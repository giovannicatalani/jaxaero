"""Run a case: build the operator, march to steady state, write the result.

    from jaxaero import load_case, run

    case = load_case("examples/naca0012_euler")
    result = run(case)              # progress=None: silent
    result.write()                  # -> <case output>/<case name>/

``run`` composes the residual (discretization), the case's pseudo-time path
(solvers) and the shared device driver; it knows nothing about reference data
or validation, which live in ``validation/``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array

from .core.state import conservative_to_primitive
from .mesh.atlas import Atlas
from .mesh.coarsen import prolong
from .mesh.quality import quality
from .numerics.residual import make_residual
from .case import Case
from .post.forces import Forces, integrate_forces, surface
from .io.output import ResultWriter
from .solvers import explicit, implicit
from .solvers.driver import ConvergenceHistory, run_to_steady

Progress = Callable[[int, float, float, Forces], None]
"""``(iteration, residual, orders dropped, forces) -> None``, once per block."""


SETTLED_CL_SPREAD = 1e-4
"""Largest CL range over the last quarter of a run for it to count as settled:
most RANS runs stall at round-off (9.8-10.8 orders) before ``tol``."""


def cl_spread(cl_history: np.ndarray) -> float:
    """CL (or any per-block history's) range over the last quarter of the
    blocks; NaN with fewer than four samples there, so a short run is never
    settled."""
    tail = np.asarray(cl_history)[3 * len(cl_history) // 4:]
    return float(tail.max() - tail.min()) if tail.size >= 4 else float("nan")


@dataclass(frozen=True)
class Timing:
    """Wall clock of a run; the first block carries the compilation, the later
    ones give the steady cost. ``cases``: advanced together (a polar's angles)."""

    total: float
    first_block: float
    iterations: int
    block_iters: int
    cells: int
    cases: int = 1
    devices: int = 1

    @classmethod
    def of(cls, history: ConvergenceHistory, block_iters: int, cells: int, **batch) -> "Timing":
        return cls(history.seconds, history.first_block_seconds, history.n_iters, block_iters,
                   cells, **batch)

    @property
    def per_iteration(self) -> float:
        """Steady seconds per iteration (of the whole batch), from every block
        after the first; NaN for a single-block run."""
        later = self.iterations - self.block_iters
        return (self.total - self.first_block) / later if later > 0 else float("nan")

    @property
    def compile(self) -> float:
        """The first block minus its iterations at the steady rate."""
        return self.first_block - self.block_iters * self.per_iteration

    @property
    def case_iterations_per_second(self) -> float:
        return self.cases / self.per_iteration

    def record(self) -> dict[str, float]:
        return {"total_s": self.total, "compile_s": self.compile,
                "per_iteration_s": self.per_iteration,
                "case_iterations_per_s": self.case_iterations_per_second,
                "ns_per_cell_iteration": 1e9 / (self.case_iterations_per_second * self.cells),
                "iterations": self.iterations, "cells": self.cells,
                "cases": self.cases, "devices": self.devices}

    def summary(self) -> str:
        if not self.per_iteration == self.per_iteration:          # one block: no steady rate
            return f"{self.total:.1f} s (one block, compilation included)"
        return (f"{self.total:.1f} s, compile ~{self.compile:.1f} s, "
                f"{1e3 * self.per_iteration:.2f} ms/iteration"
                + (f" for {self.cases} cases on {self.devices} device(s)" if self.cases > 1 else ""))


@dataclass(frozen=True)
class Result:
    """A run: ``U`` on the atlas canvas, ``(nvar, NI, NJ)`` (``grid.unpack`` gives
    the blocks), and the lift at the end of every block, ``cl_history``;
    ``levels``, the coarser grids solved first (grid sequencing), coarsest first."""

    case: Case
    grid: Atlas
    U: Array
    forces: Forces
    history: ConvergenceHistory
    cl_history: np.ndarray
    timing: Timing
    levels: tuple[dict, ...] = ()

    @property
    def seconds(self) -> float:
        return self.timing.total

    @property
    def cl_spread(self) -> float:
        return cl_spread(self.cl_history)

    @property
    def cells(self) -> int:
        return self.grid.cells

    def surface(self) -> dict[str, np.ndarray]:
        """Wall distributions per face: ``x``, ``y``, ``body``, ``cp`` [, ``cf``]."""
        return surface(self.U, self.grid, self.case.flow)

    def write(self, directory: str | Path | None = None, extra: dict | None = None) -> Path:
        """Write fields, wall distributions, residual and lift histories, and
        the resolved case with provenance, to ``directory`` (default:
        ``<case output>/<case name>``). Returns the directory."""
        directory = Path(directory) if directory is not None else self.case.output / self.case.name
        writer = ResultWriter(directory, case_name=self.case.name)
        writer.write_metadata(
            {"flow": self.case.flow, "numerics": self.case.numerics, "implicit": self.case.implicit},
            extra={
                "case_setup": self.case.raw,
                "grid": self.case.grid_spec.describe(self.case.directory),
                "cells": self.cells,
                "grid_quality": quality(self.grid.mesh).record(),
                "bodies": [dict(zip(("cl", "cd", "cm"), b)) for b in self.forces.bodies],
                "cl": self.forces.cl, "cd": self.forces.cd,
                "cd_friction": self.forces.cd_friction, "cm": self.forces.cm,
                "exit_reason": self.history.reason,
                "orders_dropped": self.history.orders_dropped,
                "iterations": self.history.n_iters,
                "seconds": self.seconds,
                "cl_spread": self.cl_spread,
                "timing": self.timing.record(),
                **({"grid_sequencing": list(self.levels)} if self.levels else {}),
                **(extra or {}),
            },
        )
        writer.write_residual_history(self.history.residual)
        writer.write_lift_history(self.cl_history, self.case.block_iters)
        writer.write_surface(self.surface())
        blocks = [np.asarray(u) for u in self.grid.unpack(self.U)]
        writer.write_fields(blocks, [np.asarray(b.nodes) for b in self.grid.mesh])
        if self.grid.dim == 3:
            for k, (U, b) in enumerate(zip(blocks, self.grid.mesh)):
                writer.write_vts(np.asarray(b.nodes), cell_fields(U, self.case.flow.gamma), k)
        return directory


def cell_fields(U: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    """Density, velocity, pressure and Mach number of a 3D block, for ParaView."""
    rho, velocity, p = (np.asarray(a) for a in conservative_to_primitive(jnp.asarray(U), gamma, 3))
    mach = np.linalg.norm(velocity, axis=0) / np.sqrt(gamma * p / rho)
    return {"density": rho, "velocity": velocity, "pressure": p, "mach": mach}


def make_stepper(case: Case, grid: Atlas, freestream: Array | None = None,
                 distance: Array | None = None) -> tuple[Callable, object]:
    """``(step, initial state)`` of the case's pseudo-time path; ``freestream``
    (``(4,)`` primitives, possibly traced) is how a polar maps it over angles."""
    flow, numerics = case.flow, case.numerics
    operator = make_residual(grid, flow, numerics, freestream=freestream, distance=distance)
    if numerics.solver == "implicit":
        jacobian = None if case.implicit.jacobian == "exact" else make_residual(
            grid, flow, numerics, first_order=True, freestream=freestream, distance=operator.distance)
        return (implicit.make_newton_step(operator, case.implicit, jacobian),
                implicit.initial_state(operator, case.implicit, dtype=numerics.dtype))
    return (explicit.make_rk_step(operator, scheme=case.rk_scheme),
            explicit.initial_state(operator, dtype=numerics.dtype))


def run(case: Case, grid: Atlas | None = None, progress: Progress | None = None,
        on_level: Callable[[int, Atlas], None] | None = None) -> Result:
    """March ``case`` to steady state on ``grid`` (default: ``case.grid()``),
    after its ``sequencing`` coarser grids, each started from the one before;
    ``on_level(level, grid)`` announces each grid (0: the case's own)."""
    grid = case.grid() if grid is None else grid
    levels, start = [], None
    for level in range(case.sequencing, 0, -1):
        coarse = case.grid(coarsen=level)
        if on_level is not None:
            on_level(level, coarse)
        r = _march(case, coarse, progress, start)
        levels.append({"cells": coarse.cells, "iterations": r.history.n_iters, "seconds": r.seconds,
                       "exit_reason": r.history.reason, "cl": r.forces.cl, "cd": r.forces.cd})
        start = (r.U, coarse)
    if levels and on_level is not None:
        on_level(0, grid)
    return replace(_march(case, grid, progress, start), levels=tuple(levels))


def _march(case: Case, grid: Atlas, progress: Progress | None, start: tuple[Array, Atlas] | None) -> Result:
    """One grid to steady state, from freestream or from ``start``, a coarser
    grid's solution ``(U, atlas)``."""
    flow, numerics = case.flow, case.numerics
    length = case.grid_spec.force_reference()
    step, state = make_stepper(case, grid)
    if start is not None:
        state = state._replace(U=prolong(*start, grid, state.U))

    cl_history: list[float] = []
    coefficients: list[tuple[float, float, float]] = []
    peak = [0.0]

    def on_block(index: int, current, residuals: np.ndarray) -> bool:
        finite = residuals[np.isfinite(residuals)]
        if finite.size:
            peak[0] = max(peak[0], float(finite.max()))
        forces = integrate_forces(current.U, grid, flow, length)
        cl_history.append(forces.cl)
        coefficients.append((forces.cl, forces.cd, forces.cm))
        if progress is not None:
            last = float(residuals[-1])
            orders = float(np.log10(peak[0] / last)) if peak[0] > 0.0 and last > 0.0 else 0.0
            progress((index + 1) * case.block_iters, last, orders, forces)
        # NaN (too few blocks) compares false: a short run never stops here.
        return numerics.forces_tol is not None and all(
            cl_spread(h) < numerics.forces_tol for h in np.transpose(coefficients))

    state, history = run_to_steady(
        step, state,
        max_iters=numerics.max_iters, tol=numerics.tol, block_iters=case.block_iters,
        on_block=on_block, finite_check=lambda s: jnp.all(jnp.isfinite(s.U)),
    )
    return Result(
        case=case, grid=grid, U=state.U,
        forces=integrate_forces(state.U, grid, flow, length),
        history=history, cl_history=np.asarray(cl_history),
        timing=Timing.of(history, case.block_iters, grid.cells),
    )
