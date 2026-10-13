"""A polar: one case at many angles of attack, batched on the GPU(s).

    python -m jaxaero.polar CASE_DIR --alpha START STOP STEP [--devices N]
                            [--set file.section.key=value ...] [--out DIR] [--quiet]

The angle of attack enters the solver only through the freestream vector
``(rho, u, v, p)``, which ``make_residual`` accepts as an array. So one step is
traced with the freestream as an argument and ``jax.vmap``-ed over the angles:
every kernel then does all angles' work at once, which fills a GPU that one
case leaves latency-bound (the line preconditioner's sequential sweep). Each
angle keeps its own CFL ramp, rejections and convergence; they never interact.
The driver (``solvers/driver.py``) marches the batch and splits it over the
devices.

Output: ``<out>/polar.dat`` (alpha, CL, CD, friction, CM, orders, iterations,
CL spread, settled), ``<out>/polar_metadata.json`` (case, timing, provenance),
and one ordinary result directory per angle, ``<out>/alpha_+2.50/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .mesh.atlas import Atlas
from .core.state import freestream_primitive
from .numerics.residual import make_residual
from .case import Case, CaseError, load_case
from .post.forces import integrate_forces, pressure_lift
from .io.output import environment_record, git_revision
from .simulation import SETTLED_CL_SPREAD, Result, Timing, make_stepper
from .solvers.driver import run_to_steady


@dataclass(frozen=True)
class Polar:
    """One ``Result`` per angle, and the timing of the batch."""

    case: Case
    alphas: np.ndarray
    results: list[Result]
    timing: Timing

    def table(self) -> np.ndarray:
        """Rows ``alpha, CL, CD, CD friction, CM, orders, iterations, CL spread, settled``."""
        return np.array([[a, r.forces.cl, r.forces.cd, r.forces.cd_friction, r.forces.cm,
                          r.history.orders_dropped, r.history.n_iters, r.cl_spread,
                          float(r.cl_spread <= SETTLED_CL_SPREAD)]
                         for a, r in zip(self.alphas, self.results)])

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(directory / "polar.dat", self.table(),
                   header="alpha_deg cl cd cd_friction cm orders_dropped iterations cl_spread settled",
                   fmt=["%.4f", "%.8f", "%.8f", "%.8f", "%.8f", "%.3f", "%d", "%.3e", "%d"])
        (directory / "polar_metadata.json").write_text(json.dumps({
            "case": self.case.name, "case_setup": self.case.raw, "alphas": self.alphas.tolist(),
            "timing": self.timing.record(), "settled_cl_spread": SETTLED_CL_SPREAD,
            "git": git_revision(), "environment": environment_record(),
        }, indent=2, default=float))
        for alpha, result in zip(self.alphas, self.results):
            result.write(directory / f"alpha_{alpha:+.2f}")
        return directory


def at_alpha(case: Case, alpha: float) -> Case:
    flow = dataclasses.replace(case.flow, alpha_deg=float(alpha))
    return dataclasses.replace(case, flow=flow, name=f"{case.name}_a{alpha:+.2f}")


def run_polar(case: Case, alphas, grid: Atlas | None = None, devices: int | None = None,
              progress=None) -> Polar:
    """``case`` at every angle of ``alphas`` (degrees) at once, split over
    ``devices`` (default all); ``progress(iteration, CL per angle)`` per block."""
    grid = case.grid() if grid is None else grid
    alphas = np.asarray(alphas, dtype=float)
    use = jax.devices()[: devices or len(jax.devices())]
    # Pad to a multiple of the device count with copies of the last angle.
    padded = np.concatenate([alphas, np.repeat(alphas[-1:], -alphas.size % len(use))])
    freestream = jnp.asarray([freestream_primitive(at_alpha(case, a).flow, grid.dim) for a in padded],
                             dtype=case.numerics.dtype)
    distance = make_residual(grid, case.flow, case.numerics).distance

    def stepper(q):
        return make_stepper(case, grid, freestream=q, distance=distance)

    def step(carry):
        states, q = carry
        states, residuals = jax.vmap(lambda s, f: stepper(f)[0](s))(states, q)
        return (states, q), residuals

    lift = jax.jit(jax.vmap(lambda U, q: pressure_lift(U, grid, q, case.flow.gamma)))
    lifts: list[np.ndarray] = []

    def on_block(index, carry, residuals):
        lifts.append(np.asarray(lift(carry[0].U, carry[1]))[: alphas.size])
        if progress is not None:
            progress((index + 1) * case.block_iters, lifts[-1])

    carry, history = run_to_steady(
        step, (jax.jit(jax.vmap(lambda f: stepper(f)[1]))(freestream), freestream),
        max_iters=case.numerics.max_iters, tol=case.numerics.tol, block_iters=case.block_iters,
        on_block=on_block, devices=use)

    timing = Timing.of(history, case.block_iters, grid.cells, cases=alphas.size, devices=len(use))
    U, lifts = np.asarray(carry[0].U), np.array(lifts)
    results = []
    for k, alpha in enumerate(alphas):
        sub, Uk = at_alpha(case, alpha), jnp.asarray(U[k])
        forces = integrate_forces(Uk, grid, sub.flow, case.grid_spec.force_reference())
        results.append(Result(sub, grid, Uk, forces, history.case(k), lifts[:, k], timing))
    return Polar(case, alphas, results, timing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jaxaero.polar", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="case directory (case.json + numerical.json)")
    parser.add_argument("--alpha", type=float, nargs=3, required=True, metavar=("START", "STOP", "STEP"),
                        help="angles START..STOP (included) by STEP, degrees")
    parser.add_argument("--devices", type=int, help="devices to split the angles over (default: all)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="FILE.SECTION.KEY=VALUE", help="override one key; repeatable")
    parser.add_argument("--out", type=Path, help="output directory (default: <save_path>/<case_name>_polar)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        case = load_case(args.case, tuple(args.overrides))
    except (CaseError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    start, stop, step = args.alpha
    alphas = np.round(np.arange(start, stop + 0.5 * step, step), 10)
    print(f"{case.name}: {alphas.size} angles {start:+g}..{stop:+g} deg on {[str(d) for d in jax.devices()]}")

    def progress(iteration, cl):
        print(f"{iteration:8d}  CL " + " ".join(f"{c:+.4f}" for c in cl))

    polar = run_polar(case, alphas, devices=args.devices, progress=None if args.quiet else progress)
    out = polar.write(args.out or case.output / f"{case.name}_polar")
    print("   alpha        CL          CD   orders  CL spread  settled")
    for row in polar.table():
        print(f"{row[0]:+8.2f}  {row[1]:+.6f}  {row[2]:+.7f}  {row[5]:6.2f}  {row[7]:9.1e}  {bool(row[8])}")
    print(f"{polar.timing.summary()}\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
