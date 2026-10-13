"""The iteration driver: blocks of steps fused into one on-device ``lax.scan``,
with Python between blocks for monitoring. It knows nothing of the equations,
so the explicit and implicit paths share it and its convergence test. A step
may advance a batch of independent cases (one residual each), split over
devices along the leading axis with no communication.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import jax
import numpy as np
from jax import Array, lax

BATCH_AXIS = "case"


class StepFunction(Protocol):
    """``state -> (state, residual norm)``, pure and jittable; the norm is ``(n_cases,)`` for a batch."""

    def __call__(self, state: Any) -> tuple[Any, Array]: ...


class ExitReason:
    """Why the driver stopped."""

    CONVERGED = "converged"
    MAX_ITERS = "max_iters"
    NON_FINITE = "non_finite"


@dataclass
class ConvergenceHistory:
    """``residual``: ``(n_iters,)`` or ``(n_iters, n_cases)``; ``reason``: ``CONVERGED``
    only if every case converged; times include compilation (the first block's)."""

    residual: np.ndarray
    reason: str
    n_iters: int
    seconds: float = float("nan")
    first_block_seconds: float = float("nan")
    case_reasons: tuple[str, ...] = ()

    def case(self, k: int) -> "ConvergenceHistory":
        """Case ``k`` of a batch, as a single run's history."""
        return ConvergenceHistory(self.residual[:, k], self.case_reasons[k], self.n_iters,
                                  self.seconds, self.first_block_seconds)

    @property
    def converged(self) -> bool:
        return self.reason == ExitReason.CONVERGED

    @property
    def peak(self) -> float:
        """Largest finite residual over the run: the convergence reference."""
        finite = self.residual[np.isfinite(self.residual)]
        return float(finite.max()) if finite.size else float("nan")

    @property
    def orders_dropped(self) -> float:
        """Orders of magnitude the residual fell from its peak (``run_to_steady``)."""
        if self.residual.size == 0:
            return 0.0
        peak, last = self.peak, self.residual[-1]
        if not (np.isfinite(peak) and np.isfinite(last)) or peak <= 0 or last <= 0:
            return float("nan")
        return float(np.log10(peak / last))

    def summary(self) -> str:
        return (
            f"{self.reason} after {self.n_iters} iterations, "
            f"residual peak {self.peak:.6e} -> {self.residual[-1]:.6e} "
            f"({self.orders_dropped:.2f} orders from peak)"
        )


def make_block(step: StepFunction, block_iters: int,
               devices: Sequence | None = None) -> Callable[[Any], tuple[Any, Array]]:
    """``block_iters`` steps as one jitted call, ``state -> (state, residuals)``;
    with two or more ``devices`` a batch is split over them (``shard_map``)."""

    def run_block(state: Any) -> tuple[Any, Array]:
        return lax.scan(lambda carry, _: step(carry), state, xs=None, length=block_iters)

    if devices is None or len(devices) < 2:
        return jax.jit(run_block)
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    return jax.jit(shard_map(run_block, mesh=_mesh(devices), in_specs=(P(BATCH_AXIS),),
                             out_specs=(P(BATCH_AXIS), P(None, BATCH_AXIS)), check_rep=False))


def _mesh(devices: Sequence):
    from jax.sharding import Mesh

    return Mesh(np.array(devices), (BATCH_AXIS,))


def run_to_steady(
    step: StepFunction,
    state: Any,
    *,
    max_iters: int,
    tol: float,
    block_iters: int = 100,
    on_block: Callable[[int, Any, np.ndarray], None] | None = None,
    finite_check: Callable[[Any], Array] | None = None,
    devices: Sequence | None = None,
) -> tuple[Any, ConvergenceHistory]:
    """Iterate until the residual is ``tol`` times its running **peak**, per case.

    The peak, not the first value: a uniform-freestream start has a near-zero
    first residual (free-stream preservation), a degenerate reference.
    ``on_block(block, state, residuals)`` runs between blocks (the only place
    for I/O); ``finite_check(state)`` is an optional device-side health test.
    """
    block = make_block(step, block_iters, devices)
    if devices is not None and len(devices) > 1:
        from jax.sharding import NamedSharding, PartitionSpec as P

        state = jax.device_put(state, NamedSharding(_mesh(devices), P(BATCH_AXIS)))
    n_blocks = max(1, -(-max_iters // block_iters))  # ceiling division

    blocks: list[np.ndarray] = []
    peak = reasons = None
    started = time.perf_counter()
    first_block = float("nan")
    for b in range(n_blocks):
        state, block_residual = block(state)
        block_residual = np.asarray(block_residual)
        if b == 0:
            first_block = time.perf_counter() - started
        blocks.append(block_residual)
        if on_block is not None:
            on_block(b, state, block_residual)

        columns = block_residual.reshape(block_residual.shape[0], -1)   # (block, n_cases)
        if peak is None:
            peak = np.zeros(columns.shape[1])
            reasons = np.full(columns.shape[1], ExitReason.MAX_ITERS, dtype=object)
        finite = np.where(np.isfinite(columns), columns, 0.0)
        peak = np.maximum(peak, finite.max(axis=0))
        last = columns[-1]
        running = reasons == ExitReason.MAX_ITERS
        failed = ~np.all(np.isfinite(columns), axis=0)
        if finite_check is not None and not bool(finite_check(state)):
            failed[:] = True
        reasons[running & failed] = ExitReason.NON_FINITE
        reasons[running & ~failed & (peak > 0) & (last <= tol * peak)] = ExitReason.CONVERGED
        if not np.any(reasons == ExitReason.MAX_ITERS):
            break

    residual = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)
    unfinished = [r for r in reasons if r != ExitReason.CONVERGED]
    return state, ConvergenceHistory(
        residual=residual,
        reason=unfinished[0] if unfinished else ExitReason.CONVERGED,
        n_iters=int(residual.shape[0]),
        seconds=time.perf_counter() - started,
        first_block_seconds=first_block,
        case_reasons=tuple(reasons) if residual.ndim > 1 else (),
    )
