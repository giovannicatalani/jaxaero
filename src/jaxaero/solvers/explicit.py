"""Explicit multistage Runge-Kutta time stepping with local time steps.

For a steady problem the time integrator is only a relaxation scheme: it does not
need to be time-accurate, only to damp the whole eigenvalue spectrum of the
spatial operator as fast as possible per residual evaluation. That is why the
coefficients below are the Jameson-type sequences tuned for maximum steady-state
damping rather than the classical RK4 weights.

The local time step is frozen at the start of each multistage cycle, which is
standard and saves recomputing the spectral radius at every stage.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
from jax import Array

from ..numerics.residual import ResidualOperator

# Stage coefficients alpha_k for U^(k) = U^n - alpha_k dt/V R(U^(k-1)).
RK_SCHEMES: dict[str, tuple[float, ...]] = {
    "euler": (1.0,),
    "rk4_jameson": (0.25, 1.0 / 3.0, 0.5, 1.0),
    "rk5_jameson": (0.25, 1.0 / 6.0, 0.375, 0.5, 1.0),
}
"""Jameson's multistage sequences for steady-state Euler. The 5-stage scheme
tolerates roughly CFL 3 and is the usual production choice; ``euler`` is a single
forward step, kept for debugging and for order checks."""


def get_rk_scheme(name: str) -> tuple[float, ...]:
    try:
        return RK_SCHEMES[name]
    except KeyError:
        raise KeyError(
            f"unknown RK scheme '{name}'. Available: {sorted(RK_SCHEMES)}"
        ) from None


class SolverState(NamedTuple):
    """``U``: conservative state on the canvas, ``(nvar, *canvas shape)``."""

    U: Array


def make_rk_step(
    operator: ResidualOperator, scheme: str = "rk5_jameson"
) -> Callable[[SolverState], tuple[SolverState, Array]]:
    """``step(state) -> (state, residual norm)`` of the ``RK_SCHEMES[scheme]`` sequence."""
    alphas = get_rk_scheme(scheme)
    volume = operator.grid.volume

    def step(state: SolverState) -> tuple[SolverState, Array]:
        u_old = state.U
        dt = operator.time_step(u_old)
        factor = dt / volume

        # The first-stage residual is the one reported: it is evaluated at U^n,
        # so the history measures the residual of the state actually reached.
        residual = operator.residual(u_old)
        norm = operator.norm(residual)

        u_new = u_old - alphas[0] * factor * residual
        for alpha in alphas[1:]:
            u_new = u_old - alpha * factor * operator.residual(u_new)

        return SolverState(U=u_new), norm

    return step


def initial_state(operator: ResidualOperator, dtype=jnp.float64) -> SolverState:
    """Uniform freestream: it satisfies the far field exactly and leaves the
    wall as the only initial inconsistency."""
    return SolverState(U=operator.initial_state(dtype))
