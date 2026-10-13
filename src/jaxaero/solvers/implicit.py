"""Pseudo-transient Newton-Krylov (JFNK) stepping.

Each step solves the backward-Euler system

    (V/dt I + dR/dU) dU = -R(U),        U <- U + dU

with the CFL that sets ``V/dt`` ramped up as the iteration succeeds. At low CFL
this is a robust implicit march; as ``V/dt -> 0`` it becomes Newton's method on
``R(U) = 0``. That is pseudo-transient continuation.

Jacobian-vector products from ``jax.linearize``, right-preconditioned GMRES
(``krylov.py``). ``ImplicitConfig.jacobian``: ``exact`` linearizes the residual
solved; ``first_order`` (default) the first-order one, the second-order
residual staying on the right-hand side (defect correction): same fixed point,
no limiter kinks in the operator, and the system the preconditioner
approximates.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ..core.config import ImplicitConfig
from ..core.state import RHO, Rows, conservative_to_primitive, dot, is_physical
from ..numerics.residual import ResidualOperator, cell_wave_speed_sum
from . import explicit
from .krylov import gmres_right
from .preconditioners import get_preconditioner


class ImplicitState(NamedTuple):
    """``U (nvar, *canvas shape)``, the CFL and the previous ``|R|``: traced, so the ramp needs no recompilation."""

    U: Array
    cfl: Array
    residual_norm: Array


def initial_state(
    operator: ResidualOperator, config: ImplicitConfig, dtype=jnp.float64
) -> ImplicitState:
    """The explicit path's freestream start, at the starting CFL."""
    return ImplicitState(
        U=explicit.initial_state(operator, dtype=dtype).U,
        cfl=jnp.asarray(config.cfl, dtype=dtype),
        residual_norm=jnp.asarray(jnp.inf, dtype=dtype),
    )


def relax(dU: Array, U: Array, gamma: float, fraction: float | None, dim: int = 2) -> tuple[Array, Array]:
    """Scale ``dU`` by one global factor so no cell's density or pressure moves
    by more than ``fraction`` of itself, nor ``nu~`` by more than ``fraction``
    of ``max(|nu~|, nu_inf)`` (``nu_hat`` is stored over ``nu_inf``, so the
    floor is 1). Returns ``(omega dU, omega)``; ``omega = 1`` when ``fraction``
    is ``None``, and the update is then returned untouched.

    The pressure change is linearized, ``dp = (gamma - 1)(dE - V.d(rho V) +
    |V|^2/2 d rho)``, which is what a small step does to it.
    """
    one = jnp.ones((), dU.dtype)
    if fraction is None:
        return dU, one
    rows = Rows(dim)
    rho, velocity, p = conservative_to_primitive(U, gamma, dim)
    d_rho = dU[RHO]
    d_e = dU[rows.energy]
    for k in range(dim):
        d_e = d_e - velocity[k] * dU[1 + k]
    d_p = (gamma - 1.0) * (d_e + 0.5 * dot(velocity, velocity) * d_rho)
    ratios = [jnp.abs(d_rho) / rho, jnp.abs(d_p) / p]
    if U.shape[0] > rows.n_flow:
        nu_hat = U[rows.nu] / rho
        d_nu_hat = (dU[rows.nu] - nu_hat * d_rho) / rho
        ratios.append(jnp.abs(d_nu_hat) / jnp.maximum(jnp.abs(nu_hat), 1.0))
    worst = jnp.max(jnp.stack([jnp.max(r) for r in ratios]))
    omega = jnp.minimum(one, fraction / jnp.maximum(worst, 1e-300))
    return omega * dU, omega


def make_newton_step(
    operator: ResidualOperator,
    config: ImplicitConfig = ImplicitConfig(),
    jacobian_operator: ResidualOperator | None = None,
) -> Callable[[ImplicitState], tuple[ImplicitState, Array]]:
    """``step(state) -> (state, |R(U)| before the update)``; ``jacobian_operator``
    (a first-order residual: defect correction) supplies the linear operator,
    ``None`` linearizes ``operator`` itself."""
    gamma = operator.gamma
    prepare_preconditioner = get_preconditioner(config.preconditioner)(
        operator.atlas, gamma, operator.viscosity, operator.model, operator.distance
    )
    exact_jacobian = jacobian_operator is None
    jacobian_source = operator if exact_jacobian else jacobian_operator

    def step(state: ImplicitState) -> tuple[ImplicitState, Array]:
        U, cfl = state.U, state.cfl

        # One linearization serves the whole Krylov solve; the branch is
        # resolved at trace time, not in the loop.
        if exact_jacobian:
            residual, jvp = jax.linearize(operator.residual, U)
        else:
            residual = operator.residual(U)
            _, jvp = jax.linearize(jacobian_source.residual, U)
        norm = operator.norm(residual)

        lambdas = operator.wave_speeds(U)
        volume_over_dt = cell_wave_speed_sum(lambdas) / cfl

        def matvec(dU: Array) -> Array:
            """Action of ``V/dt I + dR/dU``."""
            return volume_over_dt[None] * dU + jvp(dU)

        # Right preconditioning, so GMRES minimizes the true linear residual
        # that the ramp below is judged on (solvers/krylov.py).
        dU = gmres_right(
            matvec,
            -residual,
            prepare_preconditioner(U, lambdas, cfl),
            restart=config.krylov_subspace,
            maxiter=config.krylov_restarts,
            tol=config.krylov_tol,
        )

        # Bound how far an accepted step may go (ImplicitConfig.max_relative_update).
        dU, omega = relax(dU, U, gamma, config.max_relative_update, operator.dim)

        # The ramp keys off how well the linear system was solved (one more
        # matvec), not the residual history: a stagnated GMRES returns a null
        # update that leaves the residual unchanged, which looks like success.
        linear_reduction = jnp.linalg.norm(matvec(dU) + residual) / jnp.maximum(
            jnp.linalg.norm(residual), 1e-300
        )

        # Reject a step that leaves the physical domain or that GMRES failed to
        # solve, and resume at a lower CFL: one wasted step, not a lost run.
        candidate = U + dU
        physical = jnp.all(is_physical(candidate, gamma, operator.dim)) & jnp.all(jnp.isfinite(dU))
        accept = physical & (linear_reduction < 1.0)

        # Switched evolution relaxation: grow the CFL by the residual reduction
        # achieved, capped, and only when the Krylov solve earned it. Mulder and
        # van Leer, J. Comput. Phys. 59 (1985).
        reduction = state.residual_norm / jnp.maximum(norm, 1e-300)
        cfl_next = jnp.where(
            accept & (linear_reduction < config.krylov_success) & (omega >= 1.0),
            cfl * jnp.clip(reduction, 1.0, config.cfl_growth),
            jnp.where(accept, cfl, cfl / config.cfl_cutback),
        )

        return (
            ImplicitState(
                U=jnp.where(accept, candidate, U),
                cfl=jnp.clip(cfl_next, config.cfl_min, config.cfl_max),
                residual_norm=jnp.where(accept, norm, state.residual_norm),
            ),
            norm,
        )

    return step
