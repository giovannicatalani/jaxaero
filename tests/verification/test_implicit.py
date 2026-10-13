"""Verification of the implicit (Newton-Krylov) path.

The two solvers must reach the same converged solution. That is not
automatic in general, but it is structural here: both paths drive the *same* ``assemble_residual``, the explicit one by
``U -= dt/V R(U)`` and the implicit one by solving
``(V/dt I + dR/dU) dU = -R(U)``. Both updates vanish if and only if
``R(U) = 0``, so they share their fixed points exactly. What has to be tested is
that the implementation really does share that residual, and that the state the
implicit path arrives at is one the explicit scheme also considers at rest.

Budget note: these run on the coarsest level with an unlimited first-order
residual, the configuration where the exact Jacobian is appropriate and
convergence is fast. They are regression guards at a size that fits a test
suite, not a demonstration of the phase's ten-order exit criterion, which needs
a GPU and several minutes.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxaero.core.config import FlowConditions, ImplicitConfig, NumericsConfig
from jaxaero.numerics.residual import make_residual
from jaxaero.solvers import explicit, implicit
from jaxaero.solvers.driver import run_to_steady

from ..single import atlas_of

ITERATIONS = 150


@pytest.fixture(scope="module")
def converged(coarse_grid):
    """March the implicit path on a smooth residual and keep the result.

    Module scoped: the march is the expensive part and three tests read it, so
    the flow conditions are built here rather than taken from the
    function-scoped ``subsonic_flow`` fixture.
    """
    flow = FlowConditions(mach=0.5, alpha_deg=2.0)
    operator = make_residual(atlas_of(coarse_grid, 2, False), flow, NumericsConfig(limiter="none"), first_order=True
    )
    config = ImplicitConfig(cfl=2.0, cfl_growth=1.5, jacobian="exact")
    state, history = run_to_steady(
        implicit.make_newton_step(operator, config),
        implicit.initial_state(operator, config),
        max_iters=ITERATIONS,
        tol=1e-12,
        block_iters=25,
    )
    return operator, state, history


@pytest.mark.slow
def test_implicit_path_reduces_the_residual(converged) -> None:
    """The residual falls, the state stays physical and the CFL ramp engages.

    The threshold is a regression guard set below a measured 1.44 orders at this
    budget, not a convergence claim.
    """
    operator, state, history = converged
    assert history.orders_dropped > 1.2
    assert bool(jnp.all(jnp.isfinite(state.U)))
    assert float(state.cfl) > float(ImplicitConfig().cfl_min)


@pytest.mark.slow
def test_explicit_scheme_is_at_rest_at_the_implicit_solution(converged) -> None:
    """One explicit multistage step barely moves the implicit solution.

    This is the operative form of "the same converged solution": if the two
    paths had different fixed points, an explicit step at the implicit one would
    move it by an amount set by the difference between them rather than by the
    residual that remains.

    The tolerance is loose on purpose. At this budget the state is converged to
    about 1.4 orders, and the drift measured 4.9e-6 relative; a bound of 1e-4
    distinguishes "at rest" from "converging to somewhere else" without pinning
    the test to a particular iteration count.
    """
    operator, state, _ = converged

    stepped, _ = explicit.make_rk_step(operator, scheme="rk5_jameson")(
        explicit.SolverState(U=state.U)
    )
    drift = float(
        jnp.linalg.norm(stepped.U - state.U) / jnp.linalg.norm(state.U)
    )
    assert drift < 1e-4


@pytest.mark.slow
def test_both_paths_agree_on_the_residual_of_a_state(coarse_grid, transonic_flow) -> None:
    """The implicit step reports exactly the norm the explicit step would.

    Cheap but pointed: it is what guarantees the two convergence histories are
    comparable quantities rather than two differently normalized ones, which is
    what a claim like "implicit reached more orders" rests on.
    """
    operator = make_residual(atlas_of(coarse_grid, 2, False), transonic_flow, NumericsConfig(limiter="minmod")
    )
    config = ImplicitConfig()
    jacobian = make_residual(atlas_of(coarse_grid, 2, False), transonic_flow, NumericsConfig(limiter="minmod"), first_order=True
    )
    state = implicit.initial_state(operator, config)

    _, implicit_norm = implicit.make_newton_step(operator, config, jacobian)(state)
    _, explicit_norm = explicit.make_rk_step(operator)(
        explicit.SolverState(U=state.U)
    )
    assert float(implicit_norm) == pytest.approx(float(explicit_norm), rel=1e-13)
