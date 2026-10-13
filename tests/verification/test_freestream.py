"""Free-stream preservation of the assembled residual.

This is the strongest cheap test of the whole spatial discretization: it
exercises the grid metrics, ghost filling, every boundary condition, the
reconstruction and the numerical flux *together*. A uniform state is an exact
solution of the Euler equations, so its residual must vanish -- except in the
cells touching the airfoil, where it must not, because a uniform flow does not
satisfy a wall.

Unlike the geometric closure identities of the grid, this does not hold by
construction: it would fail for a reconstruction that is not exact on constants,
a flux that is not consistent, a far field that imposes the wrong state, or a
periodic seam whose ghosts are misordered.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from jaxaero.numerics.limiters import LIMITERS
from jaxaero.numerics.riemann import FLUXES

from ..single import Single
from ..conftest import uniform_state


WALL_INFLUENCE_DEPTH = 2
"""Cells from the wall whose residual a uniform flow may legitimately disturb.

The MUSCL stencil for the face at ``j = 1`` reaches cells ``-1 .. 2``, so the
mirrored ghost at ``j = -1`` enters the reconstruction for cell ``j = 1`` as well
as ``j = 0``. With a monotone limiter that ghost contributes nothing -- one of its
two differences is zero, so the slope is zero -- but the unlimited slope
``(a+b)/2`` does not vanish. Two cells is the stencil depth and therefore the
correct mask for every reconstruction.
"""


def residual_away_from_the_wall(residual, grid) -> np.ndarray:
    """Residual with the wall-influenced cells masked out."""
    return np.abs(np.asarray(residual))[:, :, WALL_INFLUENCE_DEPTH:]


@pytest.mark.parametrize("flux", sorted(FLUXES))
@pytest.mark.parametrize("limiter", sorted(LIMITERS))
def test_uniform_flow_has_zero_residual_away_from_the_wall(
    coarse_grid, subsonic_flow, flux: str, limiter: str
) -> None:
    """Every flux and limiter combination must preserve a uniform state."""
    numerics = NumericsConfig(flux=flux, limiter=limiter)
    operator = Single(coarse_grid, subsonic_flow, numerics)
    residual = operator.residual(uniform_state(coarse_grid, subsonic_flow))

    masked = residual_away_from_the_wall(residual, coarse_grid)
    # Scale by the flux magnitude a face carries, so the tolerance is relative.
    scale = float(
        jnp.max(jnp.hypot(coarse_grid.sx_j, coarse_grid.sy_j))
    ) * subsonic_flow.mach
    assert masked.max() < 1e-12 * max(scale, 1.0), (
        f"{flux}/{limiter}: max |R| = {masked.max():.3e} away from the wall"
    )


@pytest.mark.parametrize("mach,alpha", [(0.5, 0.0), (0.8, 1.25), (0.15, 10.0), (0.3, -4.0)])
def test_preservation_holds_at_every_condition(coarse_grid, mach: float, alpha: float) -> None:
    """Including at incidence, where the freestream is not grid-aligned.

    A non-zero angle of attack is the case that catches a metric or normal bug
    that cancels by symmetry at alpha = 0.
    """
    flow = FlowConditions(mach=mach, alpha_deg=alpha)
    operator = Single(coarse_grid, flow, NumericsConfig())
    residual = operator.residual(uniform_state(coarse_grid, flow))
    masked = residual_away_from_the_wall(residual, coarse_grid)
    assert masked.max() < 1e-12


def test_first_order_reconstruction_also_preserves_freestream(
    coarse_grid, subsonic_flow
) -> None:
    """The first-order path is a separate code path and needs its own check."""
    operator = Single(
        coarse_grid, subsonic_flow, NumericsConfig(), first_order=True
    )
    residual = operator.residual(uniform_state(coarse_grid, subsonic_flow))
    assert residual_away_from_the_wall(residual, coarse_grid).max() < 1e-12


def test_preservation_is_independent_of_grid_level(medium_grid, subsonic_flow) -> None:
    """Preservation is exact, so refining the grid must not degrade it.

    An error that grew with cell count would mean an accumulation rather than an
    identity holding pointwise.
    """
    operator = Single(medium_grid, subsonic_flow, NumericsConfig())
    residual = operator.residual(uniform_state(medium_grid, subsonic_flow))
    assert residual_away_from_the_wall(residual, medium_grid).max() < 1e-12


def test_the_wall_residual_is_not_zero(coarse_grid, subsonic_flow) -> None:
    """The complement of the test above, so it cannot pass vacuously.

    If the wall cells also showed zero residual, the slip wall would be doing
    nothing and the masked test above would prove nothing about the interior.
    """
    operator = Single(coarse_grid, subsonic_flow, NumericsConfig())
    residual = np.abs(
        np.asarray(operator.residual(uniform_state(coarse_grid, subsonic_flow)))
    )
    assert residual[:, :, 0].max() > 1e-6, "the slip wall is not disturbing a uniform flow"


def test_local_time_step_is_positive_and_scales_with_cfl(coarse_grid, subsonic_flow) -> None:
    """dt must be positive everywhere and strictly proportional to CFL."""
    state = uniform_state(coarse_grid, subsonic_flow)
    one = Single(coarse_grid, subsonic_flow, NumericsConfig(cfl=1.0))
    two = Single(coarse_grid, subsonic_flow, NumericsConfig(cfl=2.0))
    dt1 = np.asarray(one.time_step(state))
    dt2 = np.asarray(two.time_step(state))
    assert dt1.min() > 0.0
    assert np.allclose(dt2, 2.0 * dt1, rtol=1e-13)
    # The exponential radial growth spans orders of magnitude in cell size,
    # which is exactly why local rather than global time stepping is required.
    assert dt1.max() / dt1.min() > 1e3
