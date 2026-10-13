"""Unit tests for variable indexing and thermodynamic conversions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxaero.core import state
from jaxaero.core.config import FlowConditions

from ..conftest import freestream_state

GAMMA = 1.4

# Physically admissible states: positive density and pressure, subsonic to
# supersonic velocities. Ranges are wide enough to catch scaling mistakes but
# exclude the non-physical region, where the conversions are not defined.
densities = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
pressures = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
# Velocity as a multiple of the local sound speed, so the kinetic-to-internal
# energy ratio stays in the range this solver is built for.
mach_components = st.floats(
    min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False
)


@given(rho=densities, p=pressures, mach_x=mach_components, mach_y=mach_components)
@settings(max_examples=300, deadline=None)
def test_primitive_conservative_round_trip(
    rho: float, p: float, mach_x: float, mach_y: float
) -> None:
    """primitive -> conservative -> primitive recovers the input.

    Velocities are drawn as multiples of the local sound speed rather than as
    absolute numbers, which bounds the kinetic-to-internal energy ratio. That
    matters: recovering ``p`` from ``E`` subtracts the kinetic energy, so the
    attainable relative accuracy is ``eps * E/p``, and sampling absolute
    velocities lets hypothesis construct Mach-500 states where the ratio is 4e5
    and no implementation could do better than 1e-10. This solver targets
    M <= ~1.5, and the bound here (M <= 3 per component) covers it with margin.
    """
    a = np.sqrt(GAMMA * p / rho)
    u, v = mach_x * a, mach_y * a
    U = state.primitive_to_conservative(jnp.array(rho), jnp.array([u, v]), jnp.array(p), GAMMA)
    rho2, (u2, v2), p2 = state.conservative_to_primitive(U, GAMMA, 2)
    assert rho2 == pytest.approx(rho, rel=1e-13)
    assert u2 == pytest.approx(u, rel=1e-13, abs=1e-13)
    assert v2 == pytest.approx(v, rel=1e-13, abs=1e-13)
    assert p2 == pytest.approx(p, rel=1e-12)


def test_pressure_recovery_degrades_with_the_energy_ratio() -> None:
    """Documents the cancellation limit rather than hiding behind a tolerance.

    ``p`` is recovered as ``(gamma-1)(E - rho q^2/2)``, so when kinetic energy
    dominates the two terms nearly cancel and the relative error in ``p`` grows
    like ``eps * E/p``. This is a property of the conservative formulation, not of
    this implementation, and is recorded here so a future reader does not mistake
    it for a bug. It is also the reason reconstruction is done on primitive
    variables (numerics/reconstruction.py).
    """
    rho, p = 1.0, 1.0
    eps = float(np.finfo(np.float64).eps)

    for mach in (1.0, 30.0, 300.0, 3000.0):
        a = np.sqrt(GAMMA * p / rho)
        u = mach * a
        U = state.primitive_to_conservative(jnp.array(rho), jnp.array([u, 0.0]), jnp.array(p), GAMMA)
        energy = float(U[state.Rows(2).energy])
        recovered = float(state.conservative_to_primitive(U, GAMMA, 2)[2])
        error = abs(recovered - p) / p

        # The attainable bound: cancelling two numbers of size E to recover p
        # costs about eps * E/p in relative terms. Asserting the bound is the
        # real property; asserting that the error *grows* is not, because at any
        # particular set of values the rounding may happen to cancel exactly
        # (at M = 300 here it does).
        bound = 10.0 * eps * energy / p
        assert error <= max(bound, eps), (
            f"M={mach}: error {error:.3e} exceeds the cancellation bound {bound:.3e}"
        )

    # At the Mach numbers this solver targets the round trip is essentially exact.
    assert bound > 1e-9, "the bound should be loose by M=3000, illustrating the limit"


@pytest.mark.parametrize("dim", [2, 3])
def test_variable_indices_are_distinct_and_cover_nvar(dim: int) -> None:
    """Indexing is defined once; a duplicated index would silently alias fields."""
    rows = state.Rows(dim)
    indices = [state.RHO, *range(dim + 2)[rows.velocity], rows.energy]
    assert sorted(indices) == list(range(rows.n_flow))
    assert rows.nu == rows.n_flow


def test_three_dimensional_round_trip() -> None:
    rng = np.random.default_rng(3)
    rho, p = rng.uniform(0.5, 2.0, 50), rng.uniform(0.5, 2.0, 50)
    velocity = rng.uniform(-1.0, 1.0, (3, 50))
    U = state.primitive_to_conservative(jnp.asarray(rho), jnp.asarray(velocity), jnp.asarray(p), GAMMA)
    assert U.shape == (5, 50)
    rho2, v2, p2 = state.conservative_to_primitive(U, GAMMA, 3)
    np.testing.assert_allclose(rho2, rho, rtol=1e-14)
    np.testing.assert_allclose(v2, velocity, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(p2, p, rtol=1e-13)


@pytest.mark.parametrize("mach,alpha", [(0.5, 0.0), (0.8, 1.25), (0.15, 10.0), (0.1, -4.0)])
def test_freestream_follows_the_nondimensionalization(mach: float, alpha: float) -> None:
    """rho = 1, |velocity| = Mach, p = 1/gamma, per constants.py."""
    flow = FlowConditions(mach=mach, alpha_deg=alpha)
    rho, u, v, p = state.freestream_primitive(flow)
    assert state.freestream_primitive(flow, 3) == (rho, u, 0.0, v, p)    # incidence in x-z, span y
    assert rho == pytest.approx(1.0)
    assert p == pytest.approx(1.0 / flow.gamma)
    assert np.hypot(u, v) == pytest.approx(mach, rel=1e-14)
    assert np.rad2deg(np.arctan2(v, u)) == pytest.approx(alpha, abs=1e-12)
    # The sound speed is 1 by construction, so the local Mach number is |velocity|.
    assert np.sqrt(flow.gamma * p / rho) == pytest.approx(1.0, rel=1e-15)



def test_is_physical_flags_bad_states() -> None:
    # Build the clean base state from primitives, so it is physical by
    # construction. A naive array of ones is not: its kinetic energy exactly
    # equals its total energy, giving p = 0.
    ones = jnp.ones((2, 2))
    U = state.primitive_to_conservative(ones, 0.1 * jnp.stack([ones, ones]), 2.0 * ones, GAMMA)
    U = U.at[state.RHO, 0, 0].set(-1.0)       # negative density
    U = U.at[state.Rows(2).energy, 1, 1].set(-50.0)   # drives pressure negative
    good = np.asarray(state.is_physical(U, GAMMA, 2))
    assert not good[0, 0]
    assert not good[1, 1]
    assert good[0, 1] and good[1, 0]

    ok, (i, j) = state.first_bad_cell(U, GAMMA, 2)
    assert not ok and (i, j) == (0, 0)

    clean = jnp.broadcast_to(freestream_state(FlowConditions(mach=0.5))[:, None, None], (4, 3, 3))
    assert bool(jnp.all(state.is_physical(clean, GAMMA, 2)))
    assert state.first_bad_cell(clean, GAMMA, 2) == (True, (-1, -1))


def test_nan_is_not_physical() -> None:
    """NaN must be caught, not silently pass the positivity comparison."""
    U = jnp.broadcast_to(freestream_state(FlowConditions(mach=0.5))[:, None, None], (4, 2, 2))
    U = U.at[state.RHO, 0, 0].set(jnp.nan)
    assert not bool(jnp.all(state.is_physical(U, GAMMA, 2)))
