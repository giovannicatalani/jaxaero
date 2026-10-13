"""Unit tests for the viscous flux, the face gradients and the no-slip wall.

The face gradient must be exact for linear fields on any grid, so a uniform
shear gives a constant stress and zero net viscous momentum in every cell. A
quadratic field then measures the operator's consistency, which must be second
order once the grid is smooth -- the property that exposed the cosine station
layout of the original O-grid generator.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core import constants
from jaxaero.core.config import FlowConditions, NumericsConfig
from validation.grids.karman_trefftz import make_o_grid
from jaxaero.core.state import freestream_primitive
from jaxaero.numerics.gradients import face_gradients
from jaxaero.numerics.residual import local_time_step, viscous_fluxes
from jaxaero.physics.navier_stokes import Viscosity, conductivity, viscous_flux

from ..single import Single, fill_ghosts, ghost_centres, padded_centres, wave_speeds
from ..conftest import uniform_state

MU, GAMMA = 0.1, 1.4
PU, PV = 1, 2     # the 2D velocity rows
GRIDS = {"conformal": lambda: make_o_grid(32), "viscous": lambda: make_o_grid(64, 96, 2e-4)}


def linear_state(grid, a: float = 0.3, b: float = -0.7):
    """Primitive state with velocity linear in space, uniform rho and p."""
    x, y = padded_centres(grid)
    one = jnp.ones_like(x)
    return jnp.stack([one, a * x + b * y, -b * x + 0.2 * y, one / GAMMA])


def test_pure_shear_flux() -> None:
    """``u = y``: shear stress ``mu`` on a face normal to y, and work ``u mu``."""
    zero, one = jnp.array(0.0), jnp.array(1.0)
    u = jnp.array(0.4)
    grad_v = jnp.stack([jnp.stack([zero, one]), jnp.stack([zero, zero])])     # du/dy = 1
    flux = viscous_flux(jnp.stack([u, zero]), grad_v, jnp.stack([zero, zero]),
                        jnp.stack([zero, one]), MU, conductivity(MU, GAMMA, 0.72))
    assert np.allclose(np.asarray(flux), [0.0, MU, 0.0, 0.4 * MU], rtol=1e-15)


def test_heat_flux_coefficient() -> None:
    """``k = mu / (Pr (gamma - 1))`` with ``T = a^2`` -- the nondimensional form."""
    zero, one = jnp.array(0.0), jnp.array(1.0)
    flux = viscous_flux(jnp.stack([zero, zero]), jnp.zeros((2, 2)), jnp.stack([one, zero]),
                        jnp.stack([one, zero]), MU, conductivity(MU, GAMMA, 0.72))
    assert float(flux[3]) == pytest.approx(MU / (0.72 * (GAMMA - 1.0)), rel=1e-15)


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_face_gradients_are_exact_for_linear_fields(name: str) -> None:
    """On every face -- wall, far field and periodic seam included.

    Exact up to the conditioning of a difference quotient: round-off in values
    of size ``|x|`` divided by a diagonal of size ``h`` gives ``eps |x| / h``,
    with ``|x|`` up to ~150 at the far field and ``h`` down to the smallest cell.
    """
    grid = GRIDS[name]()
    x, y = padded_centres(grid)
    (gx_i, gy_i), (gx_j, gy_j) = face_gradients(jnp.stack([1.0 + 0.3 * x - 0.7 * y]), ghost_centres(grid))
    bound = 10.0 * np.finfo(float).eps * float(jnp.max(jnp.abs(x))) / float(jnp.sqrt(grid.area.min()))
    for got, want in ((gx_i, 0.3), (gy_i, -0.7), (gx_j, 0.3), (gy_j, -0.7)):
        assert float(jnp.max(jnp.abs(got - want))) < bound


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_uniform_shear_has_no_net_viscous_momentum(name: str) -> None:
    """A linear velocity gives a constant stress, whose divergence is zero."""
    grid = GRIDS[name]()
    visc_i, visc_j = viscous_fluxes(linear_state(grid), grid, Viscosity(MU), GAMMA, centres=ghost_centres(grid))
    net = (visc_i[:, 1:] - visc_i[:, :-1]) + (visc_j[:, :, 1:] - visc_j[:, :, :-1])
    scale = MU * float(jnp.max(jnp.hypot(grid.sx_j, grid.sy_j)))
    assert float(jnp.max(jnp.abs(net[1:3, :, 1:]))) < 1e-12 * scale


def test_viscous_operator_is_second_order() -> None:
    """``u = y^2`` has ``div tau = (2 mu, 0)``; the error must fall by ~4 per level.

    Every cell between 0.6 and 5 chords from mid-chord, maximum norm: the grid
    line leaving the trailing edge is included, which is where a non-smooth
    station layout shows up as an error that does not converge at all.
    """
    errors = []
    for n in (32, 64, 128):
        grid = make_o_grid(n)
        x, y = padded_centres(grid)
        q = jnp.stack([jnp.ones_like(x), y**2, jnp.zeros_like(x), jnp.ones_like(x) / GAMMA])
        visc_i, visc_j = viscous_fluxes(q, grid, Viscosity(MU), GAMMA, centres=ghost_centres(grid))
        net = (visc_i[1, 1:] - visc_i[1, :-1]) + (visc_j[1, :, 1:] - visc_j[1, :, :-1])
        r = np.hypot(np.asarray(grid.xc) - 0.5, np.asarray(grid.yc))
        sel = (r > 0.6) & (r < 5.0)
        errors.append(np.abs(np.asarray(net / grid.area)[sel] - 2.0 * MU).max())
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5, errors


def test_no_slip_wall_velocity_is_exactly_zero() -> None:
    """The face average of cell and ghost -- what the viscous flux sees -- vanishes."""
    grid = make_o_grid(32)
    flow = FlowConditions(mach=0.5, alpha_deg=1.0, reynolds=5000.0)
    q_inf = jnp.asarray(freestream_primitive(flow))
    rng = np.random.default_rng(0)
    interior = q_inf[:, None, None] * (1.0 + 0.1 * rng.standard_normal((4, 32, 32)))
    padded = fill_ghosts(interior, grid, 2, q_inf, flow.gamma, no_slip=True)
    for var in (PU, PV):
        assert float(jnp.max(jnp.abs(padded[var, 2:-2, 1] + padded[var, 2:-2, 2]))) == 0.0


def test_viscous_freestream_is_preserved_off_the_wall() -> None:
    """A uniform state has no gradients: only the no-slip wall row may feel it."""
    grid = make_o_grid(64, 96, 2e-4)
    flow = FlowConditions(mach=0.5, alpha_deg=1.0, reynolds=5000.0)
    residual = Single(grid, flow, NumericsConfig()).residual(uniform_state(grid, flow))
    assert float(jnp.max(jnp.abs(residual[:, :, 2:]))) < 1e-12
    assert float(jnp.max(jnp.abs(residual[:, :, 0]))) > 1e-6


def test_viscous_time_step_is_smaller_and_tends_to_the_inviscid_one() -> None:
    grid = make_o_grid(64, 96, 2e-4)
    flow = FlowConditions(mach=0.5)
    U = uniform_state(grid, flow)
    inviscid = np.asarray(local_time_step(U, grid, GAMMA, 1.0))
    viscous = np.asarray(local_time_step(U, grid, GAMMA, 1.0, viscosity=Viscosity(0.5 / 5000.0)))
    vanishing = np.asarray(local_time_step(U, grid, GAMMA, 1.0, viscosity=Viscosity(1e-12)))
    assert np.all(viscous < inviscid) and np.all(viscous > 0.0)
    assert np.allclose(vanishing, inviscid, rtol=1e-6)
    # The near-wall cells are where diffusion governs.
    assert (inviscid / viscous)[:, 0].mean() > (inviscid / viscous)[:, -1].mean()
    assert constants.PRANDTL_LAMINAR == 0.72


def test_viscous_face_wave_speed_adds_the_thin_layer_term() -> None:
    """``lambda_f += 2 (gamma/Pr) mu |S|^2 / (rho V)``; unchanged without ``mu``."""

    grid = make_o_grid(32, 44, 4e-4)
    flow = FlowConditions(mach=0.5)
    U = uniform_state(grid, flow)
    mu = 1e-4
    inv_i, inv_j = wave_speeds(U, grid, GAMMA)
    vis_i, vis_j = wave_speeds(U, grid, GAMMA, Viscosity(mu))
    # Interior j-face between cells (3, 4) and (3, 5); freestream rho = 1.
    s2 = float(grid.sx_j[3, 5] ** 2 + grid.sy_j[3, 5] ** 2)
    volume = 0.5 * float(grid.area[3, 4] + grid.area[3, 5])
    expected = 2.0 * GAMMA / constants.PRANDTL_LAMINAR * mu * s2 / volume
    assert float(vis_j[3, 5] - inv_j[3, 5]) == pytest.approx(expected, rel=1e-12)
    assert bool(jnp.all(vis_i > inv_i)) and bool(jnp.all(vis_j > inv_j))


def test_no_slip_wall_is_adiabatic_through_the_ghosts() -> None:
    """Zero wall heat flux with no special case, for any interior temperature."""
    grid = make_o_grid(64, 96, 2e-4)
    flow = FlowConditions(mach=0.5, alpha_deg=1.0, reynolds=5000.0)
    q_inf = jnp.asarray(freestream_primitive(flow))
    rng = np.random.default_rng(1)
    interior = q_inf[:, None, None] * (1.0 + 0.1 * rng.standard_normal((4, 64, 96)))
    padded = fill_ghosts(interior, grid, 2, q_inf, flow.gamma, no_slip=True)
    _, visc_j = viscous_fluxes(padded[:, 1:-1, 1:-1], grid, Viscosity(MU), GAMMA, centres=ghost_centres(grid))
    scale = float(jnp.max(jnp.abs(visc_j[3, :, 1])))
    assert float(jnp.max(jnp.abs(visc_j[3, :, 0]))) < 1e-12 * scale


# --- Sutherland's law ---------------------------------------------------------------


def test_sutherland_is_the_freestream_value_at_the_freestream_temperature() -> None:
    law = Viscosity(1e-4, 198.6 / 540.0)
    assert float(law(jnp.asarray(1.0))) == pytest.approx(1e-4, rel=1e-15)
    assert Viscosity(1e-4)(jnp.asarray(2.0)) == 1e-4   # constant law: the float itself


def test_sutherland_matches_the_dimensional_formula() -> None:
    """Against White's dimensional form at 300 K and 600 K with T_inf = 540 R:
    ``mu = mu0 (T/T0)^(3/2) (T0 + S)/(T + S)``, T0 = 491.6 R, S = 198.6 R."""
    mu0, T0, S, T_inf = 1.716e-5, 491.6, 198.6, 540.0

    def white(T):
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

    law = Viscosity(1.0, S / T_inf)
    for T in (300.0 * 1.8, 600.0 * 1.8, 250.0 * 1.8):
        assert float(law(jnp.asarray(T / T_inf))) == pytest.approx(white(T) / white(T_inf), rel=1e-13)


def test_sutherland_viscosity_rises_with_temperature_in_the_flux() -> None:
    """Same shear, hotter face, larger stress: the law reaches the flux."""
    grid = make_o_grid(32, 44, 4e-4)
    q = linear_state(grid)
    hot = q.at[3].multiply(1.5)                     # p up 50%: T up 50%
    for law, rises in ((Viscosity(MU), False), (Viscosity(MU, 0.3678), True)):
        cold_i, _ = viscous_fluxes(q, grid, law, GAMMA, centres=ghost_centres(grid))
        hot_i, _ = viscous_fluxes(hot, grid, law, GAMMA, centres=ghost_centres(grid))
        ratio = float(jnp.max(jnp.abs(hot_i[1]))) / float(jnp.max(jnp.abs(cold_i[1])))
        assert (ratio > 1.2) == rises and (abs(ratio - 1.0) < 1e-12) != rises
