"""Unit tests for the physical boundary conditions.

Each condition is tested by the property it exists to enforce rather than against
stored numbers: a slip wall must pass no mass, and a characteristic far field must
leave an undisturbed freestream alone. Both are exact statements, asserted at
round-off.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions
from jaxaero.core.state import freestream_primitive
from jaxaero.numerics import boundary as bc
from jaxaero.numerics.riemann import get_flux

from ..single import atlas_of, fill_ghosts, to_canvas, unit_j_normals, wall_slice

GAMMA = 1.4
NGHOST = 2
PRHO, PU, PV, PP = 0, 1, 2, 3     # the 2D rows (core/state.py, Rows(2))


def uniform_primitive(grid, flow: FlowConditions) -> jnp.ndarray:
    """A uniform freestream primitive field on the interior cells."""
    ones = jnp.ones((grid.ni, grid.nj))
    return jnp.stack([value * ones for value in freestream_primitive(flow)], axis=0)


def wall_ghosts(field, grid) -> jnp.ndarray:
    """The ghosts under the wall, ``(4, n_wall, g)`` in ascending ``j``."""
    q_inf = jnp.asarray(freestream_primitive(FlowConditions(mach=0.5)))
    padded = fill_ghosts(field, grid, NGHOST, q_inf, GAMMA)
    return padded[:, NGHOST:-NGHOST, :NGHOST][:, wall_slice(grid)]


def far_field_ghosts(field, grid, q_inf, gamma) -> jnp.ndarray:
    """The ghosts beyond ``j = nj``, ``(4, ni, g)`` in ascending ``j``."""
    return fill_ghosts(field, grid, NGHOST, q_inf, gamma)[:, NGHOST:-NGHOST, -NGHOST:]


# --- Velocity mirroring -----------------------------------------------------


def test_mirror_velocity_reflects_about_the_surface() -> None:
    """Normal component negated, tangential untouched."""
    nx, ny = jnp.array(0.6), jnp.array(0.8)      # a unit normal
    u, v = jnp.array(1.0), jnp.array(0.5)
    um, vm = bc.mirror_velocity(jnp.stack([u, v]), jnp.stack([nx, ny]))

    assert float(um * nx + vm * ny) == pytest.approx(-float(u * nx + v * ny), rel=1e-14)
    tx, ty = -ny, nx
    assert float(um * tx + vm * ty) == pytest.approx(float(u * tx + v * ty), rel=1e-14)
    # Reflection preserves magnitude and is an involution.
    assert float(jnp.hypot(um, vm)) == pytest.approx(float(jnp.hypot(u, v)), rel=1e-14)
    again = bc.mirror_velocity(jnp.stack([um, vm]), jnp.stack([nx, ny]))
    assert float(again[0]) == pytest.approx(float(u), rel=1e-14)


# --- Slip wall --------------------------------------------------------------


def test_slip_wall_passes_no_mass(coarse_grid, subsonic_flow) -> None:
    """The defining property of an inviscid wall: zero mass flux through it.

    The ghost is the interior state mirrored about the face, so the Roe flux
    between them has its normal velocity average to exactly zero and the two
    acoustic contributions to the mass flux cancel. Energy flux vanishes for the
    same reason, leaving a pure pressure force -- which is what makes the wall
    pressure integral in post/forces.py the whole of the inviscid load.
    """
    grid, flow = coarse_grid, subsonic_flow
    rng = np.random.default_rng(0)
    # Perturbed, so the result cannot pass by symmetry alone.
    field = uniform_primitive(grid, flow) * jnp.asarray(
        1.0 + 0.1 * rng.standard_normal((4, grid.ni, grid.nj))
    )
    ghosts = wall_ghosts(field, grid)

    wall = wall_slice(grid)
    q_inner = field[:, wall, 0]
    q_ghost = ghosts[:, :, -1]                # ascending j: the last layer is j = -1
    sx, sy = grid.sx_j[wall, 0], grid.sy_j[wall, 0]

    flux = get_flux("roe")(q_ghost, q_inner, jnp.stack([sx, sy]), GAMMA)
    scale = float(jnp.max(jnp.abs(q_inner[PP] * jnp.hypot(sx, sy))))
    assert float(jnp.max(jnp.abs(flux[0]))) < 1e-13 * scale, "mass crosses the wall"
    assert float(jnp.max(jnp.abs(flux[3]))) < 1e-13 * scale, "energy crosses the wall"


def test_slip_wall_copies_density_and_pressure(coarse_grid, subsonic_flow) -> None:
    """Zero normal gradient of the scalars, which is what the wall condition is."""
    grid = coarse_grid
    field = uniform_primitive(grid, subsonic_flow) * 1.3
    ghosts = wall_ghosts(field, grid)
    wall = wall_slice(grid)
    for layer in range(NGHOST):
        for var in (PRHO, PP):
            assert np.allclose(
                np.asarray(ghosts[var, :, -1 - layer]),   # ghost j = -1-layer
                np.asarray(field[var, wall, layer]),
                rtol=0, atol=0,
            )


def test_slip_wall_uses_the_outward_normal(coarse_grid, subsonic_flow) -> None:
    """A uniform flow must be turned *away* from the body, never into it.

    With the inward normal the ghost would push flow through the surface, and the
    wall pressure force would come out with the wrong sign.
    """
    grid = coarse_grid
    field = uniform_primitive(grid, subsonic_flow)
    ghosts = wall_ghosts(field, grid)

    wall = wall_slice(grid)
    nx, ny = unit_j_normals(grid, 0)
    vn_inner = field[PU, wall, 0] * nx + field[PV, wall, 0] * ny
    vn_ghost = ghosts[PU, :, -1] * nx + ghosts[PV, :, -1] * ny
    # Exactly opposite, so the face average is zero.
    assert np.allclose(np.asarray(vn_ghost), -np.asarray(vn_inner), rtol=1e-13)
    assert float(jnp.max(jnp.abs(vn_ghost + vn_inner))) < 1e-14


# --- Characteristic far field -----------------------------------------------


def test_farfield_leaves_an_undisturbed_freestream_exactly(coarse_grid, subsonic_flow) -> None:
    """With freestream in the interior the far field must return the freestream.

    Both Riemann invariants then carry the same state, so ``Vn_b = Vn_inf`` and
    ``a_b = a_inf`` identically. If this drifts, the boundary is quietly forcing
    the solution.
    """
    grid, flow = coarse_grid, subsonic_flow
    q_inf = jnp.asarray(freestream_primitive(flow))
    block = far_field_ghosts(uniform_primitive(grid, flow), grid, q_inf, flow.gamma)
    assert block.shape == (4, grid.ni, NGHOST)
    for var in range(4):
        assert np.allclose(
            np.asarray(block[var]), float(q_inf[var]), rtol=1e-12, atol=1e-14
        ), f"far field perturbed variable {var}"


def test_farfield_transmits_an_outgoing_disturbance(coarse_grid, subsonic_flow) -> None:
    """A perturbed interior must not be snapped back to the freestream.

    Imposing the full freestream would reflect outgoing acoustic waves back into
    the domain, which is what stalls convergence on a small domain.
    """
    grid, flow = coarse_grid, subsonic_flow
    q_inf = jnp.asarray(freestream_primitive(flow))
    field = uniform_primitive(grid, flow).at[PP, :, -1].multiply(1.05)
    block = far_field_ghosts(field, grid, q_inf, flow.gamma)
    assert np.any(
        np.abs(np.asarray(block[PP, :, 0]) - float(q_inf[PP])) > 1e-6 * float(q_inf[PP])
    ), "far field clamped the interior disturbance instead of transmitting it"


@pytest.mark.parametrize("mach,alpha", [(0.5, 0.0), (0.8, 1.25), (0.3, -4.0)])
def test_farfield_is_exact_at_every_condition(coarse_grid, mach: float, alpha: float) -> None:
    """Including at incidence, where the freestream is not grid-aligned."""
    flow = FlowConditions(mach=mach, alpha_deg=alpha)
    q_inf = jnp.asarray(freestream_primitive(flow))
    block = far_field_ghosts(uniform_primitive(coarse_grid, flow), coarse_grid, q_inf, flow.gamma)
    for var in range(4):
        assert np.allclose(np.asarray(block[var]), float(q_inf[var]), rtol=1e-12, atol=1e-14)


# --- All together -----------------------------------------------------------


def test_fill_ghosts_has_the_right_shape_and_keeps_the_interior(
    coarse_grid, subsonic_flow
) -> None:
    grid, flow = coarse_grid, subsonic_flow
    q_inf = jnp.asarray(freestream_primitive(flow))
    rng = np.random.default_rng(11)
    field = uniform_primitive(grid, flow) * jnp.asarray(
        1.0 + 0.05 * rng.standard_normal((4, grid.ni, grid.nj))
    )
    padded = fill_ghosts(field, grid, NGHOST, q_inf, flow.gamma)

    assert padded.shape == (4, grid.ni + 2 * NGHOST, grid.nj + 2 * NGHOST)
    assert np.allclose(
        np.asarray(padded[:, NGHOST:-NGHOST, NGHOST:-NGHOST]), np.asarray(field), rtol=0, atol=0
    ), "ghost filling must not touch the interior"


def test_uniform_freestream_survives_every_ghost_fill(coarse_grid, subsonic_flow) -> None:
    """A uniform freestream is unchanged everywhere except in the wall velocity.

    The wall is the one boundary that legitimately alters it -- a mirrored ghost
    differs from its interior cell wherever the flow is not already parallel to the
    surface. Density and pressure are copied by every condition here, so they must
    be uniform across the whole padded array, ghosts included.
    """
    grid, flow = coarse_grid, subsonic_flow
    q_inf = jnp.asarray(freestream_primitive(flow))
    padded = fill_ghosts(
        uniform_primitive(grid, flow), grid, NGHOST, q_inf, flow.gamma
    )
    for var in (PRHO, PP):
        assert np.allclose(np.asarray(padded[var]), float(q_inf[var]), rtol=1e-12), (
            f"variable {var} disturbed by a ghost fill"
        )

    # The far-field and periodic ghosts carry the freestream velocity exactly.
    assert np.allclose(np.asarray(padded[PU, :, -NGHOST:]), float(q_inf[PU]), rtol=1e-12)


# --- The point-vortex far field ---------------------------------------------------

Q_TRANSONIC = jnp.asarray(freestream_primitive(FlowConditions(mach=0.729, alpha_deg=2.31)))


def ring(radius: float, n: int = 64) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    return radius * jnp.cos(theta), radius * jnp.sin(theta)


def test_without_circulation_the_point_vortex_is_the_freestream() -> None:
    """Bitwise: a non-lifting case must see exactly the plain far field."""
    q = bc.point_vortex_state(*ring(20.0), Q_TRANSONIC, GAMMA, jnp.asarray(0.0))
    assert np.array_equal(np.asarray(q), np.broadcast_to(np.asarray(Q_TRANSONIC)[:, None], q.shape))


def test_the_point_vortex_keeps_total_enthalpy_and_entropy() -> None:
    q = bc.point_vortex_state(*ring(5.0), Q_TRANSONIC, GAMMA, jnp.asarray(0.4))
    rho, u, v, p = q[PRHO], q[PU], q[PV], q[PP]
    rf, uf, vf, pf = Q_TRANSONIC
    enthalpy = GAMMA / (GAMMA - 1.0) * p / rho + 0.5 * (u * u + v * v)
    assert np.allclose(enthalpy, GAMMA / (GAMMA - 1.0) * pf / rf + 0.5 * (uf * uf + vf * vf), rtol=1e-14)
    assert np.allclose(p / rho**GAMMA, pf / rf**GAMMA, rtol=1e-14)


def test_the_point_vortex_is_clockwise_and_decays_as_one_over_r() -> None:
    """Positive circulation (positive lift) speeds the flow above the body and
    slows it below; the disturbance halves when the distance doubles. Along
    the freestream direction the Prandtl-Glauert factor is beta exactly."""
    circulation = jnp.asarray(0.3)
    x, y = jnp.asarray([0.0, 0.0, 10.0]), jnp.asarray([10.0, -10.0, 0.0])
    q = bc.point_vortex_state(x, y, Q_TRANSONIC, GAMMA, circulation)
    assert float(q[PU, 0]) > float(Q_TRANSONIC[PU]) > float(q[PU, 1])
    near = bc.point_vortex_state(*ring(10.0), Q_TRANSONIC, GAMMA, circulation)[PU] - Q_TRANSONIC[PU]
    far = bc.point_vortex_state(*ring(20.0), Q_TRANSONIC, GAMMA, circulation)[PU] - Q_TRANSONIC[PU]
    assert np.allclose(np.asarray(far), 0.5 * np.asarray(near), rtol=1e-12, atol=1e-16)
    mach2 = float(Q_TRANSONIC[PU] ** 2 + Q_TRANSONIC[PV] ** 2)   # a_inf = 1
    alpha = float(jnp.arctan2(Q_TRANSONIC[PV], Q_TRANSONIC[PU]))
    r = 10.0
    q = bc.point_vortex_state(jnp.asarray([r * np.cos(alpha)]), jnp.asarray([r * np.sin(alpha)]),
                              Q_TRANSONIC, GAMMA, circulation)
    induced = np.hypot(float(q[PU, 0] - Q_TRANSONIC[PU]), float(q[PV, 0] - Q_TRANSONIC[PV]))
    assert induced == pytest.approx(0.3 * np.sqrt(1.0 - mach2) / (2.0 * np.pi * r), rel=1e-13)


def test_the_circulation_is_kutta_joukowski_of_the_wall_pressure(coarse_grid) -> None:
    """A uniform field has no lift, so the far field is the freestream bitwise;
    suction on the upper wall (lift) gives a clockwise vortex: faster above."""
    flow = FlowConditions(mach=0.729, alpha_deg=2.31)
    q = uniform_primitive(coarse_grid, flow)
    q_inf = jnp.asarray(freestream_primitive(flow))
    x, y = jnp.asarray([0.25, 0.25]), jnp.asarray([30.0, -30.0])
    still = bc.point_vortex_far_field(to_canvas(q, atlas_of(coarse_grid)), atlas_of(coarse_grid), q_inf,
                                      flow.gamma)(jnp.stack([x, y]))
    assert np.array_equal(np.asarray(still), np.broadcast_to(np.asarray(q_inf)[:, None], (4, 2)))

    upper = coarse_grid.yc[:, 0] > 0.0
    lifting = q.at[PP, :, 0].set(jnp.where(upper, 0.9, 1.1) * q_inf[PP])
    moving = bc.point_vortex_far_field(to_canvas(lifting, atlas_of(coarse_grid)), atlas_of(coarse_grid), q_inf,
                                       flow.gamma)(jnp.stack([x, y]))
    assert float(moving[PU, 0]) > float(q_inf[PU]) > float(moving[PU, 1])


def test_the_quarter_chord_does_not_assume_the_chord_line(coarse_grid) -> None:
    xc, yc = atlas_of(coarse_grid).quarter_chord
    sl = wall_slice(coarse_grid)
    x_te = float(coarse_grid.x_node[sl.start, 0])
    x_le = float(coarse_grid.x_node[sl.start : sl.stop + 1, 0].min())
    assert float(xc) == pytest.approx(x_le + 0.25 * (x_te - x_le), rel=1e-14)
    assert abs(float(yc)) < 1e-12
