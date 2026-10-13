"""The SA-RANS discretization: the fifth row, its boundary values, the eddy
viscosity in the mean flow, and its preconditioner.

The strongest cheap check is the free stream. Uniform flow with ``nu~ = 3
nu_inf`` everywhere has no gradients, so the mean-flow rows must vanish away
from the wall exactly as for laminar flow, and the turbulence row must equal
``-A rho (P - D) / nu_inf`` at every cell -- the source alone, since
convection and diffusion of a uniform field cancel. That ties the assembled
row to the pointwise model (tests/unit/test_spalart_allmaras.py) on every
topology. Grids from TMR's files skip when not on disk.
"""

from __future__ import annotations

import glob
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from validation.grids.karman_trefftz import TMR, make_o_grid

from ..single import PLATE_RULES, Single, atlas_of, fill_ghosts, from_canvas, n_symmetry, read_grid, to_canvas, wall_slice
from jaxaero.numerics.residual import make_residual, primitive_from_conservative
from jaxaero.numerics.turbulence import face_eddy_viscosity
from jaxaero.physics import spalart_allmaras as sa
from jaxaero.physics.navier_stokes import Viscosity
from jaxaero.solvers.explicit import initial_state

NVAR_RANS, P_NU, RHO_NU = 5, 4, 4     # 2D: Rows(2).nu

GRIDS = Path(__file__).resolve().parents[2] / "data" / "grids"


def flow(mach=0.15, alpha=10.0, reynolds=6e6):
    return FlowConditions(mach, alpha, reynolds, sutherland_ratio=198.6 / 540.0,
                          turbulence="sa_noft2")


def o_grid():
    return make_o_grid(32, 44, 4e-4, section=TMR)


def c_grid():
    path = GRIDS / "n0012_113-33.p2dfmt"
    if not path.is_file():
        pytest.skip("TMR C-grid not on disk")
    return read_grid(path)


def flat_plate():
    paths = glob.glob(str(GRIDS / "flatplate*35x25*"))
    if not paths:
        pytest.skip("TMR flat-plate grid not on disk")
    return read_grid(paths[0], PLATE_RULES)


TOPOLOGIES = {"o_grid": o_grid, "c_grid": c_grid, "flat_plate": flat_plate}


def wall_influenced(grid) -> np.ndarray:
    """Cells a uniform flow may disturb: within two of the wall, plus the
    cells sharing the wall's end nodes (the C-grid trailing edge, the plate's
    leading edge), which the diamond face gradient reaches through the node."""
    mask = np.zeros((grid.ni, grid.nj), bool)
    wall = wall_slice(grid)
    mask[wall, :2] = True
    for i in (wall.start - 1, wall.stop):
        if 0 <= i < grid.ni:
            mask[i, :2] = True
    return mask


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_freestream_rows_and_the_analytic_source(name) -> None:
    grid, fl = TOPOLOGIES[name](), flow(alpha=0.0 if name == "flat_plate" else 10.0)
    op = Single(grid, fl, NumericsConfig(limiter="none"))
    U = op.initial_state()
    assert U.shape == (NVAR_RANS, grid.ni, grid.nj)
    R = np.asarray(op.residual(U))
    off = ~wall_influenced(grid)
    assert np.abs(R[:4][:, off]).max() < 1e-12

    nu_inf = op.model.nu_inf
    production, destruction = sa.source(
        jnp.asarray(3.0 * nu_inf), jnp.asarray(nu_inf), jnp.asarray(0.0), from_canvas(op.distance, op.atlas)
    )
    expected = -np.asarray(grid.area) * np.asarray(production - destruction) / nu_inf
    scale = np.abs(expected[off]).max()
    assert np.abs(R[4][off] - expected[off]).max() < 1e-9 * scale


def test_nu_tilde_and_the_eddy_viscosity_vanish_on_the_wall() -> None:
    """The odd ghost makes the face value exactly zero; mu_t built from the
    face values is then exactly zero there too, while the first cell's is not."""
    grid, fl = o_grid(), flow()
    op = Single(grid, fl, NumericsConfig(limiter="none"))
    U = np.asarray(op.initial_state()).copy()
    U[RHO_NU] *= 1.0 + 50.0 * np.random.default_rng(0).random((grid.ni, grid.nj))
    q = primitive_from_conservative(jnp.asarray(U), fl.gamma, 2)
    padded = fill_ghosts(q, grid, 2, op.q_inf, fl.gamma, no_slip=True)
    face = 0.5 * (padded[P_NU, 2:-2, 1] + padded[P_NU, 2:-2, 2])
    assert float(jnp.max(jnp.abs(face))) == 0.0
    _, mu_t_j = face_eddy_viscosity(padded[:, 1:-1, 1:-1], Viscosity.of(fl), fl.gamma, op.model, 2)
    assert float(jnp.max(jnp.abs(mu_t_j[:, 0]))) == 0.0
    assert float(jnp.min(mu_t_j[:, 1])) > 0.0


def test_symmetry_plane_keeps_nu_tilde_and_the_wall_does_not() -> None:
    """Ahead of the plate the ghost nu~ is the cell's (even); on it, minus."""
    grid = flat_plate()
    rng = np.random.default_rng(3)
    q = jnp.asarray(1.0 + 0.1 * rng.random((5, grid.ni, grid.nj)))
    q_inf = jnp.asarray([1.0, 0.2, 0.0, 1 / 1.4, 3.0])
    padded = fill_ghosts(q, grid, 2, q_inf, 1.4, no_slip=True)
    n_sym = n_symmetry(grid)
    ghost, cell = np.asarray(padded[P_NU, 2:-2, 1]), np.asarray(q[P_NU, :, 0])
    assert np.array_equal(ghost[:n_sym], cell[:n_sym])
    assert np.array_equal(ghost[n_sym:], -cell[n_sym:])


def test_the_sa_line_preconditioner_inverts_its_own_blocks() -> None:
    """Dense check on the chain through one wake pair of the C-grid."""
    from jaxaero.solvers import preconditioners as pre

    grid, fl = c_grid(), flow()
    single = Single(grid, fl, NumericsConfig(limiter="none"))
    op, atlas = single.op, single.atlas
    U = np.asarray(single.initial_state()).copy()
    rng = np.random.default_rng(4)
    U[RHO_NU] *= 1.0 + 100.0 * rng.random((grid.ni, grid.nj))
    U[1] *= 1.0 + 0.2 * rng.standard_normal((grid.ni, grid.nj))
    U = to_canvas(jnp.asarray(U), atlas)
    cfl = jnp.asarray(20.0)
    lambdas = op.wave_speeds(U)
    layout = pre.ChainLayout(atlas)
    lower, diag, upper = (np.asarray(a) for a in pre.sa_chain_blocks(
        U, lambdas, cfl, atlas, fl.gamma, op.viscosity, op.model, op.distance, layout))
    r = rng.standard_normal((5,) + atlas.shape)
    z = np.asarray(pre.get_preconditioner("line_j")(
        atlas, fl.gamma, op.viscosity, op.model, op.distance)(U, lambdas, cfl)(jnp.asarray(r)))

    g, i = atlas.g, 3
    cells = np.asarray(atlas.lines.cells)
    n = int(np.argwhere(cells == (g + i) * atlas.nj + g)[0, 0])
    chain = [int(c) for c in cells[n] if c >= 0]
    assert len(chain) == 2 * grid.nj, "the wake chain runs through the cut"
    L = len(chain)
    M = np.zeros((L, L))
    for k in range(L):
        M[k, k] = diag[k, n, 0, 0]
        if k > 0:
            M[k, k - 1] = lower[k, n, 0, 0]
        if k < L - 1:
            M[k, k + 1] = upper[k, n, 0, 0]
    want = np.linalg.solve(M, r.reshape(5, -1)[4, chain])
    got = z.reshape(5, -1)[4, chain]
    assert np.allclose(got, want, rtol=1e-10, atol=1e-14 * np.abs(want).max())
    assert np.all(np.diag(M) > 0.0) and np.all(np.abs(np.diag(M, 1)) > 0.0)


def test_an_implicit_step_on_a_rans_case_stays_finite() -> None:
    """Five rows through the JVP, GMRES and both preconditioners."""
    import jax

    from jaxaero.core.config import ImplicitConfig
    from jaxaero.solvers import implicit

    grid, fl = o_grid(), flow()
    num = NumericsConfig(limiter="none", solver="implicit")
    op = make_residual(atlas_of(grid, 2, True), fl, num)
    step = jax.jit(implicit.make_newton_step(op, ImplicitConfig(), make_residual(atlas_of(grid, 2, True), fl, num, True)))
    state = implicit.initial_state(op, ImplicitConfig())
    for _ in range(3):
        state, norm = step(state)
    assert state.U.shape[0] == NVAR_RANS and bool(jnp.all(jnp.isfinite(state.U)))
    assert np.isfinite(float(norm))


def test_update_relaxation_bounds_every_relative_change() -> None:
    """omega scales the whole update so the largest relative change of rho,
    p (linearized) or nu~ is exactly the fraction; small updates pass whole."""
    from jaxaero.solvers.implicit import relax

    grid, fl = o_grid(), flow()
    U = initial_state(make_residual(atlas_of(grid, 2, True), fl, NumericsConfig(limiter="none"))).U
    rng = np.random.default_rng(8)
    dU = jnp.asarray(rng.standard_normal(U.shape)) * jnp.abs(U) * 0.5
    scaled, omega = relax(dU, U, fl.gamma, 0.2)
    assert 0.0 < float(omega) < 1.0
    d_rho = np.abs(np.asarray(scaled[0] / U[0]))
    assert d_rho.max() <= 0.2 + 1e-12
    small, one = relax(1e-6 * dU, U, fl.gamma, 0.2)
    assert float(one) == 1.0 and np.array_equal(np.asarray(small), np.asarray(1e-6 * dU))
    untouched, one = relax(dU, U, fl.gamma, None)
    assert untouched is dU and float(one) == 1.0


@pytest.mark.parametrize("dim", [2, 3])
def test_vorticity_has_a_finite_derivative_where_it_vanishes(dim) -> None:
    """A freestream start has zero vorticity; the JVPs of the implicit solver
    must stay finite there (3D: |curl V| is a square root)."""
    import jax

    from jaxaero.numerics.turbulence import vorticity_magnitude

    g = jnp.zeros((dim, dim, 5))
    _, tangent = jax.jvp(vorticity_magnitude, (g,), (jnp.ones_like(g),))
    assert np.array_equal(np.asarray(tangent), np.zeros(5))
    g = jnp.asarray(np.random.default_rng(1).standard_normal((dim, dim, 5)))
    assert bool(jnp.all(jnp.isfinite(jax.jvp(vorticity_magnitude, (g,), (jnp.ones_like(g),))[1])))
