"""The C-grid topology: the wake-cut exchange, the outflow planes, and the
discretization on a real TMR grid.

The wake cut is the most expensive thing in this code to get wrong: an error
there does not crash and does not stall convergence, it returns the wrong lift,
and it would be blamed on the turbulence model. So the exchange is checked
against explicit partner indices on random data -- a smooth field would pass a
reversed mapping -- and for the absence of a sign flip.

The grid-level tests read TMR's coarsest NACA 0012 C-grid from ``data/grids``,
which is not tracked (see its README), and skip without it.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from jaxaero.mesh.block import Grid, validate_metrics
from jaxaero.core.state import freestream_primitive
from jaxaero.numerics import reconstruction
from jaxaero.numerics.gradients import face_gradients
from jaxaero.numerics.limiters import LIMITERS, get_limiter
from jaxaero.numerics.riemann import FLUXES, get_flux
from jaxaero.post.forces import integrate_forces, surface
from jaxaero.mesh.assemble import RawBlock, assemble

from ..conftest import uniform_state
from ..single import c_grid_sides, read_grid, Single, atlas_of, c_layout, fill_ghosts, ghost_centres, padded_centres, to_canvas

G = 2
TMR_COARSE = Path(__file__).resolve().parents[2] / "data" / "grids" / "n0012_113-33.p2dfmt"


class Layout:
    """A C-grid's counts and sides, for tests of the exchange alone."""

    def __init__(self, n_wake, n_airfoil, nj):
        self.n_wake, self.n_airfoil, self.nj = n_wake, n_airfoil, nj
        self.ni = 2 * n_wake + n_airfoil

    def sides(self):
        return c_grid_sides(self.n_wake, self.n_airfoil, self.nj)

    def wake_partner(self, i: int) -> int:
        return self.ni - 1 - i


@pytest.fixture
def topology() -> Layout:
    return Layout(n_wake=5, n_airfoil=12, nj=4)


@pytest.fixture(scope="module")
def c_grid():
    if not TMR_COARSE.is_file():
        pytest.skip(f"{TMR_COARSE.name} not on disk (data/grids is not tracked)")
    return read_grid(TMR_COARSE)


def random_interior(ni: int, nj: int, seed: int = 42) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).standard_normal((4, ni, nj)))


def physical_interior(grid, flow, seed: int = 0, amplitude: float = 0.05):
    q_inf = np.asarray(freestream_primitive(flow))
    noise = np.random.default_rng(seed).standard_normal((4, grid.ni, grid.nj))
    return jnp.asarray(q_inf[:, None, None] * (1.0 + amplitude * noise))


# --- Topology -------------------------------------------------------------------


def test_the_j_min_segments_are_cut_wall_cut(topology) -> None:
    from jaxaero.mesh.block import Boundary, Interface

    ni, nw = topology.ni, topology.n_wake
    lower, wall, upper = topology.sides().j_min
    assert ni == 2 * 5 + 12
    assert lower == Interface(0, nw, 0, "j_min", ni - nw, True)
    assert wall == Boundary("wall", nw, nw + 12)
    assert upper == Interface(ni - nw, ni, 0, "j_min", 0, True)
    assert [s.patch for s in topology.sides().j_max] == ["farfield"]


def test_wake_partner_pairs_outflow_with_outflow_and_te_with_te(topology) -> None:
    ni, nw = topology.ni, topology.n_wake
    assert topology.wake_partner(0) == ni - 1
    assert topology.wake_partner(nw - 1) == ni - nw
    assert all(topology.wake_partner(topology.wake_partner(i)) == i for i in range(ni))


def test_odd_or_empty_topologies_are_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        c_grid_sides(4, 11, 4)
    with pytest.raises(ValueError, match="n_wake"):
        c_grid_sides(0, 12, 4)


# --- The exchange ---------------------------------------------------------------


def wake_ghosts(interior, topology, n_ghost):
    """The ``j < 0`` ghosts of the lower and upper cut, ``(4, n_wake, g)`` each in
    ascending ``j``, from the ghost fill on a unit-square grid of this
    topology (only the segments matter)."""
    x, y = np.meshgrid(np.arange(topology.ni + 1.0), np.arange(topology.nj + 1.0), indexing="ij")
    grid = Grid.from_nodes(x, y, topology.sides())
    q_inf = jnp.asarray([1.0, 0.5, 0.0, 1.0 / 1.4])
    ascending = fill_ghosts(interior, grid, n_ghost, q_inf, 1.4)[:, n_ghost:-n_ghost, :n_ghost]
    nw = topology.n_wake
    return ascending[:, :nw], ascending[:, topology.ni - nw:]


@pytest.mark.parametrize("n_ghost", [2, 3])
def test_exchange_matches_explicit_partner_indices(topology, n_ghost: int) -> None:
    """Ghost ``j = -1-l`` of cell ``i`` is the partner's cell at ``j = l``, bitwise."""
    interior = random_interior(topology.ni, topology.nj)
    lower, upper = wake_ghosts(interior, topology, n_ghost)
    nw, ni = topology.n_wake, topology.ni
    assert lower.shape == upper.shape == (4, nw, n_ghost)
    for layer in range(n_ghost):
        column = n_ghost - 1 - layer          # ascending j: j = -1 is the last
        for k in range(nw):
            for block, cell in ((lower, k), (upper, ni - nw + k)):
                want = interior[:, topology.wake_partner(cell), layer]
                assert np.array_equal(np.asarray(block[:, k, column]), np.asarray(want))


def test_exchange_applies_no_sign_flip(topology) -> None:
    """Momentum is Cartesian, so the exchange copies. With a strictly positive
    field any negation is a sign change, not a subtle magnitude error."""
    interior = jnp.abs(random_interior(topology.ni, topology.nj, seed=7)) + 1.0
    lower, upper = wake_ghosts(interior, topology, G)
    assert bool(jnp.all(lower > 0.0)) and bool(jnp.all(upper > 0.0))


def test_exchange_reverses_the_i_direction(topology) -> None:
    """An unreversed copy pairs outflow with trailing edge: wrong everywhere but
    the midpoint, and the subtle version of the bug."""
    interior = random_interior(topology.ni, topology.nj)
    lower, _ = wake_ghosts(interior, topology, G)
    nw, ni = topology.n_wake, topology.ni
    naive = np.asarray(interior[:, ni - nw : ni, 0])
    assert not np.allclose(np.asarray(lower[:, :, -1]), naive)
    assert np.array_equal(np.asarray(lower[:, :, -1]), naive[:, ::-1])


# --- Topology inference -------------------------------------------------------------


def c_line(n_wake: int = 6, n_half: int = 8, tilt_deg: float = -3.0, length: float = 20.0):
    """The ``j = 0`` node line of a C-grid, ``(2 n_wake + 2 n_half + 1,)`` each:
    a wake from ``length`` chords downstream along ``tilt_deg`` into the trailing
    edge at (1, 0), an ellipse clockwise (lower surface first) round the leading
    edge, and the same wake back out. The inference reads only this line."""
    s = np.linspace(1.0, 0.0, n_wake + 1) ** 2 * length          # TE-clustered
    t = np.deg2rad(tilt_deg)
    wx, wy = 1.0 + s * np.cos(t), s * np.sin(t)
    theta = np.linspace(0.0, 2.0 * np.pi, 2 * n_half + 1)[1:-1]
    ex, ey = 0.5 + 0.5 * np.cos(theta), -0.06 * np.sin(theta)
    return np.r_[wx, ex, wx[::-1]], np.r_[wy, ey, wy[::-1]]


def as_grid(x0: np.ndarray, y0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two node rows: the line, and the line moved to its left (the fluid
    side of a clockwise C-grid), so the two wake sides separate."""
    tx, ty = np.gradient(x0), np.gradient(y0)
    length = np.hypot(tx, ty)
    nx, ny = -ty / length, tx / length
    h = 0.05 * np.minimum(1.0, np.hypot(x0 - 0.5, y0))
    return np.stack([x0, x0 + h * nx], axis=1), np.stack([y0, y0 + h * ny], axis=1)


def j_min_of(x, y):
    mesh, _ = assemble([RawBlock(x, y)])
    return mesh[0].sides.j_min


@pytest.mark.parametrize("tilt_deg", [0.0, -3.0, 7.0])
def test_a_wake_off_the_x_axis_is_found(tilt_deg) -> None:
    """The NPARC RAE 2822 grid's wake follows the trailing-edge bisector, not
    ``y = 0``; the cut is found by coincident faces, wherever it lies."""
    assert j_min_of(*as_grid(*c_line(tilt_deg=tilt_deg))) == c_grid_sides(6, 16, 1).j_min


def test_the_assembly_does_not_depend_on_position_scale_or_rotation() -> None:
    x0, y0 = c_line()
    a = np.deg2rad(25.0)
    xr = 3.0 * (np.cos(a) * x0 - np.sin(a) * y0) - 40.0
    yr = 3.0 * (np.sin(a) * x0 + np.cos(a) * y0) + 7.0
    assert j_min_of(*as_grid(xr, yr)) == j_min_of(*as_grid(x0, y0))


def test_a_left_handed_grid_is_turned_round() -> None:
    """Walked the other way (i reversed), the block is made right-handed and
    comes out the same."""
    x, y = as_grid(*c_line())
    assert j_min_of(x[::-1].copy(), y[::-1].copy()) == c_grid_sides(6, 16, 1).j_min


def test_an_open_wake_is_not_a_cut() -> None:
    """Upper wake moved off the lower one: no coincident faces, no interface;
    the two wake lines become boundary faces."""
    x0, y0 = c_line()
    x, y = as_grid(x0, y0)
    y[-7:] += 0.01                                  # the upper wake nodes, both rows
    segments = j_min_of(x, y)
    assert not any(hasattr(s, "partner_start") for s in segments)


NPARC_RAE2822 = TMR_COARSE.parent / "rae2822_nparc.x.fmt"


def test_the_nparc_rae2822_grid_has_its_documented_topology() -> None:
    """NPARC: the trailing edge at I33 and I337 (1-based), the wake coupled
    I1-I32 to I338-I369, 369 x 65 nodes."""
    if not NPARC_RAE2822.is_file():
        pytest.skip(f"{NPARC_RAE2822.name} not on disk (data/grids is not tracked)")
    grid = read_grid(NPARC_RAE2822)
    assert grid.sides == c_grid_sides(32, 304, 64)
    assert validate_metrics(grid).ok


# --- On a TMR grid ----------------------------------------------------------------


def test_the_tmr_grid_has_the_documented_topology(c_grid) -> None:
    assert c_grid.sides == c_grid_sides(24, 64, 32)
    assert validate_metrics(c_grid).ok


def test_ghost_centres_across_the_wake_are_the_partners_own(c_grid) -> None:
    """Bitwise: the gradient across the cut must see the true geometry."""
    topo = c_layout(c_grid)
    x, y = padded_centres(c_grid)
    for i in (*range(topo.n_wake), *range(topo.ni - topo.n_wake, topo.ni)):
        partner = topo.wake_partner(i)
        assert float(x[1 + i, 0]) == float(c_grid.xc[partner, 0])
        assert float(y[1 + i, 0]) == float(c_grid.yc[partner, 0])


def test_face_gradients_are_exact_for_linear_fields(c_grid) -> None:
    """On every face: wall, wake cut, far field and both outflow planes.

    The bound is the round-off of a difference quotient, ``eps |x| / h``, with
    ``|x|`` up to ~700 chords and ``h`` down to the smallest wall cell.
    """
    x, y = padded_centres(c_grid)
    (gx_i, gy_i), (gx_j, gy_j) = face_gradients(jnp.stack([1.0 + 0.3 * x - 0.7 * y]), ghost_centres(c_grid))
    bound = 10.0 * np.finfo(float).eps * float(jnp.max(jnp.abs(x))) / float(
        jnp.sqrt(c_grid.area.min()))
    for got, want in ((gx_i, 0.3), (gy_i, -0.7), (gx_j, 0.3), (gy_j, -0.7)):
        assert float(jnp.max(jnp.abs(got - want))) < bound


@pytest.mark.parametrize("flux", sorted(FLUXES))
@pytest.mark.parametrize("limiter", sorted(LIMITERS))
def test_freestream_is_preserved_everywhere_but_the_wall(c_grid, flux, limiter) -> None:
    """Including the wake-cut cells: they are interior cells and must not be
    masked. Only the cells over the airfoil, within the MUSCL stencil of the
    wall, may feel a uniform flow."""
    flow = FlowConditions(mach=0.15, alpha_deg=10.0)
    operator = Single(c_grid, flow, NumericsConfig(flux=flux, limiter=limiter))
    residual = np.abs(np.asarray(operator.residual(uniform_state(c_grid, flow))))
    residual[:, c_layout(c_grid).wall, :2] = 0.0
    assert residual.max() < 1e-12


def test_viscous_freestream_is_preserved_everywhere_but_the_wall(c_grid) -> None:
    """As above, plus the two wake cells at the trailing edge, ``j = 0``.

    Their i-face ends on the trailing-edge node, and the diamond gradient there
    averages the four cells around that node -- one of them the first wall
    cell's no-slip ghost. A uniform flow past a no-slip body really has a
    velocity gradient at that node, so those two cells must feel it; every
    other wake cell must not.
    """
    flow = FlowConditions(mach=0.15, alpha_deg=10.0, reynolds=6.0e6)
    topo = c_layout(c_grid)
    operator = Single(c_grid, flow, NumericsConfig(limiter="none"))
    residual = np.abs(np.asarray(operator.residual(uniform_state(c_grid, flow))))
    wall = residual[:, topo.wall, 0].copy()
    trailing_edge = residual[1:3, [topo.n_wake - 1, topo.ni - topo.n_wake], 0].copy()
    residual[:, topo.wall, :2] = 0.0
    residual[:, [topo.n_wake - 1, topo.ni - topo.n_wake], 0] = 0.0
    assert residual.max() < 1e-12
    assert wall.max() > 1e-8, "the no-slip wall is not disturbing a uniform flow"
    assert trailing_edge.min() > 1e-12, "the trailing-edge node is not seen from the wake"


@pytest.mark.parametrize("first_order", [True, False])
def test_the_wake_cut_is_conservative(c_grid, first_order: bool) -> None:
    """The flux leaving a cell through the cut enters its partner, to round-off.

    Each wake face is computed twice, once from each side, with opposite area
    vectors and the states swapped; a consistent flux makes the two cancel.
    """
    flow = FlowConditions(mach=0.5, alpha_deg=3.0)
    q_inf = jnp.asarray(freestream_primitive(flow))
    interior = physical_interior(c_grid, flow)
    padded = fill_ghosts(interior, c_grid, G, q_inf, flow.gamma)
    ni, nj, topo = c_grid.ni, c_grid.nj, c_layout(c_grid)
    if first_order:
        ql, qr = reconstruction.first_order_faces(padded, G, (ni, nj), 1)
    else:
        ql, qr = reconstruction.face_states(padded, G, (ni, nj), get_limiter("minmod"), 1)
    flux = np.asarray(get_flux("hllc")(ql, qr, c_grid.faces[1], flow.gamma))[:, :, 0]
    lower = np.arange(topo.n_wake)
    partner = topo.ni - 1 - lower
    scale = np.abs(flux).max()
    assert np.abs(flux[:, lower] + flux[:, partner]).max() < 1e-12 * scale
    # The complement: the wall faces carry pressure, so they do not cancel.
    assert np.abs(flux[:, topo.wall]).max() > 1e-6 * scale


def test_outflow_planes_leave_a_freestream_alone(c_grid) -> None:
    """The characteristic condition reproduces an undisturbed freestream
    exactly on the outflow planes, as on the outer boundary."""
    flow = FlowConditions(mach=0.15, alpha_deg=10.0)
    q_inf = jnp.asarray(freestream_primitive(flow))
    interior = jnp.broadcast_to(q_inf[:, None, None], (4, c_grid.ni, c_grid.nj))
    padded = fill_ghosts(interior, c_grid, G, q_inf, flow.gamma)
    for column in (*range(G), *range(-G, 0)):
        block = np.asarray(padded[:, column, G:-G])
        assert np.abs(block - np.asarray(q_inf)[:, None]).max() < 1e-14


def test_wall_quantities_cover_the_airfoil_only(c_grid) -> None:
    """Cp, Cf and the stations span the wall faces, not the wake cut."""
    atlas = atlas_of(c_grid)
    viscous = FlowConditions(mach=0.15, alpha_deg=0.0, reynolds=6.0e6)
    wall = surface(to_canvas(uniform_state(c_grid, viscous), atlas), atlas, viscous)
    assert wall["x"].shape == wall["cf"].shape == (64,)
    assert wall["x"].min() > 0.0 and wall["x"].max() < 1.0
    # Uniform pressure on the closed wall polygon: no net force, to round-off.
    # Were the wake faces included, the open polyline would not close.
    inviscid = FlowConditions(mach=0.15, alpha_deg=0.0)
    forces = integrate_forces(to_canvas(uniform_state(c_grid, inviscid), atlas), atlas, inviscid)
    assert abs(forces.cl) < 1e-12 and abs(forces.cd) < 1e-12


# --- Lines through the cut ------------------------------------------------------------


@pytest.mark.parametrize("which", ["wake", "airfoil"])
def test_line_chains_run_through_the_cut_with_the_defining_blocks(c_grid, which) -> None:
    """A wake column's chain continues across the cut into its partner
    ``ni - 1 - i``; an airfoil column's stops at the wall. Every coupling on
    the chain is ``1/2 A(U_nb, S_out) - 1/2 lambda_f I``, with ``S_out`` taken
    here from the cell's own face toward the neighbour, independently of how
    the chains were built."""
    from jaxaero.core.state import primitive_to_conservative, split
    from jaxaero.physics.euler import normal_flux_jacobian
    from jaxaero.solvers import preconditioners as pre

    atlas, g = atlas_of(c_grid), atlas_of(c_grid).g
    flow = FlowConditions(mach=0.5, alpha_deg=3.0)
    gamma, cfl = flow.gamma, jnp.asarray(7.0)
    q = physical_interior(c_grid, flow, amplitude=0.1)
    U = to_canvas(primitive_to_conservative(*split(q, 2), gamma), atlas)
    single = Single(c_grid, flow, NumericsConfig())
    lam_i, lam_j = single.op.wave_speeds(U)
    layout = pre.ChainLayout(atlas)
    lower, diag, upper = (np.asarray(a) for a in pre.flow_chain_blocks(U, (lam_i, lam_j), cfl, atlas, gamma, layout))

    topo, nj = c_layout(c_grid), c_grid.nj
    i = 3 if which == "wake" else topo.n_wake + 5
    cells = np.asarray(atlas.lines.cells)
    n = int(np.argwhere(cells == (g + i) * atlas.nj + g)[0, 0])
    chain = [(int(c) // atlas.nj - g, int(c) % atlas.nj - g) for c in cells[n] if c >= 0]
    if which == "wake":
        assert sorted(chain) == sorted([(i, j) for j in range(nj)] + [(topo.wake_partner(i), j) for j in range(nj)])
    else:
        assert sorted(chain) == [(i, j) for j in range(nj)]

    qc = np.asarray(q)
    sx, sy, lj = np.asarray(c_grid.sx_j), np.asarray(c_grid.sy_j), np.asarray(lam_j)[g:-g, g:]

    def expected(a, b):
        (ia, ja), (ib, jb) = a, b
        if jb == ja + 1:
            s_out, face = (sx[ia, ja + 1], sy[ia, ja + 1]), (ia, ja + 1)
        else:                                      # down, or across the cut from j = 0
            s_out, face = (-sx[ia, ja], -sy[ia, ja]), (ia, ja)
        A = np.asarray(normal_flux_jacobian(jnp.asarray(qc[:, ib, jb]), jnp.asarray(s_out), gamma))
        return 0.5 * A - 0.5 * lj[face] * np.eye(4)

    for k in range(len(chain) - 1):
        assert np.allclose(upper[k, n], expected(chain[k], chain[k + 1]), rtol=1e-12, atol=1e-14)
        assert np.allclose(lower[k + 1, n], expected(chain[k + 1], chain[k]), rtol=1e-12, atol=1e-14)
    r = np.random.default_rng(5).standard_normal((4,) + atlas.shape)
    z = np.asarray(pre.get_preconditioner("line_j")(atlas, gamma)(U, (lam_i, lam_j), cfl)(jnp.asarray(r)))
    L = len(chain)
    M = np.zeros((4 * L, 4 * L))
    for k in range(L):
        M[4 * k:4 * k + 4, 4 * k:4 * k + 4] = diag[k, n]
        if k > 0:
            M[4 * k:4 * k + 4, 4 * (k - 1):4 * k] = lower[k, n]
        if k < L - 1:
            M[4 * k:4 * k + 4, 4 * (k + 1):4 * k + 8] = upper[k, n]
    flat = [cells[n, k] for k in range(L)]
    b = np.concatenate([r.reshape(4, -1)[:, c] for c in flat])
    got = np.concatenate([z.reshape(4, -1)[:, c] for c in flat])
    want = np.linalg.solve(M, b)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 * np.abs(want).max())
