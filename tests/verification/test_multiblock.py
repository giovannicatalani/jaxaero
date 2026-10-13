"""Split-block regression: the gate for every multi-block feature.

A single-block grid cut into blocks along ``i`` and ``j`` is the same
discretization written differently: every face sees the same two cells, so the
residual of the N-block atlas must be the 1-block atlas's, cell for cell. That
tests the interface exchange (both stages, the corners, reversed partners, a
cut running through the C-grid's wake pairing) against a known answer.

``split`` builds each block's sides from the original grid's, cell by cell:
inside the original grid a side faces the neighbouring block; on its boundary
it inherits the original segment, with an interface's partner mapped to the
block that now holds that cell.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, ImplicitConfig, NumericsConfig
from jaxaero.mesh.block import BlockSides, Boundary, Grid, Interface
from jaxaero.mesh.mesh import assign_conditions, wall_distance

from ..single import Single, read_grid
from validation.grids.karman_trefftz import TMR, make_o_grid
from jaxaero.mesh.atlas import build_atlas
from jaxaero.numerics.residual import make_residual
from jaxaero.solvers import implicit

GRIDS = Path(__file__).resolve().parents[2] / "data" / "grids"


# --- Cutting a grid into blocks ---------------------------------------------------


def split(grid: Grid, i_cuts: tuple[int, ...], j_cuts: tuple[int, ...]):
    """Cut ``grid`` at cell lines ``i_cuts`` and ``j_cuts``.

    Returns ``(mesh, windows)``: the blocks, row-major in ``(bi, bj)``, and
    each block's ``(i-slice, j-slice)`` of the original cells.
    """
    sides = grid.sides
    ie = (0, *i_cuts, grid.ni)
    je = (0, *j_cuts, grid.nj)
    nbi, nbj = len(ie) - 1, len(je) - 1
    index = lambda bi, bj: bi * nbj + bj  # noqa: E731

    def owner(side: str, cell: int) -> tuple[int, str, int]:
        """The block holding original boundary cell ``cell`` of ``side``, and
        the cell's index along that block's side."""
        if side.startswith("j"):
            bi = int(np.searchsorted(ie, cell, side="right")) - 1
            return index(bi, 0 if side == "j_min" else nbj - 1), side, cell - ie[bi]
        bj = int(np.searchsorted(je, cell, side="right")) - 1
        return index(0 if side == "i_min" else nbi - 1, bj), side, cell - je[bj]

    def original(side: str, cell: int):
        for seg in sides.side(side):
            if seg.start <= cell < seg.stop:
                if isinstance(seg, Boundary):
                    return (seg.patch, seg.kind)
                return owner(seg.side, int(seg.partner_cells()[cell - seg.start]))
        raise AssertionError

    def compress(cells: list) -> tuple:
        """Per-cell targets -> segments."""
        segments, k = [], 0
        while k < len(cells):
            target = cells[k]
            if len(target) == 2:                         # (patch, kind)
                stop = k
                while stop < len(cells) and cells[stop] == target:
                    stop += 1
                segments.append(Boundary(target[0], k, stop, target[1]))
            else:
                block, side, p = target
                step, stop = None, k + 1
                while stop < len(cells) and len(cells[stop]) == 3 \
                        and cells[stop][:2] == (block, side) \
                        and cells[stop][2] - cells[stop - 1][2] == (step or cells[stop][2] - p) \
                        and abs(cells[stop][2] - cells[stop - 1][2]) == 1:
                    step = cells[stop][2] - cells[stop - 1][2]
                    stop += 1
                last = cells[stop - 1][2]
                segments.append(Interface(k, stop, block, side, min(p, last), step == -1))
            k = stop
        return tuple(segments)

    mesh, windows = [], []
    for bi in range(nbi):
        for bj in range(nbj):
            i0, i1, j0, j1 = ie[bi], ie[bi + 1], je[bj], je[bj + 1]
            spec = {}
            for side, inner, along, lo, neighbour in (
                ("j_min", bj > 0, range(i0, i1), i0, (bi, bj - 1, "j_max")),
                ("j_max", bj < nbj - 1, range(i0, i1), i0, (bi, bj + 1, "j_min")),
                ("i_min", bi > 0, range(j0, j1), j0, (bi - 1, bj, "i_max")),
                ("i_max", bi < nbi - 1, range(j0, j1), j0, (bi + 1, bj, "i_min")),
            ):
                if inner:
                    nb, nside = index(*neighbour[:2]), neighbour[2]
                    spec[side] = (Interface(0, len(along), nb, nside, 0, False),)
                else:
                    spec[side] = compress([original(side, c) for c in along])
            topology = BlockSides(i1 - i0, j1 - j0, **spec)
            mesh.append(Grid.from_nodes(grid.x_node[i0:i1 + 1, j0:j1 + 1],
                                        grid.y_node[i0:i1 + 1, j0:j1 + 1], topology))
            windows.append((slice(i0, i1), slice(j0, j1)))
    return tuple(mesh), windows


def cut(field: jnp.ndarray, windows) -> tuple:
    return tuple(field[..., i, j] for i, j in windows)


def join(blocks, windows, like: jnp.ndarray) -> np.ndarray:
    out = np.full(like.shape, np.nan)
    for block, (i, j) in zip(blocks, windows):
        out[..., i, j] = np.asarray(block)
    return out


# --- The grids and flows -----------------------------------------------------


def o_grid():
    return make_o_grid(32, 44, 4e-4, section=TMR)


def c_grid():
    path = GRIDS / "n0012_113-33.p2dfmt"
    if not path.is_file():
        pytest.skip("TMR C-grid not on disk")
    return read_grid(path)


CUTS = {
    # O-grid: through the seam's neighbourhood, the wall, both directions.
    "o": ((5, 16, 29), (3, 20)),
    # C-grid 112 x 32, n_wake 24: one cut inside each wake segment (their
    # pairing then crosses blocks), one on the wall, one at the trailing edge.
    "c": ((10, 24, 56, 95), (7,)),
}

FLOWS = {
    "euler": FlowConditions(0.6, 3.0),
    "sa": FlowConditions(0.2, 8.0, 6e6, sutherland_ratio=0.3678, turbulence="sa_noft2"),
}


def perturbed(single: Single, seed=1):
    U = single.initial_state()
    rng = np.random.default_rng(seed)
    return U * (1.0 + 0.02 * jnp.asarray(rng.standard_normal(U.shape)))


@pytest.fixture(scope="module", params=["o", "c"])
def case(request):
    grid = o_grid() if request.param == "o" else c_grid()
    grid = assign_conditions(grid, {"wall": "no_slip_wall", "farfield": "far_field"})[0]
    mesh, windows = split(grid, *CUTS[request.param])
    return request.param, grid, mesh, windows


@pytest.fixture(scope="module")
def atlas(case):
    """``atlas(no_slip)``: the split mesh packed, its wall as the flow needs."""
    built = {}

    def make(no_slip: bool = True):
        if no_slip not in built:
            wall = "no_slip_wall" if no_slip else "slip_wall"
            built[no_slip] = build_atlas(assign_conditions(case[2], {"wall": wall, "farfield": "far_field"}))
        return built[no_slip]

    return make


def residuals(grid, atlas, windows, flow, first_order):
    """The 1-block residual and the N-block one, joined, on a perturbed state."""
    atlas = atlas(not FLOWS[flow].inviscid)
    numerics = NumericsConfig(limiter="van_albada")
    single = Single(grid, FLOWS[flow], numerics, first_order=first_order)
    other = make_residual(atlas, FLOWS[flow], numerics, first_order=first_order)
    U = perturbed(single)
    got = atlas.unpack(jax.jit(other.residual)(atlas.pack(cut(U, windows))))
    return np.asarray(jax.jit(single.residual)(U)), join(got, windows, U)


# --- The gate ------------------------------------------------------------------


def test_the_blocks_tile_the_grid_and_every_interface_is_mutual(case) -> None:
    _, grid, mesh, windows = case
    assert sum(b.ni * b.nj for b in mesh) == grid.ni * grid.nj
    for b, block in enumerate(mesh):
        for side in ("j_min", "j_max", "i_min", "i_max"):
            for seg in block.sides.side(side):
                if isinstance(seg, Interface):
                    back = [s for s in mesh[seg.block].sides.side(seg.side)
                            if isinstance(s, Interface) and s.block == b and s.side == side]
                    assert back, (b, side, seg)


def test_wall_distance_is_the_single_block_one(case) -> None:
    _, grid, mesh, windows = case
    assert np.array_equal(join(wall_distance(mesh), windows, grid.area), wall_distance(grid)[0])


@pytest.mark.parametrize("flow,first_order", [
    ("euler", True), ("sa", False),
    pytest.param("euler", False, marks=pytest.mark.slow), pytest.param("sa", True, marks=pytest.mark.slow),
])
def test_the_residual_is_the_single_block_one(case, atlas, flow, first_order) -> None:
    """Measured 4e-15 at most (a few ulps: XLA fuses the two canvases differently)."""
    name, grid, mesh, windows = case
    want, got = residuals(grid, atlas, windows, flow, first_order)
    scale = np.abs(want).max(axis=(1, 2), keepdims=True)
    assert np.max(np.abs(got - want) / scale) < 1e-13, np.max(np.abs(got - want) / scale)


def test_norm_and_time_step_are_the_single_block_ones(case, atlas) -> None:
    name, grid, mesh, windows = case
    atlas = atlas()
    numerics = NumericsConfig(limiter="van_albada")
    single = Single(grid, FLOWS["sa"], numerics)
    packed = make_residual(atlas, FLOWS["sa"], numerics)
    U = perturbed(single)
    Uc = jnp.where(atlas.interior, atlas.pack(cut(U, windows)), packed.initial_state())
    assert float(packed.norm(packed.residual(Uc))) == pytest.approx(
        float(single.norm(single.residual(U))), rel=1e-12)
    assert np.allclose(join(atlas.unpack(packed.time_step(Uc)), windows, U[0]),
                       np.asarray(single.time_step(U)), rtol=1e-14, atol=0)


def test_an_implicit_step_on_the_atlas_is_accepted_and_leaves_the_gaps_alone(case, atlas) -> None:
    """The preconditioner differs (lines stop at the cuts), so the step does too;
    it must still be a valid, accepted step, and every canvas cell that is no
    block's must keep its value."""
    _, grid, mesh, windows = case
    atlas = atlas()
    flow = FLOWS["sa"]
    numerics = NumericsConfig(limiter="van_albada", solver="implicit")
    operator = make_residual(atlas, flow, numerics)
    jacobian = make_residual(atlas, flow, numerics, first_order=True, distance=operator.distance)
    config = ImplicitConfig(cfl=10.0)
    state = implicit.initial_state(operator, config)
    U0 = state.U
    state = state._replace(U=jnp.where(atlas.interior, atlas.pack(cut(perturbed(
        Single(grid, flow, numerics)), windows)), U0))
    new, norm = jax.jit(implicit.make_newton_step(operator, config, jacobian))(state)
    assert bool(jnp.all(jnp.isfinite(new.U))) and np.isfinite(float(norm))
    assert float(new.residual_norm) == float(norm), "the step was rejected"
    assert np.array_equal(np.asarray(new.U)[:, ~atlas.interior], np.asarray(U0)[:, ~atlas.interior])
    assert not np.array_equal(np.asarray(new.U)[:, atlas.interior], np.asarray(state.U)[:, atlas.interior])


def test_forces_and_surface_are_the_single_block_ones(case, atlas) -> None:
    """Pressure and friction, face by face: the same wall faces and cells."""
    from jaxaero.post.forces import integrate_forces, surface

    _, grid, mesh, windows = case
    atlas = atlas()
    flow = FLOWS["sa"]
    single = Single(grid, flow, NumericsConfig())
    U = perturbed(single)
    want = integrate_forces(single.canvas(U), single.atlas, flow)
    got = integrate_forces(atlas.pack(cut(U, windows)), atlas, flow)
    assert got.bodies == ()
    for name in ("cl", "cd", "cm", "cd_friction"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12, abs=1e-14), name
    a, b = surface(single.canvas(U), single.atlas, flow), surface(atlas.pack(cut(U, windows)), atlas, flow)
    for key in ("x", "cp", "cf"):
        assert np.allclose(np.sort(b[key]), np.sort(a[key]), rtol=1e-12, atol=1e-15), key
