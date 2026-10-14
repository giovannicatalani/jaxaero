"""Grid sequencing: a coarsened mesh keeps the fine mesh's topology, and the
coarse solution is carried back exactly.

The coarse mesh must be a valid grid in its own right (interfaces still
point-matched, the pole still a pole: freestream preserved on it), with the
same patches, and every other node of the fine one.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero import load_case, run
from jaxaero.mesh.block import Grid
from jaxaero.mesh.block3d import box_sides
from jaxaero.mesh.coarsen import coarsen, prolong
from jaxaero.mesh.mesh import interface_node_mismatch, patches

from ..single import pole_sides
from ..verification.test_freestream3d import FLOWS, assert_preserved
from .test_grid3d import warped_box

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_a_c_grid_coarsens_with_its_wake_cut_and_patches() -> None:
    case = load_case(EXAMPLES / "naca0012_laminar")
    fine, coarse = case.grid(), case.grid(coarsen=1)
    assert coarse.cells * 4 == fine.cells
    assert np.array_equal(np.asarray(coarse.mesh[0].nodes), np.asarray(fine.mesh[0].nodes)[:, ::2, ::2])
    assert patches(coarse.mesh) == patches(fine.mesh)
    assert interface_node_mismatch(coarse.mesh) == 0.0
    assert len(coarse.walls.face) * 2 == len(fine.walls.face)


def test_freestream_is_preserved_on_the_coarsened_m6_grid() -> None:
    """The pole (an interface flipped and periodic round the O seam) and the
    symmetry plane survive coarsening."""
    from jaxaero.mesh.plot3d import read_plot3d_3d

    (nodes,) = read_plot3d_3d(EXAMPLES / "onera_m6_sa/om6_strct_16x44x24.ufmt")
    grid = Grid.from_node_array(nodes, pole_sides(tuple(n - 1 for n in nodes.shape[1:])))
    (coarse,) = coarsen(grid)
    assert coarse.shape == tuple(n // 2 for n in grid.shape)
    assert_preserved(coarse, FLOWS["sa"], {"wall": "far_field", "farfield": "far_field", "symmetry": "symmetry"})


def test_an_odd_cell_count_is_refused() -> None:
    shape = (6, 5, 4)
    grid = Grid.from_node_array(warped_box(shape), box_sides(shape, **dict.fromkeys(
        ("j_min", "j_max", "i_min", "i_max", "k_min", "k_max"), "farfield")))
    with pytest.raises(ValueError, match="odd"):
        coarsen(grid)


def test_prolongation_fills_each_child_from_its_parent() -> None:
    case = load_case(EXAMPLES / "naca0012_laminar")
    fine, coarse = case.grid(), case.grid(coarsen=1)
    U = jnp.asarray(np.random.default_rng(1).normal(size=(2,) + coarse.shape))
    start = jnp.full((2,) + fine.shape, 7.0)
    P = prolong(U, coarse, fine, start)
    block, parent = np.asarray(fine.unpack(P)[0]), np.asarray(coarse.unpack(U)[0])
    for di in (0, 1):
        for dj in (0, 1):
            assert np.array_equal(block[:, di::2, dj::2], parent)
    outside = np.ones(fine.shape, bool)
    outside[tuple(fine.windows[0])] = False
    assert np.all(np.asarray(P)[:, outside] == 7.0)          # canvas padding keeps the start


def test_a_sequenced_run_solves_the_coarse_grid_first() -> None:
    case = load_case(EXAMPLES / "naca0012_euler", (
        "case.domain.cells_airfoil=32", "case.domain.cells_normal=16",
        "numerical.convergence.max_iters=400", "numerical.convergence.grid_sequencing=1"))
    result = run(case)
    (level,) = result.levels
    assert level["cells"] * 4 == result.cells and level["iterations"] > 0
    assert np.all(np.isfinite(np.asarray(result.U)))
