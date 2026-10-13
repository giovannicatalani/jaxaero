"""Wall distance: exact distance from cell centres to the wall polyline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxaero.mesh.mesh import wall_distance
from validation.grids.karman_trefftz import TMR, make_o_grid

from ..single import c_layout, read_grid, wall_slice, with_conditions

TMR_COARSE = Path(__file__).resolve().parents[2] / "data" / "grids" / "n0012_113-33.p2dfmt"


def brute_force(grid, samples_per_segment: int = 400) -> np.ndarray:
    """Distance to a densely sampled wall polyline: an upper bound that
    converges to the exact value as ``O(h_segment / samples)``."""
    wall = wall_slice(grid)
    x = np.asarray(grid.x_node[wall.start : wall.stop + 1, 0])
    y = np.asarray(grid.y_node[wall.start : wall.stop + 1, 0])
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    sx = (x[:-1, None] + t * np.diff(x)[:, None]).ravel()
    sy = (y[:-1, None] + t * np.diff(y)[:, None]).ravel()
    px, py = np.asarray(grid.xc).ravel(), np.asarray(grid.yc).ravel()
    d = np.sqrt((px[:, None] - sx) ** 2 + (py[:, None] - sy) ** 2).min(axis=1)
    return d.reshape(grid.ni, grid.nj)


def test_matches_a_densely_sampled_wall() -> None:
    grid = make_o_grid(32, 44, 4e-4, section=TMR)
    exact, dense = wall_distance(with_conditions(grid))[0], brute_force(grid)
    assert np.all(exact <= dense + 1e-15)          # the exact value is the minimum
    segment = 0.12                                  # longest wall segment at n_c = 32, roughly
    assert np.max(dense - exact) < segment / 400


def test_first_cells_sit_at_half_their_height() -> None:
    """Away from the leading and trailing edges the first centre is half its
    own cell's wall-normal width off the wall, and distance grows outward."""
    from jaxaero.mesh.block import cell_widths

    grid = make_o_grid(64, 88, 2e-4, section=TMR)
    d = wall_distance(with_conditions(grid))[0]
    _, width_j = (np.asarray(w) for w in cell_widths(grid.x_node, grid.y_node))
    mid = slice(10, 22)                             # lower surface, mid-chord
    assert np.allclose(d[mid, 0], 0.5 * width_j[mid, 0], rtol=0.02)
    assert np.all(np.diff(d[mid], axis=1) > 0.0)


def test_on_a_c_grid_the_wake_measures_to_the_trailing_edge() -> None:
    if not TMR_COARSE.is_file():
        pytest.skip("TMR grids not on disk")
    grid = read_grid(TMR_COARSE)
    d = wall_distance(with_conditions(grid))[0]
    topo = c_layout(grid)
    xc, yc = np.asarray(grid.xc), np.asarray(grid.yc)
    wake = slice(0, topo.n_wake)
    to_te = np.hypot(xc[wake, 0] - 1.0, yc[wake, 0])
    assert np.allclose(d[wake, 0], to_te, rtol=1e-12)
    assert d.min() > 0.0
