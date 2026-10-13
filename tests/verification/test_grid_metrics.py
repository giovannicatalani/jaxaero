"""Metric identities of the grid container.

The real geometric checks -- wall on the analytic section, positive areas,
enclosed area against its closed form, symmetry, seam closure -- live with the
generator in ``tests/unit/test_karman_trefftz.py``. What is here is independent
of how the nodes were made:

* **Outward normals** -- real. An inward wall normal flips every force.
* **Normal closure, and area-sum vs perimeter** -- *identities*. Both are the
  same telescoping sum of node cross-products and hold bitwise by construction;
  perturbing a node leaves their residual at exactly zero. They catch indexing
  mistakes, not wrong formulas, and are labelled as such rather than presented as
  independent confirmation.

Tolerances are round-off multiples in float64 and are not to be relaxed: every
identity here is exact in real arithmetic, so a failure is a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxaero.mesh.block import Grid, normal_closure_residual, perimeter_area, validate_metrics
from jaxaero.mesh.mesh import interface_node_mismatch
from ..single import unit_j_normals

ROUNDOFF = 1e-12

# --- Real checks ------------------------------------------------------------



def constant_flux_residual(grid, flux_x: float = 1.0, flux_y: float = 0.0):
    """Divergence of a spatially constant flux over every cell: zero on any
    valid grid however distorted, since a closed cell's outward area vectors
    sum to zero. An indexing check (faces and cells line up), not an
    independent one: the identity holds by construction."""
    flux_i = flux_x * grid.sx_i + flux_y * grid.sy_i
    flux_j = flux_x * grid.sx_j + flux_y * grid.sy_j
    return (flux_i[1:, :] - flux_i[:-1, :]) + (flux_j[:, 1:] - flux_j[:, :-1])

def test_j_face_normals_point_away_from_the_wall(coarse_grid) -> None:
    """An inward +j normal would reverse the sign of every wall pressure force."""
    nx, ny = (np.asarray(a) for a in unit_j_normals(coarse_grid, 0))
    xw = np.asarray(coarse_grid.x_node)[:-1, 0]
    yw = np.asarray(coarse_grid.y_node)[:-1, 0]
    # The component toward mid-chord on the chord line must be negative.
    assert np.all(nx * (0.5 - xw) + ny * (-yw) < 0.0)
    assert np.allclose(np.hypot(nx, ny), 1.0, rtol=1e-14)


def test_the_gate_passes(coarse_grid, medium_grid) -> None:
    """``validate_metrics`` is what the validation script reports per level."""
    for grid in (coarse_grid, medium_grid):
        report = validate_metrics(grid)
        assert report.ok, report.summary()
        assert interface_node_mismatch(grid) == 0.0


# --- Identities (indexing checks) -------------------------------------------


def test_face_normals_close_each_cell(coarse_grid) -> None:
    """Outward area vectors sum to zero per cell.

    An identity, since the four edge vectors of a closed quadrilateral sum to zero
    and the rotation building normals is linear. It verifies that face arrays and
    cells are indexed consistently.
    """
    closure = np.abs(np.asarray(normal_closure_residual(coarse_grid)))
    assert closure.max() < ROUNDOFF


def test_area_sum_equals_the_perimeter_integral(coarse_grid) -> None:
    """Also an identity, and it holds bitwise -- see the module docstring.

    Its value is in the *sign*: walked counter-clockwise in index space, a
    right-handed grid gives a positive result, so this flags a reversed contour.
    """
    total = float(np.sum(np.asarray(coarse_grid.area)))
    assert total > 0.0
    assert total == pytest.approx(float(perimeter_area(coarse_grid)), rel=ROUNDOFF)


@pytest.mark.parametrize("flux", [(1.0, 0.0), (0.0, 1.0), (0.3, -0.7)])
def test_constant_flux_gives_zero_residual(coarse_grid, flux) -> None:
    """Geometric free-stream preservation: an indexing check, not an independent one."""
    residual = np.abs(np.asarray(constant_flux_residual(coarse_grid, *flux)))
    scale = max(abs(flux[0]), abs(flux[1])) * float(
        np.max(np.hypot(coarse_grid.sx_j, coarse_grid.sy_j))
    )
    assert residual.max() < 1e-11 * max(scale, 1.0)


def test_reversed_contour_produces_negative_areas(coarse_grid) -> None:
    """Documents the orientation requirement by violating it."""
    reversed_grid = Grid.from_nodes(
        np.asarray(coarse_grid.x_node)[::-1],
        np.asarray(coarse_grid.y_node)[::-1],
        coarse_grid.sides,
    )
    assert float(np.asarray(reversed_grid.area).max()) < 0.0
