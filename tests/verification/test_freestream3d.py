"""Freestream preservation in 3D.

A uniform flow has zero residual on any closed-cell grid when every boundary
reproduces it: the face vectors of each cell sum to zero (mesh/block.py), so
the fluxes telescope. Checked on randomly warped cells (non-planar faces) and
on the ONERA M6 sample grid, pole included, with its wall set to far field.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from jaxaero.mesh.atlas import build_atlas
from jaxaero.mesh.block import Grid
from jaxaero.mesh.block3d import box_sides
from jaxaero.mesh.mesh import assign_conditions
from jaxaero.numerics.residual import make_residual

from ..single import pole_sides
from ..unit.test_grid3d import M6_SAMPLE, warped_box

FLOWS = {
    "euler": FlowConditions(mach=0.84, alpha_deg=3.06),
    "laminar": FlowConditions(mach=0.5, alpha_deg=2.0, reynolds=1000.0),
    "sa": FlowConditions(mach=0.84, alpha_deg=3.06, reynolds=14.6e6, sutherland_ratio=198.6 / 540.0,
                         turbulence="sa_noft2"),
}


def assert_preserved(grid: Grid, flow: FlowConditions, kinds: dict[str, str]) -> None:
    for first_order in (True, False):
        op = make_residual(build_atlas(assign_conditions(grid, kinds), 2), flow,
                           NumericsConfig(limiter="van_albada"), first_order)
        R = np.asarray(op.residual(op.initial_state()))[op.atlas.interior[None].repeat(op.q_inf.shape[0], 0)]
        # Flux scale: |F| ~ (rho V^2 + p) |S|, the largest face.
        scale = max(float(jnp.abs(f).max()) for f in grid.faces) * (flow.mach**2 + 1.0)
        assert np.abs(R).max() < 1e-12 * scale, f"first_order={first_order}: {np.abs(R).max():.2e}"


@pytest.mark.parametrize("physics", sorted(FLOWS))
def test_freestream_is_preserved_on_warped_cells(physics) -> None:
    shape = (6, 5, 7)
    grid = Grid.from_node_array(warped_box(shape, amplitude=0.3, seed=5),
                                box_sides(shape, **dict.fromkeys(("j_min", "j_max", "i_min", "i_max", "k_min",
                                                                  "k_max"), "farfield")))
    assert_preserved(grid, FLOWS[physics], {"farfield": "far_field"})


@pytest.mark.skipif(not M6_SAMPLE.exists(), reason="ONERA M6 sample grid not generated")
@pytest.mark.parametrize("physics", ["euler", "sa"])
def test_freestream_is_preserved_on_the_onera_m6_grid(physics) -> None:
    from jaxaero.mesh.plot3d import read_plot3d_3d

    (nodes,) = read_plot3d_3d(M6_SAMPLE)
    shape = tuple(n - 1 for n in nodes.shape[1:])
    grid = Grid.from_node_array(nodes, pole_sides(shape))
    assert_preserved(grid, FLOWS[physics], {"wall": "far_field", "farfield": "far_field", "symmetry": "symmetry"})
