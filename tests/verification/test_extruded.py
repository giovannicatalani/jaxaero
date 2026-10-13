"""The 3D operator contains the 2D one.

A 2D grid extruded ``nk`` cells in ``z`` with symmetry planes at both ends,
and a 2D state copied into every plane with ``w = 0``, is a 3D state whose
residual must be the 2D residual times the cell depth ``dz`` in every plane,
with no ``z`` momentum: the 3D faces are the 2D ones times ``dz``, the end
planes mirror ``w = 0`` onto itself, and every ``z`` difference vanishes. Every
2D topology (O seam, C wake cut, the plate's symmetry, inflow and outflow
patches), Euler, laminar and SA, first and second order: the 2D validations
then check the 3D code too.
"""

from __future__ import annotations

import glob
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from jaxaero.mesh.atlas import build_atlas
from jaxaero.mesh.mesh import assign_conditions
from jaxaero.numerics.residual import make_residual
from validation.grids.karman_trefftz import TMR, make_o_grid

from ..single import PLATE_RULES, conditions, extrude, read_grid

GRIDS = Path(__file__).resolve().parents[2] / "data" / "grids"
NK, DZ = 3, 0.7


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
PHYSICS = {
    "euler": FlowConditions(mach=0.6, alpha_deg=2.5),
    "laminar": FlowConditions(mach=0.5, alpha_deg=1.0, reynolds=5000.0),
    "sa": FlowConditions(mach=0.15, alpha_deg=10.0, reynolds=6e6, sutherland_ratio=198.6 / 540.0,
                         turbulence="sa_noft2"),
}


def to_3d(U2: np.ndarray, nk: int) -> np.ndarray:
    """``(rho, rho u, rho v, E[, s])`` -> ``(rho, rho u, rho v, 0, E[, s])`` in every plane."""
    U3 = np.concatenate([U2[:3], np.zeros_like(U2[:1]), U2[3:]])
    return np.repeat(U3[..., None], nk, axis=-1)


@pytest.mark.parametrize("first_order", [True, False])
@pytest.mark.parametrize("physics", sorted(PHYSICS))
@pytest.mark.parametrize("topology", [pytest.param(t, marks=() if t == "o_grid" else pytest.mark.slow)
                                      for t in sorted(TOPOLOGIES)])
def test_extruded_residual_is_the_2d_residual_in_every_plane(topology, physics, first_order) -> None:
    grid2, flow = TOPOLOGIES[topology](), PHYSICS[physics]
    numerics = NumericsConfig(limiter="van_albada")
    kinds = conditions(grid2, no_slip=not flow.inviscid)
    atlas2 = build_atlas(assign_conditions(grid2, kinds), numerics.n_ghost)
    op2 = make_residual(atlas2, flow, numerics, first_order)
    atlas3 = build_atlas(assign_conditions(extrude(grid2, NK, DZ), {**kinds, "symmetry": "symmetry"}),
                         numerics.n_ghost)
    rho, u, v, p = np.asarray(op2.q_inf[:4])
    op3 = make_residual(atlas3, flow, numerics, first_order, freestream=jnp.asarray([rho, u, v, 0.0, p]))

    # A seeded perturbation of the freestream on the block cells.
    g = numerics.n_ghost
    U0_2 = np.asarray(op2.initial_state())
    rng = np.random.default_rng(42)
    U2 = U0_2 * (1.0 + 0.03 * rng.standard_normal(U0_2.shape))
    U2 = np.where(atlas2.interior, U2, U0_2)
    block2 = (slice(None), slice(g, -g), slice(g, -g))
    U3 = np.asarray(op3.initial_state()).copy()
    U3[(slice(None),) + (slice(g, -g),) * 3] = to_3d(U2[block2], NK)

    R2 = np.asarray(op2.residual(jnp.asarray(U2)))[block2]
    R3 = np.asarray(op3.residual(jnp.asarray(U3)))[(slice(None),) + (slice(g, -g),) * 3]
    scale = np.abs(R2).max(axis=(1, 2))[:, None, None]
    rows = [0, 1, 2, 4] + ([5] if R3.shape[0] == 6 else [])
    for k in range(NK):
        worst = float(np.max(np.abs(R3[rows, :, :, k] / DZ - R2) / scale))
        assert worst < 1e-11, f"plane {k}: relative difference {worst:.2e}"
    assert np.abs(R3[3]).max() < 1e-11 * scale.max() * DZ, "z momentum appeared"
