"""O-grid topology: the periodic seam and the symmetry of the operator.

The O-mesh wraps periodically in ``i`` and has one slip wall along ``j = 0``,
so what has to be established is that the seam is an *exchange* -- the two ends
are neighbours, nothing is imposed -- and that a symmetric grid gives a
symmetric discrete operator. Free-stream preservation, which also covers the
seam, is in ``test_freestream.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from jaxaero.mesh.block import o_grid_sides
from jaxaero.core.state import freestream_primitive
from jaxaero.numerics import reconstruction
from jaxaero.numerics.limiters import get_limiter

from ..single import Single, fill_ghosts, forces, wall_slice, wall_surface
from ..conftest import freestream_state, uniform_state

N_C = 32
"""The ``coarse_grid`` fixture's size."""


# --- The topology object ----------------------------------------------------


def test_airfoil_slice_covers_every_cell(coarse_grid) -> None:
    """On an O-grid the body *is* the whole ``j = 0`` line."""
    assert wall_slice(coarse_grid) == slice(0, N_C)
    assert wall_surface(uniform_state(coarse_grid, FlowConditions(0.5)), coarse_grid, FlowConditions(0.5))['x'].shape == (N_C,)


def test_odd_station_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        o_grid_sides(31, 32)
    with pytest.raises(ValueError, match="nj >= 1"):
        o_grid_sides(32, 0)


# --- The periodic seam ------------------------------------------------------


def test_i_ghosts_are_an_exact_periodic_copy(coarse_grid) -> None:
    """Ghost ``i = -1-l`` is cell ``ni-1-l``, and ``i = ni+l`` is cell ``l``.

    Random data, deliberately: a smooth field would pass even with the two
    blocks reversed, which is the bug being hunted.
    """
    g = 2
    rng = np.random.default_rng(0)
    interior = jnp.asarray(rng.standard_normal((4, N_C, N_C)))
    q_inf = jnp.asarray(freestream_primitive(FlowConditions(mach=0.5)))
    padded = fill_ghosts(interior, coarse_grid, g, q_inf, 1.4)

    body = padded[:, :, g:-g]
    for layer in range(g):
        # No reversal: both ends are traversed in the same direction.
        assert np.array_equal(body[:, g - 1 - layer, :], interior[:, N_C - 1 - layer, :])
        assert np.array_equal(body[:, g + N_C + layer, :], interior[:, layer, :])


def test_the_seam_is_one_face_reconstructed_once(coarse_grid) -> None:
    """The i-faces at ``i = 0`` and ``i = ni`` are the same face, bitwise.

    The sharpest statement of what periodicity has to mean: both approaches to the
    seam must reconstruct identical left and right states, so the two fluxes the
    assembly computes for one physical face are the same number and cancel
    exactly in the global sum. Equality has to be *exact* -- a face counted with
    two slightly different fluxes is a conservation error that no residual norm
    would reveal.

    Random data again: a smooth field would pass even with the ghost blocks
    reversed.
    """
    g = 2
    rng = np.random.default_rng(1)
    flow = FlowConditions(mach=0.5, alpha_deg=1.25)
    interior = jnp.asarray(
        np.stack(
            [
                rng.uniform(0.5, 1.5, (N_C, N_C)),
                rng.standard_normal((N_C, N_C)),
                rng.standard_normal((N_C, N_C)),
                rng.uniform(0.5, 1.5, (N_C, N_C)),
            ]
        )
    )
    q_inf = jnp.asarray(freestream_primitive(flow))
    padded = fill_ghosts(interior, coarse_grid, g, q_inf, flow.gamma)

    left, right = reconstruction.face_states(
        padded, g, (N_C, N_C), get_limiter("minmod"), 0
    )
    assert np.array_equal(left[:, 0, :], left[:, N_C, :])
    assert np.array_equal(right[:, 0, :], right[:, N_C, :])
    # And the face geometry agrees, so the two fluxes are identical outright.
    assert np.array_equal(np.asarray(coarse_grid.sx_i)[0], np.asarray(coarse_grid.sx_i)[N_C])
    assert np.array_equal(np.asarray(coarse_grid.sy_i)[0], np.asarray(coarse_grid.sy_i)[N_C])


# --- Symmetry ---------------------------------------------------------------


def test_zero_incidence_freestream_gives_exactly_zero_lift(coarse_grid) -> None:
    """The grid is symmetric bitwise, so a symmetric state integrates to CL = 0.

    Not merely small: the mirror pairing of ``o_mesh_nodes`` makes the face area
    vectors exact negatives in ``y``, so the pressure integral cancels term by
    term. This is the baseline the lifting cases are measured against -- if it
    were only approximate, a 1e-4 drag figure would not be meaningful.
    """
    flow = FlowConditions(mach=0.5, alpha_deg=0.0)
    U = uniform_state(coarse_grid, flow)
    result = forces(U, coarse_grid, flow)
    assert result.cl == 0.0
    assert result.fy == 0.0


def test_a_symmetric_residual_stays_symmetric(coarse_grid) -> None:
    """Mirroring the state mirrors the residual, to round-off.

    The discrete operator inherits the grid's symmetry: mass, x-momentum and
    energy residuals mirror, the y-momentum residual anti-mirrors. Without this
    a zero-incidence run could develop lift out of nothing but round-off bias,
    and the ``CL = 0`` reference value of ladder case 1 would be untestable.

    Note the cell mirror is ``ni - 1 - i``, not the node mirror ``ni - i``: cell
    ``i`` spans nodes ``i`` and ``i+1``, so its reflection spans nodes
    ``ni-i-1`` and ``ni-i``.
    """
    flow = FlowConditions(mach=0.5, alpha_deg=0.0)
    operator = Single(coarse_grid, flow, NumericsConfig(limiter="minmod"))

    mirror = N_C - 1 - np.arange(N_C)
    rng = np.random.default_rng(2)
    base = np.asarray(freestream_state(flow))[:, None, None]
    noise = 0.02 * rng.standard_normal((4, N_C, N_C))
    # Make the perturbation itself symmetric: even in rho, rho u and E, odd in
    # rho v, which is what a symmetric flow field looks like.
    symmetric = 0.5 * (noise + noise[:, mirror, :])
    antisymmetric = 0.5 * (noise - noise[:, mirror, :])
    U = base * (1.0 + symmetric)
    U[2] = base[2] * antisymmetric[2] + np.abs(base[0]) * antisymmetric[2]

    residual = np.asarray(operator.residual(jnp.asarray(U)))
    scale = np.abs(residual).max()
    for var in (0, 1, 3):
        assert np.abs(residual[var] - residual[var][mirror, :]).max() < 1e-11 * scale
    assert np.abs(residual[2] + residual[2][mirror, :]).max() < 1e-11 * scale


def test_surface_output_covers_the_whole_contour(coarse_grid) -> None:
    """``airfoil_slice`` is all of ``i``, so Cp covers the whole contour."""
    flow = FlowConditions(mach=0.5, alpha_deg=0.0)
    U = uniform_state(coarse_grid, flow)
    cp = np.asarray(wall_surface(U, coarse_grid, flow)['cp'])
    x = np.asarray(wall_surface(uniform_state(coarse_grid, flow), coarse_grid, flow)['x'])
    assert cp.shape == (N_C,)
    assert np.allclose(cp, 0.0, atol=1e-15)       # uniform state: p = p_inf
    assert x.min() >= -1e-12
    assert x.max() <= 1.009
