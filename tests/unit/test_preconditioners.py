"""Unit tests for the implicit preconditioners.

A preconditioner cannot make the solver wrong -- it only changes how many
Krylov iterations a step costs -- so the useful question is not "is the
approximate Jacobian accurate" but "does applying it actually invert the matrix
it claims to invert". That is where a transpose or an index offset would hide,
and it is what these tests check: the blocks of ``line_j`` are assembled into a
dense per-line matrix and the preconditioned vector is substituted back.

Whether the approximation is *worth* anything is a different question, measured
on a real case in ``tests/verification/test_implicit.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.state import primitive_to_conservative
from jaxaero.numerics.residual import face_wave_speeds

from ..single import atlas_of, to_canvas
from jaxaero.solvers import preconditioners as pre

GAMMA = 1.4
NVAR_FLOW = 4     # 2D
CFL = 7.0
NAMES = sorted(pre.PRECONDITIONERS)


@pytest.fixture(scope="module")
def perturbed_state(coarse_grid):
    """Freestream with a smooth, O(10%) perturbation.

    Uniform flow would make every block identical and hide an index error; a
    varying state makes each ``j`` row of the line system distinct.
    """
    grid = coarse_grid
    wave = jnp.cos(3.0 * grid.xc) * jnp.sin(2.0 * grid.yc)
    rho = 1.0 + 0.1 * wave
    u = 0.5 + 0.1 * wave
    v = 0.05 * wave
    p = 1.0 / GAMMA * (1.0 + 0.1 * wave)
    return to_canvas(primitive_to_conservative(rho, jnp.stack([u, v]), p, GAMMA), atlas_of(grid))


@pytest.fixture(scope="module")
def speeds(perturbed_state, coarse_grid):
    return face_wave_speeds(perturbed_state, atlas_of(coarse_grid), GAMMA)


@pytest.fixture(scope="module")
def rhs(coarse_grid):
    rng = np.random.default_rng(21)
    return jnp.asarray(rng.normal(size=(NVAR_FLOW,) + atlas_of(coarse_grid).shape))


@pytest.mark.parametrize("name", NAMES)
def test_preconditioner_preserves_shape_and_finiteness(
    name, coarse_grid, perturbed_state, speeds, rhs
) -> None:
    apply = pre.get_preconditioner(name)(atlas_of(coarse_grid), GAMMA)(
        perturbed_state, speeds, jnp.asarray(CFL)
    )
    z = apply(rhs)
    assert z.shape == rhs.shape
    assert bool(jnp.all(jnp.isfinite(z)))


def test_identity_returns_its_input(coarse_grid, perturbed_state, speeds, rhs) -> None:
    apply = pre.get_preconditioner("none")(atlas_of(coarse_grid), GAMMA)(
        perturbed_state, speeds, jnp.asarray(CFL)
    )
    assert np.array_equal(np.asarray(apply(rhs)), np.asarray(rhs))


def chain_of(atlas, i: int) -> int:
    """The chain holding block cell ``(i, 0)``."""
    cells = np.asarray(atlas.lines.cells)
    return int(np.argwhere(cells == (atlas.g + i) * atlas.nj + atlas.g)[0, 0])


@pytest.mark.parametrize("station", [0, 8, 16, 31])
def test_line_preconditioner_inverts_its_own_blocks(
    station, coarse_grid, perturbed_state, speeds, rhs
) -> None:
    """``M z = r`` for the dense matrix the chain blocks describe, at the
    trailing edge, both surfaces and the leading edge."""
    atlas = atlas_of(coarse_grid)
    cfl = jnp.asarray(CFL)
    lower, diag, upper = (np.asarray(a) for a in pre.flow_chain_blocks(
        perturbed_state, speeds, cfl, atlas, GAMMA, pre.ChainLayout(atlas)))
    z = np.asarray(pre.get_preconditioner("line_j")(atlas, GAMMA)(perturbed_state, speeds, cfl)(rhs))
    n = chain_of(atlas, station)
    cells = [int(c) for c in np.asarray(atlas.lines.cells)[n] if c >= 0]
    assert len(cells) == coarse_grid.nj                    # an O-grid line: wall to far field
    m, L = NVAR_FLOW, len(cells)
    dense = np.zeros((L * m, L * m))
    for k in range(L):
        rows = slice(k * m, (k + 1) * m)
        dense[rows, rows] = diag[k, n]
        if k > 0:
            dense[rows, slice((k - 1) * m, k * m)] = lower[k, n]
        if k < L - 1:
            dense[rows, slice((k + 1) * m, (k + 2) * m)] = upper[k, n]
    z_line = z.reshape(m, -1)[:, cells].T.reshape(-1)
    r_line = np.asarray(rhs).reshape(m, -1)[:, cells].T.reshape(-1)
    assert np.allclose(dense @ z_line, r_line, rtol=1e-9, atol=1e-11)


def test_line_blocks_vanish_at_the_chain_ends(coarse_grid, perturbed_state, speeds) -> None:
    """No coupling below a chain's first cell or above its last: the solver
    does not inspect those blocks."""
    atlas = atlas_of(coarse_grid)
    lower, _, upper = pre.flow_chain_blocks(perturbed_state, speeds, jnp.asarray(CFL), atlas, GAMMA,
                                            pre.ChainLayout(atlas))
    length = (np.asarray(atlas.lines.cells) >= 0).sum(axis=1)
    assert not np.any(np.asarray(lower[0]))
    for n, L in enumerate(length):
        assert not np.any(np.asarray(upper[L - 1, n]))


def test_unknown_preconditioner_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown preconditioner"):
        pre.get_preconditioner("magic")
