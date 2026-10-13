"""Unit tests for the block-tridiagonal factorization and solve.

Exercised as the preconditioner uses them -- factor once, then solve -- rather
than through a one-shot helper, so the test covers the path that runs.

Checked against a dense assembly of the same system: the only way to be sure the
forward and reverse sweeps, the batching and the augmented right-hand side are
wired as intended. Random non-symmetric blocks are used rather than a
hand-written example, because an index slip typically still solves a symmetric
or constant-block system correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxaero.core.linalg import (
    factor_block_tridiagonal,
    solve_block_tridiagonal,
)

M = 4


def dense_from_blocks(lower, diag, upper, batch) -> np.ndarray:
    """Assemble one batch entry's system as a dense ``(n*M, n*M)`` matrix."""
    n = diag.shape[0]
    dense = np.zeros((n * M, n * M))
    for k in range(n):
        rows = slice(k * M, (k + 1) * M)
        dense[rows, rows] = diag[k, batch]
        if k > 0:
            dense[rows, slice((k - 1) * M, k * M)] = lower[k, batch]
        if k < n - 1:
            dense[rows, slice((k + 1) * M, (k + 2) * M)] = upper[k, batch]
    return dense


@pytest.fixture
def system():
    """A diagonally dominant batch of block-tridiagonal systems."""
    rng = np.random.default_rng(7)
    n, batch = 9, 5
    shape = (n, batch, M, M)
    lower = rng.normal(size=shape)
    upper = rng.normal(size=shape)
    diag = rng.normal(size=shape) + 4.0 * M * np.eye(M)
    lower[0] = 0.0
    upper[-1] = 0.0
    rhs = rng.normal(size=(n, batch, M))
    return lower, diag, upper, rhs


def solve(lower, diag, upper, rhs) -> np.ndarray:
    """Factor and apply, the way the line preconditioner does."""
    factor = factor_block_tridiagonal(
        np.asarray(lower), np.asarray(diag), np.asarray(upper)
    )
    return np.asarray(solve_block_tridiagonal(factor, np.asarray(rhs)))


def test_matches_a_dense_solve(system) -> None:
    lower, diag, upper, rhs = system
    x = solve(*system)

    for b in range(rhs.shape[1]):
        dense = dense_from_blocks(lower, diag, upper, b)
        expected = np.linalg.solve(dense, rhs[:, b].reshape(-1)).reshape(-1, M)
        assert np.allclose(x[:, b], expected, rtol=1e-10, atol=1e-12)


def test_residual_of_the_solution_vanishes(system) -> None:
    """``L x_{k-1} + D x_k + U x_{k+1} = b`` to round-off, stated directly."""
    lower, diag, upper, rhs = system
    x = solve(*system)
    n = diag.shape[0]

    shifted_down = np.concatenate([np.zeros_like(x[:1]), x[:-1]], axis=0)
    shifted_up = np.concatenate([x[1:], np.zeros_like(x[:1])], axis=0)
    applied = (
        np.einsum("nbij,nbj->nbi", lower, shifted_down)
        + np.einsum("nbij,nbj->nbi", diag, x)
        + np.einsum("nbij,nbj->nbi", upper, shifted_up)
    )
    assert np.allclose(applied, rhs, rtol=1e-10, atol=1e-12)
    assert n == x.shape[0]


def test_diagonal_system_is_a_plain_divide() -> None:
    """With no off-diagonal blocks the solve reduces to the point-Jacobi case."""
    rng = np.random.default_rng(8)
    n, batch = 6, 3
    diag = np.einsum("nb,ij->nbij", rng.uniform(1.0, 3.0, (n, batch)), np.eye(M))
    zero = np.zeros_like(diag)
    rhs = rng.normal(size=(n, batch, M))

    x = solve(zero, diag, zero, rhs)
    assert np.allclose(x, rhs / np.einsum("nbii->nbi", diag), rtol=1e-12, atol=0.0)


def test_shape_disagreement_is_rejected(system) -> None:
    """Each function checks what it is given, rather than trusting the caller."""
    lower, diag, upper, rhs = system
    with pytest.raises(ValueError, match="block shapes"):
        factor_block_tridiagonal(lower[:-1], diag, upper)
    with pytest.raises(ValueError, match="rhs shape"):
        solve_block_tridiagonal(
            factor_block_tridiagonal(lower, diag, upper), rhs[..., :-1]
        )
