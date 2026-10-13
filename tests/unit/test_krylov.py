"""Right-preconditioned GMRES: exact on small systems, and never worse than
the zero vector -- the property left preconditioning lacks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.solvers.krylov import gmres_right


def system(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.eye(n) * 4.0 + rng.standard_normal((n, n))
    return A, rng.standard_normal(n)


def test_full_subspace_solves_exactly() -> None:
    """With ``restart = n`` the Krylov space is everything: the exact solution."""
    A, b = system()
    x = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: r,
                    restart=24, maxiter=1, tol=1e-14)
    assert np.allclose(A @ np.asarray(x), b, atol=1e-10)


def test_an_exact_preconditioner_converges_in_one_step() -> None:
    A, b = system()
    A_inv = jnp.asarray(np.linalg.inv(A))
    x = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: A_inv @ r,
                    restart=1, maxiter=1, tol=1e-14)
    assert np.allclose(A @ np.asarray(x), b, atol=1e-10)


@pytest.mark.parametrize("restart", [1, 3, 6])
def test_the_true_residual_never_grows(restart: int) -> None:
    """Even with a bad preconditioner: GMRES minimizes |b - A x| over a space
    containing zero."""
    A, b = system(seed=3)
    bad = jnp.asarray(np.random.default_rng(9).standard_normal((24, 24)))
    x = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: bad @ r,
                    restart=restart, maxiter=2, tol=1e-14)
    assert np.linalg.norm(b - A @ np.asarray(x)) <= np.linalg.norm(b) * (1 + 1e-12)


def test_array_shaped_unknowns_and_zero_rhs() -> None:
    """Fields are ``(4, ni, nj)``, not vectors; a zero right-hand side gives zero."""
    rng = np.random.default_rng(1)
    d = jnp.asarray(1.0 + rng.random((4, 5, 3)))
    b = jnp.asarray(rng.standard_normal((4, 5, 3)))
    x = gmres_right(lambda v: d * v, b, lambda r: r, restart=8, maxiter=10, tol=1e-12)
    assert np.allclose(np.asarray(d * x), np.asarray(b), atol=1e-10)
    zero = gmres_right(lambda v: d * v, jnp.zeros_like(b), lambda r: r, restart=4, maxiter=2, tol=1e-12)
    assert not np.any(np.asarray(zero))
