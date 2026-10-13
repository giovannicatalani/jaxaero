"""Dense linear algebra on batches of small blocks.

Equation-agnostic array utilities. The one subject is the block-tridiagonal
operator, which is what a grid line becomes when a cell is coupled implicitly to
its two neighbours along that line.

Factorization is separate from the solve because a preconditioner is factored
once per Newton step and applied once per Krylov iteration. Folding them
together cost 9 ms per application against 0.5 ms on the 112x32 line system.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, lax


class BlockTridiagonalFactor(NamedTuple):
    """Block Thomas factor, a pytree: ``lower``, ``pivot_inverse`` and ``c_hat``, each ``(n, ..., m, m)``."""

    lower: Array
    pivot_inverse: Array
    c_hat: Array


def _matvec(block: Array, vector: Array) -> Array:
    return jnp.einsum("...ij,...j->...i", block, vector)


def invert_blocks(A: Array) -> Array:
    """Inverse of every block in a batch, ``(..., m, m) -> (..., m, m)``.

    Gauss-Jordan on ``[A | I]``, unrolled over the ``m`` rows at trace time and
    carried as a list of row arrays, so every operation is elementwise over the
    batch and XLA fuses the whole inverse.

    Preferred over ``jnp.linalg.inv`` because this is called inside a sequential
    scan on blocks of size four, where dispatching a batched vendor LU per step
    dominates: 11.2 ms against 2.5 ms, agreeing to 6e-17. Vendor LU remains the
    right choice for large blocks.

    Unpivoted; a zero pivot yields non-finite output rather than a plausible
    wrong answer, and callers are required to detect it.
    """
    m = A.shape[-1]
    eye = jnp.eye(m, dtype=A.dtype)
    rows = [
        jnp.concatenate(
            [A[..., k, :], jnp.broadcast_to(eye[k], A.shape[:-2] + (m,))], axis=-1
        )
        for k in range(m)
    ]
    for k in range(m):
        rows[k] = rows[k] / rows[k][..., k, None]
        for r in range(m):
            if r != k:
                rows[r] = rows[r] - rows[r][..., k, None] * rows[k]
    return jnp.stack([row[..., m:] for row in rows], axis=-2)


def factor_block_tridiagonal(
    lower: Array, diag: Array, upper: Array
) -> BlockTridiagonalFactor:
    """Forward sweep of the block Thomas algorithm for
    ``L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k``, blocks ``(n, ..., m, m)``
    (``lower[0]``, ``upper[n-1]`` zero): a ``lax.scan`` along axis 0, the batch
    axes parallel. Unpivoted (``invert_blocks``): sound with a pseudo-time
    diagonal; the implicit solver cuts the time step on a non-finite result."""
    if not (lower.shape == diag.shape == upper.shape):
        raise ValueError(
            f"block shapes must agree, got lower {lower.shape}, "
            f"diag {diag.shape}, upper {upper.shape}"
        )

    def eliminate(c_prev: Array, blocks: tuple[Array, Array, Array]):
        low, dia, upp = blocks
        pivot_inverse = invert_blocks(dia - low @ c_prev)
        c_hat = pivot_inverse @ upp
        return c_hat, (pivot_inverse, c_hat)

    _, (pivot_inverse, c_hat) = lax.scan(
        eliminate, jnp.zeros_like(diag[0]), (lower, diag, upper)
    )
    return BlockTridiagonalFactor(lower, pivot_inverse, c_hat)


def solve_block_tridiagonal(factor: BlockTridiagonalFactor, rhs: Array) -> Array:
    """Apply a factored inverse to ``rhs``, ``(n, ..., m) -> (n, ..., m)``.

    Forward then backward substitution, matrix-vector products only.

    Raises:
        ValueError: if ``rhs`` does not match the factored block shape.
    """
    if rhs.shape != factor.c_hat.shape[:-1]:
        raise ValueError(
            f"rhs shape {rhs.shape} does not match block shape {factor.c_hat.shape}"
        )

    def forward(d_prev: Array, blocks):
        low, pivot_inverse, b = blocks
        d = _matvec(pivot_inverse, b - _matvec(low, d_prev))
        return d, d

    def backward(x_next: Array, blocks):
        c, d = blocks
        x = d - _matvec(c, x_next)
        return x, x

    zero = jnp.zeros_like(rhs[0])
    _, d_hat = lax.scan(forward, zero, (factor.lower, factor.pivot_inverse, rhs))
    _, x = lax.scan(backward, zero, (factor.c_hat, d_hat), reverse=True)
    return x
