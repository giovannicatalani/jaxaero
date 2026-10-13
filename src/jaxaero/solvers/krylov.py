"""Right-preconditioned restarted GMRES.

Solves ``A x = b`` as ``(A M^-1) y = b``, ``x = M^-1 y``. GMRES minimizes the
residual of the system it is given, so with the preconditioner on the right the
minimized quantity is the *true* residual ``|b - A x|``: an iteration can never
leave it larger than the zero vector does. Left preconditioning
(``jax.scipy.sparse.linalg.gmres``) minimizes ``|M^-1 (b - A x)|`` instead, and
the true residual is then unconstrained -- and the CFL ramp in ``implicit.py``
keys off the true reduction (docs/dev/notes.md).

Orthogonalization is classical Gram-Schmidt applied twice (CGS2): as stable as
modified Gram-Schmidt in practice and two matrix products per step instead of a
sequential loop (Giraud, Langou and Rozloznik, *Comput. Math. Appl.* 50 (2005)
1069). The Arnoldi basis is a fixed-size array, so each restart runs the full
``restart`` iterations; convergence is tested between restarts.

Reference: Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
(2003), Algorithms 6.9 and 9.5.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array, lax


def _dot(V: Array, w: Array) -> Array:
    """``V_k . w`` for every basis vector, ``(k,)``."""
    return jnp.tensordot(V, w, axes=w.ndim)


def gmres_right(
    matvec: Callable[[Array], Array],
    b: Array,
    precondition: Callable[[Array], Array],
    restart: int,
    maxiter: int,
    tol: float,
) -> Array:
    """Restarted GMRES(``restart``), right-preconditioned (``precondition: r -> M^-1 r``),
    for ``matvec`` on any array shape; stops at ``|b - A x| <= tol |b|`` between
    cycles, at most ``maxiter`` cycles. Zero for ``b = 0``."""
    b_norm = jnp.linalg.norm(b)
    tiny = jnp.finfo(b.dtype).tiny

    def cycle(x: Array) -> Array:
        r = b - matvec(x)
        beta = jnp.linalg.norm(r)
        V = jnp.zeros((restart + 1,) + b.shape, b.dtype).at[0].set(r / jnp.maximum(beta, tiny))
        H = jnp.zeros((restart + 1, restart), b.dtype)

        def arnoldi(j, carry):
            V, H = carry
            w = matvec(precondition(V[j]))
            active = (jnp.arange(restart + 1) <= j).astype(b.dtype)
            h = _dot(V, w) * active
            w = w - jnp.tensordot(h, V, axes=1)
            h2 = _dot(V, w) * active                  # the second pass of CGS2
            w = w - jnp.tensordot(h2, V, axes=1)
            h = h + h2
            norm = jnp.linalg.norm(w)
            H = H.at[:, j].set(h.at[j + 1].set(norm))
            V = V.at[j + 1].set(w / jnp.maximum(norm, tiny))
            return V, H

        V, H = lax.fori_loop(0, restart, arnoldi, (V, H))
        e1 = jnp.zeros(restart + 1, b.dtype).at[0].set(beta)
        y = jnp.linalg.lstsq(H, e1)[0]
        return x + precondition(jnp.tensordot(y, V[:restart], axes=1))

    def unconverged(carry):
        k, x, residual = carry
        return (k < maxiter) & (residual > tol * b_norm)

    def body(carry):
        k, x, _ = carry
        x = cycle(x)
        return k + 1, x, jnp.linalg.norm(b - matvec(x))

    x0 = jnp.zeros_like(b)
    _, x, _ = lax.while_loop(unconverged, body, (0, x0, b_norm))
    return x
