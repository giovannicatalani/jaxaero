"""Slope limiters for MUSCL reconstruction.

Every limiter here takes the two neighbouring *differences* ``a`` (backward) and
``b`` (forward) and returns a limited slope, rather than taking a ratio
``r = a/b``. That avoids a 0/0 wherever the solution is locally flat -- which on
a converged steady field is most of the domain -- and removes the need for an
arbitrary guard epsilon in the denominator.

All of them return 0 when ``a*b <= 0`` (a local extremum), which is what enforces
monotonicity, and **must** return exactly ``a`` when ``a == b``: that is the
condition for second-order accuracy in smooth flow, and a limiter that returns
some fraction of the central slope instead degrades the scheme silently while
still looking plausible. It is unit-tested for every registered limiter.

``minmod``: the most dissipative second-order TVD limiter, for shocks.
``van_albada``: smooth where active, for a shock meeting a boundary layer
(what a Newton method wants). ``none``: smooth flow and order measurements.
"""

from __future__ import annotations

from typing import Callable, Protocol

import jax.numpy as jnp
from jax import Array


class Limiter(Protocol):
    """Maps a pair of neighbouring differences to a limited slope."""

    def __call__(self, a: Array, b: Array) -> Array:
        ...


LIMITERS: dict[str, Limiter] = {}


def register_limiter(name: str) -> Callable[[Limiter], Limiter]:
    def decorate(fn: Limiter) -> Limiter:
        if name in LIMITERS:
            raise KeyError(f"limiter '{name}' already registered")
        LIMITERS[name] = fn
        return fn

    return decorate


def get_limiter(name: str) -> Limiter:
    try:
        return LIMITERS[name]
    except KeyError:
        raise KeyError(
            f"unknown limiter '{name}'. Available: {sorted(LIMITERS)}"
        ) from None


@register_limiter("none")
def unlimited(a: Array, b: Array) -> Array:
    """Unlimited central slope, ``(a + b)/2``.

    Second-order accurate and non-monotone. This is the setting for
    order-of-accuracy verification: limiters clip extrema by design and would
    mask the formal order of the scheme.
    """
    return 0.5 * (a + b)


@register_limiter("minmod")
def minmod(a: Array, b: Array) -> Array:
    """The most diffusive second-order TVD limiter.

    ``sign(a) min(|a|, |b|)`` when ``a*b > 0``, else 0. Its kink at ``a == b``
    is why the implicit path uses a first-order Jacobian (defect correction).
    """
    return jnp.where(a * b > 0.0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


@register_limiter("van_albada")
def van_albada(a: Array, b: Array) -> Array:
    """Van Albada's limiter, ``ab(a+b)/(a^2+b^2)`` when ``a*b > 0``, else 0.

    Returns the central slope ``a`` at ``a == b`` and departs from it only
    quadratically in ``(a - b)``, so smooth regions -- the boundary layer --
    keep nearly the unlimited scheme, while at a shock (``a`` and ``b`` of
    very different size) it tends to the smaller difference, as minmod does.
    Differentiable wherever ``a*b > 0``.

    Reference: van Albada, van Leer and Roberts, "A comparative study of
    computational methods in cosmic gas dynamics", Astron. Astrophys. 108 (1982).
    """
    same_sign = a * b > 0.0
    denom = jnp.where(same_sign, a * a + b * b, 1.0)
    return jnp.where(same_sign, a * b * (a + b) / denom, 0.0)
