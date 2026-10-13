"""Unit tests for slope limiters.

The second-order consistency test here caught a real bug: a limiter returning
0.75 of the central slope in smooth flow, which would have silently cost the
scheme its formal order and invalidated any order-of-accuracy study.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxaero.numerics import limiters

ALL = sorted(limiters.LIMITERS)
MONOTONE = [n for n in ALL if n != "none"]

finite = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64
)


@pytest.mark.parametrize("name", ALL)
def test_equal_differences_give_the_central_slope(name: str) -> None:
    """psi(a, a) == a, the condition for second-order accuracy.

    In smooth flow the two neighbouring differences agree, and the limited slope
    must then be the full central slope. A limiter returning any fraction of it
    degrades the scheme to below second order while still producing a plausible
    looking solution -- so this is asserted exactly, not approximately.
    """
    limiter = limiters.get_limiter(name)
    for value in (1e-8, 0.3, 1.0, 17.0, -0.3, -5.0):
        got = float(limiter(jnp.array(value), jnp.array(value)))
        assert got == pytest.approx(value, rel=1e-14), f"{name}: psi({value},{value})={got}"


@pytest.mark.parametrize("name", MONOTONE)
def test_opposite_signs_give_zero(name: str) -> None:
    """At a local extremum every *monotone* limiter must return zero slope.

    ``none`` is excluded deliberately: it is the unlimited central slope and is
    non-monotone by construction, which is the whole point of having it for
    order-of-accuracy studies.
    """
    limiter = limiters.get_limiter(name)
    for a, b in ((1.0, -1.0), (-2.0, 3.0), (0.5, -1e-6)):
        assert float(limiter(jnp.array(a), jnp.array(b))) == 0.0


@pytest.mark.parametrize("name", MONOTONE)
def test_zero_difference_gives_zero(name: str) -> None:
    """A flat neighbour means no slope, and must not divide by zero.

    This property is what makes the monotone limiters preserve a freestream right
    up against a reflecting wall, where one difference is zero and the other is
    not; ``none`` does not have it, and correspondingly needs a wider mask in the
    free-stream preservation test.
    """
    limiter = limiters.get_limiter(name)
    assert float(limiter(jnp.array(0.0), jnp.array(1.0))) == 0.0
    assert float(limiter(jnp.array(1.0), jnp.array(0.0))) == 0.0
    assert float(limiter(jnp.array(0.0), jnp.array(0.0))) == 0.0


@pytest.mark.parametrize("name", ALL)
def test_flat_field_gives_zero_slope(name: str) -> None:
    """Both differences zero must give zero slope for every limiter, ``none``
    included, and must not produce a NaN from a 0/0."""
    slope = float(limiters.get_limiter(name)(jnp.array(0.0), jnp.array(0.0)))
    assert slope == 0.0 and not np.isnan(slope)


@pytest.mark.parametrize("name", ALL)
def test_symmetry_under_sign_reversal(name: str) -> None:
    """psi(-a, -b) == -psi(a, b): the scheme must not prefer a direction."""
    limiter = limiters.get_limiter(name)
    a, b = jnp.array(0.7), jnp.array(1.9)
    assert float(limiter(-a, -b)) == pytest.approx(-float(limiter(a, b)), rel=1e-14)


@pytest.mark.parametrize("name", MONOTONE)
@given(a=finite, b=finite)
@settings(max_examples=200, deadline=None)
def test_limited_slope_stays_within_the_tvd_bound(name: str, a: float, b: float) -> None:
    """A TVD limiter's slope may not exceed twice the smaller difference.

    The Sweby region requires ``0 <= psi(r) <= min(2r, 2)``; in difference form
    that caps the slope magnitude at ``2 min(|a|, |b|)``. Exceeding it permits new
    extrema, which is exactly what produces oscillations at a shock.
    """
    limiter = limiters.get_limiter(name)
    slope = float(limiter(jnp.array(a), jnp.array(b)))
    bound = 2.0 * min(abs(a), abs(b))
    assert abs(slope) <= bound + 1e-9 * max(1.0, bound)


@pytest.mark.parametrize("name", MONOTONE)
@given(a=finite, b=finite)
@settings(max_examples=200, deadline=None)
def test_limited_slope_takes_the_sign_of_its_arguments(name: str, a: float, b: float) -> None:
    """The slope may not oppose the local trend."""
    slope = float(limiters.get_limiter(name)(jnp.array(a), jnp.array(b)))
    if a * b > 0:
        assert slope * a >= -1e-12
    else:
        assert slope == 0.0


def test_unlimited_is_the_arithmetic_mean() -> None:
    """The 'none' entry must be the plain central slope, for order studies."""
    a, b = jnp.array(0.4), jnp.array(1.2)
    assert float(limiters.unlimited(a, b)) == pytest.approx(0.8)
    # It is deliberately non-monotone: it does not vanish at an extremum.
    assert float(limiters.unlimited(jnp.array(1.0), jnp.array(-1.0))) == 0.0


def test_unknown_limiter_lists_the_available_ones() -> None:
    with pytest.raises(KeyError, match="minmod"):
        limiters.get_limiter("not_a_limiter")


def test_registry_rejects_duplicates() -> None:
    with pytest.raises(KeyError, match="already registered"):
        limiters.register_limiter("minmod")(lambda a, b: a)
