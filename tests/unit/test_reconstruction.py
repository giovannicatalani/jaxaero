"""Unit tests for MUSCL reconstruction.

On a uniform grid a second-order reconstruction must return the exact face
value of a linear field, for every registered limiter: a linear field has equal
neighbouring differences, and every limiter must return exactly ``a`` when
``a == b``. Both face families share one stencil, so both axes are checked.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.numerics import reconstruction as rc
from jaxaero.numerics.limiters import LIMITERS, get_limiter

GHOST = 2
N, OTHER = 10, 3


I_FACES, J_FACES = 0, 1     # index directions


def linear_field(axis: int, slope: float = -0.7, offset: float = 3.2):
    """Padded ``(4, ...)`` field linear in ``axis``, with unit cell spacing.

    Padded cell ``k`` has centre ``k + 1/2``, so face ``f`` (between padded cells
    ``f+1`` and ``f+2``) sits at ``f + 2``.
    """
    shape = (N + 2 * GHOST, OTHER + 2 * GHOST) if axis == I_FACES else (
        OTHER + 2 * GHOST, N + 2 * GHOST)
    centres = np.arange(N + 2 * GHOST) + 0.5
    profile = offset + slope * centres
    field = profile[:, None] if axis == I_FACES else profile[None, :]
    faces = offset + slope * (np.arange(N + 1) + 2.0)
    return jnp.asarray(np.broadcast_to(field, (4,) + shape)), faces


def counts(axis: int) -> tuple[int, int]:
    return (N, OTHER) if axis == I_FACES else (OTHER, N)


@pytest.mark.parametrize("axis", [I_FACES, J_FACES])
@pytest.mark.parametrize("name", sorted(LIMITERS))
def test_linear_field_is_reconstructed_exactly(name: str, axis: int) -> None:
    """Both sides of every face recover the exact linear value."""
    padded, expected = linear_field(axis)
    qL, qR = rc.face_states(padded, GHOST, counts(axis), get_limiter(name), axis)
    take = (0, slice(None), 0) if axis == I_FACES else (0, 0, slice(None))
    assert np.allclose(np.asarray(qL)[take], expected, rtol=1e-14)
    assert np.allclose(np.asarray(qR)[take], expected, rtol=1e-14)


@pytest.mark.parametrize("axis", [I_FACES, J_FACES])
def test_first_order_faces_are_the_cell_values(axis: int) -> None:
    rng = np.random.default_rng(3)
    shape = (4, N + 2 * GHOST, OTHER + 2 * GHOST) if axis == I_FACES else (
        4, OTHER + 2 * GHOST, N + 2 * GHOST)
    padded = jnp.asarray(rng.normal(size=shape))
    qL, qR = rc.first_order_faces(padded, GHOST, counts(axis), axis)
    _, q_m, q_p, _ = rc._stencil(padded, GHOST, counts(axis), axis)
    assert np.array_equal(np.asarray(qL), np.asarray(q_m))
    assert np.array_equal(np.asarray(qR), np.asarray(q_p))


def test_minmod_flattens_an_extremum() -> None:
    """At a local maximum the limited slope is zero, so faces take the cell value."""
    profile = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    padded = jnp.asarray(np.broadcast_to(profile[None, :, None], (4, 8, 5)))
    qL, _ = rc.face_states(padded, GHOST, (4, 1), get_limiter("minmod"), I_FACES)
    # Face 2 lies right of padded cell 3, the maximum.
    assert float(qL[0, 2, 0]) == 3.0


def test_kappa_one_third_is_exact_for_quadratic_cell_averages() -> None:
    """kappa = 1/3 recovers the face value of a quadratic from its cell
    averages (third order in 1D); every kappa is exact for a linear field."""
    h = 0.1
    left = np.arange(-GHOST, 6 + GHOST) * h                      # cell left edges
    avg = lambda f, F: (F(left + h) - F(left)) / h               # noqa: E731
    faces = np.arange(7) * h
    quadratic = avg(None, lambda x: x**3 / 3.0)                  # average of x^2
    padded = jnp.asarray(np.broadcast_to(quadratic[None, :, None], (1, len(left), 1 + 2 * GHOST)))
    qL, qR = rc.face_states(padded, GHOST, (6, 1), get_limiter("none"), I_FACES, kappa=1.0 / 3.0)
    np.testing.assert_allclose(np.asarray(qL)[0, :, 0], faces**2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(qR)[0, :, 0], faces**2, rtol=1e-12, atol=1e-14)
    linear = avg(None, lambda x: 0.5 * x**2 - 2.0 * x)
    padded = jnp.asarray(np.broadcast_to(linear[None, :, None], (1, len(left), 1 + 2 * GHOST)))
    for kappa in (-1.0, 0.0, 1.0 / 3.0, 0.5):
        qL, qR = rc.face_states(padded, GHOST, (6, 1), get_limiter("none"), I_FACES, kappa=kappa)
        np.testing.assert_allclose(np.asarray(qL)[0, :, 0], faces - 2.0, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(np.asarray(qR)[0, :, 0], faces - 2.0, rtol=1e-12, atol=1e-13)


def test_kappa_needs_the_unlimited_scheme() -> None:
    from jaxaero.core.config import NumericsConfig

    with pytest.raises(ValueError, match="limiter 'none'"):
        NumericsConfig(limiter="minmod", kappa=1.0 / 3.0)
    assert NumericsConfig(limiter="none", kappa=1.0 / 3.0).kappa == pytest.approx(1.0 / 3.0)
