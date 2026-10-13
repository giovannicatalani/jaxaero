"""Unit tests for the normal flux Jacobian.

``normal_flux_jacobian`` is obtained by differentiating ``normal_flux``, so it
cannot disagree with the flux by a transcription error. What these tests check
is that the *flux* and the differentiation together have the properties the
preconditioner relies on, and that the matrix really is the derivative that is
claimed: its eigenvalues are the acoustic and convective wave speeds, it
reproduces the flux through homogeneity, and it matches a central difference.

A failure of the eigenvalue test would invalidate every upwind scheme in the
code, not just the preconditioner.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.state import conservative_to_primitive, dot, magnitude, primitive_to_conservative, split, stack_primitive
from jaxaero.physics import euler

from ..conftest import random_face_vectors, random_physical_states

GAMMA = 1.4


@pytest.fixture(scope="module", params=[2, 3])
def states(request):
    dim = request.param
    q = jnp.asarray(random_physical_states(300, seed=11, dim=dim))
    return q, jnp.asarray(random_face_vectors(300, seed=12, dim=dim))


def test_jacobian_recovers_the_flux(states) -> None:
    """``A U = F . S``.

    The Euler flux is homogeneous of degree one in the conservative variables,
    so Euler's homogeneous-function theorem makes this an *identity*, not an
    approximation. It is the sharpest available check: it ties all sixteen
    entries to the flux at once and holds to round-off.
    """
    q, s = states
    jac = euler.normal_flux_jacobian(q, s, GAMMA)
    U = primitive_to_conservative(*split(q, s.shape[0]), GAMMA)

    recovered = jnp.einsum("rcn,cn->rn", jac, U)
    expected = euler.normal_flux(q, s, GAMMA)

    scale = jnp.maximum(jnp.abs(expected).max(axis=0), 1e-12)
    assert np.max(np.abs(recovered - expected) / scale) < 1e-12


def test_jacobian_eigenvalues_are_the_wave_speeds(states) -> None:
    """``eig(A) = {Vn (d times), Vn + a, Vn - a}`` scaled by the face area.

    The area vector scales the whole matrix linearly, so every eigenvalue
    carries one factor of ``|S|``.
    """
    q, s = states
    dim = s.shape[0]
    jac = np.asarray(euler.normal_flux_jacobian(q, s, GAMMA))

    length = np.asarray(magnitude(s))
    rho, velocity, p = (np.asarray(x) for x in split(q, dim))
    a = np.sqrt(GAMMA * p / rho)
    vn = np.asarray(dot(velocity, s)) / length

    for n in range(q.shape[1]):
        eigenvalues = np.sort(np.linalg.eigvals(jac[:, :, n]).real)
        expected = np.sort(length[n] * np.array([vn[n]] * dim + [vn[n] + a[n], vn[n] - a[n]]))
        assert np.allclose(eigenvalues, expected, rtol=1e-10, atol=1e-10)


def test_jacobian_matches_a_central_difference(states) -> None:
    """Column ``k`` equals ``dF/dU_k`` by central differences.

    Redundant with the two identities above only if the flux is exactly the
    Euler flux; it is kept because it would still catch a wrong *argument*
    (differentiating with respect to primitives, say) which homogeneity alone
    would not.
    """
    q, s = states
    q, s = q[:, :40], s[:, :40]
    dim = s.shape[0]
    jac = np.asarray(euler.normal_flux_jacobian(q, s, GAMMA))
    U = np.asarray(primitive_to_conservative(*split(q, dim), GAMMA))

    def flux_of(u_vec: np.ndarray) -> np.ndarray:
        q_vec = stack_primitive(*conservative_to_primitive(jnp.asarray(u_vec), GAMMA, dim))
        return np.asarray(euler.normal_flux(q_vec, s, GAMMA))

    for k in range(dim + 2):
        step = 1e-7 * np.maximum(np.abs(U[k]), 1.0)
        perturbation = np.zeros_like(U)
        perturbation[k] = step
        derivative = (flux_of(U + perturbation) - flux_of(U - perturbation)) / (2.0 * step)
        scale = np.maximum(np.abs(jac[:, k, :]).max(axis=0), 1e-10)
        assert np.max(np.abs(derivative - jac[:, k, :]) / scale) < 1e-6


def test_jacobian_is_linear_in_the_area_vector(states) -> None:
    """``A(q; S1 + S2) = A(q; S1) + A(q; S2)``.

    Relied on in ``preconditioners.j_line_blocks``, which negates the area
    vector by negating the matrix, and in the telescoping argument that makes
    the approximate diagonal a scalar.
    """
    q, s = states
    s2 = jnp.asarray(random_face_vectors(q.shape[1], seed=13, dim=s.shape[0]))

    combined = euler.normal_flux_jacobian(q, s + s2, GAMMA)
    separate = euler.normal_flux_jacobian(q, s, GAMMA) + euler.normal_flux_jacobian(q, s2, GAMMA)
    assert np.allclose(combined, separate, rtol=1e-12, atol=1e-14)
