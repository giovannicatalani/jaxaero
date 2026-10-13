"""Unit tests for the numerical fluxes, in 2D and 3D.

Consistency and conservation are the two properties every entry in the registry
must have; they are checked over random physical states rather than a handful of
hand-picked ones, because a sign error in a single eigenvector can survive any
particular case while breaking others.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.state import magnitude, split, stack_primitive
from jaxaero.numerics import riemann
from jaxaero.physics import euler

from ..conftest import random_face_vectors, random_physical_states

GAMMA = 1.4
NAMES = sorted(riemann.FLUXES)
DIMS = [2, 3]


@pytest.fixture(scope="module", params=DIMS)
def states(request):
    dim = request.param
    q = jnp.asarray(random_physical_states(500, seed=3, dim=dim))
    q2 = jnp.asarray(random_physical_states(500, seed=4, dim=dim))
    return q, q2, jnp.asarray(random_face_vectors(500, seed=5, dim=dim))


@pytest.mark.parametrize("name", NAMES)
def test_flux_is_consistent(name: str, states) -> None:
    """``F_num(q, q, S) == F_analytic(q) . S``.

    With no jump there is nothing for the dissipation to act on, so the numerical
    flux must collapse exactly onto the analytic one. This is the single most
    effective test of a Riemann solver: it fails for almost any algebra slip.
    """
    q, _, s = states
    length = magnitude(s)
    exact = euler.normal_flux(q, s / length, GAMMA) * length
    got = riemann.get_flux(name)(q, q, s, GAMMA)
    relative = jnp.abs(got - exact) / jnp.maximum(jnp.abs(exact), 1e-10)
    assert float(jnp.max(relative)) < 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_flux_is_conservative(name: str, states) -> None:
    """``F(qL, qR, S) == -F(qR, qL, -S)``.

    The flux leaving one cell must equal the flux entering its neighbour. Without
    this the scheme is not conservative and mass is created at every face.
    """
    q, q2, s = states
    flux = riemann.get_flux(name)
    forward = flux(q, q2, s, GAMMA)
    reverse = flux(q2, q, -s, GAMMA)
    scale = jnp.maximum(jnp.abs(forward), 1.0)
    assert float(jnp.max(jnp.abs(forward + reverse) / scale)) < 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_flux_scales_linearly_with_face_area(name: str, states) -> None:
    """Doubling the face area vector doubles the integrated flux."""
    q, q2, s = states
    flux = riemann.get_flux(name)
    once = flux(q, q2, s, GAMMA)
    twice = flux(q, q2, 2.0 * s, GAMMA)
    assert np.allclose(np.asarray(twice), 2.0 * np.asarray(once), rtol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_zero_area_face_has_zero_flux(name: str) -> None:
    """A collapsed face (the pole of a 3D grid) carries nothing, and no NaN."""
    q = jnp.asarray(random_physical_states(8, seed=9, dim=3))
    got = riemann.get_flux(name)(q, q[:, ::-1], jnp.zeros((3, 8)), GAMMA)
    assert np.array_equal(np.asarray(got), np.zeros((5, 8)))


def _rotation(dim: int) -> np.ndarray:
    if dim == 2:
        c, s = np.cos(0.7), np.sin(0.7)
        return np.array([[c, -s], [s, c]])
    q, _ = np.linalg.qr(np.random.default_rng(11).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", NAMES)
def test_flux_is_rotationally_invariant(name: str, dim: int) -> None:
    """Rotating the state and the face together rotates the flux.

    Galilean/rotational invariance is a property of the Euler equations, and a
    flux that violates it has picked up a hidden dependence on the coordinate
    axes -- which on a curvilinear grid would show up as a spurious
    orientation-dependent error.
    """
    R = jnp.asarray(_rotation(dim))
    qL = jnp.asarray(random_physical_states(20, seed=12, dim=dim))
    qR = jnp.asarray(random_physical_states(20, seed=13, dim=dim))
    s = jnp.asarray(random_face_vectors(20, seed=14, dim=dim))

    def rotate(q):
        rho, v, p = split(q, dim)
        return stack_primitive(rho, R @ v, p)

    flux = riemann.get_flux(name)
    plain = np.asarray(flux(qL, qR, s, GAMMA))
    rotated = np.asarray(flux(rotate(qL), rotate(qR), R @ s, GAMMA))
    # Mass and energy are scalars; momentum rotates with the frame.
    np.testing.assert_allclose(rotated[0], plain[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(rotated[-1], plain[-1], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(rotated[1:dim + 1], np.asarray(R) @ plain[1:dim + 1], rtol=1e-11, atol=1e-12)


def test_three_dimensional_flux_reduces_to_two_dimensional() -> None:
    """A 3D state with ``w = 0`` through a face normal to ``z`` reproduces the 2D flux."""
    q2 = jnp.asarray(random_physical_states(50, seed=21))
    s2 = jnp.asarray(random_face_vectors(50, seed=22))
    q3 = jnp.concatenate([q2[:3], jnp.zeros((1, 50)), q2[3:]])
    s3 = jnp.concatenate([s2, jnp.zeros((1, 50))])
    for name in NAMES:
        f2 = np.asarray(riemann.get_flux(name)(q2, q2[:, ::-1], s2, GAMMA))
        f3 = np.asarray(riemann.get_flux(name)(q3, q3[:, ::-1], s3, GAMMA))
        np.testing.assert_allclose(f3[[0, 1, 2, 4]], f2, rtol=1e-12, atol=1e-13)
        assert np.max(np.abs(f3[3])) < 1e-13


@pytest.mark.parametrize("dim", DIMS)
def test_roe_average_reproduces_a_uniform_state(dim: int) -> None:
    """With equal states the Roe average is that state."""
    q = jnp.asarray(random_physical_states(5, seed=31, dim=dim))
    rho, velocity, enthalpy, a = euler.roe_average(q, q, GAMMA, dim)
    r, v, p = split(q, dim)
    np.testing.assert_allclose(rho, r, rtol=1e-14)
    np.testing.assert_allclose(velocity, v, rtol=1e-14)
    np.testing.assert_allclose(a, euler.sound_speed(r, p, GAMMA), rtol=1e-14)
    np.testing.assert_allclose(enthalpy, euler.total_enthalpy(r, v, p, GAMMA), rtol=1e-14)


def test_roe_jump_condition_holds_across_a_discontinuity() -> None:
    """The defining Roe property: ``A(qL,qR) (UR - UL) == FR - FL``.

    Rather than form the matrix, this checks the equivalent statement that the Roe
    flux with the dissipation built from the *signed* eigenvalues reproduces the
    upwind flux, by confirming the wave decomposition reconstructs the exact flux
    difference. This is what makes Roe's scheme capture a steady shock crisply
    instead of smearing it.
    """
    qL = jnp.asarray([[1.0], [0.9], [0.0], [1.0 / 1.4]])
    qR = jnp.asarray([[1.6], [0.5], [0.0], [1.3]])
    n = jnp.asarray([[1.0], [0.0]])

    rho, (u, v), enthalpy, a = euler.roe_average(qL, qR, GAMMA, 2)
    d_rho = qR[0] - qL[0]
    d_p = qR[3] - qL[3]
    d_vn = qR[1] - qL[1]

    alpha_minus = (d_p - rho * a * d_vn) / (2.0 * a * a)
    alpha_entropy = d_rho - d_p / (a * a)
    alpha_plus = (d_p + rho * a * d_vn) / (2.0 * a * a)
    kinetic = 0.5 * (u * u + v * v)

    r_minus = jnp.stack([jnp.ones_like(rho), u - a, v, enthalpy - a * u])
    r_entropy = jnp.stack([jnp.ones_like(rho), u, v, kinetic])
    r_plus = jnp.stack([jnp.ones_like(rho), u + a, v, enthalpy + a * u])

    reconstructed = (
        (u - a) * alpha_minus * r_minus
        + u * alpha_entropy * r_entropy
        + (u + a) * alpha_plus * r_plus
    )
    exact = euler.normal_flux(qR, n, GAMMA) - euler.normal_flux(qL, n, GAMMA)
    assert np.allclose(np.asarray(reconstructed), np.asarray(exact), rtol=1e-11, atol=1e-13)


def test_unknown_flux_lists_the_available_ones() -> None:
    with pytest.raises(KeyError, match="roe"):
        riemann.get_flux("not_a_flux")


# --- Contact-wave resolution ------------------------------------------------
# A stationary contact -- equal pressure, zero normal velocity, jumping density --
# carries no flux except the pressure term, and a flux that resolves the contact
# reproduces that exactly. Every registered flux must: the slip wall has vn = 0
# along its whole length, so a flux that smears contacts smears exactly where CL
# and CD are computed.


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", NAMES)
def test_stationary_contact_is_resolved_exactly(name: str, dim: int) -> None:
    """``F = (0, p n, 0) |S|`` across a stationary contact.

    The flux has no mass, no momentum beyond pressure and no energy to carry,
    because nothing crosses the face. Any numerical dissipation that acts on the
    density jump shows up immediately as a non-zero mass flux, so this is the
    test that a contact-resolving solver's star states are right -- the part the
    consistency and conservation tests cannot see.
    """
    rng = np.random.default_rng(17)
    count = 64
    pressure = rng.uniform(0.5, 2.0, count)
    s = random_face_vectors(count, seed=18, dim=dim)
    length = np.linalg.norm(s, axis=0)
    n = s / length
    # Velocity purely tangential (a random vector minus its normal part).
    v = rng.uniform(-1.5, 1.5, (dim, count))
    v = v - np.sum(v * n, axis=0) * n

    def side(density):
        return jnp.asarray(np.concatenate([density[None], v, pressure[None]]))

    qL, qR = side(rng.uniform(0.2, 0.5, count)), side(rng.uniform(1.6, 4.0, count))
    flux = np.asarray(riemann.get_flux(name)(qL, qR, jnp.asarray(s), GAMMA))

    scale = float(np.max(np.abs(pressure * length)))
    assert np.max(np.abs(flux[0])) < 1e-12 * scale          # mass
    assert np.max(np.abs(flux[-1])) < 1e-12 * scale         # energy
    np.testing.assert_allclose(flux[1:dim + 1], pressure * n * length, rtol=1e-12, atol=1e-12 * scale)
