"""Unit tests for the SA-noft2 / SA-neg pointwise functions.

Two kinds. Values: each function against a scalar transcription of the TMR
formulas, written out branch by branch with ``math`` (a second, independent
reading of the page), at sample points chosen to hit every branch -- positive
and negative ``nu~``, ``S~`` clipped and not, ``r`` capped and not. Structure:
identities the model is built on, which a transcription error in either reading
would break.
"""

from __future__ import annotations


import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.physics import spalart_allmaras as sa

C = sa.SA


def reference(nu_t: float, nu: float, omega: float, d: float, rho: float = 1.0):
    """Scalar SA-noft2 + SA-neg, straight from the TMR page."""
    cb1, sigma, cb2, kappa = 0.1355, 2.0 / 3.0, 0.622, 0.41
    cw2, cw3, cv1, c2, c3, cn1 = 0.3, 2.0, 7.1, 0.7, 0.9, 16.0
    cw1 = cb1 / kappa**2 + (1 + cb2) / sigma
    chi = nu_t / nu
    fv1 = chi**3 / (chi**3 + cv1**3)
    fv2 = 1 - chi / (1 + chi * fv1)
    if nu_t >= 0:
        s_bar = nu_t * fv2 / (kappa**2 * d**2)
        if s_bar >= -c2 * omega:
            s_t = omega + s_bar
        else:
            s_t = omega + omega * (c2**2 * omega + c3 * s_bar) / ((c3 - 2 * c2) * omega - s_bar)
        r = min(nu_t / (s_t * kappa**2 * d**2), 10.0)
        g = r + cw2 * (r**6 - r)
        fw = g * ((1 + cw3**6) / (g**6 + cw3**6)) ** (1 / 6)
        P, D = cb1 * s_t * nu_t, cw1 * fw * (nu_t / d) ** 2
        fn, mu_t = 1.0, rho * nu_t * fv1
    else:
        P, D = cb1 * omega * nu_t, -cw1 * (nu_t / d) ** 2
        fn, mu_t = (cn1 + chi**3) / (cn1 - chi**3), 0.0
    return P, D, fn, mu_t


# (nu~, nu, Omega, d): log layer, viscous sublayer, wake/freestream, the S~
# clip (large negative S_bar needs chi above ~1 with fv2 < 0), r capped (tiny
# Omega), and negative nu~.
POINTS = [
    (4.1e-3, 1e-4, 50.0, 2e-3),
    (2e-5, 1e-4, 4e3, 1e-5),
    (3e-4, 1e-4, 0.5, 0.3),
    (1.5e-4, 1e-4, 0.02, 1e-3),
    (5e-2, 1e-4, 1e-6, 0.05),
    (-2e-4, 1e-4, 30.0, 1e-3),
    (-5e-5, 1e-4, 1.0, 0.2),
]


@pytest.mark.parametrize("nu_t, nu, omega, d", POINTS)
def test_source_and_viscosity_match_the_tmr_formulas(nu_t, nu, omega, d) -> None:
    P_ref, D_ref, fn_ref, mut_ref = reference(nu_t, nu, omega, d, rho=1.3)
    args = tuple(jnp.asarray(x) for x in (nu_t, nu, omega, d))
    P, D = sa.source(*args)
    assert float(P) == pytest.approx(P_ref, rel=1e-13)
    assert float(D) == pytest.approx(D_ref, rel=1e-13)
    assert float(sa.f_n(args[0], args[1])) == pytest.approx(fn_ref, rel=1e-13)
    mu_t = float(sa.eddy_viscosity(jnp.asarray(1.3), args[0], args[1]))
    assert mu_t == pytest.approx(mut_ref, rel=1e-13, abs=0.0)


def test_the_sample_points_reach_every_branch() -> None:
    """So the test above cannot pass by never entering a branch."""
    nu_t, nu, omega, d = (np.array(c) for c in zip(*POINTS))
    chi = nu_t / nu
    fv1 = chi**3 / (chi**3 + C.c_v1**3)
    s_bar = nu_t * (1 - chi / (1 + chi * fv1)) / (C.kappa * d) ** 2
    positive = nu_t >= 0
    assert np.any(positive & (s_bar < -C.c_2 * omega)), "S~ clip never taken"
    assert np.any(positive & (s_bar >= -C.c_2 * omega))
    assert np.any(~positive)
    r = np.asarray(sa.r_of(jnp.asarray(nu_t), sa.s_tilde(*map(jnp.asarray, (omega, nu_t, nu, d))),
                           jnp.asarray(d)))
    assert np.any(positive & (r == C.r_max)) and np.any(positive & (r < C.r_max))


def test_constants() -> None:
    assert C.c_w1 == pytest.approx(3.2390678, rel=1e-7)
    assert C.sigma == 2.0 / 3.0


def test_damping_functions_limits() -> None:
    assert float(sa.f_v1(jnp.asarray(0.0))) == 0.0
    assert float(sa.f_v1(jnp.asarray(1e4))) == pytest.approx(1.0, abs=1e-9)
    assert float(sa.f_v1(jnp.asarray(C.c_v1))) == pytest.approx(0.5, rel=1e-15)
    assert float(sa.f_v2(jnp.asarray(0.0))) == 1.0
    assert float(sa.f_w(jnp.asarray(1.0))) == pytest.approx(1.0, rel=1e-15)
    # f_w saturates as r -> r_max: destruction stays bounded in the wake.
    assert float(sa.f_w(jnp.asarray(10.0))) == pytest.approx((1 + 64) ** (1 / 6), rel=1e-3)


def test_log_layer_balance() -> None:
    """In the log layer ``nu~ = kappa u_tau d``, ``Omega = u_tau / (kappa d)``,
    and with ``f_v2`` neglected (chi large) ``S~ = Omega``, ``r = 1``,
    ``f_w = 1``: production minus destruction equals minus the diffusion,
    ``-(1 + c_b2) kappa^2 u_tau^2 / sigma``. That identity is what defines
    ``c_w1``."""
    u_tau, d = 0.05, 1e-3
    nu_t, omega = C.kappa * u_tau * d, u_tau / (C.kappa * d)
    r = float(sa.r_of(jnp.asarray(nu_t), jnp.asarray(omega), jnp.asarray(d)))
    assert r == pytest.approx(1.0, rel=1e-14)
    production = C.c_b1 * omega * nu_t
    destruction = C.c_w1 * float(sa.f_w(jnp.asarray(r))) * (nu_t / d) ** 2
    diffusion = (1.0 + C.c_b2) * (C.kappa * u_tau) ** 2 / C.sigma
    assert production - destruction + diffusion == pytest.approx(0.0, abs=1e-15)


def test_s_tilde_clip_is_continuous_and_positive() -> None:
    """Both branches give ``(1 - c_2) Omega`` at ``S_bar = -c_2 Omega``, and the
    clipped branch stays positive however negative ``S_bar`` gets."""
    omega = 2.0
    s_bar = jnp.asarray([-C.c_2 * omega, -10.0 * omega, -1e6 * omega])
    clipped = omega + omega * (C.c_2**2 * omega + C.c_3 * s_bar) / (
        (C.c_3 - 2 * C.c_2) * omega - s_bar)
    assert float(clipped[0]) == pytest.approx((1 - C.c_2) * omega, rel=1e-14)
    assert bool(jnp.all(clipped > 0.0))


def test_sa_neg() -> None:
    """Negative nu~: no eddy viscosity; the wall term ``-D = c_w1 (nu~/d)^2``
    is positive, so it pushes a negative nu~ back toward zero; and the
    diffusion coefficient stays positive."""
    nu = jnp.asarray(1e-4)
    for nu_t in (-1e-6, -5e-5, -2e-4):
        P, D = sa.source(jnp.asarray(nu_t), nu, jnp.asarray(10.0), jnp.asarray(1e-2))
        assert float(D) < 0.0 and float(-D) > 0.0
        assert float(sa.eddy_viscosity(jnp.asarray(1.0), jnp.asarray(nu_t), nu)) == 0.0
        coefficient = float(sa.diffusion_coefficient(jnp.asarray(1.0), jnp.asarray(nu_t), nu))
        assert coefficient > 0.0


def test_freestream_value_is_what_tmr_quotes() -> None:
    """nu~ = 3 nu_inf gives nu_t = 0.210438 nu_inf (TMR SA page)."""
    nu = 1.0
    mu_t = float(sa.eddy_viscosity(jnp.asarray(1.0), jnp.asarray(3.0 * nu), jnp.asarray(nu)))
    assert mu_t == pytest.approx(0.210438, rel=1e-5)
