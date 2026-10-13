"""Numerical fluxes ``F(qL, qR, S)``: primitive states either side, the face
area vector ``S (d, ...)`` (L to R), the flux integrated over the face, in any
dimension ``d``. Each entry is tested for consistency, ``F(q, q, S) = F(q).S``,
and conservation, ``F(qL, qR, S) = -F(qR, qL, -S)``. Both resolve a stationary
contact exactly; ``hllc`` (default) needs no entropy fix, ``roe`` is the
independent check.
"""

from __future__ import annotations

from typing import Callable, Protocol

import jax.numpy as jnp
from jax import Array

from ..core.state import dot, magnitude, primitive_to_conservative, split
from ..physics import euler


class NumericalFlux(Protocol):
    """Flux through one face, given the states either side."""

    def __call__(self, qL: Array, qR: Array, s: Array, gamma: float) -> Array:
        ...


FLUXES: dict[str, NumericalFlux] = {}


def register_flux(name: str) -> Callable[[NumericalFlux], NumericalFlux]:
    def decorate(fn: NumericalFlux) -> NumericalFlux:
        if name in FLUXES:
            raise KeyError(f"flux '{name}' already registered")
        FLUXES[name] = fn
        return fn

    return decorate


def get_flux(name: str) -> NumericalFlux:
    try:
        return FLUXES[name]
    except KeyError:
        raise KeyError(f"unknown flux '{name}'. Available: {sorted(FLUXES)}") from None


def _unit_normal(s: Array) -> tuple[Array, Array]:
    """Split an area vector into its magnitude and unit normal; a zero-area
    face (a pole) gets a zero normal, so its flux is exactly zero."""
    length = magnitude(s)
    return length, s / jnp.maximum(length, 1e-300)


@register_flux("roe")
def roe_flux(qL: Array, qR: Array, s: Array, gamma: float) -> Array:
    """Roe's flux-difference-splitting flux.

    The flux is the arithmetic mean minus the characteristic dissipation

        F = 1/2 (F_L + F_R) - 1/2 sum_k |lambda_k| alpha_k r_k

    over the waves of the Euler system: two acoustic (``Vn -/+ a``), one
    entropy and ``d - 1`` shear waves (all ``Vn``). Wave strengths ``alpha_k``
    are the projection of the state jump onto the left eigenvectors, evaluated
    at the Roe average so that the scheme is exact for an isolated
    discontinuity. The shear waves together carry the tangential velocity
    jump ``dV_t = dV - (dV . n) n``: ``alpha r = rho (0, dV_t, V . dV_t)``.
    No entropy fix (docs/dev/notes.md).
    """
    dim = s.shape[0]
    length, n = _unit_normal(s)

    rho, velocity, enthalpy, a = euler.roe_average(qL, qR, gamma, dim)
    vn = dot(velocity, n)

    rho_l, v_l, p_l = split(qL, dim)
    rho_r, v_r, p_r = split(qR, dim)
    d_rho, d_p, d_v = rho_r - rho_l, p_r - p_l, v_r - v_l
    d_vn = dot(d_v, n)
    d_vt = d_v - d_vn * n

    # Wave strengths: acoustic pair, entropy.
    a_squared = a * a
    alpha_minus = (d_p - rho * a * d_vn) / (2.0 * a_squared)
    alpha_entropy = d_rho - d_p / a_squared
    alpha_plus = (d_p + rho * a * d_vn) / (2.0 * a_squared)

    lam_minus = jnp.abs(vn - a)
    lam_mid = jnp.abs(vn)
    lam_plus = jnp.abs(vn + a)

    kinetic = 0.5 * dot(velocity, velocity)
    one = jnp.ones_like(rho)

    def vector(scalar, v, energy):
        return jnp.concatenate([scalar[None], v, energy[None]], axis=0)

    # Right eigenvectors, in conservative-variable order; the shear waves
    # already multiplied by their strengths.
    r_minus = vector(one, velocity - a * n, enthalpy - a * vn)
    r_entropy = vector(one, velocity, kinetic)
    shear = vector(jnp.zeros_like(rho), rho * d_vt, rho * dot(velocity, d_vt))
    r_plus = vector(one, velocity + a * n, enthalpy + a * vn)

    dissipation = (
        lam_minus * alpha_minus * r_minus
        + lam_mid * alpha_entropy * r_entropy
        + lam_mid * shear
        + lam_plus * alpha_plus * r_plus
    )

    central = 0.5 * (euler.normal_flux(qL, n, gamma) + euler.normal_flux(qR, n, gamma))
    return length * (central - 0.5 * dissipation)


@register_flux("hllc")
def hllc_flux(qL: Array, qR: Array, s: Array, gamma: float) -> Array:
    """Harten-Lax-van Leer flux with a restored Contact wave.

    HLL approximates the Riemann fan by two acoustic waves and one uniform state
    between them, which is robust but smears contacts and shear layers. HLLC
    puts the contact wave back, so the fan is

        S_L  (acoustic) | S_M (contact) | S_R  (acoustic)

    and the flux is whichever of the four regions the face sits in:

        F = F_L                        if      0 <= S_L
          = F_L + S_L (U*_L - U_L)     if S_L < 0 <= S_M
          = F_R + S_R (U*_R - U_R)     if S_M < 0 <= S_R
          = F_R                        if S_R < 0

    Across each acoustic wave the normal velocity becomes ``S_M`` and the
    pressure becomes ``p*``, while the tangential velocity is carried through
    unchanged; that gives the star states directly from Rankine-Hugoniot.

    Wave speeds after Einfeldt: the more extreme of the one-sided and the
    Roe-averaged value, so the fan contains the true waves.

    References: Toro, Spruce & Speares, "Restoration of the contact surface in
    the HLL Riemann solver", Shock Waves 4 (1994); Toro, *Riemann Solvers and
    Numerical Methods for Fluid Dynamics*, 3rd ed., section 10.4. Einfeldt, "On
    Godunov-type methods for gas dynamics", SIAM J. Numer. Anal. 25 (1988).
    """
    dim = s.shape[0]
    length, n = _unit_normal(s)

    rho_l, v_l, p_l = split(qL, dim)
    rho_r, v_r, p_r = split(qR, dim)
    vn_l = dot(v_l, n)
    vn_r = dot(v_r, n)
    a_l = euler.sound_speed(rho_l, p_l, gamma)
    a_r = euler.sound_speed(rho_r, p_r, gamma)

    _, v_roe, _, a_roe = euler.roe_average(qL, qR, gamma, dim)
    vn_roe = dot(v_roe, n)
    s_l = jnp.minimum(vn_l - a_l, vn_roe - a_roe)
    s_r = jnp.maximum(vn_r + a_r, vn_roe + a_roe)

    # Contact speed (Toro eq. 10.37). Both terms of the denominator have the
    # same sign for physical states, so it cannot vanish; it is clamped anyway
    # because every branch below is evaluated whether or not it is selected, and
    # a NaN in a discarded branch of `where` still propagates into gradients.
    mass_l = rho_l * (s_l - vn_l)
    mass_r = rho_r * (s_r - vn_r)
    s_m = (p_r - p_l + mass_l * vn_l - mass_r * vn_r) / jnp.minimum(
        mass_l - mass_r, -1e-300
    )

    def star_flux(rho, velocity, p, vn, s_k, flux_k, u_k, widening):
        """``F_K + S_K (U*_K - U_K)`` for one side of the contact: the normal
        velocity becomes ``S_M``, the tangential part ``V - Vn n`` is kept."""
        p_star = p + rho * (vn - s_k) * (vn - s_m)
        rho_star = rho * (s_k - vn) / widening
        energy_k = u_k[dim + 1]
        energy_star = ((s_k - vn) * energy_k - p * vn + p_star * s_m) / widening
        u_star = jnp.concatenate(
            [rho_star[None], rho_star * (velocity + (s_m - vn) * n), energy_star[None]], axis=0
        )
        return flux_k + s_k * (u_star - u_k)

    u_l, u_r = _to_conservative(qL, gamma, dim), _to_conservative(qR, gamma, dim)
    flux_l = euler.normal_flux(qL, n, gamma)
    flux_r = euler.normal_flux(qR, n, gamma)

    # S_L < S_M < S_R strictly for physical states; clamping keeps the sign and
    # protects the unselected branches, as above.
    star_l = star_flux(rho_l, v_l, p_l, vn_l, s_l, flux_l, u_l,
                       jnp.minimum(s_l - s_m, -1e-300))
    star_r = star_flux(rho_r, v_r, p_r, vn_r, s_r, flux_r, u_r,
                       jnp.maximum(s_r - s_m, 1e-300))

    flux = jnp.where(
        s_l >= 0.0,
        flux_l,
        jnp.where(s_m >= 0.0, star_l, jnp.where(s_r >= 0.0, star_r, flux_r)),
    )
    return length * flux


def _to_conservative(q: Array, gamma: float, dim: int) -> Array:
    """Stacked primitive ``(d+2, ...)`` to stacked conservative."""
    return primitive_to_conservative(*split(q, dim), gamma)
