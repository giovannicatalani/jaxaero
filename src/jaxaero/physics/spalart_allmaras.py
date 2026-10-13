"""The Spalart-Allmaras turbulence model, SA-noft2 with SA-neg: pointwise pieces.

Pure functions of local quantities, like the rest of ``physics``; nothing here
knows about grids. The transport equation, in the conservative compressible
form this code integrates for the fifth variable ``rho nu~``:

    d(rho nu~)/dt + div(rho u nu~)
        = rho P - rho D + (1/sigma) [ div((mu + rho nu~ f_n) grad nu~)
                                      + c_b2 rho |grad nu~|^2 ]

with, for ``nu~ >= 0`` (standard SA without the ft2 term, "SA-noft2"),

    P = c_b1 S~ nu~,          D = c_w1 f_w (nu~/d)^2,          f_n = 1,

and for ``nu~ < 0`` (SA-neg, which leaves a well-resolved solution unchanged
and keeps a transiently negative ``nu~`` from destabilizing the iteration),

    P = c_b1 Omega nu~,       D = -c_w1 (nu~/d)^2,
    f_n = (c_n1 + chi^3) / (c_n1 - chi^3),     mu_t = 0.

Definitions, all from the TMR SA page:

    chi  = nu~ / nu                  f_v1 = chi^3 / (chi^3 + c_v1^3)
    f_v2 = 1 - chi / (1 + chi f_v1)  S_bar = nu~ f_v2 / (kappa^2 d^2)
    S~   = Omega + S_bar                                 if S_bar >= -c_2 Omega
         = Omega + Omega (c_2^2 Omega + c_3 S_bar) / ((c_3 - 2 c_2) Omega - S_bar)
                                                          otherwise    (note 1c)
    r    = min(nu~ / (S~ kappa^2 d^2), 10)
    g    = r + c_w2 (r^6 - r)        f_w = g ((1 + c_w3^6) / (g^6 + c_w3^6))^(1/6)
    mu_t = rho nu~ f_v1

``Omega`` is the vorticity magnitude. Nondimensional ``nu~`` is scaled like
``mu / rho`` (by ``a_inf c``), so with ``mu_inf = M / Re`` no Reynolds number
appears here; the far-field value is ``3 nu_inf = 3 M / Re`` and the wall
value 0.

The TMR page writes the incompressible form; the conservative form above is the
usual compressible one, with ``rho`` on every term. At M = 0.15 the two
differ negligibly; the conservative form is the one a finite-volume code can
discretize without a source term in ``div u``.

References
----------
* NASA TMR, "The Spalart-Allmaras Turbulence Model",
  <https://tmbwg.github.io/turbmodels/spalart.html> (equations, constants,
  SA-neg, S~ clipping note 1c, SA-noft2).
* Spalart and Allmaras, "A One-Equation Turbulence Model for Aerodynamic
  Flows", *Recherche Aerospatiale* 1 (1994) 5-21.
* Allmaras, Johnson and Spalart, "Modifications and Clarifications for the
  Implementation of the Spalart-Allmaras Turbulence Model", ICCFD7-1902 (2012):
  SA-neg and the S~ clipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class SAConstants:
    """Model constants, TMR SA page. ``c_w1`` is derived, not free.

    ``c_2``, ``c_3`` are the S~ clipping constants of note 1(c); ``c_n1`` is
    SA-neg's; ``r_max`` caps ``r``.
    """

    c_b1: float = 0.1355
    sigma: float = 2.0 / 3.0
    c_b2: float = 0.622
    kappa: float = 0.41
    c_w2: float = 0.3
    c_w3: float = 2.0
    c_v1: float = 7.1
    c_2: float = 0.7
    c_3: float = 0.9
    c_n1: float = 16.0
    r_max: float = 10.0

    @property
    def c_w1(self) -> float:
        """``c_b1 / kappa^2 + (1 + c_b2) / sigma``: the value that balances
        production, destruction and diffusion in the log layer."""
        return self.c_b1 / self.kappa**2 + (1.0 + self.c_b2) / self.sigma


SA = SAConstants()


@dataclass(frozen=True)
class SAModel:
    """The model for one flow: ``nu_inf = M / Re`` scales the stored ``nu~ / nu_inf``."""

    nu_inf: float
    freestream_ratio: float = 3.0
    constants: SAConstants = SA

    @classmethod
    def of(cls, flow) -> "SAModel | None":
        """The model for a ``FlowConditions``; ``None`` when laminar."""
        if flow.turbulence is None:
            return None
        return cls(nu_inf=flow.mu_inf, freestream_ratio=flow.nu_tilde_ratio)

    def eddy_viscosity(self, rho: Array, nu_hat: Array, mu: Array) -> Array:
        """``mu_t`` from the stored, scaled variable ``nu_hat = nu~ / nu_inf``."""
        return eddy_viscosity(rho, nu_hat * self.nu_inf, mu / rho, self.constants)


def f_v1(chi: Array, c: SAConstants = SA) -> Array:
    chi3 = chi**3
    return chi3 / (chi3 + c.c_v1**3)


def f_v2(chi: Array, c: SAConstants = SA) -> Array:
    return 1.0 - chi / (1.0 + chi * f_v1(chi, c))


def eddy_viscosity(rho: Array, nu_tilde: Array, nu: Array, c: SAConstants = SA) -> Array:
    """``mu_t = rho nu~ f_v1(chi)``, and zero where ``nu~ < 0`` (SA-neg)."""
    chi = nu_tilde / nu
    return jnp.where(nu_tilde > 0.0, rho * nu_tilde * f_v1(chi, c), 0.0)


def s_tilde(omega: Array, nu_tilde: Array, nu: Array, d: Array, c: SAConstants = SA) -> Array:
    """Modified vorticity, clipped by TMR note 1(c) so it stays positive.

    Continuous at ``S_bar = -c_2 Omega``, where both branches give
    ``(1 - c_2) Omega``; the clipped branch stays above ``0.3 Omega``.
    """
    chi = nu_tilde / nu
    s_bar = nu_tilde * f_v2(chi, c) / (c.kappa * d) ** 2
    clipped = omega + omega * (c.c_2**2 * omega + c.c_3 * s_bar) / (
        (c.c_3 - 2.0 * c.c_2) * omega - s_bar
    )
    return jnp.where(s_bar >= -c.c_2 * omega, omega + s_bar, clipped)


def f_w(r: Array, c: SAConstants = SA) -> Array:
    """Near-wall destruction function. ``f_w(1) = 1``: the log layer."""
    g = r + c.c_w2 * (r**6 - r)
    cw3_6 = c.c_w3**6
    return g * ((1.0 + cw3_6) / (g**6 + cw3_6)) ** (1.0 / 6.0)


def r_of(nu_tilde: Array, s_t: Array, d: Array, c: SAConstants = SA) -> Array:
    """``min(nu~ / (S~ kappa^2 d^2), r_max)``; ``r_max`` where ``S~ <= 0``,
    which the clipping prevents but a zero vorticity does not."""
    denominator = s_t * (c.kappa * d) ** 2
    safe = jnp.where(denominator > 0.0, denominator, 1.0)
    return jnp.where(denominator > 0.0, jnp.minimum(nu_tilde / safe, c.r_max), c.r_max)


def source(
    nu_tilde: Array, nu: Array, omega: Array, d: Array, c: SAConstants = SA
) -> tuple[Array, Array]:
    """``(P, D)`` per unit mass from ``nu~``, ``nu = mu / rho``, the vorticity
    magnitude and the wall distance; the source of ``rho nu~`` is ``rho (P - D)``."""
    positive = nu_tilde >= 0.0
    s_t = s_tilde(omega, nu_tilde, nu, d, c)
    wall_term = (nu_tilde / d) ** 2
    production = jnp.where(positive, c.c_b1 * s_t, c.c_b1 * omega) * nu_tilde
    destruction = jnp.where(
        positive, c.c_w1 * f_w(r_of(nu_tilde, s_t, d, c), c), -c.c_w1
    ) * wall_term
    return production, destruction


def f_n(nu_tilde: Array, nu: Array, c: SAConstants = SA) -> Array:
    """SA-neg diffusion factor: 1 for ``nu~ >= 0``, else
    ``(c_n1 + chi^3) / (c_n1 - chi^3)``, which keeps the diffusion
    coefficient ``nu + nu~ f_n`` positive."""
    chi3 = (nu_tilde / nu) ** 3
    return jnp.where(nu_tilde >= 0.0, 1.0, (c.c_n1 + chi3) / (c.c_n1 - chi3))


def diffusion_coefficient(
    rho: Array, nu_tilde: Array, mu: Array, c: SAConstants = SA
) -> Array:
    """``(mu + rho nu~ f_n) / sigma``: what multiplies ``grad nu~`` in the
    conservative diffusion term."""
    nu = mu / rho
    return (mu + rho * nu_tilde * f_n(nu_tilde, nu, c)) / c.sigma
