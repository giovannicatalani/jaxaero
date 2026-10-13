"""Laminar viscous flux of the compressible Navier-Stokes equations.

Pure and pointwise, like ``physics/euler.py``: a face state and the velocity and
temperature gradients at the face go in, the flux comes out. With the face area
vector ``S`` (it carries the face length),

    F_v . S = ( 0,
                tau_xx Sx + tau_xy Sy,
                tau_xy Sx + tau_yy Sy,
                (tau . V) . S + k grad(T) . S )

    tau = mu (grad V + grad V^T - 2/3 (div V) I)       Stokes' hypothesis

and the residual is ``R = sum_faces (F_inv - F_v) . S``.

Nondimensionalization (core/constants.py): with ``rho_inf``, ``a_inf`` and the
chord as references, ``mu = M / Re``, and the temperature that appears in the
heat flux is ``T = a^2 = gamma p / rho`` (so ``T_inf = 1``). Fourier's law
``q = -k_dim grad T_dim`` with ``k_dim = mu c_p / Pr`` becomes, because
``c_p T_dim / a_inf^2 = a^2 / (gamma - 1)``,

    q = -k grad(T),      k = mu / (Pr (gamma - 1))

The viscosity is a law of the temperature (``Viscosity``): constant, or
Sutherland's,

    mu(T) / mu_inf = T^(3/2) (1 + s) / (T + s),      s = S / T_inf,

in these units, where ``T_inf = 1`` (White, *Viscous Fluid Flow*, 1974, p. 28;
the form TMR gives "for nondimensional codes"). Only the ratio ``s`` enters,
which is why a case states two temperatures in one unit, not a unit system.

Reference: Blazek, "Computational Fluid Dynamics: Principles and Applications",
3rd ed., section 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ..core.config import FlowConditions
from ..core.state import Rows, dot


@dataclass(frozen=True)
class Viscosity:
    """``mu(T)``: constant (``sutherland_ratio=None``; then the Python float itself,
    bitwise a scalar ``mu``) or Sutherland's; ``mu_inf = M / Re``."""

    mu_inf: float
    sutherland_ratio: float | None = None

    @classmethod
    def of(cls, flow: FlowConditions) -> "Viscosity | None":
        """The law for a flow; ``None`` when inviscid."""
        if flow.inviscid:
            return None
        return cls(flow.mu_inf, flow.sutherland_ratio)

    def __call__(self, T: Array) -> Array | float:
        """``mu`` at nondimensional temperature ``T`` (freestream: ``T = 1``)."""
        s = self.sutherland_ratio
        if s is None:
            return self.mu_inf
        return self.mu_inf * T * jnp.sqrt(T) * (1.0 + s) / (T + s)


def temperature(q: Array, gamma: float, dim: int) -> Array:
    """Nondimensional temperature ``T = a^2 = gamma p / rho``."""
    return gamma * q[Rows(dim).pressure] / q[0]


def conductivity(mu: Array | float, gamma: float, prandtl: float) -> Array | float:
    """Nondimensional conductivity ``k = mu / (Pr (gamma - 1))``.

    With an eddy viscosity the two parts add with their own Prandtl numbers,
    ``conductivity(mu, ., Pr) + conductivity(mu_t, ., Pr_t)``.
    """
    return mu / (prandtl * (gamma - 1.0))


def viscous_flux(
    velocity: Array,
    grad_v: Array,
    grad_t: Array,
    s: Array,
    mu: Array | float,
    k: Array | float,
) -> Array:
    """``F_v . S``, ``(d + 2, ...)``, from the face velocity ``(d, ...)``, its
    gradient ``grad_v[a, b] = d v_a / d x_b`` ``(d, d, ...)``, the temperature
    gradient ``(d, ...)``, the area vector ``s (d, ...)``, and effective ``mu``
    and ``k``."""
    dim = s.shape[0]
    div = grad_v[0, 0]
    for a in range(1, dim):
        div = div + grad_v[a, a]
    two_thirds_div = (2.0 / 3.0) * div

    def tau(a: int, b: int) -> Array:
        if a == b:
            return mu * (2.0 * grad_v[a, a] - two_thirds_div)
        lo, hi = min(a, b), max(a, b)
        return mu * (grad_v[lo, hi] + grad_v[hi, lo])

    force = jnp.stack([dot(jnp.stack([tau(a, b) for b in range(dim)]), s) for a in range(dim)])
    energy = dot(velocity, force) + k * dot(grad_t, s)
    return jnp.concatenate([jnp.zeros_like(energy)[None], force, energy[None]], axis=0)
