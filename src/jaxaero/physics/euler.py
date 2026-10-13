"""Perfect-gas thermodynamics and analytic Euler fluxes, in any dimension.

Pure, pointwise functions of arrays: no grid, no I/O, no global state. The same
functions serve 2D and 3D, structured and unstructured discretizations, and are
the reference the numerical fluxes are tested against for consistency.

All quantities are nondimensional (core/constants.py). Conservative state is
``U = (rho, rho V, E)``; primitive is ``Q = (rho, V, p)``, ``V`` the velocity
vector of ``d`` components (core/state.py). A normal or area vector is
``(d, ...)``; the dimension is its length.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..core.state import (
    conservative_to_primitive,
    dot,
    primitive_to_conservative,
    split,
    stack_primitive,
)


def sound_speed(rho: Array, p: Array, gamma: float) -> Array:
    """``a = sqrt(gamma p / rho)``."""
    return jnp.sqrt(gamma * p / rho)


def total_enthalpy(rho: Array, velocity: Array, p: Array, gamma: float) -> Array:
    """``H = (E + p)/rho = gamma/(gamma-1) p/rho + |V|^2/2``."""
    return gamma / (gamma - 1.0) * p / rho + 0.5 * dot(velocity, velocity)


def total_energy(rho: Array, velocity: Array, p: Array, gamma: float) -> Array:
    """``E = p/(gamma-1) + rho |V|^2 / 2``."""
    return p / (gamma - 1.0) + 0.5 * rho * dot(velocity, velocity)


def normal_flux(q: Array, n: Array, gamma: float) -> Array:
    """Euler flux through a normal ``n (d, ...)``, ``(d + 2, ...)`` from ``q = (rho, V, p)``:

        F . n = (rho Vn, rho V Vn + p n, (E + p) Vn),   Vn = V . n"""
    rho, velocity, p = split(q, n.shape[0])
    vn = dot(velocity, n)
    energy = total_energy(rho, velocity, p, gamma)
    return jnp.concatenate(
        [(rho * vn)[None], rho * velocity * vn + p * n, ((energy + p) * vn)[None]], axis=0
    )


def normal_flux_jacobian(q: Array, s: Array, gamma: float) -> Array:
    """``A = d(F . S)/dU``, ``(d+2, d+2, ...)`` indexed ``[flux row, variable, ...]``:
    eigenvalues ``{Vn (d times), Vn + a, Vn - a}`` times the face area
    (Blazek, 3rd ed., eq. A.9). JVPs of ``normal_flux``, so the flux has one
    definition."""
    dim = s.shape[0]
    rho, velocity, p = split(q, dim)
    U = primitive_to_conservative(rho, velocity, p, gamma)

    def flux_of_conservative(u_vec: Array) -> Array:
        return normal_flux(stack_primitive(*conservative_to_primitive(u_vec, gamma, dim)), s, gamma)

    columns = []
    for k in range(dim + 2):
        tangent = jnp.zeros_like(U).at[k].set(1.0)
        columns.append(jax.jvp(flux_of_conservative, (U,), (tangent,))[1])
    return jnp.stack(columns, axis=1)


def roe_average(
    qL: Array, qR: Array, gamma: float, dim: int
) -> tuple[Array, Array, Array, Array]:
    """Roe-averaged ``(rho, V (d, ...), H, a)``: the weights for which
    ``A(qL, qR)(UR - UL) = FR - FL`` exactly (Roe, J. Comput. Phys. 43 (1981))."""
    rho_l, v_l, p_l = split(qL, dim)
    rho_r, v_r, p_r = split(qR, dim)

    h_l = total_enthalpy(rho_l, v_l, p_l, gamma)
    h_r = total_enthalpy(rho_r, v_r, p_r, gamma)

    weight = jnp.sqrt(rho_r / rho_l)
    denom = 1.0 + weight

    rho = jnp.sqrt(rho_l * rho_r)
    velocity = (v_l + weight * v_r) / denom
    enthalpy = (h_l + weight * h_r) / denom
    # a^2 = (gamma-1)(H - q^2/2), guarded so a round-off-negative argument near a
    # vacuum state cannot produce a NaN that would propagate silently.
    a_squared = (gamma - 1.0) * (enthalpy - 0.5 * dot(velocity, velocity))
    return rho, velocity, enthalpy, jnp.sqrt(jnp.maximum(a_squared, 1e-30))
