"""State container, variable indexing and thermodynamic conversions.

Variable indexing is defined here once and nowhere else, for any dimension
``d`` (2 or 3). All state arrays carry the variable on axis 0:

    U: (d + 2, ...)   conservative (rho, rho V, E),   V = (u, v[, w])
    Q: (d + 2, ...)   primitive (rho, V, p)

``Rows(d)`` names the rows: density 0, velocity (momentum) ``1 .. d``,
pressure (energy) ``d + 1``. With the Spalart-Allmaras model one more row is
appended, ``Rows.nu = d + 2``: ``rho nu~ / nu_inf`` conservative,
``nu~ / nu_inf`` primitive. The scaling by ``nu_inf = M / Re`` puts it at O(1)
in the far field (3) and O(1e3) in a boundary layer instead of O(1e-8): one
GMRES norm spans all equations, and unscaled the turbulence equation would
vanish from it. Functions here read the flow rows only, so they accept either
layout.

All quantities are nondimensional (see constants.py). Every function here is
pure and pointwise: it takes arrays and returns arrays, knows nothing about the
grid, and performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from . import constants
from .config import FlowConditions

RHO = 0
"""Density: row 0 of the conservative and the primitive state, in any dimension."""


@dataclass(frozen=True)
class Rows:
    """The rows of a ``dim``-dimensional state (conservative and primitive alike)."""

    dim: int

    @property
    def velocity(self) -> slice:
        """``rho V`` conservative, ``V`` primitive: ``(dim, ...)``."""
        return slice(1, 1 + self.dim)

    @property
    def energy(self) -> int:
        """``E`` conservative, ``p`` primitive."""
        return self.dim + 1

    pressure = energy

    @property
    def n_flow(self) -> int:
        """Rows of the mean flow: ``dim + 2``."""
        return self.dim + 2

    @property
    def nu(self) -> int:
        """The Spalart-Allmaras row, after the mean flow."""
        return self.dim + 2


def dot(a: Array, b: Array) -> Array:
    """``sum_k a_k b_k`` over axis 0, summed in index order (so the 2D result
    is bitwise ``a_x b_x + a_y b_y``, as written out by hand)."""
    total = a[0] * b[0]
    for k in range(1, a.shape[0]):
        total = total + a[k] * b[k]
    return total


def magnitude(a: Array) -> Array:
    """``|a|`` over axis 0 (``hypot`` in 2D)."""
    return jnp.hypot(a[0], a[1]) if a.shape[0] == 2 else jnp.sqrt(dot(a, a))


def split(q: Array, dim: int) -> tuple[Array, Array, Array]:
    """``(rho, V (dim, ...), p)`` of a primitive state (or ``(rho, rho V, E)``
    of a conservative one)."""
    rows = Rows(dim)
    return q[RHO], q[rows.velocity], q[rows.energy]


def conservative_to_primitive(U: Array, gamma: float, dim: int) -> tuple[Array, Array, Array]:
    """``(rho, V (dim, ...), p)`` from ``U (dim + 2 or more, ...)``."""
    rho, momentum, energy = split(U, dim)
    velocity = momentum / rho
    p = (gamma - 1.0) * (energy - 0.5 * rho * dot(velocity, velocity))
    return rho, velocity, p


def primitive_to_conservative(rho: Array, velocity: Array, p: Array, gamma: float) -> Array:
    """``U (dim + 2, ...)`` from primitive fields of one shape, ``velocity (dim, ...)``."""
    energy = p / (gamma - 1.0) + 0.5 * rho * dot(velocity, velocity)
    return jnp.concatenate([rho[None], rho * velocity, energy[None]], axis=0)


def stack_primitive(rho: Array, velocity: Array, p: Array) -> Array:
    """``Q (dim + 2, ...)`` from its parts."""
    return jnp.concatenate([rho[None], velocity, p[None]], axis=0)


def freestream_primitive(flow: FlowConditions, dim: int = 2) -> tuple[float, ...]:
    """Nondimensional freestream primitives ``(rho, V, p)``.

    With the rho_inf / a_inf / chord scaling of constants.py: rho = 1,
    p = 1/gamma, and the velocity at incidence alpha in the plane of the
    first and the last axis (x-y in 2D, x-z in 3D, y the span):
    ``M (cos alpha, sin alpha)`` or ``M (cos alpha, 0, sin alpha)``.
    """
    alpha = flow.alpha_rad
    u, normal = float(flow.mach * np.cos(alpha)), float(flow.mach * np.sin(alpha))
    velocity = (u, normal) if dim == 2 else (u, 0.0, normal)
    return (1.0, *velocity, 1.0 / flow.gamma)


def is_physical(U: Array, gamma: float, dim: int) -> Array:
    """Boolean mask, True where density and pressure are both above their floors."""
    rho, _, p = conservative_to_primitive(U, gamma, dim)
    return (
        (rho > constants.MIN_DENSITY)
        & (p > constants.MIN_PRESSURE)
        & jnp.isfinite(rho)
        & jnp.isfinite(p)
    )


def first_bad_cell(U: Array, gamma: float, dim: int) -> tuple[bool, tuple[int, ...]]:
    """``(all good, index)`` of the first non-physical cell, all ``-1`` if none.
    A host transfer: for error reports between blocks only."""
    good = np.asarray(is_physical(U, gamma, dim))
    if good.all():
        return True, (-1,) * good.ndim
    return False, tuple(int(k[0]) for k in np.nonzero(~good))
