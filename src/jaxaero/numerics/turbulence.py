"""Discretization of the Spalart-Allmaras equation: the fifth residual row.

Works on the primitive state with one ghost layer, ``(d+3, *(shape+2))``,
whose last row is ``nu_hat = nu~ / nu_inf`` (core/state.py), and returns that
row's residual, scaled likewise. In finite-volume form, for cell ``c`` of
volume ``V`` (an area in 2D),

    R_c = sum_f [ mdot_f nu_hat_up  -  (mu + rho nu~ f_n)_f / sigma  grad(nu_hat)_f . S_f ]
          - V rho [ P - D + (c_b2 / sigma) |grad nu~|^2 ] / nu_inf

* **Convection**, first-order upwind on the mean flow's numerical mass flux
  ``mdot_f`` (row 0 of the Riemann flux), the standard choice for the
  turbulence variable: it keeps ``nu~`` bounded, and transport of ``rho nu~``
  stays conservative because the same mass flux carries the mean flow.
* **Diffusion**, through the face gradients the viscous flux already uses, with
  the coefficient at the face from face-averaged ``rho``, ``nu~`` and ``T``.
* **Source**, at the cell: ``P``, ``D`` from physics/spalart_allmaras.py with
  the vorticity and ``grad nu~`` from Green-Gauss over face-averaged values.

On the wall the ghost ``nu_hat`` is the negative of the cell's
(numerics/boundary.py), so the face value is exactly zero, as is the
face eddy viscosity built from it.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..core import constants
from ..core.state import RHO, Rows, dot
from ..mesh.block import Grid
from ..physics import navier_stokes
from ..physics import spalart_allmaras as sa
from ..physics.navier_stokes import Viscosity


def interior_index(p1: Array, dim: int, axis: int | None = None, offset: int = 1) -> tuple:
    """Index of ``p1 (..., *(shape+2))`` without its ghost layer, except along
    ``axis``, where it takes ``n + 1`` cells from ``offset`` (0: the cells
    before each face, 1: after)."""
    lead = (slice(None),) * (p1.ndim - dim)
    index = [slice(1, -1)] * dim
    if axis is not None:
        index[axis] = slice(offset, p1.shape[p1.ndim - dim + axis] - 1 + offset)
    return lead + tuple(index)


def face_averages(p1: Array, dim: int) -> tuple[Array, ...]:
    """Mean of the two cells either side of every face, per family:
    ``(..., *face shape)``, from ``p1`` with one ghost layer."""
    return tuple(0.5 * (p1[interior_index(p1, dim, a, 0)] + p1[interior_index(p1, dim, a, 1)])
                 for a in range(dim))


def divergence(fluxes: tuple[Array, ...]) -> Array:
    """``sum_families (F[next face] - F[this face])``: the net outflow of each
    cell, from per-family face fields (the last ``d`` axes)."""
    dim = len(fluxes)
    total = None
    for a, f in enumerate(fluxes):
        axis = f.ndim - dim + a
        lo, hi = [slice(None)] * f.ndim, [slice(None)] * f.ndim
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        diff = f[tuple(hi)] - f[tuple(lo)]
        total = diff if total is None else total + diff
    return total


def green_gauss(phi_faces: tuple[Array, ...], grid: Grid) -> Array:
    """Cell gradient ``(1/V) sum_f phi_f S_f`` from face values, ``(d, ..., *shape)``."""
    return jnp.stack([divergence(tuple(phi * s[c] for phi, s in zip(phi_faces, grid.faces))) / grid.volume
                      for c in range(grid.dim)])


def face_eddy_viscosity(
    p1: Array, viscosity: Viscosity, gamma: float, model: sa.SAModel, dim: int
) -> tuple[Array, ...]:
    """``mu_t`` on every face family from face-averaged ``rho``, ``nu~``, ``T``."""
    nu = Rows(dim).nu

    def at(face):
        T = navier_stokes.temperature(face, gamma, dim)
        return model.eddy_viscosity(face[RHO], face[nu], viscosity(T))

    return tuple(at(face) for face in face_averages(p1, dim))


def vorticity_magnitude(g: Array) -> Array:
    """``|curl V|`` (``|dv/dx - du/dy|`` in 2D) from ``g[b, a] = d v_a / d x_b``."""
    if g.shape[0] == 2:
        return jnp.abs(g[0, 1] - g[1, 0])
    curl = jnp.stack([g[1, 2] - g[2, 1], g[2, 0] - g[0, 2], g[0, 1] - g[1, 0]])
    # sqrt has an infinite derivative at 0, and a freestream start has zero
    # vorticity in most cells: inf * 0 = NaN in every JVP (ONERA M6, case 2308:
    # every Newton step rejected). Derivative 0 there, as jnp.abs's in 2D.
    square = dot(curl, curl)
    positive = square > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


def vorticity(p1: Array, grid: Grid) -> Array:
    """Vorticity magnitude per cell, by Green-Gauss over face-averaged velocities."""
    velocity = p1[Rows(grid.dim).velocity]
    return vorticity_magnitude(green_gauss(face_averages(velocity, grid.dim), grid))


def sa_residual(
    p1: Array,
    mass: tuple[Array, ...],
    grad_nu: tuple[Array, ...],
    grid: Grid,
    viscosity: Viscosity,
    gamma: float,
    model: sa.SAModel,
    distance: Array,
) -> Array:
    """The scaled SA residual row, ``shape``, from ``p1`` (one ghost layer),
    every face's mass flux and ``grad nu_hat`` ``(d, *face shape)``, and the
    wall distance."""
    c, nu_inf, dim = model.constants, model.nu_inf, grid.dim
    nu_row = Rows(dim).nu
    nu_hat = p1[nu_row]

    def face_flux(a, mass_f, face, grad, s):
        left = nu_hat[interior_index(nu_hat, dim, a, 0)]
        right = nu_hat[interior_index(nu_hat, dim, a, 1)]
        upwind = jnp.where(mass_f > 0.0, left, right)
        mu = viscosity(navier_stokes.temperature(face, gamma, dim))
        coefficient = sa.diffusion_coefficient(face[RHO], face[nu_row] * nu_inf, mu, c)
        return mass_f * upwind - coefficient * dot(grad, s)

    fluxes = tuple(face_flux(a, m, face, g, s) for a, (m, face, g, s)
                   in enumerate(zip(mass, face_averages(p1, dim), grad_nu, grid.faces)))

    cell = p1[interior_index(p1, dim)]
    rho = cell[RHO]
    mu = viscosity(navier_stokes.temperature(cell, gamma, dim))
    production, destruction = sa.source(
        cell[nu_row] * nu_inf, mu / rho, vorticity(p1, grid), distance, c
    )
    grad = green_gauss(face_averages(nu_hat, dim), grid)
    # Everything divided by nu_inf, the scale of the stored variable; the
    # gradient term carries nu_inf^2 / nu_inf.
    source = rho * ((production - destruction) / nu_inf
                    + c.c_b2 / c.sigma * nu_inf * dot(grad, grad))
    return divergence(fluxes) - source * grid.volume


def turbulent_conductivity(mu_t: Array, gamma: float) -> Array:
    """The eddy part of the heat-flux coefficient, ``mu_t / (Pr_t (gamma - 1))``."""
    return navier_stokes.conductivity(mu_t, gamma, constants.PRANDTL_TURBULENT)
