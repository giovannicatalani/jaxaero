"""The residual ``R(U)`` of the Euler, Navier-Stokes and SA-RANS equations,
in 2D and 3D.

For cell ``c`` of volume ``V_c`` (an area in 2D), with face area vectors ``S``
(mesh/block.py), the semi-discrete system is ``V_c dU_c/dt = -R_c`` with

    R_c = sum_faces (F_num - F_v) . S  -  V_c S_c

the last term the SA source (numerics/turbulence.py). Everything runs on an
atlas canvas (mesh/atlas.py): ghosts filled, the single-block assembly applied
once, the result kept on interior cells. The dimension is the grid's.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..core import constants
from ..core.config import FlowConditions, NumericsConfig
from ..core.state import (
    RHO, Rows, conservative_to_primitive, dot, freestream_primitive, magnitude,
    primitive_to_conservative, split, stack_primitive,
)
from ..mesh.atlas import Atlas, build_atlas
from ..mesh.block import Grid
from ..mesh.mesh import wall_distance
from ..physics import euler, navier_stokes
from ..physics.navier_stokes import Viscosity
from ..physics.spalart_allmaras import SAModel
from . import boundary as bc
from . import gradients, reconstruction, turbulence
from .limiters import get_limiter
from .riemann import get_flux


def primitive_from_conservative(U: Array, gamma: float, dim: int) -> Array:
    """``(rho, V, p[, nu_hat])`` from ``(rho, rho V, E[, rho nu_hat])``."""
    q = stack_primitive(*conservative_to_primitive(U, gamma, dim))
    if U.shape[0] == Rows(dim).n_flow:
        return q
    return jnp.concatenate([q, U[Rows(dim).nu:] / U[RHO]], axis=0)


def block_residual(
    padded: Array, grid: Grid, centres: Array | None, distance: Array | None, *,
    flux: Callable, limiter: Callable, gamma: float, n_ghost: int, first_order: bool,
    viscosity: Viscosity | None, model: SAModel | None, kappa: float = 0.0,
) -> Array:
    """Net flux of every cell of ``grid`` from its ghosted primitive state
    ``(nvar, *(shape + 2g))``; ``centres`` (cell centres with one ghost
    layer, ``(d, *(shape + 2))``) are needed when viscous."""
    dim, shape = grid.dim, grid.shape
    rows = Rows(dim)
    flow_rows = padded[:rows.n_flow]

    def reconstruct(axis: int) -> tuple[Array, Array]:
        if first_order:
            return reconstruction.first_order_faces(flow_rows, n_ghost, shape, axis)
        return reconstruction.face_states(flow_rows, n_ghost, shape, limiter, axis, kappa)

    fluxes = [flux(*reconstruct(a), s, gamma) for a, s in enumerate(grid.faces)]   # (d+2, *face shape)
    sa_row = None
    if viscosity is not None:
        g = n_ghost - 1
        p1 = padded[(slice(None),) + (slice(g, -g),) * dim]
        if model is None:
            visc = viscous_fluxes(p1, grid, viscosity, gamma, centres=centres)
        else:
            phi = jnp.concatenate([p1[rows.velocity], navier_stokes.temperature(p1, gamma, dim)[None],
                                   p1[rows.nu][None]])
            grads = gradients.face_gradients(phi, centres)
            mu_t = turbulence.face_eddy_viscosity(p1, viscosity, gamma, model, dim)
            visc = viscous_fluxes(p1, grid, viscosity, gamma, mu_t, grads)
            sa_row = turbulence.sa_residual(
                p1, tuple(f[0] for f in fluxes), tuple(G[:, dim + 1] for G in grads),
                grid, viscosity, gamma, model, distance,
            )
        fluxes = [f - v for f, v in zip(fluxes, visc)]

    flow = turbulence.divergence(tuple(fluxes))
    return flow if sa_row is None else jnp.concatenate([flow, sa_row[None]], axis=0)


def viscous_fluxes(
    q: Array, grid: Grid, viscosity: Viscosity, gamma: float,
    mu_t: tuple[Array, ...] | None = None, grads=None, centres: Array | None = None,
) -> tuple[Array, ...]:
    """``F_v . S`` on every face family from primitives with one ghost layer,
    ``(d+2, *(shape+2))``; ``mu_t`` adds the eddy viscosity (and, over ``Pr_t``,
    conductivity). Face velocity and temperature are two-cell means, so a
    no-slip wall's face velocity is exactly zero, and its heat flux zero to
    round-off (the ghost centre is the reflected cell centre)."""
    dim = grid.dim
    rows = Rows(dim)
    if grads is None:
        phi = jnp.concatenate([q[rows.velocity], navier_stokes.temperature(q, gamma, dim)[None]])
        grads = gradients.face_gradients(phi, centres)
    eddies = (None,) * dim if mu_t is None else mu_t

    def flux(a, grad, s, eddy):
        left = q[turbulence.interior_index(q, dim, a, 0)]
        right = q[turbulence.interior_index(q, dim, a, 1)]
        velocity = 0.5 * (left[rows.velocity] + right[rows.velocity])
        T = 0.5 * (navier_stokes.temperature(left, gamma, dim) + navier_stokes.temperature(right, gamma, dim))
        mu = viscosity(T)
        k = navier_stokes.conductivity(mu, gamma, constants.PRANDTL_LAMINAR)
        if eddy is not None:
            mu, k = mu + eddy, k + turbulence.turbulent_conductivity(eddy, gamma)
        grad_v = jnp.swapaxes(grad[:, :dim], 0, 1)          # [component, direction]
        return navier_stokes.viscous_flux(velocity, grad_v, grad[:, dim], s, mu, k)

    return tuple(flux(a, grad, s, eddy) for a, (grad, s, eddy) in enumerate(zip(grads, grid.faces, eddies)))


def spectral_radius(q: Array, s: Array, gamma: float) -> Array:
    """``|V . S| + a |S|``: the one wave speed of the time step, the
    pseudo-time term and the preconditioner."""
    rho, velocity, p = split(q, s.shape[0])
    return jnp.abs(dot(velocity, s)) + euler.sound_speed(rho, p, gamma) * magnitude(s)


def lower_upper(f: Array, axis: int) -> tuple[Array, Array]:
    """A face field's values on each cell's lower and upper face along ``axis``
    (counted from the end: the face fields' leading axes vary)."""
    lo, hi = [slice(None)] * f.ndim, [slice(None)] * f.ndim
    lo[axis], hi[axis] = slice(0, -1), slice(1, None)
    return f[tuple(lo)], f[tuple(hi)]


def local_time_step(U: Array, grid: Grid, gamma: float, cfl: float,
                    viscosity: Viscosity | None = None, model: SAModel | None = None) -> Array:
    """``dt = CFL V / sum_faces lambda_f`` per cell, the faces' spectral radii
    at the cell state, plus the diffusive limit when viscous (Blazek, 3rd ed.,
    eq. 6.22: ``(gamma/Pr)(mu/rho) sum_a S_a^2 / V``, x4, ``S_a`` the mean area
    of the two faces normal to ``a``)."""
    dim = grid.dim
    q = primitive_from_conservative(U, gamma, dim)
    total = None
    for a, s in enumerate(grid.faces):
        lo, hi = lower_upper(s, a - dim)
        lo, hi = spectral_radius(q, lo, gamma), spectral_radius(q, hi, gamma)
        total = lo + hi if total is None else total + lo + hi
    if viscosity is not None:
        mean_area = [0.5 * (magnitude(lo) + magnitude(hi)) for lo, hi in
                     (lower_upper(s, a - dim) for a, s in enumerate(grid.faces))]
        squares = mean_area[0] ** 2
        for m in mean_area[1:]:
            squares = squares + m ** 2
        mu = viscosity(navier_stokes.temperature(q, gamma, dim))
        if model is not None:
            mu = mu + model.eddy_viscosity(q[RHO], q[Rows(dim).nu], mu)
        factor = gamma / constants.PRANDTL_LAMINAR * mu / q[RHO]
        total = total + 4.0 * factor * squares / grid.volume
    return cfl * grid.volume / jnp.maximum(total, 1e-300)


def face_wave_speeds(U: Array, atlas: Atlas, gamma: float,
                     viscosity: Viscosity | None = None, model: SAModel | None = None
                     ) -> tuple[Array, ...]:
    """``lambda`` on every canvas face, per family, from the face-averaged
    state: the dissipation scale of a first-order flux, which the implicit
    path's pseudo-time term and preconditioner use. Viscous: plus the
    thin-layer ``2 (gamma/Pr) mu |S|^2 / (rho V)``, ``V`` the two cells' mean
    volume, with the eddy viscosity in RANS."""
    grid = atlas.grid
    dim = grid.dim
    q = primitive_from_conservative(U, gamma, dim)
    q_faces = atlas.face_averages(q)
    lambdas = tuple(spectral_radius(qf, s, gamma) for qf, s in zip(q_faces, grid.faces))
    if viscosity is None:
        return lambdas
    volumes = atlas.face_averages(grid.volume)
    factor = 2.0 * gamma / constants.PRANDTL_LAMINAR
    out = []
    for lam, qf, s, vol in zip(lambdas, q_faces, grid.faces, volumes):
        mu = viscosity(navier_stokes.temperature(qf, gamma, dim))
        if model is not None:
            mu = mu + model.eddy_viscosity(qf[RHO], qf[Rows(dim).nu], mu)
        out.append(lam + factor * mu * dot(s, s) / (qf[RHO] * vol))
    return tuple(out)


def cell_wave_speed_sum(lambdas: tuple[Array, ...]) -> Array:
    """``sum_faces lambda_f`` per cell: ``V/dt = sum / CFL``, and the
    preconditioner diagonal."""
    dim = len(lambdas)
    total = None
    for a, lam in enumerate(lambdas):
        lo, hi = lower_upper(lam, a - dim)
        total = lo + hi if total is None else total + lo + hi
    return total


@dataclass(frozen=True)
class ResidualOperator:
    """A configured residual on an atlas canvas, and what solvers need with it.

    ``residual``, ``time_step`` and ``norm`` take canvas fields
    ``(nvar, *canvas shape)``; ``grid`` is the canvas geometry. Canvas cells that are
    no block's keep their initial value (their residual is ``U - U_0``).
    """

    residual: Callable[[Array], Array]
    time_step: Callable[[Array], Array]
    atlas: Atlas
    q_inf: Array
    gamma: float
    viscosity: Viscosity | None = None
    model: SAModel | None = None
    distance: Array | None = None

    @property
    def grid(self) -> Grid:
        return self.atlas.grid

    @property
    def dim(self) -> int:
        return self.grid.dim

    def norm(self, residual: Array, variable: int = RHO) -> Array:
        """Volume-weighted L2 norm of one component over the block cells."""
        weight = jnp.where(self.atlas.interior, self.grid.volume, 0.0)
        return jnp.sqrt(jnp.sum(weight * residual[variable] ** 2) / jnp.sum(weight))

    def wave_speeds(self, U: Array):
        return face_wave_speeds(U, self.atlas, self.gamma, self.viscosity, self.model)

    def initial_state(self, dtype=jnp.float64) -> Array:
        """The freestream on every canvas cell."""
        return uniform_canvas(self.q_inf, self.grid, self.gamma, dtype)


def uniform_canvas(q: Array, grid: Grid, gamma: float, dtype=jnp.float64) -> Array:
    """The primitive state ``q (nvar,)`` on every cell, conservative."""
    rows, ones = Rows(grid.dim), jnp.ones(grid.shape, dtype=dtype)
    cells = (slice(None),) + (None,) * grid.dim
    rho, velocity, p = split(q, grid.dim)
    U = primitive_to_conservative(rho * ones, velocity[cells] * ones, p * ones, gamma)
    if q.shape[0] > rows.n_flow:
        U = jnp.concatenate([U, (q[RHO] * q[rows.n_flow:][cells]) * ones], axis=0)
    return U.astype(dtype)


def make_residual(
    grid: Atlas | Grid | tuple[Grid, ...],
    flow: FlowConditions,
    numerics: NumericsConfig,
    first_order: bool = False,
    freestream: Array | None = None,
    distance: Array | None = None,
) -> ResidualOperator:
    """The residual of one case. A ``Grid`` or a mesh is packed into an atlas.

    ``freestream`` replaces ``flow``'s primitives ``(d+2,)`` and may be traced (a
    polar maps the step over it); ``distance`` is a precomputed wall-distance
    canvas (with SA). Registries are resolved here, before tracing.
    """
    atlas = grid if isinstance(grid, Atlas) else build_atlas(grid, numerics.n_ghost)
    if atlas.g != numerics.n_ghost:
        raise ValueError(f"the atlas has {atlas.g} ghost layers, the scheme needs {numerics.n_ghost}")
    viscosity = Viscosity.of(flow)
    model = SAModel.of(flow)
    dtype, gamma, g, dim = numerics.dtype, flow.gamma, numerics.n_ghost, atlas.grid.dim
    if numerics.far_field not in bc.far_field_models(dim):
        raise ValueError(f"far field model {numerics.far_field!r} is not available in {dim}D; "
                         f"available: {bc.far_field_models(dim)}")
    q_inf = jnp.asarray(freestream_primitive(flow, dim) if freestream is None else freestream, dtype=dtype)
    if model is not None:
        q_inf = jnp.concatenate([q_inf, jnp.asarray([model.freestream_ratio], dtype=dtype)])
        if distance is None:
            distance = atlas.pack([jnp.asarray(d, dtype=dtype) for d in wall_distance(atlas.mesh)], 1.0)
    else:
        distance = None
    canvas = atlas.grid
    far_field = bc.FAR_FIELDS[numerics.far_field]
    flux, limiter = get_flux(numerics.flux), get_limiter(numerics.limiter)
    U_0 = uniform_canvas(q_inf, canvas, gamma, dtype)

    def residual(U: Array) -> Array:
        q = primitive_from_conservative(U, gamma, dim)
        conditions = bc.Conditions(q_inf, gamma, far_field(q, atlas, q_inf, gamma))
        padded = jnp.pad(bc.fill_ghosts(q, atlas, conditions), ((0, 0),) + ((g, g),) * dim, mode="edge")
        R = block_residual(padded, canvas, atlas.centres, distance, flux=flux, limiter=limiter,
                           gamma=gamma, n_ghost=g, first_order=first_order,
                           viscosity=viscosity, model=model, kappa=numerics.kappa)
        return jnp.where(atlas.interior, R, U - U_0)

    return ResidualOperator(
        residual=residual,
        time_step=partial(local_time_step, grid=canvas, gamma=gamma, cfl=numerics.cfl,
                          viscosity=viscosity, model=model),
        atlas=atlas, q_inf=q_inf, gamma=gamma, viscosity=viscosity, model=model, distance=distance,
    )
