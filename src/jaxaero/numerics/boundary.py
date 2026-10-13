"""Boundary conditions: ghost-cell states, and the ghost fill of an atlas.

Conditions act on primitive variables ``(rho, V, p[, nu_hat])``, in 2D or 3D.
Each boundary kind is one pointwise function of the cell it reads (the cell as
far inside the face as the ghost is outside it, for walls and symmetry planes;
the cell on the face otherwise) and of the face's outward unit normal and
midpoint. Interfaces are not conditions: their ghosts are copies of the
partner cells, made by the atlas gather (mesh/atlas.py). Velocities are
global Cartesian, so a copy is right with no sign change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..core.state import RHO, Rows, dot, magnitude, split, stack_primitive
from ..mesh.atlas import Atlas


@dataclass(frozen=True)
class Face:
    """Outward unit normals and midpoints of the faces ghosts lie across, ``(d, n)`` each."""

    normal: Array
    midpoint: Array

    @property
    def dim(self) -> int:
        return self.normal.shape[0]


Reference = Callable[[Array], Array]
"""Face midpoints ``(d, n)`` -> the far-field reference primitives ``(nvar, n)``."""


@dataclass(frozen=True)
class Conditions:
    """What the conditions need besides the cells: the freestream, the gas
    and the far-field reference state."""

    q_inf: Array
    gamma: float
    reference: Reference


def mirror_velocity(velocity: Array, n: Array) -> Array:
    """``V - 2 (V . n) n``: the tangential component kept, the normal one
    reversed. Either sign of ``n`` gives the same result, bitwise."""
    return velocity - 2.0 * dot(velocity, n) * n


def _with_scalars(flow: Array, scalars: Array) -> Array:
    return jnp.concatenate([flow, scalars], axis=0)


# --- One function per boundary kind: (cell state (nvar, n), face, conditions) -> ghost ---


def slip_wall(q: Array, face: Face, c: Conditions) -> Array:
    """Mirror image of the cell: density and pressure copied (zero normal
    gradient), velocity mirrored (zero normal velocity on the face), scalars
    negated (zero on the wall)."""
    rho, velocity, p = split(q, face.dim)
    flow = stack_primitive(rho, mirror_velocity(velocity, face.normal), p)
    return _with_scalars(flow, -q[Rows(face.dim).n_flow:])


def no_slip_wall(q: Array, face: Face, c: Conditions) -> Array:
    """Adiabatic no-slip wall: velocity reversed, so the face velocity is
    exactly zero; density and pressure copied, so temperature too (zero heat
    flux); scalars negated (``nu~`` zero on the wall)."""
    rho, velocity, p = split(q, face.dim)
    return _with_scalars(stack_primitive(rho, -velocity, p), -q[Rows(face.dim).n_flow:])


def symmetry(q: Array, face: Face, c: Conditions) -> Array:
    """The slip-wall mirror with passive scalars copied, not negated."""
    rho, velocity, p = split(q, face.dim)
    flow = stack_primitive(rho, mirror_velocity(velocity, face.normal), p)
    return _with_scalars(flow, q[Rows(face.dim).n_flow:])


def characteristic_state(edge: Array, n: Array, q_inf: Array, gamma: float) -> Array:
    """Boundary state from the Riemann invariants normal to the face,
    ``(nvar, m)``; ``n (d, m)`` outward, ``q_inf`` one reference state or one per
    face.

        R+ = Vn_cell + 2 a_cell / (gamma - 1),   R- = Vn_ref - 2 a_ref / (gamma - 1)
        Vn = (R+ + R-) / 2,   a = (gamma - 1) (R+ - R-) / 4

    Entropy, tangential velocity ``V - Vn n`` and scalars from the cell where
    the flow leaves, from the reference where it enters.
    """
    dim = n.shape[0]
    n_flow = Rows(dim).n_flow
    if q_inf.ndim == 1:
        q_inf = q_inf[:, None]
    rho_i, v_i, p_i = split(edge, dim)
    rho_f, v_f, p_f = split(q_inf, dim)
    a_i = jnp.sqrt(gamma * p_i / rho_i)
    a_f = jnp.sqrt(gamma * p_f / rho_f)
    two_over = 2.0 / (gamma - 1.0)
    vn_i, vn_f = dot(v_i, n), dot(v_f, n)
    r_plus = vn_i + two_over * a_i
    r_minus = vn_f - two_over * a_f
    vn_b = 0.5 * (r_plus + r_minus)
    a_b = 0.25 * (gamma - 1.0) * (r_plus - r_minus)
    outflow = vn_b > 0.0
    entropy_b = jnp.where(outflow, p_i / rho_i**gamma, p_f / rho_f**gamma)
    tangential = jnp.where(outflow, v_i - vn_i * n, v_f - vn_f * n)
    rho_b = (a_b * a_b / (gamma * entropy_b)) ** (1.0 / (gamma - 1.0))
    p_b = rho_b * a_b * a_b / gamma
    scalar_b = jnp.where(outflow, edge[n_flow:], q_inf[n_flow:])
    return _with_scalars(stack_primitive(rho_b, vn_b * n + tangential, p_b), scalar_b)


def far_field(q: Array, face: Face, c: Conditions) -> Array:
    """``characteristic_state`` against the far-field reference at the face."""
    return characteristic_state(q, face.normal, c.reference(face.midpoint), c.gamma)


def total_condition_inflow(edge: Array, q_inf: Array, gamma: float, dim: int) -> Array:
    """Subsonic inflow at the freestream's total pressure and temperature, in
    its direction: pressure from the cell, ``M`` from ``p_t / p``, ``T`` from
    ``T_t``, scalars from the freestream."""
    g1 = gamma - 1.0
    n_flow = Rows(dim).n_flow
    rho_f, v_f, p_f = split(q_inf, dim)
    t_f = gamma * p_f / rho_f
    speed_f = magnitude(v_f)
    t_total = t_f + 0.5 * g1 * speed_f * speed_f
    p_total = p_f * (t_total / t_f) ** (gamma / g1)
    p_b = edge[Rows(dim).pressure]
    mach2 = jnp.maximum(2.0 / g1 * ((p_total / p_b) ** (g1 / gamma) - 1.0), 0.0)
    t_b = t_total / (1.0 + 0.5 * g1 * mach2)
    speed = jnp.sqrt(mach2 * t_b)
    flow = stack_primitive(gamma * p_b / t_b, speed * v_f[:, None] / speed_f, p_b)
    scalars = jnp.broadcast_to(q_inf[n_flow:, None], (q_inf.shape[0] - n_flow, p_b.shape[0]))
    return _with_scalars(flow, scalars)


def inflow_total(q: Array, face: Face, c: Conditions) -> Array:
    return total_condition_inflow(q, c.q_inf, c.gamma, face.dim)


def outflow_static(q: Array, face: Face, c: Conditions) -> Array:
    """Subsonic outflow at the freestream static pressure."""
    p = Rows(face.dim).pressure
    return q.at[p].set(c.q_inf[p])


CONDITIONS: dict[str, Callable[[Array, Face, Conditions], Array]] = {
    "slip_wall": slip_wall,
    "no_slip_wall": no_slip_wall,
    "symmetry": symmetry,
    "far_field": far_field,
    "inflow_total": inflow_total,
    "outflow_static": outflow_static,
}


# --- Far-field reference states --------------------------------------------------


FarFieldModel = Callable[[Array, Atlas, Array, float], Reference]
"""``(primitive canvas, atlas, q_inf, gamma) -> Reference``, once per residual."""


def uniform_far_field(q: Array, atlas: Atlas, q_inf: Array, gamma: float) -> Reference:
    return lambda midpoint: jnp.broadcast_to(q_inf[:, None], (q_inf.shape[0], midpoint.shape[1]))


def point_vortex_state(dx: Array, dy: Array, q_inf: Array, gamma: float, circulation: Array) -> Array:
    """Freestream plus the Prandtl-Glauert-scaled compressible vortex of
    circulation ``Gamma`` (clockwise positive), isentropic at the freestream
    total enthalpy, at ``(dx, dy)`` from its centre (2D):

        u = u_inf + Gamma beta sin(theta) / (2 pi r (1 - M^2 sin^2(theta - alpha)))
        v = v_inf - Gamma beta cos(theta) / (2 pi r (1 - M^2 sin^2(theta - alpha)))

    Thomas and Salas, AIAA J. 24 (1986) 1074-1080.
    """
    rho_f, (u_f, v_f), p_f = split(q_inf, 2)
    a2_f = gamma * p_f / rho_f
    speed2_f = u_f * u_f + v_f * v_f
    mach2 = speed2_f / a2_f
    alpha = jnp.arctan2(v_f, u_f)
    r = jnp.hypot(dx, dy)
    theta = jnp.arctan2(dy, dx)
    strength = circulation * jnp.sqrt(1.0 - mach2) / (
        2.0 * jnp.pi * r * (1.0 - mach2 * jnp.sin(theta - alpha) ** 2))
    u = u_f + strength * jnp.sin(theta)
    v = v_f - strength * jnp.cos(theta)
    ratio = 1.0 + 0.5 * (gamma - 1.0) * (speed2_f - u * u - v * v) / a2_f
    rho = rho_f * ratio ** (1.0 / (gamma - 1.0))
    p = p_f * ratio ** (gamma / (gamma - 1.0))
    n_flow = Rows(2).n_flow
    scalars = jnp.broadcast_to(q_inf[n_flow:, None], (q_inf.shape[0] - n_flow, dx.shape[0]))
    return _with_scalars(stack_primitive(rho, jnp.stack([u, v]), p), scalars)


def wall_vectors(atlas: Atlas) -> Array:
    """Area vector of every wall face into the fluid, ``(d, n)`` (``atlas.walls`` order)."""
    return atlas.walls.into_fluid * atlas.wall_values(atlas.grid.faces)


def wall_pressure_force(p_gauge: Array, atlas: Atlas) -> tuple[Array, ...]:
    """``F = -sum (p - p_inf) S`` over every wall face, ``S`` into the fluid,
    per component; ``p_gauge`` per wall face (``atlas.walls`` order)."""
    S = wall_vectors(atlas)
    return tuple(-jnp.sum(p_gauge * S[c]) for c in range(S.shape[0]))


def point_vortex_far_field(q: Array, atlas: Atlas, q_inf: Array, gamma: float) -> Reference:
    """The vortex of the current pressure lift of all walls (Kutta-Joukowski),
    at the quarter chord of the longest body (``Atlas.quarter_chord``); 2D."""
    pressure = Rows(2).pressure
    p = q[pressure].ravel()[atlas.walls.cell] - q_inf[pressure]
    fx, fy = wall_pressure_force(p, atlas)
    _, (u_f, v_f), _ = split(q_inf, 2)
    speed = jnp.hypot(u_f, v_f)
    circulation = (fy * u_f - fx * v_f) / speed / (q_inf[RHO] * speed)
    xc, yc = atlas.quarter_chord
    return lambda m: point_vortex_state(m[0] - xc, m[1] - yc, q_inf, gamma, circulation)


FAR_FIELDS: dict[str, FarFieldModel] = {
    "characteristic": uniform_far_field,
    "characteristic_point_vortex": point_vortex_far_field,
}


def far_field_models(dim: int) -> list[str]:
    """The far-field models of ``FAR_FIELDS`` defined in ``dim`` dimensions."""
    return list(FAR_FIELDS) if dim == 2 else ["characteristic"]


# --- The fill -----------------------------------------------------------------------


def fill_ghosts(q: Array, atlas: Atlas, conditions: Conditions) -> Array:
    """Every ghost cell of a primitive canvas ``(nvar, *canvas shape)``, by the
    atlas's gather stages; interior and gap cells unchanged."""
    flat = q.reshape(q.shape[0], -1)
    for stage in atlas.stages:
        parts = [flat]
        for kind, source, normal, midpoint in stage.conditions:
            face = Face(jnp.asarray(normal), jnp.asarray(midpoint))
            parts.append(CONDITIONS[kind](flat[:, source], face, conditions))
        flat = jnp.concatenate(parts, axis=1)[:, stage.index]
    return flat.reshape(q.shape)
