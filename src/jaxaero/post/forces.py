"""Wall pressure and friction, and the integrated force coefficients.

Nondimensional as everywhere (core/constants.py): ``p_inf = 1/gamma`` and
``q_inf = M^2 / 2``. Pressure comes from the cell on each wall face, friction
from the viscous flux the residual uses there; the force on the body is

    F = sum_faces [ -(p - p_inf) S + tau . S ],     S into the fluid,

gauge pressure, so an open wall (a plate wetted on one side) is right too.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.config import FlowConditions
from ..core.state import RHO, Rows, conservative_to_primitive, freestream_primitive
from ..mesh.atlas import Atlas
from ..numerics import boundary as bc
from ..numerics.residual import primitive_from_conservative, viscous_fluxes
from ..physics.navier_stokes import Viscosity


def dynamic_pressure(flow: FlowConditions) -> float:
    return 0.5 * flow.mach**2


def wall_loads(U: Array, atlas: Atlas, flow: FlowConditions) -> tuple[Array, Array, Array]:
    """Per wall face: gauge pressure ``(n,)``, the area vector into the fluid
    ``(d, n)``, and the viscous force on the body ``(d, n)`` (zero if inviscid)."""
    gamma, dim = flow.gamma, atlas.dim
    rows = Rows(dim)
    q = primitive_from_conservative(U[:rows.n_flow], gamma, dim)
    p = q[rows.pressure].ravel()[atlas.walls.cell] - 1.0 / gamma
    S = bc.wall_vectors(atlas)
    shear = jnp.zeros((dim,) + p.shape, dtype=U.dtype)
    if not flow.inviscid:
        q_inf = jnp.asarray(freestream_primitive(flow, dim), dtype=U.dtype)
        conditions = bc.Conditions(q_inf, gamma, bc.uniform_far_field(q, atlas, q_inf, gamma))
        filled = jnp.pad(bc.fill_ghosts(q, atlas, conditions), ((0, 0),) + ((1, 1),) * dim, mode="edge")
        visc = viscous_fluxes(filled, atlas.grid, Viscosity.of(flow), gamma, centres=atlas.centres)
        shear = atlas.walls.into_fluid * atlas.wall_values(tuple(v[rows.velocity] for v in visc))
    return p, S, shear


@dataclass(frozen=True)
class Forces:
    """Lift, drag (pressure + friction) and pitching moment coefficients; the
    moment about ``(0.25, 0)``, nose-up positive, pressure only. ``bodies``:
    ``(cl, cd, cm)`` of each body when there are several (numbered by
    leading-edge ``x``: slat, main, flap)."""

    cl: float
    cd: float
    cm: float
    fx: float
    fy: float
    cd_friction: float = 0.0
    bodies: tuple[tuple[float, float, float], ...] = ()
    fz: float = 0.0
    cy: float = 0.0

    def summary(self) -> str:
        friction = f" (friction {self.cd_friction:+.6f})" if self.cd_friction else ""
        side = f"   CY = {self.cy:+.6f}" if self.cy else ""
        text = f"CL = {self.cl:+.6f}   CD = {self.cd:+.6f}{friction}   CM = {self.cm:+.6f}{side}"
        for k, (cl, cd, cm) in enumerate(self.bodies):
            text += f"\n  body {k}: CL = {cl:+.6f}   CD = {cd:+.6f}   CM = {cm:+.6f}"
        return text


@dataclass(frozen=True)
class ForceReference:
    """What the coefficients are per: ``length`` (chord), ``area`` (3D; in 2D
    the coefficients are per unit span, the area is the length) and the
    ``centre`` the pitching moment is about."""

    length: float = 1.0
    area: float | None = None
    centre: tuple[float, ...] = (0.25, 0.0, 0.0)


def integrate_forces(U: Array, atlas: Atlas, flow: FlowConditions,
                     reference: ForceReference | float = ForceReference()) -> Forces:
    """Force coefficients of every wall, and of each body; ``reference`` (or
    the reference length alone)."""
    if not isinstance(reference, ForceReference):
        reference = ForceReference(length=float(reference))
    if atlas.dim == 3:
        return _integrate_forces_3d(U, atlas, flow, reference)
    reference_length = reference.length
    xc, yc = reference.centre[:2]
    p, (sx, sy), shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    faces = atlas.walls.faces
    x, y = faces.mid[:, 0] - xc, faces.mid[:, 1] - yc
    cos_a, sin_a = np.cos(flow.alpha_rad), np.sin(flow.alpha_rad)
    q_inf = dynamic_pressure(flow) * reference_length

    def coefficients(mask):
        fx, fy = -np.sum(p[mask] * sx[mask]), -np.sum(p[mask] * sy[mask])
        tx, ty = shear[0][mask].sum(), shear[1][mask].sum()
        moment = -np.sum(x[mask] * (-p[mask] * sy[mask]) - y[mask] * (-p[mask] * sx[mask]))
        fx, fy = fx + tx, fy + ty
        return ((-fx * sin_a + fy * cos_a) / q_inf, (fx * cos_a + fy * sin_a) / q_inf,
                moment / (q_inf * reference_length), fx, fy, (tx * cos_a + ty * sin_a) / q_inf)

    cl, cd, cm, fx, fy, friction = coefficients(np.ones(len(p), dtype=bool))
    bodies = tuple(tuple(float(c) for c in coefficients(faces.body == k)[:3])
                   for k in range(faces.bodies)) if faces.bodies > 1 else ()
    return Forces(float(cl), float(cd), float(cm), float(fx), float(fy), float(friction), bodies)


def _integrate_forces_3d(U: Array, atlas: Atlas, flow: FlowConditions, reference: ForceReference) -> Forces:
    """3D: lift along ``(-sin a, 0, cos a)``, drag along ``(cos a, 0, sin a)``
    (incidence in x-z, core/state.py), side force along ``y``, per
    ``q_inf * area``; the pitching moment ``sum (r x f)_y`` about the ``y``
    axis through the centre (nose-up positive), pressure and friction (as
    TMR's CMY; 2D is pressure only), per ``q_inf * area * length``."""
    if reference.area is None:
        raise ValueError("a 3D case needs domain.reference_area")
    p, S, shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    r = atlas.walls.faces.mid - np.asarray(reference.centre, dtype=float)[:3]
    cos_a, sin_a = np.cos(flow.alpha_rad), np.sin(flow.alpha_rad)
    q_area = dynamic_pressure(flow) * reference.area
    pressure = -p * S                                     # (3, n)
    force = pressure.sum(axis=1) + shear.sum(axis=1)
    friction = shear.sum(axis=1)
    lift = np.array([-sin_a, 0.0, cos_a])
    drag = np.array([cos_a, 0.0, sin_a])
    load = pressure + shear
    moment = np.sum(r[:, 2] * load[0] - r[:, 0] * load[2])
    return Forces(float(force @ lift / q_area), float(force @ drag / q_area),
                  float(moment / (q_area * reference.length)), float(force[0]), float(force[1]),
                  float(friction @ drag / q_area), fz=float(force[2]), cy=float(force[1] / q_area))


def surface(U: Array, atlas: Atlas, flow: FlowConditions) -> dict[str, np.ndarray]:
    """Per wall face, body by body in loop order: ``x``, ``y``, ``body``,
    ``cp`` and, viscous, ``cf`` (along the downstream-pointing tangent,
    negative where separated)."""
    if atlas.dim == 3:
        return _surface_3d(U, atlas, flow)
    p, (sx, sy), shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    faces = atlas.walls.faces
    out = {"x": faces.mid[:, 0], "y": faces.mid[:, 1], "body": faces.body.astype(float),
           "cp": p / dynamic_pressure(flow)}
    if not flow.inviscid:
        length = np.hypot(sx, sy)
        tx, ty = sy / length, -sx / length
        sign = np.where(tx >= 0.0, 1.0, -1.0)
        out["cf"] = sign * (shear[0] * tx + shear[1] * ty) / length / dynamic_pressure(flow)
    return out


def _surface_3d(U: Array, atlas: Atlas, flow: FlowConditions) -> dict[str, np.ndarray]:
    """Per wall face: its running position on the wall side ``i``, ``k``
    (the wing sections of validation/sections.py use them), ``x``, ``y``, ``z``, ``cp`` and,
    viscous, the skin friction vector ``cf_x``, ``cf_y``, ``cf_z`` (wall shear
    over ``q_inf``)."""
    p, S, shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    faces = atlas.walls.faces
    mid = faces.mid
    q = dynamic_pressure(flow)
    out = {"i": faces.cell[:, 0].astype(float), "k": faces.cell[:, 1].astype(float),
           "x": mid[:, 0], "y": mid[:, 1], "z": mid[:, 2], "cp": p / q}
    if not flow.inviscid:
        area = np.linalg.norm(S, axis=0)
        for c, name in enumerate("xyz"):
            out[f"cf_{name}"] = shear[c] / area / q
    return out


def pressure_lift(U: Array, atlas: Atlas, freestream: Array, gamma: float) -> Array:
    """Pressure lift coefficient per unit reference area, traceable, for
    freestream primitives ``(rho, V, p)``: a batched polar's per-block monitor.
    Lift is normal to the freestream in the plane of incidence, the first and
    the last axis (core/state.py)."""
    dim = atlas.dim
    rows = Rows(dim)
    p = conservative_to_primitive(U[:rows.n_flow], gamma, dim)[2].ravel()[atlas.walls.cell]
    force = bc.wall_pressure_force(p - freestream[rows.pressure], atlas)
    fx, fy = force[0], force[-1]
    u, v = freestream[1], freestream[dim]
    speed2 = u * u + v * v
    return (fy * u - fx * v) / jnp.sqrt(speed2) / (0.5 * freestream[RHO] * speed2)
