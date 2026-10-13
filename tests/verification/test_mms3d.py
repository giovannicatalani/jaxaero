"""The method of manufactured solutions in 3D.

As the 2D MMS (test_mms.py), on a curvilinear 3D block: the unit cube under a
smooth map that warps every face, a smooth field with all three velocity
components, Re = 100, no limiter. Every boundary takes Dirichlet ghosts:
``U_e`` at the centres of the cells of linearly extrapolated nodes (a point
reflection of the first cells through the face midpoints, consistent over
edges and corners), and those are the ghost centres the gradients see. (The
atlas reflects through the face *plane*, which on a skewed boundary cell
puts the ghost-cell midpoint O(h) off the face midpoint: an O(1) truncation
error in the boundary cells, first order in the solution (measured),
which would hide the interior scheme this test is about.)
The source is the exact flux integral over each bilinear face,

    S_c = sum_faces int (F - F_v)(U_e(x(s, t))) . (x_s x x_t) ds dt,

3x3 Gauss per face. The discrete solution is driven to round-off and the
error measured at the cell centres: second order in every variable, max norm
and L2, verifies the 3D convective flux, MUSCL, the 3D diamond gradients and
the viscous flux together on warped cells.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxaero.core import constants
from jaxaero.core.config import FlowConditions, ImplicitConfig, NumericsConfig
from jaxaero.core.state import primitive_to_conservative, split
from jaxaero.mesh.atlas import build_atlas
from jaxaero.mesh.block import Grid
from jaxaero.mesh.block3d import SIDES_3D, box_sides
from jaxaero.mesh.mesh import assign_conditions
from jaxaero.numerics import boundary as bc
from jaxaero.numerics.residual import make_residual, primitive_from_conservative
from jaxaero.solvers import implicit
from jaxaero.solvers.driver import run_to_steady

FLOW = FlowConditions(mach=0.5, alpha_deg=0.0, reynolds=100.0)
GAMMA, MU, PR = FLOW.gamma, FLOW.mu_inf, constants.PRANDTL_LAMINAR
WARP = 0.04


def mapped_nodes(n: int) -> np.ndarray:
    """The unit cube, ``n`` cells a side, under a smooth warping map."""
    xi = np.linspace(0.0, 1.0, n + 1)
    a, b, c = np.meshgrid(xi, xi, xi, indexing="ij")
    s = np.sin
    return np.stack([a + WARP * s(2 * np.pi * b) * s(np.pi * c),
                     b + WARP * s(2 * np.pi * c) * s(np.pi * a),
                     c + WARP * s(2 * np.pi * a) * s(np.pi * b)])


def manufactured():
    """``(primitive(x), flux(x))`` as numpy callables of points ``(3, ...)``:
    ``flux`` returns ``(5, 3, ...)``, the Cartesian components of ``F - F_v``."""
    X = sp.symbols("x y z", real=True)
    x, y, z = X
    rho = 1 + sp.Rational(1, 10) * sp.sin(sp.pi * x) * sp.cos(sp.pi * y / 2) * (1 + z / 2)
    V = [sp.Rational(1, 2) + sp.Rational(1, 10) * sp.cos(sp.pi * y) * sp.sin(sp.pi * z),
         sp.Rational(1, 10) + sp.Rational(1, 10) * sp.sin(sp.pi * x) * sp.cos(sp.pi * z),
         sp.Rational(1, 5) + sp.Rational(1, 10) * sp.sin(sp.pi * (x + y))]
    p = 1 / GAMMA + sp.Rational(1, 20) * sp.cos(sp.pi * x / 2 + y) * sp.cos(sp.pi * z / 2)

    energy = p / (GAMMA - 1) + rho * sum(v**2 for v in V) / 2
    temp = GAMMA * p / rho
    div = sum(sp.diff(V[a], X[a]) for a in range(3))
    tau = [[MU * (sp.diff(V[a], X[b]) + sp.diff(V[b], X[a])) - (sp.Rational(2, 3) * MU * div if a == b else 0)
            for b in range(3)] for a in range(3)]
    k = MU / (PR * (GAMMA - 1))
    flux = []
    for b in range(3):                      # direction
        flux.append([rho * V[b]]
                    + [rho * V[a] * V[b] + (p if a == b else 0) - tau[a][b] for a in range(3)]
                    + [(energy + p) * V[b] - sum(V[a] * tau[a][b] for a in range(3)) - k * sp.diff(temp, X[b])])

    def vectorise(exprs, shape):
        flat = [e for row in exprs for e in (row if isinstance(row, list) else [row])]
        f = sp.lambdify(X, flat, "numpy")
        return lambda P: np.stack([np.broadcast_to(np.asarray(c, dtype=float), P.shape[1:])
                                   for c in f(*P)]).reshape(shape + P.shape[1:])

    return vectorise([rho, *V, p], (5,)), vectorise([list(r) for r in zip(*flux)], (5, 3))


def source(nodes: np.ndarray, flux) -> np.ndarray:
    """``sum_faces int (F - F_v) . dS`` per cell, 3x3 Gauss on each bilinear face."""
    g, w = np.polynomial.legendre.leggauss(3)
    g, w = 0.5 * (g + 1.0), 0.5 * w
    total = 0.0
    for axis in range(3):
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        n = np.moveaxis(nodes, (1 + axis, 1 + a1, 1 + a2), (1, 2, 3))
        pa, pb, pc, pd = n[:, :, :-1, :-1], n[:, :, 1:, :-1], n[:, :, 1:, 1:], n[:, :, :-1, 1:]
        face = 0.0
        for s, ws in zip(g, w):
            for t, wt in zip(g, w):
                x = (1 - s) * (1 - t) * pa + s * (1 - t) * pb + s * t * pc + (1 - s) * t * pd
                xs = (1 - t) * (pb - pa) + t * (pc - pd)
                xt = (1 - s) * (pd - pa) + s * (pc - pb)
                dS = np.cross(xs, xt, axis=0)
                face = face + ws * wt * np.einsum("vd...,d...->v...", flux(x), dS)
        face = np.moveaxis(face, (1, 2, 3), (1 + axis, 1 + a1, 1 + a2))
        total = total + (np.take(face, range(1, face.shape[1 + axis]), axis=1 + axis)
                         - np.take(face, range(face.shape[1 + axis] - 1), axis=1 + axis))
    return total


def dirichlet_ghosts(atlas, nodes: np.ndarray, primitive):
    """Ghost centres from linearly extrapolated nodes -- a point reflection of
    the first cells through the boundary face midpoints, carried over edges
    and corners -- as the operator's ghost centres, and ``U_e`` there as each
    fill stage's ghost values (keyed by the stage's size)."""
    from jaxaero.mesh.atlas3d import _extended_nodes
    from jaxaero.mesh.block import metrics

    g = atlas.g
    centres = metrics(_extended_nodes(nodes, g))[0]
    flat = centres.reshape(3, -1)
    by_size = {}
    for stage in atlas.stages:
        n_canvas = flat.shape[1]
        ghost_of_row = np.empty(int(stage.index.max()) + 1 - n_canvas, dtype=np.int64)
        rows = stage.index >= n_canvas
        ghost_of_row[stage.index[rows] - n_canvas] = np.flatnonzero(rows)
        offset = 0
        for kind, src, normal, mid in stage.conditions:
            ghosts = ghost_of_row[offset:offset + len(src)]
            by_size[normal.shape[1]] = jnp.asarray(primitive(flat[:, ghosts]))
            offset += len(src)
    assert len(by_size) == len(atlas.stages), "stages must differ in size to be told apart"
    padded = np.pad(centres, ((0, 0),) + ((1, 1),) * 3, mode="edge")
    return replace(atlas, centres=jnp.asarray(padded)), by_size


def solve(n: int, monkeypatch, fields, kappa: float = 0.0):
    primitive, flux = fields
    nodes = mapped_nodes(n)
    shape = (n, n, n)
    grid = Grid.from_node_array(nodes, box_sides(shape, **dict.fromkeys(SIDES_3D, "wall")))
    numerics = NumericsConfig(limiter="none", kappa=kappa)
    atlas = build_atlas(assign_conditions(grid, {"wall": "no_slip_wall"}), numerics.n_ghost)

    atlas, by_size = dirichlet_ghosts(atlas, nodes, primitive)
    monkeypatch.setitem(bc.CONDITIONS, "no_slip_wall", lambda q, face, c: by_size[face.normal.shape[1]])

    operator = make_residual(atlas, FLOW, numerics)
    g = numerics.n_ghost
    block = (slice(None),) + (slice(g, -g),) * 3
    s = jnp.zeros((5,) + atlas.shape).at[block].set(jnp.asarray(source(nodes, flux)))
    operator = replace(operator, residual=lambda U, base=operator.residual: base(U) - s)
    jacobian = make_residual(atlas, FLOW, numerics, first_order=True)

    q_exact = primitive(np.asarray(grid.centres))
    U0 = operator.initial_state().at[block].set(primitive_to_conservative(*split(jnp.asarray(q_exact), 3), GAMMA))
    config = ImplicitConfig()
    state = implicit.initial_state(operator, config)._replace(U=U0)
    state, history = run_to_steady(implicit.make_newton_step(operator, config, jacobian),
                                   state, max_iters=300, tol=1e-9, block_iters=20)
    assert history.residual[-1] < 1e-13, history.summary()
    q = np.asarray(primitive_from_conservative(state.U, GAMMA, 3))[block]
    return q, q_exact, np.asarray(grid.volume)


@pytest.mark.slow
@pytest.mark.parametrize("kappa", [0.0, 1.0 / 3.0])
def test_navier_stokes_mms_is_second_order_in_3d(monkeypatch, kappa) -> None:
    fields = manufactured()
    errors = []
    for n in (8, 16, 32):
        q, q_exact, volume = solve(n, monkeypatch, fields, kappa)
        e = q - q_exact
        errors.append((np.abs(e).max(axis=(1, 2, 3)), np.sqrt((volume * e**2).sum(axis=(1, 2, 3)) / volume.sum())))
    for norm, name in ((0, "Linf"), (1, "L2")):
        e = np.array([err[norm] for err in errors])
        order = np.log2(e[:-1] / e[1:])
        print(f"{name} errors (rho, u, v, w, p) per level:\n{e}\norders:\n{order}")
        assert np.all(order[-1] > 1.8), (name, order)
