"""Method of manufactured solutions for the compressible Navier-Stokes system.

A smooth field ``U_e`` that is *not* a solution is made one by adding the
source it leaves behind, ``S = div (F - F_v)(U_e)``. The discrete problem
``R(U) = S`` is then solved and the error ``U - U_e`` measured on a sequence of
grids: every variable must converge at the scheme's order, in the max norm as
well as L2. Nothing else exercises the whole viscous discretization at once --
convective flux, reconstruction, face gradients, stresses, heat flux, and their
assembly on a curvilinear grid.

Construction choices, each to keep the test about the interior scheme:

* ``U_e`` is the freestream plus a perturbation decaying like
  ``exp(-r^2 / R^2)`` about mid-chord, so the characteristic far field at ~148
  chords sees an exact freestream.
* The wall takes Dirichlet ghosts, ``U_e`` evaluated at the ghost centres (the
  reflections of the first interior centres through the wall faces). The
  no-slip wall itself is unit-tested separately.
* The source is the exact flux integral ``sum_faces int (F - F_v)(U_e) . n``,
  two-point Gauss on each straight face, so no centroid approximation enters.
* Re = 100 and no limiter: the viscous terms are not negligible on coarse
  grids, and a limiter would clip the field's smooth extrema to first order.

Reference: Roache, "Code Verification by the Method of Manufactured
Solutions", J. Fluids Eng. 124 (2002); Salari and Knupp, SAND2000-1444.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxaero.core import constants
from jaxaero.core.config import FlowConditions, ImplicitConfig, NumericsConfig
from validation.grids.karman_trefftz import make_o_grid
from jaxaero.core.state import primitive_to_conservative, split
from jaxaero.numerics import boundary as bc

from ..single import from_canvas, to_canvas, unit_j_normals, atlas_of
from jaxaero.numerics.residual import make_residual, primitive_from_conservative
from jaxaero.solvers import implicit
from jaxaero.solvers.driver import run_to_steady

FLOW = FlowConditions(mach=0.5, alpha_deg=1.0, reynolds=100.0)
GAMMA, MU, PR = FLOW.gamma, FLOW.mu_inf, constants.PRANDTL_LAMINAR


def manufactured():
    """``(primitive(x, y), flux_x(x, y), flux_y(x, y))`` as numpy callables.

    ``flux_x``, ``flux_y`` are the Cartesian components of ``F - F_v``, so the
    flux through a face with area vector ``S`` is ``flux_x Sx + flux_y Sy``.
    """
    x, y = sp.symbols("x y", real=True)
    g = sp.exp(-((x - 0.5) ** 2 + y**2) / sp.Rational(9, 16))
    u_inf, v_inf = FLOW.mach * np.cos(FLOW.alpha_rad), FLOW.mach * np.sin(FLOW.alpha_rad)
    rho = 1 + sp.Rational(1, 10) * g * (1 + sp.sin(sp.pi * x) / 2)
    u = u_inf + sp.Rational(1, 10) * g * sp.cos(sp.pi * y)
    v = v_inf + sp.Rational(1, 10) * g * sp.sin(sp.pi * x)
    p = 1 / GAMMA + sp.Rational(1, 20) * g * sp.cos(sp.pi * x / 2 + y)

    energy = p / (GAMMA - 1) + rho * (u**2 + v**2) / 2
    temp = GAMMA * p / rho
    div = sp.diff(u, x) + sp.diff(v, y)
    txx = MU * (2 * sp.diff(u, x) - sp.Rational(2, 3) * div)
    tyy = MU * (2 * sp.diff(v, y) - sp.Rational(2, 3) * div)
    txy = MU * (sp.diff(u, y) + sp.diff(v, x))
    k = MU / (PR * (GAMMA - 1))

    fx = [rho * u, rho * u**2 + p - txx, rho * u * v - txy,
          (energy + p) * u - (u * txx + v * txy) - k * sp.diff(temp, x)]
    fy = [rho * v, rho * u * v - txy, rho * v**2 + p - tyy,
          (energy + p) * v - (u * txy + v * tyy) - k * sp.diff(temp, y)]

    def vectorise(exprs):
        f = sp.lambdify((x, y), exprs, "numpy")
        return lambda a, b: np.stack([np.broadcast_to(c, np.shape(a)) for c in f(a, b)])

    return vectorise([rho, u, v, p]), vectorise(fx), vectorise(fy)


def ghost_centres(grid, n_ghost: int):
    """Centres of the wall ghosts, ``(ni, g)`` each, ordered outward ``j = -1 .. -g``."""
    nx, ny = (np.asarray(a) for a in unit_j_normals(grid, 0))
    x_node, y_node = np.asarray(grid.x_node), np.asarray(grid.y_node)
    mx, my = 0.5 * (x_node[:-1, 0] + x_node[1:, 0]), 0.5 * (y_node[:-1, 0] + y_node[1:, 0])
    xc, yc = np.asarray(grid.xc)[:, :n_ghost], np.asarray(grid.yc)[:, :n_ghost]
    d = (xc - mx[:, None]) * nx[:, None] + (yc - my[:, None]) * ny[:, None]
    gx, gy = xc - 2 * d * nx[:, None], yc - 2 * d * ny[:, None]
    return gx, gy


def source(grid, flux_x, flux_y) -> np.ndarray:
    """``sum_faces int (F - F_v)(U_e) . S``, two-point Gauss per face, ``(4, ni, nj)``."""
    x, y = np.asarray(grid.x_node), np.asarray(grid.y_node)
    weights = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))

    def face_flux(xa, ya, xb, yb, sx, sy):
        total = 0.0
        for w in weights:
            px, py = xa + w * (xb - xa), ya + w * (yb - ya)
            total = total + 0.5 * (flux_x(px, py) * sx + flux_y(px, py) * sy)
        return total

    fi = face_flux(x[:, :-1], y[:, :-1], x[:, 1:], y[:, 1:], np.asarray(grid.sx_i), np.asarray(grid.sy_i))
    fj = face_flux(x[:-1, :], y[:-1, :], x[1:, :], y[1:, :], np.asarray(grid.sx_j), np.asarray(grid.sy_j))
    return (fi[:, 1:] - fi[:, :-1]) + (fj[:, :, 1:] - fj[:, :, :-1])


def solve(n_c: int, monkeypatch, fields):
    """Converged discrete solution and the exact field at the centres."""
    primitive, flux_x, flux_y = fields
    grid = make_o_grid(n_c)
    gx, gy = ghost_centres(grid, 2)
    # One row per wall ghost, in the fill plan's order: faces along the wall,
    # the two layers of each innermost.
    wall = jnp.asarray(primitive(gx, gy)).reshape(4, -1)
    monkeypatch.setitem(bc.CONDITIONS, "no_slip_wall", lambda q, face, conditions: wall)

    numerics = NumericsConfig(limiter="none")
    operator = make_residual(atlas_of(grid, 2, True), FLOW, numerics)
    atlas = operator.atlas
    s = to_canvas(jnp.asarray(source(grid, flux_x, flux_y)), atlas, 0.0)
    operator = replace(operator, residual=lambda U, base=operator.residual: base(U) - s)
    jacobian = make_residual(atlas, FLOW, numerics, first_order=True)

    q_exact = primitive(np.asarray(grid.xc), np.asarray(grid.yc))
    U0 = to_canvas(primitive_to_conservative(*split(jnp.asarray(q_exact), 2), GAMMA), atlas, operator.initial_state())
    config = ImplicitConfig()
    state = implicit.initial_state(operator, config)._replace(U=U0)
    # Started from U_e the residual is already small (its peak ~1e-7), so the
    # criterion is absolute: converged means at round-off, not N orders down.
    state, history = run_to_steady(
        implicit.make_newton_step(operator, config, jacobian),
        state, max_iters=400, tol=1e-9, block_iters=20,
    )
    assert history.residual[-1] < 1e-13, history.summary()
    q = from_canvas(primitive_from_conservative(state.U, GAMMA, 2), atlas)
    return np.asarray(q), q_exact, np.asarray(grid.area)


@pytest.mark.slow
def test_navier_stokes_mms_is_second_order(monkeypatch) -> None:
    fields = manufactured()
    errors = []
    for n_c in (32, 64, 128):
        q, q_exact, area = solve(n_c, monkeypatch, fields)
        e = q - q_exact
        errors.append((np.abs(e).max(axis=(1, 2)), np.sqrt((area * e**2).sum(axis=(1, 2)) / area.sum())))
    for norm, name in ((0, "Linf"), (1, "L2")):
        e = np.array([err[norm] for err in errors])
        order = np.log2(e[:-1] / e[1:])
        print(f"{name} errors (rho, u, v, p) per level:\n{e}\norders:\n{order}")
        assert np.all(order[-1] > 1.8), (name, order)
