"""O- and C-grids around an airfoil, by hyperbolic marching.

The wall line (O-grid: the closed contour; C-grid: lower wake, airfoil,
upper wake) is distributed by Vinokur's stretching -- clustered at the
leading and trailing edges -- and marched outward layer by layer, each layer
solving the linearized orthogonality and cell-area equations

    x_xi x_eta + y_xi y_eta = 0,      x_xi y_eta - y_xi x_eta = dV

(Steger and Chaussee, SIAM J. Sci. Stat. Comput. 1 (1980) 431-437), with
implicit second-difference smoothing of each step in ``xi`` (after Chan and
Steger, Appl. Math. Comput. 51 (1992) 181-205). Steps grow geometrically from
the wall spacing to the far-field distance. A C-grid's outflow planes are
kept straight and normal to the wake.

What it is for: one body, convex or mildly concave. A deep concavity (a
slat cove, a flap gap) folds the marching; the quality report then refuses
the grid, and such geometry is meshed elsewhere and read from file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ..geometry.airfoil import Airfoil


# --- Distributions -------------------------------------------------------------------


def vinokur(n: int, ds0: float, ds1: float) -> np.ndarray:
    """``n + 1`` points on [0, 1], first and last intervals close to ``ds0``
    and ``ds1`` (fractions of the length; the end slopes match exactly, the
    intervals to a few percent): Vinokur's two-sided tanh/tan stretching
    (J. Comput. Phys. 50 (1983) 215-234)."""
    xi = np.linspace(0.0, 1.0, n + 1)
    s0, s1 = 1.0 / (n * ds0), 1.0 / (n * ds1)
    a, b = np.sqrt(s0 / s1), np.sqrt(s0 * s1)
    if abs(b - 1.0) < 1e-6:
        u = xi
    elif b > 1.0:
        d = brentq(lambda d: np.sinh(d) / d - b, 1e-12, 50.0)
        u = 0.5 * (1.0 + np.tanh(d * (xi - 0.5)) / np.tanh(0.5 * d))
    else:
        d = brentq(lambda d: np.sin(d) / d - b, 1e-12, np.pi - 1e-12)
        u = 0.5 * (1.0 + np.tan(d * (xi - 0.5)) / np.tan(0.5 * d))
    s = u / (a + (1.0 - a) * u)
    s[0], s[-1] = 0.0, 1.0
    return s


def geometric(n: int, first: float, total: float) -> np.ndarray:
    """``n + 1`` points on [0, total], growing geometrically from ``first``."""
    if first * n >= total:
        raise ValueError(f"a first step of {first:g} over {n} steps already exceeds {total:g}")
    ratio = brentq(lambda r: first * (r**n - 1.0) / (r - 1.0) - total, 1.0 + 1e-12, 10.0)
    return np.concatenate([[0.0], first * np.cumsum(ratio ** np.arange(n))])


# --- The wall line ----------------------------------------------------------------------


@dataclass(frozen=True)
class Spacing:
    """Surface spacings, in chords: at the leading and the trailing edge."""

    leading_edge: float = 2e-3
    trailing_edge: float = 2e-3


def surface(airfoil: Airfoil, n: int, spacing: Spacing) -> tuple[np.ndarray, np.ndarray]:
    """``n + 1`` nodes round the contour, trailing edge to trailing edge,
    ``n / 2`` cells on each side of the leading edge (``n`` even)."""
    if n % 2:
        raise ValueError(f"cells round the airfoil must be even, got {n}")
    s_le, L = float(airfoil.s[airfoil.leading_edge]), airfoil.length
    c = airfoil.chord
    lower = s_le * vinokur(n // 2, spacing.trailing_edge * c / s_le, spacing.leading_edge * c / s_le)
    upper = s_le + (L - s_le) * vinokur(n // 2, spacing.leading_edge * c / (L - s_le),
                                         spacing.trailing_edge * c / (L - s_le))
    x, y = airfoil.points(np.concatenate([lower, upper[1:]]))
    if airfoil.sharp:
        x[-1], y[-1] = x[0], y[0]
    return x, y


def wake(airfoil: Airfoil, x_te: float, y_te: float, n: int, first: float, length: float):
    """``n + 1`` nodes from the trailing edge along the bisector of its two
    surface tangents, ``length`` long, the first step ``first``."""
    fx, fy = airfoil.spline()
    L = airfoil.length
    lower = np.array([fx(0.0, 1), fy(0.0, 1)])
    upper = -np.array([fx(L, 1), fy(L, 1)])
    direction = -(lower / np.linalg.norm(lower) + upper / np.linalg.norm(upper))
    direction /= np.linalg.norm(direction)
    s = geometric(n, first, length)
    return x_te + s * direction[0], y_te + s * direction[1], direction


# --- Marching -------------------------------------------------------------------------


def _derivative(a: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))
    d = np.empty_like(a)
    d[1:-1] = 0.5 * (a[2:] - a[:-2])
    d[0], d[-1] = a[1] - a[0], a[-1] - a[-2]
    return d


def _second(a: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return np.roll(a, -1, axis=0) - 2.0 * a + np.roll(a, 1, axis=0)
    d = np.zeros_like(a)
    d[1:-1] = a[2:] - 2.0 * a[1:-1] + a[:-2]
    return d


def march(line: np.ndarray, steps: np.ndarray, periodic: bool, outflow: np.ndarray | None = None,
          smoothing: float = 3.0) -> np.ndarray:
    """Layers marched from ``line`` ``(n, 2)``, one per step of ``steps``
    (``(nj,)`` distances), ``(n, nj + 1, 2)``. ``periodic``: a closed line
    (an O-grid, its last node not repeated). Otherwise the two end nodes stay
    on the lines through them along ``outflow`` (the wake direction), level
    with their neighbours."""
    n, nj = len(line), len(steps)
    layers = np.empty((n, nj + 1, 2))
    layers[:, 0] = line
    r = line.copy()
    for k, ds in enumerate(steps):
        r_xi = _derivative(r, periodic)
        length2 = np.sum(r_xi**2, axis=1)
        area = ds * np.sqrt(length2)
        for _ in range(2):                                   # smooth the target areas
            area = 0.5 * area + 0.25 * (np.roll(area, 1) + np.roll(area, -1)) if periodic else \
                np.concatenate([[area[0]], 0.5 * area[1:-1] + 0.25 * (area[:-2] + area[2:]), [area[-1]]])
        r_eta = (area / length2)[:, None] * np.stack([-r_xi[:, 1], r_xi[:, 0]], axis=1)
        x_xi, y_xi = r_xi[:, 0], r_xi[:, 1]
        x_eta, y_eta = r_eta[:, 0], r_eta[:, 1]
        # C = B^-1 A, g = B^-1 (0, 2 dV), B^-1 = [[x_xi, -y_xi], [y_xi, x_xi]] / |r_xi|^2.
        inv = 1.0 / length2
        C = np.empty((n, 2, 2))
        C[:, 0, 0] = inv * (x_xi * x_eta - y_xi * y_eta)
        C[:, 0, 1] = inv * (x_xi * y_eta + y_xi * x_eta)
        C[:, 1, 0] = inv * (y_xi * x_eta + x_xi * y_eta)
        C[:, 1, 1] = inv * (y_xi * y_eta - x_xi * x_eta)
        g = np.stack([-y_xi, x_xi], axis=1) * (inv * 2.0 * area)[:, None]
        # Smoothing acts on the step r(k+1) - r(k), never on the positions (a
        # stretched line -- a geometric wake -- has a large second difference
        # of its positions, and smoothing those drags its nodes sideways), and
        # is the same in every layer: grown from zero at the wall it let the
        # lines leaving an O-grid's sharp trailing edge cross (NACA 0012,
        # 128 x 96: 1622 folded cells at 0.2, none at a constant 3).
        eps = smoothing
        rhs = g + r - eps * _second(r, periodic)
        rows, cols, vals = [], [], []

        def put(i, j, block):
            for a in range(2):
                for b in range(2):
                    rows.append(2 * i + a), cols.append(2 * j + b), vals.append(block[a, b])

        eye = np.eye(2)
        interior = range(n) if periodic else range(1, n - 1)
        for i in interior:
            put(i, (i - 1) % n, -0.5 * C[i] - eps * eye)
            put(i, i, (1.0 + 2.0 * eps) * eye)
            put(i, (i + 1) % n, 0.5 * C[i] - eps * eye)
        if not periodic:
            w = outflow / np.linalg.norm(outflow)
            m = np.array([-w[1], w[0]])
            for end, inner in ((0, 1), (n - 1, n - 2)):
                put(end, end, np.stack([w, m]))               # w . r_end = w . r_end(k)
                put(end, inner, np.stack([np.zeros(2), -m]))  # m . r_end = m . r_inner
                rhs[end] = [w @ r[end], 0.0]
        A = coo_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).tocsc()
        r = spsolve(A, rhs.ravel()).reshape(n, 2)
        layers[:, k + 1] = r
    return layers


# --- The grids ----------------------------------------------------------------------------


def _reach(wall: np.ndarray, outer: np.ndarray) -> float:
    """Smallest distance from an outer node to a wall node."""
    d = np.hypot(outer[:, None, 0] - wall[None, ::4, 0], outer[:, None, 1] - wall[None, ::4, 1])
    return float(d.min())


def _march_to(line, wall, cells_normal, first, far_field, iterations=4, **kwargs):
    """March so the outer layer lies ``far_field`` from the wall: the steps grow
    geometrically from ``first``; the linearized marching falls short where
    steps are large, so the total is corrected until it does (to 1%)."""
    total = far_field
    for _ in range(iterations):
        layers = march(line, np.diff(geometric(cells_normal, first, total)), **kwargs)
        reach = _reach(wall, layers[:, -1])
        if abs(reach - far_field) <= 0.01 * far_field:
            break
        total *= far_field / reach
    return layers


def o_grid(airfoil: Airfoil, cells_airfoil: int, cells_normal: int, wall_spacing: float,
           far_field: float = 100.0, spacing: Spacing = Spacing(), smoothing: float = 3.0):
    """Node arrays ``(ni + 1, nj + 1)`` of an O-grid: ``i`` round the body
    from the trailing edge (lower surface first), ``j`` outward; the seam
    node line repeated exactly. A blunt trailing edge gets its base as part
    of the contour, ``cells_airfoil // 32`` (at least 2) cells across it."""
    c = airfoil.chord
    x, y = surface(airfoil, cells_airfoil, spacing)
    if not airfoil.sharp:
        n_base = max(2, cells_airfoil // 32)
        t = np.linspace(0.0, 1.0, n_base + 1)[1:-1]
        x = np.concatenate([x, x[-1] + t * (x[0] - x[-1])])
        y = np.concatenate([y, y[-1] + t * (y[0] - y[-1])])
    else:
        x, y = x[:-1], y[:-1]
    line = np.stack([x, y], axis=1)
    layers = _march_to(line, line, cells_normal, wall_spacing * c, far_field * c,
                       periodic=True, smoothing=smoothing)
    layers = np.concatenate([layers, layers[:1]], axis=0)
    return layers[..., 0], layers[..., 1]


def c_grid(airfoil: Airfoil, cells_airfoil: int, cells_wake: int, cells_normal: int, wall_spacing: float,
           far_field: float = 100.0, spacing: Spacing = Spacing(), smoothing: float = 3.0):
    """Node arrays of a C-grid: ``i`` from the downstream end of the lower
    wake, round the airfoil (lower surface first), out along the upper wake;
    the two wake lines coincide (the cut). Needs a sharp trailing edge."""
    if not airfoil.sharp:
        raise ValueError(f"airfoil {airfoil.name!r} has a blunt trailing edge "
                         f"(gap {airfoil.trailing_edge_gap:.3g}); use an O-grid, or close it "
                         f"(case key domain.trailing_edge = \"close\")")
    c = airfoil.chord
    x, y = surface(airfoil, cells_airfoil, spacing)
    first = np.hypot(x[1] - x[0], y[1] - y[0])
    wx, wy, direction = wake(airfoil, x[0], y[0], cells_wake, first, far_field * c)
    line_x = np.concatenate([wx[::-1], x[1:-1], wx])
    line_y = np.concatenate([wy[::-1], y[1:-1], wy])
    line = np.stack([line_x, line_y], axis=1)
    layers = _march_to(line, np.stack([x, y], axis=1), cells_normal, wall_spacing * c, far_field * c,
                       periodic=False, outflow=direction, smoothing=smoothing)
    return layers[..., 0], layers[..., 1]
