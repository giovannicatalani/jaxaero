"""The elements of a multi-element (high-lift) section: read, placed, landmarked.

An element is an ``Airfoil`` (clockwise from its trailing edge, as deployed,
in the case's length unit) with a role, ``slat``, ``main`` or ``flap``.
Elements come one file per element, or all from one multi-curve file whose
curves join end to end into closed contours (the BANC ``.crv``: per curve a
point count, then its points). A closed contour is opened at its trailing
edge: the base between the two sharpest corners near the point farthest from
the leading edge, or the single corner of a sharp one.

Placement re-sets a slat or a flap against the main element: a rotation by
``deflection_deg`` from its given position (positive deploys it further:
trailing edge down for a flap, leading edge down for a slat), then the
translation giving the requested ``gap`` (smallest distance from the upstream
element's trailing-edge midpoint to the downstream element's surface) and
``overlap`` (along the main element's chord, the upstream element's trailing
edge ahead of the downstream element's leading edge is positive), by Newton's
method on the two equations.

Landmarks, on the contour as given: the leading edge (``Airfoil``'s: the
point farthest from the trailing-edge midpoint), and the cove lip of a slat
or a main element, the sharpest convex corner of its lower surface within a
window of its chord set by its role (the slat's cusp, the main element's
step ahead of the flap cove).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scipy.interpolate import CubicSpline

from .airfoil import Airfoil, from_points, read_airfoil

ROLES = ("slat", "main", "flap")

LIP_WINDOW = {"slat": (0.05, 0.9), "main": (0.4, 0.98)}
"""Where along the chord, from the leading edge, a role's cove lip is looked for."""

LIP_MIN_TURN_DEG = 45.0
"""A lower-surface corner turning less than this is no cove lip."""

TE_CORNER_DEG = 45.0
"""A closed contour's trailing-edge base ends at corners turning at least this."""

CORNER_DEG = 30.0
"""A contour point turning at least this is a corner: the surface is splined
piecewise between corners, and grids put a node on each."""


@dataclass(frozen=True)
class Element:
    """An airfoil in place, and what it is."""

    airfoil: Airfoil
    role: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"element role {self.role!r}; choose from {ROLES}")

    @property
    def trailing_edge(self) -> np.ndarray:
        a = self.airfoil
        return 0.5 * np.array([a.x[0] + a.x[-1], a.y[0] + a.y[-1]])

    @property
    def leading_edge(self) -> np.ndarray:
        k = self.airfoil.leading_edge
        return np.array([self.airfoil.x[k], self.airfoil.y[k]])

    @property
    def chord_direction(self) -> np.ndarray:
        """Unit vector from the leading to the trailing edge."""
        d = self.trailing_edge - self.leading_edge
        return d / np.linalg.norm(d)

    def moved(self, rotation_deg: float = 0.0, about=(0.0, 0.0), shift=(0.0, 0.0)) -> Element:
        """Rotated counter-clockwise by ``rotation_deg`` about ``about``, then shifted."""
        t = np.radians(rotation_deg)
        c, s = np.cos(t), np.sin(t)
        x, y = self.airfoil.x - about[0], self.airfoil.y - about[1]
        a = Airfoil(self.airfoil.name, c * x - s * y + about[0] + shift[0], s * x + c * y + about[1] + shift[1])
        return Element(a, self.role)


# --- Reading ------------------------------------------------------------------------------


def read_curves(path: str | Path) -> list[np.ndarray]:
    """The curves of a counted-curve file: per curve a line holding its point
    count, then its points, one per line (``x y`` or ``x y z``; ``z`` dropped)."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    curves, k = [], 0
    while k < len(lines):
        if len(lines[k]) != 1:
            raise ValueError(f"{Path(path).name}: expected a point count on line {k + 1}, got {' '.join(lines[k])!r}")
        n = int(lines[k][0])
        rows = lines[k + 1:k + 1 + n]
        if len(rows) < n or any(len(r) not in (2, 3) for r in rows):
            raise ValueError(f"{Path(path).name}: curve {len(curves)} (line {k + 1}) has fewer than {n} points")
        curves.append(np.array([[float(r[0]), float(r[1])] for r in rows]))
        k += 1 + n
    return curves


def join_contours(curves: list[np.ndarray], tol: float = 1e-6) -> list[np.ndarray]:
    """Curves chained end to start (either way round) into closed contours,
    ``(n, 2)`` each, the closing point not repeated. ``tol``: a fraction of
    the overall extent within which ends coincide.

    Raises:
        ValueError: if a chain does not close.
    """
    extent = np.ptp(np.concatenate(curves), axis=0).max()
    near = lambda p, q: np.hypot(*(p - q)) <= tol * extent  # noqa: E731
    unused, loops = list(range(len(curves))), []
    while unused:
        first = unused.pop(0)
        loop = [curves[first]]
        while not near(loop[-1][-1], loop[0][0]):
            for k in unused:
                c = curves[k]
                if near(c[0], loop[-1][-1]) or near(c[-1], loop[-1][-1]):
                    loop.append(c if near(c[0], loop[-1][-1]) else c[::-1])
                    unused.remove(k)
                    break
            else:
                raise ValueError(f"curve {first} does not close into a contour: its chain ends at "
                                 f"{loop[-1][-1]} with no curve starting there")
        points = np.concatenate([loop[0]] + [c[1:] for c in loop[1:]])[:-1]
        loops.append(points)
    return loops


def _turning(p: np.ndarray, closed: bool) -> np.ndarray:
    """Signed turning angle at each point, radians (left turns positive)."""
    e = np.roll(p, -1, axis=0) - p if closed else np.diff(p, axis=0)
    a, b = (np.roll(e, 1, axis=0), e) if closed else (e[:-1], e[1:])
    t = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))
    return t if closed else np.concatenate([[0.0], t, [0.0]])


def open_at_trailing_edge(points: np.ndarray, name: str) -> Airfoil:
    """A closed contour as an ``Airfoil``: cut at its trailing edge, the base
    (between the two sharpest corners within 5% of the chord of the point
    farthest from the leading edge) removed."""
    d = np.hypot(points[:, None, 0] - points[None, :, 0], points[:, None, 1] - points[None, :, 1])
    i, j = np.unravel_index(np.argmax(d), d.shape)
    chord = d[i, j]
    turn = np.abs(_turning(points, closed=True))
    # The trailing edge is the end of the diameter with the sharper corners near it.
    best = None
    for end in (i, j):
        near = np.flatnonzero(np.hypot(*(points - points[end]).T) <= 0.05 * chord)
        corners = near[turn[near] >= np.radians(TE_CORNER_DEG)]
        score = np.sort(turn[corners])[::-1][:2].sum() if corners.size else turn[end]
        if best is None or score > best[0]:
            best = (score, end, corners)
    _, end, corners = best
    n = len(points)
    if corners.size >= 2:
        two = corners[np.argsort(turn[corners])[::-1][:2]]
        a, b = sorted(two)
        # the base is the shorter way between the two corners
        inside = np.arange(a + 1, b) if (b - a) <= n - (b - a) else np.concatenate([np.arange(b + 1, n),
                                                                                    np.arange(0, a)])
        keep = np.setdiff1d(np.arange(n), inside)
        start = b if (b - a) <= n - (b - a) else a
        order = np.roll(keep, -int(np.searchsorted(keep, start)))
        contour = points[order]
    else:
        tip = corners[np.argmax(turn[corners])] if corners.size else end
        contour = np.concatenate([np.roll(points, -tip, axis=0), points[tip:tip + 1]])
    return from_points(contour[:, 0], contour[:, 1], name)


def read_elements(path: str | Path, roles: list[str], scale: float = 1.0) -> list[Element]:
    """Elements from one counted-curve file, its contours given ``roles`` in
    the order of their leading edges' ``x`` (front to back)."""
    loops = join_contours(read_curves(path))
    if len(loops) != len(roles):
        raise ValueError(f"{Path(path).name} holds {len(loops)} closed contours, the case lists "
                         f"{len(roles)} elements")
    foils = [open_at_trailing_edge(scale * p, f"{Path(path).stem}[{k}]") for k, p in enumerate(loops)]
    foils.sort(key=lambda a: a.x[a.leading_edge])
    return [Element(a, r) for a, r in zip(foils, roles)]


def read_element(path: str | Path, role: str, scale: float = 1.0) -> Element:
    """One element from a coordinate file (``read_airfoil``'s formats), as placed in it."""
    a = read_airfoil(path)
    return Element(Airfoil(a.name, scale * a.x, scale * a.y), role)


# --- Landmarks ------------------------------------------------------------------------------


def corners(element: Element) -> np.ndarray:
    """Indices of the contour's corners (its two ends excluded): the points turning at least ``CORNER_DEG``."""
    a = element.airfoil
    turn = np.abs(_turning(np.stack([a.x, a.y], axis=1), closed=False))
    return np.flatnonzero(turn >= np.radians(CORNER_DEG))


def surface(element: Element):
    """``s -> (..., 2)``: the contour at arc lengths ``s``, a cubic spline in
    arc length between consecutive corners (straight where a piece has two
    points), so corners stay sharp."""
    a = element.airfoil
    p = np.stack([a.x, a.y], axis=1)
    if a.sharp:
        p[-1] = p[0]
    breaks = np.concatenate([[0], corners(element), [a.x.size - 1]])
    s = a.s
    pieces = [(s[i], s[j], CubicSpline(s[i:j + 1], p[i:j + 1]) if j - i > 1 else None, p[i], p[j])
              for i, j in zip(breaks[:-1], breaks[1:])]

    def at(t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, a.length)
        out = np.empty(t.shape + (2,))
        for s0, s1, spline, p0, p1 in pieces:
            here = (t >= s0) & (t <= s1)
            out[here] = spline(t[here]) if spline is not None else \
                p0 + ((t[here] - s0) / (s1 - s0))[:, None] * (p1 - p0)
        return out

    return at



def chord_fraction(element: Element, points: np.ndarray) -> np.ndarray:
    """Distance of ``points`` ``(n, 2)`` along the element's chord from its leading edge, in chords."""
    c = element.airfoil.chord
    return (points - element.leading_edge) @ element.chord_direction / c


def cove_lip(element: Element, window_chords: float = 0.005) -> int | None:
    """Index of the cove lip: the lower-surface point turning most sharply
    convex (to the right, walking clockwise), the turning summed over
    ``window_chords`` of arc either side, within the role's ``LIP_WINDOW``,
    then the sharpest single point in that window; ``None`` for a flap, or if no corner turns ``LIP_MIN_TURN_DEG``."""
    if element.role not in LIP_WINDOW:
        return None
    a = element.airfoil
    p = np.stack([a.x, a.y], axis=1)
    lower = np.arange(1, a.leading_edge)
    lo, hi = LIP_WINDOW[element.role]
    f = chord_fraction(element, p[lower])
    candidates = lower[(f >= lo) & (f <= hi)]
    if not candidates.size:
        return None
    theta = np.unwrap(np.arctan2(*np.diff(p, axis=0).T[::-1]))      # heading of each edge
    s_mid = 0.5 * (a.s[1:] + a.s[:-1])
    w = window_chords * a.chord
    before = np.searchsorted(s_mid, a.s[candidates] - w, side="right") - 1
    after = np.searchsorted(s_mid, a.s[candidates] + w)
    before, after = np.clip(before, 0, len(theta) - 1), np.clip(after, 0, len(theta) - 1)
    turn = theta[after] - theta[before]                              # right turns negative
    k = int(np.argmin(turn))
    if -turn[k] < np.radians(LIP_MIN_TURN_DEG):
        return None
    # the corner itself: the sharpest single point within the window
    near = candidates[(np.abs(a.s[candidates] - a.s[candidates[k]]) <= w)
                      | (np.abs(candidates - candidates[k]) <= 1)]
    return int(near[np.argmin(_turning(p, closed=False)[near])])


# --- Placement ------------------------------------------------------------------------------


def nearest_on(element: Element, point: np.ndarray, samples: int = 2000) -> tuple[float, np.ndarray]:
    """Arc length and position of the element's surface point nearest ``point``:
    the closest of ``samples`` points, refined by Newton on the spline."""
    fx, fy = element.airfoil.spline()
    L = element.airfoil.length
    s = np.linspace(0.0, L, samples)
    k = int(np.argmin(np.hypot(fx(s) - point[0], fy(s) - point[1])))
    t = s[k]
    for _ in range(30):
        dx, dy = fx(t) - point[0], fy(t) - point[1]
        g = dx * fx(t, 1) + dy * fy(t, 1)
        h = fx(t, 1) ** 2 + fy(t, 1) ** 2 + dx * fx(t, 2) + dy * fy(t, 2)
        step = -g / h if h > 0 else 0.0
        t = float(np.clip(t + step, 0.0, L))
        if abs(step) < 1e-15 * L:
            break
    return t, np.array([fx(t), fy(t)])


def gap(upstream: Element, downstream: Element) -> float:
    """Smallest distance from the upstream element's trailing-edge midpoint to the downstream element."""
    te = upstream.trailing_edge
    return float(np.hypot(*(nearest_on(downstream, te)[1] - te)))


def overlap(upstream: Element, downstream: Element, main: Element) -> float:
    """Along the main element's chord: upstream trailing edge minus downstream
    leading edge (positive when they overlap)."""
    return float((upstream.trailing_edge - downstream.leading_edge) @ main.chord_direction)


def deflection(element: Element, angle_deg: float) -> Element:
    """Rotated by ``angle_deg`` more deployed: a flap about its leading edge,
    trailing edge down; a slat about its trailing edge, leading edge down."""
    sign = {"flap": -1.0, "slat": 1.0}.get(element.role)
    if sign is None:
        raise ValueError(f"only a slat or a flap is deflected, not a {element.role!r} element")
    pivot = element.leading_edge if element.role == "flap" else element.trailing_edge
    return element.moved(sign * angle_deg, about=pivot)


def place(element: Element, main: Element, deflection_deg: float = 0.0, gap_: float | None = None,
          overlap_: float | None = None, tol: float = 1e-13) -> Element:
    """``element`` (a slat or flap) re-set against ``main``: rotated by
    ``deflection_deg`` (``deflection``), then translated so that its gap and
    overlap with the main element take the given values (one left ``None``
    keeps its value after the rotation). A flap is downstream of the main element, a slat upstream.

    Raises:
        ValueError: if Newton's method does not converge (a gap the geometry
        cannot take at that overlap).
    """
    moved = deflection(element, deflection_deg) if deflection_deg else element
    if gap_ is None and overlap_ is None:
        return moved
    up_down = (lambda e: (main, e)) if element.role == "flap" else (lambda e: (e, main))
    target = np.array([gap(*up_down(moved)) if gap_ is None else gap_,
                       overlap(*up_down(moved), main) if overlap_ is None else overlap_])
    sign = 1.0 if element.role == "slat" else -1.0           # does the moving element carry the TE?
    shift = np.zeros(2)
    for _ in range(50):
        trial = moved.moved(shift=shift)
        up, down = up_down(trial)
        te = up.trailing_edge
        p = nearest_on(down, te)[1]
        f = np.array([np.hypot(*(te - p)), overlap(up, down, main)]) - target
        if np.max(np.abs(f)) <= tol * main.airfoil.chord:
            return trial
        n = (te - p) / np.hypot(*(te - p))
        J = np.array([sign * n, sign * main.chord_direction])     # d(gap, overlap)/d(shift)
        shift = shift - np.linalg.solve(J, f)
    raise ValueError(f"placing the {element.role}: gap {gap_} and overlap {overlap_} not reached "
                     f"(residual {np.max(np.abs(f)):.2e})")
