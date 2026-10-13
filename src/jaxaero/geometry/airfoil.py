"""An airfoil from its coordinates: read, splined, redistributed.

The contour is held as a cubic spline (not periodic: the trailing edge is a
corner) of the coordinates against arc length, running trailing edge -> lower surface -> leading edge -> upper
surface -> trailing edge (clockwise: the orientation the grids need). The
leading edge is the point farthest from the trailing-edge midpoint; the
trailing edge is sharp when the two ends coincide, blunt otherwise.

Sources: a coordinate file (Selig: a name line, then the points round the
contour; Lednicer: a name line, the point counts, the upper then the lower
surface from the leading edge; or a plain two-column table either way), or
the NACA 4-digit formula. Grids are built round ``unit_chord()``, closed
with ``closed()`` on request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class Airfoil:
    """A closed (or blunt-ended) contour, clockwise from the trailing edge.

    ``x, y``: the points as given, reordered; ``s``: their arc length;
    ``s_le``: the leading edge's.
    """

    name: str
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if self.x.size < 5:
            raise ValueError(f"airfoil {self.name!r}: at least 5 points are needed, got {self.x.size}")
        if np.any(np.hypot(np.diff(self.x), np.diff(self.y)) == 0.0):
            raise ValueError(f"airfoil {self.name!r}: repeated consecutive points")

    @property
    def s(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def trailing_edge_gap(self) -> float:
        return float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]))

    @property
    def sharp(self) -> bool:
        """Closed trailing edge: the ends coincide to 1e-6 of the chord."""
        return self.trailing_edge_gap <= 1e-6 * self.chord

    @property
    def leading_edge(self) -> int:
        """Index of the point farthest from the trailing-edge midpoint."""
        te = 0.5 * np.array([self.x[0] + self.x[-1], self.y[0] + self.y[-1]])
        return int(np.argmax(np.hypot(self.x - te[0], self.y - te[1])))

    @property
    def chord(self) -> float:
        te = 0.5 * np.array([self.x[0] + self.x[-1], self.y[0] + self.y[-1]])
        k = self.leading_edge
        return float(np.hypot(self.x[k] - te[0], self.y[k] - te[1]))

    def spline(self) -> tuple[CubicSpline, CubicSpline]:
        """``x(s), y(s)``, not periodic: a trailing edge is a corner, sharp or
        not (a sharp one's two ends are made to coincide exactly)."""
        x, y = self.x.copy(), self.y.copy()
        if self.sharp:
            x[-1], y[-1] = x[0], y[0]
        return CubicSpline(self.s, x), CubicSpline(self.s, y)

    def points(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The contour at arc lengths ``s``."""
        fx, fy = self.spline()
        return fx(s), fy(s)

    def unit_chord(self) -> Airfoil:
        """Leading edge moved to the origin, scaled to chord 1, not rotated."""
        k, c = self.leading_edge, self.chord
        return Airfoil(self.name, (self.x - self.x[k]) / c, (self.y - self.y[k]) / c)

    def closed(self) -> Airfoil:
        """The trailing edge closed at the midpoint of its gap: each surface
        moves by a displacement growing linearly along the chord, zero at the
        leading edge, half the gap at the end. Chord and leading edge unchanged."""
        if self.sharp:
            return self
        k = self.leading_edge
        mid = 0.5 * np.array([self.x[0] + self.x[-1], self.y[0] + self.y[-1]])
        le = np.array([self.x[k], self.y[k]])
        t = (mid - le) / np.linalg.norm(mid - le)
        along = (self.x - le[0]) * t[0] + (self.y - le[1]) * t[1]
        end = np.where(np.arange(self.x.size) < k, 0, -1)       # the lower surface's end, or the upper's
        w = np.clip(along / along[end], 0.0, 1.0)
        x = self.x + w * (mid[0] - self.x[end])
        y = self.y + w * (mid[1] - self.y[end])
        x[0], y[0], x[-1], y[-1] = mid[0], mid[1], mid[0], mid[1]
        return Airfoil(self.name, x, y)


def from_points(x, y, name: str = "airfoil") -> Airfoil:
    """An airfoil from points round its contour, in either direction and
    starting at either end of the trailing edge; reordered clockwise from
    the trailing edge (lower surface first)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) > 0.0])
    x, y = x[keep], y[keep]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 0.5 * (x[-1] * y[0] - x[0] * y[-1])
    if area > 0.0:                      # counter-clockwise: walk it the other way
        x, y = x[::-1], y[::-1]
    return Airfoil(name, x, y)


def read_airfoil(path: str | Path) -> Airfoil:
    """A coordinate file: Selig, Lednicer or a plain two-column table.

    Raises:
        ValueError: if the file holds fewer than five coordinate pairs.
    """
    path = Path(path)
    rows, name = [], path.stem
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            if not rows and parts:
                name = line.strip()
            continue
        if len(values) == 2:
            rows.append(values)
    data = np.array(rows)
    if len(data) < 5:
        raise ValueError(f"{path.name}: fewer than 5 coordinate pairs")
    # Lednicer: a first row of point counts (> 1), then upper and lower from the LE.
    if data[0, 0] > 1.5 and data[0, 1] > 1.5:
        n_upper, n_lower = int(data[0, 0]), int(data[0, 1])
        upper, lower = data[1:1 + n_upper], data[1 + n_upper:1 + n_upper + n_lower]
        contour = np.concatenate([upper[::-1], lower[1:]])
        return from_points(contour[:, 0], contour[:, 1], name)
    return from_points(data[:, 0], data[:, 1], name)


def naca4(code: str, n: int = 201, sharp: bool = True) -> Airfoil:
    """The NACA 4-digit section ``code`` (e.g. ``"0012"``, ``"2412"``),
    ``2 n - 1`` points with cosine spacing; ``sharp`` uses the closed
    trailing-edge coefficient (-0.1036) instead of the standard (-0.1015)."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"a NACA 4-digit code is four digits, got {code!r}")
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))
    a4 = -0.1036 if sharp else -0.1015
    yt = 5.0 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 + a4 * xc**4)
    if m > 0.0 and p > 0.0:
        front = xc < p
        yc = np.where(front, m / p**2 * (2 * p * xc - xc**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * xc - xc**2))
        dy = np.where(front, 2 * m / p**2 * (p - xc), 2 * m / (1 - p) ** 2 * (p - xc))
    else:
        yc, dy = np.zeros_like(xc), np.zeros_like(xc)
    theta = np.arctan(dy)
    xu, yu = xc - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = xc + yt * np.sin(theta), yc - yt * np.cos(theta)
    x = np.concatenate([xl[::-1], xu[1:]])
    y = np.concatenate([yl[::-1], yu[1:]])
    if sharp:
        y[0] = y[-1] = 0.5 * (y[0] + y[-1])
    return from_points(x, y, f"NACA {code}")


def airfoil_source(spec: str, base: Path | None = None) -> Airfoil:
    """``naca####`` (or ``naca####_blunt``), else a coordinate file path
    (relative to ``base`` first, then the working directory)."""
    lowered = spec.lower()
    if lowered.startswith("naca") and lowered[4:8].isdigit() and len(lowered) in (8, 14):
        return naca4(lowered[4:8], sharp=not lowered.endswith("_blunt"))
    for root in ([base] if base else []) + [Path.cwd()]:
        path = Path(spec) if Path(spec).is_absolute() else root / spec
        if path.is_file():
            return read_airfoil(path)
    raise FileNotFoundError(f"airfoil {spec!r}: not a NACA 4-digit name, nor a file in {base} or {Path.cwd()}")
