"""What lies beyond each side of a 3D structured block.

A 3D block has cells ``(ni, nj, nk)`` and six sides, named by the index
constant on them (``i_min`` ... ``k_max``). Along a side run its two other
index directions, in ascending order (``running_axes``): ``(j, k)`` on an
``i`` side, ``(i, k)`` on a ``j`` side, ``(i, j)`` on a ``k`` side. Each side
is tiled by rectangles of cells ``[start, stop)`` in those running indices:

* a ``BoundaryRect``, a named patch, which a case maps to a boundary
  condition, as in 2D (mesh/block.py);
* an ``InterfaceRect``, cells facing cells of a side of a block (this one or
  another). Running position ``a`` maps to the partner's running position

      p[m] = origin[m] + s_m (a[axes[m]] - start[axes[m]])     (mod period[m] if > 0)

  with ``s_m = -1`` where ``flip[m]``. Ghost layer ``l`` is the partner's
  ``l``-th layer inward, copied (velocities are Cartesian).

This covers an O seam (``k_min`` facing ``k_max``), a C-grid wake cut
(``j_min`` facing itself, ``i`` reversed) and a **pole**, a side collapsed
to a line: ``i_min`` facing itself with ``k`` shifted by half a turn, so the
ghost across the axis is the cell diametrically opposite (the assembly
finds poles and seams from the coordinates, mesh/assemble3d.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .block import CONDITION_KINDS, WALL_KINDS

SIDES_3D = ("j_min", "j_max", "i_min", "i_max", "k_min", "k_max")
AXIS = {"i": 0, "j": 1, "k": 2}


def normal_axis(side: str) -> int:
    return AXIS[side[0]]


def running_axes(side: str) -> tuple[int, int]:
    """The two index directions along a side, ascending."""
    n = normal_axis(side)
    return tuple(a for a in range(3) if a != n)


@dataclass(frozen=True)
class BoundaryRect:
    patch: str
    start: tuple[int, int]
    stop: tuple[int, int]
    kind: str | None = None

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in CONDITION_KINDS:
            raise ValueError(f"unknown boundary condition {self.kind!r}; known: {CONDITION_KINDS}")


@dataclass(frozen=True)
class InterfaceRect:
    start: tuple[int, int]
    stop: tuple[int, int]
    block: int
    side: str
    origin: tuple[int, int]
    axes: tuple[int, int] = (0, 1)
    flip: tuple[bool, bool] = (False, False)
    period: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.side not in SIDES_3D:
            raise ValueError(f"unknown side {self.side!r}; known: {SIDES_3D}")

    def partner(self, a: tuple[int, int]) -> tuple[int, int]:
        """The partner's running position facing my running position ``a``
        (``a`` may lie outside the rectangle: corners of the ghost fill)."""
        out = []
        for m in range(2):
            d = a[self.axes[m]] - self.start[self.axes[m]]
            p = self.origin[m] + (-d if self.flip[m] else d)
            out.append(p % self.period[m] if self.period[m] else p)
        return tuple(out)


Rect = BoundaryRect | InterfaceRect


@dataclass(frozen=True)
class Sides3D:
    """Rectangles tiling each of a block's six sides."""

    shape: tuple[int, int, int]
    j_min: tuple[Rect, ...]
    j_max: tuple[Rect, ...]
    i_min: tuple[Rect, ...]
    i_max: tuple[Rect, ...]
    k_min: tuple[Rect, ...]
    k_max: tuple[Rect, ...]

    def __post_init__(self) -> None:
        for side in SIDES_3D:
            n = self.extent(side)
            count = np.zeros(n, dtype=int)
            for r in self.side(side):
                if not all(0 <= r.start[m] < r.stop[m] <= n[m] for m in range(2)):
                    raise ValueError(f"rectangle {r} outside {side} of extent {n}")
                count[r.start[0]:r.stop[0], r.start[1]:r.stop[1]] += 1
            if np.any(count != 1):
                raise ValueError(f"the rectangles of {side} do not tile it exactly once")

    @property
    def ni(self) -> int:
        return self.shape[0]

    @property
    def nj(self) -> int:
        return self.shape[1]

    @property
    def nk(self) -> int:
        return self.shape[2]

    def side(self, name: str) -> tuple[Rect, ...]:
        return getattr(self, name)

    def extent(self, name: str) -> tuple[int, int]:
        a, b = running_axes(name)
        return self.shape[a], self.shape[b]

    def boundaries(self) -> list[tuple[str, BoundaryRect]]:
        return [(s, r) for s in SIDES_3D for r in self.side(s) if isinstance(r, BoundaryRect)]

    def walls(self) -> list[tuple[str, BoundaryRect]]:
        return [(s, r) for s, r in self.boundaries() if r.kind in WALL_KINDS]

    def with_conditions(self, kinds: dict[str, str]) -> "Sides3D":
        def assign(rects):
            return tuple(replace(r, kind=kinds[r.patch]) if isinstance(r, BoundaryRect) else r
                         for r in rects)
        return replace(self, **{s: assign(self.side(s)) for s in SIDES_3D})


def box_sides(shape: tuple[int, int, int], **sides) -> Sides3D:
    """Each side one patch (a string) or the given rectangles."""
    def side(name):
        value = sides[name]
        if isinstance(value, str):
            return (BoundaryRect(value, (0, 0), tuple(np.array(shape)[list(running_axes(name))])),)
        return tuple(value)
    return Sides3D(tuple(shape), *(side(name) for name in SIDES_3D))
