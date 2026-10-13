"""A mesh: blocks, their named patches, walls and bodies, wall distance.

``Mesh`` is a tuple of ``Grid`` blocks; an ``Interface`` names its partner by
index in it. A single-block grid is the mesh ``(grid,)``. The solver runs a
mesh packed into one canvas (mesh/atlas.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .block import SIDES, Boundary, Grid, Interface

Mesh = tuple[Grid, ...]


def as_mesh(grid: Grid | Mesh) -> Mesh:
    return (grid,) if isinstance(grid, Grid) else tuple(grid)


def patches(mesh: Grid | Mesh) -> list[str]:
    """Every patch name of the mesh's boundaries, in first-seen order."""
    names: list[str] = []
    for block in as_mesh(mesh):
        for _, seg in block.sides.boundaries():
            if seg.patch not in names:
                names.append(seg.patch)
    return names


def assign_conditions(mesh: Grid | Mesh, kinds: dict[str, str]) -> Mesh:
    """Every boundary segment given the condition of its patch.

    Raises:
        ValueError: a patch of the mesh has no condition in ``kinds``.
    """
    missing = [p for p in patches(mesh) if p not in kinds]
    if missing:
        raise ValueError(f"no boundary condition for patch(es) {missing}; the grid's patches are {patches(mesh)}")
    return tuple(replace(block, sides=block.sides.with_conditions(kinds)) for block in as_mesh(mesh))


def side_nodes(grid: Grid, side: str, segment: Boundary | Interface) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates along a segment of a side, ``(n + 1,)`` each, in the
    side's running index."""
    lo, hi = segment.start, segment.stop + 1
    index = {"j_min": (slice(lo, hi), 0), "j_max": (slice(lo, hi), -1),
             "i_min": (0, slice(lo, hi)), "i_max": (-1, slice(lo, hi))}[side]
    return np.asarray(grid.x_node)[index], np.asarray(grid.y_node)[index]


def interface_node_mismatch(mesh: Grid | Mesh) -> float:
    """Largest distance between nodes an interface pairs: zero for a
    point-matched mesh, the file precision for one read from disk."""
    mesh = as_mesh(mesh)
    worst = 0.0
    for grid in mesh:
        for side in SIDES:
            for seg in grid.sides.side(side):
                if not isinstance(seg, Interface):
                    continue
                xa, ya = side_nodes(grid, side, seg)
                span = Boundary("", seg.partner_start, seg.partner_start + seg.stop - seg.start)
                xb, yb = side_nodes(mesh[seg.block], seg.side, span)
                if seg.reversed:
                    xb, yb = xb[::-1], yb[::-1]
                worst = max(worst, float(np.max(np.hypot(xa - xb, ya - yb))))
    return worst


@dataclass(frozen=True)
class WallFaces:
    """Every wall face, grouped into bodies and ordered round each.

    Faces run with the fluid on their left (``a -> b``), so a body's faces
    chain end to start: a closed loop round an airfoil, from its trailing
    edge (the node of largest ``x``), or an open chain (a plate). Bodies are
    numbered by leading-edge ``x``: slat 0, main 1, flap 2.
    """

    block: np.ndarray
    side: np.ndarray        # index into SIDES
    cell: np.ndarray        # position along the side
    body: np.ndarray
    a: np.ndarray           # (n, 2)
    b: np.ndarray           # (n, 2)

    @property
    def bodies(self) -> int:
        return int(self.body.max()) + 1 if self.body.size else 0

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.a + self.b)


def wall_faces(mesh: Grid | Mesh) -> WallFaces:
    """The wall faces of ``mesh``.

    Raises:
        ValueError: if two faces start at one node.
    """
    rows = []
    for k, grid in enumerate(as_mesh(mesh)):
        for side, seg in grid.sides.walls():
            x, y = side_nodes(grid, side, seg)
            p = np.stack([x, y], axis=1)
            forward = side in ("j_min", "i_max")        # fluid on the left of the running order
            for n in range(seg.stop - seg.start):
                a, b = (p[n], p[n + 1]) if forward else (p[n + 1], p[n])
                rows.append((k, SIDES.index(side), seg.start + n, *a, *b))
    if not rows:
        return WallFaces(*(np.zeros(0, int),) * 4, np.zeros((0, 2)), np.zeros((0, 2)))
    rows = np.array(rows)
    a, b = rows[:, 3:5], rows[:, 5:7]
    tol = 1e-3 * np.hypot(*(b - a).T).min()
    key = lambda p: tuple(np.round(p / tol).astype(np.int64))  # noqa: E731
    starts = {key(p): n for n, p in enumerate(a)}
    if len(starts) != len(a):
        raise ValueError("two wall faces start at the same node")
    following = np.array([starts.get(key(p), -1) for p in b])
    has_predecessor = np.zeros(len(a), dtype=bool)
    has_predecessor[following[following >= 0]] = True
    body = np.full(len(a), -1)
    loops = []
    for start in [*np.flatnonzero(~has_predecessor), *range(len(a))]:   # open walls first
        if body[start] >= 0:
            continue
        loop, face = [], start
        while face >= 0 and body[face] < 0:
            body[face] = len(loops)
            loop.append(face)
            face = following[face]
        if face == start:
            trailing = int(np.argmax(a[loop, 0]))
            loop = loop[trailing:] + loop[:trailing]
        elif face >= 0:
            raise ValueError("wall faces chain into one another inconsistently")
        loops.append(loop)
    loops.sort(key=lambda loop: a[loop, 0].min())
    index = np.concatenate(loops)
    for n, loop in enumerate(loops):
        body[loop] = n
    pick = lambda c: c[index]  # noqa: E731
    return WallFaces(pick(rows[:, 0].astype(int)), pick(rows[:, 1].astype(int)),
                     pick(rows[:, 2].astype(int)), pick(body), pick(a), pick(b))


def wall_distance(mesh: Grid | Mesh, chunk: int = 4096) -> list[np.ndarray]:
    """Exact distance from every cell centre to the nearest wall face, one
    ``(ni, nj)`` array per block: the distance to the clamped projection on
    each face ``a -> b``,

        t = clip((p - a).(b - a) / |b - a|^2, 0, 1),     d = |p - a - t (b - a)|,

    minimized over all faces (in chunks of cells, to bound memory). 3D:
    ``wall3d.wall_distance_3d``."""
    mesh = as_mesh(mesh)
    if mesh[0].dim == 3:
        from .wall3d import wall_distance_3d
        return wall_distance_3d(mesh)
    faces = wall_faces(mesh)
    if not faces.a.size:
        return [np.full((g.ni, g.nj), np.inf) for g in mesh]
    ax, ay = faces.a.T
    ex, ey = (faces.b - faces.a).T
    length2 = ex**2 + ey**2
    distances = []
    for block in mesh:
        px, py = np.asarray(block.xc, dtype=np.float64).ravel(), np.asarray(block.yc, dtype=np.float64).ravel()
        out = np.empty_like(px)
        for start in range(0, px.size, chunk):
            qx = px[start:start + chunk, None] - ax[None, :]
            qy = py[start:start + chunk, None] - ay[None, :]
            t = np.clip((qx * ex + qy * ey) / length2, 0.0, 1.0)
            out[start:start + chunk] = np.sqrt(np.min((qx - t * ex) ** 2 + (qy - t * ey) ** 2, axis=1))
        distances.append(out.reshape(block.ni, block.nj))
    return distances
