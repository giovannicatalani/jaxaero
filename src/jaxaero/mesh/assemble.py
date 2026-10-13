"""Any structured grid file to a mesh: blocks, interfaces, named patches.

``load_grid(path)`` reads a Plot3D or CGNS file (mesh/plot3d.py,
mesh/cgns.py) and assembles it the same way whatever its source:

1. every block made right-handed (``i`` reversed if its areas are negative);
2. coincident faces matched into interfaces -- between blocks, or within one
   (an O-grid's seam, a C-grid's wake cut) -- never across a face the file
   declares a boundary; a file's 1-to-1 interfaces must agree;
3. every other face named: by the file (CGNS families and BC types), else by
   the case's patch rules (``PatchRule``: a name for the faces inside a box),
   else by the boundary loop it lies on. Walked with the fluid on the left,
   the outer boundary runs counter-clockwise and every body clockwise: faces
   of a counter-clockwise loop are ``farfield``, of a clockwise one ``wall``.

Faces coincide when both end nodes do, within ``pair_tol`` of the smaller of
the face's length and its cell's width across it (so a thin cell's two faces
never pass for one another).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .block import SIDES, Boundary, BlockSides, Grid, Interface, cell_widths, validate_metrics
from .cgns import RawBlock, is_cgns, read_cgns
from .mesh import Mesh, assign_conditions, interface_node_mismatch, patches, side_nodes, wall_faces
from .plot3d import read_plot3d

FORMATS = ("auto", "plot3d", "cgns")


@dataclass(frozen=True)
class PatchRule:
    """Boundary faces whose midpoints lie inside the box get the patch ``name``;
    a missing bound is unbounded (a rule without bounds takes every face left).
    ``z`` bounds apply to 3D grids only."""

    name: str
    x_min: float = -np.inf
    x_max: float = np.inf
    y_min: float = -np.inf
    y_max: float = np.inf
    z_min: float = -np.inf
    z_max: float = np.inf

    def contains(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        inside = (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)
        return inside if z is None else inside & (z >= self.z_min) & (z <= self.z_max)


@dataclass(frozen=True)
class MeshInfo:
    """What a loaded grid is, for the log and the provenance record."""

    source: str
    blocks: int
    cells: int
    interfaces: int
    patch_faces: dict[str, int]
    bodies: int
    far_field_radius: float
    min_wall_spacing: float
    interface_mismatch: float

    def summary(self) -> str:
        names = ", ".join(f"{k} ({v} faces)" for k, v in self.patch_faces.items())
        return (f"{Path(self.source).name}: {self.blocks} block(s), {self.cells} cells, "
                f"{self.interfaces} interface segment(s); patches {names}; {self.bodies} body(ies) "
                f"walled by 'wall'; outer boundary at ~{self.far_field_radius:.3g}; first cell "
                f">= {self.min_wall_spacing:.3e}; interface node mismatch {self.interface_mismatch:.1e}")


def load_grid(path: str | Path, format: str = "auto", coordinate_scale: float = 1.0,
              rules: tuple[PatchRule, ...] = (), dtype=np.float64,
              pair_tol: float = 0.05) -> tuple[Mesh, MeshInfo]:
    """Read and assemble a Plot3D or CGNS file; ``rules`` name the faces it does not,
    ``pair_tol`` is the face-coincidence tolerance (assemble)."""
    path = Path(path)
    if format not in FORMATS:
        raise ValueError(f"unknown grid format {format!r}; choose from {FORMATS}")
    if format == "cgns" or (format == "auto" and is_cgns(path)):
        raw = read_cgns(path)
    else:
        raw = [RawBlock(x, y) for x, y in read_plot3d(path)]
    raw = [RawBlock(b.x * coordinate_scale, b.y * coordinate_scale, b.declared, b.name) for b in raw]
    return assemble(raw, rules, dtype, pair_tol, source=str(path))


# --- Assembly --------------------------------------------------------------------------


def _right_handed(block: RawBlock) -> RawBlock:
    """``i`` reversed (and the declared faces with it) if the areas are negative."""
    x, y = block.x, block.y
    d1x, d1y = x[1:, 1:] - x[:-1, :-1], y[1:, 1:] - y[:-1, :-1]
    d2x, d2y = x[:-1, 1:] - x[1:, :-1], y[:-1, 1:] - y[1:, :-1]
    area = 0.5 * (d1x * d2y - d1y * d2x)
    if np.all(area > 0.0):
        return block
    if not np.all(area < 0.0):
        raise ValueError(f"block {block.name or ''} of {x.shape} nodes folds over: cell areas of both signs")
    ni = x.shape[0] - 1
    flip = {"i_min": "i_max", "i_max": "i_min"}
    declared = tuple(type(d)(flip.get(d.side, d.side),
                             *( (ni - d.stop, ni - d.start) if d.side.startswith("j") else (d.start, d.stop)),
                             d.name, d.donor) for d in block.declared)
    return RawBlock(x[::-1].copy(), y[::-1].copy(), declared, block.name)


@dataclass(frozen=True)
class _Faces:
    """Every face of every block side, flattened."""

    block: np.ndarray
    side: np.ndarray        # index into SIDES
    cell: np.ndarray        # position along the side
    a: np.ndarray           # (n, 2) first node, in the side's running order
    b: np.ndarray
    width: np.ndarray       # the cell's extent across the face

    @property
    def length(self) -> np.ndarray:
        return np.hypot(*(self.b - self.a).T)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.a + self.b)


def _nodes(x, y, side):
    index = {"j_min": (slice(None), 0), "j_max": (slice(None), -1),
             "i_min": (0, slice(None)), "i_max": (-1, slice(None))}[side]
    return x[index], y[index]


def _faces(blocks: list[RawBlock]) -> _Faces:
    parts = {k: [] for k in ("block", "side", "cell", "a", "b", "width")}
    for k, block in enumerate(blocks):
        width_i, width_j = cell_widths(block.x, block.y)
        across = {"j_min": width_j[:, 0], "j_max": width_j[:, -1],
                  "i_min": width_i[0, :], "i_max": width_i[-1, :]}
        for s, side in enumerate(SIDES):
            sx, sy = _nodes(block.x, block.y, side)
            n = sx.size - 1
            parts["block"].append(np.full(n, k))
            parts["side"].append(np.full(n, s))
            parts["cell"].append(np.arange(n))
            parts["a"].append(np.stack([sx[:-1], sy[:-1]], axis=1))
            parts["b"].append(np.stack([sx[1:], sy[1:]], axis=1))
            parts["width"].append(across[side])
    return _Faces(*(np.concatenate(parts[k]) for k in ("block", "side", "cell", "a", "b", "width")))


def _match(faces: _Faces, pair_tol: float, excluded: np.ndarray):
    """For every face, the coincident face (-1: none) and whether it runs the
    other way. ``excluded`` faces (declared boundaries) match nothing."""
    mid, tol = faces.mid, pair_tol * np.minimum(faces.length, faces.width)
    order = np.argsort(mid[:, 0])
    xs = mid[order, 0]
    lo = np.searchsorted(xs, mid[:, 0] - tol, side="left")
    hi = np.searchsorted(xs, mid[:, 0] + tol, side="right")
    me = np.repeat(np.arange(len(mid)), hi - lo)
    other = order[np.concatenate([np.arange(l, h) for l, h in zip(lo, hi)])]
    keep = (other != me) & ~excluded[me] & ~excluded[other]
    me, other = me[keep], other[keep]

    def close(p, q):
        return np.hypot(*(p - q).T) <= tol[me]

    same = close(faces.a[me], faces.a[other]) & close(faces.b[me], faces.b[other])
    flipped = close(faces.a[me], faces.b[other]) & close(faces.b[me], faces.a[other])
    hit = same | flipped
    me, other, flipped = me[hit], other[hit], flipped[hit]
    if np.unique(me).size != me.size:
        raise ValueError("a face coincides with more than one other face")
    partner = np.full(len(mid), -1)
    reverse = np.zeros(len(mid), dtype=bool)
    partner[me], reverse[me] = other, flipped
    return partner, reverse


def _segments(targets: list) -> tuple:
    """Per-cell targets along a side -> segments. A target is a patch name,
    or ``(block, side, cell, reversed)``."""
    segments, k = [], 0
    while k < len(targets):
        t = targets[k]
        stop = k + 1
        if isinstance(t, str):
            while stop < len(targets) and targets[stop] == t:
                stop += 1
            segments.append(Boundary(t, k, stop))
        else:
            block, side, cell, rev = t
            step = -1 if rev else 1
            while (stop < len(targets) and not isinstance(targets[stop], str)
                   and targets[stop][:2] == (block, side) and targets[stop][3] == rev
                   and targets[stop][2] == targets[stop - 1][2] + step):
                stop += 1
            last = targets[stop - 1][2]
            segments.append(Interface(k, stop, block, side, min(cell, last), rev))
        k = stop
    return tuple(segments)


def assemble(raw: list[RawBlock], rules: tuple[PatchRule, ...] = (), dtype=np.float64,
             pair_tol: float = 0.05, source: str = "") -> tuple[Mesh, MeshInfo]:
    """Blocks as read -> a mesh with interfaces and named patches."""
    blocks = [_right_handed(b) for b in raw]
    faces = _faces(blocks)

    def rows_of(k: int, side: str):
        rows = np.flatnonzero((faces.block == k) & (faces.side == SIDES.index(side)))
        return rows[np.argsort(faces.cell[rows])]

    declared_name = np.full(len(faces.block), None, dtype=object)
    donors = []
    for k, block in enumerate(blocks):
        for d in block.declared:
            rows = rows_of(k, d.side)[d.start:d.stop]
            if d.name is not None:
                declared_name[rows] = d.name
            elif d.donor is not None:
                donors.append((rows, d.donor))
    partner, reverse = _match(faces, pair_tol, declared_name != None)  # noqa: E711
    for rows, donor in donors:
        if np.any(partner[rows] < 0) or np.any(faces.block[partner[rows]] != donor):
            raise ValueError(f"a 1-to-1 interface the file declares is not matched by coincident faces "
                             f"to its donor zone {donor}")

    nodes = np.concatenate([np.stack([b.x.ravel(), b.y.ravel()], axis=1) for b in blocks])
    centre = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0))
    radius = float(np.hypot(*(nodes - centre).T).max())
    names = _by_loop(faces, partner < 0)
    named = np.zeros(len(names), dtype=bool)
    for rule in rules:
        hit = ~named & rule.contains(faces.mid[:, 0], faces.mid[:, 1])
        names[hit], named[hit] = rule.name, True
    names[declared_name != None] = declared_name[declared_name != None]  # noqa: E711
    names[partner >= 0] = None

    mesh, interfaces = [], 0
    for k, block in enumerate(blocks):
        spec = {}
        for side in SIDES:
            targets = [names[r] if partner[r] < 0 else
                       (int(faces.block[partner[r]]), SIDES[faces.side[partner[r]]],
                        int(faces.cell[partner[r]]), bool(reverse[r])) for r in rows_of(k, side)]
            spec[side] = _segments(targets)
            interfaces += sum(isinstance(seg, Interface) for seg in spec[side])
        grid = Grid.from_nodes(block.x, block.y, BlockSides(block.x.shape[0] - 1, block.x.shape[1] - 1, **spec),
                               dtype=dtype)
        report = validate_metrics(grid)
        if not (report.min_area > 0.0 and report.max_normal_closure < 1e-12):
            raise ValueError(f"block {k} fails the metric checks: {report.summary()}")
        mesh.append(grid)
    mesh = tuple(mesh)

    counts = {p: int(np.sum(names[partner < 0] == p)) for p in patches(mesh)}
    walled = assign_conditions(mesh, {p: "no_slip_wall" if p == "wall" else "far_field" for p in patches(mesh)})
    info = MeshInfo(source, len(mesh), sum(g.ni * g.nj for g in mesh), interfaces, counts,
                    wall_faces(walled).bodies, radius, _min_wall_spacing(walled),
                    interface_node_mismatch(mesh))
    return mesh, info


def _by_loop(faces: _Faces, boundary: np.ndarray) -> np.ndarray:
    """``farfield`` or ``wall`` for every boundary face, by the orientation of
    the loop it lies on (counter-clockwise: outer; clockwise: a body), its
    faces walked with the fluid on their left. A face on no closed loop is
    ``wall``."""
    names = np.full(len(boundary), None, dtype=object)
    rows = np.flatnonzero(boundary)
    if not rows.size:
        return names
    forward = np.isin(faces.side[rows], (SIDES.index("j_min"), SIDES.index("i_max")))
    a = np.where(forward[:, None], faces.a[rows], faces.b[rows])
    b = np.where(forward[:, None], faces.b[rows], faces.a[rows])
    tol = 1e-3 * np.hypot(*(b - a).T).min()
    key = lambda p: tuple(np.round(p / tol).astype(np.int64))  # noqa: E731
    starts = {key(p): n for n, p in enumerate(a)}
    following = np.array([starts.get(key(p), -1) for p in b])
    seen = np.zeros(len(rows), dtype=bool)
    for start in range(len(rows)):
        if seen[start]:
            continue
        loop, face = [], start
        while face >= 0 and not seen[face]:
            seen[face] = True
            loop.append(face)
            face = following[face]
        closed = face == start
        area = 0.5 * np.sum(a[loop, 0] * b[loop, 1] - b[loop, 0] * a[loop, 1])
        names[rows[loop]] = "farfield" if closed and area > 0.0 else "wall"
    return names


def _min_wall_spacing(mesh: Mesh) -> float:
    """Twice the smallest distance from a wall face to the centre of the cell on it."""
    best = np.inf
    first = {"j_min": (slice(None), 0), "j_max": (slice(None), -1),
             "i_min": (0, slice(None)), "i_max": (-1, slice(None))}
    for grid in mesh:
        xc, yc = np.asarray(grid.xc), np.asarray(grid.yc)
        for side, seg in grid.sides.walls():
            x, y = side_nodes(grid, side, seg)
            cells = slice(seg.start, seg.stop)
            d = np.hypot(xc[first[side]][cells] - 0.5 * (x[:-1] + x[1:]),
                         yc[first[side]][cells] - 0.5 * (y[:-1] + y[1:]))
            best = min(best, float(2.0 * d.min()))
    return best
