"""The atlas: every block of a mesh, with its ghost ring, packed into one canvas.

Block ``b``'s cell ``(i, j)``, ghosts included (``-g <= i < ni + g``), sits at
canvas cell ``(oi_b + g + i, oj_b + g + j)``. The solver runs the single-block
discretization once over the whole canvas, so its compiled size does not grow
with the number of blocks. Canvas cells are *interior* (a block cell), *ghost*
(a block's ghost ring) or *gap* (unused).

Everything here is static and built once in NumPy from the blocks' side
segments:

* the ghost fill as two gather stages (``j`` sides, then ``i`` sides with the
  corners), applied by ``numerics.boundary.fill_ghosts``;
* the face-neighbour map behind ``face_averages``;
* the canvas geometry: every real face keeps its block's exact area vector;
* the line chains of the line-implicit preconditioner: cells linked along
  ``j``, continued through a block's own ``j``-side cut (the C-grid wake);
* the wall faces, grouped into bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
from jax import Array

from .block import OUTWARD, SIDES, WALL_KINDS, Grid, Interface, rectangle_sides
from .mesh import Mesh, WallFaces, as_mesh, wall_faces


def interior_cell(side: str, a: int, layer: int, ni: int, nj: int) -> tuple[int, int]:
    """Block cell ``layer`` cells inward from ``side``, at position ``a`` along it."""
    return {"j_min": (a, layer), "j_max": (a, nj - 1 - layer),
            "i_min": (layer, a), "i_max": (ni - 1 - layer, a)}[side]


def ghost_cell(side: str, a: int, layer: int, ni: int, nj: int) -> tuple[int, int]:
    """Block ghost cell ``layer`` cells outward from ``side``, at position ``a``."""
    return {"j_min": (a, -1 - layer), "j_max": (a, nj + layer),
            "i_min": (-1 - layer, a), "i_max": (ni + layer, a)}[side]


@dataclass(frozen=True)
class Stage:
    """One gather stage of the ghost fill.

    ``index`` maps every canvas cell to its source in ``[canvas | condition
    values]`` (the identity except at this stage's ghosts); ``conditions``
    holds, per boundary kind, the source cell of each ghost and the outward
    unit normal and midpoint of its face, ``(d, n)`` each.
    """

    index: np.ndarray
    conditions: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class Lines:
    """Line chains, ``(n_lines, length)``, padded with ``-1``.

    ``cells`` are flat canvas indices. Between chain positions ``k`` and
    ``k + 1``, row ``k`` couples across ``up_face`` (flat index into the
    canvas ``j``-face array) with outward sign ``up_sign``, and row ``k + 1``
    across ``down_face`` with ``down_sign``: each row uses its own cell's face
    (the same face inside a block, the two sides of an interface across one).
    Padded links have sign 0.
    """

    cells: np.ndarray
    up_face: np.ndarray
    up_sign: np.ndarray
    down_face: np.ndarray
    down_sign: np.ndarray


@dataclass(frozen=True)
class Walls:
    """Every wall face located on the canvas.

    ``cell``: canvas index of the interior cell on the face; ``family``: the
    index direction the face is normal to (0 = i, 1 = j, 2 = k); ``face``: its
    flat index in the canvas face array of its family; ``into_fluid``: ``+1``
    where the canvas area vector points into the fluid (``*_min`` walls),
    ``-1`` otherwise.
    """

    faces: WallFaces
    cell: np.ndarray
    family: np.ndarray
    face: np.ndarray
    into_fluid: np.ndarray


@dataclass(frozen=True)
class Atlas:
    """A mesh packed into one canvas. See the module docstring."""

    mesh: Mesh
    g: int
    origins: tuple[tuple[int, int], ...]
    windows: tuple[tuple[slice, slice], ...]
    grid: Grid
    interior: np.ndarray
    centres: Array
    stages: tuple[Stage, Stage]
    neighbours: np.ndarray
    lines: Lines
    walls: Walls
    quarter_chord: tuple[float, float]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.interior.shape

    @property
    def dim(self) -> int:
        return self.interior.ndim

    @property
    def ni(self) -> int:
        return self.shape[0]

    @property
    def nj(self) -> int:
        return self.shape[1]

    @property
    def cells(self) -> int:
        return int(self.interior.sum())

    def pack(self, blocks, fill=0.0) -> Array:
        """Per-block fields ``(..., *block shape)`` into one canvas; ``fill``
        (scalar, or one value per leading entry) elsewhere."""
        lead = blocks[0].shape[:blocks[0].ndim - self.dim]
        fill = jnp.asarray(fill, dtype=blocks[0].dtype)
        canvas = jnp.broadcast_to(fill.reshape(fill.shape + (1,) * self.dim), lead + self.shape)
        for block, window in zip(blocks, self.windows):
            canvas = canvas.at[(Ellipsis,) + tuple(window)].set(block)
        return canvas

    def unpack(self, canvas: Array) -> tuple[Array, ...]:
        return tuple(canvas[(Ellipsis,) + tuple(window)] for window in self.windows)

    def wall_values(self, fields: tuple[Array, ...]) -> Array:
        """Per-family canvas face fields ``(..., *face shape)`` at every wall
        face, ``(..., n)`` in ``walls`` order."""
        W, out = self.walls, None
        for a, f in enumerate(fields):
            flat = f.reshape(f.shape[:f.ndim - self.dim] + (-1,))
            picked = flat[..., np.where(W.family == a, W.face, 0)]
            out = picked if out is None else jnp.where(W.family == a, picked, out)
        return out

    def face_averages(self, a: Array) -> tuple[Array, ...]:
        """Mean of the two cells either side of every canvas face, per
        family, for ``(..., *canvas shape)``: the neighbour across a block
        face is the partner cell, or the cell itself at a physical boundary."""
        lead = a.shape[:a.ndim - self.dim]
        filled = a.reshape(lead + (-1,))[..., self.neighbours].reshape(a.shape)
        out = []
        for axis in range(self.dim):
            widths = [(0, 0)] * filled.ndim
            widths[len(lead) + axis] = (1, 1)
            f = jnp.pad(filled, widths, mode="edge")
            lo, hi = [slice(None)] * f.ndim, [slice(None)] * f.ndim
            lo[len(lead) + axis], hi[len(lead) + axis] = slice(0, -1), slice(1, None)
            out.append(0.5 * (f[tuple(lo)] + f[tuple(hi)]))
        return tuple(out)


# --- Building ---------------------------------------------------------------------


def build_atlas(mesh: Mesh | Grid, g: int = 2) -> Atlas:
    """Pack ``mesh`` (or a single block), its patches' conditions assigned
    (``mesh.assign_conditions``), and build every static plan."""
    mesh = as_mesh(mesh)
    if mesh[0].dim == 3:
        from .atlas3d import build_atlas_3d
        return build_atlas_3d(mesh, g)
    unset = sorted({seg.patch for block in mesh for _, seg in block.sides.boundaries() if seg.kind is None})
    if unset:
        raise ValueError(f"patch(es) {unset} have no boundary condition; use mesh.assign_conditions")
    shapes = [(b.ni + 2 * g, b.nj + 2 * g) for b in mesh]
    origins, (NI, NJ) = _pack(shapes)

    def pos(b: int, i: int, j: int) -> int:
        oi, oj = origins[b]
        return (oi + g + i) * NJ + (oj + g + j)

    interior = np.zeros((NI, NJ), dtype=bool)
    windows = []
    for b, block in enumerate(mesh):
        oi, oj = origins[b]
        window = (slice(oi + g, oi + g + block.ni), slice(oj + g, oj + g + block.nj))
        interior[window] = True
        windows.append(window)

    grid, xc, yc = _canvas_geometry(mesh, origins, shapes, windows, g, NI, NJ)
    centres = np.stack([xc.ravel(), yc.ravel()])
    for stage in (_stage(mesh, 1, ("j_min", "j_max"), pos, NI * NJ, pad=False),
                  _stage(mesh, 1, ("i_min", "i_max"), pos, NI * NJ, pad=True)):
        centres = apply_reflections(stage, centres)
    centres = centres.reshape(2, NI, NJ)
    grid = replace(grid, centres=jnp.asarray(centres))
    padded_centres = jnp.asarray(np.stack([np.pad(c, 1, mode="edge") for c in centres]))

    stages = (_stage(mesh, g, ("j_min", "j_max"), pos, NI * NJ, pad=False),
              _stage(mesh, g, ("i_min", "i_max"), pos, NI * NJ, pad=True))
    walls = _walls(mesh, pos, origins, g, NI, NJ)
    return Atlas(tuple(mesh), g, tuple(origins), tuple(windows), grid, interior, padded_centres,
                 stages, _neighbour_map(mesh, pos, NI * NJ), _lines(mesh, pos, origins, g, NJ),
                 walls, _quarter_chord(walls.faces))


def _pack(shapes):
    """Tiles, tallest first, stacked in columns of a strip as tall as the
    tallest tile; a column is as wide as its first tile."""
    height = max(h for _, h in shapes)
    order = sorted(range(len(shapes)), key=lambda k: (-shapes[k][1], -shapes[k][0]))
    origins = [None] * len(shapes)
    columns: list[list[int]] = []          # [x, width, used height]
    for k in order:
        w, h = shapes[k]
        for col in columns:
            if w <= col[1] and col[2] + h <= height:
                origins[k] = (col[0], col[2])
                col[2] += h
                break
        else:
            x = columns[-1][0] + columns[-1][1] if columns else 0
            columns.append([x, w, h])
            origins[k] = (x, 0)
    return origins, (columns[-1][0] + columns[-1][1], height)


def _extended_nodes(x, y, g: int):
    """Nodes extended by ``g`` linearly extrapolated layers on every side: the
    ghost cells get a finite geometry, the real nodes are untouched."""

    def extend(a, axis):
        a = np.moveaxis(a, axis, 0)
        lo = [a[0] - (k + 1) * (a[1] - a[0]) for k in range(g)][::-1]
        hi = [a[-1] + (k + 1) * (a[-1] - a[-2]) for k in range(g)]
        return np.moveaxis(np.concatenate([np.stack(lo), a, np.stack(hi)]), 0, axis)

    return tuple(extend(extend(np.asarray(c, dtype=np.float64), 0), 1) for c in (x, y))


def _canvas_geometry(mesh, origins, shapes, windows, g, NI, NJ):
    area = np.ones((NI, NJ))
    li, lj = np.meshgrid(np.arange(NI), np.arange(NJ), indexing="ij")
    xc, yc = 1e3 + 10.0 * li, 1e3 + 10.0 * lj           # gaps: a lattice, never used
    sx_i, sy_i = np.zeros((NI + 1, NJ)), np.zeros((NI + 1, NJ))
    sx_j, sy_j = np.zeros((NI, NJ + 1)), np.zeros((NI, NJ + 1))
    for b, block in enumerate(mesh):
        oi, oj = origins[b]
        ti, tj = shapes[b]
        tile = Grid.from_nodes(*_extended_nodes(block.x_node, block.y_node, g),
                               rectangle_sides(ti, tj, **dict.fromkeys(SIDES, "tile")))
        a = np.asarray(tile.area)
        area[oi:oi + ti, oj:oj + tj] = np.where(np.abs(a) > 1e-300, a, 1.0)
        xc[oi:oi + ti, oj:oj + tj], yc[oi:oi + ti, oj:oj + tj] = np.asarray(tile.xc), np.asarray(tile.yc)
        sx_i[oi:oi + ti + 1, oj:oj + tj], sy_i[oi:oi + ti + 1, oj:oj + tj] = np.asarray(tile.sx_i), np.asarray(tile.sy_i)
        sx_j[oi:oi + ti, oj:oj + tj + 1], sy_j[oi:oi + ti, oj:oj + tj + 1] = np.asarray(tile.sx_j), np.asarray(tile.sy_j)
        wi, wj = windows[b]
        area[wi, wj] = np.asarray(block.area)
        sx_i[wi.start:wi.stop + 1, wj], sy_i[wi.start:wi.stop + 1, wj] = np.asarray(block.sx_i), np.asarray(block.sy_i)
        sx_j[wi, wj.start:wj.stop + 1], sy_j[wi, wj.start:wj.stop + 1] = np.asarray(block.sx_j), np.asarray(block.sy_j)
        xc[wi, wj], yc[wi, wj] = np.asarray(block.xc), np.asarray(block.yc)
    grid = Grid(jnp.zeros((2, NI + 1, NJ + 1)), None, jnp.asarray(area),
                (jnp.asarray(np.stack([sx_i, sy_i])), jnp.asarray(np.stack([sx_j, sy_j]))),
                rectangle_sides(NI, NJ, **dict.fromkeys(SIDES, "canvas")))
    return grid, xc, yc


def _face_geometry(block: Grid, side: str):
    """Outward unit normals and midpoints of a side's faces."""
    if side.startswith("j"):
        j = 0 if side == "j_min" else block.nj
        sx, sy = np.asarray(block.sx_j)[:, j], np.asarray(block.sy_j)[:, j]
        x, y = np.asarray(block.x_node)[:, j], np.asarray(block.y_node)[:, j]
    else:
        i = 0 if side == "i_min" else block.ni
        sx, sy = np.asarray(block.sx_i)[i, :], np.asarray(block.sy_i)[i, :]
        x, y = np.asarray(block.x_node)[i, :], np.asarray(block.y_node)[i, :]
    length = np.hypot(sx, sy)
    sign = OUTWARD[side]
    return sign * sx / length, sign * sy / length, 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])


def _partner_along(seg: Interface, a: int) -> int:
    offset = a - seg.start
    return seg.partner_start + (seg.stop - seg.start - 1 - offset if seg.reversed else offset)


def _stage(mesh: Mesh, g: int, sides, pos, n_canvas: int, pad: bool) -> Stage:
    """The gather plan of one fill stage. On the ``i`` sides (``pad``) each
    side extends ``g`` cells into the ``j`` ghost rows filled by stage 1, so
    the corners are the neighbour's (or the condition applied to the corner)."""
    index = np.arange(n_canvas)
    entries: dict[str, list] = {}
    for b, block in enumerate(mesh):
        ni, nj = block.ni, block.nj
        for side in sides:
            segments = block.sides.side(side)
            n = block.sides.length(side)
            ext = g if pad else 0
            geometry = _face_geometry(block, side)
            for k, seg in enumerate(segments):
                lo = seg.start - (ext if k == 0 else 0)
                hi = seg.stop + (ext if k == len(segments) - 1 else 0)
                for a in range(lo, hi):
                    for layer in range(g):
                        ghost = pos(b, *ghost_cell(side, a, layer, ni, nj))
                        if isinstance(seg, Interface):
                            partner = mesh[seg.block]
                            span = partner.sides.length(seg.side)
                            corner = g if (pad and seg.side.startswith("i")) else 0
                            p = min(max(_partner_along(seg, a), -corner), span - 1 + corner)
                            index[ghost] = pos(seg.block, *interior_cell(seg.side, p, layer, partner.ni, partner.nj))
                        else:
                            read = layer if seg.kind in (*WALL_KINDS, "symmetry") else 0
                            source = pos(b, *interior_cell(side, a, read, ni, nj))
                            face = min(max(a, 0), n - 1)
                            entries.setdefault(seg.kind, []).append((ghost, source, *(c[face] for c in geometry)))
    conditions, offset = [], n_canvas
    for kind, rows in entries.items():
        rows = np.array(rows)
        index[rows[:, 0].astype(np.int64)] = offset + np.arange(len(rows))
        offset += len(rows)
        conditions.append((kind, rows[:, 1].astype(np.int64), rows[:, 2:4].T.copy(), rows[:, 4:6].T.copy()))
    return Stage(index, tuple(conditions))


def apply_reflections(stage: Stage, centres: np.ndarray) -> np.ndarray:
    """A stage applied to cell centres ``(d, n)``: copies across interfaces,
    reflections through every physical face."""
    parts = [centres]
    for _, source, normal, midpoint in stage.conditions:
        x = centres[:, source]
        d = np.sum((x - midpoint) * normal, axis=0)
        parts.append(x - 2.0 * d * normal)
    return np.concatenate(parts, axis=1)[:, stage.index]


def _neighbour_map(mesh: Mesh, pos, n_canvas: int) -> np.ndarray:
    """Across every block face, the cell on the other side, or the cell itself."""
    index = np.arange(n_canvas)
    for b, block in enumerate(mesh):
        for side in SIDES:
            for seg in block.sides.side(side):
                for a in range(seg.start, seg.stop):
                    ghost = pos(b, *ghost_cell(side, a, 0, block.ni, block.nj))
                    if isinstance(seg, Interface):
                        partner = mesh[seg.block]
                        p = _partner_along(seg, a)
                        index[ghost] = pos(seg.block, *interior_cell(seg.side, p, 0, partner.ni, partner.nj))
                    else:
                        index[ghost] = pos(b, *interior_cell(side, a, 0, block.ni, block.nj))
    return index


def _lines(mesh: Mesh, pos, origins, g: int, NJ: int) -> Lines:
    """Chains of cells linked along ``j``, through a block's own ``j``-side cuts.

    Every cell has at most one neighbour below and one above; following those
    links gives paths (a cycle, which no grid here has, is cut once). Each
    link records, for both its cells, the canvas ``j``-face the cell couples
    across and the sign that makes its area vector point away from the cell.
    """

    def jface(b: int, i: int, j: int) -> int:
        """Flat index of the canvas j-face below block cell (i, j)."""
        oi, oj = origins[b]
        return (oi + g + i) * (NJ + 1) + (oj + g + j)

    # neighbour[(b, i, j)] = {"down"/"up": ((b', i', j'), my face, my sign)}
    links: dict[tuple, dict[str, tuple]] = {}

    def link(cell, direction, other, face, sign):
        links.setdefault(cell, {})[direction] = (other, face, sign)

    for b, block in enumerate(mesh):
        ni, nj = block.ni, block.nj
        for i in range(ni):
            for j in range(nj - 1):
                link((b, i, j), "up", (b, i, j + 1), jface(b, i, j + 1), 1)
                link((b, i, j + 1), "down", (b, i, j), jface(b, i, j + 1), -1)
        for side, j, direction, face_j, sign in (("j_min", 0, "down", 0, -1), ("j_max", nj - 1, "up", nj, 1)):
            for seg in block.sides.side(side):
                # Through a block's own j-side cut only (the C-grid wake); into
                # other blocks cost more than it gained (docs/dev/notes.md).
                if not isinstance(seg, Interface) or not seg.side.startswith("j") or seg.block != b:
                    continue
                partner = mesh[seg.block]
                for a in range(seg.start, seg.stop):
                    p = _partner_along(seg, a)
                    pj = 0 if seg.side == "j_min" else partner.nj - 1
                    link((b, a, j), direction, (seg.block, p, pj), jface(b, a, face_j), sign)

    def neighbours(cell):
        return [v[0] for v in links.get(cell, {}).values()]

    seen: set = set()
    chains: list[list[tuple]] = []
    cells = [(b, i, j) for b, block in enumerate(mesh) for i in range(block.ni) for j in range(block.nj)]
    ends = [c for c in cells if len(neighbours(c)) < 2]
    for start in ends + cells:                      # paths from their ends; then any cycle
        if start in seen:
            continue
        chain, previous, cell = [], None, start
        while cell is not None and cell not in seen:
            seen.add(cell)
            chain.append(cell)
            nxt = [c for c in neighbours(cell) if c != previous and c not in seen]
            previous, cell = cell, (nxt[0] if nxt else None)
        chains.append(chain)

    length = max(len(c) for c in chains)
    shape = (len(chains), length)
    out = {k: np.full(shape, fill, dtype=np.int64) for k, fill in
           (("cells", -1), ("up_face", 0), ("up_sign", 0), ("down_face", 0), ("down_sign", 0))}
    for n, chain in enumerate(chains):
        for k, cell in enumerate(chain):
            out["cells"][n, k] = pos(*cell)
            if k + 1 < len(chain):
                nxt = chain[k + 1]
                for direction, record in links[cell].items():
                    if record[0] == nxt:
                        out["up_face"][n, k], out["up_sign"][n, k] = record[1], record[2]
                for direction, record in links[nxt].items():
                    if record[0] == cell:
                        out["down_face"][n, k], out["down_sign"][n, k] = record[1], record[2]
    return Lines(out["cells"], out["up_face"], out["up_sign"], out["down_face"], out["down_sign"])


def _walls(mesh: Mesh, pos, origins, g: int, NI: int, NJ: int) -> Walls:
    faces = wall_faces(mesh)
    cell, family, face, sign = [], [], [], []
    for b, s, a in zip(faces.block, faces.side, faces.cell):
        block, side = mesh[b], SIDES[s]
        i, j = interior_cell(side, a, 0, block.ni, block.nj)
        cell.append(pos(b, i, j))
        ci, cj = origins[b][0] + g + i, origins[b][1] + g + j
        if side.startswith("j"):
            family.append(1)
            face.append(ci * (NJ + 1) + cj + (side == "j_max"))
        else:
            family.append(0)
            face.append((ci + (side == "i_max")) * NJ + cj)
        sign.append(1.0 if side in ("j_min", "i_min") else -1.0)
    return Walls(faces, np.array(cell, dtype=np.int64), np.array(family, dtype=np.int64),
                 np.array(face, dtype=np.int64), np.array(sign))


def _quarter_chord(faces: WallFaces) -> tuple[float, float]:
    """A quarter of the way from the leading to the trailing edge of the body
    of longest chord: the trailing edge is where its wall loop starts (the
    node of largest ``x``), the leading edge the node farthest from it. The
    point-vortex far field is centred here."""
    if faces.bodies == 0:
        return (0.25, 0.0)
    best = None
    for k in range(faces.bodies):
        nodes = faces.a[faces.body == k]
        te = nodes[0]
        le = nodes[np.argmax(np.hypot(*(nodes - te).T))]
        chord = np.hypot(*(te - le))
        if best is None or chord > best[0]:
            best = (chord, le[0] + 0.25 * (te[0] - le[0]), le[1] + 0.25 * (te[1] - le[1]))
    return float(best[1]), float(best[2])
