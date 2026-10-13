"""The atlas of a single 3D block: the block with its ghost ring as the canvas.

Builds the same static plans as the 2D atlas (mesh/atlas.py), in NumPy, so
everything downstream -- the ghost fill, face averages, the line
preconditioner, wall loads -- runs on it unchanged:

* the ghost fill as three gather stages: the ``j`` sides; the ``i`` sides,
  extended over the ``j`` ghosts; the ``k`` sides, extended over both, so edge
  and corner ghosts come from the neighbour (or the condition) as in 2D;
* ghost centres: reflections through physical faces, copies across
  interfaces (the pole's ghost is the opposite cell, centre included);
* lines along ``j`` (wall to far field) for every ``(i, k)``;
* wall faces with their family, canvas face index and orientation.

One block for now (the TMR wing grids are one block); several blocks would
need a 3D packing, which nothing here requires yet.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .atlas import Atlas, Lines, Stage, Walls, apply_reflections
from .block import WALL_KINDS, Grid, metrics
from .block3d import SIDES_3D, InterfaceRect, box_sides, normal_axis, running_axes
from .wall3d import wall_quads

STAGES = (("j_min", "j_max"), ("i_min", "i_max"), ("k_min", "k_max"))


def cell_on(side: str, a, layer: int, shape, inward: bool) -> tuple[int, int, int]:
    """Block cell ``layer`` cells inward from (or, ``inward=False``, outward
    beyond) ``side`` at running position ``a``."""
    n = normal_axis(side)
    out = [0, 0, 0]
    for m, axis in enumerate(running_axes(side)):
        out[axis] = a[m]
    low = side.endswith("min")
    if inward:
        out[n] = layer if low else shape[n] - 1 - layer
    else:
        out[n] = -1 - layer if low else shape[n] + layer
    return tuple(out)


def _extended_nodes(nodes: np.ndarray, g: int) -> np.ndarray:
    """Nodes extended by ``g`` linearly extrapolated layers on every side."""
    for axis in range(1, 4):
        a = np.moveaxis(nodes, axis, 0)
        lo = [a[0] - (k + 1) * (a[1] - a[0]) for k in range(g)][::-1]
        hi = [a[-1] + (k + 1) * (a[-1] - a[-2]) for k in range(g)]
        nodes = np.moveaxis(np.concatenate([np.stack(lo), a, np.stack(hi)]), 0, axis)
    return nodes


def _face_geometry(block: Grid, side: str):
    """Outward unit normals and midpoints ``(3, na, nb)`` of a side's faces."""
    n = normal_axis(side)
    index = 0 if side.endswith("min") else block.shape[n]
    S = np.take(np.asarray(block.faces[n]), index, axis=1 + n)
    nodes = np.take(np.asarray(block.nodes), index, axis=1 + n)
    length = np.linalg.norm(S, axis=0)
    sign = -1.0 if side.endswith("min") else 1.0
    normal = np.where(length > 0.0, sign * S / np.where(length > 0.0, length, 1.0), 0.0)
    mid = 0.25 * (nodes[:, :-1, :-1] + nodes[:, 1:, :-1] + nodes[:, :-1, 1:] + nodes[:, 1:, 1:])
    return normal, mid


def build_atlas_3d(mesh, g: int = 2) -> Atlas:
    """The atlas of a one-block 3D mesh whose patches have conditions assigned."""
    if len(mesh) != 1:
        raise ValueError(f"3D grids are single-block for now; this one has {len(mesh)} blocks")
    (block,) = mesh
    unset = sorted({r.patch for _, r in block.sides.boundaries() if r.kind is None})
    if unset:
        raise ValueError(f"patch(es) {unset} have no boundary condition; use mesh.assign_conditions")
    shape = block.shape
    canvas = tuple(n + 2 * g for n in shape)
    NI, NJ, NK = canvas
    n_canvas = NI * NJ * NK

    def pos(i, j, k) -> int:
        return ((g + i) * NJ + (g + j)) * NK + (g + k)

    window = tuple(slice(g, g + n) for n in shape)
    interior = np.zeros(canvas, dtype=bool)
    interior[window] = True

    # Geometry: extrapolated ghost layers, the block's own metrics inside.
    nodes = _extended_nodes(np.asarray(block.nodes, dtype=np.float64), g)
    centres, volume, faces = metrics(nodes)
    volume = np.where(np.abs(volume) > 1e-300, volume, 1.0)
    volume[window] = np.asarray(block.volume)

    handled = [{normal_axis(prior[0]) for prior in STAGES[:k]} for k in range(len(STAGES))]
    stage_plans = [_stage(block, g, sides, pos, n_canvas, h) for sides, h in zip(STAGES, handled)]
    one_layer = [_stage(block, 1, sides, pos, n_canvas, h) for sides, h in zip(STAGES, handled)]
    flat_centres = centres.reshape(3, -1)
    for stage in one_layer:
        flat_centres = apply_reflections(stage, flat_centres)
    centres = flat_centres.reshape((3,) + canvas)
    grid = Grid(jnp.asarray(nodes), jnp.asarray(centres), jnp.asarray(volume),
                tuple(jnp.asarray(f) for f in faces),
                box_sides(canvas, **dict.fromkeys(SIDES_3D, "canvas")))
    padded_centres = jnp.asarray(np.pad(centres, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge"))
    walls = _walls(block, pos, g, canvas)
    return Atlas(tuple(mesh), g, ((0, 0, 0),), (window,), grid, interior, padded_centres,
                 tuple(stage_plans), _neighbour_map(block, pos, n_canvas), _lines(shape, g, canvas), walls,
                 (0.25, 0.0))


def _stage(block: Grid, g: int, sides, pos, n_canvas: int, handled: set) -> Stage:
    """The gather plan of one fill stage: each side extended, along running
    axes filled by an earlier stage, ``g`` cells into those ghosts."""
    shape = block.shape
    index = np.arange(n_canvas)
    entries: dict[str, list] = {}
    for side in sides:
        normal, mid = _face_geometry(block, side)
        axes = running_axes(side)
        n = tuple(shape[a] for a in axes)
        for rect in block.sides.side(side):
            ranges = []
            for m, axis in enumerate(axes):
                ext = g if axis in handled else 0
                lo = rect.start[m] - (ext if rect.start[m] == 0 else 0)
                hi = rect.stop[m] + (ext if rect.stop[m] == n[m] else 0)
                ranges.append(range(lo, hi))
            for a in ((x, y) for x in ranges[0] for y in ranges[1]):
                for layer in range(g):
                    ghost = pos(*cell_on(side, a, layer, shape, inward=False))
                    if isinstance(rect, InterfaceRect):
                        p = list(rect.partner(a))
                        for m, axis in enumerate(running_axes(rect.side)):
                            corner = g if axis in handled else 0
                            p[m] = min(max(p[m], -corner), shape[axis] - 1 + corner)
                        index[ghost] = pos(*cell_on(rect.side, p, layer, shape, inward=True))
                    else:
                        read = layer if rect.kind in (*WALL_KINDS, "symmetry") else 0
                        source = pos(*cell_on(side, a, read, shape, inward=True))
                        f = tuple(min(max(a[m], 0), n[m] - 1) for m in range(2))
                        entries.setdefault(rect.kind, []).append(
                            (ghost, source, *normal[:, f[0], f[1]], *mid[:, f[0], f[1]]))
    conditions, offset = [], n_canvas
    for kind, rows in entries.items():
        rows = np.array(rows)
        index[rows[:, 0].astype(np.int64)] = offset + np.arange(len(rows))
        offset += len(rows)
        conditions.append((kind, rows[:, 1].astype(np.int64), rows[:, 2:5].T.copy(), rows[:, 5:8].T.copy()))
    return Stage(index, tuple(conditions))


def _neighbour_map(block: Grid, pos, n_canvas: int) -> np.ndarray:
    """Across every block face, the cell on the other side, or the cell itself."""
    shape = block.shape
    index = np.arange(n_canvas)
    for side in SIDES_3D:
        for rect in block.sides.side(side):
            for a in ((x, y) for x in range(rect.start[0], rect.stop[0])
                      for y in range(rect.start[1], rect.stop[1])):
                ghost = pos(*cell_on(side, a, 0, shape, inward=False))
                if isinstance(rect, InterfaceRect):
                    index[ghost] = pos(*cell_on(rect.side, rect.partner(a), 0, shape, inward=True))
                else:
                    index[ghost] = pos(*cell_on(side, a, 0, shape, inward=True))
    return index


def _lines(shape, g: int, canvas) -> Lines:
    """One chain along ``j`` per ``(i, k)``; links across the j-faces between
    its cells (flat index into the canvas j-face array, ``(NI, NJ+1, NK)``)."""
    ni, nj, nk = shape
    _, NJ, NK = canvas
    i, j, k = (g + a for a in np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij"))
    order = lambda a: a.transpose(0, 2, 1).reshape(ni * nk, nj)          # noqa: E731  (i, k) rows, j along
    cells = order((i * NJ + j) * NK + k).astype(np.int64)
    face = order((i * (NJ + 1) + j + 1) * NK + k).astype(np.int64)        # the face above each cell
    up_sign = np.ones_like(cells)
    up_sign[:, -1] = 0
    face[:, -1] = 0
    return Lines(cells, face, up_sign, face.copy(), -up_sign)


def _walls(block: Grid, pos, g: int, canvas) -> Walls:
    quads = wall_quads((block,))
    NI, NJ, NK = canvas
    cell, family, face, sign = [], [], [], []
    for s, a in zip(quads.side, quads.cell):
        side = SIDES_3D[s]
        c = cell_on(side, tuple(a), 0, block.shape, inward=True)
        cell.append(pos(*c))
        n = normal_axis(side)
        f = [g + x for x in c]
        f[n] += side.endswith("max")
        dims = [NI, NJ, NK]
        dims[n] += 1
        family.append(n)
        face.append((f[0] * dims[1] + f[1]) * dims[2] + f[2])
        sign.append(1.0 if side.endswith("min") else -1.0)
    return Walls(quads, np.array(cell, dtype=np.int64), np.array(family, dtype=np.int64),
                 np.array(face, dtype=np.int64), np.array(sign))
