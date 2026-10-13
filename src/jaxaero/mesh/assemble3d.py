"""A 3D structured grid file to a mesh: one block, its interfaces and patches.

The 3D counterpart of mesh/assemble.py, for a single block (as the wing grids
of TMR's generator are), assembled from the geometry alone:

1. the block made right-handed (``k`` reversed if its volumes are negative);
2. a side whose nodes coincide with another side's, node for node, is an
   interface to it (an O seam: ``k_min`` facing ``k_max``);
3. a side collapsed to a line (every face of zero area) is a **pole**: an
   interface with itself across the axis, half a turn round the periodic
   running direction (mesh/block3d.py); that direction must be an O seam;
4. every other face is named by the case's patch rules (``PatchRule``, boxes
   in ``x, y, z``), first match wins; a side whose faces get several names is
   cut into rectangles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .assemble import MeshInfo, PatchRule
from .block import Grid
from .block3d import SIDES_3D, BoundaryRect, InterfaceRect, Sides3D, normal_axis, running_axes
from .mesh import Mesh
from .plot3d import read_plot3d_3d


def is_3d_plot3d(path: str | Path) -> bool:
    """True for a Plot3D file of 3D blocks that are not a thin extrusion of a
    2D grid (more than two nodes in every direction)."""
    try:
        blocks = read_plot3d_3d(path)
    except ValueError:
        return False
    return all(min(b.shape[1:]) > 2 for b in blocks)


def load_grid_3d(path: str | Path, coordinate_scale: float = 1.0, rules: tuple[PatchRule, ...] = (),
                 dtype=np.float64, tol: float = 1e-10) -> tuple[Mesh, MeshInfo]:
    blocks = read_plot3d_3d(path)
    if len(blocks) != 1:
        raise ValueError(f"{Path(path).name}: 3D grids are single-block for now; this one has {len(blocks)}")
    nodes = coordinate_scale * blocks[0].astype(dtype)
    grid = assemble_3d(nodes, rules, tol)
    return (grid,), _info(grid, str(path))


def _plane(nodes: np.ndarray, side: str) -> np.ndarray:
    """Nodes of a side, ``(3, na + 1, nb + 1)`` in its running order."""
    n = normal_axis(side)
    return np.take(nodes, 0 if side.endswith("min") else nodes.shape[1 + n] - 1, axis=1 + n)


def assemble_3d(nodes: np.ndarray, rules: tuple[PatchRule, ...] = (), tol: float = 1e-10) -> Grid:
    from .block import metrics

    _, volume, faces = metrics(nodes)
    if np.all(volume < 0.0):
        nodes = nodes[:, :, :, ::-1]
        _, volume, faces = metrics(nodes)
    if not np.all(volume > 0.0):
        raise ValueError(f"the block has {int(np.sum(volume <= 0.0))} cells of non-positive volume")
    shape = volume.shape
    scale = float(np.abs(nodes).max())
    sides: dict[str, tuple] = {}

    # Interfaces between whole sides with coincident nodes, in running order.
    for a in SIDES_3D:
        for b in SIDES_3D:
            if a >= b or a in sides or b in sides or running_axes(a) != running_axes(b):
                continue
            pa, pb = _plane(nodes, a), _plane(nodes, b)
            if pa.shape == pb.shape and np.abs(pa - pb).max() <= tol * scale and _area(faces, a).max() > 0.0:
                extent = tuple(shape[x] for x in running_axes(a))
                sides[a] = (InterfaceRect((0, 0), extent, 0, b, (0, 0)),)
                sides[b] = (InterfaceRect((0, 0), extent, 0, a, (0, 0)),)

    # Poles: a side of zero area round a periodic direction.
    for side in SIDES_3D:
        if side in sides or _area(faces, side).max() > tol * scale**2:
            continue
        extent = tuple(shape[x] for x in running_axes(side))
        periodic = [m for m, axis in enumerate(running_axes(side))
                    if any(isinstance(r, InterfaceRect) and normal_axis(s) == axis
                           for s in sides for r in sides[s])]
        if len(periodic) != 1 or extent[periodic[0]] % 2:
            raise ValueError(f"side {side} is collapsed but not round one periodic direction of even extent")
        m = periodic[0]
        origin, period = [0, 0], [0, 0]
        origin[m], period[m] = extent[m] // 2, extent[m]
        sides[side] = (InterfaceRect((0, 0), extent, 0, side, tuple(origin), period=tuple(period)),)

    # Patches by rule, as rectangles.
    for side in SIDES_3D:
        if side in sides:
            continue
        plane = _plane(nodes, side)
        mid = 0.25 * (plane[:, :-1, :-1] + plane[:, 1:, :-1] + plane[:, :-1, 1:] + plane[:, 1:, 1:])
        names = np.full(mid.shape[1:], "", dtype=object)
        for rule in rules:
            take = (names == "") & rule.contains(mid[0], mid[1], mid[2])
            names[take] = rule.name
        if np.any(names == ""):
            raise ValueError(f"faces of {side} match no patch rule; add a rule without bounds (`domain.patches`)")
        sides[side] = _rectangles(names)
    return Grid.from_node_array(nodes, Sides3D(shape, *(sides[s] for s in SIDES_3D)))


def _area(faces, side: str) -> np.ndarray:
    n = normal_axis(side)
    S = np.take(faces[n], 0 if side.endswith("min") else faces[n].shape[1 + n] - 1, axis=1 + n)
    return np.linalg.norm(S, axis=0)


def _rectangles(names: np.ndarray) -> tuple[BoundaryRect, ...]:
    """Rectangles of equal name tiling a side: runs along the second running
    index, merged across consecutive rows of the first where identical."""
    rows = []
    for a in range(names.shape[0]):
        runs, start = [], 0
        for b in range(1, names.shape[1] + 1):
            if b == names.shape[1] or names[a, b] != names[a, start]:
                runs.append((start, b, names[a, start]))
                start = b
        rows.append(runs)
    out, open_ = [], {}
    for a, runs in enumerate(rows + [[]]):
        keys = set(runs)
        for key, first in list(open_.items()):
            if key not in keys:
                out.append(BoundaryRect(key[2], (first, key[0]), (a, key[1])))
                del open_[key]
        for key in runs:
            open_.setdefault(key, a)
    return tuple(sorted(out, key=lambda r: (r.start, r.stop)))


def _info(grid: Grid, source: str) -> MeshInfo:
    counts: dict[str, int] = {}
    for _, r in grid.sides.boundaries():
        counts[r.patch] = counts.get(r.patch, 0) + (r.stop[0] - r.start[0]) * (r.stop[1] - r.start[1])
    interfaces = sum(isinstance(r, InterfaceRect) for s in SIDES_3D for r in grid.sides.side(s))
    nodes = np.asarray(grid.nodes)
    return MeshInfo(source, 1, int(np.prod(grid.shape)), interfaces, counts, 1,
                    float(np.linalg.norm(nodes, axis=0).max()), _first_spacing(grid), 0.0)


def _first_spacing(grid: Grid) -> float:
    """Smallest first-cell height off ``j_min`` (the wall side of the wing grids)."""
    nodes = np.asarray(grid.nodes)
    return float(np.linalg.norm(nodes[:, :, 1] - nodes[:, :, 0], axis=0).min())
