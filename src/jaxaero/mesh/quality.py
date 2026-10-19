"""Grid quality, for any mesh: what a user should know before running on it.

Per mesh, over all blocks: cell areas (a folded cell is an error), the
smallest angle between a cell's ``i`` and ``j`` directions, the largest
growth ratio between neighbouring cells along each index, the aspect ratio,
the wall's first-cell heights and its orthogonality, and how far the outer
boundary lies from the walls; separately, the growth across block
interfaces (``interface_growth``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .block import Interface, cell_widths
from .mesh import Mesh, as_mesh, side_nodes, wall_faces

WARN_MIN_ANGLE = 20.0
WARN_GROWTH = 1.5


@dataclass(frozen=True)
class Quality:
    cells: int
    min_area: float
    min_angle_deg: float
    max_growth: float
    max_aspect: float
    wall_spacing: tuple[float, float]
    wall_angle_deg: float
    far_field: float

    @property
    def folded(self) -> bool:
        return not self.min_area > 0.0

    def warnings(self) -> list[str]:
        out = []
        if self.min_angle_deg < WARN_MIN_ANGLE:
            out.append(f"cells as skewed as {self.min_angle_deg:.1f} deg (under {WARN_MIN_ANGLE:g})")
        if self.max_growth > WARN_GROWTH:
            out.append(f"neighbouring cells growing by up to {self.max_growth:.2f}x (over {WARN_GROWTH:g})")
        return out

    def summary(self) -> str:
        lo, hi = self.wall_spacing
        text = (f"grid quality: {self.cells} cells, min area {self.min_area:.2e}, min angle "
                f"{self.min_angle_deg:.1f} deg, max growth {self.max_growth:.2f}, max aspect "
                f"{self.max_aspect:.3g}; wall: first cell {lo:.2e}..{hi:.2e}, within "
                f"{self.wall_angle_deg:.1f} deg of orthogonal; far field >= {self.far_field:.3g} from the walls")
        return text + "".join(f"\n  warning: {w}" for w in self.warnings())

    def record(self) -> dict:
        return asdict(self)


def _cell_angles(x, y) -> np.ndarray:
    """Angle between each cell's mean ``i`` and ``j`` edges, degrees."""
    ex = 0.5 * (x[1:, :-1] - x[:-1, :-1] + x[1:, 1:] - x[:-1, 1:])
    ey = 0.5 * (y[1:, :-1] - y[:-1, :-1] + y[1:, 1:] - y[:-1, 1:])
    fx = 0.5 * (x[:-1, 1:] - x[:-1, :-1] + x[1:, 1:] - x[1:, :-1])
    fy = 0.5 * (y[:-1, 1:] - y[:-1, :-1] + y[1:, 1:] - y[1:, :-1])
    cos = np.abs(ex * fx + ey * fy) / (np.hypot(ex, ey) * np.hypot(fx, fy))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def quality(mesh: Mesh) -> Quality:
    """The quality of ``mesh`` (walls: its patches whose condition is a wall)."""
    mesh = as_mesh(mesh)
    if mesh[0].dim == 3:
        return _quality_3d(mesh)
    areas, angles, growth, aspect = [], [], [1.0], [1.0]
    for block in mesh:
        x, y = np.asarray(block.x_node), np.asarray(block.y_node)
        areas.append(np.asarray(block.area).min())
        angles.append(_cell_angles(x, y).min())
        wi, wj = cell_widths(x, y)
        for w, axis in ((wi, 0), (wj, 1)):
            if w.shape[axis] > 1:
                a, b = np.take(w, range(w.shape[axis] - 1), axis), np.take(w, range(1, w.shape[axis]), axis)
                growth.append(float(np.max(np.maximum(a / b, b / a))))
        aspect.append(float(np.max(np.maximum(wi / wj, wj / wi))))
    heights, wall_angles, far = [], [0.0], np.inf
    faces = wall_faces(mesh)
    for block in mesh:
        x, y = np.asarray(block.x_node), np.asarray(block.y_node)
        for side, seg in block.sides.walls():
            wx, wy = side_nodes(block, side, seg)
            inner = {"j_min": (slice(seg.start, seg.stop + 1), 1), "j_max": (slice(seg.start, seg.stop + 1), -2),
                     "i_min": (1, slice(seg.start, seg.stop + 1)), "i_max": (-2, slice(seg.start, seg.stop + 1))}[side]
            ix, iy = x[inner], y[inner]
            heights.append(np.hypot(ix - wx, iy - wy))
            tx, ty = np.gradient(wx), np.gradient(wy)
            cos = np.abs(tx * (ix - wx) + ty * (iy - wy)) / (np.hypot(tx, ty) * np.hypot(ix - wx, iy - wy))
            if cos.size > 2:          # a segment's end nodes may be corners (a trailing edge)
                wall_angles.append(float(np.degrees(np.arcsin(np.clip(cos[1:-1], 0.0, 1.0))).max()))
    if faces.a.size:
        for block in mesh:
            for side, seg in block.sides.boundaries():
                if seg.kind == "far_field":
                    px, py = side_nodes(block, side, seg)
                    d = np.hypot(px[:, None] - faces.a[None, ::4, 0], py[:, None] - faces.a[None, ::4, 1])
                    far = min(far, float(d.min()))
    heights = np.concatenate(heights) if heights else np.array([np.nan])
    return Quality(sum(b.ni * b.nj for b in mesh), float(min(areas)), float(min(angles)),
                   float(max(growth)), float(max(aspect)), (float(heights.min()), float(heights.max())),
                   float(max(wall_angles)), far)


def interface_growth(mesh: Mesh) -> np.ndarray:
    """Across every interface face of a 2D multi-block mesh (each pair once),
    the larger of the two cells' widths across it over the smaller: how
    abruptly the grid changes from one block to the next."""
    mesh = as_mesh(mesh)
    widths = [cell_widths(np.asarray(g.x_node), np.asarray(g.y_node)) for g in mesh]
    across = {"j_min": lambda w: w[1][:, 0], "j_max": lambda w: w[1][:, -1],
              "i_min": lambda w: w[0][0, :], "i_max": lambda w: w[0][-1, :]}
    out = [np.ones(0)]
    for b, g in enumerate(mesh):
        for side, read in across.items():
            for seg in g.sides.side(side):
                if isinstance(seg, Interface) and (seg.block, seg.side, seg.partner_start) > (b, side, seg.start):
                    mine = read(widths[b])[seg.start:seg.stop]
                    theirs = across[seg.side](widths[seg.block])[seg.partner_cells()]
                    out.append(np.maximum(mine / theirs, theirs / mine))
    return np.concatenate(out)


def _mean_edges(nodes: np.ndarray) -> list[np.ndarray]:
    """Each cell's mean edge along ``i``, ``j``, ``k``, ``(3, ni, nj, nk)`` each."""
    out = []
    for axis in range(3):
        d = np.diff(nodes, axis=1 + axis)
        for other in range(3):
            if other != axis:
                d = 0.5 * (np.take(d, range(d.shape[1 + other] - 1), axis=1 + other)
                           + np.take(d, range(1, d.shape[1 + other]), axis=1 + other))
        out.append(d)
    return out


def _quality_3d(mesh: Mesh) -> Quality:
    """``quality`` of single 3D blocks: smallest volume and angle between the
    mean edges, growth and aspect from the widths between opposite face
    centroids, first-cell heights and orthogonality off every wall, and the
    far field's distance from the walls. Degenerate edges (a pole) are skipped."""
    from .block3d import normal_axis
    from .wall3d import wall_quads

    (block,) = mesh
    nodes = np.asarray(block.nodes)
    edges = _mean_edges(nodes)
    length = [np.linalg.norm(e, axis=0) for e in edges]
    angles = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.abs(np.sum(edges[a] * edges[b], axis=0)) / (length[a] * length[b])
        angles.append(np.degrees(np.arccos(np.clip(cos[np.isfinite(cos)], 0.0, 1.0))).min())
    growth, widths = [1.0], length
    for axis, w in enumerate(widths):
        a, b = np.take(w, range(w.shape[axis] - 1), axis), np.take(w, range(1, w.shape[axis]), axis)
        ok = (a > 0) & (b > 0)
        growth.append(float(np.max(np.maximum(a / b, b / a)[ok])))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.maximum.reduce([w1 / w2 for w1 in widths for w2 in widths if w1 is not w2])
    aspect = float(np.max(ratio[np.isfinite(ratio)]))
    heights, wall_angles = [], [0.0]
    for side, rect in block.sides.walls():
        n = normal_axis(side)
        first, second = (0, 1) if side.endswith("min") else (-1, -2)
        wall = np.take(nodes, first, axis=1 + n)
        inner = np.take(nodes, second, axis=1 + n)
        h = inner - wall
        heights.append(np.linalg.norm(h, axis=0).ravel())
        S = np.take(np.asarray(block.faces[n]), 0 if first == 0 else block.shape[n], axis=1 + n)
        hc = 0.25 * (h[:, :-1, :-1] + h[:, 1:, :-1] + h[:, :-1, 1:] + h[:, 1:, 1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.abs(np.sum(S * hc, axis=0)) / (np.linalg.norm(S, axis=0) * np.linalg.norm(hc, axis=0))
        wall_angles.append(float(np.degrees(np.arccos(np.clip(cos[np.isfinite(cos)], 0.0, 1.0))).max()))
    far = np.inf
    quads = wall_quads(mesh)
    if len(quads.corners):
        wall_points = quads.mid[:: max(1, len(quads.mid) // 2000)]
        for side, rect in block.sides.boundaries():
            if rect.kind == "far_field":
                plane = np.take(nodes, 0 if side.endswith("min") else nodes.shape[1 + normal_axis(side)] - 1,
                                axis=1 + normal_axis(side)).reshape(3, -1)[:, ::7]
                d = np.linalg.norm(plane.T[:, None, :] - wall_points[None], axis=-1)
                far = min(far, float(d.min()))
    heights = np.concatenate(heights) if heights else np.array([np.nan])
    return Quality(int(np.prod(block.shape)), float(np.asarray(block.volume).min()), float(min(angles)),
                   float(max(growth)), aspect, (float(heights.min()), float(heights.max())),
                   float(max(wall_angles)), far)
