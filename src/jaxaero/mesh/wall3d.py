"""Wall faces of 3D blocks, and the exact distance to them.

A wall face is the bilinear patch of its four nodes,

    x(u, v) = a + u (b - a) + v (d - a) + u v (a - b + c - d),   0 <= u, v <= 1,

the surface the cell metrics assume. Its distance from a point is the
smaller of the interior minimum (Newton on |x(u, v) - p|^2) and the
distances to its four edges, which are straight segments. Every cell first
ranks all faces by their distance to the face's two triangles (Ericson,
*Real-Time Collision Detection*, 2005, 5.1.5; in JAX, in chunks of cells, on
the GPU when there is one), then takes the exact patch distance over the
nearest ``candidates``. The triangles alone are not enough: on the rounded
tip of TMR's M6 grids the faces are warped by 40 times the first cell's
height, and the triangle distance of the first cells there was 15 times too
large (SA then runs away at the tip). A KD-tree over face centroids
cannot certify the minimum away from the wall either (it failed for 99% of
the cells on the 135k grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block3d import SIDES_3D, normal_axis


@dataclass(frozen=True)
class WallQuads:
    """Every wall face: block, side (index into ``SIDES_3D``), running cell
    position on it ``(n, 2)``, body (one wing: 0) and corners ``(n, 4, 3)`` in
    running order ``a b c d``."""

    block: np.ndarray
    side: np.ndarray
    cell: np.ndarray
    body: np.ndarray
    corners: np.ndarray

    @property
    def bodies(self) -> int:
        return int(self.body.max()) + 1 if self.body.size else 0

    @property
    def mid(self) -> np.ndarray:
        """Mean of the four corners, ``(n, 3)``."""
        return self.corners.mean(axis=1)


def wall_quads(mesh) -> WallQuads:
    rows = []
    for b, grid in enumerate(mesh):
        nodes = np.asarray(grid.nodes)
        for side, rect in grid.sides.walls():
            n = normal_axis(side)
            plane = np.take(nodes, 0 if side.endswith("min") else grid.shape[n], axis=1 + n)
            for x in range(rect.start[0], rect.stop[0]):
                for y in range(rect.start[1], rect.stop[1]):
                    quad = [plane[:, x, y], plane[:, x + 1, y], plane[:, x + 1, y + 1], plane[:, x, y + 1]]
                    rows.append((b, SIDES_3D.index(side), x, y, np.stack(quad)))
    if not rows:
        empty = np.zeros(0, dtype=int)
        return WallQuads(empty, empty, np.zeros((0, 2), int), empty, np.zeros((0, 4, 3)))
    return WallQuads(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                     np.array([r[2:4] for r in rows]), np.zeros(len(rows), dtype=int),
                     np.stack([r[4] for r in rows]))


def triangle_distance(p, a, b, c, xp=np):
    """Distance from points ``p (..., 3)`` to triangles ``a b c (..., 3)``,
    broadcast; a degenerate triangle (zero area) is handled as its edges.
    ``xp``: ``numpy`` or ``jax.numpy``."""
    def dot(u, v):
        return xp.sum(u * v, axis=-1)

    ab, ac = b - a, c - a
    n = xp.cross(ab, ac)
    nn = dot(n, n)
    # Barycentric coordinates of the projection on the plane.
    ap = p - a
    safe = xp.where(nn > 0.0, nn, 1.0)
    v = dot(xp.cross(ap, ac), n) / safe
    w = dot(xp.cross(ab, ap), n) / safe
    inside = (nn > 0.0) & (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
    plane = xp.abs(dot(ap, n)) / xp.sqrt(safe)
    best = xp.where(inside, plane, xp.inf)
    for s, e in ((a, b), (b, c), (c, a)):
        edge = e - s
        t = xp.clip(dot(p - s, edge) / xp.maximum(dot(edge, edge), 1e-300), 0.0, 1.0)
        best = xp.minimum(best, xp.sqrt(dot(p - s - t[..., None] * edge, p - s - t[..., None] * edge)))
    return best


def _segment_distance(p, s, e, xp):
    edge = e - s
    t = xp.clip(xp.sum((p - s) * edge, -1) / xp.maximum(xp.sum(edge * edge, -1), 1e-300), 0.0, 1.0)
    r = p - s - t[..., None] * edge
    return xp.sqrt(xp.sum(r * r, -1))


def patch_distance(p, quads, xp=np, iterations: int = 12):
    """Distance from points ``p (..., 3)`` to bilinear patches ``(..., 4, 3)``
    (corners ``a b c d`` in order): Newton from the centre for the interior
    stationary point, kept only if it lands inside, and the four edges."""
    a, b, c, d = (quads[..., k, :] for k in range(4))
    eu, ev, w = b - a, d - a, a - b + c - d
    u = xp.full(p.shape[:-1], 0.5)
    v = xp.full(p.shape[:-1], 0.5)
    for _ in range(iterations):
        x_u = eu + v[..., None] * w
        x_v = ev + u[..., None] * w
        r = a + u[..., None] * eu + v[..., None] * ev + (u * v)[..., None] * w - p
        g_u, g_v = xp.sum(r * x_u, -1), xp.sum(r * x_v, -1)
        h_uu, h_vv = xp.sum(x_u * x_u, -1), xp.sum(x_v * x_v, -1)
        h_uv = xp.sum(x_u * x_v, -1) + xp.sum(r * w, -1)
        det = h_uu * h_vv - h_uv * h_uv
        safe = xp.where(xp.abs(det) > 1e-300, det, 1.0)
        u = xp.where(xp.abs(det) > 1e-300, u - (h_vv * g_u - h_uv * g_v) / safe, u)
        v = xp.where(xp.abs(det) > 1e-300, v - (h_uu * g_v - h_uv * g_u) / safe, v)
        u, v = xp.clip(u, -0.5, 1.5), xp.clip(v, -0.5, 1.5)
    r = a + u[..., None] * eu + v[..., None] * ev + (u * v)[..., None] * w - p
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    best = xp.where(inside, xp.sqrt(xp.sum(r * r, -1)), xp.inf)
    for s_, e_ in ((a, b), (b, c), (c, d), (d, a)):
        best = xp.minimum(best, _segment_distance(p, s_, e_, xp))
    return best


def wall_distance_3d(mesh, chunk: int = 256, candidates: int = 16) -> list[np.ndarray]:
    """Exact distance from every cell centre to the nearest wall face (a
    bilinear patch), one array of the cell shape per block."""
    import jax
    import jax.numpy as jnp

    quads = wall_quads(mesh)
    if not len(quads.corners):
        return [np.full(g.shape, np.inf) for g in mesh]
    q = jnp.asarray(quads.corners)                                               # (F, 4, 3)
    k = min(candidates, q.shape[0])

    @jax.jit
    def nearest(points):                                                         # (chunk, 3)
        p = points[:, None, :]
        tri = jnp.minimum(triangle_distance(p, q[None, :, 0], q[None, :, 1], q[None, :, 2], xp=jnp),
                          triangle_distance(p, q[None, :, 0], q[None, :, 2], q[None, :, 3], xp=jnp))
        _, idx = jax.lax.top_k(-tri, k)                                          # (chunk, k)
        return patch_distance(p, q[idx], xp=jnp).min(axis=1)

    out = []
    for grid in mesh:
        points = np.moveaxis(np.asarray(grid.centres, dtype=np.float64), 0, -1).reshape(-1, 3)
        n = len(points)
        padded = np.concatenate([points, np.repeat(points[-1:], -n % chunk, axis=0)])
        d = jax.lax.map(nearest, jnp.asarray(padded.reshape(-1, chunk, 3)))
        out.append(np.asarray(d).ravel()[:n].reshape(grid.shape))
    return out
