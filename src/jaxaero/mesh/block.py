"""One structured block: its geometry and what lies beyond each of its sides.

Nodes are ``(ni+1, nj+1)``, cells ``(ni, nj)``. Face area vectors point toward
increasing index and carry the face length, so a residual is a plain sum of
``flux . S``:

* i-faces ``(ni+1, nj)``, between cells ``(i-1, j)`` and ``(i, j)``;
* j-faces ``(ni, nj+1)``, between cells ``(i, j-1)`` and ``(i, j)``.

A block must be right-handed (positive areas). Each of its four sides is tiled
by segments: a ``Boundary`` (a named patch, which a case maps to a boundary
condition) or an ``Interface`` to cells of a side of a block, this one or
another. That is the whole of a block's connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

SIDES = ("j_min", "j_max", "i_min", "i_max")
"""The four sides of a block, named by the index constant on them."""

OUTWARD = {"j_min": -1, "j_max": 1, "i_min": -1, "i_max": 1}
"""Sign turning a side's index-direction normal into the outward one."""

CONDITION_KINDS = ("slip_wall", "no_slip_wall", "symmetry", "far_field", "inflow_total", "outflow_static")
"""Boundary conditions a patch may be given (numerics/boundary.py)."""

WALL_KINDS = ("slip_wall", "no_slip_wall")


@dataclass(frozen=True)
class Boundary:
    """Cells ``[start, stop)`` along a side on the patch ``patch``; ``kind``
    is its boundary condition, set by the case (``mesh.assign_conditions``).
    Cells along a side are counted in its running index: ``i`` on a ``j``
    side, ``j`` on an ``i`` side."""

    patch: str
    start: int
    stop: int
    kind: str | None = None

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in CONDITION_KINDS:
            raise ValueError(f"unknown boundary condition {self.kind!r}; known: {CONDITION_KINDS}")


@dataclass(frozen=True)
class Interface:
    """Cells ``[start, stop)`` facing cells ``[partner_start, partner_start + n)``
    of side ``side`` of block ``block`` (in order, or reversed). Ghost layer
    ``l`` is the partner's ``l``-th layer inward from its side, copied:
    velocities are Cartesian. The O-grid seam is ``i_min`` facing ``i_max`` of
    the same block; the C-grid wake cut ``j_min`` facing ``j_min``, reversed."""

    start: int
    stop: int
    block: int
    side: str
    partner_start: int
    reversed: bool

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"unknown side {self.side!r}; known: {SIDES}")

    def partner_cells(self) -> np.ndarray:
        cells = self.partner_start + np.arange(self.stop - self.start)
        return cells[::-1] if self.reversed else cells


Segment = Boundary | Interface


@dataclass(frozen=True)
class BlockSides:
    """Segments tiling each side in order: ``j_min``/``j_max`` over
    ``[0, ni)``, ``i_min``/``i_max`` over ``[0, nj)``."""

    ni: int
    nj: int
    j_min: tuple[Segment, ...]
    j_max: tuple[Segment, ...]
    i_min: tuple[Segment, ...]
    i_max: tuple[Segment, ...]

    def __post_init__(self) -> None:
        for side in SIDES:
            n = self.length(side)
            stops = [0] + [s.stop for s in self.side(side)]
            starts = [s.start for s in self.side(side)] + [n]
            if stops != starts or any(s.stop <= s.start for s in self.side(side)):
                raise ValueError(f"segments of {side} do not tile [0, {n}) in order: {self.side(side)}")

    def side(self, name: str) -> tuple[Segment, ...]:
        return getattr(self, name)

    def length(self, name: str) -> int:
        return self.ni if name.startswith("j") else self.nj

    def boundaries(self) -> list[tuple[str, Boundary]]:
        return [(side, s) for side in SIDES for s in self.side(side) if isinstance(s, Boundary)]

    def walls(self) -> list[tuple[str, Boundary]]:
        return [(side, s) for side, s in self.boundaries() if s.kind in WALL_KINDS]

    def with_conditions(self, kinds: dict[str, str]) -> "BlockSides":
        """Every boundary segment given the condition of its patch."""
        def assign(segments):
            return tuple(replace(s, kind=kinds[s.patch]) if isinstance(s, Boundary) else s
                         for s in segments)
        return replace(self, **{side: assign(self.side(side)) for side in SIDES})


def rectangle_sides(ni: int, nj: int, **segments) -> BlockSides:
    """A block whose sides are each one patch or the given segments, e.g.
    ``rectangle_sides(ni, nj, j_min="wall", j_max="farfield", i_min=..., ...)``;
    a string names a patch over the whole side."""
    def side(name):
        value = segments[name]
        return (Boundary(value, 0, ni if name.startswith("j") else nj),) if isinstance(value, str) else tuple(value)
    return BlockSides(ni, nj, *(side(name) for name in SIDES))


@dataclass(frozen=True)
class Grid:
    """A block's geometry (a JAX pytree) and its ``sides``, in ``d = 2`` or 3
    dimensions; cells ``shape = (ni, nj[, nk])``.

    ``nodes (d, ni+1, nj+1[, nk+1])``; cell ``centres (d, *shape)`` and
    ``volume (*shape)`` (an area in 2D); ``faces[a]``, the area vectors of the
    faces normal to index direction ``a``, ``(d, *shape)`` with one more along
    ``a``. In 2D the named views ``x_node, xc, area, sx_i, ...`` read the same.
    """

    nodes: Array
    centres: Array
    volume: Array
    faces: tuple[Array, ...]
    sides: object

    @property
    def dim(self) -> int:
        return self.volume.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.volume.shape)

    @property
    def ni(self) -> int:
        return self.volume.shape[0]

    @property
    def nj(self) -> int:
        return self.volume.shape[1]

    # 2D views
    x_node = property(lambda self: self.nodes[0])
    y_node = property(lambda self: self.nodes[1])
    xc = property(lambda self: self.centres[0])
    yc = property(lambda self: self.centres[1])
    area = property(lambda self: self.volume)
    sx_i = property(lambda self: self.faces[0][0])
    sy_i = property(lambda self: self.faces[0][1])
    sx_j = property(lambda self: self.faces[1][0])
    sy_j = property(lambda self: self.faces[1][1])

    @classmethod
    def from_nodes(cls, x_node, y_node, sides, dtype=np.float64) -> Grid:
        """A 2D block from its node coordinates ``(ni+1, nj+1)`` each."""
        return cls.from_node_array(np.stack([np.asarray(x_node, dtype=dtype), np.asarray(y_node, dtype=dtype)]),
                                   sides, dtype)

    @classmethod
    def from_node_array(cls, nodes, sides, dtype=np.float64) -> Grid:
        """A block from its nodes ``(d, ni+1, nj+1[, nk+1])``; all metrics
        computed once in NumPy (eager JAX would compile anew for every block
        shape), see ``metrics``."""
        nodes = np.asarray(nodes, dtype=dtype)
        expected = tuple(n + 1 for n in sides_shape(sides))
        if nodes.shape[1:] != expected:
            raise ValueError(f"node arrays must be {expected} for these sides, got {nodes.shape[1:]}")
        centres, volume, faces = metrics(nodes)
        return cls(jnp.asarray(nodes), jnp.asarray(centres), jnp.asarray(volume),
                   tuple(jnp.asarray(f) for f in faces), sides=sides)


jax.tree_util.register_dataclass(
    Grid, data_fields=["nodes", "centres", "volume", "faces"], meta_fields=["sides"],
)


def sides_shape(sides) -> tuple[int, ...]:
    """The cell shape a block's sides describe."""
    return tuple(sides.shape) if hasattr(sides, "shape") else (sides.ni, sides.nj)


def metrics(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """``(centres, volumes, face area vectors)`` from nodes ``(d, ...)``.

    2D: areas are half the cross product of the cell diagonals; a face's area
    vector is its edge turned clockwise, ``(dy, -dx)``.

    3D: a quadrilateral face ``a b c d`` (in the order of its two running
    indices) has ``S = (x_c - x_a) x (x_d - x_b) / 2``, the vector area of
    every surface its edges bound, so the six vectors of a cell sum to zero
    exactly (freestream preservation on warped faces) and a collapsed face
    (a pole) has ``S = 0`` exactly. Volumes from the divergence theorem,
    ``V = (1/3) sum_f S_f . x_f`` (outward ``S``, ``x_f`` the mean of the face's
    nodes), the volume of the trilinear hexahedron (tests/unit/test_grid3d.py).

    Centres are the mean of the cell's nodes; every vector points toward
    increasing index.
    """
    dim = nodes.shape[0]
    if dim == 2:
        x, y = nodes
        x1, y1, x2, y2 = x[:-1, :-1], y[:-1, :-1], x[1:, :-1], y[1:, :-1]
        x3, y3, x4, y4 = x[1:, 1:], y[1:, 1:], x[:-1, 1:], y[:-1, 1:]
        d1x, d1y, d2x, d2y = x3 - x1, y3 - y1, x4 - x2, y4 - y2
        area = 0.5 * (d1x * d2y - d1y * d2x)
        centres = np.stack([0.25 * (x1 + x2 + x3 + x4), 0.25 * (y1 + y2 + y3 + y4)])
        ex_i, ey_i = x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1]      # i-face edge (i,j)->(i,j+1)
        ex_j, ey_j = x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :]      # j-face edge (i,j)->(i+1,j)
        return centres, area, (np.stack([ey_i, -ex_i]), np.stack([-ey_j, ex_j]))
    if dim != 3:
        raise ValueError(f"grids are 2D or 3D, got {dim} coordinates")
    faces = []
    for axis in range(3):
        # The face's running indices, in cyclic order after `axis`, so that
        # (e1 x e2) points toward increasing `axis` on a right-handed block.
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        n = np.moveaxis(nodes, (1 + axis, 1 + a1, 1 + a2), (1, 2, 3))
        pa, pb, pc, pd = n[:, :, :-1, :-1], n[:, :, 1:, :-1], n[:, :, 1:, 1:], n[:, :, :-1, 1:]
        S = 0.5 * np.cross(pc - pa, pd - pb, axis=0)
        mid = 0.25 * (pa + pb + pc + pd)
        faces.append((np.moveaxis(S, (1, 2, 3), (1 + axis, 1 + a1, 1 + a2)),
                      np.moveaxis(mid, (1, 2, 3), (1 + axis, 1 + a1, 1 + a2))))
    volume = 0.0
    for axis, (S, mid) in enumerate(faces):
        flux = np.sum(S * mid, axis=0)
        volume = volume + (np.take(flux, range(1, flux.shape[axis]), axis=axis)
                           - np.take(flux, range(0, flux.shape[axis] - 1), axis=axis))
    volume = volume / 3.0
    corners = [nodes[:, i:i + nodes.shape[1] - 1, j:j + nodes.shape[2] - 1, k:k + nodes.shape[3] - 1]
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    centres = sum(corners) / 8.0
    return centres, volume, tuple(S for S, _ in faces)


# --- Checks ------------------------------------------------------------------------


def cell_widths(x_node, y_node) -> tuple[np.ndarray, np.ndarray]:
    """Cell extent along each index direction, between opposite face midpoints."""
    x, y = np.asarray(x_node), np.asarray(y_node)
    mx_i, my_i = 0.5 * (x[:, :-1] + x[:, 1:]), 0.5 * (y[:, :-1] + y[:, 1:])
    mx_j, my_j = 0.5 * (x[:-1, :] + x[1:, :]), 0.5 * (y[:-1, :] + y[1:, :])
    return (np.hypot(mx_i[1:] - mx_i[:-1], my_i[1:] - my_i[:-1]),
            np.hypot(mx_j[:, 1:] - mx_j[:, :-1], my_j[:, 1:] - my_j[:, :-1]))


def normal_closure_residual(grid: Grid) -> np.ndarray:
    """Sum of outward area vectors round each cell, ``(d, *shape)``: zero for a
    closed cell, so anything else is round-off or a metric bug."""
    total = 0.0
    for a, S in enumerate(grid.faces):
        S = np.asarray(S)
        total = total + (np.take(S, range(1, S.shape[1 + a]), axis=1 + a)
                         - np.take(S, range(S.shape[1 + a] - 1), axis=1 + a))
    return total


def polygon_area(x, y) -> float:
    """Signed shoelace area of a closed polyline, positive counter-clockwise."""
    x, y = np.asarray(x), np.asarray(y)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def perimeter_area(grid: Grid) -> float:
    """The shoelace area round the block's boundary, walked counter-clockwise
    in index space. An identity with ``sum(area)`` (both telescope the same
    cross products), so it checks indexing and orientation, not the area formula."""
    x, y = np.asarray(grid.x_node), np.asarray(grid.y_node)
    xs = np.concatenate([x[:, 0], x[-1, :], x[::-1, -1], x[0, ::-1]])
    ys = np.concatenate([y[:, 0], y[-1, :], y[::-1, -1], y[0, ::-1]])
    return polygon_area(xs, ys)


@dataclass(frozen=True)
class MetricReport:
    min_area: float
    max_area: float
    sum_area: float
    perimeter_area: float
    area_closure_error: float
    max_normal_closure: float

    @property
    def ok(self) -> bool:
        return self.min_area > 0.0 and self.area_closure_error < 1e-12 and self.max_normal_closure < 1e-12

    def summary(self) -> str:
        return (f"block metrics [{'PASS' if self.ok else 'FAIL'}]: area {self.min_area:.3e}..{self.max_area:.3e}, "
                f"closure {self.area_closure_error:.1e}, normals {self.max_normal_closure:.1e}")


def boundary_volume(grid: Grid) -> float:
    """3D: ``(1/3) sum S . x_f`` over the block's boundary faces (outward ``S``,
    ``x_f`` the mean of the face's nodes) -- an identity with ``sum(volume)``,
    as ``perimeter_area`` is in 2D."""
    nodes = np.asarray(grid.nodes)
    total = 0.0
    for a, S in enumerate(grid.faces):
        S = np.asarray(S)
        moved = np.moveaxis(nodes, 1 + a, 1)
        for index, sign in ((0, -1.0), (-1, 1.0)):
            plane = moved[:, index]
            mid = 0.25 * (plane[:, :-1, :-1] + plane[:, 1:, :-1] + plane[:, :-1, 1:] + plane[:, 1:, 1:])
            total += sign * float(np.sum(np.take(S, index, axis=1 + a) * mid))
    return total / 3.0


def validate_metrics(grid: Grid) -> MetricReport:
    """Positive areas (volumes), the boundary identity, closed cells. In 3D the
    closure is relative to the largest face (far fields reach 100 chords)."""
    area = np.asarray(grid.volume)
    closure = float(np.abs(normal_closure_residual(grid)).max())
    if grid.dim == 2:
        boundary = perimeter_area(grid)
    else:
        boundary = boundary_volume(grid)
        closure /= max(float(np.abs(np.asarray(S)).max()) for S in grid.faces)
    sum_area = float(area.sum())
    return MetricReport(float(area.min()), float(area.max()), sum_area, boundary,
                        abs(sum_area - boundary) / max(abs(boundary), 1e-300), closure)


# --- The common single-block layouts ---------------------------------------------


def o_grid_sides(ni: int, nj: int) -> BlockSides:
    """An O-grid: ``wall`` on ``j_min``, ``farfield`` on ``j_max``, the seam
    ``i_min`` <-> ``i_max``. ``ni`` even, so the leading edge is a node."""
    if ni < 4 or ni % 2 or nj < 1:
        raise ValueError(f"an O-grid needs ni even and >= 4, nj >= 1; got {ni} x {nj}")
    return rectangle_sides(ni, nj, j_min="wall", j_max="farfield",
                           i_min=(Interface(0, nj, 0, "i_max", 0, False),),
                           i_max=(Interface(0, nj, 0, "i_min", 0, False),))
