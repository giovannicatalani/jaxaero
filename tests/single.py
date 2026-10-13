"""Single-block views of the atlas, for tests written cell by cell.

A one-block atlas's canvas is the block with its ghost ring: ``(nvar, ni + 2g,
nj + 2g)``, interior at ``[g:-g, g:-g]``. These helpers move block-shaped
fields in and out of it, so a test can state its expectation on the block.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxaero.core.config import NumericsConfig
from jaxaero.mesh.atlas import Atlas, build_atlas
from jaxaero.mesh.block import BlockSides, Boundary, Grid, Interface, rectangle_sides
from jaxaero.mesh.block3d import BoundaryRect, InterfaceRect, Sides3D, box_sides
from jaxaero.mesh.wall3d import triangle_distance
from jaxaero.mesh.mesh import assign_conditions, patches
from jaxaero.numerics import boundary as bc
from jaxaero.numerics.residual import ResidualOperator, make_residual

G = NumericsConfig().n_ghost


_ATLASES: dict[tuple, tuple] = {}


def conditions(grid, no_slip: bool = False) -> dict[str, str]:
    """The case layer's default condition for every patch of ``grid``, the
    wall slip or no-slip."""
    kinds = {"wall": "no_slip_wall" if no_slip else "slip_wall", "farfield": "far_field",
             "symmetry": "symmetry", "inflow": "inflow_total", "outflow": "outflow_static"}
    return {p: kinds[p] for p in patches(grid)}


def atlas_of(grid, g: int = G, no_slip: bool = False) -> Atlas:
    """The grid's one-block atlas, its patches given the default conditions;
    built once per grid object."""
    key = (id(grid), g, no_slip)
    if key not in _ATLASES:
        _ATLASES[key] = (grid, build_atlas(assign_conditions(grid, conditions(grid, no_slip)), g))
    return _ATLASES[key][1]


def interior(atlas: Atlas):
    g = atlas.g
    return (slice(g, atlas.ni - g), slice(g, atlas.nj - g))


def to_canvas(U, atlas: Atlas, fill=None):
    """A block field ``(..., ni, nj)`` on the canvas; ``fill`` elsewhere
    (default: edge copies, harmless for every test here)."""
    g = atlas.g
    pad = [(0, 0)] * (U.ndim - 2) + [(g, g), (g, g)]
    if fill is None:
        return jnp.pad(U, pad, mode="edge")
    return jnp.where(atlas.interior, jnp.pad(U, pad, mode="edge"), fill)


def from_canvas(C, atlas: Atlas):
    i, j = interior(atlas)
    return C[..., i, j]


def fill_ghosts(q, grid, n_ghost: int, q_inf, gamma: float, no_slip: bool = False, far_field=None):
    """The block's primitive state with ``n_ghost`` ghost layers filled,
    ``(nvar, ni + 2g, nj + 2g)``: the old single-block ``fill_ghosts``."""
    atlas = atlas_of(grid, n_ghost, no_slip)
    canvas = to_canvas(q, atlas)
    model = far_field or bc.uniform_far_field
    return bc.fill_ghosts(canvas, atlas, bc.Conditions(q_inf, gamma, model(canvas, atlas, q_inf, gamma)))


def padded_centres(grid):
    """Cell centres with one ghost layer, ``(ni + 2, nj + 2)`` each."""
    atlas = atlas_of(grid)
    g = atlas.g
    c = atlas.centres[:, g : atlas.ni - g + 2, g : atlas.nj - g + 2]
    return c[0], c[1]


def face_slices(atlas: Atlas):
    """The block's own faces among the canvas faces: i-faces, j-faces."""
    g = atlas.g
    return ((slice(g, atlas.ni - g + 1), slice(g, atlas.nj - g)),
            (slice(g, atlas.ni - g), slice(g, atlas.nj - g + 1)))


class Single:
    """A residual operator seen block-wise: ``residual``, ``time_step``,
    ``norm`` and ``wave_speeds`` on block-shaped fields."""

    def __init__(self, grid, flow, numerics, *args, **kwargs):
        atlas = atlas_of(grid, numerics.n_ghost, not flow.inviscid)
        self.op: ResidualOperator = make_residual(atlas, flow, numerics, *args, **kwargs)
        self.atlas = self.op.atlas
        self.U0 = self.op.initial_state()

    def canvas(self, U):
        return to_canvas(U, self.atlas, self.U0)

    def residual(self, U):
        return from_canvas(self.op.residual(self.canvas(U)), self.atlas)

    def time_step(self, U):
        return from_canvas(self.op.time_step(self.canvas(U)), self.atlas)

    def norm(self, R):
        return self.op.norm(to_canvas(R, self.atlas, 0.0))

    def wave_speeds(self, U):
        li, lj = self.op.wave_speeds(self.canvas(U))
        fi, fj = face_slices(self.atlas)
        return li[fi], lj[fj]

    def initial_state(self):
        return from_canvas(self.U0, self.atlas)

    def __getattr__(self, name):
        return getattr(self.op, name)


def unit_j_normals(grid, j: int):
    """Unit normals of the ``j``-constant faces, toward increasing ``j``."""
    sx, sy = grid.sx_j[:, j], grid.sy_j[:, j]
    length = jnp.hypot(sx, sy)
    return sx / length, sy / length


def ghost_centres(grid):
    """``(2, ni + 2, nj + 2)``: what ``face_gradients`` takes."""
    x, y = padded_centres(grid)
    return jnp.stack([x, y])


def forces(U, grid, flow, reference_length: float = 1.0):
    """``integrate_forces`` of a block-shaped state."""
    from jaxaero.post.forces import integrate_forces

    atlas = atlas_of(grid, G, not flow.inviscid)
    return integrate_forces(to_canvas(U, atlas), atlas, flow, reference_length)


def wall_surface(U, grid, flow):
    """``surface`` of a block-shaped state: ``x``, ``y``, ``body``, ``cp`` [, ``cf``]."""
    from jaxaero.post.forces import surface

    atlas = atlas_of(grid, G, not flow.inviscid)
    return surface(to_canvas(U, atlas), atlas, flow)


def wave_speeds(U, grid, gamma, viscosity=None, model=None):
    """``face_wave_speeds`` of a block-shaped state, on the block's faces."""
    from jaxaero.numerics.residual import face_wave_speeds

    atlas = atlas_of(grid, G, viscosity is not None)
    li, lj = face_wave_speeds(to_canvas(U, atlas), atlas, gamma, viscosity, model)
    fi, fj = face_slices(atlas)
    return li[fi], lj[fj]


def wall_slice(grid) -> slice:
    """The cells of the one ``wall`` segment on ``j_min``."""
    (seg,) = [s for s in grid.sides.j_min if getattr(s, "patch", None) == "wall"]
    return slice(seg.start, seg.stop)


def n_symmetry(grid) -> int:
    """Cells of the ``symmetry`` segment on ``j_min`` (the flat plate)."""
    (seg,) = [s for s in grid.sides.j_min if getattr(s, "patch", None) == "symmetry"]
    return seg.stop - seg.start


class CLayout:
    """A single-block C-grid's ``j = 0`` line, read off its sides: lower cut,
    wall, upper cut, cell ``i`` of a cut facing ``ni - 1 - i``."""

    def __init__(self, grid):
        cut, wall, _ = grid.sides.j_min
        self.n_wake, self.n_airfoil, self.nj = cut.stop, wall.stop - wall.start, grid.nj
        self.ni = grid.ni
        self.wall = slice(wall.start, wall.stop)

    def wake_partner(self, i: int) -> int:
        return self.ni - 1 - i


def c_layout(grid) -> CLayout:
    return CLayout(grid)


def read_grid(path, rules=()):
    """The one block of a grid file, patches named (mesh/assemble.py)."""
    from jaxaero.mesh.assemble import PatchRule, load_grid

    mesh, _ = load_grid(path, rules=tuple(PatchRule(**r) for r in rules))
    assert len(mesh) == 1
    return mesh[0]


PLATE_RULES = (
    {"name": "symmetry", "x_max": 0.0, "y_max": 1e-9},
    {"name": "wall", "x_min": 0.0, "y_max": 1e-9},
    {"name": "inflow", "x_max": -0.33333},
    {"name": "outflow", "x_min": 2.0},
    {"name": "farfield"},
)
"""The TMR flat plate's patches (its grid names none)."""


def body_area(grid) -> float:
    """Area enclosed by the ``wall`` nodes on ``j_min``."""
    import numpy as np

    from jaxaero.mesh.block import polygon_area

    sl = wall_slice(grid)
    return abs(polygon_area(np.asarray(grid.x_node)[sl.start:sl.stop + 1, 0],
                            np.asarray(grid.y_node)[sl.start:sl.stop + 1, 0]))


def with_conditions(grid, no_slip: bool = True):
    """The grid with its patches given the default conditions."""
    return assign_conditions(grid, conditions(grid, no_slip))[0]


def c_grid_sides(n_wake: int, n_airfoil: int, nj: int) -> BlockSides:
    """A C-grid: ``j_min`` is lower wake cut, ``wall``, upper wake cut, the
    cuts facing each other reversed (cell ``i`` faces ``ni - 1 - i``); the
    rest ``farfield``, the outflow planes included."""
    if n_wake < 1 or n_airfoil < 4 or n_airfoil % 2 or nj < 1:
        raise ValueError(f"a C-grid needs n_wake >= 1, n_airfoil even and >= 4, nj >= 1; "
                         f"got {n_wake}, {n_airfoil}, {nj}")
    ni = 2 * n_wake + n_airfoil
    return rectangle_sides(ni, nj, j_max="farfield", i_min="farfield", i_max="farfield",
                           j_min=(Interface(0, n_wake, 0, "j_min", ni - n_wake, True),
                                  Boundary("wall", n_wake, n_wake + n_airfoil),
                                  Interface(ni - n_wake, ni, 0, "j_min", 0, True)))


# --- 3D test geometry (moved from the library: only tests use them) ---------------


def pole_sides(shape: tuple[int, int, int], *, wall: str = "wall", farfield: str = "farfield",
               root: str = "symmetry") -> Sides3D:
    """The single-block wing topology of TMR's ONERA M6 generator: ``i`` from
    a pole (the tip, ``i_min``) to the root plane (``i_max``), ``j`` from the
    wall to the far field, ``k`` round the section with an O seam. ``nk`` even."""
    ni, nj, nk = shape
    if nk % 2:
        raise ValueError(f"a pole needs an even number of cells round it, got nk = {nk}")
    seam = lambda other: (InterfaceRect((0, 0), (ni, nj), 0, other, (0, 0)),)   # noqa: E731
    return box_sides(shape, j_min=wall, j_max=farfield, i_max=root,
                     i_min=(InterfaceRect((0, 0), (nj, nk), 0, "i_min", (0, nk // 2), period=(0, nk)),),
                     k_min=seam("k_max"), k_max=seam("k_min"))


def extruded_sides(sides: BlockSides, nk: int, ends: str = "symmetry") -> Sides3D:
    """A 2D block's sides carried across ``nk`` cells in ``k``, the two end
    planes the patch ``ends``: the 2D segment ``[start, stop)`` along a side
    becomes the rectangle ``[start, stop) x [0, nk)``."""
    def carry(seg):
        if isinstance(seg, Boundary):
            return BoundaryRect(seg.patch, (seg.start, 0), (seg.stop, nk), seg.kind)
        assert isinstance(seg, Interface)
        return InterfaceRect((seg.start, 0), (seg.stop, nk), seg.block, seg.side,
                             (seg.partner_start + (seg.stop - seg.start - 1 if seg.reversed else 0), 0),
                             flip=(seg.reversed, False))
    rects = {s: tuple(carry(seg) for seg in sides.side(s)) for s in ("j_min", "j_max", "i_min", "i_max")}
    return box_sides((sides.ni, sides.nj, nk), k_min=ends, k_max=ends, **rects)


def extrude(grid, nk: int, dz: float = 1.0, ends: str = "symmetry"):
    """A 2D block extruded ``nk`` cells of ``dz`` along ``z``, the end planes
    the patch ``ends``: the 2D flow in every plane is then a 3D solution, the
    check that the 3D operator contains the 2D one."""
    x, y = (np.asarray(c) for c in grid.nodes)
    z = dz * np.arange(nk + 1)
    shape = x.shape + (nk + 1,)
    nodes = np.stack([np.broadcast_to(x[..., None], shape), np.broadcast_to(y[..., None], shape),
                      np.broadcast_to(z, shape)])
    return Grid.from_node_array(nodes, extruded_sides(grid.sides, nk, ends))


def quad_distance(p: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Distance from ``p (..., 3)`` to quadrilaterals ``(..., 4, 3)``."""
    a, b, c, d = (quads[..., k, :] for k in range(4))
    return np.minimum(triangle_distance(p, a, b, c), triangle_distance(p, a, c, d))
