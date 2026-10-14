"""Grid sequencing: a mesh coarsened by taking every other node, and a coarse
solution carried back to the fine mesh.

Coarsening keeps the topology: every boundary and interface segment keeps its
patch, partner and orientation, its cell indices halved. A coarse cell is the
union of its ``2^d`` children, so prolongation by injection (each child
takes its parent's value) is conservative.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from jax import Array

from .atlas import Atlas
from .block import SIDES, BlockSides, Boundary, Grid
from .block3d import SIDES_3D, BoundaryRect, Sides3D
from .mesh import Mesh, as_mesh


def _half(n: int, what: str) -> int:
    if n % 2:
        raise ValueError(f"cannot coarsen the grid: {what} is {n}, odd (every block's cell counts and every "
                         "patch and interface limit must be even)")
    return n // 2


def _segment(seg, side: str):
    if isinstance(seg, Boundary):
        return replace(seg, start=_half(seg.start, f"a {side} limit"), stop=_half(seg.stop, f"a {side} limit"))
    return replace(seg, start=_half(seg.start, f"a {side} limit"), stop=_half(seg.stop, f"a {side} limit"),
                   partner_start=_half(seg.partner_start, f"a {side} interface offset"))


def _rect(rect, side: str):
    pair = lambda v, what: tuple(_half(x, f"a {side} {what}") for x in v)  # noqa: E731
    if isinstance(rect, BoundaryRect):
        return replace(rect, start=pair(rect.start, "limit"), stop=pair(rect.stop, "limit"))
    # ``origin`` is the partner cell facing ``start``; flipped, the coarse
    # start's two children face ``origin`` and ``origin - 1``.
    origin = tuple(_half(o + 1, f"{side} interface origin + 1") - 1 if f else _half(o, f"{side} interface origin")
                   for o, f in zip(rect.origin, rect.flip))
    return replace(rect, start=pair(rect.start, "limit"), stop=pair(rect.stop, "limit"), origin=origin,
                   period=pair(rect.period, "interface period"))


def coarsen(mesh: Grid | Mesh) -> Mesh:
    """Every block with every other node.

    Raises:
        ValueError: a cell count, a segment limit or an interface offset is odd.
    """
    out = []
    for block in as_mesh(mesh):
        s = block.sides
        if isinstance(s, Sides3D):
            sides = Sides3D(tuple(_half(n, "a cell count") for n in s.shape),
                            *(tuple(_rect(r, name) for r in s.side(name))
                              for name in SIDES_3D))
        else:
            sides = BlockSides(_half(s.ni, "a cell count"), _half(s.nj, "a cell count"),
                               *(tuple(_segment(seg, name) for seg in s.side(name))
                                 for name in SIDES))
        every_other = (slice(None),) + (slice(None, None, 2),) * block.dim
        out.append(Grid.from_node_array(np.asarray(block.nodes)[every_other], sides, np.asarray(block.nodes).dtype))
    return tuple(out)


def prolong(U: Array, coarse: Atlas, fine: Atlas, start: Array) -> Array:
    """The coarse canvas solution ``U`` on the fine canvas: each fine cell
    takes its parent's value; cells outside every block keep ``start``'s."""
    canvas = start
    for block, window in zip(coarse.unpack(U), fine.windows):
        for axis in range(1, block.ndim):
            block = jnp.repeat(block, 2, axis=axis)
        canvas = canvas.at[(Ellipsis,) + tuple(window)].set(block)
    return canvas
