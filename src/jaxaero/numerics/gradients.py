"""Gradients at cell faces, for the viscous flux.

The diamond construction, in structured form, in 2D and 3D. Across a face lie
the two cells either side; within it, its nodes. The face gradient is the
unique linear field that reproduces the differences along the line joining
the two cell centres and along the face's node diagonals,

    grad(phi) . d_cells = phi_R - phi_L,
    grad(phi) . d_m     = phi_N(m)' - phi_N(m),     m = 1 .. d-1,

a ``d x d`` solve per face; in 2D ``d_1`` is the face itself (two nodes), in
3D the two diagonals of the quadrilateral. Node values *and* positions are
the average of the ``2^d`` surrounding cell centres, which makes the result
exact for any linear field on any hexahedral grid, however stretched or
skewed -- the property the unit tests pin. Using the true mesh node positions
with averaged values would not be: on a graded grid the average of the
centres is not at the node.

Everything is slicing on arrays carrying one ghost layer, so a face on the
wall, on the far field, across an O-grid seam, a C-grid wake cut or a pole is
treated like any other. The ghost *values* come from the boundary conditions;
the ghost *centres* from the same side segments (mesh/atlas.py).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _corner_average(a: Array, dim: int) -> Array:
    """Average of the ``2^d`` cells around each node, over the last ``d`` axes."""
    if dim == 2:
        return 0.25 * (a[..., :-1, :-1] + a[..., 1:, :-1] + a[..., :-1, 1:] + a[..., 1:, 1:])
    for axis in range(a.ndim - dim, a.ndim):
        lo, hi = [slice(None)] * a.ndim, [slice(None)] * a.ndim
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        a = 0.5 * (a[tuple(lo)] + a[tuple(hi)])
    return a


def _solve2(d1x, d1y, d2x, d2y, dphi1, dphi2) -> Array:
    """``grad . d1 = dphi1``, ``grad . d2 = dphi2`` by Cramer's rule: ``(2, ...)``."""
    det = d1x * d2y - d2x * d1y
    return jnp.stack([(dphi1 * d2y - dphi2 * d1y) / det, (d1x * dphi2 - d2x * dphi1) / det])


def _cross(a: Array, b: Array) -> Array:
    return jnp.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _solve3(d1: Array, d2: Array, d3: Array, dphi1, dphi2, dphi3) -> Array:
    """``grad . d_m = dphi_m``: ``grad = sum_m dphi_m (d_m+1 x d_m+2) / det``, ``(3, nvar, ...)``."""
    c23, c31, c12 = _cross(d2, d3), _cross(d3, d1), _cross(d1, d2)
    det = d1[0] * c23[0] + d1[1] * c23[1] + d1[2] * c23[2]
    return (dphi1 * c23[:, None] + dphi2 * c31[:, None] + dphi3 * c12[:, None]) / det[None, None]


def face_gradients(phi: Array, centres: Array) -> tuple[Array, ...]:
    """Per face family ``a``, the gradient ``(d, nvar, *face shape)`` (direction
    first) of ``phi (nvar, *(shape + 2))`` (one ghost layer; ``centres (d, ...)``
    likewise); the face shape is the cell shape with one more along ``a``."""
    dim = centres.shape[0]
    shape = tuple(n - 2 for n in centres.shape[1:])
    xn, phin = _corner_average(centres, dim), _corner_average(phi, dim)
    out = []
    for a in range(dim):
        # Face f along a lies between padded cells f and f + 1, and at node f.
        def cell(offset):
            index = [slice(1, 1 + n) for n in shape]
            index[a] = slice(offset, offset + shape[a] + 1)
            return tuple(index)

        def node(shifts):
            """Nodes of the faces' corner ``shifts`` (one 0/1 per other axis)."""
            index, others = [], iter(shifts)
            for b in range(dim):
                start = 0 if b == a else next(others)
                index.append(slice(start, start + shape[b] + (b == a)))
            return tuple(index)

        left, right = cell(0), cell(1)
        d_cells = centres[(slice(None),) + right] - centres[(slice(None),) + left]
        dphi_cells = phi[(slice(None),) + right] - phi[(slice(None),) + left]
        if dim == 2:
            n0, n1 = node((0,)), node((1,))
            d_node = xn[(slice(None),) + n1] - xn[(slice(None),) + n0]
            dphi_node = phin[(slice(None),) + n1] - phin[(slice(None),) + n0]
            out.append(_solve2(d_cells[0], d_cells[1], d_node[0], d_node[1], dphi_cells, dphi_node))
            continue
        diagonals = [(node((0, 0)), node((1, 1))), (node((1, 0)), node((0, 1)))]
        d = [xn[(slice(None),) + hi] - xn[(slice(None),) + lo] for lo, hi in diagonals]
        dphi = [phin[(slice(None),) + hi] - phin[(slice(None),) + lo] for lo, hi in diagonals]
        out.append(_solve3(d_cells, d[0], d[1], dphi_cells, dphi[0], dphi[1]))
    return tuple(out)
