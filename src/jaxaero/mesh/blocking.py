"""Multi-block structured grids from a block graph: counts, transfinite
interpolation, elliptic smoothing. No geometry knowledge: a template
(mesh/highlift.py) supplies the graph and the nodes of its edges.

A ``Block`` has four sides, each a chain of edges: ``bottom`` and ``top``
run along ``i`` (at ``j = 0`` and ``j = nj``), ``left`` and ``right`` along
``j`` (at ``i = 0`` and ``i = ni``); an edge is used forward or reversed.

1. **Counts.** Opposite sides carry the same number of cells. Every edge
   asks for a count (rounded up to a multiple of ``multiple``); a side short
   of its opposite gains the difference on its longest edge, until no side
   is short (a monotone fixed point; a chain of single-edge sides is a
   union-find class taking the largest request of its members).
2. **Transfinite interpolation** of each block from its four sides (the Coons
   patch, Thompson, Warsi and Mastin, *Numerical Grid Generation*, 1985,
   ch. 3), blended by normalised arc length so the sides' clustering reaches
   the interior; a bilinear map is reproduced exactly.
3. **Elliptic smoothing** of all blocks together (Winslow's equations with
   Thomas-Middlecoff control functions, AIAA J. 18 (1980) 652):

       alpha (r_xx + phi r_x) - 2 beta r_xy + gamma (r_yy + psi r_y) = 0,
       alpha = |r_y|^2,  beta = r_x . r_y,  gamma = |r_x|^2

   (``x``, ``y`` the block indices ``xi``, ``eta``), ``phi`` and ``psi``
   taken from the spacing along each block's sides, ``phi`` interpolated
   between bottom and top, ``psi`` between left and right. Nodes on edges
   marked fixed (walls, layer contours, an outer loop) and every edge end
   stay put; the other edges' nodes are free, each with a stencil spanning
   the two blocks it joins, so the grid is smooth across them. Each Picard
   step freezes the coefficients and solves the linear system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .cgns import RawBlock
from .generate import vinokur

SIDES = ("bottom", "right", "top", "left")
Chain = tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class Block:
    """Four sides, each a chain of ``(edge, reversed)``."""

    name: str
    bottom: Chain
    right: Chain
    top: Chain
    left: Chain

    def side(self, which: str) -> Chain:
        return getattr(self, which)

    @property
    def edges(self) -> set[str]:
        return {e for s in SIDES for e, _ in self.side(s)}


# --- Edge geometry -------------------------------------------------------------------


def arc_length(points: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])


def distribute(curve: np.ndarray, n: int, ds0: float | None = None, ds1: float | None = None) -> np.ndarray:
    """``n + 1`` nodes along the polyline ``curve`` ``(m, 2)`` (splined in arc
    length), end intervals ``ds0`` and ``ds1`` (lengths; ``None``: the mean
    interval) by Vinokur's stretching."""
    s = arc_length(curve)
    L = s[-1]
    mean = L / n
    t = vinokur(n, (ds0 or mean) / L, (ds1 or mean) / L) * L
    if len(curve) == 2:
        return curve[0] + (t / L)[:, None] * (curve[1] - curve[0])
    out = np.stack([CubicSpline(s, curve[:, 0])(t), CubicSpline(s, curve[:, 1])(t)], axis=1)
    out[0], out[-1] = curve[0], curve[-1]
    return out


def hermite(p0, t0, p1, t1, samples: int = 200) -> np.ndarray:
    """A cubic from ``p0`` leaving along ``t0`` to ``p1`` arriving along ``t1``
    (unit tangents, scaled by the chord), ``samples`` points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.hypot(*(p1 - p0))
    t0, t1 = L * np.asarray(t0) / np.hypot(*t0), L * np.asarray(t1) / np.hypot(*t1)
    u = np.linspace(0.0, 1.0, samples)[:, None]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


# --- Counts ----------------------------------------------------------------------------


def solve_counts(blocks: list[Block], requested: dict[str, int], multiple: int = 1,
                 max_rounds: int = 1000) -> dict[str, int]:
    """Cells per edge: each at least its request, a multiple of ``multiple``,
    opposite sides of every block equal.

    Raises:
        ValueError: an edge without a request, or counts that never settle
        (a block graph whose constraints contradict each other).
    """
    missing = sorted(set().union(*(b.edges for b in blocks)) - set(requested))
    if missing:
        raise ValueError(f"edges without a requested count: {missing}")
    counts = {e: -(-max(int(n), 1) // multiple) * multiple for e, n in requested.items()}
    pairs = [(b.side(a), b.side(c)) for b in blocks for a, c in (("bottom", "top"), ("left", "right"))]
    for _ in range(max_rounds):
        changed = False
        for one, other in pairs:
            a, b = sum(counts[e] for e, _ in one), sum(counts[e] for e, _ in other)
            if a != b:
                short = one if a < b else other
                counts[max((e for e, _ in short), key=lambda e: counts[e])] += abs(a - b)
                changed = True
        if not changed:
            return counts
    raise ValueError("cell counts do not settle: opposite sides of the block graph contradict each other")


# --- Transfinite interpolation --------------------------------------------------------


def chain_nodes(chain: Chain, nodes: dict[str, np.ndarray]) -> np.ndarray:
    """The nodes along a side, its edges joined (the shared end node once).

    Raises:
        ValueError: if consecutive edges do not meet.
    """
    parts = [nodes[e][::-1] if rev else nodes[e] for e, rev in chain]
    for (e, _), a, b in zip(chain, parts[:-1], parts[1:]):
        if np.hypot(*(a[-1] - b[0])) > 1e-9 * max(np.ptp(a, axis=0).max(), 1e-300):
            raise ValueError(f"edge {e!r} does not end where the next edge of its side begins")
    return np.concatenate([parts[0]] + [p[1:] for p in parts[1:]])


def tfi(bottom: np.ndarray, right: np.ndarray, top: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Coons patch ``(ni + 1, nj + 1, 2)`` of four sides (bottom and top along
    ``i``, left and right along ``j``), blended by normalised arc length:
    ``xi = (1 - eta) s_b + eta s_t``, ``eta = (1 - xi) s_l + xi s_r``.

    Raises:
        ValueError: if the sides' corners do not meet or opposite counts differ.
    """
    if len(bottom) != len(top) or len(left) != len(right):
        raise ValueError(f"opposite sides differ: {len(bottom)} vs {len(top)} nodes along i, "
                         f"{len(left)} vs {len(right)} along j")
    scale = max(np.ptp(np.concatenate([bottom, top, left, right]), axis=0).max(), 1e-300)
    for p, q in ((bottom[0], left[0]), (bottom[-1], right[0]), (top[0], left[-1]), (top[-1], right[-1])):
        if np.hypot(*(p - q)) > 1e-9 * scale:
            raise ValueError(f"block sides do not meet at a corner: {p} vs {q}")
    sb, st = (arc_length(c) / arc_length(c)[-1] for c in (bottom, top))
    sl, sr = (arc_length(c) / arc_length(c)[-1] for c in (left, right))
    SB, SL = np.meshgrid(sb, sl, indexing="ij")
    ST, SR = np.meshgrid(st, sr, indexing="ij")
    d = 1.0 - (ST - SB) * (SR - SL)
    xi = (SB + SL * (ST - SB)) / d
    eta = (SL + SB * (SR - SL)) / d
    xi, eta = xi[..., None], eta[..., None]
    B, T = bottom[:, None, :], top[:, None, :]
    L, R = left[None, :, :], right[None, :, :]
    corners = ((1 - xi) * (1 - eta) * bottom[0] + xi * (1 - eta) * bottom[-1]
               + (1 - xi) * eta * top[0] + xi * eta * top[-1])
    out = (1 - eta) * B + eta * T + (1 - xi) * L + xi * R - corners
    out[:, 0], out[:, -1], out[0, :], out[-1, :] = bottom, top, left, right
    return out


def fill(blocks: list[Block], nodes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every block by ``tfi`` of its sides' nodes."""
    return {b.name: tfi(*(chain_nodes(b.side(s), nodes) for s in SIDES)) for b in blocks}


# --- Elliptic smoothing ------------------------------------------------------------------


def _global_ids(blocks: list[Block], nodes: dict[str, np.ndarray], grids: dict[str, np.ndarray]):
    """One id per distinct node: edge nodes shared by every block using the
    edge, edge ends merged where they coincide; block-interior nodes their own."""
    names = sorted(nodes)
    ends = np.array([nodes[e][k] for e in names for k in (0, -1)])
    extent = max(np.ptp(ends, axis=0).max(), 1e-300)
    parent = list(range(len(ends)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(ends).query_pairs(1e-9 * extent):
        parent[root(a)] = root(b)
    end_id = {r: k for k, r in enumerate(sorted({root(a) for a in range(len(ends))}))}
    next_id = len(end_id)
    edge_ids = {}
    for n, e in enumerate(names):
        m = len(nodes[e])
        ids = np.empty(m, dtype=np.int64)
        ids[0], ids[-1] = end_id[root(2 * n)], end_id[root(2 * n + 1)]
        ids[1:-1] = np.arange(next_id, next_id + m - 2)
        next_id += m - 2
        edge_ids[e] = ids
    n_edge_nodes = next_id
    block_ids = {}
    for b in blocks:
        ni, nj = grids[b.name].shape[:2]
        ids = np.empty((ni, nj), dtype=np.int64)
        ids[1:-1, 1:-1] = np.arange(next_id, next_id + (ni - 2) * (nj - 2)).reshape(ni - 2, nj - 2)
        next_id += (ni - 2) * (nj - 2)
        line = lambda s: np.concatenate([(edge_ids[e][::-1] if r else edge_ids[e])[(k > 0):]  # noqa: E731
                                         for k, (e, r) in enumerate(b.side(s))])
        ids[:, 0], ids[:, -1], ids[0, :], ids[-1, :] = line("bottom"), line("top"), line("left"), line("right")
        block_ids[b.name] = ids
    X = np.empty((next_id, 2))
    for b in blocks:
        X[block_ids[b.name]] = grids[b.name]
    return X, edge_ids, block_ids, n_edge_nodes


def _side_lines(ids: np.ndarray, side: str):
    """The node ids along a side and one row inward, in the block's index order;
    whether the side runs along ``i``; the sign of the inward index direction."""
    return {"bottom": (ids[:, 0], ids[:, 1], True, 1), "top": (ids[:, -1], ids[:, -2], True, -1),
            "left": (ids[0, :], ids[1, :], False, 1), "right": (ids[-1, :], ids[-2, :], False, -1)}[side]


def _control(grid: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Thomas-Middlecoff ``phi`` (along ``i``) and ``psi`` (along ``j``) over a
    block: ``-r_x . r_xx / |r_x|^2`` on its bottom and top sides, interpolated
    linearly in ``j``; the same along ``j`` on left and right, interpolated in ``i``."""

    def along(line):
        f = np.zeros(len(line))
        d1 = 0.5 * (line[2:] - line[:-2])
        d2 = line[2:] - 2.0 * line[1:-1] + line[:-2]
        f[1:-1] = -np.sum(d1 * d2, axis=1) / np.sum(d1 * d1, axis=1)
        f[0], f[-1] = f[1], f[-2]
        return np.clip(f, -limit, limit)

    ni, nj = grid.shape[:2]
    eta = np.linspace(0.0, 1.0, nj)[None, :]
    xi = np.linspace(0.0, 1.0, ni)[:, None]
    phi = (1 - eta) * along(grid[:, 0])[:, None] + eta * along(grid[:, -1])[:, None]
    psi = (1 - xi) * along(grid[0, :])[None, :] + xi * along(grid[-1, :])[None, :]
    return phi, psi


@dataclass(frozen=True)
class _Stencils:
    free: np.ndarray        # (m,) global ids of the free nodes
    nb: np.ndarray          # (m, 8) ids of W, E, S, N, SW, SE, NW, NE
    phi: np.ndarray         # (m,)
    psi: np.ndarray


def _stencils(blocks, grids, edge_ids, block_ids, free_edges, limit) -> _Stencils:
    free, nb, phi, psi = [], [], [], []
    controls = {b.name: _control(grids[b.name], limit) for b in blocks}
    for b in blocks:                                   # block interiors
        ids = block_ids[b.name]
        ph, ps = controls[b.name]
        c = (slice(1, -1), slice(1, -1))
        free.append(ids[c].ravel())
        nb.append(np.stack([ids[:-2, 1:-1], ids[2:, 1:-1], ids[1:-1, :-2], ids[1:-1, 2:],
                            ids[:-2, :-2], ids[2:, :-2], ids[:-2, 2:], ids[2:, 2:]], axis=-1).reshape(-1, 8))
        phi.append(ph[c].ravel())
        psi.append(ps[c].ravel())
    seen: dict[int, list] = {}                         # free-edge nodes: where each one lies on a side
    for b in blocks:
        ph, ps = controls[b.name]
        for s in SIDES:
            line0, line1, along_i, inward = _side_lines(block_ids[b.name], s)
            index = {"bottom": (slice(None), 0), "top": (slice(None), -1),
                     "left": (0, slice(None)), "right": (-1, slice(None))}[s]
            f_along = (ph if along_i else ps)[index]
            f_across = inward * (ps if along_i else ph)[index]
            for e, _ in b.side(s):
                if e not in free_edges:
                    continue
                for g in edge_ids[e][1:-1]:
                    k = int(np.flatnonzero(line0 == g)[0])
                    seen.setdefault(int(g), []).append((line0, line1, k, f_along[k], f_across[k]))
    for g, places in seen.items():
        if len(places) != 2:
            raise ValueError(f"a free edge node lies on {len(places)} block side(s); a free edge must "
                             "join exactly two blocks")
        (a0, a1, ka, pa, qa), (b0, b1, kb, pb, qb) = places
        same = b0[kb + 1] == a0[ka + 1]
        sw, se = (b1[kb - 1], b1[kb + 1]) if same else (b1[kb + 1], b1[kb - 1])
        free.append(np.array([g]))
        nb.append(np.array([[a0[ka - 1], a0[ka + 1], b1[kb], a1[ka], sw, se, a1[ka - 1], a1[ka + 1]]]))
        phi.append(np.array([0.5 * (pa + (pb if same else -pb))]))
        psi.append(np.array([0.5 * (qa - qb)]))
    return _Stencils(np.concatenate(free), np.concatenate(nb), np.concatenate(phi), np.concatenate(psi))


def smooth(blocks: list[Block], nodes: dict[str, np.ndarray], grids: dict[str, np.ndarray],
           fixed: set[str], iterations: int = 200, tol: float = 1e-9, relax: float = 0.5,
           control_limit: float = 1.0) -> dict[str, np.ndarray]:
    """``grids`` (from ``fill``) smoothed together by Winslow's equations with
    Thomas-Middlecoff control functions (clipped to ``control_limit``): nodes
    on ``fixed`` edges and all edge ends stay; Picard steps, each a linear
    solve, under-relaxed by ``relax``, until the largest move is under
    ``tol`` of the grid's extent.

    Raises:
        ValueError: if the result folds (a block with cells of both signs).
    """
    free_edges = set(nodes) - set(fixed)
    X, edge_ids, block_ids, _ = _global_ids(blocks, nodes, grids)
    st = _stencils(blocks, grids, edge_ids, block_ids, free_edges, control_limit)
    m = len(st.free)
    local = np.full(len(X), -1)
    local[st.free] = np.arange(m)
    extent = np.ptp(X, axis=0).max()
    for _ in range(iterations):
        W, E, S, N, SW, SE, NW, NE = (X[st.nb[:, k]] for k in range(8))
        r_x, r_y = 0.5 * (E - W), 0.5 * (N - S)
        alpha, gamma = np.sum(r_y**2, axis=1), np.sum(r_x**2, axis=1)
        beta = np.sum(r_x * r_y, axis=1)
        coef = np.stack([alpha * (1 - 0.5 * st.phi), alpha * (1 + 0.5 * st.phi),
                         gamma * (1 - 0.5 * st.psi), gamma * (1 + 0.5 * st.psi),
                         -0.5 * beta, 0.5 * beta, 0.5 * beta, -0.5 * beta], axis=1)
        diag = 2.0 * (alpha + gamma)
        cols = local[st.nb]
        inside = cols >= 0
        rows = np.repeat(np.arange(m)[:, None], 8, axis=1)
        A = coo_matrix((np.concatenate([diag, -coef[inside]]),
                        (np.concatenate([np.arange(m), rows[inside]]),
                         np.concatenate([np.arange(m), cols[inside]]))), shape=(m, m)).tocsc()
        rhs = np.einsum("mk,mkd->md", np.where(inside, 0.0, coef), X[st.nb])
        new = splu(A).solve(rhs)
        move = new - X[st.free]
        X[st.free] += relax * move
        if np.abs(move).max() <= tol * extent:
            break
    out = {b.name: X[block_ids[b.name]] for b in blocks}
    for name, g in out.items():
        a = cell_areas(g)
        if not (np.all(a > 0) or np.all(a < 0)):
            raise ValueError(f"block {name!r} folds: {int(np.sum(a <= 0))} of {a.size} cells inverted")
    return out


def cell_areas(grid: np.ndarray) -> np.ndarray:
    """Signed cell areas ``(ni, nj)`` (positive for a right-handed block)."""
    x, y = grid[..., 0], grid[..., 1]
    d1x, d1y = x[1:, 1:] - x[:-1, :-1], y[1:, 1:] - y[:-1, :-1]
    d2x, d2y = x[:-1, 1:] - x[1:, :-1], y[:-1, 1:] - y[1:, :-1]
    return 0.5 * (d1x * d2y - d1y * d2x)


def raw_blocks(grids: dict[str, np.ndarray]) -> list[RawBlock]:
    return [RawBlock(g[..., 0].copy(), g[..., 1].copy(), name=name) for name, g in grids.items()]
