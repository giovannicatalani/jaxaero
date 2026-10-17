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
3. **Elliptic smoothing** of each block, its sides fixed (Winslow's
   equations with Thomas-Middlecoff control functions, AIAA J. 18 (1980) 652):

       alpha (r_xx + phi r_x) - 2 beta r_xy + gamma (r_yy + psi r_y) = 0,
       alpha = |r_y|^2,  beta = r_x . r_y,  gamma = |r_x|^2

   (``x``, ``y`` the block indices ``xi``, ``eta``), ``phi`` and ``psi``
   taken from the spacing along the block's sides, ``phi`` interpolated
   between bottom and top, ``psi`` between left and right, so the sides'
   clustering carries into the interior; or ``psi`` from the cell heights
   wanted off the bottom and top (spacing control, as GRAPE's: Sorenson,
   NASA TM-81198, 1980, here without its iteration), so a block continues
   the cells of the layer it is built on. Each Picard step freezes the
   coefficients and solves the linear system. The edges between blocks are
   not moved: their shape and spacing are the template's (smoothing them too,
   with stencils across the blocks, let corners collapse where three blocks
   meet; docs/dev/notes.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

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


def chain_values(chain: Chain, values: dict[str, np.ndarray]) -> np.ndarray:
    """Per-node values along a side, its edges' joined (the shared end once, from the earlier edge)."""
    parts = [values[e][::-1] if rev else values[e] for e, rev in chain]
    return np.concatenate([parts[0]] + [p[1:] for p in parts[1:]])


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


def _spaced(grid: np.ndarray, first: np.ndarray, last: np.ndarray, limit: float) -> np.ndarray:
    """``psi`` along every ``j`` line from the distribution it should have:
    Vinokur's with end intervals ``first`` and ``last`` ``(ni,)`` over the
    distance between the line's two boundary nodes (spacing control at the
    bottom and top sides; the boundary alone sets it)."""
    ni, nj = grid.shape[:2]
    length = np.hypot(*(grid[:, -1] - grid[:, 0]).T)
    psi = np.empty((ni, nj))
    for i in range(ni):
        s = vinokur(nj - 1, min(first[i] / length[i], 0.5), min(last[i] / length[i], 0.5))
        d1, d2 = 0.5 * (s[2:] - s[:-2]), s[2:] - 2.0 * s[1:-1] + s[:-2]
        psi[i, 1:-1] = -d2 / d1
        psi[i, 0], psi[i, -1] = psi[i, 1], psi[i, -2]
    return np.clip(psi, -limit, limit)


def _stencil(ni: int, nj: int):
    """Interior node ids and their eight neighbours (W, E, S, N, SW, SE, NW, NE)."""
    ids = np.arange(ni * nj).reshape(ni, nj)
    centre = ids[1:-1, 1:-1].ravel()
    nb = np.stack([ids[:-2, 1:-1], ids[2:, 1:-1], ids[1:-1, :-2], ids[1:-1, 2:],
                   ids[:-2, :-2], ids[2:, :-2], ids[:-2, 2:], ids[2:, 2:]], axis=-1).reshape(-1, 8)
    return centre, nb


def _picard(X: np.ndarray, phi: np.ndarray, psi: np.ndarray, iterations: int, tol: float,
            relax: float) -> np.ndarray:
    """Winslow's equations for the interior of ``X`` (its boundary fixed), the
    controls ``phi``, ``psi`` at the interior nodes: Picard steps, each a
    sparse solve with the coefficients frozen, under-relaxed by ``relax``."""
    ni, nj = X.shape[:2]
    X = X.copy()
    centre, nb = _stencil(ni, nj)
    phi, psi = phi[1:-1, 1:-1].ravel(), psi[1:-1, 1:-1].ravel()
    local = np.full(ni * nj, -1)
    local[centre] = np.arange(centre.size)
    cols = local[nb]
    inside = cols >= 0
    rows = np.repeat(np.arange(centre.size)[:, None], 8, axis=1)
    flat = X.reshape(-1, 2)
    extent = np.ptp(flat, axis=0).max()
    for _ in range(iterations):
        W, E, S, N, SW, SE, NW, NE = (flat[nb[:, k]] for k in range(8))
        r_x, r_y = 0.5 * (E - W), 0.5 * (N - S)
        alpha, gamma = np.sum(r_y**2, axis=1), np.sum(r_x**2, axis=1)
        beta = np.sum(r_x * r_y, axis=1)
        coef = np.stack([alpha * (1 - 0.5 * phi), alpha * (1 + 0.5 * phi), gamma * (1 - 0.5 * psi),
                         gamma * (1 + 0.5 * psi), -0.5 * beta, 0.5 * beta, 0.5 * beta, -0.5 * beta], axis=1)
        m = centre.size
        A = coo_matrix((np.concatenate([2.0 * (alpha + gamma), -coef[inside]]),
                        (np.concatenate([np.arange(m), rows[inside]]),
                         np.concatenate([np.arange(m), cols[inside]]))), shape=(m, m)).tocsc()
        rhs = np.einsum("mk,mkd->md", np.where(inside, 0.0, coef), flat[nb])
        move = splu(A).solve(rhs) - flat[centre]
        flat[centre] += relax * move
        if np.abs(move).max() <= tol * extent:
            break
    return X


def winslow(grid: np.ndarray, first: np.ndarray | None = None, last: np.ndarray | None = None,
            iterations: int = 200, tol: float = 1e-9, relax: float = 0.5, control_limit: float = 1.0,
            rounds: int = 3, spacing_tol: float = 0.05) -> np.ndarray:
    """The block's interior by Winslow's equations with Thomas-Middlecoff
    control functions (clipped to ``control_limit``), its boundary fixed
    (``_picard``). Given ``first`` and ``last`` ``(ni,)``, the cell heights
    wanted off the bottom and top sides, ``psi`` comes from them (``_spaced``)
    and is corrected, as GRAPE's controls are, up to ``rounds`` times: the
    heights asked of ``_spaced`` scaled by the square root of wanted over
    obtained, until every one is within ``spacing_tol``."""
    ni, nj = grid.shape[:2]
    if ni < 3 or nj < 3:
        return grid.copy()
    phi, psi = _control(grid, control_limit)
    if first is None:
        return _picard(grid, phi, psi, iterations, tol, relax)
    X, asked = grid, [np.asarray(first, float).copy(), np.asarray(last, float).copy()]
    for _ in range(rounds):
        X = _picard(X, phi, _spaced(grid, *asked, control_limit), iterations, tol, relax)
        got = [np.hypot(*(X[1:-1, 1] - X[1:-1, 0]).T), np.hypot(*(X[1:-1, -1] - X[1:-1, -2]).T)]
        ratio = [w[1:-1] / g for w, g in zip((first, last), got)]
        if max(np.abs(r - 1.0).max() for r in ratio) <= spacing_tol:
            break
        for a, r, w in zip(asked, ratio, (first, last)):
            a[1:-1] = np.clip(a[1:-1] * np.sqrt(r), 0.01 * w[1:-1], 100.0 * w[1:-1])
    return X


def smooth(grids: dict[str, np.ndarray], spacing: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
           **kwargs) -> dict[str, np.ndarray]:
    """Every block by ``winslow``, its sides (the graph's edges) fixed;
    ``spacing``: a block's wanted cell heights off its bottom and top.

    Raises:
        ValueError: if a block folds.
    """
    spacing = spacing or {}
    out = {name: winslow(g, *spacing.get(name, (None, None)), **kwargs) for name, g in grids.items()}
    unfolded(out)
    return out


def unfolded(grids: dict[str, np.ndarray]) -> None:
    """Raise ``ValueError`` naming the first block whose cells are not all of one sign."""
    for name, g in grids.items():
        a = cell_areas(g)
        bad = min(int(np.sum(a <= 0)), int(np.sum(a >= 0)))
        if bad:
            raise ValueError(f"block {name!r} folds: {bad} of {a.size} cells inverted")


def cell_areas(grid: np.ndarray) -> np.ndarray:
    """Signed cell areas ``(ni, nj)`` (positive for a right-handed block)."""
    x, y = grid[..., 0], grid[..., 1]
    d1x, d1y = x[1:, 1:] - x[:-1, :-1], y[1:, 1:] - y[:-1, :-1]
    d2x, d2y = x[:-1, 1:] - x[1:, :-1], y[:-1, 1:] - y[1:, :-1]
    return 0.5 * (d1x * d2y - d1y * d2x)


def raw_blocks(grids: dict[str, np.ndarray]) -> list[RawBlock]:
    return [RawBlock(g[..., 0].copy(), g[..., 1].copy(), name=name) for name, g in grids.items()]
