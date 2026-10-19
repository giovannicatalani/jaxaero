"""The block-graph core: counts, transfinite interpolation, elliptic smoothing.

Synthetic graphs whose answers are known: a bilinear map, a quarter annulus
in one block and cut along its own lines into three, a 2 x 2 square under a
bump; the output must assemble with coincident interfaces and pass the
quality checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxaero.mesh.assemble import assemble
from jaxaero.mesh.blocking import (Block, cell_areas, chain_nodes, distribute, fill, raw_blocks, smooth,
                                   solve_counts, tfi, unfolded, winslow)
from jaxaero.mesh.generate import geometric
from jaxaero.mesh.quality import _cell_angles, interface_growth, quality


def _arc(r, t0, t1, n):
    t = np.radians(np.linspace(t0, t1, n + 1))
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _ray(theta, radii):
    t = np.radians(theta)
    return np.stack([radii * np.cos(t), radii * np.sin(t)], axis=1)


def test_tfi_reproduces_a_bilinear_map() -> None:
    corners = np.array([[0.0, 0.0], [2.0, 0.3], [-0.2, 1.0], [1.7, 1.6]])     # (0,0), (1,0), (0,1), (1,1)
    u, v = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 6), indexing="ij")
    exact = ((1 - u) * (1 - v))[..., None] * corners[0] + (u * (1 - v))[..., None] * corners[1] \
        + ((1 - u) * v)[..., None] * corners[2] + (u * v)[..., None] * corners[3]
    got = tfi(exact[:, 0], exact[-1, :], exact[:, -1], exact[0, :])
    np.testing.assert_allclose(got, exact, atol=1e-14)


def test_tfi_refuses_sides_that_do_not_meet() -> None:
    line = np.stack([np.linspace(0, 1, 5), np.zeros(5)], axis=1)
    with pytest.raises(ValueError, match="corner"):
        tfi(line, line + [1.0, 0.0], line + [0.0, 1.0], line)


def test_counts_propagate_and_round_up() -> None:
    # Two blocks side by side over a split bottom: [a1 a2] under one block,
    # its top t; the second block beside it, left side shared.
    blocks = [Block("A", (("a1", False), ("a2", False)), (("m", False),), (("t", False),), (("l", False),)),
              Block("B", (("b", False),), (("r", False),), (("u", False),), (("m", False),))]
    counts = solve_counts(blocks, {"a1": 5, "a2": 7, "t": 9, "l": 3, "m": 10, "b": 2, "r": 1, "u": 6}, 4)
    assert all(n % 4 == 0 for n in counts.values())
    assert counts["a1"] + counts["a2"] == counts["t"] and counts["b"] == counts["u"]
    assert counts["l"] == counts["m"] == counts["r"] == 12
    assert counts["t"] >= 16 and counts["a1"] >= 8 and counts["a2"] >= 8
    with pytest.raises(ValueError, match="without a requested count"):
        solve_counts(blocks, {"a1": 4})


def test_contradicting_counts_are_refused() -> None:
    # bottom = [x, y] opposite top = [x]: y would have to vanish
    blocks = [Block("A", (("x", False), ("y", False)), (("r", False),), (("x", False),), (("l", False),))]
    with pytest.raises(ValueError, match="do not settle"):
        solve_counts(blocks, {"x": 2, "y": 2, "l": 2, "r": 2})


def _annulus(cuts, rays=None):
    """A quarter annulus r in [1, 2] cut along ``cuts`` (degrees) into blocks,
    36 cells round, 16 out, geometric from 0.02; ``rays``: the cut lines'
    nodes, else straight."""
    radii = 1.0 + geometric(16, 0.02, 1.0)
    angles = [0.0] + list(cuts) + [90.0]
    nodes, blocks = {}, []
    for k, a in enumerate(angles):
        nodes[f"ray{k}"] = _ray(a, radii)
    for k, r in (rays or {}).items():
        nodes[f"ray{k}"] = r
    for k in range(len(angles) - 1):
        n = round(36 * (angles[k + 1] - angles[k]) / 90.0)
        nodes[f"in{k}"], nodes[f"out{k}"] = _arc(1.0, angles[k], angles[k + 1], n), _arc(2.0, angles[k], angles[k + 1], n)
        blocks.append(Block(f"b{k}", ((f"in{k}", False),), ((f"ray{k + 1}", False),), ((f"out{k}", False),),
                            ((f"ray{k}", False),)))
    return blocks, nodes


def _skew(grid):
    return 90.0 - _cell_angles(grid[..., 0], grid[..., 1])


def _shaken(grids, seed=0):
    """Interiors moved at random, so the smoother has work to do."""
    rng, out = np.random.default_rng(seed), {}
    for name, g in grids.items():
        g = g.copy()
        g[1:-1, 1:-1] += rng.uniform(-0.01, 0.01, g[1:-1, 1:-1].shape)
        out[name] = g
    return out


def test_a_quarter_annulus_is_the_same_grid_in_one_block_or_three() -> None:
    # With the cell heights off the walls given (spacing control, one round:
    # the corrections of later rounds stop at the cuts), every j line takes
    # its control from its own length, so the block equations do not depend
    # on how the annulus is cut: cut along the single block's own lines
    # i = 12 and 24, three blocks reproduce it.
    first, last = 2e-3, 0.1
    blocks, nodes = _annulus([])
    grids = fill(blocks, nodes)
    one = smooth(_shaken(grids), {"b0": (np.full(37, first), np.full(37, last))}, tol=1e-13, iterations=500,
                 rounds=1)["b0"]
    blocks, nodes = _annulus([30.0, 60.0], rays={1: one[12], 2: one[24]})
    spacing = {b.name: (np.full(13, first), np.full(13, last)) for b in blocks}
    three = smooth(_shaken(fill(blocks, nodes), seed=1), spacing, tol=1e-13, iterations=500, rounds=1)
    joined = np.concatenate([three["b0"], three["b1"][1:], three["b2"][1:]], axis=0)
    assert np.max(np.hypot(*(joined - one).transpose(2, 0, 1)) / _nearest(one)) < 1e-6


def _nearest(grid):
    """Each node's distance to its nearest neighbour along i or j."""
    d = np.full(grid.shape[:2], np.inf)
    di = np.hypot(*np.diff(grid, axis=0).transpose(2, 0, 1))
    dj = np.hypot(*np.diff(grid, axis=1).transpose(2, 0, 1))
    d[1:], d[:-1] = np.minimum(d[1:], di), np.minimum(d[:-1], di)
    d[:, 1:], d[:, :-1] = np.minimum(d[:, 1:], dj), np.minimum(d[:, :-1], dj)
    return d


def _bump_square():
    """The unit square under a bump, 2 x 2 blocks round a centre vertex."""
    n = 12
    x = np.linspace(0, 1, 2 * n + 1)
    bump = 0.15 * np.sin(np.pi * x) ** 2
    floor = np.stack([x, bump], axis=1)
    centre = np.array([0.5, 0.5])
    straight = lambda p, q: np.stack([np.linspace(p[0], q[0], n + 1), np.linspace(p[1], q[1], n + 1)], 1)  # noqa: E731
    nodes = {"f0": floor[:n + 1], "f1": floor[n:],
             "r0": straight([1, 0], [1, 0.5]), "r1": straight([1, 0.5], [1, 1]),
             "t1": straight([0.5, 1], [1, 1]), "t0": straight([0, 1], [0.5, 1]),
             "l1": straight([0, 0.5], [0, 1]), "l0": straight([0, 0], [0, 0.5]),
             "cs": distribute(np.array([floor[n], centre]), n), "ce": straight(centre, [1, 0.5]),
             "cn": straight(centre, [0.5, 1]), "cw": straight([0, 0.5], centre)}
    blocks = [Block("sw", (("f0", False),), (("cs", False),), (("cw", False),), (("l0", False),)),
              Block("se", (("f1", False),), (("r0", False),), (("ce", False),), (("cs", False),)),
              Block("nw", (("cw", False),), (("cn", False),), (("t0", False),), (("l1", False),)),
              Block("ne", (("ce", False),), (("r1", False),), (("t1", False),), (("cn", False),))]
    return blocks, nodes


def test_smoothing_keeps_the_sides_reduces_skew_and_assembles() -> None:
    blocks, nodes = _bump_square()
    start = _shaken(fill(blocks, nodes))
    out = smooth(start)
    for b in blocks:
        g = out[b.name]
        assert np.all(cell_areas(g) > 0)
        for s, idx in (("bottom", (slice(None), 0)), ("top", (slice(None), -1)),
                       ("left", (0, slice(None))), ("right", (-1, slice(None)))):
            np.testing.assert_allclose(g[idx], chain_nodes(b.side(s), nodes), rtol=0, atol=1e-15)
    worst = lambda grids: max(_skew(g).max() for g in grids.values())  # noqa: E731
    mean = lambda grids: np.mean(np.concatenate([_skew(g).ravel() for g in grids.values()]))  # noqa: E731
    assert worst(out) < worst(start) and mean(out) < 0.5 * mean(start)
    mesh, info = assemble(raw_blocks(out))
    assert info.interface_mismatch == 0.0 and info.blocks == 4
    assert info.interfaces == 8                       # each of the four cuts seen from both sides
    assert set(info.patch_faces) == {"farfield"}      # one counter-clockwise outer loop
    q = quality(mesh)
    assert not q.folded and q.min_angle_deg > 45.0 and q.max_growth < 1.3
    growth = interface_growth(mesh)
    # each cut's 12 faces once; the cut up from the bump is stretched, the others uniform
    assert growth.size == 4 * 12 and np.all(growth >= 1.0) and growth.max() < 1.5


def test_spacing_control_gives_the_first_cells_asked_for() -> None:
    blocks, nodes = _annulus([])
    grid = fill(blocks, nodes)["b0"]
    first = np.full(grid.shape[0], 2e-3)               # the sides' own first cell is 0.02
    out = winslow(_shaken({"b": grid})["b"], first, np.full(grid.shape[0], 0.1), tol=1e-12, iterations=500)
    height = np.hypot(*(out[:, 1] - out[:, 0]).T)
    assert np.all(cell_areas(out) < 0)                 # i round counter-clockwise, j out: left-handed
    # within 5 % (the rounds' tolerance) away from the sides, whose own first
    # cell is ten times larger
    np.testing.assert_allclose(height[6:-6], 2e-3, rtol=0.05)


def test_a_fold_is_refused() -> None:
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing="ij"), axis=-1)
    grid[2, 2] = [1.5, 1.5]
    with pytest.raises(ValueError, match="folds"):
        unfolded({"b": grid})
