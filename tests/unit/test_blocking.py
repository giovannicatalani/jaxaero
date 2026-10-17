"""The block-graph core: counts, transfinite interpolation, elliptic smoothing.

Synthetic graphs whose answers are known: a bilinear map, a quarter annulus
cut into one or three blocks (the global smoothing makes them the same
grid), a 2 x 2 square under a bump; the output must assemble with
coincident interfaces and pass the quality checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxaero.mesh.assemble import assemble
from jaxaero.mesh.blocking import (Block, cell_areas, chain_nodes, distribute, fill, raw_blocks, smooth,
                                   solve_counts, tfi)
from jaxaero.mesh.generate import geometric
from jaxaero.mesh.quality import _cell_angles, quality


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


def _annulus(cuts):
    """A quarter annulus r in [1, 2], cut along ``cuts`` (degrees) into blocks;
    36 cells round, 16 out, geometric from 0.02."""
    radii = 1.0 + geometric(16, 0.02, 1.0)
    angles = [0.0] + list(cuts) + [90.0]
    nodes, blocks = {}, []
    for k, a in enumerate(angles):
        nodes[f"ray{k}"] = _ray(a, radii)
    for k in range(len(angles) - 1):
        n = round(36 * (angles[k + 1] - angles[k]) / 90.0)
        nodes[f"in{k}"], nodes[f"out{k}"] = _arc(1.0, angles[k], angles[k + 1], n), _arc(2.0, angles[k], angles[k + 1], n)
        blocks.append(Block(f"b{k}", ((f"in{k}", False),), ((f"ray{k + 1}", False),), ((f"out{k}", False),),
                            ((f"ray{k}", False),)))
    fixed = {e for e in nodes if not e.startswith("ray")} | {"ray0", f"ray{len(angles) - 1}"}
    return blocks, nodes, fixed


def _skew(grid):
    return 90.0 - _cell_angles(grid[..., 0], grid[..., 1])


def test_a_quarter_annulus_is_the_same_grid_in_one_block_or_three() -> None:
    one = _annulus([])
    three = _annulus([30.0, 60.0])
    rng = np.random.default_rng(0)

    def shaken(grids):                     # interiors moved, so the smoother has work to do
        out = {}
        for name, g in grids.items():
            g = g.copy()
            g[1:-1, 1:-1] += rng.uniform(-0.01, 0.01, g[1:-1, 1:-1].shape)
            out[name] = g
        return out

    g1 = smooth(one[0], one[1], shaken(fill(one[0], one[1])), one[2], tol=1e-13, iterations=500)["b0"]
    g3 = smooth(three[0], three[1], shaken(fill(three[0], three[1])), three[2], tol=1e-13, iterations=500)
    joined = np.concatenate([g3["b0"], g3["b1"][1:], g3["b2"][1:]], axis=0)
    np.testing.assert_allclose(joined, g1, atol=1e-9)


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
    fixed = {"f0", "f1", "r0", "r1", "t0", "t1", "l0", "l1"}
    return blocks, nodes, fixed


def test_smoothing_keeps_fixed_nodes_reduces_skew_and_assembles() -> None:
    blocks, nodes, fixed = _bump_square()
    # bend the inner cut lines, as a poor starting point
    for e in ("cs", "cw", "cn", "ce"):
        t = np.linspace(0, 1, len(nodes[e]))
        nodes[e] = nodes[e] + 0.08 * np.sin(np.pi * t)[:, None] * np.array([1.0, -1.0])
    start = fill(blocks, nodes)
    out = smooth(blocks, nodes, start, fixed)
    for b in blocks:
        g = out[b.name]
        assert np.all(cell_areas(g) > 0)
        for s, idx in (("bottom", (slice(None), 0)), ("top", (slice(None), -1)),
                       ("left", (0, slice(None))), ("right", (-1, slice(None)))):
            if all(e in fixed for e, _ in b.side(s)):
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


def test_a_free_edge_must_join_two_blocks() -> None:
    blocks, nodes, fixed = _bump_square()
    with pytest.raises(ValueError, match="free edge"):
        smooth(blocks, nodes, fill(blocks, nodes), fixed - {"f0"})
