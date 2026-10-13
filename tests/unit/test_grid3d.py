"""3D block metrics, sides and wall distance (mesh/block.py, block3d.py, wall3d.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxaero.mesh.block import Grid, metrics
from jaxaero.mesh.block3d import InterfaceRect, box_sides
from jaxaero.mesh.wall3d import patch_distance, wall_distance_3d

from ..single import pole_sides, quad_distance

M6_SAMPLE = Path(__file__).resolve().parents[2] / "data/grids/onera_m6/om6_sample_40x64x48.ufmt"


def warped_box(shape=(5, 4, 6), amplitude=0.2, seed=0) -> np.ndarray:
    """Nodes of a stretched box with every node moved at random (warped faces)."""
    rng = np.random.default_rng(seed)
    axes = [np.cumsum(np.r_[0.0, rng.uniform(0.5, 2.0, n)]) for n in shape]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"))
    return nodes + amplitude * rng.uniform(-1.0, 1.0, nodes.shape)


def closure(faces) -> np.ndarray:
    """Sum of the outward area vectors of each cell, ``(3, *shape)``."""
    return sum(np.take(f, range(1, f.shape[1 + a]), axis=1 + a) - np.take(f, range(f.shape[1 + a] - 1), axis=1 + a)
               for a, f in enumerate(faces))


def test_face_vectors_close_every_warped_cell() -> None:
    _, volume, faces = metrics(warped_box())
    scale = max(np.abs(f).max() for f in faces)
    assert np.abs(closure(faces)).max() < 1e-14 * scale
    assert volume.min() > 0.0


def trilinear_volume(corners: np.ndarray) -> float:
    """Volume of the trilinear map of ``corners (2, 2, 2, 3)`` by 3-point Gauss
    quadrature of its Jacobian determinant (a polynomial of degree <= 2 in each
    variable, so the rule is exact)."""
    x, w = np.polynomial.legendre.leggauss(3)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    total = 0.0
    for a, wa in zip(x, w):
        for b, wb in zip(x, w):
            for c, wc in zip(x, w):
                t = np.array([[1 - a, a], [1 - b, b], [1 - c, c]])
                dt = np.array([[-1.0, 1.0]] * 3)
                J = np.stack([np.einsum("i,j,k,ijkd->d", *(dt[0], t[1], t[2]), corners),
                              np.einsum("i,j,k,ijkd->d", *(t[0], dt[1], t[2]), corners),
                              np.einsum("i,j,k,ijkd->d", *(t[0], t[1], dt[2]), corners)])
                total += wa * wb * wc * np.linalg.det(J)
    return total


def test_volume_is_that_of_the_trilinear_hexahedron() -> None:
    nodes = warped_box((3, 3, 3), amplitude=0.3, seed=4)
    _, volume, _ = metrics(nodes)
    for i, j, k in [(0, 0, 0), (1, 2, 0), (2, 1, 2)]:
        corners = np.moveaxis(nodes[:, i:i + 2, j:j + 2, k:k + 2], 0, -1)
        assert volume[i, j, k] == pytest.approx(trilinear_volume(corners), rel=1e-13)


def test_volumes_sum_to_the_boundary_identity() -> None:
    """``sum V = (1/3) sum_boundary S . x_f``: interior faces cancel, which
    checks the indexing and orientation of every family."""
    nodes = warped_box()
    _, volume, faces = metrics(nodes)
    total = 0.0
    for a, S in enumerate(faces):
        n = np.moveaxis(nodes, 1 + a, 1)          # the other two axes keep their order, as in S
        for plane, sign in ((n[:, 0], -1.0), (n[:, -1], 1.0)):
            m = 0.25 * (plane[:, :-1, :-1] + plane[:, 1:, :-1] + plane[:, :-1, 1:] + plane[:, 1:, 1:])
            s = np.take(S, 0 if sign < 0 else S.shape[1 + a] - 1, axis=1 + a)
            total += sign * np.sum(s * m)
    assert volume.sum() == pytest.approx(total / 3.0, rel=1e-13)


def test_a_collapsed_face_has_zero_area_exactly() -> None:
    nodes = warped_box((3, 2, 4))
    nodes[:, 0] = nodes[:, 0, :, :1]           # i = 0: every k node at the k = 0 one (a pole)
    _, volume, faces = metrics(nodes)
    assert np.array_equal(faces[0][:, 0], np.zeros_like(faces[0][:, 0]))
    assert np.abs(closure(faces)).max() < 1e-13 * np.abs(faces[1]).max()


def test_from_nodes_matches_metrics_and_checks_the_shape() -> None:
    nodes = warped_box()
    grid = Grid.from_node_array(nodes, sides=box_sides((5, 4, 6), **dict.fromkeys(
        ("j_min", "j_max", "i_min", "i_max", "k_min", "k_max"), "farfield")))
    assert grid.dim == 3 and grid.shape == (5, 4, 6) and len(grid.faces) == 3
    with pytest.raises(ValueError, match="node arrays"):
        Grid.from_node_array(nodes[:, :-1], sides=grid.sides)


def test_sides_must_tile_each_side_once() -> None:
    with pytest.raises(ValueError, match="tile"):
        box_sides((2, 2, 2), j_min="wall", j_max="farfield", i_min="farfield", i_max="farfield",
                  k_min="symmetry", k_max=())


def test_the_pole_faces_the_opposite_cell() -> None:
    sides = pole_sides((4, 3, 8))
    (pole,) = sides.i_min
    assert isinstance(pole, InterfaceRect) and pole.side == "i_min"
    assert pole.partner((1, 0)) == (1, 4)
    assert pole.partner((2, 6)) == (2, 2)
    assert pole.partner((0, 7)) == (0, 3)


@pytest.mark.skipif(not M6_SAMPLE.exists(), reason="ONERA M6 sample grid not generated (data/grids/README.md)")
def test_the_onera_m6_sample_grid_has_sound_metrics() -> None:
    from jaxaero.mesh.plot3d import read_plot3d_3d

    (nodes,) = read_plot3d_3d(M6_SAMPLE)
    centres, volume, faces = metrics(nodes)
    assert volume.shape == (40, 64, 48)
    assert volume.min() > 0.0
    assert np.array_equal(faces[0][:, 0], np.zeros_like(faces[0][:, 0])), "the pole faces have area"
    scale = np.abs(faces[1][:, :, 0]).max()
    assert np.abs(closure(faces)).max() < 1e-12 * max(scale, 1.0) * 1e4


def test_quad_distance_matches_brute_force() -> None:
    rng = np.random.default_rng(2)
    quad = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 1.0, 0.0], [0.0, 1.0, -0.1]])
    points = rng.uniform(-1.0, 2.0, (200, 3))
    got = quad_distance(points, quad[None].repeat(200, axis=0))
    s, t = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401), indexing="ij")
    # Sample the two triangles of the a-c diagonal finely.
    a, b, c, d = quad
    tri1 = a + s[..., None] * (b - a) + t[..., None] * (c - b)
    tri2 = a + s[..., None] * (d - a) + t[..., None] * (c - d)
    mask = (t <= s)[..., None]
    samples = np.concatenate([tri1[np.broadcast_to(mask, tri1.shape)].reshape(-1, 3),
                              tri2[np.broadcast_to(mask, tri2.shape)].reshape(-1, 3)])
    brute = np.array([np.min(np.linalg.norm(samples - p, axis=1)) for p in points[:40]])
    assert np.all(got[:40] <= brute + 1e-12)
    assert np.allclose(got[:40], brute, atol=5e-3)


def sampled_patch_distance(p: np.ndarray, quad: np.ndarray, n: int = 801, window=((0, 1), (0, 1))) -> float:
    s, t = np.meshgrid(np.linspace(*window[0], n), np.linspace(*window[1], n), indexing="ij")
    a, b, c, d = quad
    x = ((1 - s) * (1 - t))[..., None] * a + (s * (1 - t))[..., None] * b + (s * t)[..., None] * c \
        + ((1 - s) * t)[..., None] * d
    return float(np.linalg.norm(x - p, axis=-1).min())


def test_patch_distance_on_strongly_warped_faces() -> None:
    """Warp (out of plane) far larger than the distances measured: where the
    two-triangle approximation is off by up to the warp (the M6 tip)."""
    rng = np.random.default_rng(5)
    quad = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.0, 1.0, 0.0], [0.0, 1.0, 0.2]])
    on = np.array([0.3, 0.6])
    surface = (1 - on[0]) * (1 - on[1]) * quad[0] + on[0] * (1 - on[1]) * quad[1] \
        + on[0] * on[1] * quad[2] + (1 - on[0]) * on[1] * quad[3]
    points = np.vstack([surface + [0.0, 0.0, 1e-3], rng.uniform(-0.5, 1.5, (30, 3))])
    got = patch_distance(points, np.broadcast_to(quad, (len(points), 4, 3)))
    want = np.array([sampled_patch_distance(x, quad) for x in points])
    want[0] = sampled_patch_distance(points[0], quad, 801, ((0.29, 0.31), (0.59, 0.61)))   # resolve the 1e-3 offset
    assert np.all(got <= want + 1e-12), "a patch distance above a sampled one"
    np.testing.assert_allclose(got, want, atol=2e-6, rtol=1e-4)
    tri = quad_distance(points[:1], quad[None])[0]
    assert got[0] == pytest.approx(1e-3 / np.sqrt(1 + 0.2**2), rel=0.05) and tri > 10 * got[0]


def test_wall_distance_is_the_brute_force_minimum() -> None:
    nodes = warped_box((6, 5, 4), amplitude=0.1, seed=7)
    nodes[2, :, 0, :] += 0.15 * np.sin(3.0 * nodes[0, :, 0, :]) * np.cos(4.0 * nodes[1, :, 0, :])   # warp the wall
    sides = box_sides((6, 5, 4), j_min="wall", j_max="farfield", i_min="farfield", i_max="farfield",
                      k_min="farfield", k_max="farfield").with_conditions(
        {"wall": "no_slip_wall", "farfield": "far_field"})
    grid = Grid.from_node_array(nodes, sides=sides)
    (d,) = wall_distance_3d((grid,), chunk=7)            # chunks not dividing the cell count
    from jaxaero.mesh.wall3d import wall_quads

    quads = wall_quads((grid,)).corners
    p = np.moveaxis(np.asarray(grid.centres), 0, -1).reshape(-1, 3)
    brute = np.array([patch_distance(x[None], quads).min() for x in p]).reshape(grid.shape)
    np.testing.assert_allclose(d, brute, rtol=1e-12)
    sampled = min(sampled_patch_distance(p[0], q, 201) for q in quads)
    assert d.ravel()[0] <= sampled + 1e-12 and d.ravel()[0] == pytest.approx(sampled, abs=5e-4)


def test_the_metric_check_passes_valid_3d_blocks() -> None:
    """``validate_metrics`` (what ``validate.py --grids-only`` reports) in 3D:
    positive volumes, closed cells, the boundary identity."""
    from jaxaero.mesh.block import validate_metrics

    nodes = warped_box((5, 4, 6), amplitude=0.3, seed=3)
    grid = Grid.from_node_array(nodes, box_sides((5, 4, 6), **dict.fromkeys(
        ("j_min", "j_max", "i_min", "i_max", "k_min", "k_max"), "farfield")))
    report = validate_metrics(grid)
    assert report.ok, report.summary()
    assert report.area_closure_error < 1e-13
    mirrored = nodes[:, :, :, ::-1]                         # left-handed: every volume negative
    assert not validate_metrics(Grid.from_node_array(mirrored, grid.sides)).ok


@pytest.mark.skipif(not M6_SAMPLE.exists(), reason="ONERA M6 sample grid not generated (data/grids/README.md)")
def test_the_onera_m6_sample_passes_the_metric_check() -> None:
    from jaxaero.mesh.block import validate_metrics
    from jaxaero.mesh.plot3d import read_plot3d_3d

    (nodes,) = read_plot3d_3d(M6_SAMPLE)
    shape = tuple(n - 1 for n in nodes.shape[1:])
    assert validate_metrics(Grid.from_node_array(nodes, pole_sides(shape))).ok
