"""Airfoil geometry and the O- and C-grid generator.

The generator is judged by what a solver needs of a grid: no folded cell,
the asked-for first-cell height and far-field distance, walls orthogonal
(C-grid), and a wake cut the assembly recognises.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxaero.geometry.airfoil import airfoil_source, from_points, naca4, read_airfoil
from jaxaero.mesh.assemble import RawBlock, assemble
from jaxaero.mesh.block import Interface
from jaxaero.mesh.generate import c_grid, geometric, o_grid, vinokur
from jaxaero.mesh.mesh import assign_conditions, patches
from jaxaero.mesh.quality import quality


def test_vinokur_comes_close_to_both_end_spacings() -> None:
    for ds0, ds1 in ((1e-3, 2e-2), (5e-2, 5e-2), (2e-2, 1e-4)):
        s = vinokur(64, ds0, ds1)
        assert s[0] == 0.0 and s[-1] == 1.0 and np.all(np.diff(s) > 0.0)
        assert np.diff(s)[0] == pytest.approx(ds0, rel=0.1) and np.diff(s)[-1] == pytest.approx(ds1, rel=0.1)


def test_geometric_starts_at_its_first_step_and_ends_at_the_total() -> None:
    s = geometric(50, 1e-5, 100.0)
    assert s[1] == pytest.approx(1e-5) and s[-1] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="exceeds"):
        geometric(10, 20.0, 100.0)


def test_a_naca_section_runs_clockwise_from_its_trailing_edge() -> None:
    a = naca4("0012")
    assert a.sharp and a.chord == pytest.approx(1.0)
    assert a.x[0] == a.x[-1] == 1.0 and a.y[1] < 0.0          # lower surface first
    assert a.x[a.leading_edge] == pytest.approx(0.0, abs=1e-12)
    assert not naca4("2412", sharp=False).sharp


@pytest.mark.parametrize("layout", ["selig", "lednicer", "counter-clockwise"])
def test_coordinate_files_read_to_the_same_contour(tmp_path, layout) -> None:
    ref = naca4("2412", n=41)
    le = ref.leading_edge
    upper = np.stack([ref.x[le:], ref.y[le:]], 1)                # LE -> TE
    lower = np.stack([ref.x[:le + 1][::-1], ref.y[:le + 1][::-1]], 1)
    if layout == "selig":                                      # TE -> upper -> LE -> lower -> TE
        rows = np.concatenate([upper[::-1], lower[1:]])
        text = "NACA 2412\n" + "\n".join(f"{x:.12f} {y:.12f}" for x, y in rows)
    elif layout == "lednicer":
        text = (f"NACA 2412\n{len(upper)}. {len(lower)}.\n\n" + "\n".join(f"{x:.12f} {y:.12f}" for x, y in upper)
                + "\n\n" + "\n".join(f"{x:.12f} {y:.12f}" for x, y in lower))
    else:
        rows = np.concatenate([lower[::-1][:-1], upper])[::-1]    # any start, either direction
        text = "\n".join(f"{x:.12f} {y:.12f}" for x, y in rows)
    path = tmp_path / "foil.dat"
    path.write_text(text)
    got = read_airfoil(path)
    assert got.x.size == ref.x.size
    assert np.allclose(got.x, ref.x, atol=1e-11) and np.allclose(got.y, ref.y, atol=1e-11)
    assert airfoil_source(str(path)).x.size == ref.x.size


def judge(x, y):
    mesh, info = assemble([RawBlock(x, y)])
    kinds = {p: "no_slip_wall" if p == "wall" else "far_field" for p in patches(mesh)}
    return mesh[0], info, quality(assign_conditions(mesh, kinds))


def test_a_c_grid_is_orthogonal_at_the_wall_and_reaches_its_far_field() -> None:
    x, y = c_grid(naca4("0012"), 128, 32, 64, 1e-6, far_field=50.0)
    grid, info, q = judge(x, y)
    assert not q.folded and q.min_angle_deg > 60.0
    assert q.wall_spacing[0] == pytest.approx(1e-6, rel=0.05) and q.wall_spacing[1] == pytest.approx(1e-6, rel=0.1)
    assert q.wall_angle_deg < 5.0 and q.far_field == pytest.approx(50.0, rel=0.02)
    cut = [s for s in grid.sides.j_min if isinstance(s, Interface)]
    assert len(cut) == 2 and cut[0].stop == 32 and cut[0].reversed           # the wake, found
    assert info.patch_faces["wall"] == 128 and info.bodies == 1


def test_an_o_grid_round_a_sharp_trailing_edge_does_not_fold() -> None:
    """The lines leaving a sharp trailing edge turn through ~360 deg; with
    the smoothing grown from zero at the wall they crossed (1622 folded cells
    at these settings)."""
    x, y = o_grid(naca4("0012"), 128, 96, 1e-3)
    _, info, q = judge(x, y)
    assert not q.folded and q.far_field == pytest.approx(100.0, rel=0.02)
    assert info.patch_faces == {"wall": 128, "farfield": 128}


def test_a_blunt_trailing_edge_takes_an_o_grid_and_refuses_a_c_grid() -> None:
    blunt = naca4("0012", sharp=False)
    x, y = o_grid(blunt, 128, 64, 1e-4)
    grid, info, q = judge(x, y)
    assert not q.folded and info.patch_faces["wall"] == x.shape[0] - 1     # the base is wall too
    with pytest.raises(ValueError, match="blunt trailing edge"):
        c_grid(blunt, 128, 32, 64, 1e-4)


def test_any_airfoil_from_its_points() -> None:
    """RAE 2822's measured ordinates (validation/references/rae2822/geom.txt, first rows)."""
    xs = np.array([0.0, 0.0006, 0.00241, 0.00541, 0.00961, 0.01498, 0.02923, 0.04801, 0.07114, 0.09840,
                   0.12952, 0.16422, 0.20215, 0.24295, 0.28622, 0.33148, 0.37814, 0.42566, 0.47358, 0.52142,
                   0.56872, 0.61509, 0.66014, 0.70352, 0.74487, 0.78380, 0.82003, 0.85322, 0.88309, 0.90941,
                   0.93197, 0.95060, 0.96509, 0.97546, 0.98319, 0.98923, 1.0])
    thickness = 0.12 / 0.2 * (0.2969 * np.sqrt(xs) - 0.126 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1036 * xs**4)
    a = from_points(np.r_[xs[::-1], xs[1:]], np.r_[-0.8 * thickness[::-1], thickness[1:]], "lop-sided")
    x, y = c_grid(a, 96, 24, 48, 1e-5)
    _, _, q = judge(x, y)
    assert not q.folded and q.min_angle_deg > 45.0


def test_closing_a_trailing_edge_moves_each_surface_by_at_most_half_the_gap() -> None:
    blunt = naca4("2412", sharp=False)
    closed = blunt.closed()
    assert closed.sharp and not blunt.sharp
    shift = np.hypot(closed.x - blunt.x, closed.y - blunt.y)
    assert shift.max() <= 0.5 * blunt.trailing_edge_gap * (1 + 1e-12)
    k = blunt.leading_edge
    assert shift[k] == 0.0 and closed.leading_edge == k and closed.chord == pytest.approx(blunt.chord)
    x, y = c_grid(closed, 96, 24, 48, 1e-5)
    assert not judge(x, y)[2].folded


def test_an_airfoil_in_millimetres_is_scaled_to_unit_chord() -> None:
    a = naca4("2412")
    mm = from_points(150.0 * a.x + 30.0, 150.0 * a.y - 5.0, "in mm")
    unit = mm.unit_chord()
    k = unit.leading_edge
    assert unit.chord == pytest.approx(1.0) and (unit.x[k], unit.y[k]) == (0.0, 0.0)
    np.testing.assert_allclose(np.c_[unit.x, unit.y], np.c_[a.unit_chord().x, a.unit_chord().y], atol=1e-12)


def test_a_c_grid_case_closes_an_open_trailing_edge_on_request(tmp_path) -> None:
    from jaxaero.case.domain import GridSpec
    spec = dict(grid="c_grid", airfoil="naca2412_blunt", cells_airfoil=64, cells_wake=16, cells_normal=24,
                wall_spacing=1e-4)
    with pytest.raises(ValueError, match="trailing_edge"):
        GridSpec(**spec).build(tmp_path)
    closing = GridSpec(**spec, trailing_edge="close")
    closing.build(tmp_path)
    assert "gap 0.00252 closed" in closing.describe(tmp_path)
