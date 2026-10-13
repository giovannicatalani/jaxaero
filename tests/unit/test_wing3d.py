"""A 3D wing grid from a file: assembly, sections and forces (mesh/assemble3d.py,
validation/sections.py, post/forces.py)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, NumericsConfig
from jaxaero.mesh.assemble import PatchRule
from jaxaero.mesh.assemble3d import _rectangles, assemble_3d, load_grid_3d
from jaxaero.mesh.atlas import build_atlas
from jaxaero.mesh.block3d import BoundaryRect, InterfaceRect
from jaxaero.mesh.mesh import assign_conditions
from jaxaero.numerics.residual import make_residual
from jaxaero.post.forces import ForceReference, integrate_forces, surface
from validation.sections import section

from .test_grid3d import M6_SAMPLE, warped_box

WING_RULES = (PatchRule("symmetry", y_max=1e-9),
              PatchRule("wall", x_min=-1.0, x_max=3.0, y_max=3.0, z_min=-1.0, z_max=1.0),
              PatchRule("farfield"))
needs_sample = pytest.mark.skipif(not M6_SAMPLE.exists(), reason="ONERA M6 sample grid not generated")


@needs_sample
def test_the_wing_grid_assembles_from_its_geometry() -> None:
    (grid,), info = load_grid_3d(M6_SAMPLE, rules=WING_RULES)
    sides = grid.sides
    (pole,) = sides.i_min
    assert isinstance(pole, InterfaceRect) and pole.side == "i_min" and pole.period == (0, 48)
    assert pole.origin == (0, 24)
    assert [type(r) for r in sides.k_min + sides.k_max] == [InterfaceRect, InterfaceRect]
    assert sides.k_min[0].side == "k_max" and sides.k_max[0].side == "k_min"
    assert [(r.patch, r.start, r.stop) for r in sides.j_min] == [("wall", (0, 0), (40, 48))]
    assert [r.patch for r in sides.j_max] == ["farfield"] and [r.patch for r in sides.i_max] == ["symmetry"]
    assert info.cells == 40 * 64 * 48 and info.patch_faces["wall"] == 40 * 48


def test_a_face_matching_no_rule_is_an_error() -> None:
    nodes = warped_box((2, 2, 2), amplitude=0.0)
    with pytest.raises(ValueError, match="no patch rule"):
        assemble_3d(nodes, (PatchRule("wall", x_max=0.5),))


def test_mixed_labels_become_rectangles_tiling_the_side() -> None:
    names = np.array([["a", "a", "b"], ["a", "a", "b"], ["c", "c", "c"]], dtype=object)
    rects = _rectangles(names)
    painted = np.full(names.shape, "", dtype=object)
    for r in rects:
        assert isinstance(r, BoundaryRect)
        block = painted[r.start[0]:r.stop[0], r.start[1]:r.stop[1]]
        assert np.all(block == "")
        painted[r.start[0]:r.stop[0], r.start[1]:r.stop[1]] = r.patch
    assert np.array_equal(painted, names) and len(rects) == 3


@needs_sample
def test_a_section_cut_reproduces_a_field_linear_in_y() -> None:
    (grid,), _ = load_grid_3d(M6_SAMPLE, rules=WING_RULES)
    mesh = assign_conditions(grid, {"wall": "slip_wall", "farfield": "far_field", "symmetry": "symmetry"})
    atlas = build_atlas(mesh, 2)
    mid, cell = atlas.walls.faces.mid, atlas.walls.faces.cell
    fake = {"i": cell[:, 0], "k": cell[:, 1], "x": mid[:, 0], "y": mid[:, 1], "z": mid[:, 2],
            "cp": 0.3 + 2.0 * mid[:, 1]}
    y = 0.44 * 1.4760179762
    cut = section(fake, y)
    assert len(cut["cp"]) == 48                                   # one point per line round the section
    np.testing.assert_allclose(cut["y"], y, rtol=1e-14)
    np.testing.assert_allclose(cut["cp"], 0.3 + 2.0 * y, rtol=1e-13)
    assert cut["x_over_c"].min() == 0.0 and cut["x_over_c"].max() == 1.0


@needs_sample
def test_3d_forces_of_a_uniform_overpressure() -> None:
    """Gauge pressure ``dp`` everywhere on the wall: ``F = -dp sum S``, which for
    the half wing (open at the root) is ``dp`` times the root section's area
    vector; checked against that sum computed independently, with the moment."""
    (grid,), _ = load_grid_3d(M6_SAMPLE, rules=WING_RULES)
    flow = FlowConditions(mach=0.5, alpha_deg=4.0)
    mesh = assign_conditions(grid, {"wall": "slip_wall", "farfield": "far_field", "symmetry": "symmetry"})
    atlas = build_atlas(mesh, 2)
    op = make_residual(atlas, flow, NumericsConfig())
    U = op.initial_state()
    dp = 0.05
    U = U.at[4].add(dp / (flow.gamma - 1.0))               # energy: pressure + dp everywhere
    ref = ForceReference(length=0.8, area=1.15, centre=(0.25, 0.0, 0.0))
    f = integrate_forces(U, atlas, flow, ref)
    S = np.asarray(atlas.walls.into_fluid * atlas.wall_values(atlas.grid.faces))
    force = -dp * S.sum(axis=1)
    q = 0.5 * flow.mach**2 * ref.area
    a = np.deg2rad(4.0)
    assert f.cl == pytest.approx((-force[0] * np.sin(a) + force[2] * np.cos(a)) / q, rel=1e-12, abs=1e-15)
    assert f.cd == pytest.approx((force[0] * np.cos(a) + force[2] * np.sin(a)) / q, rel=1e-12, abs=1e-15)
    r = atlas.walls.faces.mid - np.array([0.25, 0.0, 0.0])
    moment = np.sum(r[:, 2] * (-dp * S[0]) - r[:, 0] * (-dp * S[2]))
    assert f.cm == pytest.approx(moment / (q * ref.length), rel=1e-12, abs=1e-15)
    cp = surface(U, atlas, flow)["cp"]
    np.testing.assert_allclose(cp, dp / (0.5 * flow.mach**2), rtol=1e-12)


def test_a_3d_case_needs_a_reference_area() -> None:
    from jaxaero.post.forces import _integrate_forces_3d

    with pytest.raises(ValueError, match="reference_area"):
        _integrate_forces_3d(jnp.zeros(1), None, FlowConditions(mach=0.5), ForceReference())


@needs_sample
def test_check_reports_a_3d_grid(tmp_path, capsys) -> None:
    """``jaxaero CASE --check`` on a 3D wing: patches by faces, the wall's
    extent, no 2D picture."""
    import json

    from jaxaero.__main__ import main

    case = {"general": {"case_name": "m6_check"},
            "domain": {"grid": "file", "file": str(M6_SAMPLE), "reference_area": 1.15,
                       "patches": [{"name": "symmetry", "y_max": 1e-9},
                                   {"name": "wall", "x_min": -1.0, "x_max": 3.0, "y_max": 3.0,
                                    "z_min": -1.0, "z_max": 1.0},
                                   {"name": "farfield"}]},
            "freestream": {"mach": 0.84, "alpha_deg": 3.06}}
    (tmp_path / "case.json").write_text(json.dumps(case))
    (tmp_path / "numerical.json").write_text("{}")
    assert main([str(tmp_path), "--check", "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "patch 'wall': slip_wall, 1920 faces" in out and "patch 'symmetry': symmetry, 3072 faces" in out
    assert "wall: 1920 faces" in out and "picture is 2D only" in out


def test_2d_figures_refuse_a_3d_result(tmp_path) -> None:
    np.savez(tmp_path / "w_fields.npz", n_blocks=1, U_0=np.zeros((5, 1, 1, 1)), x_node_0=np.zeros((2, 2, 2)),
             y_node_0=np.zeros((2, 2, 2)), z_node_0=np.zeros((2, 2, 2)))
    (tmp_path / "w_metadata.json").write_text("{}")
    from jaxaero.plot import load_result

    with pytest.raises(SystemExit, match="3D result"):
        load_result(tmp_path)
