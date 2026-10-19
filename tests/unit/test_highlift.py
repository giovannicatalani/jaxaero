"""Multi-element grids: the three families round the 30P30N's elements.

Judged by what the solver and the generator's targets ask of a grid: no fold; walls only
on the near-wall layers' ``j = 0``; one body per element; patches ``wall``
and ``farfield`` only; coincident interfaces; two coarsenings possible; the
first cell at the asked wall spacing and the near-wall lines orthogonal to
the wall, away from the fans at trailing edges and convex corners. Where the
targets are not met yet (smallest angle 20 deg, interface growth 1.3:
both missed at the main element's concave cove corner and round the gap; see
docs/dev/notes.md), the tests hold the measured state, so it cannot get worse
unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jaxaero import CaseError, load_case
from jaxaero.case import parse_case
from jaxaero.geometry.elements import corners, fan_points, gap, overlap, place, read_elements
from jaxaero.mesh.assemble import assemble
from jaxaero.mesh.coarsen import coarsen
from jaxaero.mesh.highlift import Settings, family, multi_element
from jaxaero.mesh.mesh import assign_conditions, patches
from jaxaero.mesh.quality import _cell_angles, interface_growth, quality

CRV = Path(__file__).resolve().parents[2] / "validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv"
PICK = {"main_flap": [1, 2], "slat_main": [0, 1], "slat_main_flap": [0, 1, 2]}
SETTINGS = Settings(wall_spacing=1.5e-5, cells_per_chord=150)


@pytest.fixture(scope="module")
def thirty_p():
    return read_elements(CRV, ["slat", "main", "flap"], 1 / 18)


@pytest.fixture(scope="module", params=sorted(PICK))
def built(request, thirty_p):
    elements = [thirty_p[k] for k in PICK[request.param]]
    raw, record = multi_element(elements, SETTINGS)
    mesh, info = assemble(raw)
    walled = assign_conditions(mesh, {p: "no_slip_wall" if p == "wall" else "far_field" for p in patches(mesh)})
    return request.param, elements, raw, record, walled, info


def _outside_fans(points, elements, margin: float = 0.0) -> np.ndarray:
    """Points farther than a layer's thickness (plus ``margin``) from every
    trailing edge and convex corner, where the layers fan."""
    fans = np.concatenate([fan_points(e) for e in elements])
    d = np.hypot(points[:, None, 0] - fans[None, :, 0], points[:, None, 1] - fans[None, :, 1])
    return d.min(axis=1) > SETTINGS.layer_thickness + margin


def test_each_family_has_its_bodies_two_patches_and_coincident_interfaces(built) -> None:
    name, elements, raw, record, mesh, info = built
    assert record["family"] == name
    assert info.bodies == len(elements)
    assert set(info.patch_faces) == {"wall", "farfield"}
    assert info.interface_mismatch < 1e-12
    assert not quality(mesh).folded
    assert info.far_field_radius > 90.0


def test_walls_lie_on_the_layers_j_min_only(built) -> None:
    *_, mesh, _ = built
    sides = {side for g in mesh for side, _ in g.sides.walls()}
    assert sides == {"j_min"}
    walled = [b for b, g in enumerate(mesh) if list(g.sides.walls())]
    assert all(b < len(built[1]) for b in walled)             # the layer blocks come first


def test_it_coarsens_twice(built) -> None:
    *_, mesh, _ = built
    twice = coarsen(coarsen(mesh))
    assert sum(g.ni * g.nj for g in twice) * 16 == sum(g.ni * g.nj for g in mesh)


def test_the_first_cell_and_its_orthogonality_outside_the_fans(built) -> None:
    _, elements, _, _, mesh, _ = built
    for g in mesh:
        x, y = np.asarray(g.x_node), np.asarray(g.y_node)
        for side, seg in g.sides.walls():
            sl = slice(seg.start, seg.stop + 1)
            wall, inner = np.stack([x[sl, 0], y[sl, 0]], 1), np.stack([x[sl, 1], y[sl, 1]], 1)
            keep = _outside_fans(wall, elements)
            keep[[0, -1]] = False
            # concave corners: their node steps along the bisector, a wall cell from both walls
            for e in elements:
                for c in corners(e):
                    keep &= np.hypot(*(wall - [e.airfoil.x[c], e.airfoil.y[c]]).T) > 1e-7
            h = np.hypot(*(inner - wall).T)[keep]
            tangent = np.gradient(wall, axis=0)[keep]
            cos = np.abs(np.sum(tangent * (inner - wall)[keep], 1)) / (np.hypot(*tangent.T) * h)
            assert np.all(np.abs(h / SETTINGS.wall_spacing - 1.0) <= 0.05)
            assert np.degrees(np.arcsin(cos.max())) <= 5.0


def test_angles_and_interface_growth_hold_their_measured_state(built) -> None:
    _, elements, _, _, mesh, _ = built
    worst = 90.0
    for g in mesh:
        x, y = np.asarray(g.x_node), np.asarray(g.y_node)
        centres = np.stack([0.25 * (x[1:, 1:] + x[:-1, 1:] + x[1:, :-1] + x[:-1, :-1]).ravel(),
                            0.25 * (y[1:, 1:] + y[:-1, 1:] + y[1:, :-1] + y[:-1, :-1]).ravel()], 1)
        far = _outside_fans(centres, elements, margin=0.02).reshape(g.ni, g.nj)
        if far.any():
            worst = min(worst, float(_cell_angles(x, y)[far].min()))
    assert worst >= 12.0                       # target 20 deg (missed at the cove and gap corners)
    growth = interface_growth(mesh)
    assert np.median(growth) <= 1.5 and growth.max() <= 35.0        # target 1.3 everywhere


def test_unknown_role_orders_are_refused() -> None:
    with pytest.raises(ValueError, match="no multi-element template"):
        family(["flap", "main"])
    assert family(["slat", "main", "flap"]) == "slat_main_flap"


def test_a_gap_too_narrow_for_the_layers_is_refused(thirty_p) -> None:
    slat, main, flap = thirty_p
    tight = place(flap, main, 0.0, 1e-4, 0.0)               # the flap all but touching the main element
    with pytest.raises(ValueError, match="too thin"):
        multi_element([main, tight], SETTINGS)


def test_a_section_in_another_unit_is_refused(thirty_p) -> None:
    inches = [e.moved(shift=(0.0, 0.0)) for e in read_elements(CRV, ["slat", "main", "flap"])]   # 18 in chord
    with pytest.raises(ValueError, match="coordinate_scale"):
        multi_element(inches[1:], SETTINGS)


def test_re_placed_flaps_build(thirty_p) -> None:
    slat, main, flap = thirty_p
    for setting in ((5.0, 0.02, 0.0), (-5.0, 0.01, 0.005)):
        raw, _ = multi_element([main, place(flap, main, *setting)], SETTINGS)
        mesh, info = assemble(raw)
        assert info.bodies == 2 and info.interface_mismatch < 1e-12


EXAMPLE = Path(__file__).resolve().parents[2] / "examples/main_flap_sa"


def test_the_example_case_reads_and_describes_its_grid() -> None:
    case = load_case(EXAMPLE)
    spec = case.grid_spec
    assert spec.grid == "multi_element" and [e["role"] for e in spec.elements] == ["main", "flap"]
    assert "main+flap" in spec.describe()
    elements = spec.placed_elements(EXAMPLE)
    assert [e.role for e in elements] == ["main", "flap"]
    assert elements[1].airfoil.chord == pytest.approx(0.30, abs=1e-3)


@pytest.mark.parametrize("elements, message", [
    ([{"role": "main", "file": "main.dat"}, {"role": "rudder", "file": "flap.dat"}], "needs a 'role'"),
    ([{"role": "main", "file": "main.dat"}, {"role": "flap"}], "needs a 'file'"),
    ([{"role": "flap", "file": "flap.dat"}, {"role": "main", "file": "main.dat"}], "no multi-element template"),
    ([{"role": "main", "file": "main.dat", "gap": 0.01}, {"role": "flap", "file": "flap.dat"}], "not re-placed"),
    ([{"role": "main", "file": "main.dat"}, {"role": "flap", "file": "flap.dat", "gap": "wide"}], "must be a number"),
])
def test_malformed_elements_are_refused(elements, message) -> None:
    case, numerical = (json.loads((EXAMPLE / f).read_text()) for f in ("case.json", "numerical.json"))
    case["domain"]["elements"] = elements
    with pytest.raises(CaseError, match=message):
        parse_case(case, numerical, EXAMPLE)


def test_a_re_placed_flap_is_where_the_case_puts_it() -> None:
    case, numerical = (json.loads((EXAMPLE / f).read_text()) for f in ("case.json", "numerical.json"))
    case["domain"]["elements"][1].update({"deflection_deg": 5.0, "gap": 0.02, "overlap": 0.0})
    main, flap = parse_case(case, numerical, EXAMPLE).grid_spec.placed_elements(EXAMPLE)
    assert gap(main, flap) == pytest.approx(0.02, abs=1e-10)
    assert overlap(main, flap, main) == pytest.approx(0.0, abs=1e-10)
