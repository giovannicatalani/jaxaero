"""High-lift elements: read, placed, landmarked.

Judged on the 30P30N geometry (the BANC ``.crv``, eleven curves making three
contours) and on synthetic NACA-based elements whose cove lips are known by
construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxaero.geometry.airfoil import from_points, naca4
from jaxaero.geometry.elements import (Element, cove_lip, deflection, gap, join_contours, overlap, place,
                                       read_curves, read_element, read_elements)

CRV = Path(__file__).resolve().parents[2] / "validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv"
INCH = 1.0 / 18.0


@pytest.fixture(scope="module")
def thirty_p():
    return read_elements(CRV, ["slat", "main", "flap"], INCH)


def _signed_area(a) -> float:
    x, y = a.x, a.y
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def test_the_30p30n_reads_to_three_clockwise_contours(thirty_p) -> None:
    assert len(read_curves(CRV)) == 11
    slat, main, flap = thirty_p
    assert [e.role for e in thirty_p] == ["slat", "main", "flap"]
    for e in thirty_p:
        assert _signed_area(e.airfoil) < 0.0                         # clockwise
        assert e.airfoil.y[1] < e.airfoil.y[-2]                       # lower surface first
    # Element chords in stowed chords: the slat and the flap are 0.15 and 0.30
    # (the published values); the main element alone is 0.83 (the stowed chord
    # also spans the stowed slat and flap).
    assert slat.airfoil.chord == pytest.approx(0.15, abs=0.005)
    assert main.airfoil.chord == pytest.approx(0.8315, abs=0.001)
    assert flap.airfoil.chord == pytest.approx(0.30, abs=0.001)
    # The thickened slat trailing edge, 0.03 in (0.76 mm); the others blunt too.
    assert slat.airfoil.trailing_edge_gap == pytest.approx(0.03 * INCH, rel=0.01)
    assert not main.airfoil.sharp and not flap.airfoil.sharp


def test_the_30p30n_settings_measure_as_published(thirty_p) -> None:
    slat, main, flap = thirty_p
    assert gap(main, flap) == pytest.approx(0.0127, abs=2e-4)
    assert gap(slat, main) == pytest.approx(0.0295, abs=1e-3)


def test_unclosed_curves_are_refused() -> None:
    a, b = np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="does not close"):
        join_contours([a, b])
    c = np.array([[0.0, 0.0], [1.0, 1.0]])                           # closes, walked either way
    assert len(join_contours([a, b, c])) == 1


def test_a_sharp_contour_split_into_curves_opens_at_its_trailing_edge(tmp_path) -> None:
    ref = naca4("2412", n=81)
    le = ref.leading_edge
    pts = np.stack([ref.x, ref.y], 1)
    lower, upper = pts[:le + 1], pts[le:][::-1]                       # TE->LE, and TE->LE the other way
    text = "".join(f"{len(c)}\n" + "".join(f"{x:.15e} {y:.15e} 0.0\n" for x, y in c) for c in (lower, upper))
    path = tmp_path / "foil.crv"
    path.write_text(text)
    (e,) = read_elements(path, ["main"])
    assert e.airfoil.sharp and e.airfoil.chord == pytest.approx(1.0)
    np.testing.assert_allclose(e.trailing_edge, [1.0, ref.y[0]], atol=1e-12)


@pytest.mark.parametrize("element", ["flap", "slat"])
@pytest.mark.parametrize("setting", [(0.0, 0.0127, 0.0025), (5.0, 0.02, -0.01), (-4.0, 0.035, 0.0)])
def test_placement_round_trips(thirty_p, element, setting) -> None:
    slat, main, flap = thirty_p
    moving = flap if element == "flap" else slat
    angle, g, o = setting
    placed = place(moving, main, angle, g, o)
    up, down = (main, placed) if element == "flap" else (placed, main)
    assert gap(up, down) == pytest.approx(g, abs=1e-10)
    assert overlap(up, down, main) == pytest.approx(o, abs=1e-10)
    (ax, ay), (bx, by) = moving.chord_direction, placed.chord_direction
    turn = np.degrees(np.arctan2(ax * by - ay * bx, ax * bx + ay * by))
    assert turn == pytest.approx(-angle if element == "flap" else angle, abs=1e-10)
    assert placed.airfoil.chord == pytest.approx(moving.airfoil.chord, rel=1e-12)


def test_only_slats_and_flaps_deflect(thirty_p) -> None:
    with pytest.raises(ValueError, match="only a slat or a flap"):
        deflection(thirty_p[1], 5.0)


def test_the_30p30n_cove_lips_are_its_corners(thirty_p) -> None:
    slat, main, flap = thirty_p
    curves = [INCH * c for c in read_curves(CRV)]
    cusp, step = curves[9][-1], curves[1][-1]                        # where the file's curves meet
    for e, corner in ((slat, cusp), (main, step)):
        k = cove_lip(e)
        np.testing.assert_allclose([e.airfoil.x[k], e.airfoil.y[k]], corner, atol=1e-12)
    assert cove_lip(flap) is None


def _cove_main(n: int = 101):
    """NACA 2412 with a flap cove: the lower surface aft of 80% chord replaced
    by a step up (the lip at its foot) and a ceiling to the trailing edge."""
    a = naca4("2412", n=n)
    le = a.leading_edge
    xl, yl = a.x[:le + 1][::-1], a.y[:le + 1][::-1]                   # LE -> TE, lower
    xu, yu = a.x[le:], a.y[le:]
    lip = np.array([0.8, np.interp(0.8, xl, yl)])
    roof = np.interp(0.8, xl, yl) + 0.6 * (np.interp(0.8, xu, yu) - np.interp(0.8, xl, yl))
    ceiling_x = np.linspace(1.0, 0.8, 12)
    ceiling_y = a.y[0] + (roof - a.y[0]) * np.sin(0.5 * np.pi * (1.0 - ceiling_x) / 0.2)
    wall_y = np.linspace(roof, lip[1], 8)[1:-1]
    front = xl <= 0.8 - 1e-9
    x = np.concatenate([ceiling_x, np.full(wall_y.size, 0.8), [lip[0]], xl[front][::-1], xu[1:]])
    y = np.concatenate([ceiling_y, wall_y, [lip[1]], yl[front][::-1], yu[1:]])
    return from_points(x, y, "cove main"), lip


def _synthetic_slat(n: int = 101):
    """The front of a NACA 4412 cut into a slat: upper surface to 25% chord,
    lower to 10% (the cusp), and a concave cove between them."""
    a = naca4("4412", n=n)
    le = a.leading_edge
    xl, yl = a.x[:le + 1][::-1], a.y[:le + 1][::-1]
    xu, yu = a.x[le:], a.y[le:]
    cusp = np.array([0.1, np.interp(0.1, xl, yl)])
    te = np.array([0.25, np.interp(0.25, xu, yu)])
    ctrl = np.array([0.11, np.interp(0.11, xu, yu) - 0.015])
    t = np.linspace(0.0, 1.0, 25)[:, None]
    cove = (1 - t) ** 2 * te + 2 * t * (1 - t) * ctrl + t**2 * cusp      # TE -> cusp, a Bezier arc
    lo, up = xl < 0.1 - 1e-9, (xu > 0.0) & (xu < 0.25 - 1e-9)
    x = np.concatenate([cove[:, 0], xl[lo][::-1], xu[up]])
    y = np.concatenate([cove[:, 1], yl[lo][::-1], yu[up]])
    return from_points(x, y, "synthetic slat"), cusp


@pytest.mark.parametrize("make, role", [(_cove_main, "main"), (_synthetic_slat, "slat")])
@pytest.mark.parametrize("n", [61, 101, 201])
def test_synthetic_cove_lips_are_found_to_one_surface_cell(make, role, n) -> None:
    airfoil, lip = make(n)
    e = Element(airfoil, role)
    k = cove_lip(e)
    assert k is not None
    cell = np.hypot(np.diff(airfoil.x), np.diff(airfoil.y))[max(k - 1, 0):k + 1].max()
    assert np.hypot(airfoil.x[k] - lip[0], airfoil.y[k] - lip[1]) <= cell


def test_a_smooth_main_element_has_no_cove_lip() -> None:
    assert cove_lip(Element(naca4("2412"), "main")) is None


def test_one_file_per_element_with_a_scale(tmp_path) -> None:
    a = naca4("0012", n=41)
    path = tmp_path / "flap.dat"
    path.write_text("flap\n" + "\n".join(f"{x:.12f} {y:.12f}" for x, y in zip(a.x, a.y)))
    e = read_element(path, "flap", scale=0.3)
    assert e.role == "flap" and e.airfoil.chord == pytest.approx(0.3)


def test_a_gap_out_of_reach_names_the_smallest_there(thirty_p) -> None:
    slat, main, flap = thirty_p
    with pytest.raises(ValueError, match=r"out of reach at an overlap of -0.024.*at least 0.024"):
        place(slat, main, 0.0, 0.015, None)        # the slat's trailing edge 2.4 %c ahead of the nose
    assert gap(place(slat, main, 0.0, 0.015, -0.01), main) == pytest.approx(0.015, abs=1e-10)
