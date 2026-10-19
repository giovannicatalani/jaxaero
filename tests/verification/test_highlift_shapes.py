"""The multi-element generator beyond the 30P30N: synthetic sections built
from NACA shapes, each re-placed as a designer would, with what the 30P30N
lacks -- sharp trailing edges, a main element without a cove step, a
smooth slat cove. Each must build unfolded, with its bodies, two
coarsenings, and quality at least the 30P30N's (minutes: marked slow)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))

from test_elements import _cove_main, _synthetic_slat  # noqa: E402

from jaxaero.geometry.airfoil import Airfoil, from_points, naca4  # noqa: E402
from jaxaero.geometry.elements import Element, gap, place  # noqa: E402
from jaxaero.mesh.assemble import assemble  # noqa: E402
from jaxaero.mesh.coarsen import coarsen  # noqa: E402
from jaxaero.mesh.highlift import Settings, multi_element  # noqa: E402
from jaxaero.mesh.quality import _cell_angles, quality  # noqa: E402

pytestmark = pytest.mark.slow


def naca23012(sharp: bool = True, n: int = 161) -> Airfoil:
    """The NACA 23012 (five-digit, 15 % camber position)."""
    b = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(b))
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3
                + (-0.1036 if sharp else -0.1015) * x**4)
    m, k1 = 0.2025, 15.957
    yc = np.where(x < m, k1 / 6 * (x**3 - 3 * m * x**2 + m * m * (3 - m) * x), k1 * m**3 / 6 * (1 - x))
    th = np.arctan(np.where(x < m, k1 / 6 * (3 * x**2 - 6 * m * x + m * m * (3 - m)), -k1 * m**3 / 6))
    xu, yu, xl, yl = x - yt * np.sin(th), yc + yt * np.cos(th), x + yt * np.sin(th), yc - yt * np.cos(th)
    X, Y = np.concatenate([xl[::-1], xu[1:]]), np.concatenate([yl[::-1], yu[1:]])
    if sharp:
        Y[0] = Y[-1] = 0.5 * (Y[0] + Y[-1])
    return from_points(X, Y, "NACA 23012")


def _moved(a: Airfoil, chord: float, dx: float, dy: float, deg: float) -> Airfoil:
    t = np.radians(deg)
    x, y = a.x * chord, a.y * chord
    return Airfoil(a.name, np.cos(t) * x - np.sin(t) * y + dx, np.sin(t) * x + np.cos(t) * y + dy)


def _flap(main: Element, shape: Airfoil, deflection: float, g: float, o: float) -> Element:
    te = main.trailing_edge
    return place(Element(_moved(shape, 0.3, te[0] - 0.02, te[1] - 0.03, -deflection), "flap"), main, 0.0, g, o)


def _slat(main: Element, shape: Airfoil) -> Element:
    le = main.leading_edge
    return place(Element(_moved(shape, 1.0, le[0] - 0.18, le[1] - 0.03, 25.0), "slat"), main, 0.0, 0.025, -0.01)


def _configurations():
    sharp, blunt = Element(naca23012(), "main"), Element(naca23012(sharp=False), "main")
    cove = Element(_cove_main(161)[0], "main")
    plain = Element(naca4("4412"), "main")
    slat = _synthetic_slat(161)[0]
    return {
        "sharp main and flap": lambda: [sharp, _flap(sharp, naca23012(), 20.0, 0.015, 0.01)],
        "blunt main and flap": lambda: [blunt, _flap(blunt, naca23012(sharp=False), 20.0, 0.015, 0.01)],
        "stepped cove and flap": lambda: [cove, _flap(cove, naca4("0012"), 25.0, 0.012, 0.005)],
        "slat and plain main": lambda: [_slat(plain, slat), plain],
        "slat, stepped cove, flap": lambda: [_slat(cove, slat), cove, _flap(cove, naca4("0012"), 25.0, 0.012, 0.005)],
    }


@pytest.mark.parametrize("name", list(_configurations()))
def test_a_synthetic_section_builds(name) -> None:
    elements = _configurations()[name]()
    raw, record = multi_element(elements, Settings(wall_spacing=1.5e-5, cells_per_chord=150))
    mesh, info = assemble(raw)
    assert info.bodies == len(elements) and set(info.patch_faces) == {"wall", "farfield"}
    assert info.interface_mismatch < 1e-12 and not quality(mesh).folded
    coarsen(coarsen(mesh))
    # outside the trailing-edge fans (2 % of the overall chord), as skewed as the 30P30N at worst
    tes = np.array([e.trailing_edge for e in elements])
    worst = 90.0
    for g in mesh:
        x, y = np.asarray(g.x_node), np.asarray(g.y_node)
        cx, cy = 0.25 * (x[1:, 1:] + x[:-1, 1:] + x[1:, :-1] + x[:-1, :-1]), 0.25 * (y[1:, 1:] + y[:-1, 1:] + y[1:, :-1] + y[:-1, :-1])
        far = np.min(np.hypot(cx[..., None] - tes[:, 0], cy[..., None] - tes[:, 1]), axis=-1) > 0.04
        if far.any():
            worst = min(worst, float(_cell_angles(x, y)[far].min()))
    assert worst >= 15.0, f"{name}: cells as skewed as {worst:.1f} deg"
    if name.startswith("slat"):
        assert gap(elements[0], elements[1]) == pytest.approx(0.025, abs=1e-9)
