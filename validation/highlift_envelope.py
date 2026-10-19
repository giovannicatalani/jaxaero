#!/usr/bin/env python3
"""The multi-element generator across the 30P30N's slat and flap settings.

Each setting re-places the slat and the flap of the 30P30N (BANC geometry,
validation/references/jaxa_30p30n/) and builds the three-element grid; it
either builds, and its quality is measured as tests/unit/test_highlift.py
measures it, or it is refused with the generator's message. Deflections are
changes from the given 30 deg (positive: further deployed), gaps and
overlaps in stowed chords:

    python validation/highlift_envelope.py                 # the sample, 6 processes
    python validation/highlift_envelope.py --jobs 2 --cells-per-chord 120

The markdown table it prints is docs/multi-element.md's envelope.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxaero.geometry.elements import fan_points, gap, overlap, place, read_elements  # noqa: E402
from jaxaero.mesh.assemble import assemble  # noqa: E402
from jaxaero.mesh.highlift import Settings, multi_element  # noqa: E402
from jaxaero.mesh.quality import _cell_angles, interface_growth, quality  # noqa: E402

CRV = ROOT / "validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv"
GIVEN = {"flap": (0.0, 0.0127, 0.0014), "slat": (0.0, 0.0295, None)}


def sample() -> list[dict]:
    """The settings: the flap swept with the slat as given, the slat swept
    with the flap as given, and the extreme corners of both."""
    out = [{"flap": f, "slat": GIVEN["slat"]}
           for f in itertools.product((-10.0, 0.0, 10.0), (0.005, 0.0127, 0.03), (-0.01, 0.0014, 0.02))]
    # a slat gap under its given overlap's 2.4 %c is out of reach: the small gaps at -1 %c
    out += [{"flap": GIVEN["flap"], "slat": (d, g, None if g > 0.025 else -0.01)}
            for d, g in itertools.product((-10.0, -5.0, 0.0), (0.015, 0.0295, 0.04)) if (d, g) != (0.0, 0.0295)]
    out += [{"flap": (fd, fg, 0.0014), "slat": (sd, sg, None if sg > 0.025 else -0.01)}
            for fd, fg, sd, sg in itertools.product((-10.0, 10.0), (0.005, 0.03), (-10.0,), (0.015, 0.04))]
    return out


def measure(elements, settings: Settings) -> dict:
    """Build and measure one grid (or the refusal)."""
    start = time.time()
    try:
        raw, record = multi_element(elements, settings)
        mesh, info = assemble(raw)
    except ValueError as error:
        return {"built": False, "why": str(error), "seconds": time.time() - start}
    fans = np.concatenate([fan_points(e) for e in elements])
    worst_angle = 90.0
    for g in mesh:
        x, y = np.asarray(g.x_node), np.asarray(g.y_node)
        c = np.stack([0.25 * (x[1:, 1:] + x[:-1, 1:] + x[1:, :-1] + x[:-1, :-1]).ravel(),
                      0.25 * (y[1:, 1:] + y[:-1, 1:] + y[1:, :-1] + y[:-1, :-1]).ravel()], 1)
        d = np.hypot(c[:, None, 0] - fans[None, :, 0], c[:, None, 1] - fans[None, :, 1]).min(axis=1)
        far = (d > settings.layer_thickness + 0.02).reshape(g.ni, g.nj)
        if far.any():
            worst_angle = min(worst_angle, float(_cell_angles(x, y)[far].min()))
    growth = interface_growth(mesh)
    return {"built": True, "cells": record["cells"], "bodies": info.bodies, "folded": quality(mesh).folded,
            "min_angle": worst_angle, "growth_peak": float(growth.max()), "growth_median": float(np.median(growth)),
            "seconds": time.time() - start}


def run(setting: dict, cells_per_chord: float) -> dict:
    slat, main, flap = read_elements(CRV, ["slat", "main", "flap"], 1.0 / 18.0)
    try:
        flap = place(flap, main, *setting["flap"])
        slat = place(slat, main, *setting["slat"])
    except ValueError as error:
        return {**setting, "built": False, "why": f"placement: {error}", "seconds": 0.0}
    placed = {"flap_gap": gap(main, flap), "flap_overlap": overlap(main, flap, main),
              "slat_gap": gap(slat, main)}
    return {**setting, **placed, **measure([slat, main, flap], Settings(1.5e-5, cells_per_chord))}


def table(rows: list[dict]) -> str:
    head = ("| flap defl. | flap gap | flap overlap | slat defl. | slat gap | result | cells | min angle "
            "| interface growth (median, peak) | s |")
    out = [head, "|" + "---|" * 10]
    for r in rows:
        (fd, fg, fo), (sd, sg, _) = r["flap"], r["slat"]
        left = f"| {fd:+.0f} | {fg:.4f} | {fo:+.4f} | {sd:+.0f} | {sg:.4f} |"
        if r["built"]:
            res = "folded" if r["folded"] else "built"
            out.append(f"{left} {res} | {r['cells']} | {r['min_angle']:.1f} | {r['growth_median']:.2f}, "
                       f"{r['growth_peak']:.1f} | {r['seconds']:.0f} |")
        else:
            out.append(f"{left} refused: {r['why'][:110]} | | | | {r['seconds']:.0f} |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cells-per-chord", type=float, default=150.0)
    args = parser.parse_args()
    settings = sample()
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run, settings, [args.cells_per_chord] * len(settings)))
    print(table(rows))


if __name__ == "__main__":
    main()
