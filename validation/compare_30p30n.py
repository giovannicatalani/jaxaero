#!/usr/bin/env python3
"""A 30P30N run's wall Cp against JAXA's RANS and the experiment, per element.

The experiment (validation/references/jaxa_30p30n/cp_exp_aoa*.dat) gives x/c
per element (slat, main, flap zones) round each contour; its stations take
their y from JAXA's RANS curve of that element (the RANS point of nearest x
on the side whose Cp is nearer). At each station our Cp (the wall point
nearest in x and y) is judged against the band between the experiment and
the RANS, widened by ``--margin``: the fraction of stations inside it, and
the largest excursion outside, per element.

    python validation/compare_30p30n.py RESULT_DIR [--alpha 5.5] [--margin 0.05]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "validation/references/jaxa_30p30n"
NAMES = ("slat", "main", "flap")


def experiment(alpha: float) -> list[np.ndarray]:
    """Per element, ``(n, 2)``: x/c and Cp."""
    zones, rows = [], None
    for line in (REFS / f"cp_exp_aoa{alpha:.1f}.dat").read_text().splitlines():
        if line.startswith("ZONE"):
            rows = []
            zones.append(rows)
        elif rows is not None:
            try:
                rows.append([float(v) for v in line.split()[:2]])
            except ValueError:
                continue
    return [np.array(z) for z in zones]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("result")
    p.add_argument("--alpha", type=float, default=5.5)
    p.add_argument("--margin", type=float, default=0.05)
    args = p.parse_args(argv)
    ours = np.loadtxt(next(Path(args.result).glob("*_surface.dat")), comments="#")   # x y body cp cf
    rans = np.loadtxt(REFS / f"cp_rans_jaxa_aoa{args.alpha:.1f}.dat", comments="#")      # x y cp
    bodies = sorted(set(ours[:, 2].astype(int)))
    if len(bodies) != 3:
        sys.exit(f"{len(bodies)} bodies in the run; the 30P30N has three")
    # each RANS point belongs to the body of ours nearest it
    d = np.hypot(rans[:, None, 0] - ours[None, :, 0], rans[:, None, 1] - ours[None, :, 1])
    rans_body = ours[np.argmin(d, axis=1), 2].astype(int)
    print(f"{Path(args.result).name}, alpha {args.alpha}: our Cp against the band between experiment and "
          f"JAXA's RANS, widened by {args.margin}")
    for b, exp in zip(bodies, experiment(args.alpha)):
        r, o = rans[rans_body == b], ours[ours[:, 2] == b]
        inside, worst = 0, 0.0
        for x, cp_exp in exp:
            near = np.flatnonzero(np.abs(r[:, 0] - x) <= np.sort(np.abs(r[:, 0] - x))[:4].max())
            k = near[np.argmin(np.abs(r[near, 2] - cp_exp))]                 # the side nearer in Cp
            j = np.argmin(np.hypot(o[:, 0] - x, o[:, 1] - r[k, 1]))
            lo, hi = sorted((cp_exp, r[k, 2]))
            out = max(lo - args.margin - o[j, 3], o[j, 3] - hi - args.margin, 0.0)
            inside += out == 0.0
            worst = max(worst, out)
        print(f"  {NAMES[b]:4s}: {inside}/{len(exp)} stations inside, worst excursion {worst:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
