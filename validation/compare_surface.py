#!/usr/bin/env python3
"""Compare a run's wall Cp and Cf with published distributions.

Reads the ``*_surface.dat`` that a run writes (x, Cp, Cf around the wall:
trailing edge, lower surface, leading edge, upper surface), splits it into
surfaces at the leading edge, interpolates ours onto each reference's stations
and prints the differences over 0.02 <= x <= 0.98 -- away from the
leading-edge suction peak, whose height is a resolution question, and the
trailing edge, where TMR itself warns of numerical error.

Reference sets (``--reference``, also named in each case's ``reference.json``):

* ``tmr_naca0012:<alpha>`` (0, 10, 15): CFL3D's SA Cp and upper-surface
  Cf on 897x257 (references/tmr_naca0012). CFL3D's lift has the point-vortex
  correction and ours does not; the shape of Cp is comparable, its level at the
  1e-3 scale is not.
* ``rae2822``: AGARD AR-138 case 6 experiment and WIND SA on the NPARC 369x65
  grid (references/rae2822; both files store -Cp).

Where the upper surface is supersonic the shock position is printed too: where the upper-surface Cp first rises
  through the critical value Cp* (sonic at the wall),

      Cp* = 2 / (gamma M^2) [ ((2 + (gamma-1) M^2) / (gamma+1))^(gamma/(gamma-1)) - 1 ],

  interpolated linearly between stations -- one definition applied to every
  curve alike.

Examples:
    python validation/compare_surface.py results/validation/tmr_naca0012_sa_a10_n896 --reference tmr_naca0012:10
    python validation/compare_surface.py results/validation/nparc_rae2822_case6_n368 --reference rae2822
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from jaxaero.plot import read_table

REFERENCES = Path(__file__).resolve().parent / "references"
GAMMA = 1.4

Curve = tuple[np.ndarray, np.ndarray]


# --- Reading ------------------------------------------------------------------------


def read_zones(path: Path) -> dict[str, np.ndarray]:
    """Tecplot-style zones of two columns, keyed by zone title."""
    zones: dict[str, list[list[float]]] = {}
    current = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("zone"):
            current = stripped.split("=", 1)[1].strip().strip('"')
            zones[current] = []
        elif current is not None and stripped and stripped[0] in "-+.0123456789":
            zones[current].append([float(v) for v in stripped.split()[:2]])
    return {k: np.asarray(v) for k, v in zones.items()}


def read_genplot(path: Path) -> np.ndarray:
    """The two-column data rows of a single-curve GENPLOT file (NPARC/WIND):
    a header, then a point count, then ``x value`` rows."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.asarray(rows)


def split_at_leading_edge(x: np.ndarray, y: np.ndarray) -> tuple[Curve, Curve]:
    """Two branches at the minimum ``x``, each sorted by ``x``."""
    k = int(np.argmin(x))
    first, second = (x[: k + 1], y[: k + 1]), (x[k:], y[k:])
    return tuple(tuple(a[np.argsort(b[0])] for a in b) for b in (first, second))


def upper_lower(x: np.ndarray, cp: np.ndarray) -> tuple[Curve, Curve]:
    """Split a Cp loop at the leading edge; the upper surface is the one with
    the more negative Cp at quarter chord."""
    a, b = split_at_leading_edge(x, cp)
    at = lambda c: np.interp(0.25, *c)  # noqa: E731
    return (a, b) if at(a) < at(b) else (b, a)


# --- Measures -----------------------------------------------------------------------


def differences(ours: Curve, theirs: Curve, window=(0.02, 0.98)) -> tuple[float, float]:
    """Max and RMS of ours minus theirs at their stations inside ``window``."""
    (xo, yo), (xt, yt) = ours, theirs
    inside = (xt >= window[0]) & (xt <= window[1])
    d = np.interp(xt[inside], xo, yo) - yt[inside]
    return float(np.abs(d).max()), float(np.sqrt(np.mean(d * d)))


def critical_cp(mach: float, gamma: float = GAMMA) -> float:
    """Pressure coefficient at which the local flow is sonic (isentropic)."""
    ratio = (2.0 + (gamma - 1.0) * mach**2) / (gamma + 1.0)
    return 2.0 / (gamma * mach**2) * (ratio ** (gamma / (gamma - 1.0)) - 1.0)


def shock_position(upper: Curve, cp_star: float, start: float = 0.1) -> float:
    """First ``x >= start`` where the upper-surface Cp rises through ``cp_star``
    (supersonic -> subsonic), linearly interpolated; NaN if there is none."""
    x, cp = upper
    keep = x >= start
    x, cp = x[keep], cp[keep]
    for k in np.flatnonzero((cp[:-1] < cp_star) & (cp[1:] >= cp_star)):
        return float(x[k] + (cp_star - cp[k]) / (cp[k + 1] - cp[k]) * (x[k + 1] - x[k]))
    return float("nan")


# --- Reference sets -----------------------------------------------------------------


def tmr_naca0012(alpha: int) -> dict[str, dict[str, Curve]]:
    """CFL3D, SA, 897x257, point-vortex far field."""
    base = REFERENCES / "tmr_naca0012"
    ref_cp = read_zones(base / "n0012cp_cfl3d_sa.dat")[f"alpha={alpha}"]
    ref_cf = read_zones(base / "n0012cf_cfl3d_sa.dat")[f"alpha={alpha}, upper surface"]
    upper, lower = upper_lower(ref_cp[:, 0], ref_cp[:, 1])
    cf = ref_cf[np.argsort(ref_cf[:, 0])]
    return {"CFL3D 897x257": {"Cp upper": upper, "Cp lower": lower,
                              "Cf upper": (cf[:, 0], cf[:, 1])}}


def rae2822() -> dict[str, dict[str, Curve]]:
    """AGARD AR-138 case 6 experiment and WIND SA on the same grid; both
    files store -Cp. WIND's Cf file is not in Cf units (see the README)."""
    base = REFERENCES / "rae2822"
    out = {}
    for label, name in (("experiment", "cp.exp.gen"), ("WIND SA 369x65", "cp.wind.gen")):
        data = read_genplot(base / name)
        upper, lower = upper_lower(data[:, 0], -data[:, 1])
        out[label] = {"Cp upper": upper, "Cp lower": lower}
    return out


REFERENCE_SETS = {"tmr_naca0012": tmr_naca0012, "rae2822": rae2822}


def reference_set(spec: str) -> dict[str, dict[str, Curve]]:
    """A set by its ``reference.json`` name: ``"rae2822"``, ``"tmr_naca0012:10"``."""
    name, _, arg = spec.partition(":")
    return REFERENCE_SETS[name](int(arg)) if arg else REFERENCE_SETS[name]()


def joined(curves: dict[str, Curve], *keys: str) -> Curve:
    """Surfaces as one point set, NaN between them so a line does not join
    one surface's trailing edge to the next's leading edge."""
    gap = np.array([np.nan])
    return (np.concatenate([np.r_[curves[k][0], gap] for k in keys]),
            np.concatenate([np.r_[curves[k][1], gap] for k in keys]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result", type=Path, help="a result directory written by a run")
    p.add_argument("--reference", required=True, help="rae2822 | tmr_naca0012:0|10|15")
    args = p.parse_args(argv)
    references = reference_set(args.reference)
    surface = read_table(Path(glob.glob(str(args.result / "*_surface.dat"))[0]))
    mach = json.loads(Path(glob.glob(str(args.result / "*_metadata.json"))[0]).read_text())["config"]["flow"]["mach"]
    x, cp = surface["x"], surface["cp"]
    upper, lower = upper_lower(x, cp)
    ours = {"Cp upper": upper, "Cp lower": lower}
    if "cf" in surface:
        # Ours runs TE -> lower -> LE -> upper: the second branch is the upper surface.
        ours["Cf upper"] = split_at_leading_edge(x, surface["cf"])[1]

    cp_star = critical_cp(mach)
    print(f"{args.result.name} against {args.reference} (0.02 <= x <= 0.98):")
    for label, curves in [("this code", ours), *references.items()]:
        if label != "this code":
            for name, theirs in curves.items():
                worst, rms = differences(ours[name], theirs)
                print(f"  {label}, {name}: max |diff| {worst:.3e}, rms {rms:.3e}")
        shock = shock_position(curves["Cp upper"], cp_star)
        print(f"  {label}: suction peak {curves['Cp upper'][1].min():+.3f}"
              + ("" if np.isnan(shock) else f", shock (Cp rises through Cp* = {cp_star:+.4f}) at {shock:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
