#!/usr/bin/env python3
"""Compare a 3D wing run's sectional Cp with the ONERA M6 references.

Cuts the run's wall table (``*_surface.dat``: ``i k x y z cp ...``) at the
seven span stations of the experiment, ``eta = y / b`` = 0.20, 0.44, 0.65, 0.80,
0.90, 0.96, 0.99 (sections.py), and prints per station and surface the
max and RMS difference over 0.02 <= x/c <= 0.98 against

* the experiment, Schmitt and Charpin, AGARD AR-138 (``case_2308.dat`` or
  ``case_2565.dat``, taps at ``X/L``; ±0.02 at M 0.84), and
* CFL3D on the same grid level, when the run's cell count is one of TMR's
  structured levels (``CFL3D_OM6_A3p06_CPCF.dat``, L4 = 135,168 cells ...
  L1 = 69M),

with the upper-surface shock position (where Cp rises through the sonic
value, compare_surface.py) of each. Sources: references/tmr_onera_m6/README.md.

Examples:
    python validation/compare_sections.py results/validation/tmr_onera_m6_euler_n135168
    python validation/compare_sections.py RESULT --plot sections.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from jaxaero.plot import read_table
from sections import section

from compare_surface import critical_cp, differences, shock_position

REFERENCE = Path(__file__).resolve().parent / "references" / "tmr_onera_m6"
SEMISPAN = 1.4760179762
STATIONS = (0.20, 0.44, 0.65, 0.80, 0.90, 0.96, 0.99)
CFL3D_LEVELS = {135168: "L4", 1081344: "L3", 8650752: "L2", 69206016: "L1"}


def experiment(alpha_deg: float) -> list[tuple]:
    """Per station, ``(upper, lower)`` curves ``(x/c, cp)`` of the run nearest ``alpha``."""
    path = REFERENCE / ("case_2565.dat" if alpha_deg > 4.5 else "case_2308.dat")
    zones = re.split(r"^ZONE.*$", path.read_text(), flags=re.M)[1:]
    out = []
    for zone in zones:
        rows = np.array([[float(v) for v in line.split()] for line in zone.splitlines() if line.strip()])
        x, z, cp = rows[:, 2], rows[:, 3], rows[:, 4]
        surfaces = []
        for side in (z > 0.0, z <= 0.0):
            order = np.argsort(x[side])
            surfaces.append((x[side][order], cp[side][order]))
        out.append(tuple(surfaces))
    return out


def cfl3d(level: str) -> list[tuple]:
    """Per station, CFL3D's ``(upper, lower)`` on ``level``."""
    text = (REFERENCE / "CFL3D_OM6_A3p06_CPCF.dat").read_text()
    out = {}
    for block in re.split(r"^ZONE", text, flags=re.M)[1:]:
        title, *lines = block.splitlines()
        if f" {level}," not in title:
            continue
        eta = float(re.search(r"eta=([0-9.]+)", title).group(1))
        rows = np.array([[float(v) for v in line.split()] for line in lines if line.strip()])
        out[eta] = closed_loop(rows[:, 7], rows[:, 3], rows[:, 2])
    return [out[eta] for eta in STATIONS]


def ours(surface: dict[str, np.ndarray], eta: float) -> tuple:
    """Our ``(upper, lower)`` at ``eta``: the cut, its loop started at the
    trailing edge (largest ``x``) so that it splits at the leading edge."""
    cut = section(surface, eta * SEMISPAN)
    return closed_loop(cut["x_over_c"], cut["cp"], cut["z"])


def closed_loop(x: np.ndarray, cp: np.ndarray, z: np.ndarray) -> tuple:
    """``(upper, lower)`` of a loop round a section starting anywhere: rotated
    to start at the trailing edge, split at the leading edge, the upper branch
    the one of larger mean ``z`` (geometry, not Cp: near the tip both
    surfaces have similar Cp at quarter chord)."""
    start = int(np.argmax(x))
    x, cp, z = (np.roll(a, -start) for a in (x, cp, z))
    x, cp, z = (np.append(a, a[0]) for a in (x, cp, z))
    le = int(np.argmin(x))
    branches = []
    for part in (slice(0, le + 1), slice(le, None)):
        order = np.argsort(x[part])
        branches.append(((x[part][order], cp[part][order]), z[part].mean()))
    (a, za), (b, zb) = branches
    return (a, b) if za > zb else (b, a)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", help="a 3D run's result directory")
    parser.add_argument("--plot", type=Path, help="also draw the sections to this figure")
    args = parser.parse_args()
    directory = Path(args.result)
    meta = json.loads(next(directory.glob("*_metadata.json")).read_text())["extra"]
    flow = meta["case_setup"]["case"]["freestream"]
    surface = read_table(next(directory.glob("*_surface.dat")))
    surface = {{"x_over_c": "x", "y_over_c": "y", "z_over_c": "z"}.get(k, k): v for k, v in surface.items()}
    cp_star = critical_cp(flow["mach"])
    exp = experiment(flow.get("alpha_deg", 0.0))
    level = CFL3D_LEVELS.get(int(meta["cells"]))
    cfd = cfl3d(level) if level else None
    print(f"{directory.name}: {meta['cells']} cells, M {flow['mach']}, alpha {flow.get('alpha_deg', 0.0)}; "
          f"CFL3D level {level or 'none'}; Cp* {cp_star:.3f}")
    print(" eta  | vs experiment: upper max/rms   lower max/rms | vs CFL3D: upper rms  lower rms "
          "| shock x/c: ours  exp  CFL3D")
    for n, eta in enumerate(STATIONS):
        upper, lower = ours(surface, eta)
        (eu, el) = exp[n]
        du, dl = differences(upper, eu), differences(lower, el)
        text = f" {eta:.2f} | {du[0]:.3f}/{du[1]:.3f}   {dl[0]:.3f}/{dl[1]:.3f}"
        shocks = [shock_position(upper, cp_star), shock_position(eu, cp_star)]
        if cfd:
            cu, cl = cfd[n]
            text += f" | {differences(upper, cu)[1]:.3f}  {differences(lower, cl)[1]:.3f}"
            shocks.append(shock_position(cu, cp_star))
        print(text + " | " + "  ".join(f"{s:.3f}" for s in shocks))
    if args.plot:
        plot_sections(surface, exp, cfd, level, f"{directory.name}: {meta['cells']} cells, M {flow['mach']}, "
                      f"alpha {flow.get('alpha_deg', 0.0)} deg", args.plot)
        print(f"written {args.plot}")
    return 0


def plot_sections(surface, exp, cfd, level, title: str, out: Path) -> None:
    """Cp at the seven stations: this code (solid), CFL3D on the same grid
    (dashed), the experiment (markers); the stations on the planform."""
    from jaxaero.plot import SERIES, SURFACE, TEXT_SECONDARY, _figure, _save

    fig, axes = _figure(nrows=2, ncols=4, figsize=(15.0, 7.4))
    for n, (ax, eta) in enumerate(zip(axes.flat, STATIONS)):
        upper, lower = ours(surface, eta)
        for curve in (upper, lower):
            ax.plot(*curve, color=SERIES[0], linewidth=1.8, label="this code" if curve is upper else None)
        if cfd:
            for curve in cfd[n]:
                ax.plot(*curve, color=SERIES[1], linewidth=1.4, linestyle=(0, (5, 2)),
                        label=f"CFL3D, same grid ({level})" if curve is cfd[n][0] else None)
        for curve in exp[n]:
            ax.plot(*curve, linestyle="none", marker="o", markersize=4.5, color=SERIES[2],
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label="experiment (AGARD AR-138)" if curve is exp[n][0] else None)
        ax.set_title(f"η = {eta:.2f}")
        ax.set(xlim=(-0.03, 1.03), ylim=(1.2, -1.5))      # Cp: suction up
    for ax in axes[1, :3]:
        ax.set_xlabel("x / c (local)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cp")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    key = axes[1, 3]
    key.set_title("stations on the planform")
    tip = (1.4760179762 * np.tan(np.radians(30.0)), 1.4760179762)
    te_tip = tip[0] + 0.5625159852668158
    key.fill([0.0, tip[0], te_tip, 1.0], [0.0, tip[1], tip[1], 0.0], color=TEXT_SECONDARY, alpha=0.25)
    for eta in STATIONS:
        y = eta * SEMISPAN
        key.plot([y * np.tan(np.radians(30.0)), 1.0 + y * (te_tip - 1.0) / tip[1]], [y, y], color=SERIES[0],
                 linewidth=1.2)
    key.set(aspect="equal", xlabel="x / c_root", ylabel="y / c_root", ylim=(0.0, 1.6))
    fig.suptitle(title, fontweight="bold")
    _save(fig, out)


if __name__ == "__main__":
    raise SystemExit(main())
