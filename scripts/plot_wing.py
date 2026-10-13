#!/usr/bin/env python3
"""Wall-pressure figures of a 3D wing result: user post-processing, not part of
the library. Reads only the files a run writes (``*_surface.dat`` with columns
``i k x y z cp``, ``*_fields.npz`` for the wall nodes, ``*_metadata.json``).

* ``wing_cp_surfaces.png``: the upper and the lower surface in planform, each a
  filled contour plot of Cp on viridis (bands every 0.05, isobars every 0.1,
  the sonic isobar Cp* in white: on the upper surface, the shocks' feet);
* ``wing_3d.png``: the wing alone in true-scale perspective, shaded by Cp, with
  isobars every 0.05 on the upper surface.

Assumes the wing-grid layout of TMR's generator: the wall is ``j = 0`` of one
block, ``i`` spanwise, ``k`` round the section from the leading edge.

    python scripts/plot_wing.py RESULT_DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, TEXT, TEXT_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = "viridis"          # perceptually uniform, colour-blind safe: low Cp (suction) dark, high light


def read_table(path: Path) -> dict[str, np.ndarray]:
    names = path.read_text().splitlines()[0].lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    short = {"x_over_c": "x", "y_over_c": "y", "z_over_c": "z"}
    return {short.get(n, n): data[:, k] for k, n in enumerate(names)}


def load(directory: Path):
    """Wall nodes ``(3, ni+1, nk+1)``, face Cp ``(ni, nk)``, the upper and lower
    ``k`` ranges of faces, and the freestream."""
    one = lambda suffix: next(directory.glob(f"*{suffix}"))  # noqa: E731
    fields = np.load(one("_fields.npz"))
    nodes = np.stack([fields[f"{c}_node_0"][:, 0, :] for c in "xyz"])
    surface = read_table(one("_surface.dat"))
    i, k = surface["i"].astype(int), surface["k"].astype(int)
    cp = np.full((i.max() + 1, k.max() + 1), np.nan)
    cp[i, k] = surface["cp"]
    mid = nodes[:, nodes.shape[1] // 2, :-1]              # a mid-span section loop
    te = int(np.argmax(mid[0]))
    first, second = slice(0, te), slice(te, nodes.shape[2] - 1)
    upper, lower = (first, second) if mid[2, first].mean() > mid[2, second].mean() else (second, first)
    meta = json.loads(one("_metadata.json").read_text())
    return nodes, cp, upper, lower, meta["extra"]["case_setup"]["case"]["freestream"], directory.name


def node_values(face: np.ndarray) -> np.ndarray:
    """Face values ``(n, m)`` moved to the nodes ``(n+1, m+1)``: the mean of the faces around."""
    p = np.pad(face, 1, mode="edge")
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


def critical_cp(mach: float, gamma: float = 1.4) -> float:
    ratio = (2.0 + (gamma - 1.0) * mach**2) / (gamma + 1.0)
    return 2.0 / (gamma * mach**2) * (ratio ** (gamma / (gamma - 1.0)) - 1.0)


def colour_scale(cp: np.ndarray, step: float):
    import matplotlib
    from matplotlib.colors import Normalize

    # The range of the bulk of the wall (1st-99th percentile): the leading-edge
    # suction spike and the stagnation line would otherwise squeeze the rest
    # into a few colours; values beyond it are clipped (the colour bar's ends).
    finite = cp[np.isfinite(cp)]
    lo = np.floor(np.percentile(finite, 1.0) / step) * step
    hi = np.ceil(np.percentile(finite, 99.0) / step) * step
    return np.arange(lo, hi + 0.5 * step, step), matplotlib.colormaps[CMAP], Normalize(lo, hi)


def draw_sonic(lines) -> None:
    """The sonic isobar white with a black edge, legible on every part of the map."""
    import matplotlib.patheffects as pe

    lines.set(path_effects=[pe.Stroke(linewidth=3.4, foreground=TEXT), pe.Normal()])


def style(plt) -> None:
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT, "text.color": TEXT,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False})


def plot_surfaces(data, out: Path, step: float = 0.05) -> None:
    """Upper and lower surface in planform: filled contours on the wall grid's
    own triangulation (no hull across the swept tip)."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    nodes, cp, upper, lower, flow, name = data
    levels, cmap, norm = colour_scale(cp, step)
    cp_star = critical_cp(flow["mach"])
    style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4), sharey=True, layout="constrained")
    for ax, (title, ks) in zip(axes, (("Upper surface", upper), ("Lower surface", lower))):
        x = nodes[0][:, ks.start:ks.stop + 1]
        y = nodes[1][:, ks.start:ks.stop + 1]
        value = node_values(cp[:, ks])
        n, m = x.shape
        index = np.arange(n * m).reshape(n, m)
        a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
        triangles = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        tri = Triangulation(x.ravel(), y.ravel(), triangles)
        filled = ax.tricontourf(tri, value.ravel(), levels=levels, cmap=cmap, norm=norm, extend="both")
        ax.tricontour(tri, value.ravel(), levels=levels[::2], colors=TEXT, linewidths=0.35, alpha=0.35,
                      linestyles="solid")
        if np.nanmin(value) < cp_star:
            draw_sonic(ax.tricontour(tri, value.ravel(), levels=[cp_star], colors="white", linewidths=1.8,
                                     linestyles="solid"))
        ax.set(aspect="equal", title=title, xlabel="x / c_root")
        ax.set_ylim(0.0, 1.03 * nodes[1].max())
    axes[0].set_ylabel("y / c_root (span; root at y = 0)")
    bar = fig.colorbar(filled, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label(f"Cp   (bands {step:g}; white line: sonic, Cp* = {cp_star:.3f})")
    re = flow.get("reynolds")
    fig.suptitle(f"Wall pressure: {name}   (M {flow['mach']:g}, α {flow.get('alpha_deg', 0.0):g}°"
                 + (f", Re {re:.3g}" if re else ", inviscid") + ")", fontweight="bold")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def refine(nodes: np.ndarray, value: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear refinement of node arrays ``(..., n+1, m+1)``."""
    t = np.linspace(0.0, 1.0, factor + 1)[:-1]

    def one(a):
        for axis in (-2, -1):
            a = np.moveaxis(a, axis, -1)
            fine = (a[..., :-1, None] * (1.0 - t) + a[..., 1:, None] * t).reshape(a.shape[:-1] + (-1,))
            a = np.moveaxis(np.concatenate([fine, a[..., -1:]], axis=-1), -1, axis)
        return a

    return one(nodes), one(value)


def plot_3d(data, out: Path, step: float = 0.05, factor: int = 4) -> None:
    """The wing alone: smooth Cp shading, isobars on the upper surface."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scipy.ndimage import map_coordinates

    nodes, cp, upper, lower, flow, name = data
    levels, cmap, norm = colour_scale(cp, step)
    style(plt)
    fig = plt.figure(figsize=(10.0, 7.0))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    scratch = Figure().add_subplot()
    for ks in (lower, upper):                              # the surface in view drawn last
        fine, value = refine(nodes[:, :, ks.start:ks.stop + 1], node_values(cp[:, ks]), factor)
        face = np.clip(0.25 * (value[:-1, :-1] + value[1:, :-1] + value[:-1, 1:] + value[1:, 1:]),
                       levels[0], levels[-1])
        ax.plot_surface(*fine, facecolors=cmap(norm(face)), linewidth=0, antialiased=False, shade=False,
                        rcount=fine.shape[1], ccount=fine.shape[2])
        if ks is lower:
            continue
        for segments in scratch.contour(np.arange(value.shape[1]), np.arange(value.shape[0]), value,
                                        levels=levels).allsegs:
            for seg in segments:
                if len(seg) > 1:
                    at = np.stack([seg[:, 1], seg[:, 0]])
                    ax.plot(*[map_coordinates(c, at, order=1) for c in fine], color=TEXT, linewidth=0.45,
                            alpha=0.4)
    lo, hi = nodes.min(axis=(1, 2)), nodes.max(axis=(1, 2))
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
    ax.set(xlim=(centre[0] - half, centre[0] + half), ylim=(centre[1] - half, centre[1] + half),
           zlim=(centre[2] - half, centre[2] + half))
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.15)
    ax.view_init(elev=38, azim=-62)
    ax.set_axis_off()
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="a 3D wing result directory")
    parser.add_argument("--out", type=Path, help="figure directory (default: RESULT/figures)")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    out = args.out or args.result / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.result)
    plot_surfaces(data, out / "wing_cp_surfaces.png")
    plot_3d(data, out / "wing_3d.png")
    print(out / "wing_cp_surfaces.png", out / "wing_3d.png", sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
