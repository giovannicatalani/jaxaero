#!/usr/bin/env python3
"""A case's grid, each block in its own colour, in one or more windows side
by side: user post-processing, not part of the library (``jaxaero CASE
--check`` draws the whole grid in one colour per body).

    python scripts/plot_blocks.py CASE_DIR [--set file.section.key=value ...]
        [--window x0 x1 y0 y1 ...] [--out grid_blocks.png]

Each ``--window`` adds a panel (chords); without one, the grid's walls fill
the panel.
"""

from __future__ import annotations

import argparse

import numpy as np

SURFACE, TEXT = "#fcfcfb", "#0b0b0b"
COLOURS = ["#2a6fdb", "#e0782d", "#2f9e66", "#c23b5b", "#7a5bd1", "#8a6a3f", "#d45fb4", "#5f6b75",
           "#b5a12a", "#2aa5b5"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case")
    parser.add_argument("--set", action="append", default=[], metavar="file.section.key=value")
    parser.add_argument("--window", action="append", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--out", default="grid_blocks.png")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jaxaero import load_case

    case = load_case(args.case, tuple(args.set))
    mesh = case.grid_spec.build(case.directory)
    walls = [np.stack([np.asarray(g.x_node), np.asarray(g.y_node)], -1)[:, 0] for g in mesh
             if any(seg.patch == "wall" for _, seg in g.sides.boundaries())]
    windows = args.window or [[*(lambda p: (p[:, 0].min() - 0.05, p[:, 0].max() + 0.05,
                                           p[:, 1].min() - 0.1, p[:, 1].max() + 0.1))(np.concatenate(walls))]]
    fig, axes = plt.subplots(1, len(windows), figsize=(7 * len(windows), 5.5), squeeze=False,
                             facecolor=SURFACE)
    for ax, (x0, x1, y0, y1) in zip(axes[0], windows):
        ax.set_facecolor(SURFACE)
        for k, g in enumerate(mesh):
            x, y = np.asarray(g.x_node), np.asarray(g.y_node)
            inside = (x.max() >= x0) & (x.min() <= x1) & (y.max() >= y0) & (y.min() <= y1)
            if inside:
                c = COLOURS[k % len(COLOURS)]
                ax.plot(x, y, color=c, lw=0.3)
                ax.plot(x.T, y.T, color=c, lw=0.3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.tick_params(colors=TEXT, labelsize=8)
    fig.suptitle(f"{case.name}: {len(mesh)} blocks, {sum(g.ni * g.nj for g in mesh)} cells", color=TEXT)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
