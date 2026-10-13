"""Sectional distributions of a 3D wing: the wall cut at a span station.

The wall of a wing grid is a surface of faces ``(i, k)``: ``i`` spanwise, ``k``
round the section (TMR's wing grids). Along each line of
constant ``k`` the face centres run monotonically in ``y`` over the semispan,
so a cut at ``y`` interpolates every face quantity linearly in ``y`` on each
such line: one point per ``k``, in order round the section. Its chord is from
the cut's smallest to its largest ``x``; ``x/c`` is measured from the first.
Works on the wall table a run writes (post/forces.py ``surface``: columns
``i``, ``k``, ``x``, ``y``, ``z``, ``cp`` ...).
"""

from __future__ import annotations

import numpy as np


def section(surface: dict[str, np.ndarray], y: float) -> dict[str, np.ndarray]:
    """The wall distributions cut at span ``y``: per surface line round the
    section, ``x``, ``z``, ``x_over_c`` and every other column. Lines that do
    not reach ``y`` are left out."""
    i, k = surface["i"].astype(int), surface["k"].astype(int)
    shape = (i.max() + 1, k.max() + 1)
    names = [n for n in surface if n not in ("i", "k")]
    grid = {name: np.full(shape, np.nan) for name in names}
    for name in names:
        grid[name][i, k] = surface[name]
    out: dict[str, list] = {name: [] for name in names}
    for line in range(shape[1]):
        ys = grid["y"][:, line]
        order = np.argsort(ys)
        ys = ys[order]
        if not (ys[0] <= y <= ys[-1]):
            continue
        for name in names:
            out[name].append(np.interp(y, ys, grid[name][order, line]))
    cut = {name: np.asarray(v) for name, v in out.items()}
    x_le, x_te = cut["x"].min(), cut["x"].max()
    cut["x_over_c"] = (cut["x"] - x_le) / (x_te - x_le)
    return cut
