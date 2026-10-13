#!/usr/bin/env python3
"""Write the Karman-Trefftz O-grids the validation cases read, as Plot3D.

Vassberg and Jameson's conformal meshes (and the HiOCFD / TMR-section
families built the same way) come from a closed-form mapping that only these
references need, so it lives here (karman_trefftz.py), not in the library;
the cases read its output like any grid file. Every level of every case
whose reference.json names a ``data/grids/kt_*`` file is written:

    python validation/grids/make_grids.py                         # all, into data/grids/
    python validation/grids/make_grids.py hiocfd 512 704 2.5e-5   # one more

File names: ``kt_<section>_<cells around>[x<cells normal>_<wall spacing>].p2dfmt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.grids.karman_trefftz import HIOCFD, TMR, VJ, make_o_grid  # noqa: E402

SECTIONS = {"vj": VJ, "hiocfd": HIOCFD, "tmr": TMR}

LEVELS = {
    "vj": [(n, None, None) for n in (32, 64, 128, 256)],
    "hiocfd": [(32, 44, 4e-4), (64, 88, 2e-4), (128, 176, 1e-4), (256, 352, 5e-5)],
    "tmr": [(64, 88, 2e-4), (128, 176, 1e-4), (256, 352, 5e-5)],
}


def name(section: str, n: int, nj: int | None, spacing: float | None) -> str:
    return f"kt_{section}_{n}.p2dfmt" if nj is None else f"kt_{section}_{n}x{nj}_{spacing:g}.p2dfmt"


def write_plot3d(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """Formatted 2D Plot3D, 17 significant digits: read back bitwise."""
    with open(path, "w") as f:
        f.write(f"{x.shape[0]} {x.shape[1]}\n")
        for c in (x, y):
            np.savetxt(f, c.T.reshape(-1, 1), fmt="%.17e")


def main(argv: list[str]) -> int:
    out = ROOT / "data" / "grids"
    out.mkdir(parents=True, exist_ok=True)
    if argv:
        section, n, *rest = argv
        todo = {section: [(int(n), int(rest[0]), float(rest[1])) if rest else (int(n), None, None)]}
    else:
        todo = LEVELS
    for section, levels in todo.items():
        for n, nj, spacing in levels:
            grid = make_o_grid(n, dtype=np.float64, section=SECTIONS[section]) if nj is None else \
                make_o_grid(n, nj, spacing, dtype=np.float64, section=SECTIONS[section])
            path = out / name(section, n, nj, spacing)
            write_plot3d(path, np.asarray(grid.x_node), np.asarray(grid.y_node))
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
