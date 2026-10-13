"""Command line: run one case and write its result.

    python -m jaxaero CASE_DIR [--set file.section.key=value ...] [--out DIR] [--quiet] [--check]

The case directory holds ``case.json`` and ``numerical.json`` (format:
``docs/guide.md``). ``--set`` changes one key for this run; the resolved
files are saved with the result. ``--check`` builds the grid, prints its
patches, conditions and bodies, draws it, and stops before solving.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from . import CaseError, load_case, run
from .core.state import first_bad_cell
from .mesh.quality import quality


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jaxaero", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="case directory (case.json + numerical.json)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="FILE.SECTION.KEY=VALUE", help="override one key; repeatable")
    parser.add_argument("--out", type=Path, help="output directory (default: <save_path>/<case_name>)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    parser.add_argument("--check", action="store_true",
                        help="build and report the grid, write grid.png, do not solve")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # readable while a long run is going

    try:
        case = load_case(args.case, tuple(args.overrides))
        grid = case.grid()
    except (CaseError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{case.name}: {case.description}")
    print(f"grid {case.grid_spec.describe(case.directory)}: {len(grid.mesh)} block(s), {grid.cells} cells, "
          f"{grid.walls.faces.bodies} body(ies); {case.numerics.solver}, {case.numerics.flux}, "
          f"limiter {case.numerics.limiter}")
    print(quality(grid.mesh).summary())
    if args.check:
        return check(case, grid, args.out)

    def progress(iteration, residual, orders, forces):
        print(f"{iteration:8d}  res {residual:.4e} ({orders:5.2f} orders)  "
              f"{forces.summary().splitlines()[0]}")

    result = run(case, grid, progress=None if args.quiet else progress)
    directory = result.write(args.out)
    print(f"{result.history.summary()}\n{result.forces.summary()}\n{result.timing.summary()}\n"
          f"written to {directory}")
    physical, cell = first_bad_cell(result.U, case.flow.gamma, grid.dim)
    if not physical:
        at = ", ".join(f"{float(c):.4f}" for c in grid.grid.centres[(slice(None),) + cell])
        print(f"non-physical state, first at canvas cell {cell}: ({at})", file=sys.stderr)
    return 0 if physical and result.history.reason != "non_finite" else 1


def check(case, grid, out: Path | None) -> int:
    """Patches with their conditions, bodies with their extent, and a picture."""
    faces: dict[str, list] = {}
    for block in grid.mesh:
        for _, seg in block.sides.boundaries():
            n = int(np.prod(np.subtract(seg.stop, seg.start)))      # a segment, or a 3D rectangle
            faces.setdefault(seg.patch, [seg.kind, 0])[1] += n
    for patch, (kind, n) in faces.items():
        print(f"patch {patch!r}: {kind}, {n} faces")
    walls = grid.walls.faces
    if grid.dim == 3:
        lo, hi = walls.corners.min(axis=(0, 1)), walls.corners.max(axis=(0, 1))
        print(f"wall: {len(walls.mid)} faces, x {lo[0]:.4f} to {hi[0]:.4f}, y {lo[1]:.4f} to {hi[1]:.4f}, "
              f"z {lo[2]:.4f} to {hi[2]:.4f}")
        print("no grid picture: the picture is 2D only")
        return 0
    for body in range(walls.bodies):
        x = walls.a[walls.body == body]
        print(f"body {body}: x {x[:, 0].min():.4f} to {x[:, 0].max():.4f}, "
              f"y {x[:, 1].min():.4f} to {x[:, 1].max():.4f}, {len(x)} wall faces")
    if importlib.util.find_spec("matplotlib") is None:
        print("no grid picture: pip install matplotlib")
        return 0
    from .plot import plot_grid

    path = (out if out is not None else case.output / case.name) / "grid.png"
    plot_grid(grid, out=path)
    print(f"grid drawn in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
