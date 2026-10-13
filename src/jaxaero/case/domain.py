"""The ``domain`` section of a case: which grid, and how to obtain it.

One grid per case: generated round an airfoil (``o_grid``, ``c_grid``:
mesh/generate.py, the airfoil a NACA 4-digit name or a coordinate file), or
read from any structured grid file (``file``: Plot3D or CGNS, single- or
multi-block, mesh/assemble.py). Paths are resolved against the case
directory first, the working directory second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..geometry.airfoil import Airfoil, airfoil_source
from ..mesh.assemble import PatchRule, RawBlock, assemble, load_grid
from ..mesh.assemble3d import is_3d_plot3d, load_grid_3d
from ..mesh.generate import Spacing, c_grid, o_grid
from ..mesh.mesh import Mesh
from ..post.forces import ForceReference


@dataclass(frozen=True)
class GridSpec:
    """How to obtain the grid (``docs/case_reference.md`` lists every key)."""

    grid: str = "c_grid"
    airfoil: str | None = None
    cells_airfoil: int | None = None
    cells_wake: int | None = None
    cells_normal: int | None = None
    wall_spacing: float | None = None
    far_field: float = 100.0
    trailing_edge: str = "as_given"
    leading_edge_spacing: float = Spacing.leading_edge
    trailing_edge_spacing: float = Spacing.trailing_edge
    file: str | None = None
    format: str = "auto"
    coordinate_scale: float = 1.0
    patches: tuple[dict[str, Any], ...] = field(default=())
    reference_length: float = 1.0
    reference_area: float | None = None
    moment_centre: tuple[float, ...] = (0.25, 0.0, 0.0)

    def force_reference(self) -> ForceReference:
        return ForceReference(self.reference_length, self.reference_area, tuple(self.moment_centre))

    def build(self, base: Path, dtype=np.float64) -> Mesh:
        """The grid's blocks, their patches named but not yet given conditions."""
        return GRIDS[self.grid](self, base, dtype)

    def rules(self) -> tuple[PatchRule, ...]:
        return tuple(PatchRule(**rule) for rule in self.patches)

    def describe(self, base: Path | None = None) -> str:
        if self.grid == "file":
            return f"file {Path(self.file).name}"
        wake = f" + 2x{self.cells_wake} wake" if self.grid == "c_grid" else ""
        text = (f"{self.grid} round {self.airfoil}: {self.cells_airfoil}{wake} x {self.cells_normal} cells, "
                f"wall spacing {self.wall_spacing:g}, far field {self.far_field:g}")
        if base is not None:
            text += "".join(f"; {note}" for note in self.airfoil_shape(base)[1])
        return text

    def airfoil_shape(self, base: Path) -> tuple[Airfoil, list[str]]:
        """The airfoil the grid is built round, and what was done to it."""
        given = airfoil_source(self.airfoil, base)
        airfoil, notes = given.unit_chord(), []
        if abs(given.chord - 1.0) > 1e-6:
            notes.append(f"airfoil chord {given.chord:g} scaled to 1")
        if self.trailing_edge == "close" and not airfoil.sharp:
            notes.append(f"trailing-edge gap {airfoil.trailing_edge_gap:.3g} closed")
            airfoil = airfoil.closed()
        return airfoil, notes

    def resolve(self, base: Path, name: str | None = None) -> Path:
        """A file, relative to ``base`` (the case directory) or the working directory."""
        name = name or self.file
        for root in (base, Path.cwd()):
            path = Path(name) if Path(name).is_absolute() else root / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"{name!r} not found in {base} or {Path.cwd()}")


def _generated(spec: GridSpec, base: Path, dtype) -> Mesh:
    airfoil, _ = spec.airfoil_shape(base)
    spacing = Spacing(spec.leading_edge_spacing, spec.trailing_edge_spacing)
    common = dict(cells_normal=spec.cells_normal, wall_spacing=spec.wall_spacing,
                  far_field=spec.far_field, spacing=spacing)
    if spec.grid == "o_grid":
        x, y = o_grid(airfoil, spec.cells_airfoil, **common)
    else:
        x, y = c_grid(airfoil, spec.cells_airfoil, spec.cells_wake, **common)
    mesh, _ = assemble([RawBlock(x, y)], dtype=dtype, source=spec.describe())
    return mesh


def _file(spec: GridSpec, base: Path, dtype) -> Mesh:
    path = spec.resolve(base)
    if spec.format in ("auto", "plot3d") and is_3d_plot3d(path):
        mesh, _ = load_grid_3d(path, spec.coordinate_scale, spec.rules(), dtype)
        return mesh
    mesh, _ = load_grid(path, spec.format, spec.coordinate_scale, spec.rules(), dtype)
    return mesh


GRIDS: dict[str, Callable[[GridSpec, Path, object], Mesh]] = {
    "o_grid": _generated, "c_grid": _generated, "file": _file,
}
"""Grid sources by ``domain.grid``."""

PATCH_RULE_KEYS = {"name", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
