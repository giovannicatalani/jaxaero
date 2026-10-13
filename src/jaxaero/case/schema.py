"""Every key of ``case.json`` and ``numerical.json``, declared once.

The loader checks documents against this table (unknown keys, types,
choices, required keys, defaults), and ``docs/case_reference.md`` is
generated from it: ``python -m jaxaero.case --write``. Rules that tie
keys together (Sutherland's law needs a temperature, a turbulence model a
Reynolds number, ...) stay in the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import constants
from ..core.config import TURBULENCE_MODELS, ImplicitConfig, NumericsConfig
from ..mesh.assemble import FORMATS
from ..mesh.block import CONDITION_KINDS
from ..mesh.generate import Spacing
from ..numerics.limiters import LIMITERS
from ..numerics.riemann import FLUXES
from ..solvers.explicit import RK_SCHEMES
from ..solvers.preconditioners import PRECONDITIONERS
from .domain import GRIDS

REQUIRED = "required"
GENERATED = ("o_grid", "c_grid")


@dataclass(frozen=True)
class Key:
    name: str
    type: str                                 # number, integer, string, list; "|null" if null is allowed
    default: Any
    help: str
    choices: tuple[str, ...] | None = None
    grids: tuple[str, ...] | None = None      # domain keys: the grid sources that take it
    required_for: tuple[str, ...] = ()        # domain keys: required for these sources
    default_text: str | None = None           # the default as documented, when it depends on other keys

    def check(self, path: str, value: Any) -> None:
        """Raise ``TypeError``/``ValueError`` (the loader words them) if ``value`` is not of this key's type."""
        if value is None:
            if not self.type.endswith("|null"):
                raise TypeError(f"'{path}' may not be null")
            return
        base = self.type.split("|")[0]
        ok = {"number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "string": lambda v: isinstance(v, str),
              "list": lambda v: isinstance(v, list)}[base](value)
        if not ok:
            raise TypeError(f"'{path}' must be {'an' if base[0] in 'aeiou' else 'a'} {base}, got {value!r}")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"'{path}' = {value!r} is not implemented; choose from {sorted(self.choices)}")


def _sorted(registry) -> tuple[str, ...]:
    return tuple(sorted(registry))


_N, _I = NumericsConfig, ImplicitConfig

CASE: dict[str, tuple[Key, ...]] = {
    "general": (
        Key("case_name", "string", REQUIRED, "names the output directory"),
        Key("description", "string", None, "one line, printed in the log", default_text="`case_name`"),
        Key("save_path", "string", "results", "output root, relative to the case directory"),
    ),
    "domain": (
        Key("grid", "string", REQUIRED, "where the grid comes from: generated round one airfoil "
            "(`o_grid`, `c_grid`) or read from a file", choices=_sorted(GRIDS)),
        Key("airfoil", "string", None, "`naca` + 4 digits (closed trailing edge; `naca0012_blunt` for the "
            "standard open one), or a coordinate file (Selig, Lednicer or two columns), scaled to unit chord",
            grids=GENERATED, required_for=GENERATED),
        Key("cells_airfoil", "integer", None, "cells round the airfoil", grids=GENERATED, required_for=GENERATED),
        Key("cells_wake", "integer", None, "cells along each side of the wake cut",
            grids=("c_grid",), required_for=("c_grid",)),
        Key("cells_normal", "integer", None, "cells from the wall to the far field",
            grids=GENERATED, required_for=GENERATED),
        Key("wall_spacing", "number", None, "first-cell height, chords; for y+ ~ 1 see docs/own-airfoil.md",
            grids=GENERATED, required_for=GENERATED),
        Key("trailing_edge", "string", "as_given", "`close` closes an open trailing edge (needed for a C-grid); "
            "`as_given` keeps it, and an O-grid meshes its base",
            choices=("as_given", "close"), grids=GENERATED),
        Key("far_field", "number", 100.0, "distance of the outer boundary from the airfoil, chords",
            grids=GENERATED),
        Key("leading_edge_spacing", "number", Spacing.leading_edge, "surface cell length at the leading edge, "
            "chords", grids=GENERATED),
        Key("trailing_edge_spacing", "number", Spacing.trailing_edge, "surface cell length at the trailing "
            "edge, chords", grids=GENERATED),
        Key("file", "string", None, "the grid file: Plot3D (formatted or unformatted, either byte order, "
            "2D or 3D, single- or multi-block, iblank, gzip) or CGNS in HDF5 storage (structured zones; "
            "`pip install h5py`)", grids=("file",), required_for=("file",)),
        Key("format", "string", "auto", "`auto` reads it from the contents", choices=_sorted(FORMATS),
            grids=("file",)),
        Key("coordinate_scale", "number", 1.0, "factor on every coordinate, to the length the Reynolds "
            "number is per (1/18 for a grid in inches with an 18 in chord)", grids=("file",)),
        Key("patches", "list", [], "rules naming the boundary faces the file does not name, first match "
            "wins: `{\"name\": N, \"x_min\": .., \"x_max\": .., \"y_min\": .., \"y_max\": ..}` (3D grids: "
            "also `z_min`, `z_max`), every bound optional", grids=("file",)),
        Key("reference_length", "number", 1.0, "length the force coefficients are per (3D: the moment's)"),
        Key("reference_area", "number|null", None, "3D grids (required there): area the force "
            "coefficients are per; 2D coefficients are per unit span"),
        Key("moment_centre", "list", [0.25, 0.0, 0.0], "point the pitching moment is about, `[x, y]` or "
            "`[x, y, z]`; nose-up positive about the spanwise axis (z in 2D, y in 3D)"),
    ),
    "freestream": (
        Key("mach", "number", REQUIRED, "subsonic"),
        Key("alpha_deg", "number", 0.0, "incidence, degrees"),
        Key("reynolds", "number|null", None, "per unit length of the grid (per chord for a grid of unit "
            "chord); `null` is inviscid"),
        Key("temperature", "number", None, "absolute, in the unit of `sutherland_temperature`; "
            "Sutherland's law needs it"),
    ),
    "material_properties": (
        Key("specific_heat_ratio", "number", constants.GAMMA_AIR, ""),
        Key("dynamic_viscosity", "string", "constant", "viscous cases only", choices=("constant", "sutherland")),
        Key("sutherland_temperature", "number", None, "198.6 (R) or 110.3 (K)"),
        Key("prandtl_number", "number", constants.PRANDTL_LAMINAR, "the one implemented"),
    ),
    "turbulence": (
        Key("model", "string", "none", "`sa_noft2`: Spalart-Allmaras without ft2, with SA-neg",
            choices=("none", *TURBULENCE_MODELS)),
        Key("freestream_nu_tilde_ratio", "number", 3.0, "far-field nu~/nu (3 to 5)"),
    ),
}

NUMERICAL: dict[str, tuple[Key, ...]] = {
    "convective_fluxes": (
        Key("riemann_solver", "string", _N.flux, "", choices=_sorted(FLUXES)),
        Key("limiter", "string", _N.limiter, "`minmod` where a shock may form; `none` for smooth flow "
            "(minmod stalls viscous convergence); `van_albada` for a shock with a boundary layer",
            choices=_sorted(LIMITERS)),
        Key("muscl_kappa", "number", _N.kappa, "unlimited MUSCL family: 0 Fromm (the default), 1/3 "
            "third-order upwind-biased (CFL3D's default), -1 fully upwind; other than 0 only with "
            "limiter `none`"),
    ),
    "time_integration": (
        Key("solver", "string", _N.solver, "`explicit` (multistage Runge-Kutta, local time step) or "
            "`implicit` (Newton-Krylov; use it for viscous cases)", choices=("explicit", "implicit")),
        Key("rk_scheme", "string", "rk5_jameson", "explicit path", choices=_sorted(RK_SCHEMES)),
        Key("cfl", "number", _N.cfl, "explicit path"),
    ),
    "implicit": (
        Key("cfl", "number", _I.cfl, "starting CFL"),
        Key("cfl_max", "number", _I.cfl_max, "ceiling of the CFL ramp"),
        Key("cfl_growth", "number", _I.cfl_growth, "largest CFL growth per step"),
        Key("cfl_cutback", "number", _I.cfl_cutback, "CFL divisor after a rejected step"),
        Key("cfl_min", "number", _I.cfl_min, "CFL floor"),
        Key("preconditioner", "string", _I.preconditioner, "", choices=_sorted(PRECONDITIONERS)),
        Key("jacobian", "string", _I.jacobian, "of the Krylov operator: `first_order` (defect correction) "
            "or `exact`", choices=("exact", "first_order")),
        Key("krylov_tol", "number", _I.krylov_tol, "relative linear residual to reach"),
        Key("krylov_subspace", "integer", _I.krylov_subspace, "GMRES vectors before a restart"),
        Key("krylov_restarts", "integer", _I.krylov_restarts, "GMRES restarts"),
        Key("krylov_success", "number", _I.krylov_success, "linear residual reduction for the CFL to grow"),
        Key("max_relative_update", "number|null", _I.max_relative_update, "bound on a step's relative change "
            "of density, pressure and nu~ (0.2 for SA at incidence)"),
    ),
    "convergence": (
        Key("max_iters", "integer", _N.max_iters, "iteration ceiling"),
        Key("tol", "number", _N.tol, "residual drop from its running peak"),
        Key("block_iters", "integer", None, "iterations per device call; forces are monitored at this "
            "interval", default_text="200 explicit, 20 implicit"),
    ),
}

FILES = {"case.json": CASE, "numerical.json": NUMERICAL}

FAR_FIELD_MODELS = {"freestream": "characteristic", "point_vortex": "characteristic_point_vortex"}
"""A ``far_field`` patch's ``model`` -> its ``numerics.boundary.FAR_FIELDS`` key."""

BOUNDARY_KEYS = (
    Key("type", "string", REQUIRED, "the condition", choices=tuple(CONDITION_KINDS)),
    Key("model", "string", "freestream", "`far_field` only: `point_vortex` adds the compressible vortex "
        "of the current lift at the quarter chord of the longest body, for a far field nearer than ~100 "
        "chords; one model per case", choices=tuple(FAR_FIELD_MODELS)),
)

CONDITIONS_HELP = {
    "slip_wall": "inviscid wall (default `wall` when inviscid)",
    "no_slip_wall": "adiabatic no-slip wall (default `wall` when viscous)",
    "symmetry": "mirror plane",
    "far_field": "Riemann invariants against the freestream (default `farfield`)",
    "inflow_total": "subsonic inflow at the freestream total pressure and temperature, in its direction (default `inflow`)",
    "outflow_static": "subsonic outflow at the freestream static pressure (default `outflow`)",
}


def keys(file: str, section: str) -> dict[str, Key]:
    return {k.name: k for k in FILES[file][section]}


# --- The generated reference ------------------------------------------------------------


def _value(v: Any) -> str:
    if v is REQUIRED:
        return "required"
    if v is None:
        return "`null`"
    if isinstance(v, str):
        return f"`{v}`"
    return f"`{v!r}`"


def _row(k: Key, with_grids: bool) -> str:
    default = k.default_text or ("required" if k.required_for and k.default is None else
                                 "" if k.default is None and k.type != "number|null" else _value(k.default))
    help_ = k.help
    if k.choices:
        help_ = (help_ + "; " if help_ else "") + "one of " + ", ".join(f"`{c}`" for c in k.choices)
    cells = [f"`{k.name}`", k.type.replace("|", " or "), default]
    if with_grids:
        cells.append(", ".join(f"`{g}`" for g in k.grids) if k.grids else "all")
    return "| " + " | ".join(cells + [help_]) + " |"


def reference() -> str:
    """``docs/case_reference.md``."""
    out = ["# Case reference",
           "",
           "Generated from `src/jaxaero/case/schema.py` (`python -m jaxaero.case --write`); "
           "do not edit. How to write a case: [the guide](guide.md).",
           "",
           "Every section may carry a `\"notes\"` string. Unknown keys are errors, a missing key takes its "
           "default, and an option that is not implemented is an error listing the alternatives.",
           ""]
    for file, sections in FILES.items():
        out += [f"## `{file}`", ""]
        for section, ks in sections.items():
            with_grids = section == "domain"
            head = ["key", "type", "default"] + (["grids"] if with_grids else []) + ["meaning"]
            out += [f"### `{section}`", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            out += [_row(k, with_grids) for k in ks]
            out.append("")
            if file == "case.json" and section == "domain":
                out += _boundary_section()
    return "\n".join(out).rstrip() + "\n"


def _boundary_section() -> list[str]:
    out = ["### `boundary_conditions`", "",
           "One entry per patch of the grid, `\"<patch>\": {\"type\": ..., \"model\": ...}`; a patch left out "
           "takes the default for its name (below).", "",
           "| key | type | default | meaning |", "|---|---|---|---|"]
    out += ["| " + " | ".join([f"`{k.name}`", k.type, _value(k.default),
                               k.help + "; one of " + ", ".join(f"`{c}`" for c in k.choices)]) + " |"
            for k in BOUNDARY_KEYS]
    out += ["", "| type | condition |", "|---|---|"]
    out += [f"| `{kind}` | {CONDITIONS_HELP[kind]} |" for kind in CONDITION_KINDS]
    return out + [""]


REFERENCE = Path(__file__).resolve().parents[3] / "docs" / "case_reference.md"
