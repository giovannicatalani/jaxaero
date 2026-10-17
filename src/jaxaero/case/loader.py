"""A case: a directory holding ``case.json`` (what is solved) and
``numerical.json`` (how), parsed into the frozen configs the solver uses, so
nothing below this layer knows about JSON. Keys, types, defaults and choices
come from ``schema.py``; the rules that tie keys together are here.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import constants
from ..core.config import FlowConditions, ImplicitConfig, NumericsConfig
from ..mesh.atlas import Atlas, build_atlas
from ..mesh.coarsen import coarsen as coarsen_mesh
from ..mesh.mesh import assign_conditions, patches
from ..mesh.quality import quality
from . import schema
from ..geometry.elements import ROLES
from ..mesh.highlift import family
from .domain import ELEMENT_KEYS, PATCH_RULE_KEYS, GridSpec

CASE_FILE, NUMERICAL_FILE = "case.json", "numerical.json"


class CaseError(ValueError):
    """A case that cannot be run as written."""


@dataclass(frozen=True)
class Case:
    """One case, resolved. ``conditions``: those the case names (patch -> kind);
    ``raw``: the two documents after overrides, saved with every result."""

    name: str
    description: str
    output: Path
    grid_spec: GridSpec
    conditions: dict[str, str]
    flow: FlowConditions
    numerics: NumericsConfig
    implicit: ImplicitConfig
    rk_scheme: str
    block_iters: int
    directory: Path
    raw: dict[str, Any] = field(compare=False, repr=False)
    sequencing: int = 0

    def grid(self, coarsen: int = 0) -> Atlas:
        """The grid, every patch given its condition, packed for the solver;
        ``coarsen`` times with every other node (grid sequencing)."""
        mesh = self.grid_spec.build(self.directory, self.numerics.dtype)
        kinds = resolve_conditions(self.conditions, patches(mesh), self.flow)
        mesh = assign_conditions(mesh, kinds)
        for _ in range(coarsen):
            mesh = coarsen_mesh(mesh)
        report = quality(mesh)
        if report.folded:
            raise CaseError(f"the grid has folded cells: {report.summary()}")
        return build_atlas(mesh, self.numerics.n_ghost)


# --- Parsing ------------------------------------------------------------------


def _section(doc: dict, file: str, name: str, required: bool = False) -> dict:
    """``doc[name]`` with unknown keys refused and ``notes`` dropped."""
    if name not in doc:
        if required:
            raise CaseError(f"missing section '{name}'")
        return {}
    section = doc[name]
    if not isinstance(section, dict):
        raise CaseError(f"'{name}' must be an object, got {type(section).__name__}")
    allowed = schema.keys(file, name)
    unknown = set(section) - set(allowed) - {"notes"}
    if unknown:
        raise CaseError(f"unknown key(s) {sorted(f'{name}.{k}' for k in unknown)}; "
                        f"allowed: {sorted(allowed)}")
    return {k: v for k, v in section.items() if k != "notes"}


def _values(section: dict, file: str, name: str) -> dict:
    """Every key of the section: checked, or its default (``None`` where it has none)."""
    out = {}
    for key in schema.FILES[file][name]:
        path = f"{name}.{key.name}"
        if key.name in section:
            try:
                key.check(path, section[key.name])
            except (TypeError, ValueError) as error:
                raise CaseError(str(error)) from None
            out[key.name] = section[key.name]
        elif key.default is schema.REQUIRED:
            raise CaseError(f"missing '{path}'")
        else:
            out[key.name] = key.default
    return out


def _top_level(doc: dict, file: str) -> None:
    unknown = set(doc) - set(schema.FILES[file]) - ({"boundary_conditions"} if file == CASE_FILE else set())
    if unknown:
        raise CaseError(f"{file}: unknown section(s) {sorted(unknown)}; allowed: {sorted(schema.FILES[file])}")


def _grid_spec(case_doc: dict) -> GridSpec:
    raw = _section(case_doc, CASE_FILE, "domain", required=True)
    if "grid" not in raw:
        raise CaseError(f"missing 'domain.grid'; choose from {sorted(schema.keys(CASE_FILE, 'domain')['grid'].choices)}")
    domain = _values(raw, CASE_FILE, "domain")
    kind = domain["grid"]
    keys = schema.keys(CASE_FILE, "domain")
    foreign = sorted(k for k in raw if keys[k].grids is not None and kind not in keys[k].grids)
    if foreign:
        takes = sorted(k for k, key in keys.items() if key.grids and kind in key.grids)
        raise CaseError(f"'domain.{foreign[0]}' does not apply to grid {kind!r}; it takes {takes}")
    for name, key in keys.items():
        if kind in key.required_for and domain[name] is None:
            raise CaseError(f"'domain.{name}' is required for grid {kind!r}")
    rules = domain["patches"]
    if not all(isinstance(r, dict) for r in rules):
        raise CaseError("'domain.patches' must be a list of objects")
    for k, rule in enumerate(rules):
        unknown = set(rule) - PATCH_RULE_KEYS
        if unknown or "name" not in rule:
            raise CaseError(f"'domain.patches[{k}]' needs a 'name' and takes {sorted(PATCH_RULE_KEYS)}; "
                            f"got {sorted(rule)}")
    elements = domain["elements"]
    for k, element in enumerate(elements):
        if not isinstance(element, dict) or element.get("role") not in ROLES or set(element) - ELEMENT_KEYS:
            raise CaseError(f"'domain.elements[{k}]' needs a 'role' (one of {list(ROLES)}) and takes "
                            f"{sorted(ELEMENT_KEYS)}; got {element!r}")
        for key in ("deflection_deg", "gap", "overlap"):
            if key in element and (not isinstance(element[key], (int, float)) or isinstance(element[key], bool)):
                raise CaseError(f"'domain.elements[{k}].{key}' must be a number, got {element[key]!r}")
        if _moved(element) and element["role"] == "main":
            raise CaseError(f"'domain.elements[{k}]': the main element is not re-placed; place the others")
        if kind == "multi_element" and "file" not in element and domain["file"] is None:
            raise CaseError(f"'domain.elements[{k}]' needs a 'file', or 'domain.file' must hold every element")
    if kind == "multi_element":
        try:
            family([e["role"] for e in elements])
        except ValueError as error:
            raise CaseError(f"'domain.elements': {error}") from None
    return GridSpec(**{**{k: v for k, v in domain.items() if v is not None}, "patches": tuple(rules),
                       "elements": tuple(elements)})


def _moved(element: dict) -> bool:
    return any(k in element for k in ("deflection_deg", "gap", "overlap"))


def _flow(case_doc: dict) -> FlowConditions:
    given = {name: _section(case_doc, CASE_FILE, name, required=name == "freestream")
             for name in ("freestream", "material_properties", "turbulence")}
    freestream, material, turbulence = (_values(given[name], CASE_FILE, name) for name in given)
    inviscid = freestream["reynolds"] is None

    sutherland_ratio = None
    if inviscid:
        for key in ("dynamic_viscosity", "prandtl_number", "sutherland_temperature"):
            if key in given["material_properties"]:
                raise CaseError(f"'material_properties.{key}' given for an inviscid case "
                                f"(freestream.reynolds is null)")
    else:
        if material["dynamic_viscosity"] == "sutherland":
            if freestream["temperature"] is None or material["sutherland_temperature"] is None:
                raise CaseError("Sutherland's law needs 'freestream.temperature' and "
                                "'material_properties.sutherland_temperature', in one absolute unit")
            sutherland_ratio = material["sutherland_temperature"] / freestream["temperature"]
        elif material["sutherland_temperature"] is not None:
            raise CaseError("'material_properties.sutherland_temperature' given with a constant viscosity")
        if material["prandtl_number"] != constants.PRANDTL_LAMINAR:
            raise CaseError(f"'material_properties.prandtl_number' = {material['prandtl_number']} is not "
                            f"implemented; only {constants.PRANDTL_LAMINAR}")

    model = turbulence["model"]
    if model != "none" and inviscid:
        raise CaseError("'turbulence.model' needs a viscous case (freestream.reynolds)")
    if model == "none" and "freestream_nu_tilde_ratio" in given["turbulence"]:
        raise CaseError("'turbulence.freestream_nu_tilde_ratio' given without a model")

    return FlowConditions(
        mach=freestream["mach"],
        alpha_deg=freestream["alpha_deg"],
        reynolds=freestream["reynolds"],
        gamma=material["specific_heat_ratio"],
        sutherland_ratio=sutherland_ratio,
        turbulence=None if model == "none" else model,
        nu_tilde_ratio=turbulence["freestream_nu_tilde_ratio"],
    )


def default_conditions(flow: FlowConditions) -> dict[str, str]:
    """The condition of a patch the case does not name, by the patch's name."""
    return {"wall": "slip_wall" if flow.inviscid else "no_slip_wall", "farfield": "far_field",
            "symmetry": "symmetry", "inflow": "inflow_total", "outflow": "outflow_static"}


def _boundaries(case_doc: dict, flow: FlowConditions) -> tuple[dict[str, str], str]:
    """``boundary_conditions``: per patch name, ``{"type": ..., "model": ...}``.

    Returns the conditions the case names (patch -> kind) and the far-field
    model's ``FAR_FIELDS`` key (one model for all far-field patches).
    """
    section = case_doc.get("boundary_conditions", {})
    if not isinstance(section, dict):
        raise CaseError("'boundary_conditions' must be an object")
    kinds, models = {}, set()
    for patch, entry in section.items():
        if patch == "notes":
            continue
        where = f"boundary_conditions.{patch}"
        if not isinstance(entry, dict):
            raise CaseError(f"'{where}' must be an object with a 'type', got {entry!r}")
        allowed = {k.name: k for k in schema.BOUNDARY_KEYS}
        unknown = set(entry) - set(allowed) - {"notes"}
        if unknown:
            raise CaseError(f"unknown key(s) {sorted(f'{where}.{k}' for k in unknown)}; allowed: {sorted(allowed)}")
        if "type" not in entry:
            raise CaseError(f"missing '{where}.type'")
        for name in set(entry) - {"notes"}:
            try:
                allowed[name].check(f"{where}.{name}", entry[name])
            except (TypeError, ValueError) as error:
                raise CaseError(str(error)) from None
        if "model" in entry and entry["type"] != "far_field":
            raise CaseError(f"'{where}.model' applies to a far_field patch only")
        if entry["type"] == "far_field":
            models.add(entry.get("model", "freestream"))
        if entry["type"] == "no_slip_wall" and flow.inviscid:
            raise CaseError(f"'{where}.type' = 'no_slip_wall' in an inviscid case (freestream.reynolds is null)")
        kinds[patch] = entry["type"]
    if len(models) > 1:
        raise CaseError(f"far-field patches name different models {sorted(models)}; one model per case")
    return kinds, schema.FAR_FIELD_MODELS[models.pop() if models else "freestream"]


def resolve_conditions(named: dict[str, str], grid_patches: list[str], flow: FlowConditions) -> dict[str, str]:
    """Every grid patch's condition: the case's, else the default for its name;
    ``CaseError`` for a patch the grid lacks or one with neither."""
    unknown = [p for p in named if p not in grid_patches]
    if unknown:
        raise CaseError(f"'boundary_conditions' names patch(es) {unknown} the grid does not have; "
                        f"its patches are {grid_patches}")
    defaults = default_conditions(flow)
    missing = [p for p in grid_patches if p not in named and p not in defaults]
    if missing:
        raise CaseError(f"no boundary condition for patch(es) {missing}: give each a 'type' in "
                        f"'boundary_conditions' (the grid's patches are {grid_patches})")
    return {p: named.get(p, defaults.get(p)) for p in grid_patches}


def parse_case(case_doc: dict, numerical_doc: dict, directory: Path = Path(".")) -> Case:
    """A ``Case`` from the two documents; ``CaseError`` names the offending key
    (value ranges: ``ValueError`` from the configs)."""
    _top_level(case_doc, CASE_FILE)
    _top_level(numerical_doc, NUMERICAL_FILE)
    general = _values(_section(case_doc, CASE_FILE, "general", required=True), CASE_FILE, "general")
    spec = _grid_spec(case_doc)
    flow = _flow(case_doc)
    conditions, far_field = _boundaries(case_doc, flow)

    fluxes, stepping, implicit, convergence = (
        _values(_section(numerical_doc, NUMERICAL_FILE, name), NUMERICAL_FILE, name)
        for name in ("convective_fluxes", "time_integration", "implicit", "convergence"))
    implicit = ImplicitConfig(**implicit)
    numerics = NumericsConfig(
        flux=fluxes["riemann_solver"],
        limiter=fluxes["limiter"],
        kappa=fluxes["muscl_kappa"],
        cfl=stepping["cfl"],
        max_iters=convergence["max_iters"],
        tol=convergence["tol"],
        forces_tol=convergence["forces_tol"],
        solver=stepping["solver"],
        far_field=far_field,
    )
    rk_scheme = stepping["rk_scheme"]
    block_iters = convergence["block_iters"] or (200 if numerics.solver == "explicit" else 20)
    if block_iters < 1:
        raise CaseError(f"'convergence.block_iters' must be >= 1, got {block_iters}")
    if convergence["grid_sequencing"] < 0:
        raise CaseError(f"'convergence.grid_sequencing' must be >= 0, got {convergence['grid_sequencing']}")

    directory = Path(directory)
    output = Path(general["save_path"])
    return Case(
        name=general["case_name"],
        description=general["description"] or general["case_name"],
        output=output if output.is_absolute() else directory / output,
        grid_spec=spec,
        conditions=conditions,
        flow=flow,
        numerics=numerics,
        implicit=implicit,
        rk_scheme=rk_scheme,
        block_iters=block_iters,
        directory=directory,
        sequencing=convergence["grid_sequencing"],
        raw={"case": case_doc, "numerical": numerical_doc},
    )


# --- Loading and overrides ---------------------------------------------------------


def apply_override(docs: dict[str, dict], assignment: str) -> None:
    """Apply ``file.section.key=value`` to the raw documents, in place.

    ``file`` is ``case`` or ``numerical``; ``value`` is read as JSON, falling
    back to a bare string, so ``limiter=none`` and ``reynolds=null`` both work.
    """
    if "=" not in assignment:
        raise CaseError(f"override {assignment!r} is not of the form file.section.key=value")
    path, text = assignment.split("=", 1)
    parts = path.strip().split(".")
    if len(parts) != 3 or parts[0] not in docs:
        raise CaseError(f"override path {path!r} must be {sorted(docs)[0]}|{sorted(docs)[1]}"
                        f".section.key, e.g. numerical.convective_fluxes.limiter")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    docs[parts[0]].setdefault(parts[1], {})[parts[2]] = value


def load_case(directory: str | Path, overrides: tuple[str, ...] = ()) -> Case:
    """Read a case directory, apply ``file.section.key=value`` overrides, parse.

    Raises:
        FileNotFoundError: no ``case.json`` there.
        CaseError: the files are not a runnable case; the message names the
            directory and the key.
    """
    directory = Path(directory)
    if not (directory / CASE_FILE).is_file():
        raise FileNotFoundError(f"{directory} holds no {CASE_FILE}")
    docs = {}
    for label, filename in (("case", CASE_FILE), ("numerical", NUMERICAL_FILE)):
        path = directory / filename
        try:
            docs[label] = json.loads(path.read_text()) if path.is_file() else {}
        except json.JSONDecodeError as error:
            raise CaseError(f"{path}: {error}") from None
    docs = copy.deepcopy(docs)
    for assignment in overrides:
        apply_override(docs, assignment)
    try:
        return parse_case(docs["case"], docs["numerical"], directory)
    except CaseError as error:
        raise CaseError(f"{directory}: {error}") from None
