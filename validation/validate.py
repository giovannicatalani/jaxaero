#!/usr/bin/env python3
"""Run validation cases over their grid levels and judge them against references.

A validation case is a case directory (``case.json`` + ``numerical.json``, run
by the solver as any other) plus a ``reference.json``, read only here:

    suite, order          the suite it belongs to, and its place in it
    levels                grid levels: each a set of ``domain`` keys merged
                          into case.json, coarsest first
    reference             sourced CL / CD (and optionally Cf at one station),
                          the spread between sources, tolerances, citations
    settled_cl_spread     largest CL range over the last quarter of blocks
                          that counts as settled
    gallery               how report.py draws the case: plot "window", the
                          "surface_reference" set of compare_surface.py, "fields"

Examples:
    python validation/validate.py --suite ladder
    python validation/validate.py hiocfd_c12 --levels 0,1,2
    python validation/validate.py tmr_naca0012_sa_a10 --levels 449-129 --set numerical.convergence.max_iters=6000
    python validation/validate.py tmr_flatplate_sa --grids-only

Exits 1 if a judged case fails its criteria (expected at coarse levels).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
sys.path.insert(0, str(HERE.parent / "src"))

from jaxaero import CaseError, Result, load_case, run  # noqa: E402
from jaxaero.mesh.block import validate_metrics  # noqa: E402


# --- reference.json -------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """Sourced values and the criteria a run is judged by. Tolerances are
    relative to a non-zero reference and absolute for a zero one; ``cl: null``
    means none is quoted (the flat plate)."""

    cl: float | None
    cd: float
    cl_spread: float = 0.0
    cd_spread: float = 0.0
    cl_tol: float = 0.01
    cd_tol: float = 0.01
    cf_x: float | None = None
    cf: float | None = None
    cf_tol: float = 0.01
    isolates: str = ""
    sources: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Validation:
    name: str
    directory: Path
    suite: str
    order: int
    levels: tuple[dict, ...]
    reference: Reference | None
    settled_cl_spread: float = 1e-4
    gallery: dict | None = None
    notes: str = ""


def load_validation(name: str) -> Validation:
    directory = Path(name) if Path(name).is_dir() else CASES / name
    doc = json.loads((directory / "reference.json").read_text())
    allowed = {f.name for f in fields(Validation)} - {"name", "directory"}
    unknown = set(doc) - allowed
    if unknown:
        raise CaseError(f"{directory}/reference.json: unknown key(s) {sorted(unknown)}")
    ref = doc.get("reference")
    if ref is not None:
        unknown = set(ref) - {f.name for f in fields(Reference)}
        if unknown:
            raise CaseError(f"{directory}/reference.json: unknown reference key(s) {sorted(unknown)}")
        ref = Reference(**{**ref, "sources": tuple(ref.get("sources", ()))})
    return Validation(name=directory.name, directory=directory, suite=doc["suite"],
                      order=int(doc["order"]), levels=tuple(doc["levels"]), reference=ref,
                      settled_cl_spread=float(doc.get("settled_cl_spread", 1e-4)),
                      gallery=doc.get("gallery"), notes=doc.get("notes", ""))


def suite(name: str) -> list[Validation]:
    members = [load_validation(d.name) for d in sorted(CASES.iterdir()) if (d / "reference.json").is_file()]
    chosen = sorted((v for v in members if v.suite == name), key=lambda v: (v.order, v.name))
    if not chosen:
        raise KeyError(f"no validation case in suite {name!r}; suites: {sorted({v.suite for v in members})}")
    return chosen


def level_overrides(level: dict) -> tuple[str, ...]:
    return tuple(f"case.domain.{k}={json.dumps(v)}" for k, v in level.items())


def select_levels(levels: tuple[dict, ...], spec: str | None) -> list[dict]:
    """All levels, or those named by index or by a substring of a value."""
    if not spec:
        return list(levels)
    chosen = []
    for token in spec.split(","):
        if token.isdigit() and int(token) < len(levels):
            chosen.append(levels[int(token)])
            continue
        matches = [lv for lv in levels if any(token in str(v) for v in lv.values())]
        if len(matches) != 1:
            # Not a configured level: a grid file of that name next to them.
            matches = [{"file": str(p)} for p in sorted(Path("data/grids").glob(f"*{token}*.p2dfmt*"))]
        if len(matches) != 1:
            raise CaseError(f"level {token!r} matches {len(matches)} levels; use an index 0..{len(levels) - 1}")
        chosen.append(matches[0])
    return chosen


# --- Judging --------------------------------------------------------------------


@dataclass(frozen=True)
class Judged:
    result: Result
    reference: Reference | None
    settled_cl_spread: float
    cf_station: float = float("nan")

    @property
    def cl_spread(self) -> float:
        return self.result.cl_spread

    @property
    def settled(self) -> bool:
        return self.cl_spread <= self.settled_cl_spread

    def error(self, value: float, reference: float | None) -> float:
        if reference is None:
            return float("nan")
        return abs(value) if reference == 0.0 else abs(value / reference - 1.0)

    def failures(self) -> list[str]:
        ref, forces = self.reference, self.result.forces
        if ref is None:
            return []
        out = []
        if self.result.history.reason == "non_finite":
            out.append("non-finite state")
        if not self.settled:
            out.append(f"CL still moving by {self.cl_spread:.2e} (settled means < {self.settled_cl_spread:.0e})")
        if ref.cl is not None and self.error(forces.cl, ref.cl) > ref.cl_tol:
            out.append(f"CL {forces.cl:+.5f} vs {ref.cl:+.5f} ({100 * self.error(forces.cl, ref.cl):.2f}% > {100 * ref.cl_tol:.2f}%)")
        if self.error(forces.cd, ref.cd) > ref.cd_tol:
            out.append(f"CD {forces.cd:+.6f} vs {ref.cd:.6f} ({100 * self.error(forces.cd, ref.cd):.2f}% > {100 * ref.cd_tol:.2f}%)")
        if ref.cf is not None and not self.error(self.cf_station, ref.cf) <= ref.cf_tol:
            out.append(f"Cf({ref.cf_x}) {self.cf_station:.6e} vs {ref.cf:.6e}")
        return out


def cf_at(result: Result, x_station: float | None) -> float:
    """Cf at ``x_station`` on the first wall run where x increases through it
    (the plate; an airfoil's upper surface)."""
    if x_station is None or result.case.flow.inviscid:
        return float("nan")
    surface = result.surface()
    x, cf = surface["x"], surface["cf"]
    for k in np.flatnonzero(np.diff(x) > 0.0):
        if x[k] <= x_station <= x[k + 1]:
            w = (x_station - x[k]) / (x[k + 1] - x[k])
            return float((1.0 - w) * cf[k] + w * cf[k + 1])
    return float("nan")


@dataclass(frozen=True)
class RichardsonFit:
    """Observed order from three levels refined by ``r``: ``p = log(d1/d2) / log r``,
    NaN when the differences change sign (no asymptotic range)."""

    order: float
    extrapolated: float
    monotone: bool
    differences: tuple[float, float] = field(default=(float("nan"), float("nan")))


def richardson(values: tuple[float, float, float], refinement: float = 2.0) -> RichardsonFit:
    coarse, medium, fine = values
    d1, d2 = medium - coarse, fine - medium
    if d1 * d2 <= 0.0 or d2 == 0.0:
        return RichardsonFit(float("nan"), float("nan"), False, (d1, d2))
    order = float(np.log(abs(d1 / d2)) / np.log(refinement))
    return RichardsonFit(order, float(fine + d2 / (refinement**order - 1.0)), True, (d1, d2))


# --- Running --------------------------------------------------------------------


def validate(v: Validation, levels: list[dict], args) -> bool | None:
    ref = v.reference
    print(f"\n{'=' * 78}\n{v.name} ({v.suite} {v.order})" + (f" -- {ref.isolates}" if ref and ref.isolates else ""))
    judged: list[Judged] = []
    for level in levels:
        case = load_case(v.directory, level_overrides(level) + tuple(args.overrides))
        grid = case.grid()
        ok = all(validate_metrics(block).ok for block in grid.mesh)
        size = grid.mesh[0].ni if len(grid.mesh) == 1 else grid.cells      # the level's label
        print(f"  grid {case.grid_spec.describe()}: {grid.cells} cells, metrics {'ok' if ok else 'FAILED'}")
        if args.grids_only or not ok:
            continue
        result = run(case, grid)
        j = Judged(result, ref, v.settled_cl_spread, cf_at(result, ref and ref.cf_x))
        judged.append(j)
        f, h = result.forces, result.history
        cf = f"  Cf({ref.cf_x}) {j.cf_station:.5e}" if ref and ref.cf is not None else ""
        print(f"  {'ok' if not j.failures() else 'XX'} {size:4d}  CL {f.cl:+.5f}  CD {f.cd:+.6f} "
              f"(friction {f.cd_friction:.6f}){cf}  spread {j.cl_spread:.1e}  "
              f"{h.orders_dropped:.2f} orders  {h.n_iters} it  {result.seconds:.0f}s", flush=True)
        result.write(Path(args.out) / f"{v.name}_n{size}",
                     extra={"settled": j.settled, "cf_station": j.cf_station,
                            "reference": None if ref is None else ref.__dict__})
    if len(judged) >= 3:
        for label, values in (("CL", [j.result.forces.cl for j in judged]),
                              ("CD", [j.result.forces.cd for j in judged])):
            fit = richardson(tuple(values[-3:]))
            print(f"  {label}: " + (f"order {fit.order:.2f}, extrapolated {fit.extrapolated:+.6f}"
                                    if fit.monotone else "differences change sign: no asymptotic range"))
    if args.grids_only or ref is None or not judged:
        return None
    failures = judged[-1].failures()
    print(f"  {'PASS' if not failures else 'FAIL'}: {v.name}" + "".join(f"\n    - {x}" for x in failures))
    return not failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", nargs="*", help=f"names under {CASES.relative_to(HERE.parent)} or paths")
    p.add_argument("--suite", help="ladder | laminar | topology | rans | transonic | generated")
    p.add_argument("--levels", help="comma-separated indices or unique parts of a level (e.g. 449-129)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="FILE.SECTION.KEY=VALUE")
    p.add_argument("--grids-only", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("--out", default="results/validation")
    args = p.parse_args(argv)
    if bool(args.cases) == bool(args.suite):
        p.error("give case names or --suite")
    sys.stdout.reconfigure(line_buffering=True)
    try:
        validations = suite(args.suite) if args.suite else [load_validation(c) for c in args.cases]
        plan = [(v, select_levels(v.levels, args.levels)) for v in validations]
    except (CaseError, KeyError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    verdicts = {}
    for v, levels in plan:
        verdicts[v.name] = validate(v, levels, args)
        if verdicts[v.name] is False and args.suite and not args.continue_on_failure:
            print("\nstopping: later cases are uninterpretable while this one fails (--continue-on-failure)")
            break
    print("\nsummary: " + ", ".join(f"{k} {'PASS' if x else 'FAIL' if x is False else 'done'}" for k, x in verdicts.items()))
    return 1 if False in verdicts.values() else 0


if __name__ == "__main__":
    raise SystemExit(main())
