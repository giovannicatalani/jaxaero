#!/usr/bin/env python3
"""Build the validation gallery: figures, forces against references, run times.

    python validation/report.py [--results results] [--out validation/gallery]

Every validation case whose ``reference.json`` has a ``gallery`` entry is drawn
from its finest result in ``<results>/validation/<case>_n<N>/`` (written by
validate.py), with the published distributions its ``surface_reference``
names; the polar study is drawn from ``<results>/polar/``. Writes
``<out>/README.md`` and its figures. Runs nothing; a case without a result is
listed as missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from jaxaero.plot import (SERIES, SURFACE, TEXT_SECONDARY, WINDOW, _figure, _save,  # noqa: E402
                          load_polar, load_result, plot_convergence, plot_cp_family, plot_field,
                          plot_polar, plot_surface)
from jaxaero.simulation import cl_spread  # noqa: E402
from compare_surface import critical_cp, joined, reference_set, shock_position, upper_lower  # noqa: E402
from validate import CASES, load_validation  # noqa: E402

SUITES = ("ladder", "rans", "transonic", "laminar", "topology")
"""Page order."""


def finest(results: Path, name: str) -> Path | None:
    runs = sorted(results.glob(f"{name}_n*"), key=lambda d: int(d.name.rsplit("_n", 1)[1]))
    return runs[-1] if runs else None


def curves(spec: str | None) -> dict[str, dict]:
    """Comparison points per plotted quantity from a ``surface_reference``."""
    out: dict[str, dict] = {"cp": {}, "cf": {}}
    for label, ref in (reference_set(spec) if spec else {}).items():
        out["cp"][label] = joined(ref, "Cp upper", "Cp lower")
        if "Cf upper" in ref:
            out["cf"][f"{label}, upper surface"] = ref["Cf upper"]
    return out


def _pct(value: float, ref: float | None) -> str:
    return "—" if ref is None else (f"{abs(value):.1e} abs" if ref == 0 else f"{100 * (value / ref - 1):+.2f}%")


def case_section(v, directory: Path, figures: Path) -> tuple[list[str], dict]:
    res, g = load_result(directory), v.gallery
    refs = curves(g.get("surface_reference"))
    stem, images = directory.name, []

    def draw(name, fn, *args):
        fn(res, *args, figures / f"{stem}_{name}.png")
        images.append(f"{stem}_{name}.png")

    if g.get("fields", True):
        window = tuple(g.get("window", WINDOW))
        draw("mach", plot_field, "mach", window)
        draw("cp_field", plot_field, "cp", window)
        draw("cp", plot_surface, "cp", refs["cp"])
    if "cf" in res.surface:
        draw("cf", plot_surface, "cf", refs["cf"])
    draw("convergence", plot_convergence)

    e, ref = res.extra, v.reference
    case_doc = res.metadata["extra"]["case_setup"]["case"]
    spread = cl_spread(res.cl_history)
    lines = [f"## {case_doc['general']['description']}", "",
             f"`{v.name}`, {e['grid']}, {e['cells']:,} cells. Isolates: {ref.isolates}.", "",
             "| | CL | CD | CD friction | CM |", "|---|---|---|---|---|",
             f"| this code | {e['cl']:.5f} | {e['cd']:.6f} | {e['cd_friction']:.6f} | {e['cm']:.4f} |",
             f"| reference | {'—' if ref.cl is None else f'{ref.cl:.5f}'} | {ref.cd:.6f} | | |",
             f"| difference | {_pct(e['cl'], ref.cl)} | {_pct(e['cd'], ref.cd)} | | |", ""]
    upper = upper_lower(res.surface["x"], res.surface["cp"])[0]
    shock = shock_position(upper, critical_cp(res.flow["mach"]))
    if not np.isnan(shock):
        others = {k: shock_position(c["Cp upper"], critical_cp(res.flow["mach"]))
                  for k, c in (reference_set(g["surface_reference"]).items() if g.get("surface_reference") else [])}
        lines += [f"Shock (upper-surface Cp rising through Cp*): this code {shock:.4f} c"
                  + "".join(f", {k} {s:.4f} c" for k, s in others.items()) + ".", ""]
    lines += [f"Residual fell {e['orders_dropped']:.1f} orders in {e['iterations']:,} iterations; CL moved "
              f"{spread:.1e} over the last quarter ({'settled' if spread <= v.settled_cl_spread else '**not settled**'}). "
              f"Reference: {ref.notes}", ""]
    lines += [f"![{name}](figures/{name})" for name in images] + [""]
    return lines, {"case": v.name, "cells": e["cells"], "solver": res.metadata["config"]["numerics"]["solver"],
                   "iterations": e["iterations"], "orders": e["orders_dropped"], **e.get("timing", {})}


def runtime_table(rows: list[dict]) -> list[str]:
    out = ["| case | cells | solver | iterations | orders | total | compile | per iteration |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| `{r['case']}` | {r['cells']:,} | {r['solver']} | {r['iterations']:,} | {r['orders']:.1f} | "
                   f"{r.get('total_s', float('nan')):.0f} s | {r.get('compile_s', float('nan')):.0f} s | "
                   f"{1e3 * r.get('per_iteration_s', float('nan')):.1f} ms |")
    return out


def plot_scaling(bench: dict, out: Path) -> None:
    """Throughput against batch size, one line per device count."""
    fig, ax = _figure(figsize=(7.0, 4.4))
    for slot, n in enumerate(sorted({r["devices"] for r in bench["rows"]})):
        rows = sorted((r for r in bench["rows"] if r["devices"] == n), key=lambda r: r["cases"])
        x, y = [r["cases"] for r in rows], [r["case_iterations_per_s"] for r in rows]
        ax.plot(x, y, "-o", color=SERIES[slot], markeredgecolor=SURFACE, label=f"{n} GPU{'s' * (n > 1)}")
        ax.annotate(f"{rows[-1]['speedup']:.0f}x", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TEXT_SECONDARY)
    ax.set(xscale="log", yscale="log", xlabel="angles advanced together", ylabel="angle-iterations per second",
           title=f"Batched Newton-Krylov throughput, {bench['case']} ({bench['cells']:,} cells)")
    ax.legend(loc="upper left")
    _save(fig, out)


def polar_section(polar: Path, bench_path: Path, figures: Path) -> list[str]:
    table, meta = load_polar(polar)
    plot_polar(polar, figures / "polar.png")
    plot_cp_family(polar, out=figures / "polar_cp_family.png")
    t, env = meta["timing"], meta["environment"]
    lines = ["## Transonic polar: RAE 2822, M 0.729, Re 6.5e6", "",
             f"`{meta['case']}` at {table.shape[0]} angles, all advanced together (`python -m jaxaero.polar`) "
             f"on {t['devices']} x {env.get('jax_device_kind', 'GPU')}: **{t['total_s'] / 60:.1f} min**, "
             f"compile {t['compile_s']:.0f} s, {1e3 * t['per_iteration_s']:.0f} ms per iteration of the batch. "
             "Hollow points did not settle (CL still moving over the last quarter of the run).", "",
             "![polar](figures/polar.png)", "", "![Cp family](figures/polar_cp_family.png)", "",
             "| α | CL | CD | CM | orders | CL spread |", "|---|---|---|---|---|---|"]
    lines += [f"| {a:+.1f} | {cl:.5f} | {cd:.6f} | {cm:.4f} | {o:.1f} | {s:.1e}{'' if ok else ' (not settled)'} |"
              for a, cl, cd, _, cm, o, _, s, ok in table]
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text())
        plot_scaling(bench, figures / "polar_scaling.png")
        lines += ["", f"Throughput (`scripts/benchmark.py --batches`): {bench['iterations']} identical Newton "
                  "steps per configuration, steady rate; speed-up against one angle on one GPU.", "",
                  "![scaling](figures/polar_scaling.png)", "",
                  "| GPUs | angles | ms per batch iteration | angle-iterations / s | speed-up |", "|---|---|---|---|---|"]
        lines += [f"| {r['devices']} | {r['cases']} | {1e3 * r['per_iteration_s']:.1f} | "
                  f"{r['case_iterations_per_s']:.1f} | {r['speedup']:.1f}x |" for r in bench["rows"]]
    return lines + [""]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=HERE / "gallery")
    p.add_argument("--polar", default="polar/rae2822_a100x4", help="polar directory under --results")
    p.add_argument("--benchmark", default="polar/benchmark_a100.json", help="benchmark JSON under --results")
    args = p.parse_args(argv)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    cases = [load_validation(d.name) for d in sorted(CASES.iterdir()) if (d / "reference.json").is_file()]
    cases = sorted((v for v in cases if v.gallery is not None), key=lambda v: (SUITES.index(v.suite), v.order))
    sections, rows, missing = [], [], []
    for v in cases:
        directory = finest(args.results / "validation", v.name)
        if directory is None:
            missing.append(v.name)
            continue
        lines, row = case_section(v, directory, figures)
        sections += lines
        rows.append(row)
    polar = args.results / args.polar
    if (polar / "polar.dat").is_file():
        sections += polar_section(polar, args.results / args.benchmark, figures)

    commits = sorted({str(json.loads(next((args.results / "validation").glob(f"{r['case']}_n*/*_metadata.json"))
                                      .read_text())["git"]["commit"])[:10] for r in rows})
    page = ["# Validation gallery", "",
            f"Generated by `validation/report.py` from runs on one NVIDIA A100 each, code at "
            f"{', '.join(f'`{c}`' for c in commits)}. Mach on a one-hue scale, Cp diverging about Cp = 0, Cp "
            "axes inverted. Compile is the first block minus its share at the steady rate.", "",
            *runtime_table(rows), ""]
    page += [f"Missing (no result yet): {', '.join(missing)}.", ""] if missing else []
    (args.out / "README.md").write_text("\n".join(page + sections) + "\n")
    print(args.out / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
