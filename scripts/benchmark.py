#!/usr/bin/env python3
"""Time the pieces of one implicit Newton step of a case, on this device.

The residual, one Jacobian-vector product, the preconditioner (set-up and one
application) and a whole Newton step, each compiled first and then timed with
``block_until_ready``; in milliseconds and nanoseconds per cell, the number
that compares devices and grid sizes. ``JAX_PLATFORMS=cpu`` forces the CPU.

    python scripts/benchmark.py examples/naca0012_sa
    python scripts/benchmark.py validation/cases/tmr_naca0012_sa_a10 --set case.domain.file=data/grids/n0012_897-257.p2dfmt.gz

With ``--batches`` it measures instead the throughput of a batched polar
(``jaxaero.polar``) for each batch size on each device count: the same
``--iterations`` Newton steps every time (the residual tolerance is made
unreachable), steady rate, compilation excluded, written as JSON for
validation/report.py:

    python scripts/benchmark.py validation/cases/nparc_rae2822_case6 --batches 1,4,16,64 \
        --devices 1,4 --out results/polar/benchmark_a100.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxaero import load_case
from jaxaero.numerics.residual import make_residual
from jaxaero.solvers import implicit
from jaxaero.solvers.explicit import initial_state
from jaxaero.solvers.preconditioners import get_preconditioner


def timed(fn, *args, repeats: int) -> float:
    """Seconds per call after one compiling call."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / repeats


def batch_throughput(args) -> int:
    from jaxaero.io.output import environment_record, git_revision
    from jaxaero.polar import run_polar

    overrides = (*args.overrides, f"numerical.convergence.max_iters={args.iterations}",
                 "numerical.convergence.tol=1e-300")
    case = load_case(args.case, overrides)
    grid = case.grid()
    rows = []
    for devices in map(int, args.devices.split(",")):
        for cases in (b for b in map(int, args.batches.split(",")) if b >= devices):
            timing = run_polar(case, np.linspace(-2.0, 5.5, cases), grid=grid, devices=devices).timing
            rows.append(timing.record())
            print(f"{devices} device(s), {cases:3d} angles: {1e3 * timing.per_iteration:8.1f} ms per iteration, "
                  f"{timing.case_iterations_per_second:7.1f} angle-iterations/s")
    for row in rows:
        row["speedup"] = row["case_iterations_per_s"] / rows[0]["case_iterations_per_s"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"case": case.name, "cells": grid.cells, "iterations": args.iterations,
                                    "rows": rows, "git": git_revision(), "environment": environment_record()},
                                   indent=2, default=float))
    print(f"written to {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--batches", help="batched-polar throughput for these batch sizes, e.g. 1,4,16,64")
    p.add_argument("--devices", default="1", help="with --batches: device counts, e.g. 1,4")
    p.add_argument("--iterations", type=int, default=100, help="with --batches: Newton steps per configuration")
    p.add_argument("--out", type=Path, default=Path("results/polar/benchmark.json"))
    args = p.parse_args(argv)
    if args.batches:
        return batch_throughput(args)

    case = load_case(args.case, tuple(args.overrides))
    grid = case.grid()
    op = make_residual(grid, case.flow, case.numerics)
    jop = make_residual(grid, case.flow, case.numerics, first_order=True)
    g, cells = case.flow.gamma, grid.cells
    U = initial_state(op).U
    U = U * (1.0 + 1e-3 * jnp.sin(jnp.arange(U.size).reshape(U.shape)))  # not a fixed point

    _, linear = jax.linearize(jop.residual, U)
    li, lj = op.wave_speeds(U)
    prepare = get_preconditioner(case.implicit.preconditioner)(op.atlas, g, op.viscosity, op.model, op.distance)
    cfl = jnp.asarray(100.0)
    step = jax.jit(implicit.make_newton_step(op, case.implicit, jop))
    state = implicit.ImplicitState(U=U, cfl=cfl, residual_norm=jnp.asarray(jnp.inf))
    t = {
        "residual": timed(jax.jit(op.residual), U, repeats=args.repeats),
        "jvp": timed(jax.jit(linear), U, repeats=args.repeats),
        "precond setup": timed(jax.jit(lambda u: prepare(u, li, lj, cfl)(u)), U, repeats=args.repeats),
        "precond apply": timed(jax.jit(prepare(U, li, lj, cfl)), U, repeats=args.repeats),
        "newton step": timed(lambda s: step(s)[0].U, state, repeats=max(2, args.repeats // 3)),
    }
    device = jax.devices()[0]
    print(f"{case.name}: {cells} cells on {device.device_kind} "
          f"({os.cpu_count()} CPUs visible), jax {jax.__version__}")
    for name, seconds in t.items():
        print(f"  {name:14s} {1e3 * seconds:9.2f} ms  {1e9 * seconds / cells:10.1f} ns/cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
