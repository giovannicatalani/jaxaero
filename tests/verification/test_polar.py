"""A batched polar is the same computation as its single runs.

Each angle in a vmapped batch must follow exactly the path of its own
standalone run: same CFL ramp, same rejections, same state. Compared after a
fixed number of implicit steps on a coarse O-grid, to the round-off that
batching reductions (norms, GMRES dot products) may reorder. The multi-device
path is run in a subprocess with the CPU split into two XLA devices, which
must be set before JAX starts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from jaxaero import load_case, run
from jaxaero.polar import main, run_polar

pytestmark = pytest.mark.slow   # ~6 min on a CPU: the implicit step compiles several times

ROOT = Path(__file__).resolve().parents[2]
SMALL = ("case.domain.cells_airfoil=32", "case.domain.cells_normal=16", "numerical.time_integration.solver=implicit",
         "numerical.convergence.max_iters=40", "numerical.convergence.block_iters=20")
ALPHAS = (0.5, 2.0, 3.5)


@pytest.fixture(scope="module")
def case():
    return load_case(ROOT / "examples" / "naca0012_euler", SMALL)


@pytest.fixture(scope="module")
def batched(case):
    return run_polar(case, ALPHAS)


def test_each_batched_angle_matches_its_own_run(case, batched) -> None:
    import dataclasses

    for alpha, result in zip(ALPHAS, batched.results):
        single = run(dataclasses.replace(case, flow=dataclasses.replace(case.flow, alpha_deg=alpha)))
        assert result.history.n_iters == single.history.n_iters == 40
        # Round-off, amplified by 40 Newton-Krylov steps on wall-stretched cells: 7e-12 at most on O(1) fields.
        assert np.allclose(np.asarray(result.U), np.asarray(single.U), rtol=1e-10, atol=1e-11)
        assert result.forces.cl == pytest.approx(single.forces.cl, rel=1e-9, abs=1e-12)
        # The monitor's lift is the pressure lift of the same iterates (inviscid: all of it).
        assert np.allclose(result.cl_history, single.cl_history, rtol=1e-9, atol=1e-12)


def test_the_polar_is_written_with_one_result_per_angle(batched, tmp_path) -> None:
    out = batched.write(tmp_path / "polar")
    table = np.loadtxt(out / "polar.dat", ndmin=2)
    assert np.allclose(table[:, 0], ALPHAS) and table.shape == (3, 9)
    assert table[2, 1] > table[0, 1] > 0.0              # lift grows with incidence
    assert all((out / f"alpha_{a:+.2f}").is_dir() for a in ALPHAS)
    assert batched.timing.cases == 3 and batched.timing.devices == 1


def test_the_command_line_takes_a_negative_start(tmp_path) -> None:
    """``--alpha START STOP STEP``: a negative START is a number, not an option."""
    out = tmp_path / "cli"
    assert main([str(ROOT / "examples" / "naca0012_euler"), "--alpha", "-1", "1", "1",
                 *(f"--set={s}" for s in SMALL), "--out", str(out), "--quiet"]) == 0
    assert np.allclose(np.loadtxt(out / "polar.dat")[:, 0], [-1.0, 0.0, 1.0])


def test_two_devices_give_the_one_device_answer(batched) -> None:
    """shard_map over two (CPU) devices; three angles, so one is padding."""
    script = textwrap.dedent(f"""
        import jax, numpy as np
        from jaxaero import load_case
        from jaxaero.polar import run_polar
        assert len(jax.devices()) == 2, jax.devices()
        case = load_case({str(ROOT / "examples" / "naca0012_euler")!r}, {SMALL!r})
        polar = run_polar(case, {ALPHAS!r})
        assert polar.timing.devices == 2
        np.save("U.npy", np.stack([np.asarray(r.U) for r in polar.results]))
    """)
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    workdir = Path(__file__).parent / "_two_devices"
    workdir.mkdir(exist_ok=True)
    try:
        done = subprocess.run([sys.executable, "-c", script], cwd=workdir, env=env,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stderr[-2000:]
        U = np.load(workdir / "U.npy")
    finally:
        for f in workdir.glob("*"):
            f.unlink()
        workdir.rmdir()
    for k, result in enumerate(batched.results):
        assert np.allclose(U[k], np.asarray(result.U), rtol=1e-10, atol=1e-11)          # as above: amplified round-off
