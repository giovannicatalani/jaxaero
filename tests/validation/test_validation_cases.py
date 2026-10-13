"""The validation layer (validation/validate.py): every validation case loads,
its levels build the grid family it documents, and the judging helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from jaxaero import load_case
from jaxaero.mesh.block import cell_widths
from jaxaero.post.forces import Forces
from jaxaero.solvers.driver import ConvergenceHistory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
import validate as v  # noqa: E402

NAMES = sorted(d.name for d in (ROOT / "validation" / "cases").iterdir() if d.is_dir())


@pytest.mark.parametrize("name", NAMES)
def test_every_validation_case_and_level_parses(name) -> None:
    val = v.load_validation(name)
    assert val.levels and val.suite
    for level in val.levels:
        case = load_case(val.directory, v.level_overrides(level))
        assert case.name == name


def test_suites_are_ordered() -> None:
    assert [c.name for c in v.suite("ladder")] == ["vj_m05a0", "vj_m05a125", "vj_m08a0", "vj_m08a125"]
    assert [c.name for c in v.suite("rans")][0] == "tmr_flatplate_sa"
    with pytest.raises(KeyError, match="ladder"):
        v.suite("nope")


def test_the_laminar_family_is_n_by_11n_over_8_with_halving_spacing() -> None:
    val = v.load_validation("hiocfd_c12")
    grid = load_case(val.directory, v.level_overrides(val.levels[0])).grid().mesh[0]
    assert (grid.ni, grid.nj) == (32, 44)
    _, width_j = cell_widths(grid.x_node, grid.y_node)
    assert float(np.mean(width_j[:, 0])) == pytest.approx(1.28e-2 / 32, rel=0.01)


def test_levels_are_selected_by_index_or_name() -> None:
    levels = v.load_validation("tmr_naca0012_sa_a10").levels
    assert v.select_levels(levels, "0,2") == [levels[0], levels[2]]
    assert v.select_levels(levels, "225-65") == [levels[1]]
    assert v.select_levels(levels, None) == list(levels)


def test_richardson_recovers_order_and_limit() -> None:
    f_star, c, p = 0.0553, 0.4, 1.7
    fit = v.richardson(tuple(f_star + c * h**p for h in (1.0, 0.5, 0.25)))
    assert fit.monotone and fit.order == pytest.approx(p, rel=1e-12)
    assert fit.extrapolated == pytest.approx(f_star, rel=1e-12)
    assert np.isnan(v.richardson((1.0, 0.9, 0.95)).order)


def judged(cl_history):
    from jaxaero.simulation import Result

    case = load_case(ROOT / "validation" / "cases" / "hiocfd_c12")
    result = Result(case, None, None, Forces(cl=0.0183, cd=0.0553, cm=0.0, fx=0.0, fy=0.0),
                    ConvergenceHistory(residual=np.ones(3), reason="converged", n_iters=3),
                    np.asarray(cl_history), 0.0)
    return v.Judged(result, v.load_validation("hiocfd_c12").reference, 1e-4)


def test_a_short_run_never_counts_as_settled() -> None:
    assert not judged([0.02, 0.02]).settled
    assert judged([0.02] * 16).settled
    assert not judged([0.02] * 12 + [0.02, 0.03, 0.02, 0.03]).settled
    assert judged([0.0183] * 16).failures() == []
