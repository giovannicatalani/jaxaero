"""The iteration driver, on toy steps whose residuals are known in closed form.

A batched step returns one residual per case; each case must be judged on its
own peak, and the march must end only when every case is finished.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxaero.solvers.driver import ExitReason, run_to_steady


def decay(rates):
    """Case k's residual is ``x_k``, multiplied by ``rates[k]`` every step."""
    rates = jnp.asarray(rates)
    return lambda x: (x * rates, x * rates)


def test_a_single_case_converges_from_its_peak() -> None:
    _, history = run_to_steady(decay(0.5), jnp.asarray(1.0), max_iters=1000, tol=1e-6, block_iters=10)
    assert history.converged and history.residual.ndim == 1
    # Peak 0.5 (the first residual); 0.5^n <= 1e-6 * 0.5 from n = 21, seen at the block end, 30.
    assert history.n_iters == 30
    assert history.seconds >= history.first_block_seconds > 0.0


def test_each_case_of_a_batch_is_judged_on_its_own() -> None:
    _, history = run_to_steady(decay([0.5, 1.0, 0.9]), jnp.asarray([1.0, 3.0, 1e-3]),
                               max_iters=200, tol=1e-6, block_iters=10)
    assert history.residual.shape == (200, 3)
    assert history.case_reasons == (ExitReason.CONVERGED, ExitReason.MAX_ITERS, ExitReason.CONVERGED)
    assert history.reason == ExitReason.MAX_ITERS     # not every case converged
    # Case 2 starts 1000x smaller, but is judged against its own peak, not case 1's.
    assert history.case(2).converged and history.case(2).orders_dropped > 6.0
    assert np.allclose(history.case(1).residual, 3.0)


def test_the_march_stops_when_every_case_is_finished() -> None:
    _, history = run_to_steady(decay([0.5, 0.25]), jnp.asarray([1.0, 1.0]),
                               max_iters=1000, tol=1e-6, block_iters=10)
    assert history.converged and history.n_iters == 30     # the slower case, as above


def test_a_non_finite_case_fails_alone() -> None:
    step = lambda x: (x * jnp.asarray([0.5, jnp.inf]), x * jnp.asarray([0.5, jnp.inf]))  # noqa: E731
    _, history = run_to_steady(step, jnp.asarray([1.0, 1.0]), max_iters=100, tol=1e-6, block_iters=10)
    assert history.case_reasons == (ExitReason.CONVERGED, ExitReason.NON_FINITE)


def test_the_caller_can_stop_the_march_as_settled() -> None:
    """``on_block`` returning true ends a run that has not reached ``tol``, as
    ``SETTLED``, at the end of that block."""
    _, history = run_to_steady(decay(1.0), jnp.asarray(1.0), max_iters=1000, tol=1e-6, block_iters=10,
                               on_block=lambda b, state, residuals: b == 4)
    assert history.reason == ExitReason.SETTLED and not history.converged
    assert history.n_iters == 50


def test_a_settled_stop_does_not_hide_a_failure() -> None:
    _, history = run_to_steady(decay([1.0, jnp.inf]), jnp.asarray([1.0, 1.0]), max_iters=100, tol=1e-6,
                               block_iters=10, on_block=lambda b, state, residuals: True)
    assert history.case_reasons == (ExitReason.SETTLED, ExitReason.NON_FINITE)
    assert history.reason == ExitReason.NON_FINITE
