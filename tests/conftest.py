"""Shared fixtures.

Importing ``airfoil`` enables float64 before any array exists. Every test runs in
double precision: a verification suite in single precision cannot tell a bug from
round-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # validation/ (Karman-Trefftz grids)

import jaxaero  # noqa: F401  (imported for its float64 side effect)
from jaxaero.core.config import FlowConditions, NumericsConfig
from validation.grids.karman_trefftz import make_o_grid
from jaxaero.core.state import freestream_primitive, primitive_to_conservative, split


@pytest.fixture(scope="session")
def coarse_grid():
    """The 32 x 32 conformal O-grid. Built once per session."""
    return make_o_grid(32)


@pytest.fixture(scope="session")
def medium_grid():
    """The 64 x 64 O-grid, for refinement comparisons."""
    return make_o_grid(64)


@pytest.fixture
def subsonic_flow() -> FlowConditions:
    """M = 0.5 at zero incidence: the symmetry verification condition."""
    return FlowConditions(mach=0.5, alpha_deg=0.0)


@pytest.fixture
def transonic_flow() -> FlowConditions:
    """The Vassberg-Jameson transonic case: M = 0.8, alpha = 1.25 degrees."""
    return FlowConditions(mach=0.8, alpha_deg=1.25)


@pytest.fixture
def numerics() -> NumericsConfig:
    return NumericsConfig(cfl=1.5, max_iters=200)


def random_physical_states(n: int, seed: int = 0, dim: int = 2) -> np.ndarray:
    """``(dim + 2, n)`` primitive states with positive density and pressure.

    Velocities span subsonic through supersonic so the tests exercise every wave
    pattern, not just the one the airfoil case happens to produce.
    """
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.2, 3.0, n)
    velocity = [rng.uniform(-2.5, 2.5, n) for _ in range(dim)]
    p = rng.uniform(0.2, 3.0, n)
    return np.stack([rho, *velocity, p])


def random_face_vectors(n: int, seed: int = 1, dim: int = 2) -> np.ndarray:
    """``(dim, n)`` face area vectors of assorted lengths and directions."""
    rng = np.random.default_rng(seed)
    length = rng.uniform(1e-4, 5.0, n)
    if dim == 2:
        angle = rng.uniform(0.0, 2.0 * np.pi, n)
        return np.stack([length * np.cos(angle), length * np.sin(angle)])
    direction = rng.standard_normal((dim, n))
    return length * direction / np.linalg.norm(direction, axis=0)


def freestream_state(flow: FlowConditions, dim: int = 2):
    """Freestream conservative vector, ``(dim + 2,)``."""
    return primitive_to_conservative(*split(jnp.asarray(freestream_primitive(flow, dim)), dim), flow.gamma)


def uniform_state(grid, flow: FlowConditions):
    """Freestream broadcast over every cell, ``(4, ni, nj)``."""
    return jnp.broadcast_to(freestream_state(flow)[:, None, None], (4, grid.ni, grid.nj))
