"""Frozen configs, validated once at construction so the numerics need no
input checks. Hashable and static under ``jit``; what a batch varies (a
polar's freestream) travels as an array instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants


TURBULENCE_MODELS: tuple[str, ...] = ("sa_noft2",)
"""Spalart-Allmaras without the ft2 term, with SA-neg (physics/spalart_allmaras.py)."""


@dataclass(frozen=True)
class FlowConditions:
    """Freestream and gas, nondimensional (constants.py). ``reynolds=None`` is
    inviscid; ``sutherland_ratio`` is ``S / T_inf`` (``None``: constant
    viscosity); ``nu_tilde_ratio`` is ``nu~_inf / nu_inf``."""

    mach: float
    alpha_deg: float = 0.0
    reynolds: float | None = None
    gamma: float = constants.GAMMA_AIR
    sutherland_ratio: float | None = None
    turbulence: str | None = None
    nu_tilde_ratio: float = 3.0

    def __post_init__(self) -> None:
        if not self.mach > 0.0:
            raise ValueError(f"mach must be positive, got {self.mach}")
        if self.mach >= 1.0:
            # Subsonic/transonic freestream only: the characteristic far-field
            # BC assumes a subsonic outer boundary.
            raise ValueError(
                f"freestream mach must be subsonic for the characteristic "
                f"far-field BC, got {self.mach}"
            )
        if not self.gamma > 1.0:
            raise ValueError(f"gamma must exceed 1, got {self.gamma}")
        if self.reynolds is not None and not self.reynolds > 0.0:
            raise ValueError(f"reynolds must be positive or None, got {self.reynolds}")
        if self.sutherland_ratio is not None:
            if self.reynolds is None:
                raise ValueError("sutherland_ratio given for an inviscid case")
            if not self.sutherland_ratio > 0.0:
                raise ValueError(f"sutherland_ratio must be positive, got {self.sutherland_ratio}")
        if self.turbulence is not None:
            if self.turbulence not in TURBULENCE_MODELS:
                raise ValueError(f"turbulence must be one of {TURBULENCE_MODELS}, "
                                 f"got {self.turbulence!r}")
            if self.reynolds is None:
                raise ValueError("a turbulence model needs a Reynolds number")
        if not self.nu_tilde_ratio > 0.0:
            raise ValueError(f"nu_tilde_ratio must be positive, got {self.nu_tilde_ratio}")
        if abs(self.alpha_deg) > 90.0:
            raise ValueError(f"alpha_deg out of range, got {self.alpha_deg}")

    @property
    def inviscid(self) -> bool:
        return self.reynolds is None

    @property
    def alpha_rad(self) -> float:
        return float(np.deg2rad(self.alpha_deg))

    @property
    def mu_inf(self) -> float:
        """``mu = M / Re``, at the freestream temperature."""
        if self.reynolds is None:
            raise ValueError("mu_inf is undefined for an inviscid case")
        return self.mach / self.reynolds


SOLVERS: tuple[str, ...] = ("explicit", "implicit")
"""Pseudo-time paths: multistage Runge-Kutta, pseudo-transient Newton-Krylov."""


@dataclass(frozen=True)
class NumericsConfig:
    """Discretization and stepping (``docs/case_reference.md``). ``tol`` is relative
    to the residual's running peak; ``forces_tol`` the largest spread of CL, CD
    and CM over the last quarter of a run for it to stop as settled (``None``:
    never); ``far_field`` keys ``numerics.boundary.FAR_FIELDS``."""

    flux: str = "hllc"
    limiter: str = "minmod"
    cfl: float = 1.5
    max_iters: int = 40_000
    tol: float = 1.0e-12
    forces_tol: float | None = 1.0e-4
    n_ghost: int = 2
    solver: str = "explicit"
    far_field: str = "characteristic"
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if not self.cfl > 0.0:
            raise ValueError(f"cfl must be positive, got {self.cfl}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if not 0.0 < self.tol < 1.0:
            raise ValueError(f"tol must lie in (0, 1), got {self.tol}")
        if self.forces_tol is not None and not self.forces_tol > 0.0:
            raise ValueError(f"forces_tol must be positive or null, got {self.forces_tol}")
        if self.n_ghost < 2:
            raise ValueError(
                f"n_ghost must be >= 2 for MUSCL reconstruction, got {self.n_ghost}"
            )
        if not -1.0 <= self.kappa <= 1.0:
            raise ValueError(f"muscl_kappa must lie in [-1, 1], got {self.kappa}")
        if self.kappa != 0.0 and self.limiter != "none":
            raise ValueError("muscl_kappa other than 0 is implemented without a limiter only "
                             f"(limiter 'none'), got limiter {self.limiter!r}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")

    @property
    def dtype(self) -> np.dtype:
        """Always float64: every verification gate runs in double (docs/dev/notes.md)."""
        return np.dtype("float64")


@dataclass(frozen=True)
class ImplicitConfig:
    """Pseudo-transient Newton-Krylov settings (solvers/implicit.py; meanings in
    ``docs/case_reference.md``, reasons in ``docs/dev/notes.md``)."""

    cfl: float = 10.0
    cfl_max: float = 1.0e3
    cfl_growth: float = 1.3
    cfl_cutback: float = 4.0
    cfl_min: float = 0.5
    preconditioner: str = "line_j"
    jacobian: str = "first_order"
    krylov_tol: float = 1.0e-2
    krylov_subspace: int = 10
    krylov_restarts: int = 2
    krylov_success: float = 0.5
    max_relative_update: float | None = None

    def __post_init__(self) -> None:
        if not self.cfl > 0.0:
            raise ValueError(f"cfl must be positive, got {self.cfl}")
        if self.cfl_max < self.cfl:
            raise ValueError(
                f"cfl_max must be at least the starting cfl {self.cfl}, "
                f"got {self.cfl_max}"
            )
        if not 0.0 < self.cfl_min <= self.cfl:
            raise ValueError(
                f"cfl_min must lie in (0, cfl], got {self.cfl_min}"
            )
        if self.cfl_growth < 1.0:
            raise ValueError(f"cfl_growth must be >= 1, got {self.cfl_growth}")
        if self.cfl_cutback <= 1.0:
            raise ValueError(f"cfl_cutback must exceed 1, got {self.cfl_cutback}")
        if self.jacobian not in ("exact", "first_order"):
            raise ValueError(
                f"jacobian must be 'exact' or 'first_order', got {self.jacobian!r}"
            )
        if not 0.0 < self.krylov_tol < 1.0:
            raise ValueError(f"krylov_tol must lie in (0, 1), got {self.krylov_tol}")
        if self.krylov_subspace < 1:
            raise ValueError(
                f"krylov_subspace must be >= 1, got {self.krylov_subspace}"
            )
        if self.krylov_restarts < 1:
            raise ValueError(
                f"krylov_restarts must be >= 1, got {self.krylov_restarts}"
            )
        if self.max_relative_update is not None and not self.max_relative_update > 0.0:
            raise ValueError(
                f"max_relative_update must be positive or None, got {self.max_relative_update}"
            )
        if not 0.0 < self.krylov_success < 1.0:
            raise ValueError(
                f"krylov_success must lie in (0, 1), got {self.krylov_success}"
            )
