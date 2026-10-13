"""Named physical constants, each with its units and source.

One source of truth: nothing else in the package may hard-code these numbers.
The solver integrates nondimensional variables only; a case is specified by
Mach and Reynolds numbers, plus -- for Sutherland's law -- the ratio of the
Sutherland temperature to the freestream temperature, so no dimensional
constant is needed.

Nondimensionalization used throughout the package
-------------------------------------------------
Reference quantities are freestream density rho_inf, freestream speed of sound
a_inf, and airfoil chord c. With that choice the nondimensional freestream is

    rho = 1
    u   = M_inf * cos(alpha)
    v   = M_inf * sin(alpha)
    p   = 1 / gamma
    E   = p/(gamma-1) + 0.5*rho*(u^2+v^2)

because a^2 = gamma*p/rho implies p_inf = rho_inf * a_inf^2 / gamma, and the
nondimensional velocity is the Mach number by construction.

The nondimensional dynamic viscosity follows from the definition
Re = rho_inf * u_inf * c / mu_inf:

    mu = M_inf / Re

Time is nondimensionalized by c / a_inf.
"""

from __future__ import annotations

# --- Perfect gas, air -------------------------------------------------------

GAMMA_AIR: float = 1.4
"""Ratio of specific heats for air at moderate temperature [-]."""

PRANDTL_LAMINAR: float = 0.72
"""Laminar Prandtl number for air [-]."""

PRANDTL_TURBULENT: float = 0.9
"""Turbulent Prandtl number [-], constant, as in the NASA TMR cases."""


# --- Numerical safety floors ------------------------------------------------
# These are *detection* thresholds, not silent clips: the monitor reports the
# cell index where a state falls below them rather than quietly repairing it.

MIN_DENSITY: float = 1.0e-10
"""Nondimensional density below which a state is declared non-physical."""

MIN_PRESSURE: float = 1.0e-10
"""Nondimensional pressure below which a state is declared non-physical."""
