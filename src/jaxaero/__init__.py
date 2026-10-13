"""GPU airfoil solver in JAX.

Cell-centred finite-volume solver for steady 2D airfoil flow on structured
grids (generated O-grids; C-grids and flat-plate rectangles from Plot3D):
compressible Euler, laminar Navier-Stokes and Spalart-Allmaras RANS.

    from jaxaero import load_case, run
    result = run(load_case("examples/naca0012_euler"))
    result.write()

or, from the shell, ``python -m jaxaero examples/naca0012_euler``.

Float64 is enabled here, before any submodule creates an array: JAX only
honours the x64 flag if it is set before the first allocation.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .case import Case, CaseError, load_case  # noqa: E402
from .simulation import Result, run  # noqa: E402

__all__ = ["Case", "CaseError", "Result", "load_case", "run"]
__version__ = "0.1.0"
