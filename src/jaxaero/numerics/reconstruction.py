"""MUSCL reconstruction: cell averages to face states.

The scheme
----------
For the face between cells ``m`` and ``p``, with ``mm`` and ``pp`` their outer
neighbours, the limited slopes and face states are

    sigma_L = limiter(q_m - q_mm, q_p - q_m)
    sigma_R = limiter(q_p - q_m, q_pp - q_p)
    qL = q_m + sigma_L / 2          qR = q_p - sigma_R / 2

With unlimited slopes this is the symmetric second-order average; all the
upwinding comes from the Riemann solver, never from the reconstruction. Both
face families use the same formula on different index windows, so they share one
implementation and take the axis as an argument.

Reconstruction is in **primitive** variables, on raw index differences, not
spacing-weighted gradients (docs/dev/notes.md).
"""

from __future__ import annotations

from jax import Array

from .limiters import Limiter



def _stencil(padded: Array, n_ghost: int, shape: tuple[int, ...], axis: int
             ) -> tuple[Array, Array, Array, Array]:
    """The four cell values straddling every face normal to index direction
    ``axis`` (0 for i, 1 for j, 2 for k).

    Returns ``(q_mm, q_m, q_p, q_pp)``, each of the face-array shape: ``shape``
    with one more along ``axis``, after the variable axis. Pure slicing, which
    is why the ghosted array is built rather than patched.
    """
    g = n_ghost

    def window(offset: int) -> Array:
        index = [slice(None)] + [slice(g, g + n) for n in shape]
        index[1 + axis] = slice(g - 2 + offset, g - 2 + offset + shape[axis] + 1)
        return padded[tuple(index)]

    return window(0), window(1), window(2), window(3)


def face_states(
    padded: Array, n_ghost: int, shape: tuple[int, ...], limiter: Limiter, axis: int,
    kappa: float = 0.0,
) -> tuple[Array, Array]:
    """Second-order face states for one family, ``(qL, qR)``: limited, or the
    unlimited kappa family (van Leer 1979; ``kappa`` other than 0 needs
    ``limiter`` 'none', which config enforces),

        qL = q_m + [(1 - kappa) (q_m - q_mm) + (1 + kappa) (q_p - q_m)] / 4,

    and its mirror for qR. kappa = 0 is the limiter-free branch below (Fromm),
    1/3 third-order upwind-biased in 1D (CFL3D's default)."""
    q_mm, q_m, q_p, q_pp = _stencil(padded, n_ghost, shape, axis)
    d_minus, d_mid, d_plus = q_m - q_mm, q_p - q_m, q_pp - q_p
    if kappa != 0.0:
        lo, hi = 0.25 * (1.0 - kappa), 0.25 * (1.0 + kappa)
        return q_m + lo * d_minus + hi * d_mid, q_p - lo * d_plus - hi * d_mid
    return (
        q_m + 0.5 * limiter(d_minus, d_mid),
        q_p - 0.5 * limiter(d_mid, d_plus),
    )


def first_order_faces(
    padded: Array, n_ghost: int, shape: tuple[int, ...], axis: int
) -> tuple[Array, Array]:
    """Piecewise-constant face states: the two cell values, no slope.

    A robust start-up, the control when a second-order result looks wrong, and
    the Jacobian source for the implicit path's defect correction.
    """
    _, q_m, q_p, _ = _stencil(padded, n_ghost, shape, axis)
    return q_m, q_p
