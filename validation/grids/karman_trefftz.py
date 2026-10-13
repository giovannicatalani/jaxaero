"""Conformal Karman-Trefftz O-meshes around the NACA 0012.

Why this module exists
----------------------
The TMR C-grids this project started on were built for Re = 6e6 RANS: at
448x128, 31 of the 128 wall-normal cells lie within 0.01 chords of the wall,
resolving a boundary layer that an inviscid solution does not have, and the
wall-normal grading reaches 2.13 between adjacent cells. The transonic Euler
gate has never passed on them. The meshes built here are the ones the reference
values were computed on, so a disagreement can be charged to the scheme rather
than to the grid.

The construction is transcribed from Vassberg and Jameson, "In Pursuit of Grid
Convergence for Two-Dimensional Euler Solutions", J. Aircraft 47 (2010)
1152-1166 (preprint AIAA 2009-4114), sections III-IV. Everything is closed
form: there is no marching and no iteration, so the cell inversion that
defeated the abandoned hyperbolic generator cannot occur here.

Geometry
--------
The section is the standard NACA 0012 thickness law with its **coefficients
unchanged** and the chord *extended* until the curve closes:

    y(x) = +- 0.6 (0.2969 sqrt(x) - 0.126 x - 0.3516 x^2
                   + 0.2843 x^3 - 0.1015 x^4)

whose next root past x = 1 is ``X_TE = 1.00893041136514``. The reference chord
for force coefficients stays 1.0, not ``X_TE``. This is **not** the TMR
section, whose coefficients are rescaled to close at x = 1; the two are
different airfoils and their force coefficients are not interchangeable.

The mapping
-----------
Karman-Trefftz, with its two singular points at the trailing edge and half a
leading-edge radius inside the contour::

    ((zeta - 1)/(zeta + 1))^P = (z - z1)/(z - z2)
    P  = 2 - tau/pi          tau = included trailing-edge angle
    z1 = X_TE                (the trailing edge; image of zeta = +1)
    z2 = RHO_LE / 2          (inside the leading edge; image of zeta = -1)

The exponent turns the corner of angle ``tau`` at the trailing edge into the
smooth point ``zeta = +1``, which is what lets a polar grid in ``zeta`` come
back as a body-fitted grid in ``z``. Both branch cuts lie on the segment
``-1 <= zeta <= 1``, i.e. *inside* the body, so the principal branch is
continuous everywhere in the flow; ``test_karman_trefftz.py`` asserts that
rather than trusting it.

Mesh construction
-----------------
1. Place the contour stations in ``z`` so that their images in ``zeta`` are
   equally spaced in angle (``airfoil_contour``). The image is a *near* circle
   (radius varies by 4.5%), not an exact one, because a NACA 0012 is not
   exactly a Karman-Trefftz section.
2. Centre it on the midpoint of the mapped chordline, giving a radius
   ``r[i, 0]`` and an angle ``theta[i]`` per station.
3. ``R_1`` = (discrete arclength of the near circle) / 2 pi, the radius of the
   circle with the same perimeter.
4. ``R_j = R_1 exp(2 pi (j-1) / N_C)``. Exponential growth in a polar grid is
   what makes the mesh conformal: the cells stay square, because the arc length
   ``R dtheta`` and the radial step ``dR`` grow together.
5. Blend linearly in ``R_j`` from the near circle at the wall to a true circle
   at the far field, so the outer boundary is round and the inner one hugs the
   body.
6. Map back to ``z``.
7. Symmetrise about ``y = 0`` by averaging each station with its mirror
   partner, which makes the grid symmetric *bitwise* rather than to round-off,
   so a zero-incidence case can return CL = 0 exactly.

With ``i_max = j_max = N_C + 1`` the far field lands ~148 chords out by
construction -- the same order as the ~150 chords of the reference
computations, and far enough that the published values are comparable. A
boundary at 10 chords is worth 25-35 lift counts, so this is not a free choice.

Measured quality, N_C = 128
---------------------------
=========================================  ====================================
cell aspect ratio, whole mesh              0.98 to 1.03
wall-normal width growth, median           1.0503 = exp(2 pi / 128) exactly
wall-normal width growth, max              2.07, at the single trailing-edge
                                           cell (the corner the map opens out)
far-field radius                           148.1 to 149.1 chords
=========================================  ====================================

For comparison the TMR levels grade at 1.73 / 1.32 / 1.15 in the median with
maxima 4.04 / 2.84 / 2.13.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jaxaero.mesh.block import Grid, o_grid_sides

# --- The section -------------------------------------------------------------

NACA0012_COEFFICIENTS: tuple[float, float, float, float, float] = (
    0.2969,
    -0.126,
    -0.3516,
    0.2843,
    -0.1015,
)
"""Standard NACA four-digit thickness coefficients, scaled below by ``t/0.2``
with ``t = 0.12``. Unchanged from the standard section: it is the *chord* that
is extended to close the trailing edge, not the coefficients."""

THICKNESS_SCALE: float = 0.6
"""``t / 0.2`` for a 12% section."""

X_TE: float = 1.0089304113651427
"""Chordwise station where the thickness law returns to zero.

The root of ``thickness`` past ``x = 1``, to full double precision. Vassberg
and Jameson publish ``1.0089304115``, which differs by 1.3e-10 chords; the
rounded value leaves the trailing edge open by 4e-11, which would be the
largest departure from the analytic section anywhere on the generated contour.
The exact root is used so that closure is a property of the geometry rather
than a tolerance."""

X_TE_PUBLISHED: float = 1.0089304115
"""The value as printed in the reference, kept for provenance only."""

RHO_LE: float = 0.01586736
"""Leading-edge radius, ``1.1019 t^2`` for ``t = 0.12``, as published. The
thickness series itself implies ``(0.6 * 0.2969)^2 / 2 = 0.0158669``; they
differ in the fourth digit and only the placement of the inner singular point
depends on it, which is not a sensitive choice."""


@dataclass(frozen=True)
class Section:
    """A NACA four-digit-style symmetric section and the map constants it fixes.

    ``y(x) = +-scale (a0 sqrt(x) + a1 x + a2 x^2 + a3 x^3 + a4 x^4)`` on
    ``0 <= x <= x_te``, closed (zero thickness) at ``x_te``. Two are used: V&J's
    extended chord (``VJ``) and the HiOCFD workshop's closed trailing edge
    (``HIOCFD``). Both have reference chord 1.

    Attributes:
        coefficients: ``(a0, a1, a2, a3, a4)``.
        x_te: Trailing-edge station, a root of the thickness law.
        rho_le: Leading-edge radius, placing the inner singular point.
        scale: ``t / 0.2``; 0.6 for a 12% section.
    """

    coefficients: tuple[float, float, float, float, float]
    x_te: float
    rho_le: float
    scale: float = THICKNESS_SCALE

    def thickness(self, x: np.ndarray | float) -> np.ndarray:
        """Half-thickness ``y(x)`` in chords; zero at ``x = 0`` and ``x_te``."""
        x = np.asarray(x, dtype=np.float64)
        a0, a1, a2, a3, a4 = self.coefficients
        return self.scale * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)

    def thickness_slope(self, x: np.ndarray | float) -> np.ndarray:
        """``dy/dx`` of the half-thickness. Singular at ``x = 0``."""
        x = np.asarray(x, dtype=np.float64)
        a0, a1, a2, a3, a4 = self.coefficients
        return self.scale * (
            0.5 * a0 / np.sqrt(x) + a1 + 2 * a2 * x + 3 * a3 * x**2 + 4 * a4 * x**3
        )

    @property
    def tau(self) -> float:
        """Included trailing-edge angle, ``2 arctan|y'(x_te)|``: the two
        surfaces leave at equal and opposite slopes."""
        return float(2.0 * np.arctan(abs(self.thickness_slope(self.x_te))))

    @property
    def power(self) -> float:
        """Karman-Trefftz exponent ``P = 2 - tau/pi``. ``P = 2`` is Joukowski,
        with its cusped trailing edge."""
        return 2.0 - self.tau / np.pi

    @property
    def z1(self) -> complex:
        """Trailing-edge singular point; image of ``zeta = +1``."""
        return complex(self.x_te, 0.0)

    @property
    def z2(self) -> complex:
        """Inner singular point, half a leading-edge radius aft of the nose;
        image of ``zeta = -1``. Strictly inside the contour, which keeps the
        branch cut out of the flow."""
        return complex(0.5 * self.rho_le, 0.0)


VJ = Section(NACA0012_COEFFICIENTS, X_TE, RHO_LE)
"""Vassberg and Jameson's section: standard coefficients, chord extended to the
next root. The Euler validation grids."""

HIOCFD = Section(
    (0.2969, -0.1260, -0.3516, 0.2843, -0.1036),
    x_te=1.0,
    rho_le=0.5 * (THICKNESS_SCALE * 0.2969) ** 2,
)
"""The closed-trailing-edge NACA 0012 of the High-Order CFD Workshop, case C1.2:
``a4 = -0.1036`` instead of ``-0.1015`` so the coefficients sum to zero and the
section closes at ``x = 1``. The laminar validation geometry. Its leading-edge
radius is the one the series implies, ``(0.6 a0)^2 / 2``."""

TMR = Section(
    (0.298222773, -0.127125232, -0.357907906, 0.291984971, -0.105174606),
    x_te=1.0,
    rho_le=0.5 * (0.594689181 * 0.298222773) ** 2,
    scale=0.594689181,
)
"""The NASA Turbulence Modeling Resource NACA 0012: the coefficients rescaled
so the section closes at ``x = 1`` with the original maximum thickness,
``y = +-0.594689181 (0.298222773 sqrt(x) - 0.127125232 x - 0.357907906 x^2
+ 0.291984971 x^3 - 0.105174606 x^4)``, from
<https://turbmodels.larc.nasa.gov/naca0012_val.html>. The RANS validation
geometry; its grids are TMR's C-grids, read from file (mesh/plot3d.py). A third
airfoil, distinct from ``VJ`` and ``HIOCFD``."""


def thickness(x: np.ndarray | float, section: Section = VJ) -> np.ndarray:
    """Half-thickness of ``section`` (V&J's by default)."""
    return section.thickness(x)


def thickness_slope(x: np.ndarray | float, section: Section = VJ) -> np.ndarray:
    """``dy/dx`` of the half-thickness of ``section``."""
    return section.thickness_slope(x)


# --- The conformal map -------------------------------------------------------


def forward_map(zeta: np.ndarray, section: Section = VJ) -> np.ndarray:
    """Map the circle plane to the physical plane, ``zeta -> z``.

    Inverting ``((zeta-1)/(zeta+1))^P = (z-z1)/(z-z2)`` for ``z`` gives

        z = (z1 - W z2) / (1 - W),        W = ((zeta-1)/(zeta+1))^P

    which is evaluated as written where ``|zeta-1| <= |zeta+1|`` and, where it is
    not, in the algebraically identical reciprocal form

        z = (z1 Wi - z2) / (Wi - 1),      Wi = ((zeta+1)/(zeta-1))^P = 1/W

    Choosing the branch on the *distances* rather than on ``|W|`` is what makes
    the two singular points work: at ``zeta = -1`` the first ratio is a division
    by zero, and at ``zeta = +1`` the second one is, so neither ratio may be
    formed before the choice is made. Whichever form is selected then has a
    factor of modulus at most one, so nothing overflows, and the limits come out
    exactly -- ``zeta = +1 -> z1``, ``zeta = -1 -> z2``.

    Args:
        zeta: Complex points in the circle plane, any shape.

    Returns:
        Complex physical coordinates, same shape.
    """
    zeta = np.asarray(zeta, dtype=np.complex128)
    minus, plus = zeta - 1.0, zeta + 1.0
    reciprocal = np.abs(minus) > np.abs(plus)

    # Both branches are evaluated over the whole array -- they are cheap, and
    # boolean indexing would not broadcast over the mesh shape. The dividend and
    # divisor are swapped *before* dividing, so the discarded branch is never
    # even formed and no 0/0 appears.
    numerator = np.where(reciprocal, plus, minus)
    denominator = np.where(reciprocal, minus, plus)
    factor = np.power(numerator / denominator, section.power)  # modulus <= 1
    z1, z2 = section.z1, section.z2
    near = (z1 - factor * z2) / (1.0 - factor)
    far = (z1 * factor - z2) / (factor - 1.0)
    return np.where(reciprocal, far, near)


def inverse_map(z: np.ndarray, section: Section = VJ) -> np.ndarray:
    """Map the physical plane to the circle plane, ``z -> zeta``.

        zeta = (1 + V) / (1 - V),        V = ((z-z1)/(z-z2))^(1/P)

    The principal branch is taken. Along the contour ``arg((z-z1)/(z-z2))``
    runs smoothly from +pi at the trailing edge through 0 at the nose to -pi
    back at the trailing edge, where the ratio is exactly zero; it never crosses
    the cut, because the cut's pre-image is the segment between the two singular
    points, which lies inside the body.

    Args:
        z: Complex physical coordinates, any shape.

    Returns:
        Complex circle-plane coordinates, same shape.
    """
    z = np.asarray(z, dtype=np.complex128)
    v = np.power((z - section.z1) / (z - section.z2), 1.0 / section.power)
    return (1.0 + v) / (1.0 - v)


# --- Mesh construction -------------------------------------------------------


def _lower_surface(t: np.ndarray, section: Section) -> np.ndarray:
    """Lower-surface point at contour parameter ``t`` in ``[0, pi]``, TE to LE.

    ``x = x_te cos^2(t/2)`` -- the half-angle form of ``x_te (1 + cos t) / 2``,
    because at the nose ``1 + cos t`` loses every significant digit while
    ``cos(t/2)`` keeps them all, and the ``sqrt(x)`` thickness law amplifies an
    absolute error in ``x`` without bound there.
    """
    x = section.x_te * np.cos(0.5 * t) ** 2
    return x - 1j * section.thickness(x)


def airfoil_contour(n_c: int, section: Section = VJ) -> tuple[np.ndarray, np.ndarray]:
    """The section at ``n_c`` stations, traversed trailing edge -> lower -> upper.

    Stations are equally spaced in angle on the near circle -- the contour's
    image in the circle plane, measured about the midpoint of the mapped
    chordline -- as V&J prescribe (section IV: "redistributed ... with constant
    angular spacing"). That is what makes the mesh smooth across the
    trailing-edge grid line. A cosine law in ``x`` instead puts the tightest
    spacing at the trailing edge from both sides, a cusp that leaves the cell
    width jumping by 7% across that line at every resolution, and an O(1)
    truncation error in any second-derivative operator there.

    Each lower-surface station solves ``angle(zeta(t) - centre) = theta_i`` for
    the contour parameter ``t`` by bisection (the angle decreases monotonically
    from 0 at the trailing edge to ``-pi`` at the nose), so it lies on the
    analytic curve exactly; V&J reach the same points by interpolating and
    projecting back until converged. The upper surface is the mirror image,
    bitwise.

    ``theta`` runs *negative* so that increasing ``i`` traverses the contour
    trailing edge -> lower surface -> leading edge -> upper surface. That is the
    direction ``mesh.block`` requires: it makes the j-face normals point out of
    the body and every cell area positive.

    Args:
        n_c: Stations around the body, even so that the leading edge lands on a
            node. The wrap point is *not* repeated.
        section: The airfoil; V&J's by default.

    Returns:
        ``(z, theta)``: complex contour points and their circle-plane angles,
        ``(n_c,)``.

    Raises:
        ValueError: if ``n_c`` is not a positive even number.
    """
    if n_c < 4 or n_c % 2 != 0:
        raise ValueError(f"n_c must be even and >= 4, got {n_c}")
    theta = -2.0 * np.pi * np.arange(n_c, dtype=np.float64) / n_c
    half = n_c // 2
    centre = 0.5 * (1.0 + inverse_map(np.array([0.0 + 0.0j]), section)[0].real)

    target = theta[1:half]
    lo, hi = np.zeros_like(target), np.full_like(target, np.pi)
    for _ in range(64):  # bisection to the last bit of t
        mid = 0.5 * (lo + hi)
        above = np.angle(inverse_map(_lower_surface(mid, section), section) - centre) > target
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    lower = _lower_surface(0.5 * (lo + hi), section)

    z = np.empty(n_c, dtype=np.complex128)
    z[0], z[half] = section.x_te + 0.0j, 0.0 + 0.0j
    z[1:half] = lower
    z[half + 1 :] = np.conj(lower[::-1])
    return z, theta


@dataclass(frozen=True)
class NearCircle:
    """The contour's image in the circle plane, described in polar form.

    A NACA 0012 is not exactly a Karman-Trefftz section, so its pre-image is a
    *near* circle: at ``N_C = 128`` the radius varies from 1.03700 to 1.08594,
    4.5%. That departure is the only thing standing between this construction
    and an exactly conformal mesh, and the blend of step 5 confines it to the
    first few grid lines.

    Attributes:
        centre: Midpoint of the mapped chordline. Real, hence the symmetry.
        radius: Distance of each station from the centre, ``(n_c,)``.
        angle: Angle of each station about the centre, ``(n_c,)``, decreasing
            with ``i`` because the contour is traversed clockwise.
        perimeter_radius: Radius of the circle with the same perimeter, i.e.
            the discrete arclength over ``2 pi``. This -- not a mean radius --
            is ``R_1``, because it is the perimeter that fixes the arc length a
            square cell has to match.
    """

    centre: complex
    radius: np.ndarray
    angle: np.ndarray
    perimeter_radius: float


def near_circle(z_contour: np.ndarray, section: Section = VJ) -> NearCircle:
    """Map a contour back to the circle plane and describe it in polar form.

    The centre is the midpoint of the mapped chordline: the mapped trailing
    edge, which is ``zeta = +1`` by construction, and the mapped nose.

    Args:
        z_contour: Contour points from ``airfoil_contour``, ``(n_c,)``, with
            the wrap point not repeated.

    Returns:
        The polar description.
    """
    zeta = inverse_map(z_contour, section)
    n_c = zeta.size
    centre = complex(0.5 * (zeta[0] + zeta[n_c // 2]))
    offset = zeta - centre
    arclength = float(np.abs(np.roll(zeta, -1) - zeta).sum())
    return NearCircle(
        centre=centre,
        radius=np.abs(offset),
        angle=np.angle(offset),
        perimeter_radius=arclength / (2.0 * np.pi),
    )


def radial_levels(
    perimeter_radius: float, n_c: int, nj: int | None = None, wall_step: float | None = None
) -> np.ndarray:
    """Radii of the ``nj + 1`` circle-plane grid lines, increasing.

    In log-radius ``s = ln(R / R_1)`` the conformal mesh is uniform,
    ``ds = 2 pi / n_c``: one angular step subtends an arc ``R dtheta`` and one
    radial step adds ``R ds``, so the cells are square. After ``n_c`` steps
    ``s = 2 pi`` and the radius has grown by ``exp(2 pi) = 535.49``, which puts
    the far field ~148 chords out with no separate choice of extent.

    With ``wall_step`` set the steps are instead clustered at the wall for a
    viscous boundary layer,

        ds_k = min(wall_step r^k, 2 pi / n_c),   k = 0 .. nj - 1,

    geometric growth until the cells are square, conformal beyond, with ``r``
    chosen so that ``sum ds_k = 2 pi`` and the far field stays where it was.

    Args:
        perimeter_radius: ``R_1``, from ``NearCircle``.
        n_c: Cells around the body.
        nj: Radial cells; ``n_c`` (the conformal mesh) when ``wall_step`` is
            ``None``, otherwise more, since the extra cells are what resolve the
            boundary layer.
        wall_step: First log-radius step ``ds_0``, or ``None`` for conformal.

    Returns:
        ``R_j`` for ``j = 0 .. nj``.

    Raises:
        ValueError: if ``nj`` cells cannot reach ``s = 2 pi`` with growth in (1, 2].
    """
    if wall_step is None:
        return perimeter_radius * np.exp(2.0 * np.pi * np.arange(n_c + 1) / n_c)
    ds_square = 2.0 * np.pi / n_c
    k = np.arange(nj)

    def total(r: float) -> float:
        return float(np.minimum(wall_step * r**k, ds_square).sum())

    # A growth ratio above 2 is never acceptable in a boundary layer, so it
    # bounds the search -- and keeps r**k finite.
    if not total(1.0) < 2.0 * np.pi < total(2.0):
        raise ValueError(
            f"nj = {nj} cannot span s = 2 pi from a wall step {wall_step:.3e} "
            f"with growth between 1 and 2: need more than {n_c} radial cells"
        )
    lo, hi = 1.0, 2.0
    for _ in range(200):  # bisection; total(r) is increasing in r
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if total(mid) < 2.0 * np.pi else (lo, mid)
    steps = np.minimum(wall_step * hi**k, ds_square)
    steps *= 2.0 * np.pi / steps.sum()  # land the far field exactly
    return perimeter_radius * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def blend_radii(circle: NearCircle, r_level: np.ndarray) -> np.ndarray:
    """Radius of every node, ``(n_c, n_c+1)``, blended wall-to-far-field.

        r[i, j] = ( r_wall[i] (R_out - R_j) + R_out (R_j - R_1) ) / (R_out - R_1)

    a linear interpolation *in* ``R_j`` between the near circle at ``R_1`` and a
    true circle at ``R_out``. At ``j = 1`` it returns ``r_wall`` exactly and at
    ``j = n_c + 1`` exactly ``R_out``, so the wall line hugs the body and the
    outer line is round.

    Args:
        circle: Polar description of the contour's image.
        r_level: Radial levels from ``radial_levels``.

    Returns:
        Node radii about ``circle.centre``.
    """
    r_1 = r_level[0]
    r_out = r_level[-1]
    return (
        circle.radius[:, None] * (r_out - r_level[None, :])
        + r_out * (r_level[None, :] - r_1)
    ) / (r_out - r_1)


def o_mesh_nodes(
    n_c: int,
    nj: int | None = None,
    wall_spacing: float | None = None,
    section: Section = VJ,
) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates of an O-mesh, ``(n_c+1, nj+1)`` each.

    Index ``i`` wraps around the body (trailing edge -> lower -> leading edge ->
    upper -> trailing edge) with node ``i = n_c`` repeating node ``i = 0``
    *bitwise*, and ``j`` runs from the wall at ``j = 0`` out to the far field.

    Args:
        n_c: Cells around the body. Even, at least 4.
        nj: Radial cells; see ``radial_levels``.
        wall_spacing: Target first-cell height at the wall, in chords, or
            ``None`` for the conformal mesh. Converted to a log-radius step
            through the contour perimeter ``L``: a conformal cell has physical
            height ``(L / 2 pi) ds``. It is a target, not a guarantee -- the
            map's stretching varies around the body.
        section: The airfoil; V&J's by default.

    Returns:
        ``(x_node, y_node)``, float64, ready for ``Grid.from_nodes`` with an
        ``o_grid_sides(n_c, nj)``.
    """
    z_contour, _ = airfoil_contour(n_c, section)
    circle = near_circle(z_contour, section)
    perimeter = float(np.abs(np.roll(z_contour, -1) - z_contour).sum())
    wall_step = None if wall_spacing is None else 2.0 * np.pi * wall_spacing / perimeter
    r_level = radial_levels(circle.perimeter_radius, n_c, nj, wall_step)
    radius = blend_radii(circle, r_level)

    z = forward_map(circle.centre + radius * np.exp(1j * circle.angle[:, None]), section)

    # Close the wrap by *repeating* node 0 rather than by evaluating the map at
    # theta = -2 pi, which would differ in the last bit and leave the periodic
    # ghost exchange in numerics/boundary.py with a seam.
    z = np.concatenate([z, z[:1]], axis=0)

    # Symmetrise about y = 0. Station i mirrors station n_c - i (modulo the
    # wrap), so averaging x and anti-averaging y makes the grid symmetric to the
    # last bit: 0.5*(a-b) and 0.5*(b-a) are exact negatives in IEEE arithmetic.
    # Without this the symmetry is only good to round-off, and a zero-incidence
    # case cannot return CL = 0 exactly.
    mirror = np.concatenate([(n_c - np.arange(n_c)) % n_c, [0]])
    x_node = 0.5 * (z.real + z.real[mirror])
    y_node = 0.5 * (z.imag - z.imag[mirror])
    return x_node, y_node


def make_o_grid(
    n_c: int,
    nj: int | None = None,
    wall_spacing: float | None = None,
    dtype: np.dtype | type = np.float64,
    section: Section = VJ,
) -> Grid:
    """Build an O-grid, metrics and all.

    Args:
        n_c: Cells around the body; 32, 64, 128, 256 is the family the
            validation ladder uses.
        nj: Radial cells. Defaults to ``n_c``, the conformal mesh.
        wall_spacing: First-cell height in chords for a viscous mesh, or
            ``None`` (conformal, the Euler validation grids).
        dtype: Array dtype. The *generation* is always float64.
        section: The airfoil; V&J's by default, ``HIOCFD`` for the laminar case.

    Returns:
        A ``Grid`` with ``o_grid_sides(n_c, nj)``.
    """
    nj = n_c if nj is None else nj
    x_node, y_node = o_mesh_nodes(n_c, nj, wall_spacing, section)
    return Grid.from_nodes(x_node, y_node, o_grid_sides(n_c, nj), dtype=dtype)
