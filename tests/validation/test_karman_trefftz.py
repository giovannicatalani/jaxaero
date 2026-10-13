"""The conformal Karman-Trefftz O-mesh generator.

What is checked, and what each check is worth:

* the conformal map round-trips, and sends ``zeta = +-1`` to the two singular
  points. These are inverse-function identities; they catch a transcription
  error in the exponent or the branch, nothing more.
* the generated surface lies on the **analytic** NACA 0012 thickness law. This
  is a real check: the formula is not derivable from the grid arrays.
* every cell area is strictly positive, on every level. The whole reason for a
  closed-form generator is that it cannot fold over; this asserts it did not.
* the radial growth ratio equals ``exp(2 pi / N_C)``. This is the property that
  makes the mesh conformal and is the reason these grids exist, so it is
  measured rather than assumed.
* the grid is symmetric about ``y = 0`` **bitwise**, which is what lets a
  zero-incidence case return exactly zero lift.
* the far field sits ~150 chords out, matching the reference computations.
  Published values at a near boundary are worth 25-35 lift counts and are not
  comparable.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.grids import karman_trefftz as kt
from jaxaero.mesh.block import o_grid_sides, validate_metrics

from ..single import body_area, wall_slice

LEVELS = (32, 64, 128)
"""Sizes exercised by the fast suite. 256 is left to the validation script."""


# --- The section ------------------------------------------------------------


def test_trailing_edge_closes() -> None:
    """``X_TE`` is the root of the thickness law past x = 1, to double precision."""
    assert kt.thickness(kt.X_TE) == pytest.approx(0.0, abs=1e-15)
    assert kt.X_TE == pytest.approx(kt.X_TE_PUBLISHED, abs=1e-9)
    assert kt.thickness(0.0) == 0.0


def test_published_section_properties_are_reproduced() -> None:
    """Every geometric constant the reference prints comes out of the formula.

    Leading-edge radius excepted: that is quoted as ``1.1019 t^2`` and the
    series implies 0.0158669, a fourth-digit disagreement that only shifts the
    inner singular point.
    """
    assert kt.VJ.tau == pytest.approx(0.2818725, abs=1e-7)
    assert np.rad2deg(kt.VJ.tau) == pytest.approx(16.1501066197, abs=1e-8)
    assert kt.VJ.power == pytest.approx(1.9102755, abs=2e-6)

    x = np.linspace(1e-12, kt.X_TE, 2_000_001)
    total = 2.0 * kt.thickness(x)
    assert total.max() == pytest.approx(0.1200345462, abs=1e-9)
    assert x[total.argmax()] == pytest.approx(0.2998278780, abs=1e-6)


# --- The map ----------------------------------------------------------------


def test_singular_points_map_as_specified() -> None:
    """``zeta = +1 -> z1`` (trailing edge) and ``zeta = -1 -> z2`` (inside)."""
    assert kt.forward_map(np.array([1.0 + 0j]))[0] == pytest.approx(kt.VJ.z1, abs=1e-15)
    # zeta = -1 makes W infinite; the reciprocal branch is what keeps this finite.
    assert kt.forward_map(np.array([-1.0 + 0j]))[0] == pytest.approx(kt.VJ.z2, abs=1e-15)
    assert kt.inverse_map(np.array([kt.VJ.z1]))[0] == pytest.approx(1.0 + 0j, abs=1e-15)


def test_map_round_trips_on_field_points() -> None:
    """``z -> zeta -> z`` to round-off on a scatter well away from the body."""
    rng = np.random.default_rng(0)
    z = 0.3 + 4.0 * (rng.standard_normal(4000) + 1j * rng.standard_normal(4000))
    z = z[np.abs(z - 0.5) > 1.0]
    assert np.abs(kt.forward_map(kt.inverse_map(z)) - z).max() < 1e-12


def test_map_round_trips_on_the_contour() -> None:
    """The contour is where the branch cut is closest, so it is checked alone."""
    z, _ = kt.airfoil_contour(256)
    assert np.abs(kt.forward_map(kt.inverse_map(z)) - z).max() < 1e-14


def test_map_is_continuous_around_the_contour() -> None:
    """No branch-cut jump: the pre-image is a near circle, not a torn curve.

    The cut's pre-image is the segment between the two singular points, which
    lies inside the body. If the principal branch were wrong the mapped radius
    would jump by order one somewhere; here it varies by a few percent.
    """
    z, _ = kt.airfoil_contour(2048)
    circle = kt.near_circle(z)
    assert circle.radius.max() / circle.radius.min() < 1.1
    step = np.abs(np.diff(np.unwrap(circle.angle)))
    assert step.max() < 3.0 * np.median(step)


def test_near_circle_centre_is_real() -> None:
    """The centre is the mapped chordline midpoint, hence on the axis."""
    circle = kt.near_circle(kt.airfoil_contour(128)[0])
    assert abs(circle.centre.imag) < 1e-15


# --- The mesh ---------------------------------------------------------------


@pytest.mark.parametrize("n_c", LEVELS)
def test_surface_lies_on_the_analytic_section(n_c: int) -> None:
    """Wall nodes satisfy the thickness law, to 2.5e-16.

    The leading-edge node is excluded from the *vertical* comparison and checked
    for position instead. The thickness law leaves the nose with an infinite
    slope, so there ``|y| - thickness(x)`` amplifies the absolute error in ``x``
    without bound: the node lands 2.4e-17 from the curve -- round-off of the
    conformal round trip -- and the vertical gap reads 9e-10. The node position
    is the well-conditioned quantity at that one station, and it is what the
    solver uses.

    Only one node is affected: ``x = X_TE cos^2(pi i / n_c)`` falls below 1e-6
    only at ``i = n_c / 2`` for any size up to 2048.
    """
    x, y = kt.o_mesh_nodes(n_c)
    xs, ys = x[:, 0], y[:, 0]

    away_from_the_nose = xs >= 1e-6
    assert away_from_the_nose.sum() == n_c           # all but the nose node
    gap = np.abs(ys[away_from_the_nose]) - kt.thickness(xs[away_from_the_nose])
    assert np.abs(gap).max() < 1e-12

    # The wrap node is the trailing edge, on the axis at x = X_TE.
    assert ys[0] == 0.0
    assert xs[0] == pytest.approx(kt.X_TE, abs=1e-12)
    # The leading edge falls on a node, at the origin.
    assert xs[n_c // 2] == pytest.approx(0.0, abs=1e-12)
    assert ys[n_c // 2] == 0.0


@pytest.mark.parametrize("n_c", LEVELS)
def test_wrap_closes_bitwise(n_c: int) -> None:
    """Node ``i = ni`` repeats node ``i = 0`` exactly, so the seam is watertight.

    Exactness matters: a seam a few ULP wide would break the bitwise symmetry
    the zero-lift cases rely on, and the periodic ghost exchange would be an
    interpolation rather than a copy.
    """
    x, y = kt.o_mesh_nodes(n_c)
    assert np.array_equal(x[0, :], x[-1, :])
    assert np.array_equal(y[0, :], y[-1, :])


@pytest.mark.parametrize("n_c", LEVELS)
def test_grid_is_symmetric_about_the_axis_bitwise(n_c: int) -> None:
    """Station ``i`` and station ``n_c - i`` are exact mirror images."""
    x, y = kt.o_mesh_nodes(n_c)
    mirror = np.concatenate([(n_c - np.arange(n_c)) % n_c, [0]])
    assert np.array_equal(x, x[mirror])
    assert np.array_equal(y, -y[mirror])


@pytest.mark.parametrize("n_c", LEVELS)
def test_all_cell_areas_are_strictly_positive(n_c: int) -> None:
    """The closed-form construction cannot fold over. Asserted, not assumed."""
    report = validate_metrics(kt.make_o_grid(n_c))
    assert report.min_area > 0.0
    assert report.ok, report.summary()


@pytest.mark.parametrize("n_c", LEVELS)
def test_radial_growth_matches_the_conformal_ratio(n_c: int) -> None:
    """Wall-normal cell width grows by ``exp(2 pi / N_C)``.

    This is the defining property of the family, and it holds exactly in the
    median and to 1.3e-4 over the outer half of the mesh. Two deliberate
    departures, both measured:

    * Over roughly the first ten radial layers the growth is *milder* than the
      conformal ratio -- 1.014 rising to 1.050 at ``N_C = 128`` -- because the
      modulus of ``dz/dzeta`` varies quickly within a chord of the body. Milder
      than conformal is the harmless direction.
    * At the trailing-edge station alone the first cell is compressed, giving a
      ratio of 2.22 to 2.26 across the family, because the map opens a corner of
      angle ``tau`` there and ``dz/dzeta`` vanishes. That is the single worst
      cell on the mesh; the TMR levels it replaces grade at 4.04 / 2.84 / 2.13
      in their maxima and 1.73 / 1.32 / 1.15 in their medians.
    """
    from jaxaero.mesh.block import cell_widths

    grid = kt.make_o_grid(n_c)
    _, width_j = cell_widths(grid.x_node, grid.y_node)
    width_j = np.asarray(width_j)
    ratio = width_j[:, 1:] / width_j[:, :-1]
    target = np.exp(2.0 * np.pi / n_c)

    assert np.median(ratio) == pytest.approx(target, rel=1e-4)
    assert np.abs(ratio[:, n_c // 2 :] / target - 1.0).max() < 1e-3
    assert ratio.min() >= 1.0                 # coarsens outward, never contracts
    assert ratio.max() < 2.3
    # The worst cell is the trailing-edge corner, nowhere else.
    assert np.unravel_index(ratio.argmax(), ratio.shape) == (0, 0)


@pytest.mark.parametrize("n_c", LEVELS)
def test_cells_are_nearly_square(n_c: int) -> None:
    """Aspect ratio stays within 0.7 to 1.1 -- the point of a conformal mesh.

    For contrast the TMR C-grids this replaces carry near-wall cells of aspect
    ratio in the hundreds, sized for a boundary layer an Euler solution has not
    got.
    """
    from jaxaero.mesh.block import cell_widths

    grid = kt.make_o_grid(n_c)
    width_i, width_j = cell_widths(grid.x_node, grid.y_node)
    ratio = np.asarray(width_i) / np.asarray(width_j)
    assert ratio.min() > 0.7
    assert ratio.max() < 1.1


@pytest.mark.parametrize("n_c", LEVELS)
def test_far_field_is_about_150_chords_out(n_c: int) -> None:
    """``i_max = j_max`` places the outer boundary there with no extra choice.

    The reference computations use ~150 chords and their published values are
    only comparable against a boundary of that order.
    """
    x, y = kt.o_mesh_nodes(n_c)
    radius = np.hypot(x[:, -1], y[:, -1])
    assert 140.0 < radius.min()
    assert radius.max() < 160.0
    assert radius.max() / radius.min() < 1.02    # nearly a circle in z as well


def test_topology_matches_the_node_array() -> None:
    """``make_o_grid`` builds a square O-grid of the requested size."""
    grid = kt.make_o_grid(64)
    assert grid.sides == o_grid_sides(grid.ni, grid.nj)
    assert (grid.ni, grid.nj) == (64, 64)
    assert grid.x_node.shape == (65, 65)
    assert wall_slice(grid) == slice(0, 64)


def test_airfoil_area_matches_the_analytic_section() -> None:
    """Enclosed area against the closed-form integral of the thickness law.

    An independent check, unlike the closure identities: the integral cannot be
    recovered from the grid arrays. It converges as the chordwise count rises,
    so the tolerance is on the finest level only.
    """

    a0, a1, a2, a3, a4 = kt.NACA0012_COEFFICIENTS
    x_te = kt.X_TE
    exact = 2.0 * kt.THICKNESS_SCALE * (
        a0 * (2.0 / 3.0) * x_te**1.5
        + a1 * x_te**2 / 2.0
        + a2 * x_te**3 / 3.0
        + a3 * x_te**4 / 4.0
        + a4 * x_te**5 / 5.0
    )
    assert body_area(kt.make_o_grid(512)) == pytest.approx(exact, rel=1e-4)


@pytest.mark.parametrize("n_c", (3, 5, 2, 0, -4))
def test_odd_or_tiny_station_counts_are_rejected(n_c: int) -> None:
    """The leading edge must fall on a node, so the count must be even."""
    with pytest.raises(ValueError, match="even"):
        kt.airfoil_contour(n_c)


# --- Viscous (wall-clustered) meshes -----------------------------------------

VISCOUS = ((64, 96, 2e-4), (128, 176, 1e-4))


@pytest.mark.parametrize("n_c, nj, wall_spacing", VISCOUS)
def test_viscous_mesh_passes_the_gate_and_keeps_the_far_field(n_c, nj, wall_spacing) -> None:
    grid = kt.make_o_grid(n_c, nj, wall_spacing)
    assert (grid.ni, grid.nj) == (n_c, nj)
    report = validate_metrics(grid)
    assert report.ok, report.summary()
    x, y = np.asarray(grid.x_node), np.asarray(grid.y_node)
    assert np.hypot(x[:, -1], y[:, -1]).mean() == pytest.approx(148.6, abs=1.0)


@pytest.mark.parametrize("n_c, nj, wall_spacing", VISCOUS)
def test_viscous_first_cell_height_meets_its_target(n_c, nj, wall_spacing) -> None:
    """The mean first-cell height is the requested spacing, to 2%.

    The mean, because that is what the construction fixes: cells stay near
    square at the wall, and the mean wall arc is the perimeter over ``n_c``.
    Locally the map clusters toward both edges (about 0.4x the target there).
    """
    from jaxaero.mesh.block import cell_widths

    grid = kt.make_o_grid(n_c, nj, wall_spacing)
    _, width_j = cell_widths(grid.x_node, grid.y_node)
    assert np.mean(np.asarray(width_j)[:, 0]) == pytest.approx(wall_spacing, rel=0.02)


def test_viscous_mesh_grows_smoothly_and_stays_symmetric() -> None:
    from jaxaero.mesh.block import cell_widths

    n_c, nj = 128, 176
    x, y = kt.o_mesh_nodes(n_c, nj, 1e-4)
    mirror = np.concatenate([(n_c - np.arange(n_c)) % n_c, [0]])
    assert np.array_equal(x, x[mirror]) and np.array_equal(y, -y[mirror])
    _, width_j = cell_widths(x, y)
    growth = np.asarray(width_j)[:, 1:] / np.asarray(width_j)[:, :-1]
    assert np.median(growth) < 1.06
    assert growth.max() < 1.2


def test_too_few_radial_cells_are_rejected() -> None:
    """With ``nj <= n_c`` the clustered steps cannot reach the far field."""
    with pytest.raises(ValueError, match="cannot span"):
        kt.make_o_grid(64, 64, 1e-4)


# --- The HiOCFD C1.2 section --------------------------------------------------


def test_hiocfd_section_closes_at_unit_chord() -> None:
    """``a4 = -0.1036`` makes the coefficients sum to zero: closed at ``x = 1``."""
    section = kt.HIOCFD
    assert section.x_te == 1.0
    assert abs(float(section.thickness(1.0))) < 1e-16
    assert sum(section.coefficients) == pytest.approx(0.0, abs=1e-15)
    # Blunter than V&J's at the trailing edge, since it closes sooner.
    assert section.tau > kt.VJ.tau


@pytest.mark.parametrize("nj, wall_spacing", ((None, None), (96, 2e-4)))
def test_hiocfd_mesh_hugs_its_own_section(nj, wall_spacing) -> None:
    x, y = kt.o_mesh_nodes(64, nj, wall_spacing, section=kt.HIOCFD)
    # Every wall node on the analytic curve except the nose, where sqrt(x)
    # turns the round-off in x (~1e-17) into ~5e-10 of thickness.
    away = x[:, 0] > 1e-6
    assert np.abs(np.abs(y[away, 0]) - kt.HIOCFD.thickness(x[away, 0])).max() < 1e-12
    assert x[:, 0].max() == 1.0
    grid = kt.make_o_grid(64, nj, wall_spacing, section=kt.HIOCFD)
    assert validate_metrics(grid).ok
