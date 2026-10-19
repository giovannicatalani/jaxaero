# Design notes

Why some things are done the way they are, with the measurements behind
each choice, grouped by module.

## solvers/implicit.py

- **Defect correction is the default Jacobian.** On 112x32 over 400 steps, a
  limited second-order residual dropped 0.06 orders with the exact Jacobian
  and 2.8 with the first-order one; an unlimited residual gave 4.9 orders
  exact. The exact Jacobian of a limited scheme carries the limiter's kinks.
- **The CFL ramp keys off the linear solve, not the residual history.** A
  stagnated GMRES returns a null update, which leaves the residual unchanged
  and looks like success to a residual-based test; a freestream start makes
  the residual rise through the transient, which looks like failure to
  textbook SER (Mulder and van Leer, J. Comput. Phys. 59 (1985)). One extra
  matvec, about 1% of a step, measures the linear reduction instead.
- **`max_relative_update`.** Rejection alone only stops steps that leave the
  physical domain. On TMR NACA 0012 at 15 deg, 449x129, accepted steps built
  an odd-even checkerboard in nu~ and rho within 600 steps, after which every
  step was rejected; bounding the relative change (0.2) cured it.

## solvers/driver.py

- **Force-based stopping is the default** (`convergence.forces_tol` 1e-4,
  the gallery's settledness bar, applied to CL, CD and CM). On the M6 1.08M
  level from freestream it stops at 2540 steps with the forces within 3e-6 of
  the 3000-step values; 1e-5 did not stop within 3000 (the last-quarter
  window still holds the early drift). RANS residuals stall at
  round-off 10-11 orders down, and some cases at 2-3 (30P30N L1, NACA 0012
  at 0 deg) with the forces settled to six digits long before. On RAE 2822
  the run stops at 940 of 6000 steps with the 6000-step forces to every
  printed digit; on the M6 135k level at 1600, the same forces to six digits
  whatever the Krylov settings.
- **A frozen run stops as `stalled`, never `settled`.** When every step is
  rejected the state does not move, so the forces are flat and would pass
  the settledness test (a mis-started 1.08M M6 run stopped as settled after
  260 steps that way). A block whose residual is bitwise constant is no
  progress, whatever the equations.

- **Convergence is judged from the running peak.** Started from uniform
  freestream, the first residual is near zero (every cell away from the wall
  is in equilibrium), so normalising by it reports a residual that appears
  to grow through the transient.

## core/config.py

- **float64 only.** A float32 path existed and was removed. The 5.9x
  double-precision penalty on a 4 GB laptop card is real,
  but every verification gate must run in double: a suite that cannot tell a
  bug from round-off is no use, and no result was ever quoted from
  the single-precision path.
- **HLLC by default**: it resolves a stationary contact exactly and needs no
  entropy fix, so the scheme carries no tunable dissipation.
- **Krylov sizes.** The Arnoldi loop (solvers/krylov.py) runs to full length
  whether or not it has converged, testing only between restarts, so a large
  subspace is paid in full every step: small subspace (10), two restarts.
  Inexact Newton tolerance 1e-2 (Eisenstat and Walker, SIAM J. Sci. Comput.
  17 (1996)).
- **CFL ramp defaults.** Start at 10 (five times what the explicit path
  tolerates); growth capped at 1.3 per step and earned by the residual
  reduction; `cfl_max` is a backstop, not a tuning knob: the ramp settles at
  whatever the preconditioner sustains, limited in practice by
  `krylov_success`.

## numerics/riemann.py

- **Why HLLC.** Roe's scheme is shock-unstable on high-aspect-ratio cells, and
  the TMR grids reach such ratios at the shock foot: on 448x128 at M 0.8 the
  transonic shock never settled (CL oscillating 0.04-0.28), curable only by
  an entropy-fix coefficient large enough to bias CL by 16%. HLLC takes its
  dissipation from wave-speed estimates, has no vanishing eigenvalue, and
  needs no fix. On the O-grid at M 0.8, alpha 1.25, 128 squared, the two agree
  to three digits (CL 0.33567 vs 0.33585, CD 0.025711 vs 0.025698, observed
  orders 0.94/1.37 vs 0.92/1.33), so Roe stays as the independent check, 15%
  cheaper per iteration.
- **Contact resolution is required**: a slip wall has `vn = 0` along its
  length, so a flux that smears contacts smears exactly where CL and CD are
  computed. A Rusanov entry was built and removed for that reason.

## numerics/reconstruction.py, numerics/limiters.py

- **Raw index differences, not spacing-weighted gradients.** Limiting a
  gradient over the real cell spacings is the formally correct second-order
  form on a graded grid; it was built, measured and removed. On the strongly
  graded TMR C-grids it lost badly (transonic CL 0.1854 against 0.2461,
  oscillation 4.5x larger): the TVD region of a limiter depends on the spacing
  ratio and these limiters ignore it. On the mildly graded conformal O-grids
  it lost marginally, for no gain in order (CD 0.003103 against 0.002831 at
  128 squared, observed order 1.16 against 1.13). The weighted variant is in
  the git history.
- **Limiters take differences `(a, b)`, not ratios**: no 0/0 on a flat field,
  no guard epsilon. Every limiter must return exactly `a` at `a == b` (second
  order in smooth flow); this is unit-tested.
- **minmod vs van Albada.** On the explicit path on TMR's C-grids minmod held
  a transonic shock where van Albada reached 2.29 orders against minmod's
  4.39 (later confounded by those grids' own limit cycle). Van Albada came
  back for RAE 2822: smooth where active, close to the central
  slope in smooth flow. Van Leer is in the git history.
- **Venkatakrishnan is deliberately absent**: it is a smooth Barth-Jespersen
  for unstructured grids, not a slope-ratio limiter; fed `r = a/b` it returns
  `phi(1) = 3/4`, quietly losing second order.

## mesh/coarsen.py

- **Grid sequencing coarsens the case's own grid** (every other node, the
  segments' indices halved), so it needs no second grid file and works for
  generated, file and multi-block grids alike. Injection (each child takes its
  parent's value) is enough: from the settled 135k M6 level, the 1.08M level
  had its forces within 1e-4 of the 3000-step values for good from step 620,
  against 1760 from freestream. The canvas padding cells must keep the
  freestream start (a zero density there fails every step's physicality test).

## core/linalg.py

- **The line solve is block cyclic reduction**, Thomas kept as the reference
  in the tests. A Thomas sweep is `2L` dependent steps per application
  (`L` the chain length), each a small kernel: on an A100 the preconditioner
  was 89% of an RAE 2822 Newton step (121.7 against 13.6 ms with full-length
  GMRES). Odd-even reduction is `ceil(log2 L)` levels for about twice the
  flops: per apply 3.52 -> 0.136 ms on RAE 2822 (flow chains of 128), 2.08 ->
  0.24 ms on M6 135k, and still faster at 1.08M cells (4.81 -> 2.89 ms) once
  the power-of-two padding was dropped (with it, 1.08M was no faster).
  `lax.scan(unroll=8)` gave only 2-4x and was dropped.

## solvers/krylov.py

- **Right preconditioning.** On TMR's 112x32 C-grid at CFL 1000, ten
  left-preconditioned GMRES iterations returned a true residual 3.1x larger
  than they started with; ten right-preconditioned ones 0.47x (0.18x with the
  lines continued through the wake cut).

## mesh/atlas.py

- **Line chains stop at other blocks.** Continued through every interface,
  30P30N L1 ran 2.1x slower per step (chains of 190 cells instead of 66) for
  no convergence gain; only a block's own j-cut (the C-grid wake) is crossed.

## mesh/highlift.py, mesh/blocking.py (multi-element grids)

Measured on the 30P30N's elements at 150 wall cells per chord (75.9k cells
main + flap, 113k all three), quality as `tests/unit/test_highlift.py`
measures it.

- **One chain template, not three.** An O-layer's outer contour is rounded at
  every trailing edge (and the 30P30N's are blunt), so a cut leaving the nose
  tip meets the contour tangentially: blocks cannot be split there. The
  families are compositions per element pair: a slot block over the
  downstream element (left side the gap cut and the upstream base, top the
  upstream wake), a cove block (upstream lip to base, gap cut, downstream
  nose, mouth), a drop from every lip. Fill blocks have `j` from a wall where
  they touch one.
- **The gap is a side, never a chain member.** With the gap cut in the
  chain of one block over everything, 80 i-lines packed into the 0.005 gap
  ran up to the middle loop and Winslow's `psi` lost authority there (cells
  of 0.004-0.007 against 3e-4 wanted, 95x across the cut). As a side of both
  the cove and the slot it is 5-6x at worst, median 3.
- **Wakes are offsets of the slot's floor.** A Hermite wake from a slat's
  trailing edge cannot follow the main element's surface; the floor (the
  downstream contour and wake) offset node for node, by the gap growing 10 %
  per unit length, does, and gives the slot straight near-normal lines. The
  normals are smoothed over two offsets and the curve over one, then the
  nodes put back at their arc fractions (averaging alone slid them: 12.7x
  at the main element's base); unsmoothed, the flap's base creased the main
  wake (17-24x across it).
- **Connectors fixed, each block smoothed alone.** Smoothing all blocks at
  once with the connectors' nodes free (stencils across blocks) let corners
  collapse where three blocks meet (the slat cusp folded); anchoring the
  cuts' ends folded where anchored met free. Fixed Hermite cuts with spacing
  matched at their ends: no fold in any family.
- **Spacing control, iterated.** Thomas-Middlecoff `psi` interpolated from a
  block's sides ignores the layer it sits on (first cells 10-20x the layer's
  last step mid-block). `psi` from the heights wanted off the bottom and top
  (per j line, from its end nodes alone) fixed the medians; three
  GRAPE-style rounds (heights rescaled by sqrt(wanted/got)) brought the
  concave-corner peak from 26.6x to 22.8x and medians to 1.1-1.4; more
  rounds skewed the cove and slot (12 rounds: 11 deg) and cost 100 s.
- **Near-wall layer.** Hyperbolic marching folds into a concave corner
  (the main element's cove step) at any thickness above about a wall cell:
  its smoothing is raised 31x there, ramped from zero at the wall so the
  first steps stay normal; a residual fold halves the thickness round it and
  marches again. A sharp convex corner (the slat cusp, 18 deg) twisted its
  first cells until the corner's two neighbours were placed equally far
  (the step is normal to the chord through them). The march's smoothing at
  the wall tilted the first step 5-8 deg on the slat and main noses; from
  zero at the wall (except within two nodes of a convex corner, whose fan
  needs it) the wall is orthogonal to 0.05 deg. The first cell is
  re-marched until within 1 % of the wall spacing (up to 6 passes; the
  marching's area smoothing moved it 3-12 %).
- **Sharp trailing edges.** A sharp trailing edge is the layer's seam node:
  the convex-corner test (heading jumps, unwrapped) missed it, the wall got
  no smoothing there and the layer folded, then was thinned to the floor
  and refused (every sharp main element with a flap). Turning is taken from
  each node's two edges instead, and the edge's two neighbours placed equally
  far, as at the slat cusp. NACA-based sections then build at 22-38 deg.
- **Targets not met yet.** Smallest angle 20 deg: 17.4 deg at the
  main element's concave cove corner (one layer cell), 12.9 deg where the
  cove narrows into the flap gap (66 cells), 16.4 deg where the slot's left
  side turns from the gap cut onto the upstream base. Interface growth 1.3:
  22.8x at the concave corner (the layer is thinned there and the cove's
  lines fan), 5-7x round the gap; medians 1.1-1.4. Tried and dropped:
  splitting the cove at its corner (to the downstream nose in proportion, or
  at its nearest point: slivers of 0.1-7.5 deg), and starting the slot where
  the channel narrows to three gaps (its left side then ran along the
  ceiling, parallel to its floor, and folded).

## 3D

- **Face vectors from the diagonals' cross product**, `S = (x_c - x_a) x
  (x_d - x_b) / 2`: the vector area of every surface the face's edges bound,
  so a cell's six vectors sum to zero exactly on warped faces and a
  collapsed face (a pole) is exactly zero. Volumes `(1/3) sum S . x_f`
  equal the trilinear hexahedron's (tested against Gauss quadrature).
- **Face gradients: the 2D diamond generalized** (centre-to-centre vector
  plus the face's two diagonals, node values and positions both averaged
  from the 8 surrounding centres), not Green-Gauss on a dual cell: exact
  for linear fields on any hexahedra, the same property and code as 2D.
- **The pole is an interface**, `i_min` facing itself half a turn round the
  O seam: the ghost across the axis is the diametrically opposite cell. Its
  faces have zero area, so it only feeds slopes and gradients.
- **Plane-reflection ghosts on skewed cells** (walls, symmetry; 2D and 3D):
  the face state is off by O(h tan theta) on a cell skewed by theta, an
  O(1) local truncation error. Measured with the 3D MMS on a warped cube:
  first order in the solution with Dirichlet ghosts placed that way, second
  order with ghosts at the cells of linearly extrapolated nodes (a point
  reflection through the face midpoint). Kept for the physical conditions
  (standard; the M6 wall is within 2 deg of orthogonal except at the sharp
  trailing edge and the leading-edge seam); the MMS uses the latter.
- **3D incidence is in x-z, span y** (TMR's wing grids): `freestream_primitive(flow, 3)`
  is `M (cos a, 0, sin a)`; a 2D grid extruded in z needs the freestream
  passed explicitly (tests/verification/test_extruded.py does).
