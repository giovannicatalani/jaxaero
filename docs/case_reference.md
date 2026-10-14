# Case reference

Generated from `src/jaxaero/case/schema.py` (`python -m jaxaero.case --write`); do not edit. How to write a case: [the guide](guide.md).

Every section may carry a `"notes"` string. Unknown keys are errors, a missing key takes its default, and an option that is not implemented is an error listing the alternatives.

## `case.json`

### `general`

| key | type | default | meaning |
|---|---|---|---|
| `case_name` | string | required | names the output directory |
| `description` | string | `case_name` | one line, printed in the log |
| `save_path` | string | `results` | output root, relative to the case directory |

### `domain`

| key | type | default | grids | meaning |
|---|---|---|---|---|
| `grid` | string | required | all | where the grid comes from: generated round one airfoil (`o_grid`, `c_grid`) or read from a file; one of `c_grid`, `file`, `o_grid` |
| `airfoil` | string | required | `o_grid`, `c_grid` | `naca` + 4 digits (closed trailing edge; `naca0012_blunt` for the standard open one), or a coordinate file (Selig, Lednicer or two columns), scaled to unit chord |
| `cells_airfoil` | integer | required | `o_grid`, `c_grid` | cells round the airfoil |
| `cells_wake` | integer | required | `c_grid` | cells along each side of the wake cut |
| `cells_normal` | integer | required | `o_grid`, `c_grid` | cells from the wall to the far field |
| `wall_spacing` | number | required | `o_grid`, `c_grid` | first-cell height, chords; for y+ ~ 1 see docs/own-airfoil.md |
| `trailing_edge` | string | `as_given` | `o_grid`, `c_grid` | `close` closes an open trailing edge (needed for a C-grid); `as_given` keeps it, and an O-grid meshes its base; one of `as_given`, `close` |
| `far_field` | number | `100.0` | `o_grid`, `c_grid` | distance of the outer boundary from the airfoil, chords |
| `leading_edge_spacing` | number | `0.002` | `o_grid`, `c_grid` | surface cell length at the leading edge, chords |
| `trailing_edge_spacing` | number | `0.002` | `o_grid`, `c_grid` | surface cell length at the trailing edge, chords |
| `file` | string | required | `file` | the grid file: Plot3D (formatted or unformatted, either byte order, 2D or 3D, single- or multi-block, iblank, gzip) or CGNS in HDF5 storage (structured zones; `pip install h5py`) |
| `format` | string | `auto` | `file` | `auto` reads it from the contents; one of `auto`, `cgns`, `plot3d` |
| `coordinate_scale` | number | `1.0` | `file` | factor on every coordinate, to the length the Reynolds number is per (1/18 for a grid in inches with an 18 in chord) |
| `patches` | list | `[]` | `file` | rules naming the boundary faces the file does not name, first match wins: `{"name": N, "x_min": .., "x_max": .., "y_min": .., "y_max": ..}` (3D grids: also `z_min`, `z_max`), every bound optional |
| `reference_length` | number | `1.0` | all | length the force coefficients are per (3D: the moment's) |
| `reference_area` | number or null | `null` | all | 3D grids (required there): area the force coefficients are per; 2D coefficients are per unit span |
| `moment_centre` | list | `[0.25, 0.0, 0.0]` | all | point the pitching moment is about, `[x, y]` or `[x, y, z]`; nose-up positive about the spanwise axis (z in 2D, y in 3D) |

### `boundary_conditions`

One entry per patch of the grid, `"<patch>": {"type": ..., "model": ...}`; a patch left out takes the default for its name (below).

| key | type | default | meaning |
|---|---|---|---|
| `type` | string | required | the condition; one of `slip_wall`, `no_slip_wall`, `symmetry`, `far_field`, `inflow_total`, `outflow_static` |
| `model` | string | `freestream` | `far_field` only: `point_vortex` adds the compressible vortex of the current lift at the quarter chord of the longest body, for a far field nearer than ~100 chords; one model per case; one of `freestream`, `point_vortex` |

| type | condition |
|---|---|
| `slip_wall` | inviscid wall (default `wall` when inviscid) |
| `no_slip_wall` | adiabatic no-slip wall (default `wall` when viscous) |
| `symmetry` | mirror plane |
| `far_field` | Riemann invariants against the freestream (default `farfield`) |
| `inflow_total` | subsonic inflow at the freestream total pressure and temperature, in its direction (default `inflow`) |
| `outflow_static` | subsonic outflow at the freestream static pressure (default `outflow`) |

### `freestream`

| key | type | default | meaning |
|---|---|---|---|
| `mach` | number | required | subsonic |
| `alpha_deg` | number | `0.0` | incidence, degrees |
| `reynolds` | number or null | `null` | per unit length of the grid (per chord for a grid of unit chord); `null` is inviscid |
| `temperature` | number |  | absolute, in the unit of `sutherland_temperature`; Sutherland's law needs it |

### `material_properties`

| key | type | default | meaning |
|---|---|---|---|
| `specific_heat_ratio` | number | `1.4` |  |
| `dynamic_viscosity` | string | `constant` | viscous cases only; one of `constant`, `sutherland` |
| `sutherland_temperature` | number |  | 198.6 (R) or 110.3 (K) |
| `prandtl_number` | number | `0.72` | the one implemented |

### `turbulence`

| key | type | default | meaning |
|---|---|---|---|
| `model` | string | `none` | `sa_noft2`: Spalart-Allmaras without ft2, with SA-neg; one of `none`, `sa_noft2` |
| `freestream_nu_tilde_ratio` | number | `3.0` | far-field nu~/nu (3 to 5) |

## `numerical.json`

### `convective_fluxes`

| key | type | default | meaning |
|---|---|---|---|
| `riemann_solver` | string | `hllc` | one of `hllc`, `roe` |
| `limiter` | string | `minmod` | `minmod` where a shock may form; `none` for smooth flow (minmod stalls viscous convergence); `van_albada` for a shock with a boundary layer; one of `minmod`, `none`, `van_albada` |
| `muscl_kappa` | number | `0.0` | unlimited MUSCL family: 0 Fromm (the default), 1/3 third-order upwind-biased (CFL3D's default), -1 fully upwind; other than 0 only with limiter `none` |

### `time_integration`

| key | type | default | meaning |
|---|---|---|---|
| `solver` | string | `explicit` | `explicit` (multistage Runge-Kutta, local time step) or `implicit` (Newton-Krylov; use it for viscous cases); one of `explicit`, `implicit` |
| `rk_scheme` | string | `rk5_jameson` | explicit path; one of `euler`, `rk4_jameson`, `rk5_jameson` |
| `cfl` | number | `1.5` | explicit path |

### `implicit`

| key | type | default | meaning |
|---|---|---|---|
| `cfl` | number | `10.0` | starting CFL |
| `cfl_max` | number | `1000.0` | ceiling of the CFL ramp |
| `cfl_growth` | number | `1.3` | largest CFL growth per step |
| `cfl_cutback` | number | `4.0` | CFL divisor after a rejected step |
| `cfl_min` | number | `0.5` | CFL floor |
| `preconditioner` | string | `line_j` | one of `line_j`, `none` |
| `jacobian` | string | `first_order` | of the Krylov operator: `first_order` (defect correction) or `exact`; one of `exact`, `first_order` |
| `krylov_tol` | number | `0.01` | relative linear residual to reach |
| `krylov_subspace` | integer | `10` | GMRES vectors before a restart |
| `krylov_restarts` | integer | `2` | GMRES restarts |
| `krylov_success` | number | `0.5` | linear residual reduction for the CFL to grow |
| `max_relative_update` | number or null | `null` | bound on a step's relative change of density, pressure and nu~ (0.2 for SA at incidence) |

### `convergence`

| key | type | default | meaning |
|---|---|---|---|
| `max_iters` | integer | `40000` | iteration ceiling |
| `tol` | number | `1e-12` | residual drop from its running peak |
| `forces_tol` | number or null | `1e-05` | stop once CL, CD and CM each vary by less than this over the last quarter of the iterations (at least four blocks there); null: run to `tol` or `max_iters` |
| `block_iters` | integer | 200 explicit, 20 implicit | iterations per device call; forces are monitored at this interval |
