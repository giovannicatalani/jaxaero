# Developer guide

## Layout

Each layer only imports from the layers below it.

```
simulation.py, __main__.py   load a case, build the operator, iterate, write the result
polar.py      many angles at once: the step mapped over the freestream, split over GPUs
plot.py       figures from written results
case/         schema (every key, declared once), loader, domain (where the grid comes from)
io/           result files and provenance
post/         forces: Cp, Cf, CL, CD, CM, per body (2D and 3D)
solvers/      explicit (Runge-Kutta), implicit (Newton-Krylov), krylov (GMRES),
              preconditioners (line-implicit), driver (the iteration loop, batches, GPUs)
numerics/     residual, reconstruction, limiters, Riemann solvers, gradients,
              boundary conditions, SA source terms
physics/      Euler, Navier-Stokes and SA pointwise functions
mesh/         blocks and metrics (2D and 3D), meshes (patches, bodies, wall distance),
              the atlas (all blocks packed into one array; atlas3d: one 3D block),
              Plot3D and CGNS readers, assembly (interfaces and patch names; assemble3d),
              3D sides (block3d: rectangles, O seams, poles), grid generation, quality
geometry/     airfoil coordinates: file readers, NACA 4-digit, splines
core/         state, configuration, constants, linear algebra
```

## Dimensions

One code for 2D and 3D: the dimension `d` is the grid's (`Grid.dim`). State
rows come from `core.state.Rows(d)`; physics takes a velocity vector and a
normal `(d, ...)`; the residual loops over the `d` face families
(`Grid.faces`). Write new numerics that way: never `u, v` or `sx, sy` outside
2D-only code (generators, plots). The 2D named views of a `Grid`
(`x_node`, `area`, `sx_i`, ...) exist for that 2D-only code.
`tests/verification/test_extruded.py` checks that the 3D operator contains
the 2D one.

## Adding a scheme, a boundary condition or a key

Interchangeable pieces live in registries: dictionaries from the name used in
the case files to a function. They are looked up before tracing, so jitted
code never branches on a string. To add one, write the function, add it to
the registry, and if it needs a new case key, add one line to
`case/schema.py`. Then regenerate the case reference with
`python -m jaxaero.case --write` (a test fails if you forget).

| registry | entries |
|---|---|
| fluxes, `numerics/riemann.py` | `hllc` (default), `roe` |
| limiters, `numerics/limiters.py` | `minmod` (default), `van_albada`, `none` |
| Runge-Kutta schemes, `solvers/explicit.py` | `euler`, `rk4_jameson`, `rk5_jameson` (default) |
| preconditioners, `solvers/preconditioners.py` | `line_j` (default), `none` |
| grid sources, `case/domain.py` | `o_grid`, `c_grid`, `file` |
| boundary conditions, `numerics/boundary.py` | `slip_wall`, `no_slip_wall`, `symmetry`, `far_field`, `inflow_total`, `outflow_static` |
| far-field models, `numerics/boundary.py` | `freestream` (default), `point_vortex` |

The reasons behind the defaults, and the alternatives that were tried and
dropped, are in the [design notes](dev/notes.md). Read them before changing
a default.

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests -q -m "not slow"   # about 350 tests, 10 minutes
JAX_PLATFORMS=cpu python -m pytest tests -q -m slow         # manufactured solutions, batched polars, 20 minutes
```

- `tests/unit/`: each piece on its own (fluxes, limiters, boundary
  conditions, grid generation and readers, viscous terms, SA terms, GMRES,
  preconditioners, the driver, the schema).
- `tests/verification/`: metric identities, freestream preservation, the
  implicit path, the Navier-Stokes manufactured solution, batched polars
  against single runs.
- `tests/validation/`: the validation scripts.

Tests that need downloaded grids skip when `data/grids` is missing. Run
`python validation/grids/make_grids.py` once to write the grids the
validation tests generate themselves.

Comparisons with published data are in `validation/`
(`python validation/validate.py --suite rans`), with figures in
`validation/gallery/`.
