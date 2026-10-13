# Running a case

A case is a directory with two files, following
[JAX-Fluids](https://github.com/tumaer/JAXFLUIDS): `case.json` describes the
problem (grid, freestream, gas, turbulence model, boundary conditions) and
`numerical.json` the method (flux, limiter, solver, iterations). Every key is
listed in the [case reference](case_reference.md). To set up your own
geometry, see [your own airfoil](own-airfoil.md),
[a multi-element airfoil](multi-element.md) or [a 3D wing](wing-3d.md).

## Examples

| example | flow | grid |
|---|---|---|
| `naca0012_euler` | Euler, M 0.8, 1.25 deg | generated O-grid |
| `naca0012_laminar` | laminar, M 0.5, 1 deg, Re 5000 | generated C-grid |
| `clarky_sa` | SA RANS, M 0.2, 4 deg, Re 3e6 | C-grid round a downloaded coordinate file |
| `naca0012_sa` | SA RANS, M 0.15, 10 deg, Re 6e6 | NASA TMR C-grid from file |
| `rae2822_transonic_sa` | transonic SA RANS, M 0.729, 2.31 deg, Re 6.5e6 | NASA NPARC C-grid from file |
| `flatplate_sa` | SA RANS boundary layer, M 0.2, Re 5e6 | NASA TMR flat-plate grid |
| `30p30n_sa` | three-element high lift, M 0.17, 5.5 deg, Re 1.71e6 | JAXA 117-block grid from file |

The grids read from files are not in the repository; `data/grids/README.md`
says where to download each. The first three examples need nothing else.

## Running

```bash
jaxaero examples/naca0012_euler              # or: python -m jaxaero examples/naca0012_euler
jaxaero examples/naca0012_euler --check      # build and draw the grid, do not solve
jaxaero examples/naca0012_sa --set case.freestream.alpha_deg=12
jaxaero my_case --out /scratch/me/run1 --quiet
```

`--set file.section.key=value` changes one key for this run, and can be
repeated. From Python:

```python
from jaxaero import load_case, run

case = load_case("examples/naca0012_sa", overrides=("case.freestream.alpha_deg=12",))
result = run(case)
result.write()
print(result.forces.cl, result.forces.cd, result.history.orders_dropped)
```

The validation cases in `validation/cases/` run the same way.

## Output

Results go to `<case>/results/<case_name>/`, or to `--out`:

| file | content |
|---|---|
| `*_fields.npz` | the solution `U` and the grid nodes, per block |
| `*_surface.dat` | x, Cp and Cf on every wall face; with several bodies also y and a `body` column |
| `*_residual.dat`, `*_lift.dat` | convergence histories |
| `*_metadata.json` | forces, convergence, the case files as run, git hash, versions, device |

`python -m jaxaero.plot <result dir>` writes figures to `figures/` there:
Mach and Cp fields, wall Cp and Cf, convergence. It only reads these files,
so you can plot results copied back from a cluster without a GPU.

`U` is conservative, `(rho, rho u, rho v, E)`, plus `rho nu~ / nu_inf` with
SA. Everything is nondimensional, based on freestream density, freestream
speed of sound and the chord, so the freestream is `rho = 1`,
`(u, v) = M (cos a, sin a)`, `p = 1/gamma`.

## Polars

```bash
python -m jaxaero.polar examples/rae2822_transonic_sa --alpha -2 5.5 0.5
```

`--alpha` takes start, stop and step. All the angles run together as one
batch, split over every visible GPU. The output is `polar.dat` (one row per
angle), `polar_metadata.json` and a normal result directory per angle.

## Boundary conditions

Every grid has named patches on its boundaries, and `boundary_conditions`
gives each one a type:

```json
"boundary_conditions": {
    "wall": {"type": "no_slip_wall"},
    "farfield": {"type": "far_field", "model": "point_vortex"}
}
```

The types are `slip_wall`, `no_slip_wall` (adiabatic), `symmetry`,
`far_field` (characteristic; `model` is `freestream` or `point_vortex`),
`inflow_total` and `outflow_static`.

A patch you leave out gets the default for its name: `wall` is no-slip in a
viscous case and slip otherwise, `farfield` is `far_field`, `symmetry` is
`symmetry`, `inflow` is `inflow_total`, `outflow` is `outflow_static`. Any
other name must be given a type.

Where the patch names come from:

- **Generated grids** have `wall` and `farfield`.
- **CGNS files**: the family name of each boundary, else its BC type
  (`BCWall*` is `wall`, `BCFarfield` is `farfield`, and so on), else the
  name of the BC node.
- **Faces a file does not name** (all of them, in Plot3D) are named by
  `domain.patches` box rules if you give any, else by the loop they lie on:
  the outer boundary is `farfield`, each body is `wall`.

Block interfaces, the seam of an O-grid and the wake cut of a C-grid are
found from coincident faces. They are not patches, and you never declare
them.

CGNS files in the older ADF format are refused. Convert them with the CGNS
tools: `adf2hdf in.cgns out.cgns`.
