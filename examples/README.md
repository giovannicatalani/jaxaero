# Examples

| example | flow | grid |
|---|---|---|
| `naca0012_euler` | Euler, M 0.8, 1.25 deg | generated O-grid |
| `naca0012_laminar` | laminar, M 0.5, 1 deg, Re 5000 | generated C-grid |
| `clarky_sa` | SA RANS, M 0.2, 4 deg, Re 3e6 | C-grid round a downloaded coordinate file |
| `naca0012_sa` | SA RANS, M 0.15, 10 deg, Re 6e6 | NASA TMR C-grid from file |
| `rae2822_transonic_sa` | transonic SA RANS, M 0.729, 2.31 deg, Re 6.5e6 | NASA NPARC C-grid from file |
| `flatplate_sa` | SA RANS boundary layer, M 0.2, Re 5e6 | NASA TMR flat-plate grid |
| `30p30n_sa` | three-element high lift, M 0.17, 5.5 deg, Re 1.71e6 | JAXA 117-block grid from file |
| `main_flap_sa` | two-element high lift (30P30N main and flap), M 0.17, 5.5 deg, Re 1.71e6 | generated multi-element grid |
| `onera_m6_sa` | 3D wing, transonic SA RANS, M 0.84, 3.06 deg, Re 14.6e6 | NASA TMR one-block 3D grid, in the example directory |

Run one with `jaxaero examples/naca0012_euler`. The grid files of
`naca0012_sa`, `rae2822_transonic_sa`, `flatplate_sa` and `30p30n_sa` are
downloaded separately (`data/grids/README.md`); `onera_m6_sa` carries its own.

`clarky_sa` is the template for your own airfoil
([docs/own-airfoil.md](../docs/own-airfoil.md)); `main_flap_sa` for a slat,
main element and flap from their coordinates, `30p30n_sa` for a grid from
your own mesher ([docs/multi-element.md](../docs/multi-element.md));
`onera_m6_sa` for a 3D wing ([docs/wing-3d.md](../docs/wing-3d.md)).
