# NASA TMR 2D zero-pressure-gradient flat plate, SA: reference data

Downloaded 2026-10-05 from <https://tmbwg.github.io/turbmodels/flatplate_sa.html>,
unmodified.

**Case:** M = 0.2, Re = 5e6 per unit length, T_ref = 540 °R, Sutherland,
Pr = 0.72, Pr_t = 0.9, ν̃ = 3 ν∞ at the inflow. Plate from x = 0 to 2. CD is
per plate length 2.

| file | content |
|---|---|
| `cf_convergence.dat` | Cf at x = 0.97008 per grid (N = cells), CFL3D and FUN3D |
| `drag_convergence.dat` | CD per grid, CFL3D and FUN3D |
| `cfl3d_typical.inp` | the CFL3D input: the boundary conditions (2009 total-condition inflow p_t/p∞ 1.02828, T_t/T∞ 1.008; 2002 outflow p/p∞ = 1; 1001 symmetry ahead of the plate; 2004 adiabatic wall; 1003 Riemann top), kappa = 1/3 MUSCL, `tur10 3.0` |

CFL3D approaches the limit from above and FUN3D from below, so a third code's
coarse-grid value should fall between theirs on the same grid.
