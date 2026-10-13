# NASA TMR NACA 0012, SA model: reference data

Downloaded 2026-10-05 from the Turbulence Modeling Resource, now at
<https://tmbwg.github.io/turbmodels/> (the old `turbmodels.larc.nasa.gov`
redirects there). The files are unmodified.

**Case:** M = 0.15, Re = 6e6 per chord, T_ref = 540 °R, Sutherland's law
(T0 = 491.6 °R, S = 198.6 °R, mu0 = 1.716e-5 kg/m/s; White 1974, p. 28),
Pr = 0.72, Pr_t = 0.9, gamma = 1.4, far-field ν̃ = 3 ν∞, fully turbulent.
Airfoil: the TMR section (`karman_trefftz.TMR`), far field ~500 chords.

| file | source page | content |
|---|---|---|
| `n0012clcd_cfl3d_sa.dat` | [naca0012_val_sa](https://tmbwg.github.io/turbmodels/naca0012_val_sa.html) | CFL3D CL, CD at 0/10/15°, 897x257, **with** point-vortex far field |
| `n0012cp_cfl3d_sa.dat`, `n0012cf_cfl3d_sa.dat` | same | CFL3D surface Cp and Cf, same runs |
| `n0012_cfl3d_typical_sa.inp` | same | the CFL3D input: kappa = 1/3 MUSCL, `tur10 3.` |
| `cfl3d_results_sa_nopv_withN.dat` | [naca0012numerics_val_sa_withoutpv](https://tmbwg.github.io/turbmodels/naca0012numerics_val_sa_withoutpv.html) | CFL3D per grid, 10°, **without** point vortex, families I-III; `N` = cells |
| `fun3d_results_sa_nopv_withN.dat` | same | FUN3D (SA-neg, 2nd-order turbulence advection), likewise; `N` = nodes |

Which numbers to compare against:

- Our far field is plain Riemann invariants, **no point-vortex correction**.
  Compare with the `nopv` files, not the validation table. Grid-converged at
  10° (family II, three codes): CL 1.09085-1.09103, CD 0.0122715-0.0122740,
  CDp 0.006066-0.006068, CDv 0.0062057-0.0062061.
- Family I has the trailing-edge streamwise spacing of the validation grids.
  Its sizes match the files in `data/grids` (225x65 nodes = 14336 cells), so
  the per-grid family I values are the like-for-like comparison on each level.
  Per-mesh, as with the Euler work: a coarse level is judged against the
  reference codes' value on that mesh, not against the continuum.
- The seven-code validation table (897x257, with point vortex) at 0/10/15°:
  CL ≈ 0 / 1.0891-1.1000 / 1.5446-1.5642, CD 0.00812-0.00830 /
  0.01225-0.01245 / 0.02073-0.02159.
