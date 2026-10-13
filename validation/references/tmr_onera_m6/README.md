# ONERA M6 wing: reference data (NASA TMR and NPARC)

Downloaded 2026-10-07. The files are unmodified.

**Case 2308** (TMR's): M 0.84, alpha 3.06 deg, Re 14.6e6 per **root** chord
(the grids have root chord 1), T 540 °R, Sutherland, Pr 0.72, Pr_t 0.9,
fully turbulent, far-field ν̃ = 3 ν∞, SA-neg. Sharp trailing edge, rounded
tip. **Case 2565**: the same at alpha 6.06 deg (separation behind the shock).
The classic Euler conditions (NPARC/WIND) are M 0.8395, alpha 3.06 deg.

Coefficients (CFL3D's input, `cfl3d_OM6sampleinput_L2.inp`): reference area
`sref = 1.15315084119` (the trapezoidal semispan), length `cref =
0.8016729585` (the mean aerodynamic chord), moment about the origin (the root
leading edge), `CMY` = the moment about +y by the right-hand rule, so
negative is nose-down -- the convention of `post/forces.py` -- pressure and
friction. Semispan `b = 1.4760179762`; span stations `eta = y / b`.

| file | source page | content |
|---|---|---|
| `combined_forces_pitchmom_maxmut.dat` | [onerawingnumerics_val_sa](https://tmbwg.github.io/turbmodels/onerawingnumerics_val_sa.html) | CL, CD, CDp, CDv, CMY per grid for CFL3D (structured), FUN3D, USM3D; `N` = cells for the structured family |
| `CFL3D_OM6_A3p06_CPCF.dat` | same | CFL3D Cp and skin friction at the 7 stations, levels L1 (finest, 69M cells) .. L4 (coarsest, 135k) |
| `cfl3d_OM6sampleinput_L2.inp` | same | CFL3D's input on L2: conditions, reference values |
| `case_2308.dat`, `case_2565.dat` | [onerawingnumerics_val](https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html) | Schmitt and Charpin, AGARD AR-138 (1979), runs 308 and 565: Cp at the taps of 7 sections, `X/L` = x over the local chord; accuracy ±0.02 at M 0.84 |

Experimental sections: `eta = 0.20, 0.44, 0.65, 0.80, 0.90, 0.96, 0.99`
(0.96, not 0.95: corrected by ONERA). The experiment's Reynolds number is
11.72e6 per mean aerodynamic chord; TMR prescribes 14.6e6 per root chord for
the computations, as here.

CFL3D per level, case 2308 (the structured family, each level the next
finer coarsened by 2 in every direction):

| level | cells | CL | CD | CDp | CDv | CMY |
|---|---|---|---|---|---|---|
| L4 | 135,168 = 32 x 88 x 48 | 0.261198 | 0.0190461 | 0.0128390 | 0.0062071 | -0.184332 |
| L3 | 1,081,344 | 0.266033 | 0.0173521 | 0.0118074 | 0.0055447 | -0.186608 |
| L2 | 8,650,752 | 0.266410 | 0.0169172 | 0.0115981 | 0.0053192 | -0.186594 |
| L1 | 69,206,016 | 0.269315 | 0.0169562 | 0.0116536 | 0.0053026 | -0.189760 |

Judge each level against CFL3D on the same grid (as for the NACA 0012).
FUN3D on 60.8M nodes: CL 0.269546, CD 0.0169473, CMY -0.190665.

Grids: generated with TMR's own program (`hcf_wing_v5p0.f90`, input
`input.nml_wing_strct_stt`) and coarsened with `hcf_coarsening_v3p9.f90`;
see `data/grids/README.md`.
