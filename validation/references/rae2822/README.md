# RAE 2822, AGARD AR-138 case 6: reference data

Downloaded 2026-10-05 from the NASA NPARC Alliance Validation Archive, RAE 2822
transonic airfoil, study #4 (WIND 5.0 validation, J. W. Slater):
<https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf04/raetaf04.html>.
The files are unmodified.

**Case:** AGARD AR-138 (Cook, McDonald and Firmin, 1979) case 6 at the
wind-tunnel-*corrected* conditions NPARC runs: M = 0.729, alpha = 2.31 deg
(nominal 0.725 / 2.92), Re = 6.5e6 per chord, T_inf = 460 °R, fully
turbulent. Grid: the NPARC 369x65 C-grid (`data/grids/rae2822_nparc.x.fmt`),
trailing edge at I33 / I337, wall spacing 1e-5 c, far field 20-38 chords.
WIND imposed freestream static pressure on the two outflow planes; this code
uses Riemann invariants there.

| file | content |
|---|---|
| `cp.exp.gen` | experiment, case 6: **-Cp**, one loop (upper surface TE -> LE, then lower LE -> TE), 103 points |
| `cp.wind.gen` | WIND, SA, 369x65: **-Cp** at the 305 wall nodes, TE -> lower -> LE -> upper |
| `cp.nparc.gen` | the older NPARC code's Cp (as plotted by NPARC; not used) |
| `cf.wind.gen` | WIND "Cf": values ~8e-6, not skin-friction coefficients (CFPOST units); **unusable** |
| `forces.lis` | WIND's integrated forces on this grid (below) |
| `run.sa.dat` | WIND's input: SA, CFL 5, dq limiter, 9 orders |
| `geom.txt` | the measured ordinates (Table 6.1 of AR-138) |

Both Cp files store **-Cp** (`cfpost.cp.com`: `variables x; Cp scale -1`);
`validation/compare_surface.py` negates them on reading.

**Forces.**

| source | CL | CD | CD pressure | CD friction | CM (c/4, nose-up +) |
|---|---|---|---|---|---|
| WIND, SA, this grid | 0.73096 | 0.012117 | 0.006200 | 0.005916 | -0.0901 |
| experiment (AR-138, corrected) | 0.743 | 0.0127 | | | -0.095 |

WIND's `CMZ` = +0.0901 is about the quarter chord with its own sign
(nose-down positive); with ours (nose-up positive) it is -0.0901.
Experiment: Cook et al. (1979) via e.g. CEAS Aeronautical Journal 11 (2020)
33-47, <https://link.springer.com/content/pdf/10.1007/s13272-019-00389-y.pdf>.

**Shock position** (upper surface, where Cp first rises through
Cp* = -0.6658, the sonic value at M 0.729; `compare_surface.py`):
experiment x/c = 0.578, WIND 0.546. The references themselves differ by
3% chord.

Which numbers to compare against: WIND on the same grid is the like-for-like
per-mesh reference (as CFL3D was for TMR). The experiment carries
wind-tunnel corrections and a transition strip the computation does not
model; on this grid WIND's SA shock is 3.2% chord upstream of it.

WIND's input (`run.sa.dat`) names no far-field correction, but our solution
matches WIND's only with one: on this grid, with van Albada, the plain
characteristic far field gives CL 0.6972 and the shock at 0.540; with the
point-vortex far field CL 0.7212 and the shock at 0.5455, WIND's to four
digits, and the Cp rms difference falls from 3.4e-2 to 7.0e-3 (upper
surface). Treat WIND's far-field model as unknown but effectively
circulation-corrected.
