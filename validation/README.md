# Validation

Everything needed to reproduce published results is here. The solver in
`src/jaxaero` knows nothing about these cases.

```
cases/<name>/        case.json + numerical.json, plus reference.json
references/<source>/ published data, each with a README naming the source
gallery/             README.md with every result and run time, and the figures
validate.py          runs a case on each of its grid levels and compares with the reference
compare_surface.py   wall Cp and Cf against published distributions
report.py            builds gallery/ from results/
```

The results are in [gallery/README.md](gallery/README.md). Raw run output
goes to `results/` at the repository root, which is not tracked.

A validation case is a normal case directory (run it with
`jaxaero validation/cases/<name>`) plus a `reference.json` that only the
scripts here read:

| key | meaning |
|---|---|
| `suite`, `order` | the suite, and the case's place in it |
| `levels` | grid levels, coarsest first: each a set of `domain` keys merged into `case.json` |
| `reference` | sourced CL / CD (`cl: null` if none quoted), optional Cf at one station (`cf_x`, `cf`), spread between sources, tolerances (relative; absolute for a zero reference), `isolates`, `sources`, `notes` |
| `settled_cl_spread` | largest CL range over the last quarter of blocks that counts as settled |
| `gallery` | present: the case is in the gallery; `window` (field plot, chords), `surface_reference` (a `compare_surface.py` set, e.g. `tmr_naca0012:10`), `fields` (false: no field plots) |

```bash
python validation/validate.py --suite ladder                 # in order, stops at the first failure
python validation/validate.py hiocfd_c12 --levels 0,1,2      # by index
python validation/validate.py tmr_naca0012_sa_a10 --levels 449-129,897-257   # or by grid name
python validation/validate.py tmr_flatplate_sa --grids-only
python validation/compare_surface.py results/validation/tmr_naca0012_sa_a10_n896 --reference tmr_naca0012:10
python validation/report.py                                  # results/ -> gallery/
```

Results go to `results/validation/<case>_n<cells around>/`, with the judgement
(`cl_spread`, `settled`, `reference`) added to the metadata. Reference data and
their sources are in `references/`.

## Suites and results (Oct 2026)

| suite | case | source | finest result |
|---|---|---|---|
| ladder | `vj_m05a0`, `vj_m05a125`, `vj_m08a0`, `vj_m08a125` | Vassberg & Jameson (2010), Euler | second order; CD 6.52e-4 at 256² vs CFL3D 5.73e-4 (M 0.5, a 0) |
| laminar | `hiocfd_c12` | HiOCFD workshop C1.2 | CD 0.05600 at 256x352, 1.26% from the DG reference (accepted) |
| topology | `tmr_laminar_cgrid`, `tmr_laminar_ogrid` | cross-check, no reference | C- and O-grid CD extrapolations agree to 0.6% |
| rans | `tmr_flatplate_sa` | NASA TMR 2DZP | Cf(0.97) 2.70546e-3 vs 2.70562e-3 (CFL3D, 545x385) at 137x97 |
| rans | `tmr_naca0012_sa_a0` / `_a10` / `_a15` | NASA TMR NACA 0012 | at 897x257: CD 0.008120; CL 1.09076, CD 0.012349 (CFL3D same grid 1.09112, 0.012371); CL 1.54411, CD 0.021383 |
| transonic | `nparc_rae2822_case6` | AGARD AR-138 case 6; WIND SA on the NPARC grid | 369x65, van Albada, point-vortex far field: CL 0.72118, CD 0.012452, shock at 0.5455 c (WIND same grid: 0.73096, 0.012117, 0.5455; experiment 0.743, 0.0127, 0.578) |

Per-level tables are in the [gallery](gallery/README.md).
