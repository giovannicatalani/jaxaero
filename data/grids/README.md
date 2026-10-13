# Third-party grids (not tracked)

Input grids from public sources, kept locally, never in git
(see `.gitignore`). Every case that reads one names it in its `case.json`;
grids the library generates (`o_grid`, `c_grid`) are never stored.

The validation cases' Karman-Trefftz NACA 0012 O-grids (`kt_*.p2dfmt`) are
not third-party but are files too: `python validation/grids/make_grids.py`
writes them here (the generator, `validation/grids/karman_trefftz.py`, is
validation code, not library).

| files | source | used by |
|---|---|---|
| `n0012_113-33` ... `n0012_1793-513` (`.p2dfmt[.gz]`, `.nmf`, CGNS) | NASA TMR NACA 0012 grids, family I: <https://tmbwg.github.io/turbmodels/naca0012_grids.html> | `tmr_naca0012_sa_*`, `tmr_laminar_cgrid` |
| `flatplate_clust2_*levelsdown_*.p2dfmt` (35x25, 69x49, 137x97) | NASA TMR 2DZP flat plate: <https://tmbwg.github.io/turbmodels/flatplate_grids.html> (finer levels: the zip there) | `tmr_flatplate_sa` |
| `rae2822_nparc.x.fmt` (369x65 C-grid) | NASA NPARC archive, RAE 2822 study 1: <https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf01/raetaf01.html> (`raetaf.x.fmt`) | `nparc_rae2822_case6` |
| `2Ddomain_L{1..5}_*_r1.p3d` (30P30N, 117 blocks, 64k-1.04M cells) | JAXA APC-V, 2D section grids: <https://cfdws.chofu.jaxa.jp/apc/apc5/external/2D_domain.tar.gz> (`2D_domain/plot3d/`) | `jaxa_30p30n_*` |
| `onera_m6/om6_strct_{4x11x6 .. 64x176x96}.ufmt` (ONERA M6, one 3D block, 264-1.08M cells) | generated with NASA TMR's program, see below | `tmr_onera_m6_*` |
| `onera_m6/om6_sample_40x64x48.ufmt` (+ `.nmf`, `.mapbc`) | the same program's sample input (`input.nml_strct`, y+ 1): small, for tests | `tests/unit/test_grid3d.py`, `test_freestream3d.py` |

TMR airfoil grids: C-grids; the wrap index runs downstream wake -> trailing
edge -> lower surface -> leading edge -> upper surface -> trailing edge ->
wake, and `j` increases outward. Wall spacing ~3.7e-6 chords at 225x65, far
field ~500 chords. The `.nmf` files give the boundary topology and are
cross-checked against the coordinates by `mesh/plot3d.py`.

The NPARC RAE 2822 grid has the same index layout (trailing edge at I33 and
I337, 1-based; wall spacing 1e-5 c; far field 20-38 chords), but its wake cut
follows the trailing-edge bisector to `y = -1.43` at 27 chords. `mesh/assemble.py`
finds the wake cut by matching coincident faces, so neither layout is
assumed.

JAXA's 30P30N grids are multi-block, unformatted double-precision Plot3D
(one `k` plane), in inches (stowed chord 18 in): cases scale them with
`domain.coordinate_scale` = 1/18. No map file: `mesh/assemble.py` infers the
block connectivity by matching faces (L1: 482 interface segments, node
mismatch 6.4e-8 in as JAXA states, walls closing into 3 bodies). Sources and
reference data: `validation/references/jaxa_30p30n/README.md`.

`n0012_113-33_hdf5.cgns` is TMR's `n0012_113-33.cgns` converted from ADF to
HDF5 (`cgnsconvert -h`, CGNS tools from conda-forge `cgns`): the reader
handles HDF5 CGNS only. `tests/unit/test_grid_files.py` checks it against
the Plot3D file of the same grid.

The ONERA M6 grids are not downloaded: TMR ships the generator, not the
grids. Its release <https://www.nasa.gov/wp-content/uploads/2026/04/wing-release-072319.zip>
was built with gcc 12.2 (`gfortran -O2 -o hcf_wing
hcf_wing_v5p0.f90`, likewise `hcf_coarsening_v3p9.f90`) and run with TMR's
`input.nml_wing_strct_stt` and `input_coarsen.nml_wing_strct_stt`
(<https://tmbwg.github.io/turbmodels/onerawingnumerics_grids.html>): 12 min
for the 69M-cell grid and its six coarsenings, 27 GB in all, on one core
Generator level `n` (1 finest) is written
`wing_strct.n.ufmt`; renamed here by cells: level 4 = `32x88x48` (135,168,
TMR's coarsest), level 3 = `64x176x96` (1,081,344). Levels 1 (69M, 1.7 GB)
and 2 (8.65M, 211 MB) are not kept here: regenerate them the same way.
Unformatted Plot3D, one block, nodes `(ni+1, nj+1, nk+1)`: `i` from the tip
pole to the root symmetry plane, `j` from the wall to the hemispherical far
field (100 root chords), `k` round the section (an O, starting at the leading
edge); root chord 1, semispan 1.476. `mesh/assemble3d.py` finds the O seam and
the pole from the coordinates; the patches are named by `domain.patches` rules.
