# Changelog

## Unreleased

- `domain.trailing_edge = "close"` closes an open trailing edge so that a
  C-grid can be built round a downloaded coordinate file.
- Airfoils are scaled to unit chord with the leading edge at the origin, so
  coordinates in other units give correct coefficients.
- `jaxaero CASE --check` builds the grid, prints its patches, conditions and
  bodies, writes `grid.png` and stops.
- New examples: `clarky_sa` (a UIUC coordinate file as downloaded) and
  `30p30n_sa` (a multi-block grid from file).
- Documentation: walkthroughs for your own airfoil and for a multi-element
  airfoil.
- **3D**: one code for 2D and 3D -- the dimension is the
  grid's. Single-block 3D grids from Plot3D files (O seams and poles found
  from the coordinates, patches named by box rules with z bounds), Euler,
  laminar and SA in 3D, 3D forces (`domain.reference_area`,
  `domain.moment_centre`), ParaView output (`.vts`); `muscl_kappa` (the
  unlimited kappa = 1/3 reconstruction, as CFL3D runs the M6). Validated on the M6,
  Euler and SA (CL +0.19%, CD -1.74% vs CFL3D on the same 1.08M grid).
  Example `onera_m6_sa` (its grid in the example directory) and the
  walkthrough [docs/wing-3d.md](docs/wing-3d.md).
  2D results unchanged (to 5e-15). Limits: single-block 3D grids from Plot3D;
  `jaxaero.plot` is 2D only (3D wings: `scripts/plot_wing.py` or ParaView).
- Internal API (Python users of library internals; JSON cases unaffected):
  Riemann fluxes take an area vector `S (d, ...)`; state conversions take
  `dim`; `Grid` stores `nodes`, `centres`, `volume`, `faces`.

## 0.1.0 -- first release

2D steady compressible flow on structured grids, run from two JSON files.

- **Physics**: Euler, laminar Navier-Stokes (constant or Sutherland
  viscosity), Spalart-Allmaras RANS (SA-noft2, SA-neg).
- **Numerics**: cell-centred finite volume; MUSCL with minmod, van Albada or
  no limiter; HLLC and Roe fluxes; diamond face gradients. Explicit
  multistage Runge-Kutta, or pseudo-transient Newton-Krylov with
  right-preconditioned GMRES and a line-implicit preconditioner.
- **Grids**: O- and C-grids generated round any airfoil (NACA 4-digit or a
  coordinate file: Selig, Lednicer, plain); any structured grid from Plot3D
  (every variant) or CGNS/HDF5, single- or multi-block, interfaces found from
  the coordinates; a quality report before every run.
- **Boundary conditions per named patch**: slip and no-slip walls, symmetry,
  characteristic far field (optionally with the point vortex), total-condition
  inflow, static-pressure outflow.
- **Polars**: every angle in one batched run, split over GPUs.
- **Case files**: strict schema; unknown keys, wrong types and unimplemented
  options are errors naming the key; the reference page is generated from the
  schema.
- **Validation** against NASA TMR (NACA 0012, flat plate), NPARC/AGARD
  AR-138 (RAE 2822) and JAXA (30P30N high-lift) in `validation/`.
