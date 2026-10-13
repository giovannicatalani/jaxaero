# JAX-Aero

A steady airfoil and wing solver written in JAX: 2D, and 3D on single-block
grids. It solves the compressible Euler, laminar Navier-Stokes and Spalart-Allmaras RANS equations with a
cell-centred finite-volume method on structured grids, in double precision,
on CPU or GPU.

You can mesh a single airfoil from its coordinate file, or read any
structured grid (Plot3D or CGNS, one block or many), so multi-element
high-lift sections work too; a 3D wing is a one-block Plot3D grid
([docs/wing-3d.md](docs/wing-3d.md)). A case is two JSON files; you don't write any
code. Results are checked against published data from NASA, AGARD and JAXA
([validation/gallery](validation/gallery/README.md)).

## Install

```bash
pip install -e ".[plot,cgns]"
```

For a GPU, install jax with CUDA support first
([instructions](https://docs.jax.dev/en/latest/installation.html)).

## Run something

```bash
jaxaero examples/naca0012_euler
python -m jaxaero.plot examples/naca0012_euler/results/naca0012_euler
```

## Run your own airfoil

1. Make a directory with your coordinate file (Selig or Lednicer `.dat`, as
   downloaded) and copy `case.json` and `numerical.json` from
   `examples/clarky_sa`.
2. Set `domain.airfoil` to your file name, and set the freestream.
3. `jaxaero my_airfoil --check` to see the grid, then `jaxaero my_airfoil`.

The full walkthrough is [docs/own-airfoil.md](docs/own-airfoil.md). For a
slat-main-flap section or any grid from your own mesher, see
[docs/multi-element.md](docs/multi-element.md).

## Documentation

- [Running a case](docs/guide.md): examples, command line, output, polars, boundary conditions
- [Your own airfoil](docs/own-airfoil.md), [a multi-element airfoil](docs/multi-element.md) and [a 3D wing](docs/wing-3d.md)
- [Case reference](docs/case_reference.md): every key of the two files
- [Equations and method](docs/theory.md)
- [Developer guide](docs/developer.md)
- [Validation](validation/README.md)

`mkdocs serve` (after `pip install -e ".[docs]"`) shows the docs as a website.

## Repository

```
src/jaxaero/   the solver
docs/          documentation
examples/      example cases
validation/    cases against published data, reference data, gallery
tests/         unit, verification and validation tests
scripts/       benchmark and cluster job scripts
data/grids/    downloaded grid files (not tracked; see its README)
```

Licensed under the GNU GPL v3.
