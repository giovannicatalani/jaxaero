# JAX-Aero

A steady airfoil and wing solver in JAX: compressible Euler, laminar
Navier-Stokes and Spalart-Allmaras RANS on structured grids, in 2D and, on
single-block grids, in 3D, on CPU or GPU. Grids are generated round a single
airfoil from its coordinates, or read from Plot3D and CGNS files with any
number of blocks (3D: one Plot3D block). A case is two JSON files.

```bash
pip install jax-aero[plot,cgns]
jaxaero my_case --check     # build and draw the grid
jaxaero my_case             # solve
python -m jaxaero.polar my_case --alpha -2 10 1
```

- [Running a case](guide.md)
- [Your own airfoil](own-airfoil.md)
- [A multi-element airfoil](multi-element.md)
- [A 3D wing](wing-3d.md)
- [Case reference](case_reference.md)
- [Equations and method](theory.md)
- [Developer guide](developer.md) and [design notes](dev/notes.md)

Licensed under the GNU GPL v3.
