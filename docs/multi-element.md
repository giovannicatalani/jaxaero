# A multi-element airfoil

The solver does not generate grids round several bodies. For a slat, main
element and flap, or any geometry with coves and gaps, make a structured
multi-block grid with your mesher and point the case at the file. The worked
example is `examples/30p30n_sa`: the 30P30N three-element airfoil on JAXA's
117-block grid (download instructions in `data/grids/README.md`).

## 1. What the grid file must be

- **Format**: Plot3D (any variant: formatted or binary, single or double
  precision, 2D, or 3D with one plane used) or CGNS saved in HDF5. Pointwise,
  ICEM CFD and most structured meshers write one of these.
- **Blocks meet node to node.** Interfaces are found by matching coincident
  block faces, so no connectivity file is needed, but the nodes on both sides
  of an interface must coincide. Non-matching (patched) interfaces are not
  supported.
- **Every wall closes round its body**, and the outer boundary closes round
  everything.

In a CGNS file, boundary names come from the file (family names, else BC
types). Plot3D has no names, so faces that are not interfaces are named by
the loop they lie on: the outer loop becomes `farfield`, and each closed loop
round a body becomes `wall`. For other cases, such as a symmetry plane,
an inflow or an outflow, name faces with box rules in `domain.patches`
(the flat-plate example does this).

## 2. case.json

```json
{
    "general": {"case_name": "30p30n_sa"},
    "domain": {
        "grid": "file",
        "file": "data/grids/2Ddomain_L1_coarse_r1.p3d",
        "coordinate_scale": 0.05555555555555555
    },
    "freestream": {"mach": 0.17, "alpha_deg": 5.5, "reynolds": 1.71e6, "temperature": 295.56},
    "material_properties": {"dynamic_viscosity": "sutherland", "sutherland_temperature": 110.4},
    "turbulence": {"model": "sa_noft2"}
}
```

**Units.** The solver works in the length your Reynolds number is based on,
and the force coefficients use the same length. JAXA's grid is in inches,
with an 18 inch stowed chord, so `coordinate_scale` is 1/18. If your grid is
already in chords, leave it out. If you would rather keep the grid as it is
and quote coefficients per some other length, set `reference_length`; the
Reynolds number is then per that length too.

The pitching moment is taken about (0.25, 0) in the scaled coordinates.

## 3. Check the grid

```bash
jaxaero examples/30p30n_sa --check
```

```
grid file 2Ddomain_L1_coarse_r1.p3d: 117 block(s), 63957 cells, 3 body(ies); implicit, hllc, limiter van_albada
grid quality: 63957 cells, min area 8.64e-10, min angle 34.4 deg, max growth 2.16, ...; far field >= 97.6 from the walls
  warning: neighbouring cells growing by up to 2.16x (over 1.5)
patch 'wall': no_slip_wall, 610 faces
patch 'farfield': far_field, 560 faces
body 0: x -0.0848 to 0.0188, y -0.1162 to 0.0066, 266 wall faces
body 1: x 0.0438 to 0.8741, y -0.0546 to 0.0619, 254 wall faces
body 2: x 0.8714 to 1.1307, y -0.1460 to 0.0218, 90 wall faces
```

Look at three things. The number of bodies should be the number of elements:
if it is lower, a wall did not close, usually because two blocks meet with
nodes that do not coincide. The body extents should be in chords. And
`grid.png` should show every element outlined in its own colour:

![30P30N grid](figures/grid_30p30n.png)

Bodies are numbered by their leading edge, from the front: here slat 0,
main 1, flap 2.

## 4. Run

```bash
jaxaero examples/30p30n_sa
```

Use the implicit solver with `van_albada`; high-lift flows at low Mach
number converge slowly, so give it several thousand iterations (8000 here).
The coarsest 30P30N level (63,957 cells) runs its 8000 iterations in 7
minutes on an A100 (49 ms each); its forces keep moving by a little more
than the `forces_tol` bar, so it does not stop earlier. On a small GPU, run
a grid of this size on a cluster instead. Its block sizes are odd, so
`grid_sequencing` refuses it.

The forces are printed for the whole configuration and for each body. On
this coarsest level, at 5.5 degrees:

| | CL | CD |
|---|---|---|
| slat (body 0) | 0.151 | -0.065 |
| main (body 1) | 2.183 | -0.135 |
| flap (body 2) | 0.453 | 0.259 |
| total | 2.787 | 0.059 |
| JAXA's RANS, for comparison | 2.877 | 0.040 |

On the third grid level (261 thousand cells) the lift comes within 0.03 %
of JAXA's. Per-element drag is the difference of large pressure forces and
depends strongly on the grid.

`*_surface.dat` lists Cp and Cf body by body, each starting from its
trailing edge, with a `body` column to tell them apart.

## When the grid is refused

- *a face coincides with more than one other face*: two blocks overlap, or
  a block is duplicated in the file.
- *block k folds over*: that block has cells of both orientations; fix it in
  the mesher.
- *no boundary condition for patch(es) [...]*: the file names a boundary the
  case does not mention, and the name has no default. Add it to
  `boundary_conditions`; the message lists the grid's patch names.
- *a 1-to-1 interface the file declares is not matched*: a CGNS file says two
  faces connect, but their nodes do not coincide.
