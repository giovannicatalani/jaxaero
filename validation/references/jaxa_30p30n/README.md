# 30P30N three-element high-lift airfoil (JAXA, APC-IV/V): reference data

Downloaded 2026-10-06 from JAXA's Aerodynamics Prediction Challenge pages,
<https://cfdws.chofu.jaxa.jp/apc/apc5/> (APC-V, 2019; APC-IV, 2018, carries the
same data). Experimental and geometry files are unmodified; the two
spreadsheets were converted to text with their values unmodified.

**Configuration.** 30P30N, *modified slat config F*: the McDonnell Douglas
30P30N (slat and flap at 30 deg) with the slat trailing edge thickened from
zero to 0.76 mm for the BANC slat-noise workshops, the geometry of AIAA BANC
category 7 (APC-V `config.html`). Stowed chord **c = 18 in**, the reference
length; slat chord 0.15 c, flap chord 0.30 c; moment centre (0.25 c, 0).

**Conditions** (APC-IV case definition, `case/caseI.pdf`): **M = 0.17,
Re = 1.71e6** on the stowed chord, T = 295.56 K. 2D steady task: angles
0/4/5.5/8/9.5/12/14/16/20/22/24/26 deg, with **5.5 and 9.5 required**; forces
to be reported per element.

**Angles.** The wind-tunnel data are given at the *open-air CFD-equivalent*
angle: 5.5 / 9.5 / 14 deg correspond to geometric (uncorrected) 7 / 11 / 16
deg in JAXA's LWT2 Kevlar-wall tunnel (README of the data, 2023, citing
Murayama et al. 2018). The equivalence was set with SA computations, so
comparing a computation at 5.5 deg with the data is the intended use.

| file | content |
|---|---|
| `cp_exp_aoa{5.5,9.5,14.0}.dat` | experiment, mean Cp, JAXA LWT2 Kevlar wall 2017 (AoAu 7 / 11 / 16): Tecplot text, three zones = slat (23 points), main (29), flap (17); x/c on the stowed chord |
| `cp_rans_jaxa_aoa{5.5,9.5,14.0}.dat` | JAXA's 2D RANS Cp, 739 surface points of all three elements, `x/c y/c cp` |
| `forces_rans_jaxa.dat` | JAXA's 2D RANS, total CL, CD, CM from -2 to 26 deg (sheet `30P30N_bl`, "total") |
| `2010_30p30n_thik_te.crv` | the geometry, three curves (inch), from the BANC category 7 archive (`30P30N_modified_slat_configF.tar`) |

**What the RANS data are, and are not.** The APC pages give no solver, grid
or turbulence model for the RANS files; they are JAXA's own reference
computations (FaSTAR or UPACS family, not stated). The data README calls the
angle equivalence "open-air-CFD (SA turbulence model)", so SA is likely but
not stated for these files. Treat them as one code's answer, not a
grid-converged reference. Points with two values at one angle (5.5, 7 deg)
are in the file as given.

| alpha | 5.5 | 8 | 9.5 | 14 | 24 (CL max) |
|---|---|---|---|---|---|
| CL, JAXA RANS | 2.8766 | 3.1648 | 3.3305 | 3.7766 | 4.3272 |

**Grids** (not tracked; `data/grids/README.md`): JAXA's five-level family of
2D multi-block structured grids, 117 blocks, `2Ddomain_L{1..5}_*.p3d`
(unformatted double-precision Plot3D, one k plane, inch), from
<https://cfdws.chofu.jaxa.jp/apc/apc5/external/2D_domain.tar.gz>: 63,957 /
112,474 / 260,909 / 583,226 / 1,043,636 cells, wall spacing 2.6e-4 to
6.5e-5 in, interface node mismatch up to 4e-7 in (JAXA's readme, checked on
L1 by `io/multiblock.py`: 3.6e-9 c = 6.4e-8 in; 3 closed bodies; far field
at ~107 c).

**References.**
1. Murayama, M., Nakakita, K., Yamamoto, K., Ura, H., Ito, Y., and
   Choudhari, M. M., "Experimental Study on Slat Noise from 30P30N
   Three-Element High-Lift Airfoil at JAXA Hard-Wall Low-Speed Wind Tunnel",
   AIAA 2014-2080.
2. Murayama, M., Yokokawa, Y., Ura, H., et al., "Experimental Study of Slat
   Noise from 30P30N Three-Element High-Lift Airfoil in JAXA Kevlar-Wall
   Low-Speed Wind Tunnel", AIAA 2018-3460.
3. JAXA-SP-18-008 (APC-IV) and JAXA-SP-19-008 (APC-V) proceedings, with the
   participants' results: <https://repository.exst.jaxa.jp/dspace/handle/a-is/909702>,
   <https://repository.exst.jaxa.jp/dspace/handle/a-is/948661>.

Not used here: the LTPT 30P30N data (M 0.2, Re 9e6; Klausmeyer and Lin,
NASA TM-112858) are for the sharp-slat geometry at other conditions, and no
public structured grid for them was found.
