# ML_rep_alchemical

The 8-descriptor "alchemical" representation of each molecule's GCMC simulation model,
built from `../Structures/gcmc/*.xyz`.

    alchemical_gcmc.csv       the representation (one row per molecule)
    build_alchemical.py       builds it (method in its docstring)
    forcefield/               RASPA files used in GCMC (GenericMOFs_corrected):
                              force_field_mixing_rules.def (epsilon, sigma), pseudo_atoms.def (masses, charges)

## Expected layout

The script reads the GCMC structures from a `Structures/gcmc/` folder that sits **next to**
this folder (path resolved relative to `build_alchemical.py`, so it runs from any directory):

    Representations/
      Structures/
        gcmc/
          <id>.xyz             one file per molecule; <id> becomes the row id (e.g. co2.xyz)
      ML_rep_alchemical/
        build_alchemical.py
        forcefield/
          GenericMOFs_corrected/   force_field_mixing_rules.def, pseudo_atoms.def
        alchemical_gcmc.csv    written here

Every `*.xyz` file in `Structures/gcmc/` is processed. Each must have the gcmc format
(line 2 = `<name> | element  x [Angstrom]  y [Angstrom]  z [Angstrom]  label  charge [e]  mass [g/mol]`);
only the element, x, y, z and label columns are read. Every label must exist in the
forcefield files, otherwise the build stops and names the missing type.

## Requirements

Python 3 and numpy only; everything else is the Python standard library (`csv`, `pathlib`).
No internet access, no other packages. Tested with Python 3.10.12 and numpy 2.2.6.

    pip install numpy

## Run

    python build_alchemical.py

The calculation is deterministic (no random numbers): the same structures and forcefield
files always give the same CSV.

**Q and fle are filled by hand** (RASPA 2.0 values): replace the `###` cells directly in
the CSV. Rebuilding keeps whatever you typed; only `###` cells stay `###`.

| column | descriptor | unit | computed from |
|---|---|---|---|
| epsi_eff_K | sum of LJ epsilon over all sites | K | force_field_mixing_rules.def |
| a_A | largest extent along the principal axes | Angstrom | XYZ + sigma/2 radii + pseudo_atoms.def masses |
| elo | a / b | - | |
| fla | b / c | - | |
| c_A | smallest extent (effective kinetic diameter) | Angstrom | |
| mu_D | dipole, \|sum q_i r_i\| | Debye | pseudo_atoms.def charges |
| Q | quadrupole (RASPA 2.0) | as entered | by hand (`###`) |
| fle | ln Rosenbluth weight (RASPA 2.0) | - | by hand (`###`) |

Every site is looked up by its TraPPE type (the `label` column of the gcmc XYZ), so editing the
files in `forcefield/` carries straight through to the representation.

## How a, b, c are computed

The molecule is treated as a set of hard spheres, and a, b, c are its widths measured
along three perpendicular directions: the principal axes of inertia. Step by step:

1. **Sites.** Take every site with a Lennard-Jones term (including H2's centre site `H_com`)
   and every real atom. Each is a sphere of radius R_i = sigma_i / 2 from
   `force_field_mixing_rules.def`. Real atoms without an LJ term (H_alc, H_nh3, H_h2) use the
   generic `H_` radius, 1.286 A. Charge-only dummy sites (element `X`) are left out.
2. **Masses.** Each site's mass m_i [g/mol] is read from `pseudo_atoms.def` by its label.
3. **Centre of mass.** r_cm = sum_i m_i r_i / sum_i m_i, and every position is shifted to
   d_i = r_i - r_cm.
4. **Inertia tensor.** I = sum_i m_i ( |d_i|^2 * 1 - d_i d_i^T ), a symmetric 3x3 matrix.
5. **Principal axes.** The three eigenvectors of I are the axes e_1, e_2, e_3
   (perpendicular unit vectors); the eigenvalues are the principal moments.
6. **Extent along each axis.** L_k = max_i( d_i . e_k + R_i ) - min_i( d_i . e_k - R_i ),
   i.e. the distance between the two outermost sphere surfaces along e_k.
7. **Sort.** a >= b >= c are the three L_k sorted by size (sorted by length, not by moment).
   elo = a / b, fla = b / c, and c is the effective kinetic diameter.

**Role of the masses.** Masses only choose the *directions* of the three axes (steps 3-5); the
lengths themselves come from positions and radii (step 6). Heavy sites dominate the inertia
tensor, so the axes follow the heavy-atom frame and light H atoms barely tilt them. Sites
with zero mass (e.g. H2's `H_com`) still count for size but not for the axes. Changing a mass in
`pseudo_atoms.def` can therefore rotate the axes and change a, b, c, elo and fla, but never
epsi_eff or mu.

**Special cases.**
- One site (He, Ar, Kr, Xe, methane): a = b = c = sigma.
- Equal moments (relative difference < 0.1%; e.g. benzene, NH3): inside the plane (or space) of
  equal moments any perpendicular pair of directions is a valid set of axes. To make the result
  unique and independent of how the molecule is oriented in the XYZ file, the first axis is
  taken along the longest extent within that plane, and the second perpendicular to it.

This is the method of the project's base representation; the equal-moment rule is the only
addition.

## Author

    J. Fernando Fajardo-Rojas
    Tatiane G. de Vilas
    Ryther Anderson
    Diego A. Gómez-Guadlrón
