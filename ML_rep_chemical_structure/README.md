# ML_rep_chemical_structure

36 descriptors per molecule: Tc, Pc and the acentric factor (copied from
`../ML_rep_thermodynamic/thermodynamic.csv`) plus the 33 molecular-graph counts of
Yu, Choi, Tang, Medford and Sholl, *J. Phys. Chem. C* 2021, doi:10.1021/acs.jpcc.1c05266
(Table S5). The paper computed them with a Jmol SMARTS script
(github.com/medford-group/predict_K_H, `descriptor_calculation/molecular/jmol_script.txt`);
this folder reproduces that script in Python without Jmol.

    chemical_structure.csv        the representation (one row per molecule; 37 columns, see nAT-H / nAT below)
    build_chemical_structure.py   builds it; every SMARTS pattern and counting rule is in its docstring

## Expected layout

    Representations/
      Structures/all_atom/<id>.xyz             all-atom structures, explicit H (input)
      ML_rep_thermodynamic/thermodynamic.csv   source of Tc_K, Pc_MPa, omega (input)
      ML_rep_chemical_structure/
        build_chemical_structure.py
        chemical_structure.csv                 written here

Every `*.xyz` in `Structures/all_atom/` is processed; its file name is the row id and must
also appear in other representations of this work.

## Method
Bonds and bond orders are perceived from the XYZ geometry with RDKit (`rdDetermineBonds`,
xyz2mol algorithm, neutral molecule); sulfur keeps an expanded octet (SO2 = O=S=O). After
that only the graph is used. Each count is the number of distinct matches of the paper's
SMARTS pattern (same patterns, same counting as Jmol's `find("smarts", ...).count`). Two
Jmol conventions are reproduced on purpose:
- a bond left unspecified in a pattern matches any bond order, as in the Jmol script of
  the reference paper (written `~` here). This lets C=O, C≡O, C≡N bonds count as carbon-oxygen
  or carbon-nitrogen neighbours in the F01/F02 descriptors (e.g., DMF, CO2, CO). Under the strict
  SMARTS rule applied by RDKit (unespecified = single or aromatic), this bonds would be missed.
- Table S5 in ref. paper describes `nAT-H` as "total number of atoms (H-excluded)", but the published
  values are the count **including hydrogens** (methane = 5). The CSV therefore carries both:
  `nAT-H` = non-hydrogen atoms (as described) and `nAT` = all atoms (as published).
  **They are two versions of the same descriptor: use only one of them when training a
  model** (37 columns in the file, 36 in the representation).

With these rules the script reproduces the published values of all 33 counts for all 45
molecules of the paper. Aromatic rings (benzene) are outside that set: their bonds match
neither `=` nor `-`, so benzene has n2 = 0, nSB = 6, nVinylC = 0.

## Requirements
Python 3 and RDKit (`pip install rdkit`); everything else is the standard library.
Tested with Python 3.10.12 and RDKit 2026.03.6. Deterministic: same XYZ files, same CSV.

    python build_chemical_structure.py
