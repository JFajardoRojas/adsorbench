# ML_rep_string-based

Morgan (extended-connectivity) fingerprint of each molecule, computed from its canonical SMILES.

    smiles.csv               canonical SMILES of each molecule (RDKit)
    string_based.csv         the representation: id + 100 binary features (fp_000 ... fp_099)
    build_string_based.py    builds both files (method in its docstring)

## Expected layout

    Representations/
      Structures/all_atom/<id>.xyz      all-atom structures, explicit H (input)
      ML_rep_string-based/
        build_string_based.py
        smiles.csv, string_based.csv    written here

Every `*.xyz` in `Structures/all_atom/` is processed; its file name is the row id.

## Method
1. Molecular graph from the all-atom XYZ: bonds and bond orders perceived with RDKit
   (`rdDetermineBonds`, neutral molecule); SO2 is written O=S=O.
2. Canonical SMILES with RDKit (hydrogens implicit, aromaticity perceived).
3. Morgan fingerprint of that SMILES: radius 3 bonds (ECFP6), 100 bits, binary, default ECFP
   atom invariants, bond types used, chirality ignored. Each atom's neighbourhood up to 3 bonds
   away is a substructure; each substructure is hashed to one of the 100 positions, set to 1
   when present.

## Things to know
- **Why radius 3.** A binary fingerprint records which atom environments exist, not how many.
  With radius 2 every n-alkane from heptane up has the same set, so heptane and octane were
  identical. Radius 3 separates all 22 molecules (the same collapse returns for n-alkanes from
  C9 up). Counts were not used, because they would duplicate the chemical-structure representation.
- **Bit collisions.** With 100 bits, unrelated substructures can share a position, so a 1 does
  not identify a single substructure.
- The fingerprint depends only on the molecular graph, so it cannot describe alchemical
  (force-field-only) molecules.

## Requirements
Python 3 and RDKit (`pip install rdkit`). Tested with Python 3.10.12 and RDKit 2026.03.6.
Deterministic: same XYZ files, same CSVs.

    python build_string_based.py
