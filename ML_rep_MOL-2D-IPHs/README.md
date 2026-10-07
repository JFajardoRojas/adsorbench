# ML_rep_MOL-2D-IPHs

Interaction-parameter histograms of each molecule's GCMC model on a grid: for each of the three
force-field parameters (epsilon, sigma, q) a 10 x 10 density matrix of (distance to the molecule)
x (parameter value), flattened and concatenated into 300 features.

    mol_2d_iphs.csv          the representation (one row per molecule, 300 densities; each
                             10 x 10 block sums to 1)
    bin_edges.csv            distance and parameter bin edges used (needed to featurise a new molecule)
    build_mol_2d_iphs.py     builds both files (full method in its docstring)
    forcefield/GenericMOFs_corrected/   force_field_mixing_rules.def (epsilon, sigma),
                                        pseudo_atoms.def (charges, masses); copies of the GCMC files

## Expected layout

    Representations/
      Structures/gcmc/<id>.xyz      GCMC site models (positions + site labels)
      ML_rep_MOL-2D-IPHs/
        build_mol_2d_iphs.py
        forcefield/
        mol_2d_iphs.csv, bin_edges.csv   written here

## Method (short)
1. The molecule's sites, with epsilon, sigma, q looked up by TraPPE label, centre of mass at the origin.
2. A 20 A cubic box, 1 A grid (20 x 20 x 20 = 8000 cell-centre points).
3. Each grid point takes the parameters of its nearest site; points farther than r_s = 10 A from
   every site are dropped. For epsilon and sigma only sites with a Lennard-Jones term are
   candidates (charge-only dummies have none); for q all sites are.
4. Per parameter, the fraction of the kept grid points in each of 10 distance bins (1 A each,
   0-10 A) x 10 parameter bins (equal width between the minimum and maximum of that parameter in
   the whole force field: all LJ entries of `force_field_mixing_rules.def` for epsilon and sigma,
   all charges of `pseudo_atoms.def` for q; edges in `bin_edges.csv`). Densities, not counts, so
   molecules of different size are comparable.
5. Columns `eps_d<k>_p<j>`, `sig_d<k>_p<j>`, `q_d<k>_p<j>`: k = distance bin, j = parameter bin.

## Things to know
- **Rotation.** The grid is fixed in the molecule's frame, so the features change if the input
  coordinates are rotated by an arbitrary angle (tested: 33-89 of 300 features move for a random
  rotation). Rotations by 90 degrees about a box axis and any translation leave them unchanged
  (translations are removed by the centre-of-mass centring). The gcmc XYZ files are all in the
  canonical frame defined in `../Structures/check_frame.py` (centre of mass at the origin,
  principal axes along x, y, z, degenerate axes fixed by a longest-extent rule), which is what
  makes the features comparable across molecules and reproducible. Any order or sign of the
  three axes gives the same features (the grid has cubic symmetry), so only molecules with
  degenerate moments (benzene, NH3) depend on that rule. Check a new molecule's file with
  `python ../Structures/check_frame.py <file>` and rotate it with `--fix` before building.
- **Bins are fixed by the force field**, not by the molecules, so adding molecules never moves
  them: epsilon 0.8-342.19 K, sigma 2.183-6.38 A, q -1.23 to 1.077 e. The sigma range is stretched
  by TraPPE's quaternary carbon (6.38 A), so the 22 benchmark molecules occupy only sigma bins 0-4
  and epsilon bins 0-6 of 10. Any molecule the force field can describe falls inside the range.
- With r_s = 10 A the last distance bin is [9, 10] A; nothing lies beyond the shell.

## Requirements
Python 3 and numpy. Tested with Python 3.10.12 and numpy 2.2.6. Deterministic.

    python build_mol_2d_iphs.py
