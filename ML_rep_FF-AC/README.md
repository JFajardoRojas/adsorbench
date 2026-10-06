# ML_rep_FF-AC

Force-field autocorrelation (FF-AC) representation: Moreau-Broto autocorrelations (full-scope
RACs, Janet & Kulik) of the force-field parameters of each molecule's GCMC model.

    ff_ac.csv          the representation (one row per molecule, 12 descriptors)
    ff_ac_edges.csv    the graph used for each molecule: one row per edge, with its origin
                       (def = bond listed in the RASPA .def; added = see note below)
    build_ff_ac.py     builds both files (definitions in its docstring)
    forcefield/        copies of the RASPA files used in the GCMC simulations:
      GenericMOFs_corrected/      force_field_mixing_rules.def (epsilon, sigma), pseudo_atoms.def (charges)
      molecules/<name>.def                          TraPPE molecule definitions (site list + bonds)

## Expected layout

    Representations/
      Structures/gcmc/<id>.xyz      GCMC site models (input: the site labels, in order)
      ML_rep_FF-AC/
        build_ff_ac.py
        forcefield/
        ff_ac.csv                   written here

Every `*.xyz` in `Structures/gcmc/` is processed; its file name is the row id. Its site list
must match one of the `forcefield/molecules/*.def` files, which supplies the bonds.

## Definition
Each molecule is a graph whose nodes are the interaction sites of the RASPA model (real atoms
and dummy sites alike, since all carry parameters). Its edges are the real-real bonds and the
dummy-parent bonds listed in the molecule's `.def` file, plus the direct bond between two real
atoms that the `.def` connects only through a dummy centre site (see note below). For a site
property P and a depth d,

    P_d = sum_i sum_j P_i P_j delta(d_ij, d)

with d_ij the number of bonds on the shortest path from i to j; the sum runs over all ordered
pairs (each pair i != j counts twice, d = 0 gives sum_i P_i^2). Three properties, looked up by
the site's TraPPE type: epsilon/k_B [K] and sigma [A] from `force_field_mixing_rules.def`
(sites with no LJ term count as 0), charge [e] from `pseudo_atoms.def`. Depths 0-3 give
eps_0..eps_3 [K^2], sig_0..sig_3 [A^2], q_0..q_3 [e^2].

## Things to know
- **Note on the graph.** Dummy sites are nodes, bonded to their parent atom as in the `.def`
  files (M_nh3 to N, M_co to C, ...). In H2 and N2 the `.def` joins the two atoms only through
  the centre site (H-H_com-H, N-N_com-N), which would put the two atoms at distance 2. The
  direct atom-atom bond is therefore added, so the graph holds both the real bond and the
  dummy-parent bonds: for N2, eps_1 = 2 x 36 x 36 and eps_2 = 0. The added edges are marked
  `added` in `ff_ac_edges.csv` (h2 and n2 only, for the 22 molecules).
- One parameter set for all molecules. H2 was simulated with `feynman_corrected`, but its site
  types (H_h2, H_com) have identical epsilon, sigma and charge in `GenericMOFs_corrected`; the
  Feynman-Hibbs quantum correction lives in a separate file and does not change them.
- Single-site molecules (noble gases, methane) have only the d = 0 terms.
- Only force-field parameters and connectivity are used, so the same code describes
  alchemical molecules: give them a gcmc XYZ with labels, a `.def` with bonds and entries in the
  forcefield files.

## Requirements
Python 3 and numpy. Tested with Python 3.10.12 and numpy 2.2.6. Deterministic.

    python build_ff_ac.py
