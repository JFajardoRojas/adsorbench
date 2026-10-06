"""
String-based representation of the benchmark molecules: a Morgan (extended-connectivity)
fingerprint computed from each molecule's canonical SMILES.

    Structures/all_atom/<id>.xyz  ->  smiles.csv          id, canonical SMILES
                                  ->  string_based.csv    id, fp_000 ... fp_099 (0/1)

STEPS
  1. Molecular graph from the all-atom XYZ: bonds and bond orders perceived from the geometry
     with RDKit (rdDetermineBonds, xyz2mol algorithm, neutral molecule); sulfur keeps an
     expanded octet (SO2 = O=S=O). Same graph as ML_rep_chemical_structure.
  2. Canonical SMILES (RDKit, hydrogens implicit, aromaticity perceived) -> smiles.csv.
  3. The SMILES is parsed back and turned into a Morgan fingerprint:
       radius 3 bonds (ECFP6), 100 bits, binary, default ECFP atom invariants (element,
       heavy-atom degree, hydrogen count, formal charge, ring membership), bond types used,
       chirality ignored. Each atom's neighbourhood up to 3 bonds away is a substructure; every
       substructure is hashed to one of the 100 positions, set to 1 when the substructure is present.
     Radius 3 rather than 2: with radius 2 a binary fingerprint cannot tell heptane from octane
     (every n-alkane from C7 up contains the same set of radius-2 environments). Radius 3
     separates all 22 benchmark molecules; the same collapse returns for n-alkanes from C9 up.
     Binary rather than counts on purpose: counts would duplicate the information of the
     chemical-structure representation.

NOTES
  * 100 bits is short: different substructures can hash to the same bit (bit collisions),
    so a 1 does not identify a single substructure.
  * Two molecules can share the same fingerprint, but this is not observed in 22 becnchmark molecules. 
  * H2 has no heavy atom, so its SMILES keeps explicit hydrogens ([H][H]).

Molecules = every id found in Structures/all_atom.
Run:  python build_string_based.py     (needs rdkit)
"""
import csv
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
STRUCT = HERE.parent / "Structures" / "all_atom"
RADIUS, N_BITS = 3, 100
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
BITS = [f"fp_{i:03d}" for i in range(N_BITS)]


def read_mol(path):
    """RDKit molecule from an all-atom XYZ (explicit H, bond orders perceived from the geometry)"""
    lines = open(path).read().splitlines(); n = int(lines[0])
    mol = Chem.MolFromXYZBlock(f"{n}\n\n" + "\n".join(lines[2:2 + n]))
    if n > 1:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    rw = Chem.RWMol(mol)                                   # expanded octet on S: [S+]-[O-] -> S=O
    for b in rw.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        s, o = (a1, a2) if a1.GetSymbol() == "S" else (a2, a1)
        if s.GetSymbol() == "S" and o.GetSymbol() == "O" and s.GetFormalCharge() > 0 and o.GetFormalCharge() < 0 \
                and b.GetBondType() == Chem.BondType.SINGLE:
            b.SetBondType(Chem.BondType.DOUBLE); s.SetFormalCharge(s.GetFormalCharge() - 1); o.SetFormalCharge(0)
    mol = rw.GetMol(); Chem.SanitizeMol(mol)
    return mol


def canonical_smiles(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def fingerprint(smiles):
    return list(GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(int))


def main():
    ids = sorted(p.stem for p in STRUCT.glob("*.xyz"))
    if not ids:
        raise SystemExit(f"no structures found: expected {STRUCT}/<id>.xyz")
    smi = {mid: canonical_smiles(read_mol(STRUCT / f"{mid}.xyz")) for mid in ids}
    fps = {mid: fingerprint(smi[mid]) for mid in ids}
    with open(HERE / "smiles.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["id", "smiles"])
        w.writerows([mid, smi[mid]] for mid in ids)
    with open(HERE / "string_based.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["id"] + BITS)
        w.writerows([mid] + fps[mid] for mid in ids)
    groups = defaultdict(list)
    for mid in ids:
        groups[tuple(fps[mid])].append(mid)
    same = [g for g in groups.values() if len(g) > 1]
    print(f"wrote smiles.csv + string_based.csv ({len(ids)} molecules, {N_BITS} bits, radius {RADIUS})")
    for g in same:
        print("  identical fingerprints:", ", ".join(g))


if __name__ == "__main__":
    main()
