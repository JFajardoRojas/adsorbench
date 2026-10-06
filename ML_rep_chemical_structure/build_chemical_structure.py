"""
Chemical-structure representation (36 descriptors + nAT, see COUNTING RULES) of the benchmark molecules, as defined by
Yu, Choi, Tang, Medford and Sholl, J. Phys. Chem. C 2021 (doi:10.1021/acs.jpcc.1c05266),
Table S5, and computed there with a Jmol SMARTS script
(github.com/medford-group/predict_K_H, descriptor_calculation/molecular/jmol_script.txt).

    Structures/all_atom/<id>.xyz  +  ML_rep_thermodynamic/thermodynamic.csv  ->  chemical_structure.csv

COLUMNS
  Tc_K, Pc_MPa, omega   copied from ../ML_rep_thermodynamic/thermodynamic.csv
  Reference paper uses Pc_Bar, but this difference is negligeble when normalization is applied for ML training.
  33 structural counts  computed from the molecular graph with the SMARTS below

MOLECULAR GRAPH
  Atoms and coordinates come from the all-atom XYZ (explicit H). Bonds and bond orders are
  perceived from the geometry with RDKit (rdDetermineBonds, the xyz2mol algorithm, neutral
  molecule). Two conventions are applied on top:
    * expanded octet on sulfur: a perceived [S+]-[O-] pair is rewritten as S=O, so SO2 is
      O=S=O (2 double bonds) rather than the charge-separated form.
    * aromatic rings keep their aromatic bonds, which match neither "=" nor "-" (standard
      SMARTS semantics, also Jmol's): benzene has n2 = 0, nSB = 6 (the C-H bonds), nTB = 12,
      nVinylC = 0. The paper's 45 molecules contain no aromatic ring, so this case could not
      be checked against published values.
  CO is perceived as the usual Lewis structure [C-]#[O+] (one triple bond).
  The descriptors do not depend on bond lengths or angles, only in the graph description.

COUNTING RULES (reproduce the published Jmol values for all 45 molecules of the reference paper)
  * every count = number of distinct SMARTS matches (distinct sets of atoms); e.g. acetone
    has nCarbonylC = 2 because [CX3](=[OX1])~C matches once per methyl.
  * a bond left unspecified in the Jmol script matches ANY bond order (Jmol semantics);
    in RDKit an unspecified bond means single-or-aromatic, so those bonds are written "~".
  * bonds to H count in nTB, nSB and the F0x pair counts (H atoms are explicit).
  * nAT-H / nAT: Table S5 in reference paper describes nAT-H as "total number of atoms (H-excluded)", but the
    published values are the TOTAL atom count, hydrogens included (methane 5, ethane 8): the
    script's selector {* and !H} did not exclude H in Jmol. Both are written:
      nAT-H  non-hydrogen atoms   (the descriptor as described)
      nAT    all atoms, H included (the descriptor as published)
    They carry the same information twice: USE ONLY ONE OF THEM when training a model.
  * nCarbonyl2 adds the charge-separated form [CX3+]-[OX1-] to nCarbonyl1; for neutral
    molecules the two are identical.

DESCRIPTOR DEFINITIONS (SMARTS exactly as in the Jmol script, "~" added where needed)
  nC, nO, nN   number of C / O / N atoms
  n2           [*]=[*]       double bonds
  n3           [*]#[*]       triple bonds
  nAT-H        number of non-hydrogen atoms      (as described; see COUNTING RULES)
  nAT          number of atoms, H included       (as published; use one of the two)
  nTB          [*]~[*]       all bonds
  nSB          [*]-[*]       single bonds
  nAlkylC      [CX4]
  nVinylC      [$([CX3]=[CX3])]
  nCarbonyl1   [CX3]=[OX1]
  nCarbonyl2   [$([CX3]=[OX1]),$([CX3+]-[OX1-])]
  nCarbonylC   [CX3](=[OX1])~C
  nAldehyde    [CX3H1](=O)~[#6]
  nKetone      [#6]~[CX3](=O)~[#6]
  nEther       [OD2](~[#6])~[#6]
  nAmine12     [NX3;H2,H1;!$(N~C=O)]
  nAmine1      [NX3;H2;!$(N~C=[!#6]);!$(N~C#[!#6])]~[#6]
  nNitrile     [NX1]#[CX2]
  nOH          [OX2H]
  nOHAlcohol   [#6]~[OX2H]
  nN#          [$([NX1]#*)]
  n-O-         [$([OX2])]
  F01[A-B]     [A]~[B]        A-B atom pairs at topological distance 1
  F02[A-B]     [A]~*~[B]      A-B atom pairs at topological distance 2
               for A-B = H-C, H-N, H-O, C-N, C-O

Molecules = every id found in Structures/all_atom.
Run:  python build_chemical_structure.py     (needs rdkit)
"""
import csv
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
REP = HERE.parent
STRUCT = REP / "Structures" / "all_atom"
THERMO = REP / "ML_rep_thermodynamic" / "thermodynamic.csv"

PAIRS = [("H", "C"), ("H", "N"), ("H", "O"), ("C", "N"), ("C", "O")]
NUM = {"H": 1, "C": 6, "N": 7, "O": 8}
SMARTS = {
    "n2": "[*]=[*]", "n3": "[*]#[*]", "nTB": "[*]~[*]", "nSB": "[*]-[*]",
    "nAlkylC": "[CX4]", "nVinylC": "[$([CX3]=[CX3])]",
    "nCarbonyl1": "[CX3]=[OX1]", "nCarbonyl2": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]", "nCarbonylC": "[CX3](=[OX1])~C",
    "nAldehyde": "[CX3H1](=O)~[#6]", "nKetone": "[#6]~[CX3](=O)~[#6]", "nEther": "[OD2](~[#6])~[#6]",
    "nAmine12": "[NX3;H2,H1;!$(N~C=O)]", "nAmine1": "[NX3;H2;!$(N~C=[!#6]);!$(N~C#[!#6])]~[#6]",
    "nNitrile": "[NX1]#[CX2]", "nOH": "[OX2H]", "nOHAlcohol": "[#6]~[OX2H]", "nN#": "[$([NX1]#*)]", "n-O-": "[$([OX2])]",
}
for a, b in PAIRS:
    SMARTS[f"F01[{a}-{b}]"] = f"[#{NUM[a]}]~[#{NUM[b]}]"
for a, b in PAIRS:
    SMARTS[f"F02[{a}-{b}]"] = f"[#{NUM[a]}]~*~[#{NUM[b]}]"
PATT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
GRAPH = ["nC", "nO", "nN", "n2", "n3", "nAT-H", "nAT", "nTB", "nSB"] + [k for k in SMARTS if k not in ("n2", "n3", "nTB", "nSB")]
COLS = ["id", "Tc_K", "Pc_MPa", "omega"] + GRAPH


def read_mol(path):
    """RDKit molecule (explicit H, bond orders perceived from the geometry)"""
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


def count(mol, key):
    """number of distinct matches (distinct atom sets), as Jmol's find(...).count"""
    return len(mol.GetSubstructMatches(PATT[key], uniquify=True))


def descriptors(mol):
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    d = {"nC": sym.count("C"), "nO": sym.count("O"), "nN": sym.count("N"),
         "nAT-H": sum(x != "H" for x in sym), "nAT": len(sym)}
    for k in SMARTS:
        d[k] = count(mol, k)
    return d


def main():
    thermo = {r["id"]: r for r in csv.DictReader(open(THERMO))}
    ids = sorted(p.stem for p in STRUCT.glob("*.xyz"))
    if not ids:
        raise SystemExit(f"no structures found: expected {STRUCT}/<id>.xyz")
    missing = [i for i in ids if i not in thermo]
    if missing:
        raise SystemExit(f"not in {THERMO.name}: {', '.join(missing)}")
    rows = []
    for mid in ids:
        t = thermo[mid]
        rows.append({"id": mid, "Tc_K": t["Tc_K"], "Pc_MPa": t["Pc_MPa"], "omega": t["omega"],
                     **descriptors(read_mol(STRUCT / f"{mid}.xyz"))})
    with open(HERE / "chemical_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(f"wrote chemical_structure.csv ({len(rows)} molecules, {len(COLS) - 1} columns; "
          f"nAT-H and nAT are alternatives, use one)")


if __name__ == "__main__":
    main()
