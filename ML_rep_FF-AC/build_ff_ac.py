"""
Force-field autocorrelation (FF-AC) representation of the GCMC model of each molecule:
Moreau-Broto autocorrelations (full-scope RACs, Janet & Kulik) of force-field site parameters.

    Structures/gcmc/<id>.xyz  +  forcefield/  ->  ff_ac.csv        the representation
                                             ->  ff_ac_edges.csv  the graph used for each molecule

DESCRIPTORS (12)
  For a site property P and a depth d:   P_d = sum_i sum_j P_i P_j delta(d_ij, d)
  where the double sum runs over ALL ordered pairs of sites (i = j included, so each
  unordered pair i != j contributes twice) and d_ij is the number of bonds on the shortest
  path between i and j. Depths d = 0, 1, 2, 3; properties:
    eps   Lennard-Jones well depth epsilon/k_B [K]   ->  eps_0 .. eps_3   [K^2]
    sig   Lennard-Jones size sigma [A]               ->  sig_0 .. sig_3   [A^2]
    q     point charge [e]                           ->  q_0 .. q_3       [e^2]
  d = 0 is sum_i P_i^2; a depth with no pair at that distance gives 0.

GRAPH
  Nodes = every interaction site of the RASPA molecule definition, in the order of the gcmc
  XYZ (real atoms and charge-only / LJ-only dummy sites alike: they all carry parameters).
  Edges = (a) the bonds listed in the RASPA molecule file forcefield/molecules/<def>.def
              (RIGID_BOND / HARMONIC_BOND lines): real-real bonds and dummy-parent bonds
              (M_nh3-N, M_S-S, M_so2-S, M_co-C, H_com-H, N_com-N);
          (b) ADDED: the direct bond between two real atoms whose .def connection runs only
              through a dummy site, i.e. a dummy bonded to two or more real atoms gets those
              atoms bonded to each other. This affects H2 and N2 (H_com, N_com sit on the bond
              midpoint): the two atoms are at distance 1 AND each is at distance 1 from the
              centre site. Every edge used, with its origin (def / added), is written to
              ff_ac_edges.csv.
  The XYZ site labels must match the .def site list in order; otherwise the build stops.

PARAMETERS (forcefield/GenericMOFs_corrected/, copies of the RASPA files used in GCMC),
  looked up by the site label (TraPPE type):
  force_field_mixing_rules.def -> epsilon, sigma. RASPA lookup: a name ending in "_" is a
      wildcard for every type starting with the text before it; entries are read in file
      order and the last match wins. "none" -> epsilon = sigma = 0.
  pseudo_atoms.def -> charge.
  One parameter set for all molecules: GenericMOFs_corrected. H2 was simulated with the
  feynman_corrected force field, but its H2 site types (H_h2, H_com) have identical epsilon,
  sigma and charge in both files (the Feynman-Hibbs quantum correction lives in a separate
  force_field.def and does not change these parameters), so only GenericMOFs_corrected is kept.

Molecules = every id found in Structures/gcmc.
Run:  python build_ff_ac.py     (needs numpy)
"""
import csv
from collections import deque
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STRUCT = HERE.parent / "Structures" / "gcmc"
FFDIR = HERE / "forcefield"
DEPTHS = (0, 1, 2, 3)
PROPS = ("eps", "sig", "q")
COLS = ["id"] + [f"{p}_{d}" for p in PROPS for d in DEPTHS]


# ------------------------------------------------------------------ inputs
def read_xyz(path):
    """(species, site labels) in file order; species 'X' marks a dummy site"""
    lines = open(path).read().splitlines()
    n = int(lines[0]); rows = [l.split() for l in lines[2:2 + n]]
    return [r[0] for r in rows], [r[4] for r in rows]


def forcefield_of(mid):
    """one parameter set for all molecules (h2's feynman_corrected types are identical, see docstring)"""
    return "GenericMOFs_corrected"


def def_file_of(mid, labels):
    """the RASPA molecule file whose site list equals the XYZ labels, in order"""
    for p in sorted((FFDIR / "molecules").glob("*.def")):
        if read_def(p)[0] == labels:
            return p
    raise SystemExit(f"{mid}: no forcefield/molecules/*.def has the site list {labels}")


def read_def(path):
    """(site labels in order, list of bonded index pairs) from a RASPA molecule .def"""
    lines = [l.rstrip() for l in open(path)]
    i = next(k for k, l in enumerate(lines) if "atomic positions" in l)
    labels = []
    for l in lines[i + 1:]:
        t = l.split()
        if not t or l.strip().startswith("#") or not t[0].isdigit():
            break
        labels.append(t[1])
    bonds = [(int(t[0]), int(t[1])) for t in (l.split() for l in lines)
             if len(t) >= 3 and t[2] in ("RIGID_BOND", "HARMONIC_BOND")]
    return labels, bonds


def load_mixing_rules(ff):
    out = []
    for ln in open(FFDIR / ff / "force_field_mixing_rules.def"):
        t = ln.split()
        if len(t) >= 2 and not ln.lstrip().startswith("#"):
            kind = t[1].lower()
            if kind == "lennard-jones" and len(t) >= 4:
                out.append((t[0], float(t[2]), float(t[3])))
            elif kind == "none":
                out.append((t[0], 0.0, 0.0))
    return out


def lj(rules, typ):
    hit = None
    for name, eps, sig in rules:
        if name == typ or (name.endswith("_") and typ.startswith(name[:-1])):
            hit = (eps, sig)
    if hit is None:
        raise KeyError(f"no LJ entry for {typ}")
    return hit


def load_charges(ff):
    q = {}
    for ln in open(FFDIR / ff / "pseudo_atoms.def"):
        t = ln.split()
        if len(t) >= 7 and not ln.lstrip().startswith("#") and t[1] in ("yes", "no"):
            try:
                q[t[0]] = float(t[6])
            except ValueError:
                pass
    return q


def graph_edges(species, def_bonds):
    """[(i, j, origin)]: the .def bonds plus a direct bond between real atoms that the .def
    joins only through a dummy site (H2, N2)."""
    edges = [(min(i, j), max(i, j), "def") for i, j in def_bonds]
    have = {(i, j) for i, j, _ in edges}
    for k, sp in enumerate(species):
        if sp != "X":
            continue
        real = sorted({j if i == k else i for i, j in def_bonds if k in (i, j) and species[j if i == k else i] != "X"})
        for a in range(len(real)):
            for b in range(a + 1, len(real)):
                if (real[a], real[b]) not in have:
                    edges.append((real[a], real[b], "added")); have.add((real[a], real[b]))
    return edges


# ------------------------------------------------------------------ autocorrelation
def distance_matrix(n, bonds):
    """shortest-path length in bonds between every pair of sites (-1 if not connected)"""
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j); adj[j].append(i)
    D = -np.ones((n, n), dtype=int)
    for s in range(n):
        D[s, s] = 0; todo = deque([s])
        while todo:
            u = todo.popleft()
            for v in adj[u]:
                if D[s, v] < 0:
                    D[s, v] = D[s, u] + 1; todo.append(v)
    return D


def autocorrelation(P, D, d):
    P = np.asarray(P, float)
    return float(np.sum(np.outer(P, P)[D == d]))


def ff_ac(mid):
    """(descriptors, edges) for one molecule"""
    species, labels = read_xyz(STRUCT / f"{mid}.xyz")
    _, def_bonds = read_def(def_file_of(mid, labels))
    edges = graph_edges(species, def_bonds); bonds = [(i, j) for i, j, _ in edges]
    ff = forcefield_of(mid); rules = load_mixing_rules(ff); charges = load_charges(ff)
    missing = sorted({l for l in labels if l not in charges})
    if missing:
        raise KeyError(f"{mid}: not in {ff}/pseudo_atoms.def: {', '.join(missing)}")
    P = {"eps": [lj(rules, l)[0] for l in labels], "sig": [lj(rules, l)[1] for l in labels],
         "q": [charges[l] for l in labels]}
    D = distance_matrix(len(labels), bonds)
    desc = {f"{p}_{d}": autocorrelation(P[p], D, d) for p in PROPS for d in DEPTHS}
    return desc, [(i, labels[i], j, labels[j], o) for i, j, o in edges]


def main():
    ids = sorted(p.stem for p in STRUCT.glob("*.xyz"))
    if not ids:
        raise SystemExit(f"no gcmc structures found: expected {STRUCT}/<id>.xyz")
    rows, edges = [], []
    for mid in ids:
        desc, e = ff_ac(mid)
        rows.append({"id": mid, **desc}); edges += [[mid, *x] for x in e]
    with open(HERE / "ff_ac.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(HERE / "ff_ac_edges.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "site_i", "label_i", "site_j", "label_j", "origin"])   # origin: def = RASPA .def bond, added = real-real bond through a dummy centre site
        w.writerows(edges)
    added = sorted({e[0] for e in edges if e[-1] == "added"})
    print(f"wrote ff_ac.csv ({len(rows)} molecules, {len(COLS) - 1} descriptors) + ff_ac_edges.csv "
          f"({len(edges)} edges; real-real bonds added through a centre site for: {', '.join(added) or 'none'})")


if __name__ == "__main__":
    main()
