"""
MOL-2D-IPHs representation of the GCMC model of each molecule: 2D interaction-parameter
histograms (distance x parameter value) on a grid around the molecule.

    Structures/gcmc/<id>.xyz  +  forcefield/  ->  mol_2d_iphs.csv   300 features per molecule
                                              ->  bin_edges.csv     the bins used (needed to
                                                                    featurise new molecules)
METHOD
  1. Sites and parameters. Every site of the gcmc XYZ (real atoms and dummy sites) with its
     TraPPE label; epsilon/k_B [K] and sigma [A] from force_field_mixing_rules.def, charge
     [e] and mass from pseudo_atoms.def (GenericMOFs_corrected; H2's feynman types are
     identical). The molecule is placed with its centre of mass at the origin.
  2. Grid. Cubic box of edge L = 20 A centred on the molecule, spacing h = 1 A: 20 points per
     axis at the cell centres (-9.5, -8.5, ..., 9.5), 8000 grid points.
  3. Nearest site. For each grid point g, d_g = min_i |r_g - r_i| and the parameters of that
     nearest site are assigned to g. Only grid points with d_g <= r_s = 10 A are kept.
     For epsilon and sigma the nearest site is searched among the sites that carry a
     Lennard-Jones term (charge-only dummies such as M_nh3, N_com are skipped: they have no
     epsilon/sigma); for q it is searched among all sites (every site carries a charge, 0 or
     not). See NOTES.
  4. Histograms. For each parameter P in (epsilon, sigma, q): a 10 x 10 matrix of DENSITIES,
     i.e. the count of kept grid points in each (distance, parameter) cell divided by the
     total number of kept grid points of that histogram, so every 10 x 10 block sums to 1 and
     molecules of different size are comparable.
     Distance bins: 1 A wide, bin k = [k, k+1) for k = 0..8, bin 9 = [9, 10] (with r_s = 10 A
     nothing lies beyond; if r_s were larger the last bin would collect all counts >= 9 A).
     Parameter bins: 10 equal-width bins between the minimum and the maximum value of P over
     the WHOLE force field (every lennard-jones entry of force_field_mixing_rules.def for
     epsilon and sigma, every charge of pseudo_atoms.def for q), not over the molecules
     built, so the edges are fixed by the force field and any molecule it can describe falls
     inside them. The last bin is closed at the maximum. Edges -> bin_edges.csv.
  5. Features. The three 10 x 10 matrices are flattened row by row (distance bin outer,
     parameter bin inner) and concatenated: eps_d0_p0 ... eps_d9_p9, sig_..., q_... = 300
     densities in [0, 1].

NOTES
  * Not rotation invariant: the grid is fixed in the molecule frame, so rotating the input
    changes which cells fall in which bin (rotations by 90 degrees about a box axis map the
    grid onto itself and ARE exact). Translations are undone by the centre-of-mass centring.
    The gcmc XYZ files are all in the canonical frame of ../Structures/check_frame.py (centre
    of mass at the origin, principal axes along x, y, z, degenerate axes fixed by a
    longest-extent rule), so the features are comparable across molecules and reproducible.
    Axis order and signs do not matter (cubic symmetry of the grid). A new molecule's XYZ must
    be put in that frame first: python ../Structures/check_frame.py --fix <file>.
  * The parameter bins depend only on the force-field files, not on the molecules built, so
    adding molecules never moves them. Edges for GenericMOFs_corrected: epsilon 0.8-342.19 K,
    sigma 2.183-6.38 A (the 6.38 A entry is TraPPE's quaternary C_sp3), q -1.23 to 1.077 e.
    The 22 benchmark molecules occupy epsilon bins 0-6 and sigma bins 0-4 of 10.
  * Single-site molecules give one non-zero column per distance bin.

Molecules = every id found in Structures/gcmc.
Run:  python build_mol_2d_iphs.py     (needs numpy)
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STRUCT = HERE.parent / "Structures" / "gcmc"
FFDIR = HERE / "forcefield" / "GenericMOFs_corrected"
L, H, R_SHELL = 20.0, 1.0, 10.0
N_DIST, N_PAR = 10, 10
PROPS = ("eps", "sig", "q")
COLS = ["id"] + [f"{p}_d{k}_p{j}" for p in PROPS for k in range(N_DIST) for j in range(N_PAR)]


# ------------------------------------------------------------------ inputs
def read_xyz(path):
    lines = open(path).read().splitlines(); n = int(lines[0])
    rows = [l.split() for l in lines[2:2 + n]]
    return np.array([[float(x) for x in r[1:4]] for r in rows]), [r[4] for r in rows]


def load_mixing_rules():
    out = []
    for ln in open(FFDIR / "force_field_mixing_rules.def"):
        t = ln.split()
        if len(t) >= 2 and not ln.lstrip().startswith("#"):
            kind = t[1].lower()
            if kind == "lennard-jones" and len(t) >= 4:
                out.append((t[0], float(t[2]), float(t[3])))
            elif kind == "none":
                out.append((t[0], None, None))
    return out


def lj(rules, typ):
    """(epsilon, sigma) or (None, None) for a site without LJ term; RASPA lookup (wildcard 'X_', last match wins)"""
    hit = None
    for name, eps, sig in rules:
        if name == typ or (name.endswith("_") and typ.startswith(name[:-1])):
            hit = (eps, sig)
    if hit is None:
        raise KeyError(f"no LJ entry for {typ}")
    return hit


def load_pseudo_atoms():
    """label -> (mass, charge)"""
    d = {}
    for ln in open(FFDIR / "pseudo_atoms.def"):
        t = ln.split()
        if len(t) >= 7 and not ln.lstrip().startswith("#") and t[1] in ("yes", "no"):
            try:
                d[t[0]] = (float(t[5]), float(t[6]))
            except ValueError:
                pass
    return d


def sites(mid, rules, pseudo):
    pos, labels = read_xyz(STRUCT / f"{mid}.xyz")
    missing = sorted({l for l in labels if l not in pseudo})
    if missing:
        raise KeyError(f"{mid}: not in pseudo_atoms.def: {', '.join(missing)}")
    mass = np.array([pseudo[l][0] for l in labels]); q = np.array([pseudo[l][1] for l in labels])
    ljp = [lj(rules, l) for l in labels]
    eps = np.array([e if e is not None else np.nan for e, _ in ljp])
    sig = np.array([s if s is not None else np.nan for _, s in ljp])
    pos = pos - np.average(pos, axis=0, weights=mass if mass.sum() > 0 else None)   # centre of mass at origin
    return pos, {"eps": eps, "sig": sig, "q": q}


# ------------------------------------------------------------------ grid
def grid_points():
    c = (np.arange(int(round(L / H))) + 0.5) * H - L / 2
    return np.array(np.meshgrid(c, c, c, indexing="ij")).reshape(3, -1).T


GRID = grid_points()


def nearest(pos, mask):
    """(distance to nearest kept site, index of that site) for every grid point"""
    idx = np.flatnonzero(mask)
    d = np.linalg.norm(GRID[:, None, :] - pos[idx][None, :, :], axis=2)
    k = d.argmin(axis=1)
    return d[np.arange(len(GRID)), k], idx[k]


def histogram(dist, values, edges):
    """10 x 10 density matrix: counts of kept grid points / total kept (sums to 1)"""
    keep = dist <= R_SHELL
    kd = np.minimum((dist[keep] // H).astype(int), N_DIST - 1)
    kp = np.clip(np.searchsorted(edges, values[keep], side="right") - 1, 0, N_PAR - 1)
    Hm = np.zeros((N_DIST, N_PAR), dtype=float)
    np.add.at(Hm, (kd, kp), 1.0)
    return Hm / Hm.sum()


def features(pos, P, edges):
    out = {}
    for p in PROPS:
        vals = P[p]; mask = ~np.isnan(vals)
        dist, who = nearest(pos, mask)
        Hm = histogram(dist, vals[who], edges[p])
        out.update({f"{p}_d{k}_p{j}": float(Hm[k, j]) for k in range(N_DIST) for j in range(N_PAR)})
    return out


def parameter_edges(rules, pseudo):
    """10 equal-width bins between the min and max of each parameter over the whole force field"""
    eps = [e for _, e, _ in rules if e is not None]; sig = [g for _, _, g in rules if g is not None]
    q = [c for _, c in pseudo.values()]
    return {p: np.linspace(min(v), max(v), N_PAR + 1) for p, v in (("eps", eps), ("sig", sig), ("q", q))}


def main():
    ids = sorted(p.stem for p in STRUCT.glob("*.xyz"))
    if not ids:
        raise SystemExit(f"no gcmc structures found: expected {STRUCT}/<id>.xyz")
    rules, pseudo = load_mixing_rules(), load_pseudo_atoms()
    data = {mid: sites(mid, rules, pseudo) for mid in ids}
    edges = parameter_edges(rules, pseudo)
    rows = [{"id": mid, **features(pos, P, edges)} for mid, (pos, P) in data.items()]
    with open(HERE / "mol_2d_iphs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(HERE / "bin_edges.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["parameter", "unit"] + [f"edge_{j}" for j in range(N_PAR + 1)])
        for p, unit in zip(PROPS, ("K", "A", "e")):
            w.writerow([p, unit] + [f"{e:.6f}" for e in edges[p]])
        w.writerow(["distance", "A"] + [f"{k * H:.1f}" for k in range(N_DIST + 1)])
    print(f"wrote mol_2d_iphs.csv ({len(rows)} molecules, {len(COLS) - 1} features) + bin_edges.csv; "
          f"grid {len(GRID)} points, shell r_s = {R_SHELL} A")


if __name__ == "__main__":
    main()
