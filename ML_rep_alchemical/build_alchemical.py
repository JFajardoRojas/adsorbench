"""
Alchemical representation (8 descriptors) of the benchmark molecules, built from the GCMC
simulation model of each molecule:

    Structures/gcmc/<id>.xyz  ->  alchemical_gcmc.csv

DESCRIPTORS
  epsi_eff_K  sum of the LJ epsilon of every site of the molecule                 [K]
  a_A         largest extent along the principal axes                              [A]
  elo         a / b   (b = second extent)                                          [-]
  fla         b / c                                                                [-]
  c_A         c, taken as the effective kinetic diameter                           [A]
              (smallest extent along the principal axes; see SIZES)
  mu_D        |sum_i q_i r_i| over all point charges                               [D]
  Q           quadrupole, RASPA 2.0         -> MANUAL: type into the output CSV
  fle         ln(Rosenbluth weight), RASPA  -> MANUAL: type into the output CSV
              Unfilled cells hold "###". A rebuild keeps whatever was typed there.

FORCEFIELD (forcefield/<ff>/, copies of the RASPA files used in GCMC;
  ff = feynman_corrected for h2, GenericMOFs_corrected otherwise). Every site is looked up
  by its TraPPE type, the `label` column of the gcmc XYZ:
  force_field_mixing_rules.def -> epsilon, sigma. Lookup follows RASPA: a name ending in "_"
      is a wildcard for every type starting with the text before it; entries are read in
      file order and the last match wins. "none" -> epsilon = sigma = 0.
  pseudo_atoms.def -> mass and charge. The mass/charge columns of the XYZ are not used, so
      edits to pseudo_atoms.def reach this pipeline directly.

SIZES for a, b, c  (same method as the project's base representation)
  Every site is a sphere of radius sigma/2: sites with an LJ term (incl. H2's H_com) +
  real atoms; real atoms with no LJ (H_alc, H_nh3, H_h2) take the generic forcefield
  radius "H_" sigma/2 = 1.286 A (base used 1.20 A); charge-only dummies (X) dropped.
  Axes = principal axes of the mass-weighted inertia tensor (pseudo_atoms.def masses).
  Extent along axis k = max(p_k + R) - min(p_k - R); sorted a >= b >= c.
  Degenerate moments (benzene, NH3): axes inside the degenerate subspace chosen by
  successive longest extent, so results do not depend on orientation. This is the one
  addition to base, whose benzene value depended on how the .def file happened to lay out
  the ring (benzene elo/fla 1.0997/1.8446 here vs 1.0617/1.9048).

DIPOLE: all sites, dummies included, with the pseudo_atoms.def charges.

Molecules = every id found in Structures/gcmc.
Run:  python build_alchemical.py
"""
import csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REP = HERE.parent                                     # Representations/
STRUCT = REP / "Structures"
FFDIR = HERE / "forcefield"                           # copies of the RASPA forcefield files
E_A_TO_DEBYE = 1 / 0.20819434                         # 1 e.A = 4.80320 D
COLS = ["id", "epsi_eff_K", "a_A", "elo", "fla", "c_A", "mu_D", "Q", "fle"]


# ------------------------------------------------------------------ inputs
def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0]); rows = [l.split() for l in lines[2:2 + n]]
    return dict(species=[r[0] for r in rows],
                pos=np.array([[float(x) for x in r[1:4]] for r in rows]),
                label=[r[4] for r in rows] if len(rows[0]) > 4 else None)


def load_mixing_rules(ff):
    """ordered list of (name, epsilon_K, sigma_A); 'none' -> (0, 0)"""
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
    """RASPA lookup: wildcard 'X_' matches types starting with 'X'; last match wins."""
    hit = None
    for name, eps, sig in rules:
        if name == typ or (name.endswith("_") and typ.startswith(name[:-1])):
            hit = (eps, sig)
    if hit is None:
        raise KeyError(f"no LJ entry for {typ}")
    return hit


def load_pseudo_atoms(ff):
    """pseudo-atom type -> (mass [g/mol], charge [e]) from forcefield/<ff>/pseudo_atoms.def"""
    table = {}
    for ln in open(FFDIR / ff / "pseudo_atoms.def"):
        t = ln.split()
        if len(t) >= 7 and not ln.lstrip().startswith("#") and t[1] in ("yes", "no"):
            try:
                table[t[0]] = (float(t[5]), float(t[6]))
            except ValueError:
                pass
    return table


def read_gcmc(mid):
    """gcmc XYZ (positions, species, labels) + mass and charge of every site, looked up by
    its label (TraPPE type) in the copied pseudo_atoms.def. The XYZ mass/charge columns are
    not used, so edits to pseudo_atoms.def reach this pipeline directly."""
    ff = forcefield_of(mid)
    gc = read_xyz(STRUCT / "gcmc" / f"{mid}.xyz"); table = load_pseudo_atoms(ff)
    missing = sorted({l for l in gc["label"] if l not in table})
    if missing:
        raise KeyError(f"not in {ff}/pseudo_atoms.def: {', '.join(missing)}")
    gc["mass"] = np.array([table[l][0] for l in gc["label"]])
    gc["charge"] = np.array([table[l][1] for l in gc["label"]])
    return gc


def forcefield_of(mid):
    return "feynman_corrected" if mid == "h2" else "GenericMOFs_corrected"


# ------------------------------------------------------------------ geometry
def width(P, R, u):
    """extent of the union of spheres along unit vector u"""
    x = P @ u
    return (x + R).max() - (x - R).min()


def _diameter_dir(P, R):
    """direction of max_ij |p_i - p_j| + R_i + R_j (the longest extent); None if undefined"""
    best, v = -1.0, None
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = P[i] - P[j]; n = np.linalg.norm(d)
            if n > 1e-9 and n + R[i] + R[j] > best + 1e-12:
                best, v = n + R[i] + R[j], d / n
    return v


def _resolve_degenerate(P, R, vecs):
    """Inside a degenerate inertia subspace any orthonormal basis is a principal frame.
    Pick it by successive longest extent so the result is orientation independent."""
    k = vecs.shape[1]
    if k == 1:
        return vecs
    Pc = P @ vecs                                      # coordinates inside the subspace
    v = _diameter_dir(Pc, R)
    if v is None:
        return vecs
    first = vecs @ v
    if k == 2:
        return np.c_[first, vecs @ np.array([-v[1], v[0]])]
    rest = vecs - np.outer(first, first @ vecs)        # k == 3: project out, recurse on the plane
    q, _ = np.linalg.qr(rest); q = q[:, :2]
    return np.c_[first, _resolve_degenerate(P, R, q)]


def dimensions(pos, radii, mass, rtol=1e-3):
    """a >= b >= c of a set of spheres (centres pos, radii, masses); base-representation method.
      1. r_cm = sum m_i r_i / sum m_i ;  d_i = r_i - r_cm
      2. I = sum m_i (|d_i|^2 * 1 - d_i d_i^T)          (3x3 inertia tensor)
      3. eigenvectors of I = principal axes e_k          (masses set only these directions)
      4. L_k = max(d_i . e_k + R_i) - min(d_i . e_k - R_i)
      5. a, b, c = L_k sorted by length
    Moments equal within rtol (relative): the axes in that subspace are not unique and are
    fixed by successive longest extent (_resolve_degenerate), so the result does not depend
    on the orientation of the input. All masses zero -> unweighted."""
    P, R, m = np.asarray(pos, float), np.asarray(radii, float), np.asarray(mass, float)
    if len(P) == 1:
        return 2 * R[0], 2 * R[0], 2 * R[0], np.eye(3)
    w_ = m if m.sum() > 0 else np.ones(len(P))
    C = P - np.average(P, axis=0, weights=w_)
    I = sum(mi * (r @ r * np.eye(3) - np.outer(r, r)) for mi, r in zip(w_, C))
    val, vec = np.linalg.eigh(I)
    scale = max(abs(val).max(), 1e-12)
    groups = [[0]]                                     # eigenvalues are ascending
    for k in (1, 2):
        if abs(val[k] - val[groups[-1][-1]]) < rtol * scale:
            groups[-1].append(k)
        else:
            groups.append([k])
    axes = np.hstack([_resolve_degenerate(C, R, vec[:, g]) for g in groups])
    ext = sorted([width(C, R, axes[:, k]) for k in range(3)], reverse=True)
    return ext[0], ext[1], ext[2], axes


def dipole_D(pos, q):
    return float(np.linalg.norm(np.asarray(q) @ np.asarray(pos)) * E_A_TO_DEBYE)


# ------------------------------------------------------------------ descriptors
def descriptors(eps, pos_shape, radii, mass, pos_q, q):
    a, b, c, _ = dimensions(pos_shape, radii, mass)
    return dict(epsi_eff_K=float(np.sum(eps)), a_A=a, elo=a / b, fla=b / c, c_A=c,
                mu_D=dipole_D(pos_q, q))


def shape_sites_gcmc(gc, rules, h_radius=None):
    """sites that occupy space: every site with an LJ term (e.g. H2's H_com) plus every
    real atom; real atoms without LJ (H_alc, H_nh3, H_h2) take the forcefield's generic
    element radius ("H_" sigma/2), or h_radius if given (base used 1.20 A).
    Charge-only dummies are dropped."""
    keep, radii = [], []
    for i, (sp, lab) in enumerate(zip(gc["species"], gc["label"])):
        sig = lj(rules, lab)[1]
        if sig > 0:
            keep.append(i); radii.append(sig / 2)
        elif sp != "X":
            keep.append(i); radii.append(h_radius if h_radius else lj(rules, sp)[1] / 2)
    return keep, radii


def gcmc_row(mid):
    gc = read_gcmc(mid); rules = load_mixing_rules(forcefield_of(mid))
    eps = [lj(rules, t)[0] for t in gc["label"]]
    keep, radii = shape_sites_gcmc(gc, rules)
    return descriptors(eps, gc["pos"][keep], radii, gc["mass"][keep], gc["pos"], gc["charge"])


MANUAL, PLACEHOLDER = ("Q", "fle"), "###"


def previous_manual(path):
    """Q / fle already typed into an existing output CSV are kept on rebuild"""
    if not path.exists():
        return {}
    return {r["id"]: {k: r.get(k, PLACEHOLDER) or PLACEHOLDER for k in MANUAL}
            for r in csv.DictReader(open(path))}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ids = sorted(p.stem for p in (STRUCT / "gcmc").glob("*.xyz"))
    if not ids:
        raise SystemExit(f"no gcmc structures found: expected {STRUCT / 'gcmc'}/<id>.xyz (see README)")
    out = HERE / "alchemical_gcmc.csv"
    old = previous_manual(out); blank = {k: PLACEHOLDER for k in MANUAL}
    rows = [dict(id=mid, **gcmc_row(mid), **old.get(mid, blank)) for mid in ids]
    write(out, rows)
    todo = sum(r[k] == PLACEHOLDER for r in rows for k in MANUAL)
    print(f"wrote {out.name} ({len(ids)} molecules); {todo} cells still '{PLACEHOLDER}' (Q, fle)")


if __name__ == "__main__":
    main()
