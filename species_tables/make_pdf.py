"""
Render results.pkl (species.py) into species_tables.tex, species_data.json and ame_catalogue.json; pdflatex turns
species_tables.tex into species_tables.pdf (run it until the .aux and .toc stop changing, three runs from a clean
directory).

    python -B make_pdf.py [results.pkl]

The PDF: the contents; Sec. 1 what is tabulated (the coupling tensors, their 3j blocks, the scope, the conventions and
how to read a group section); Sec. 2 the catalogue of AME states (ame.CATALOGUE) with its notes; Sec. 3 the N = 2
spaces; Sec. 4 the summary; then one section per family of groups, each group with a legend of the catalogue states it
meets and one table per Gamma_mol, a row per triple with the states of its copies (an Abelian group: one grid per
Gamma_mol, side by side), the legends and tables set in up to three panels side by side as the Point Group Atlas sets
its snake tables (snake_table). The layout is the atlas's (atlas_style.tex), with its macros (atlas_macros.tex).
Groups, irreps and the catalogue's LaTeX go through names_tex and atlas_tex, so they print in the atlas's notation, and
the catalogue ids through the PREAMBLE commands \\catP, \\catB, \\catT. Every build re-checks on results.pkl the tables
and every stored copy (check_copies), the rules that Secs. 2 and 3 state and the groups of the occurs-in column of
Table 1 (check_rules), the LU equivalence of the copies of each state (check_lu_classes), the members of the N = 2
spaces that Sec. 3 identifies with named states (check_special_members) and the other claims of Sec. 3
(check_n2_spaces), and it computes the numbers that the text prints.

species_data.json: a list of group dicts {group, atlas_record, order, irreps [{label, id, dim}], tables {Gamma_mol:
[row]}}, keyed by this code's labels. A row of a non-Abelian group is one copy of a coupling space: {r, n, v, dims, N,
tau, state, K (N = 2 only), tensor [[i, j, k, Re, Im], ...] (the entries above 1e-10)}, with tau = 1..N in ascending
K and state its catalogue id (P, B1..B4, T1..T16). A row of an Abelian group is {r, n, v, dims, N, state}, state "P".
ame_catalogue.json: ame.catalogue_json(results), one catalogue row per line.
"""
import itertools
import json
import os
import pickle
import re
import sys
from collections import OrderedDict
from fractions import Fraction

import numpy as np

import ame
import atlas_tex
from species import SCHEMA, input_digest, phase_fix, real1d, tkey

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = {row["id"]: row for row in ame.CATALOGUE}
B_IDS = ("B1", "B2", "B3", "B4")
NO_TENSORS = {"I", "Ih"}      # groups whose tensors are not written out (they are in species_data.json)
CUBIC = ("T", "Th", "Td", "O", "Oh")
AXIAL = ("C1", "Cs", "Ci", "Cn", "Cnv", "Cnh", "S2n", "Dn", "Dnh", "Dnd", "Cinfv", "Dinfh")
STANDIN = {"Cinfv": (r"\grp{C_{13v}}", 13), "Dinfh": (r"\grp{D_{14h}}", 14)}   # pointgroups.linear_groups
NO_EEE = ("C4v", "D4", "D2d", "D4h")    # the non-Abelian axial groups with no admissible (E_a, E_b, E_c) (check_rules)


def names_tex(s):
    """fill the placeholders @@G:<group>@@ and @@L:<group>:<label>@@ of a hand-written string with
    the atlas's printed group and irrep (atlas_tex; this code's group names and labels), so the prose
    names them as the tables do"""
    s = re.sub(r"@@G:(\w+)@@", lambda m: atlas_tex.group_tex(m.group(1)), s)
    s = re.sub(r"@@L:(\w+):(.+?)@@", lambda m: atlas_tex.label_tex(m.group(1), m.group(2)), s)
    assert not re.search(r"@@[GL]:", s), s
    return s


def letters(k):
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(97 + r) + s
    return s


# ----------------------------------------------------------------------------
# coefficient / tensor rendering
# ----------------------------------------------------------------------------
def phase_tex(z):
    """the phase of z with its sign, as a root of unity of order up to 48 when it is one"""
    ph = np.angle(z) / (2 * np.pi) % 1.0
    f = Fraction(ph).limit_denominator(48)
    if abs(float(f) - ph) > 1e-6 and abs(float(f) - ph - 1) > 1e-6:
        return r"+\expe^{\iunit %.4f}" % (2 * np.pi * ph)
    f = f % 1
    table = {Fraction(0): "+", Fraction(1, 2): "-", Fraction(1, 4): r"+\iunit", Fraction(3, 4): r"-\iunit",
             Fraction(1, 3): r"+\rootunity{3}", Fraction(2, 3): r"+\rootunity{3}^{2}",
             Fraction(1, 6): r"-\rootunity{3}^{2}", Fraction(5, 6): r"-\rootunity{3}"}
    if f in table:
        return table[f]
    p, q = f.numerator, f.denominator
    if p * 2 > q:
        return r"+\rootunity{%d}^{-%d}" % (q, q - p)
    return r"+\rootunity{%d}^{%d}" % (q, p)


def mag_tex(x):
    f = Fraction(x * x).limit_denominator(400)
    if abs(float(f) - x * x) > 1e-7:
        return "%.4f" % x
    if f == 1:
        return ""
    # the atlas's rationalised form: sqrt(p/q) = (a/q) sqrt(b) with pq = a^2 b, b squarefree
    a, b = 1, f.numerator * f.denominator
    k = 2
    while k * k <= b:
        while b % (k * k) == 0:
            a, b = a * k, b // (k * k)
        k += 1
    c = Fraction(a, f.denominator)
    root = r"\sqrt{%d}" % b if b > 1 else ""
    if c == 1:
        return root
    return ("%d" % c.numerator if c.denominator == 1 else r"\tfrac{%d}{%d}" % (c.numerator, c.denominator)) + root


def ket_labels(d, cubic):
    if d == 1:
        return [None]
    if d == 2:
        return ["1", "2"]
    if d == 3 and cubic:
        return ["x", "y", "z"]
    return [str(i + 1) for i in range(d)]


def tensor_tex(T, cubic, irreps):
    """irreps: the printed labels of the triple; a one-dimensional partner is written |Gamma>, as the atlas does"""
    dims = T.shape
    labs = [ket_labels(d, cubic) for d in dims]
    terms = []
    for idx in np.ndindex(*dims):
        c = T[idx]
        if abs(c) > 1e-8:
            ket = r"\,".join(labs[p][idx[p]] for p in range(3) if labs[p][0] is not None)
            terms.append((c, ket))
    if all(d == 1 for d in dims):
        return "$" + "".join(r"\ket{%s}" % l for l in irreps) + "$"
    mags = np.array([abs(c) for c, _ in terms])
    common = np.allclose(mags, mags[0], atol=1e-7)
    s = ""
    for c, ket in terms:
        ph = phase_tex(c)
        coef = ph
        if not common:
            m = mag_tex(abs(c))
            coef = ph + m if m else ph
        s += r"%s|%s\rangle" % (coef, ket)
    s = s.lstrip("+")
    if common:
        pre = mag_tex(mags[0])
        if pre:
            s = pre + r"\bigl(" + s + r"\bigr)"
    return "$" + s + "$"


# ----------------------------------------------------------------------------
# catalogue states in the group tables
# ----------------------------------------------------------------------------
def state_tex(sid, var=""):
    """the catalogue id, linked to its row of the catalogue table, followed by the group's variant letter"""
    return r"\hyperlink{cat:%s}{$%s$}%s" % (sid, CAT[sid]["id_tex"], var)


def sort_rows(rows):
    """in catalogue order of the (first) copy's state, then by dims and irrep indices"""
    return sorted(rows, key=lambda r: (ame.INDEX[r["comps"][0]["state"]], r["dims"], r["r"], r["n"], r["v"]))


def assign_variants(g, printable):
    """the legend of the catalogue states met in g, in catalogue order: [(state, letter, info)], one entry per
    different tensor of each state (species.tkey: up to the order of the sectors and a global phase), info = its
    tensor and the first triple (sort_rows order) that has it, with Gamma_mol and K. The letter is "" when the state
    has one tensor in g, and for every state of a group whose tensors are not printed. Sets c["var"], the letter of
    each copy."""
    variants = {}
    for m in g["mols"]:
        for row in sort_rows(g["tables"][m]):
            for c in row["comps"]:
                key = tkey(c["T"]) if printable else None
                vs = variants.setdefault(c["state"], OrderedDict())
                if key not in vs:
                    vs[key] = dict(T=c["T"], rep=(row["r"], row["n"], row["v"]), mol=m, K=c["K"])
                c["var"] = key
    legend, mark = [], {}
    for sid in sorted(variants, key=ame.INDEX.get):
        vs = variants[sid]
        for j, (key, info) in enumerate(vs.items()):
            mark[(sid, key)] = letters(j) if len(vs) > 1 else ""
            legend.append((sid, mark[(sid, key)], info))
    for m in g["mols"]:
        for row in g["tables"][m]:
            for c in row["comps"]:
                c["var"] = mark[(c["state"], c["var"])]
    return legend


def state_counts(g, m):
    """copies per catalogue state in the Gamma_mol = m table, in catalogue order"""
    cnt = OrderedDict((i, 0) for i in ame.IDS)
    for row in g["tables"][m]:
        for c in row["comps"]:
            cnt[c["state"]] += 1
    return cnt


def counts_tex(cnt):
    return ", ".join("%s:\\,%d" % (state_tex(k), v) for k, v in cnt.items() if v)


# ----------------------------------------------------------------------------
# the stored copies, re-checked on results
# ----------------------------------------------------------------------------
def check_copies(results):
    """the 75 groups in pointgroups' order, a table for each real one-dimensional irrep (Gamma_mol) listing each
    admissible triple once with its N (by the characters), and every stored copy: of the shape of its carriers, of unit
    norm, invariant under the current irreps (times chi_mol), with flat marginals, and ame.lookup equal to its stored
    state. Returns the number of copies checked."""
    import pointgroups as pg
    groups = {G.name: G for G in pg.all_finite_groups(10)}
    if [g["name"] for g in results] != list(groups):
        sys.exit("make_pdf.py: results.pkl does not hold the 75 groups in order; rerun species.py without arguments")
    ncopies = 0
    for g in results:
        G = groups[g["name"]]
        if list(G.labels) != g["repo_labels"]:
            sys.exit("make_pdf.py: the irreps of pointgroups and %s differ; rerun species.py" % g["name"])
        if g["mols"] != real1d(G):
            sys.exit("make_pdf.py: the Gamma_mol of %s are not its real one-dimensional irreps; rerun species.py"
                     % g["name"])
        C = np.asarray(G.chars)
        for m in g["mols"]:
            Nm = np.rint(np.einsum("rg,ng,vg,g->rnv", C, C, C, C[m].conj()).real / G.order).astype(int)
            want = {tuple(map(int, t)): int(Nm[t]) for t in zip(*np.nonzero(Nm))}
            have = {(row["r"], row["n"], row["v"]): row["N"] for row in g["tables"][m]}
            if have != want or len(have) != len(g["tables"][m]):
                sys.exit("make_pdf.py: the %s table of %s does not list each admissible triple once, with its N; "
                         "rerun species.py" % (G.labels[m], g["name"]))
            for row in g["tables"][m]:
                idx = (row["r"], row["n"], row["v"])
                D = [np.asarray(G.D[i]) for i in idx]
                carriers = tuple(G.dims[i] for i in idx)
                for c in row["comps"]:
                    T = np.asarray(c["T"])
                    if T.shape != tuple(row["dims"]) or T.shape != carriers:
                        sys.exit("make_pdf.py: a stored copy of %s has shape %s, its row dims %s and its carriers %s; "
                                 "rerun species.py" % (g["name"], T.shape, tuple(row["dims"]), carriers))
                    nrm = float(np.linalg.norm(T))
                    dev = max(np.abs(np.einsum("ia,jb,kc,abc->ijk", D[0][k], D[1][k], D[2][k], T)
                                     - G.chars[m][k] * T).max() for k in range(G.order))
                    flat = ame.flat_dev(T)
                    try:
                        sid = ame.lookup(T)
                    except (LookupError, np.linalg.LinAlgError):
                        sid = None
                    ok = abs(nrm - 1) <= 1e-10 and dev <= 1e-10 and flat <= ame.TOL_FLAT     # NaN fails too
                    if not ok or sid != c["state"]:
                        sys.exit("make_pdf.py: a stored copy of %s fails (norm %.12g, invariance %.1e, flat marginals "
                                 "%.1e, catalogue id %s, stored %s); rerun species.py"
                                 % (g["name"], nrm, dev, flat, sid, c["state"]))
                    ncopies += 1
    return ncopies


# ----------------------------------------------------------------------------
# the rules of Secs. 2 and 3, checked on results
# ----------------------------------------------------------------------------
def swap_t(label):
    """T_{1} <-> T_{2} (with any tag), other labels unchanged"""
    return label[:3] + {"1": "2", "2": "1"}[label[3]] + label[4:] if label.startswith("T_{") else label


def check_rules(results):
    """the statements of Secs. 2 and 3 that hold row by row, among them the two facts of the Labels note: two
    admissible triples that differ only in g/u tags (of the carriers or of Gamma_mol, compared on the atlas's ids, the
    printed names) have the same states, and in I and Ih, replacing every T1 of a triple by T2 and every T2 by T1 keeps
    it admissible with the same states. It also checks the groups, types, K and states of the N = 2 triples, and the
    groups that the occurs-in column of Table 1 names for each state. Returns the numbers the text prints."""
    n333 = n2 = 0
    for g in results:
        L = g["repo_labels"]
        by_untagged, by_triple = {}, {}
        for m in g["mols"]:
            mol = L[m]
            for row in g["tables"][m]:
                labs = [L[i] for i in (row["r"], row["n"], row["v"])]
                states = [c["state"] for c in row["comps"]]
                fmt = tuple(sorted(row["dims"]))
                key = tuple(re.sub(r"[gu]$", "", g["ids"][i]) for i in (row["r"], row["n"], row["v"], m))
                assert by_untagged.setdefault(key, states) == states, ("a g/u tag changes the state", g["name"], key)
                by_triple[(tuple(labs), m)] = states
                if g["abelian"]:
                    assert states == ["P"], (g["name"], labs)
                elif g["family"] in AXIAL:            # non-Abelian axial groups, C_inf v, D_inf h
                    assert states == [{(1, 1, 1): "P", (1, 2, 2): "B1", (2, 2, 2): "T1"}[fmt]], (g["name"], labs)
                if g["family"] in ("O", "Td", "Oh") and fmt == (3, 3, 3):
                    t2 = sum(l.startswith("T_{2") for l in labs) + mol.startswith("A_{2")
                    assert states == ["T3" if t2 % 2 == 0 else "T4"], (g["name"], labs, mol, states)
                    n333 += 1
                if g["family"] in ("I", "Ih") and fmt == (3, 3, 5):
                    ts = sorted(l[3] for l in labs if l.startswith("T_{"))
                    assert states == ["T6" if ts[0] == ts[1] else "T7"], (g["name"], labs, states)
                where = "%s (%s) -> %s" % (g["name"], ", ".join(labs), mol)
                if row["N"] == 2:
                    typ = "".join(sorted(l[0] for l in labs))
                    want = {"TTT": ([1, 2], ["T3", "T4"]), "GHH": ([3, 4], ["T13", "T14"]),
                            "HHH": ([2, 4], ["T15", "T16"])}
                    if g["name"] not in ("T", "Th", "I", "Ih") or typ not in want:
                        n2_fail("an N = 2 triple other than (T,T,T), (G,H,H) and (H,H,H) in T, Th, I and Ih", where)
                    got = ([c["K"] for c in row["comps"]], states)
                    if got != want[typ]:
                        n2_fail("the copies have K %s and states %s, not %s and %s" % (got + want[typ]), where)
                    n2 += 1
                elif row["N"] != 1 or len(row["comps"]) != 1 or row["comps"][0]["K"] is not None:
                    n2_fail("N = %d with K %s outside the N = 2 triples" % (row["N"], [c["K"] for c in row["comps"]]),
                            where)
        if g["family"] in ("I", "Ih"):
            for (labs, m), states in by_triple.items():
                swapped = (tuple(swap_t(x) for x in labs), m)
                assert by_triple.get(swapped) == states, ("T1 <-> T2 changes the state", g["name"], labs)
    if n2 != 45:
        n2_fail("%d N = 2 triples, expected 45" % n2)
    no_eee = {g["name"] for g in results if g["family"] in AXIAL and not g["abelian"]
              and not any(c["state"] == "T1" for m in g["mols"] for row in g["tables"][m] for c in row["comps"])}
    assert no_eee == set(NO_EEE), sorted(no_eee)
    occurs = {}
    for g in results:
        for m in g["mols"]:
            for row in g["tables"][m]:
                for c in row["comps"]:
                    occurs.setdefault(c["state"], set()).add(g["name"])
    axial = {g["name"] for g in results if g["family"] in AXIAL and not g["abelian"]}
    generic = {"P": {g["name"] for g in results}, "B1": axial | {"Td", "O", "Oh"},
               "T1": (axial - set(NO_EEE)) | {"Td", "O", "Oh"}}
    for row in ame.CATALOGUE:
        want = generic.get(row["id"], set(re.findall(r"@@G:(\w+)@@", row["occurs_tex"])))
        if want != occurs.get(row["id"], set()):
            sys.exit("make_pdf.py: Table 1 puts %s in %s, the tables in %s"
                     % (row["id"], sorted(want), sorted(occurs.get(row["id"], set()))))
    # B1 and T1 name the finite non-Abelian axial groups in words, and T1 names its four exceptions (NO_EEE)
    finite = {g["name"] for g in results if g["family"] in AXIAL and not g["abelian"] and not g["truncated"]}
    named = {row["id"]: set(re.findall(r"@@G:(\w+)@@", row["occurs_tex"])) for row in ame.CATALOGUE}
    if named["B1"] != occurs["B1"] - finite or named["T1"] != (occurs["T1"] - finite) | set(NO_EEE):
        sys.exit("make_pdf.py: the B1 or T1 row of Table 1 names groups other than those of the tables")
    return dict(n333=n333)


# ----------------------------------------------------------------------------
# the special members of the N = 2 spaces (Sec. 3), located and checked on results
# ----------------------------------------------------------------------------
TOL_LU = 1e-10        # the local unitaries and exact forms of the named members
_CUBIC = {}


def cubic_group(name):
    """the pointgroups group T, Th, Td, O, Oh, I or Ih, built once (the checks of Sec. 3 use its irrep matrices)"""
    if not _CUBIC:
        import pointgroups as pg
        _CUBIC.update((G.name, G) for G in pg.cubic_groups())
    return _CUBIC[name]


def gl_map(T):
    """the map (X, Y, Z) -> (X.1.1 + 1.Y.1 + 1.1.Z) T on triples of complex matrices, one column per matrix unit E_ab
    of each leg. It is linear in T. For a critical T (every AME state) its kernel dimension is ame.sym_dim(T), since
    the stabiliser of a critical point is the complexification of the compact one (Kempf-Ness)."""
    cols = []
    for leg, d in enumerate(T.shape):
        Tl = np.moveaxis(T, leg, 0)
        for a, b in itertools.product(range(d), repeat=2):
            out = np.zeros_like(Tl)
            out[a] = Tl[b]
            cols.append(np.moveaxis(out, 0, leg).ravel())
    return np.array(cols).T


def gl_dim(T):
    M = gl_map(np.asarray(T, complex) / np.linalg.norm(T))
    s = np.linalg.svd(M, compute_uv=False)
    return int((s <= ame.TOL_SVD).sum()) + M.shape[1] - len(s)


def jump_points(A, B, rng, tries=3):
    """(k0, [psi]): k0 = ame.sym_dim of a generic member of span(A, B), and every member psi (unit norm, one per ray)
    whose ame.sym_dim exceeds k0. With psi = U + lam V for a random orthonormal pair (U, V) of the span, that happens
    exactly where the pencil gl_map(U) + lam gl_map(V) drops below its generic rank r. Every r x r compression of the
    pencil has those lam among its eigenvalues, two random compressions share no other, and the one ray it misses, V,
    is generic. A draw that fails this by bad luck (U or V not generic, or a common eigenvalue that is not a jump) is
    repeated, up to `tries` draws."""
    for _ in range(tries):
        q = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        U, V = q[0, 0] * A + q[1, 0] * B, q[0, 1] * A + q[1, 1] * B
        k0 = ame.sym_dim(V)
        if not ame.sym_dim(U) == k0 == gl_dim(U) == gl_dim(V):
            continue
        MU, MV = gl_map(U), gl_map(V)
        n = MU.shape[1]
        r = n - k0
        ev = []
        for _ in range(2):
            P = rng.normal(size=(r, MU.shape[0])) + 1j * rng.normal(size=(r, MU.shape[0]))
            Q = rng.normal(size=(n, r)) + 1j * rng.normal(size=(n, r))
            ev.append(np.linalg.eigvals(-np.linalg.solve(P @ MV @ Q, P @ MU @ Q)))
        clusters = []
        for x in ev[0]:
            if np.min(abs(ev[1] - x)) > 1e-6 * max(1.0, abs(x)):
                continue                                   # an eigenvalue of this compression only
            for c in clusters:
                if abs(np.mean(c) - x) < 1e-5 * max(1.0, abs(x)):
                    c.append(x)
                    break
            else:
                clusters.append([x])
        pts = [U + np.mean(c) * V for c in clusters]
        pts = [p / np.linalg.norm(p) for p in pts]
        if all(ame.sym_dim(p) > k0 and ame.sym_dim(p) == gl_dim(p) for p in pts):
            return k0, pts
    raise AssertionError("no jump points in %d draws: U, V not generic or a common eigenvalue not a jump" % tries)


def same_ray(S, T):
    """S and T span one ray: the part of T/|T| orthogonal to S/|S| is below TOL_LU"""
    S, T = S / np.linalg.norm(S), T / np.linalg.norm(T)
    return float(np.abs(T - np.vdot(S, T) * S).max()) < TOL_LU


def herm_basis(d):
    """a real basis of the d x d Hermitian matrices"""
    out = []
    for i in range(d):
        for j in range(i, d):
            for z in ((1,) if i == j else (1, 1j)):
                M = np.zeros((d, d), complex)
                M[i, j], M[j, i] = z, np.conj(z)
                out.append(M)
    return out


def leg_apply(M, T, leg):
    return np.moveaxis(np.tensordot(M, T, axes=([1], [leg])), 0, leg)


def diagonalised(T, legs, rng):
    """T in the eigenbases, on the given legs, of a generic Hermitian triple (X, Y, Z) with (X.1.1 + 1.Y.1 + 1.1.Z) T
    = 0, the kernel that ame.sym_dim counts (the one of eight random elements with the widest eigenvalue gaps there),
    and that smallest gap"""
    T = np.asarray(T, complex) / np.linalg.norm(T)
    Hs = [herm_basis(d) for d in T.shape]
    A = np.array([leg_apply(h, T, leg).ravel() for leg, H in enumerate(Hs) for h in H])
    assert A.shape[0] <= 2 * A.shape[1]                    # fewer parameters than real coordinates: U is square
    U, s, _ = np.linalg.svd(np.hstack([A.real, A.imag]), full_matrices=False)
    ker = U[:, s <= ame.TOL_SVD]
    best = (-1.0, None)
    for _ in range(8):
        c = ker @ rng.normal(size=ker.shape[1])
        gap, k, Ws = np.inf, 0, {}
        for leg, H in enumerate(Hs):
            if leg in legs:
                w, Ws[leg] = np.linalg.eigh(sum(ck * h for ck, h in zip(c[k:k + len(H)], H)))
                gap = min(gap, float(np.diff(w).min()))
            k += len(H)
        best = max(best, (gap, Ws), key=lambda x: x[0])
    for leg, W in best[1].items():
        T = leg_apply(W.conj().T, T, leg)
    return T, best[0]


def perm_blocks(T, legs):
    """the nonzero blocks of T over the given legs (the remaining leg grouped), if they sit on a permutation pattern
    (one block for each index of each of these legs); else None"""
    Tt = np.transpose(T, list(legs) + [x for x in range(3) if x not in legs])
    supp = [tuple(i) for i in np.argwhere(np.linalg.norm(Tt.reshape(Tt.shape[:len(legs)] + (-1,)), axis=-1) > 1e-8)]
    d = T.shape[legs[0]]
    if len(supp) != d or any(len({s[p] for s in supp}) != d for p in range(len(legs))):
        return None
    return [Tt[s] for s in supp]


def is_ghz(T, rng):
    """GHZ_d up to LU: in the eigenbases of a generic kernel element (diagonalised), T sits on a permutation of the
    three legs with moduli 1/sqrt(d)"""
    Td, gap = diagonalised(T, (0, 1, 2), rng)
    blocks = perm_blocks(Td, (0, 1, 2))
    return gap > 1e-3 and blocks is not None and np.allclose(np.abs(blocks), 1 / np.sqrt(T.shape[0]), atol=TOL_LU)


def is_simplex_ghz(T, gleg, rng):
    """5^(-1/2) sum_i |v_i>|i>|i> up to LU, the v_i at the vertices of a regular simplex in the 4-dimensional leg
    gleg: in the eigenbases of a generic kernel element on the other two legs, T sits on a permutation of them, with
    the vectors v_i/sqrt5 on gleg. Their Gram matrix G_ij = <v_i|v_j> has diagonal 1, |G_ij| = 1/4 and every triangle
    product G_ij G_jk G_ki = -1/64, the simplex's (so a diagonal unitary makes G the simplex's Gram matrix)"""
    legs = tuple(x for x in range(3) if x != gleg)
    Td, gap = diagonalised(T, legs, rng)
    v = perm_blocks(Td, legs)                              # the v_i/sqrt5
    if gap < 1e-3 or v is None:
        return False
    G = 5 * np.array([[np.vdot(x, y) for y in v] for x in v])
    off = [abs(G[i, j]) for i, j in itertools.permutations(range(5), 2)]
    tri = [G[i, j] * G[j, k] * G[k, i] for i, j, k in itertools.permutations(range(5), 3)]
    return (np.allclose(np.diag(G), 1, atol=TOL_LU) and np.allclose(off, 1 / 4, atol=TOL_LU)
            and np.allclose(tri, -1 / 64, atol=TOL_LU))


def outer_automorphism(G):
    """an automorphism alpha of I with chi_T1(alpha(g)) = chi_T2(g), as the index array g -> alpha(g): the images of
    generators a (order 5), b (order 2, ab of order 3), extended along words and checked on all products"""
    mul = np.asarray(G.mul)
    ch = {lab: np.asarray(c) for lab, c in zip(G.labels, G.chars)}

    def order(g):
        x, k = g, 1
        while x != 0:
            x, k = mul[x, g], k + 1
        return k

    def extend(gens, imgs):
        """the map x -> x' along words in gens -> imgs, or None if a word has two images or the map is not onto"""
        img, frontier = {0: 0}, [0]
        while frontier:
            nxt = []
            for x in frontier:
                for s, s2 in zip(gens, imgs):
                    y, y2 = mul[x, s], mul[img[x], s2]
                    if y not in img:
                        img[y] = y2
                        nxt.append(y)
                    elif img[y] != y2:
                        return None
            frontier = nxt
        if len(img) != G.order or len(set(img.values())) != G.order:
            return None
        return np.array([img[g] for g in range(G.order)])

    ords = np.array([order(g) for g in range(G.order)])
    a = next(g for g in range(G.order) if ords[g] == 5)
    b = next(g for g in range(G.order) if ords[g] == 2 and ords[mul[a, g]] == 3)
    assert extend((a, b), (a, b)) is not None, "a and b do not generate I"
    for a2 in [g for g in range(G.order) if ords[g] == 5 and abs(ch["T_{1}"][g] - ch["T_{2}"][a]) < 1e-9]:
        for b2 in [g for g in range(G.order) if ords[g] == 2 and ords[mul[a2, g]] == 3]:
            al = extend((a, b), (a2, b2))
            if al is not None and (mul[al[:, None], al[None, :]] == al[mul]).all():
                assert np.allclose(ch["T_{1}"][al], ch["T_{2}"]) and \
                    all(np.allclose(ch[lab][al], ch[lab]) for lab in ("A", "G", "H"))
                return al
    raise AssertionError("no outer automorphism of I")


def aut_unitary(G, al, i):
    """the unitary V on irrep i of I with D_i(alpha(g)) = V D_i(g) V^dag for every g (the twirl of a random matrix,
    a multiple of V by Schur's lemma)"""
    D = np.asarray(G.D[i])
    X = np.random.default_rng(3).normal(size=D.shape[1:]).astype(complex)
    u, _, wh = np.linalg.svd(sum(D[al[g]] @ X @ D[g].conj().T for g in range(G.order)))
    V = u @ wh
    res = max(float(np.abs(D[al[g]] @ V - V @ D[g]).max()) for g in range(G.order))
    assert res < TOL_LU, ("outer automorphism", G.labels[i], res)
    return V


def check_formats():
    """the formats of Sec. 2, by Bryan, Reichstein and Van Raamsdonk's expected dimension Delta = d1 d2 d3 - d1^2 -
    d2^2 - d3^2 + 2, their prod_i d_i - 1 - sum_i (d_i^2 - 1): among the tripartite formats of the catalogue, (2,2,2)
    and (2,3,3), where AME fixes the state (their Thm. 1.3), are the only ones with Delta <= 0, and every other has
    Delta > 0, so its AME states form, up to LU, a space of positive dimension (their Thm. 1.2, with Kempf-Ness). Also
    the four formats with two catalogue states each (Sec. 2, Two states in one format)"""
    fmts = sorted({row["format"] for row in ame.CATALOGUE if min(row["format"]) > 1})
    delta = {f: f[0] * f[1] * f[2] - sum(d * d for d in f) + 2 for f in fmts}
    assert all(delta[f] == int(np.prod(f)) - 1 - sum(d * d - 1 for d in f) for f in fmts)
    assert {f: v for f, v in delta.items() if v <= 0} == {(2, 2, 2): -2, (2, 3, 3): -2}, delta
    rows = {f: sum(r["format"] == f for r in ame.CATALOGUE) for f in fmts}
    assert max(rows.values()) == 2 and [f for f in fmts if rows[f] == 2] == [(3, 3, 3), (3, 3, 5), (4, 5, 5),
                                                                             (5, 5, 5)], rows


def check_special_members(results):
    """Sec. 3's special members, on the stored copies of every N = 2 row: each member that the text calls
    LU-equivalent to a named state, with an exact certificate of the equivalence. In (T,T,T) these are the exact forms
    of epsilon, |epsilon| and the [[3,1,2]]_3 codewords, and a diagonal unitary of cube roots of unity on two legs; in
    (G,H,H) the simplex-GHZ form and in (H,H,H) the GHZ_5 form (diagonalised); for (2 K2 + 3 sqrt5 K4)/7 a local
    unitary V x V x V, V the unitary that an outer automorphism of I induces on its H (internal; the PDF prints only the
    equivalence). The members are located, and delta computed, as the rays of the span where gl_map drops rank
    (jump_points); as an internal check, these rays are exactly the listed members, with the internal invariant
    ame.sym_dim each has (the PDF prints neither). The copies of one type and leg order are the same tensors in every
    group, so V is applied to the (H,H,H) row of I. Returns the relative phase delta of the two (G,H,H) points,
    sqrt(5/8) K3 +- e^{i delta} sqrt(3/8) K4 with 0 <= delta < pi, the same in all 27 rows."""
    rng = np.random.default_rng(0)
    w = np.exp(2j * np.pi / 3)
    eps = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        eps[p] = np.linalg.det(np.eye(3)[list(p)]) / np.sqrt(6)
    cyc = np.zeros((3, 3, 3))
    for k in range(3):
        cyc[k, (k + 1) % 3, (k + 2) % 3] = 1 / np.sqrt(3)  # |k,k+1,k+2>, a codeword of the [[3,1,2]]_3 code
    first, deltas, n2 = {}, [], 0
    for g in results:
        for m in g["mols"]:
            for row in g["tables"][m]:
                if row["N"] != 2:
                    continue
                n2 += 1
                labs = [g["repo_labels"][i] for i in (row["r"], row["n"], row["v"])]
                where = "%s %s -> %s" % (g["name"], labs, g["repo_labels"][m])
                A, B = (np.asarray(c["T"]) for c in row["comps"])
                Ks = [c["K"] for c in row["comps"]]
                A0, B0 = first.setdefault("".join(lab[0] for lab in labs), (A, B))
                assert max(np.abs(A - A0).max(), np.abs(B - B0).max()) < 1e-12, ("copies differ", where)
                k0, pts = jump_points(A, B, rng)
                if Ks == [1, 2]:                            # (T,T,T): K = 1 is epsilon, K = 2 is |epsilon|
                    assert k0 == 4 and np.abs(A - eps).max() < TOL_LU and np.abs(B - abs(eps)).max() < TOL_LU, where
                    assert np.abs((A + B) / np.sqrt(2) - cyc).max() < TOL_LU, where
                    assert np.abs((B - A) / np.sqrt(2) - cyc.transpose(0, 2, 1)).max() < TOL_LU, where
                    named = [(A, 10), ((A + B) / np.sqrt(2), 6), ((B - A) / np.sqrt(2), 6)]
                    for s in (1, -1):         # LU-equivalent to epsilon: diagonal cube roots of unity on two legs
                        psi = (A + s * 1j * np.sqrt(3) * B) / 2
                        named.append((psi, 10))
                        imgs = (leg_apply(np.diag(w ** np.array(e[3:])),
                                          leg_apply(np.diag(w ** np.array(e[:3])), psi, l1), l2)
                                for l1, l2 in itertools.combinations(range(3), 2)
                                for e in itertools.product(range(3), repeat=6))
                        assert any(np.abs(X - np.vdot(A, X) * A).max() < TOL_LU for X in imgs), ("epsilon", where)
                elif Ks == [3, 4]:                          # (G,H,H): the two simplex-GHZ points
                    mus = [np.vdot(B, p) / np.vdot(A, p) for p in pts]
                    assert k0 == 2 and len(pts) == 2 and abs(mus[0] + mus[1]) < 1e-9 and \
                        all(abs(abs(mu) ** 2 - 3 / 5) < 1e-9 for mu in mus), where
                    delta = float(np.angle(mus[0]) % np.pi)
                    deltas.append(delta)
                    named = [(np.sqrt(5 / 8) * A + s * np.exp(1j * delta) * np.sqrt(3 / 8) * B, 6) for s in (1, -1)]
                    gleg = [lab[0] for lab in labs].index("G")
                    assert all(is_simplex_ghz(T, gleg, rng) for T, _ in named), ("simplex-GHZ", where)
                else:                                       # (H,H,H): K = 2, its image, the two GHZ_5 points
                    assert Ks == [2, 4] and k0 == 2, where
                    ghz5 = [np.sqrt(4 / 7) * A + np.sqrt(3 / 7) * (np.sqrt(15) + 7j * s) / 8 * B for s in (1, -1)]
                    named = [(A, 5), ((2 * A + 3 * np.sqrt(5) * B) / 7, 5)] + [(T, 10) for T in ghz5]
                    assert all(is_ghz(T, rng) for T in ghz5), ("GHZ_5", where)
                # internal: the jump points are exactly the listed members, with these sym_dim values (not printed)
                assert len(pts) == len(named), ("%d jump points, %d named" % (len(pts), len(named)), where)
                for T, d in named:
                    hits = [p for p in pts if same_ray(p, T)]
                    assert len(hits) == 1 and ame.sym_dim(hits[0]) == d, ("named member", d, where)
    if n2 != 45 or len(deltas) != 27:
        n2_fail("%d N = 2 triples, %d of them (G,H,H), expected 45 and 27" % (n2, len(deltas)))
    if max(deltas) - min(deltas) >= 1e-9:
        n2_fail("delta varies by %.1e over the (G,H,H) triples" % (max(deltas) - min(deltas)))
    # (2 K2 + 3 sqrt5 K4)/7 is LU-equivalent to K2, the Wigner 3j(2,2,2): V x V x V maps K2 onto its ray, V the
    # unitary on H of an outer automorphism of I. The copies of (H,H,H) are the same in every group (above).
    G = cubic_group("I")
    gI = next(x for x in results if x["name"] == "I")
    assert gI["repo_labels"] == list(G.labels) and gI["dims"] == list(G.dims)
    V = aut_unitary(G, outer_automorphism(G), G.labels.index("H"))
    hhh = [row for m in gI["mols"] for row in gI["tables"][m]
           if [gI["repo_labels"][i] for i in (row["r"], row["n"], row["v"])] == ["H", "H", "H"]]
    assert len(hhh) == 1 and [c["K"] for c in hhh[0]["comps"]] == [2, 4], len(hhh)
    A, B = (np.asarray(c["T"]) for c in hhh[0]["comps"])
    image = np.einsum("ia,jb,kc,abc->ijk", V, V, V, A)
    assert same_ray(image, (2 * A + 3 * np.sqrt(5) * B) / 7), "(2 K2 + 3 sqrt5 K4)/7"
    return float(np.mean(deltas))


def check_lu_classes(results):
    """every copy of T3..T16 is LU-equivalent, up to carrier order, to every other copy of its id: equal to it to
    TOL_LU up to carrier order and phase, or linked by the outer automorphism of I, whose intertwiners W
    (D_i(alpha(g)) = W_i D_sw(i)(g) W_i^dag, sw = swap_t) map each N = 1 copy of I onto the copy of the swapped
    triple. T1 and T2 need no check: AME fixes the state in (2,2,2) and (2,3,3)"""
    def canon(T):
        Xs = [phase_fix(np.transpose(T, p)) for p in itertools.permutations(range(3))]
        return min(Xs, key=lambda X: (X.shape, tuple(np.round(X.real.ravel() * 1e5).astype(int)),
                                      tuple(np.round(X.imag.ravel() * 1e5).astype(int))))
    forms, parent = {}, {}

    def node(sid, T):
        X, fs = canon(np.asarray(T)), forms.setdefault(sid, [])
        k = next((k for k, Y in enumerate(fs) if X.shape == Y.shape and np.abs(X - Y).max() < TOL_LU), len(fs))
        if k == len(fs):
            fs.append(X)
            parent[(sid, k)] = (sid, k)
        return sid, k

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for g in results:
        for m in g["mols"]:
            for row in g["tables"][m]:
                for c in row["comps"]:
                    if min(row["dims"]) > 1 and c["state"] not in ("T1", "T2"):
                        node(c["state"], c["T"])
    G, gI = cubic_group("I"), next(x for x in results if x["name"] == "I")
    al, L = outer_automorphism(G), gI["repo_labels"]
    sw = [L.index(swap_t(x)) for x in L]
    W = []
    for i, j in enumerate(sw):
        Di, Dj = np.asarray(G.D[i]), np.asarray(G.D[j])
        X = np.random.default_rng(3).normal(size=(Di.shape[1], Dj.shape[1]))
        u, _, wh = np.linalg.svd(sum(Di[al[g]] @ X @ Dj[g].conj().T for g in range(G.order)))
        W.append(u @ wh)
        if max(float(np.abs(Di[al[g]] @ W[i] - W[i] @ Dj[g]).max()) for g in range(G.order)) > TOL_LU:
            sys.exit("make_pdf.py: no intertwiner of the outer automorphism on %s" % L[i])
    rows = {(x["r"], x["n"], x["v"]): x for x in gI["tables"][gI["mols"][0]]}
    for idx, row in rows.items():
        if row["N"] == 1 and min(row["dims"]) > 1:
            sid, T = row["comps"][0]["state"], row["comps"][0]["T"]
            other = rows[tuple(sw[i] for i in idx)]["comps"][0]["T"]
            img = np.einsum("ai,bj,ck,abc->ijk", *[W[i].conj() for i in idx], np.asarray(T))
            if node(sid, img) != node(sid, other):
                sys.exit("make_pdf.py: the outer automorphism of I does not map %s onto its swapped triple" % (idx,))
            parent[root(node(sid, T))] = root(node(sid, other))
    bad = sorted(s for s, fs in forms.items() if len({root((s, k)) for k in range(len(fs))}) > 1)
    if bad:
        sys.exit("make_pdf.py: copies of %s are not shown LU-equivalent" % ", ".join(bad))


# ----------------------------------------------------------------------------
# the other claims of Sec. 3 on the N = 2 spaces, checked on results
# ----------------------------------------------------------------------------
TOL_N2 = 1e-10        # every identity that Sec. 3 states


def n2_fail(what, where=""):
    raise SystemExit("make_pdf.py: a claim of Sec. 3 fails: %s%s" % (what, " (%s)" % where if where else ""))


def n2_named(claim, named, where):
    """the largest deviation of the checks named = [(what fails, deviation)] of a claim, after n2_fail with each
    check above TOL_N2 if there is one"""
    bad = ["%s (%.1e)" % (what, x) for what, x in named if x > TOL_N2]
    if bad:
        n2_fail("%s: %s" % (claim, "; ".join(bad)), where)
    return max(x for _, x in named)


def schur_dev(C):
    """max over the carriers X and the tensors a, b of C of |Tr_{other two} |C_b><C_a| - delta_ab I/d_X|"""
    out = 0.0
    for X, d in enumerate(C[0].shape):
        M = [legs_first(c, (X,)) for c in C]
        out = max([out] + [float(np.abs(M[b] @ M[a].conj().T - (a == b) * np.eye(d) / d).max())
                           for a in range(len(C)) for b in range(len(C))])
    return out


def legs_first(T, legs):
    """T as a matrix, with the given legs (in this order) as its row index and the remaining legs as its column"""
    rest = [x for x in range(3) if x not in legs]
    return np.transpose(T, list(legs) + rest).reshape(int(np.prod([T.shape[x] for x in legs])), -1)


def span_projector(vecs):
    Q, _ = np.linalg.qr(np.array([np.ravel(v) for v in vecs]).T)
    return Q @ Q.conj().T


def commutant(G, i, k, cache):
    """an orthonormal basis of the operators on the carriers i, k (in this order) that commute with D_i(g) x D_k(g)
    for every g: the eigenvectors of eigenvalue 1 of the twirl superoperator (1/|G|) sum_g D(g) x conj D(g) on
    row-major vec(O). The twirl is a projector of rank (1/|G|) sum_g |chi_i(g) chi_k(g)|^2. Cached per
    (group, i, k)."""
    key = (G.name, i, k)
    if key not in cache:
        D = np.einsum("gab,gcd->gacbd", np.asarray(G.D[i]), np.asarray(G.D[k]))
        n = D.shape[1] * D.shape[2]
        D = D.reshape(G.order, n * n)                          # vec D(g), D(g) = D_i(g) x D_k(g)
        P = D.T @ D.conj() / G.order                           # P[(a,b),(c,d)] = mean D_ab conj D_cd
        S = P.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)
        w, V = np.linalg.eigh((S + S.conj().T) / 2)
        rank = int(round(float(np.sum(np.abs(G.chars[i] * G.chars[k]) ** 2)) / G.order))
        if np.abs(w * (1 - w)).max() > TOL_N2 or int((w > 0.5).sum()) != rank:
            n2_fail("the twirl is not a projector of rank %d" % rank, "%s, %s x %s" % (G.name, G.labels[i],
                                                                                         G.labels[k]))
        cache[key] = V[:, w > 0.5].T.reshape(-1, n, n)
    return cache[key]


def on_space(C, legs, ops):
    """the 2 x 2 matrices <C_a|O|C_b> of the operators O (an array) on the carriers `legs` of the two copies C, and
    the largest entry of the parts of the O C_b outside the space"""
    M = np.array([legs_first(c, legs) for c in C])
    img = np.einsum("kij,bjr->kbir", ops, M)
    L = np.einsum("air,kbir->kab", M.conj(), img)
    return L, float(np.abs(img - np.einsum("kab,air->kbir", L, M)).max())


def extension_space(C):
    """S, the orthogonal complement of the one-carrier images E C_a of the space (E the matrix units of a carrier), as
    an orthonormal basis (columns), and the singular values of the images"""
    cols = [leg_apply(E, c, X).ravel() for X, d in enumerate(C[0].shape) for E in np.eye(d * d).reshape(-1, d, d)
            for c in C]
    U, s, _ = np.linalg.svd(np.array(cols).T)
    return U[:, int((s > TOL_N2).sum()):], s


def codeword(s):
    """the codeword 3^(-1/2) sum_k |k, k+s, k+2s> (mod 3) of the three-qutrit code [[3,1,2]]_3"""
    T = np.zeros((3, 3, 3))
    for k in range(3):
        T[k, (k + s) % 3, (k + 2 * s) % 3] = 1 / np.sqrt(3)
    return T


def antisymmetriser(d):
    """the projector onto the totally antisymmetric subspace of (C^d)^(x3)"""
    A = np.zeros((d ** 3, d ** 3))
    base = np.arange(d ** 3).reshape(d, d, d)
    for p in itertools.permutations(range(3)):
        A[base.ravel(), np.transpose(base, p).ravel()] += np.linalg.det(np.eye(3)[list(p)]) / 6
    return A


def check_n2_spaces(results):
    """the claims of Sec. 3 that check_rules and check_special_members leave, on the stored copies of every N = 2
    triple, with SystemExit on the first mismatch. The N = 2 triples are (T,T,T), (G,H,H) and (H,H,H), counted by type.
    The first nonzero entry of each copy, the first above 1e-10 in C order (as species_data.json lists them), is real
    and positive. The Schur identity Tr_{other two} |C_b><C_a| = delta_ab I/d holds on every carrier. For every pair of
    carriers, the operators that commute with G (commutant) map the space into itself and give all 2 x 2 matrices on it.
    For two equal carriers of spin j (ame.SPIN), with the spin matrices U^dag J U of the subduction isometry U
    (ame.intertwiner, which needs b = 1), (J + J')^2 = K(K+1) and J.J' = K(K+1)/2 - j(j+1) on the copies. S is the
    orthogonal complement of the one-carrier images (extension_space). For (T,T,T), the T matrices at the proper
    rotations are the rotation matrices, the copies are (|1> -+ |2>)/sqrt2 in the codewords |s> (codeword), and
    1 x X^2 x X maps the space onto span{|0>, |1>} (Cleve, Gottesman and Lo). There S = span{|000>, |111>, |222>}, and
    for random phases theta, 3^(-1/2) sum_k e^(i theta_k)|kkk> extends the space to a code that diag(e^(-i theta_k/3))
    on every carrier maps onto [[3,1,2]]_3, fixing the space. For (G,H,H), S = 0. For (H,H,H), the three H carriers have
    the same matrices at the proper rotations, S is their totally antisymmetric subspace, and every marginal of three
    random states of S has the top eigenvalue 1/3. Returns the counts, the eigenvalues of J.J' by type and the measured
    deviations."""
    rng = np.random.default_rng(2)
    cache, count, jdot = {}, {}, {}
    dev = dict(phase=0.0, schur=0.0, leak=0.0, jj=0.0, ttt=0.0, hhh=0.0, rank4=np.inf, ghh=np.inf)
    xs = np.roll(np.eye(3), 1, axis=0)                     # |k> -> |k+1>
    for g in results:
        for m in g["mols"]:
            for row in g["tables"][m]:
                if row["N"] != 2:
                    continue
                G = cubic_group(g["name"])
                if list(G.labels) != g["repo_labels"]:
                    n2_fail("the irreps of pointgroups and results.pkl differ", g["name"])
                idx = (row["r"], row["n"], row["v"])
                labs = [G.labels[i] for i in idx]
                where = "%s (%s) -> %s" % (g["name"], ", ".join(labs), G.labels[m])
                typ = "".join(sorted(lab[0] for lab in labs))
                if g["name"] not in {"TTT": ("T", "Th"), "GHH": ("I", "Ih"), "HHH": ("I", "Ih")}.get(typ, ()):
                    n2_fail("an N = 2 triple other than (T,T,T) in T, Th and (G,H,H), (H,H,H) in I, Ih", where)
                count[typ] = count.get(typ, 0) + 1
                C = [np.asarray(c["T"], complex) for c in row["comps"]]
                Ks = [c["K"] for c in row["comps"]]
                for c, K in zip(C, Ks):
                    f = c.ravel()[np.argmax(np.abs(c.ravel()) > 1e-10)]
                    dev["phase"] = max(dev["phase"], abs(f.imag))
                    if not (f.real > 0 and abs(f.imag) <= TOL_N2):
                        n2_fail("the first nonzero entry of the K = %d copy is %.3g%+.3gi, not real and positive"
                                % (K, f.real, f.imag), where)
                dev["schur"] = max(dev["schur"], n2_named("the Schur identity", [("deviation", schur_dev(C))], where))
                vals = set()
                for legs in itertools.combinations(range(3), 2):
                    L, leak = on_space(C, legs, commutant(G, idx[legs[0]], idx[legs[1]], cache))
                    s = np.linalg.svd(L.reshape(len(L), 4), compute_uv=False)
                    s4 = s[3] if len(s) > 3 else 0.0
                    dev["leak"], dev["rank4"] = max(dev["leak"], leak), min(dev["rank4"], s4)
                    if leak > TOL_N2 or s4 < 1e-6:
                        n2_fail("two-carrier couplings: leak %.1e, fourth singular value %.1e" % (leak, s4), where)
                    if labs[legs[0]][0] != labs[legs[1]][0]:
                        continue
                    j = ame.SPIN[labs[legs[0]][0]]
                    if any(G.dims[idx[x]] != 2 * j + 1 for x in legs):
                        n2_fail("the %s carriers are not the whole D^(%d)" % (labs[legs[0]][0], j), where)
                    try:
                        Js = [[u.conj().T @ Ja @ u for Ja in ame.spin(j)]
                              for u in (ame.intertwiner(G, idx[x], j) for x in legs)]
                    except ValueError as e:
                        n2_fail(str(e), where)
                    one = np.eye(2 * j + 1)
                    tot = [np.kron(a, one) + np.kron(one, b) for a, b in zip(*Js)]
                    L, leak = on_space(C, legs, np.array([sum(np.kron(a, b) for a, b in zip(*Js)),
                                                          sum(t @ t for t in tot)]))
                    want = [np.diag([K * (K + 1) / 2 - j * (j + 1) for K in Ks]), np.diag([K * (K + 1) for K in Ks])]
                    dev["jj"] = max(dev["jj"], n2_named("two equal carriers", [(
                        "J.J' and (J + J')^2 are not K(K+1)/2 - j(j+1) and K(K+1) on the copies",
                        max(leak, float(np.abs(L - np.array(want)).max())))], where))
                    vals.add(tuple(int(round(x)) for x in np.diag(want[0])))
                npairs = sum(labs[a][0] == labs[b][0] for a, b in itertools.combinations(range(3), 2))
                if npairs != {"TTT": 3, "GHH": 1, "HHH": 3}[typ]:
                    n2_fail("%d pairs of equal carriers" % npairs, where)
                if len(vals) != 1 or jdot.setdefault(typ, vals) != vals:
                    n2_fail("the eigenvalues of J.J' are %s, those of the first %s triple %s"
                            % (sorted(vals), typ, sorted(jdot[typ])), where)
                S, s = extension_space(C)
                if typ == "TTT":
                    th = rng.uniform(0, 2 * np.pi, 3)
                    psi = np.zeros((3, 3, 3), complex)
                    psi[range(3), range(3), range(3)] = np.exp(1j * th) / np.sqrt(3)
                    diag = np.diag(np.exp(-1j * th / 3))
                    img = [leg_apply(diag, leg_apply(diag, leg_apply(diag, c, 0), 1), 2) for c in C + [psi]]
                    named = [
                        ("partners not Cartesian", max(float(np.abs(np.asarray(G.D[i])[k] - R).max()) for i in idx
                                                       for k, R in enumerate(G.elems) if np.linalg.det(R) > 0)),
                        ("K = 1 copy not (|1> - |2>)/sqrt2",
                         float(np.abs(C[0] - (codeword(1) - codeword(2)) / np.sqrt(2)).max())),
                        ("K = 2 copy not (|1> + |2>)/sqrt2",
                         float(np.abs(C[1] - (codeword(1) + codeword(2)) / np.sqrt(2)).max())),
                        ("space not span{|1>, |2>}",
                         float(np.abs(span_projector(C) - span_projector([codeword(1), codeword(2)])).max())),
                        ("1 x X^2 x X does not map it onto span{|0>, |1>}",
                         float(np.abs(span_projector([leg_apply(xs, leg_apply(xs @ xs, c, 1), 2) for c in C])
                                      - span_projector([codeword(0), codeword(1)])).max())),
                        ("S not span{|000>, |111>, |222>}",
                         float(np.abs(S @ S.conj().T - span_projector([np.eye(27)[13 * k] for k in range(3)])).max())),
                        ("an AME vector of S does not extend the space", schur_dev(C + [psi])),
                        ("the diagonal unitary does not give [[3,1,2]]_3",
                         float(np.abs(span_projector(img) - span_projector([codeword(k) for k in range(3)])).max())),
                        ("the diagonal unitary moves the space",
                         float(np.abs(span_projector(img[:2]) - span_projector(C)).max()))]
                    dev["ttt"] = max(dev["ttt"], n2_named("(T,T,T) against [[3,1,2]]_3", named, where))
                elif typ == "GHH":
                    dev["ghh"] = min(dev["ghh"], float(s[-1]))
                    if S.shape[1]:
                        n2_fail("(G,H,H): S has dimension %d, not 0" % S.shape[1], where)
                else:
                    lam = 0.0
                    for _ in range(3):                     # a spot check: no state of S is AME
                        psi = (S @ (rng.normal(size=S.shape[1]) + 1j * rng.normal(size=S.shape[1]))).reshape(5, 5, 5)
                        psi /= np.linalg.norm(psi)
                        lam = max([lam] + [abs(float(np.linalg.eigvalsh(legs_first(psi, (X,)) @ legs_first(psi, (X,))
                                                                        .conj().T)[-1]) - 1 / 3) for X in range(3)])
                    named = [
                        ("H matrices differ at a proper rotation",
                         max(float(np.abs(np.asarray(G.D[i])[k] - np.asarray(G.D[idx[0]])[k]).max()) for i in idx
                             for k, R in enumerate(G.elems) if np.linalg.det(R) > 0)),
                        ("S not the totally antisymmetric subspace",
                         float(np.abs(S @ S.conj().T - antisymmetriser(5)).max())),
                        ("a marginal of a state of S without the top eigenvalue 1/3", lam)]
                    dev["hhh"] = max(dev["hhh"], n2_named("(H,H,H) against Lambda^3 C^5", named, where))
    if set(count) != {"TTT", "GHH", "HHH"}:
        n2_fail("the N = 2 types are %s" % sorted(count))
    return dict(count=count, jdot={typ: vals.pop() for typ, vals in jdot.items()}, dev=dev, commutants=len(cache))


def standin_signs(g):
    """the printed irreps of a linear group whose partner 2, in the stand-in's own basis, is minus the atlas's partner
    2 (atlas_print.json: sign2); every other irrep of dimension 2 must have the identity map"""
    out = []
    for lab, key, d in zip(g["labels"], g["repo_labels"], g["dims"]):
        pm = atlas_tex.partner_map(g["name"], key)
        if d == 2 and pm == "sign2":
            out.append("$%s$" % lab)
        else:
            assert d == 1 or pm == "identity", (g["name"], key, pm)
    return out


def and_list(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ----------------------------------------------------------------------------
# the layout of the tables
# ----------------------------------------------------------------------------
# A list-like table is set as the atlas sets its snake tables. Its columns repeat in up to SNAKE_PANELS panels side
# by side, SNAKE_GAP pt apart. There are as many panels as fit the text width by the estimated width of a panel
# (cell_width, times SNAKE_SLACK), with at least SNAKE_MIN_ROWS rows each. The rows run down the first panel, then
# down the next, in groups of about SNAKE_GROUP_LINES lines (snake_layout). A group does not break across pages, and a
# heavier rule (\partrule) separates the groups. A page can end only after such a rule, which then closes the page like
# a bottom rule.
TEXT_WIDTH = 506.0          # pt: a4paper with margins of 1.6 cm
SNAKE_PANELS = 3
SNAKE_GAP = 18.0
SNAKE_MIN_ROWS = 4
SNAKE_GROUP_LINES = 12
SNAKE_SLACK = 1.02
# \needspace keeps a heading with what follows it, up to the first place where a page may end. Its room (pt) is that of
# the heading and its lines of text, plus a table's head and first group or a line of grids.
HEADING_ROOM = {"section": 44.0, "subsection": 36.0, "subsubsection": 34.0}
TEXT_LINE = 12.0            # a line of text
TABLE_HEAD = 36.0           # the space before a table (\LTpre), its head and the rules around its first group, with
                            # a margin of a few pt
TABLE_LINE = 9.5            # a line of a \footnotesize table
GRID_LINE = 9.0             # a line of a \scriptsize grid
GRID_ROOM = 30.0            # the space before a line of grids, the title and the rules of a grid

# cell_width: the width (pt) of a cell at \footnotesize (Latin Modern at 8 pt, scripts at 6 pt), from the widths of its
# glyphs. A script is SCRIPT times as wide, plus \scriptspace, and two scripts on one nucleus stack. A binary operator
# or a relation counts with its spaces, and a comma in math with its thin space. On the cells of these tables it is at
# most 2% below TeX's width and at most 10% above it.
SCRIPT, SCRIPT_SPACE, PRIME = 0.87, 0.5, 2.2
MATH_GLYPHS = {"(": 3.31, ")": 3.31, "|": 2.36, ",": 3.78, ".": 2.36, "+": 10.39, "-": 10.39, "=": 11.33}
TEXT_GLYPHS = {" ": 2.83, "i": 2.36, "l": 2.36, "t": 3.31, "r": 3.33, "s": 3.35, "c": 3.78, "e": 3.78, "a": 4.25,
               "o": 4.25, "b": 4.72, "d": 4.72, "n": 4.72, "p": 4.72, "u": 4.72, "m": 7.08}
SMALL_CAPS = {"a": 4.91, "b": 4.64, "e": 4.46, "t": 4.73, "g": 5.13, "h": 4.91}          # \irr{a}, \irr{b}, ...
COMMAND_WIDTHS = {r"\,": 1.42, r"\!": -1.42, r"\downarrow": 8.97, r"\rangle": 3.31, r"\bigl": 4.58, r"\bigr": 4.58,
                  r"\sqrt": 7.08, r"\backslash": 4.25, r"\Gamma": 5.31, r"\pairK": 7.73, r"\catP": 6.61,
                  r"\iunit": 2.36, r"\expe": 3.78}
NUCLEI = {r"\catB": 5.55, r"\catT": 4.69, r"\rootunity": 4.28, r"\conjpair": 0.0}    # a glyph, its argument a script
ALIASES = {r"\Grot": r"\Gamma_{\mathrm r}", r"\Gspin": r"\Gamma_{\mathrm n}", r"\Gvib": r"\Gamma_{\mathrm v}",
           r"\ket": r"|#1\rangle"}


def tex_tokens(s):
    """control words (without the spaces after them), control symbols and single characters"""
    return [t.rstrip() if t[0] == "\\" else t for t in re.findall(r"\\[A-Za-z]+ *|\\.|[^\\]", s)]


def tex_arg(tok, i):
    """the tokens of the argument that starts at tok[i], a braced group or one token, and the index after it"""
    if tok[i] != "{":
        return tok[i:i + 1], i + 1
    depth, j = 0, i
    while True:
        depth += (tok[j] == "{") - (tok[j] == "}")
        if not depth:
            return tok[i + 1:j], j + 1
        j += 1


def tex_width(tok, math=False, scale=1.0):
    """the width (pt) of the tokens tok (tex_tokens), in math mode or not, set at `scale` times \\footnotesize"""
    w, i = 0.0, 0
    while i < len(tok):
        t, i = tok[i], i + 1
        if t == "$":
            math = not math
        elif t in ("_", "^", "'") and math:          # a script; a second one on the same nucleus stacks with it
            s = 0.0
            while True:
                if t == "'":                         # primes, a superscript
                    n = 1
                    while i < len(tok) and tok[i] == "'":
                        n, i = n + 1, i + 1
                    s = max(s, PRIME * n * scale)
                else:
                    a, i = tex_arg(tok, i)
                    s = max(s, tex_width(a, True, scale * SCRIPT))
                if i < len(tok) and tok[i] in ("_", "^", "'") and tok[i] != t:
                    t, i = tok[i], i + 1
                    continue
                break
            w += s + SCRIPT_SPACE * scale
        elif t == "{":
            a, i = tex_arg(tok, i - 1)
            w += 6.61 * scale if math and a in (["="], ["+"], ["-"]) else tex_width(a, math, scale)  # {=}: no spaces
        elif t in (r"\hyperlink", r"\hypertarget"):  # the link target is not printed
            _, i = tex_arg(tok, i)
        elif t == r"\tfrac":
            a, i = tex_arg(tok, i)
            b, i = tex_arg(tok, i)
            w += max(tex_width(a, True, scale * SCRIPT), tex_width(b, True, scale * SCRIPT)) + 2.4 * scale
        elif t == r"\irr":
            a, i = tex_arg(tok, i)
            w += scale * sum(SMALL_CAPS.get(x, 4.9) for x in a)
        elif t in (r"\mathrm", r"\ptag"):
            a, i = tex_arg(tok, i)
            w += tex_width(a, False, scale)
        elif t in NUCLEI:
            a, i = tex_arg(tok, i)
            w += (NUCLEI[t] + SCRIPT_SPACE) * scale + tex_width(a, True, scale * SCRIPT)
        elif t in ALIASES:
            body = ALIASES[t]
            if "#1" in body:
                a, i = tex_arg(tok, i)
                body = body.replace("#1", "".join(a))
            w += tex_width(tex_tokens(body), True, scale)
        elif t in COMMAND_WIDTHS:
            w += COMMAND_WIDTHS[t] * scale
        elif t[0] == "\\" or t == "}" or (t == " " and math):
            pass                                     # a font or size switch, a space in math
        elif math:
            w += scale * MATH_GLYPHS.get(t, 4.25 if t.isdigit() else 6.9 if t.isupper() else 4.5)
        else:
            w += scale * TEXT_GLYPHS.get(t, 4.25 if t.isdigit() else 6.4 if t.isupper() else 4.5)
    return w


def cell_width(tex):
    return tex_width(tex_tokens(tex))


def snake_layout(heights, panels):
    """the units (blocks of rows, of the given heights in lines) in groups of at most `panels` panels, as
    [[[unit, ...] for each panel] for each group]. There are ceil(total / (panels SNAKE_GROUP_LINES)) groups of about
    equal height. In each, the units run down the first panel, then down the next, cut where the heights of the panels
    differ least, then where the tallest is least tall, then with the taller panels first."""
    total = sum(heights)
    ngroups = -(-total // (panels * SNAKE_GROUP_LINES))
    cum = list(itertools.accumulate([0] + list(heights)))
    cuts = [0]
    for k in range(1, ngroups):
        cuts.append(min(range(cuts[-1] + 1, len(heights) - ngroups + k + 1),
                        key=lambda j: abs(cum[j] - total * k / ngroups)))
    cuts.append(len(heights))
    groups = []
    for a, b in zip(cuts, cuts[1:]):
        def spread(c):
            h = [cum[y] - cum[x] for x, y in zip((a,) + c, c + (b,))]
            return max(h) - min(h), max(h), [-x for x in h]
        best = min(itertools.combinations(range(a + 1, b), min(panels, b - a) - 1), key=spread)
        groups.append([list(range(x, y)) for x, y in zip((a,) + best, best + (b,))])
    return groups


def snake_table(head, rows, spec, sep, out, cont, wrap=None):
    """the longtable of the rows (each a list of cells) under the header cells `head`, in the most panels that fit
    (snake_layout). A panel has the column specifiers `spec` and \\tabcolsep `sep` (pt). In a single panel too wide for
    the text, column `wrap` becomes a paragraph column of the remaining width. A continued page repeats the head, with
    the title `cont` above it in the heading font (zero width, so it never widens the table). Returns the room (pt) of
    the head and the first group, for \\needspace."""
    ncol = len(head)
    widths = [SNAKE_SLACK * max(cell_width(r[k]) for r in [head] + rows) for k in range(ncol)]
    width = sum(widths) + 2 * sep * (ncol - 1)
    panels = max(1, min(SNAKE_PANELS, len(rows) // SNAKE_MIN_ROWS))
    while panels > 1 and panels * width + (panels - 1) * SNAKE_GAP > TEXT_WIDTH:
        panels -= 1
    if panels == 1 and width > TEXT_WIDTH and wrap is not None:
        spec = spec[:wrap] + [r">{\raggedright\arraybackslash}p{%dpt}" % (TEXT_WIDTH - width + widths[wrap])] \
            + spec[wrap + 1:]
    groups = snake_layout([1] * len(rows), panels)
    gap = r"@{\hspace{%gpt}}" % SNAKE_GAP
    out.append(r"\setlength\tabcolsep{%gpt}" % sep)
    out.append(r"\begin{longtable}{@{}%s@{}}" % gap.join(["".join(spec)] * panels))
    first = r"\toprule" + "\n" + " & ".join(head * panels) + r"\\\midrule"
    out.append(first + r"\endfirsthead")
    out.append(r"\multicolumn{%d}{@{}l@{}}{\rlap{\legendhead{%s} \small(continued)}}\\[2pt]" % (ncol * panels, cont))
    out.append(first + r"\endhead")
    for k, group in enumerate(groups):
        cols = [[rows[i] for i in col] for col in group]
        cols += [[]] * (panels - len(cols))
        for line in range(max(len(c) for c in cols)):         # no page break inside a group
            out.append(" & ".join(x for c in cols for x in (c[line] if line < len(c) else [""] * ncol)) + r"\\*")
        out.append(r"\partrule" if k < len(groups) - 1 else r"\bottomrule")
    out.append(r"\end{longtable}")
    return TABLE_HEAD + TABLE_LINE * max(len(col) for col in groups[0])


def text_lines(tex, size=1.25):
    """the lines that a paragraph takes, by cell_width (a \\footnotesize width; size: the font's scale to it)"""
    return -(-size * cell_width(tex) // TEXT_WIDTH)


# ----------------------------------------------------------------------------
# the group sections
# ----------------------------------------------------------------------------
def lab(g, i):
    return "$" + g["labels"][i] + "$"


def dims_tex(d):
    return "$(%d,%d,%d)$" % d


def nontrivial_table(g, m, out):
    """the Gamma_mol = m table of a non-Abelian group, a row per triple, with the states of its copies in ascending K
    (snake_table). Returns its room."""
    rows = [[lab(g, row["r"]), lab(g, row["n"]), lab(g, row["v"]), dims_tex(row["dims"]),
             ", ".join(state_tex(c["state"], c["var"]) for c in row["comps"])] for row in sort_rows(g["tables"][m])]
    head = [r"$\Grot$", r"$\Gspin$", r"$\Gvib$", r"$(d_{\mathrm r},d_{\mathrm n},d_{\mathrm v})$", "state"]
    out.append(r"{\footnotesize")
    room = snake_table(head, rows, ["c"] * 4 + ["l"], 2.6, out,
                       cont=r"$%s$, $\Gmol=%s$" % (g["latex"], g["labels"][m]))
    out.append("}")
    return room


def abelian_grids(g, out):
    """the grids of an Abelian group, one per Gamma_mol, all in \\scriptsize and side by side as many as fit a line
    (TeX breaks the lines). Returns the room of a line of grids."""
    labs = g["labels"]
    n = len(labs)
    head = [r"$\Grot\backslash\Gspin$"] + ["$" + l + "$" for l in labs]
    # every grid fits at \scriptsize: its estimated width at \footnotesize, every column as wide as the widest label, does
    widest = max(map(cell_width, head[1:]))
    assert max(cell_width(head[0]), widest) + n * widest + 6 * (n + 1) <= TEXT_WIDTH, g["name"]
    boxes = []
    for m in g["mols"]:
        grid = {(row["r"], row["n"]): row["v"] for row in g["tables"][m]}
        assert len(g["tables"][m]) == len(grid) == n * n, g["name"]          # one Gamma_v for every (Gamma_r, Gamma_n)
        rows = [" & ".join(head) + r"\\\midrule"]
        rows += ["$" + labs[r] + "$ & " + " & ".join("$" + labs[grid[(r, k)]] + "$" for k in range(n)) + r"\\"
                 for r in range(n)]
        boxes.append("\n".join([r"\begin{tabular}[t]{@{}c@{}}\footnotesize$%s$, $\Gmol=%s$\\[2pt]"
                                % (g["latex"], labs[m]),
                                r"\scriptsize\setlength\tabcolsep{3pt}\begin{tabular}{c@{\hspace{8pt}}%s}\toprule"
                                % ("c" * n)]
                               + rows + [r"\bottomrule\end{tabular}\end{tabular}"]))
    out.append(r"\begin{center}\setlength{\lineskip}{10pt}%")
    out.append((r"\hspace{%gpt}%%" % SNAKE_GAP + "\n").join(boxes))
    out.append(r"\clubpenalty=0 \widowpenalty=0 \end{center}")      # a page may end after any line of grids
    return GRID_ROOM + GRID_LINE * (n + 1)


GROUP_TITLES = OrderedDict([       # (title through names_tex, its PDF bookmark); in a family the groups follow n
    ("nonaxial", ("Non-axial groups $@@G:C1@@$, $@@G:Cs@@$, $@@G:Ci@@$", "Non-axial groups C1, C1h, S2")),
    ("Cn", ("Cyclic groups $\\grp{C_{\\mathnormal{n}}}$", "Cyclic groups Cn")),
    ("S2n", ("Improper cyclic groups $\\grp{S_{2\\mathnormal{n}}}$", "Improper cyclic groups S2n")),
    ("Cnh", ("Groups $\\grp{C_{\\mathnormal{n}h}}$", "Groups Cnh")),
    ("Cnv", ("Groups $\\grp{C_{\\mathnormal{n}v}}$", "Groups Cnv")),
    ("Dn", ("Dihedral groups $\\grp{D_{\\mathnormal{n}}}$", "Dihedral groups Dn")),
    ("Dnh", ("Groups $\\grp{D_{\\mathnormal{n}h}}$", "Groups Dnh")),
    ("Dnd", ("Groups $\\grp{D_{\\mathnormal{n}d}}$", "Groups Dnd")),
    ("cubic", ("Tetrahedral and octahedral groups", "Tetrahedral and octahedral groups")),
    ("icosa", ("Icosahedral groups", "Icosahedral groups")),
    ("linear", ("Linear (continuous axial) groups", "Linear (continuous axial) groups")),
])


def plain_group(latex):
    """the PDF bookmark of a printed group: \\grp{C_{3v}} -> C3v, \\grp{C_{\\infty v}} -> C\\textinfty v (hyperref's
    PU encoding writes U+221E)"""
    s = re.sub(r"[{}_ ]", "", latex.replace(r"\grp", "").replace(r"\infty", "@"))
    assert re.fullmatch(r"[A-Za-z0-9@]+", s), latex
    return s.replace("@", r"\textinfty ")


def section_of(g):
    f = g["family"]
    if f in ("C1", "Cs", "Ci"):
        return "nonaxial"
    if f in CUBIC:
        return "cubic"
    if f in ("I", "Ih"):
        return "icosa"
    if f in ("Cinfv", "Dinfh"):
        return "linear"
    return f


def group_section(g, data):
    """the LaTeX lines of the section of group g, and its room (\\needspace) up to the first place where a page may
    end. It appends the group's tables to data (species_data.json)."""
    out = []
    labs = g["labels"]                 # the atlas's printed forms
    keys = g["repo_labels"]            # this code's labels, kept as the JSON keys
    out.append(r"\subsection{\texorpdfstring{$%s$}{%s}}" % (g["latex"], plain_group(g["latex"])))
    out.append(r"\label{sec:%s}" % g["name"])
    order = r"$|\grp{G}|=\infty$" if g["truncated"] else (r"$|\grp{G}|=%d$" % g["order"])
    irr = ", ".join("$%s$\\,(%d)" % (l, d) for l, d in zip(labs, g["dims"]))
    intro = [r"\noindent %s; irreps (dimension): %s." % (order, irr)]
    if g["truncated"]:
        sub, n = STANDIN[g["name"]]
        intro.append(r" The tables are exact for these irreps. The stand-in $%s$ has order %d, and its invariants "
                     r"coincide with those of the continuous group when \mbox{$\Xi_1+\Xi_2+\Xi_3<%d$}."
                     % (sub, g["order"], n))
    mols = ", ".join("$%s$" % labs[m] for m in g["mols"])
    if g["abelian"]:                   # the grids are explained once, in Sec. 1
        counts = {len(g["tables"][m]) for m in g["mols"]}
        assert len(counts) == 1, g["name"]
        count = counts.pop()
        intro.append(r" Possible $\Gmol$: %s, %s %s." % (
            mols, "each with" if len(g["mols"]) > 1 else "with",
            "%d admissible triples, all %s" % (count, state_tex("P")) if count > 1
            else "1 admissible triple, %s" % state_tex("P")))
    else:
        intro.append(r" Possible $\Gmol$: %s." % mols)
    out += intro
    room = HEADING_ROOM["subsection"] + TEXT_LINE * text_lines("".join(intro))
    ids = g.get("ids") or [None] * len(keys)
    gd = dict(group=g["name"], atlas_record=g.get("atlas_record"), order=g["order"],
              irreps=[dict(label=k, id=i, dim=d) for k, i, d in zip(keys, ids, g["dims"])], tables={})
    data.append(gd)
    if g["abelian"]:
        room += abelian_grids(g, out)
        for m in g["mols"]:
            gd["tables"][keys[m]] = [dict(r=keys[x["r"]], n=keys[x["n"]], v=keys[x["v"]], dims=[1, 1, 1], N=1,
                                          state=x["comps"][0]["state"]) for x in g["tables"][m]]
        return room, out
    printable = g["name"] not in NO_TENSORS
    legend = assign_variants(g, printable)
    cubic = g["family"] in CUBIC
    tensors = [tensor_tex(info["T"], cubic, [labs[x] for x in info["rep"]]) if printable else None
               for _, _, info in legend]
    label = r"\legendhead{Catalogue states.}"      # what every legend shares is said once, in Sec. 1
    out.append(r"\par\smallskip\noindent" + label)
    rows = []
    for (sid, var, info), tens in zip(legend, tensors):
        r, n, v = info["rep"]
        rep = r"$(%s,%s,%s)\!\downarrow\!%s$" % (labs[r], labs[n], labs[v], labs[info["mol"]])
        if info["K"] is not None:
            rep += r", $\pairK{=}%d$" % info["K"]
        rows.append([state_tex(sid, var), rep] + ([tens] if printable else []))
    head = ["state", "representative"] + (["tensor"] if printable else [])
    out.append(r"{\footnotesize")
    room += TEXT_LINE * text_lines(label) + snake_table(
        head, rows, ["l"] * len(head), 4.0, out, wrap=2 if printable else None,
        cont=r"$%s$, catalogue states" % g["latex"])
    out.append("}")
    for m in g["mols"]:
        heading = (r"\subsubsection*{$%s$, $\Gmol=%s$ \normalfont\small\color{black}(%d admissible triples; %s)}"
                   % (g["latex"], labs[m], len(g["tables"][m]), counts_tex(state_counts(g, m))))
        table = []
        need = HEADING_ROOM["subsubsection"] + TEXT_LINE * (text_lines(heading, 1.125) - 1) \
            + nontrivial_table(g, m, table)
        out += [r"\needspace{%dpt}" % need, heading] + table
        rows_json = []
        for row in sort_rows(g["tables"][m]):
            for t, c in enumerate(row["comps"]):
                T = c["T"]
                ent = [[int(i), int(j), int(k), float(T[i, j, k].real), float(T[i, j, k].imag)]
                       for i, j, k in zip(*np.nonzero(abs(T) > 1e-10))]
                assert ent and ent[0][3] > 0 and abs(ent[0][4]) <= TOL_N2, (       # as the README states
                    "first entry not real and positive", g["name"], keys[m], keys[row["r"]], keys[row["n"]],
                    keys[row["v"]], t + 1)
                rj = dict(r=keys[row["r"]], n=keys[row["n"]], v=keys[row["v"]], dims=list(row["dims"]), N=row["N"],
                          tau=t + 1, state=c["state"])
                if row["N"] > 1:
                    rj["K"] = c["K"]
                rj["tensor"] = ent
                rows_json.append(rj)
        gd["tables"][keys[m]] = rows_json
    return room, out


def summary_table(results, out):
    def family(g):      # the summary's families: the group sections, with the exceptional groups (cubic, icosa) as one
        return "exceptional" if section_of(g) in ("cubic", "icosa") else section_of(g)
    out.append(r"{\footnotesize\setlength\tabcolsep{3.5pt}")
    out.append(r"\begin{longtable}{@{}l r r r l r r r r r@{}}")
    hdr = (r"\toprule $\grp{G}$ & $|\grp{G}|$ & \#irr & $d_{\max}$ & $\Gmol$ choices & \#adm. & %s & $\catfont{B}$ & "
           r"$\catfont{T}$ & $N=2$\\\midrule" % state_tex("P"))
    out.append(hdr + r"\endfirsthead")
    out.append(r"\multicolumn{10}{@{}l@{}}{\rlap{\legendhead{Summary} \small(continued)}}\\[2pt]")   # as snake_table
    out.append(hdr + r"\endhead")
    for k, g in enumerate(results):
        if k and family(g) != family(results[k - 1]):
            # between the families of the group sections; the penalty puts a page break after the rule's space
            out.append(r"\midrule\noalign{\penalty-1}")
        m = g["mols"][0]
        cnt = state_counts(g, m)
        tri = sum(row["comps"][0]["state"][0] == "T" for row in g["tables"][m])   # tripartite triples, N = 2 once
        n2 = [sum(row["N"] == 2 for row in g["tables"][x]) for x in g["mols"]]
        assert all(state_counts(g, x) == cnt for x in g["mols"]) and len(set(n2)) == 1, g["name"]  # every Gamma_mol
        assert cnt["P"] + sum(cnt[x] for x in B_IDS) + tri == len(g["tables"][m]), g["name"]      # P, B, T partition
        mols = ", ".join("$%s$" % g["labels"][x] for x in g["mols"])
        if len(g["mols"]) > 4:
            mols = r"%d (all real 1D)" % len(g["mols"])
        last = k + 1 == len(results) or family(results[k + 1]) != family(g)
        out.append(r"\hyperref[sec:%s]{$%s$} & %s & %d%s & %d & %s & %d & %d & %d & %d & %d%s" % (
            g["name"], g["latex"].split("=")[0], r"$\infty$" if g["truncated"] else g["order"], len(g["labels"]),
            r"\rlap{$^*$}" if g["truncated"] else "", max(g["dims"]), mols, len(g["tables"][m]),
            cnt["P"], sum(cnt[x] for x in B_IDS), tri, n2[0],
            r"\\" if last else r"\\*"))       # a page breaks only between families
    out.append(r"\bottomrule\end{longtable}}")


TITLE = "Admissible rovibrational species and their symmetry-enforced entanglement"
# cited once, at the opening of Sec. 1, for its main result
CITATION = (r"E.~R.~Das, J.-C.~Zhang, V.~V.~Albert and Y.~Liu, \emph{Symmetry classification of hyperfine "
            r"rovibrational molecular phase space}, in preparation")

PREAMBLE = r"""\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=-1
\documentclass[10pt,a4paper]{article}
@@ATLAS_STYLE@@
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsthm}
% Proposition 2.1: amsthm's plain style, with the head in the font of the atlas's \paragraph headings
\newtheoremstyle{atlasplain}{\topsep}{\topsep}{\itshape}{}{\normalfont\small\sffamily\bfseries\color{parttitle}}{.}
  {5pt plus 1pt minus 1pt}{}
\theoremstyle{atlasplain}
\newtheorem{proposition}{Proposition}[section]
\hypersetup{pdftitle={@@TITLE@@}}
\fancyhead[L]{\small\textsc{Molecular Entanglement Zoo}}
@@ATLAS_MACROS@@
% the layout. \partrule: the heavy rule between two parts of a snake table, across the columns only, as longtable and
% booktabs draw their rules. The rows end with \\*, so a page may end after the space below this rule and nowhere else
% in the table: that space stays on the page that ends, and the next page starts with the table's head. \keeprule: the
% same rule at the light width, with no page break after it, so Table 1, whose rows also end with \\*, is never split.
% \LTpre, \LTpost: the space above and below a longtable. \cellmax{<cell>}: \colwidth raised to the width of the cell,
% for a paragraph column of the remaining width (\fillwidth), as the atlas sets it. \legendhead{<text>}: a run-in
% label in the font of the atlas's \paragraph headings
\makeatletter
\newcommand{\partrule}{\noalign{\nobreak\vskip\aboverulesep}\multispan\LT@cols\unskip\leaders\hrule\@height
  \heavyrulewidth\hfill\cr\noalign{\nobreak\vskip\belowrulesep\penalty\z@}}
\newcommand{\keeprule}{\noalign{\nobreak\vskip\aboverulesep}\multispan\LT@cols\unskip\leaders\hrule\@height
  \lightrulewidth\hfill\cr\noalign{\nobreak\vskip\belowrulesep}}
\makeatother
\newcommand{\legendhead}[1]{{\normalfont\small\sffamily\bfseries\color{parttitle}#1}}
\setlength{\LTpre}{6pt plus 2pt minus 2pt}\setlength{\LTpost}{6pt plus 2pt minus 2pt}
\newlength{\fillwidth}\newlength{\cellwidth}\newlength{\colwidth}
\newcommand{\cellmax}[1]{\settowidth{\cellwidth}{#1}\ifdim\cellwidth>\colwidth\setlength{\colwidth}{\cellwidth}\fi}
\newcommand{\ket}[1]{|#1\rangle}
% the role commands, each defined here and nowhere else
\newcommand{\gel}{\elt{g}}      % a group element, in the atlas's element font
% the sector species
\newcommand{\Gmol}{\Gamma_{\mathrm{mol}}}
\newcommand{\Grot}{\Gamma_{\mathrm{r}}}
\newcommand{\Gspin}{\Gamma_{\mathrm{n}}}
\newcommand{\Gvib}{\Gamma_{\mathrm{v}}}
% the rotational levels and the coupling space (Sec. 1)
\newcommand{\rotJ}{J}           % the angular momentum of a rotational level (Secs. 1 and 3)
\newcommand{\molproj}{P_{\Gmol}}  % the projector onto the Gamma_mol component (Secs. 1 and 3)
% the 3j blocks (Sec. 1)
\newcommand{\brmult}{b}         % the multiplicity of an irrep in a restriction, the atlas's b_Gamma(J), written
                                % b_Gamma(j) here
\newcommand{\sdc}[4]{\bigl\langle{}^{#1}_{\,#2}\bigm\vert{}^{#3}_{\,#4}\bigr\rangle}   % <j m|Gamma mu>, subduction
% the catalogue ids (Sec. 2): the product state P, the pairs B_k and the tripartite states T_k, in one font
\newcommand{\catfont}{\mathcal}
\newcommand{\catP}{\catfont{P}}
\newcommand{\catB}[1]{\catfont{B}_{#1}}
\newcommand{\catT}[1]{\catfont{T}_{#1}}
% the letter of the AME catalogue (Sec. 2)
\newcommand{\redstate}{\rho}    % a reduced state of the carrier state |C>
% the letters of the N = 2 spaces (Sec. 3)
\newcommand{\pairK}{K}          % the SO(3) angular momentum of the pair of equal carriers of an N = 2 triple
\newcommand{\carJ}{\mathbf{J}}  % the spin matrices of such a carrier, a whole D^(j), through its subduction coefficients
\newcommand{\secop}{O}          % an operator on one carrier
\newcommand{\extsp}{S}          % the orthogonal complement of the one-carrier images of an N = 2 space
\newcommand{\extphase}{\theta}  % a phase of an AME vector of S for (T,T,T)
\newcommand{\cwket}[1]{\ket{\overline{#1}}}  % a codeword of the three-qutrit code [[3,1,2]]_3
\newcommand{\ghhphase}{\delta}  % the relative phase of the two simplex-GHZ members of a (G,H,H) coupling space
\begin{document}
\thispagestyle{plain}
\begin{center}
{\LARGE\sffamily\bfseries\color{parttitle}@@TITLE@@\par}
\end{center}
{\setlength{\parskip}{3.5pt}\contentspage}  % 3.5pt between the entries (the atlas: 4pt), so they fit the first page
"""

INTRO = r"""
\section{What is tabulated}\label{sec:tabulated}
Identical-particle statistics selects the physical states of a molecule. These states factor into multiplicity spaces
that symmetry leaves free and coupling tensors that it fixes, independently of the molecular Hamiltonian
(@@CITATION@@). The tables below give these coupling tensors, with the entanglement that each tensor carries between
rotation, nuclear spin and vibration.

For a point group $\grp{G}\subset@@G:Kh@@$ and a required molecular species $\Gmol$, a triple
$(\Grot,\Gspin,\Gvib)$ of rotational, nuclear-spin and vibrational species is admissible when
\begin{equation}
N^{\Gmol}_{\Grot\Gspin\Gvib}=\frac1{|\grp{G}|}\sum_{\gel\in\grp{G}}
\irchar{\Grot}(\gel)\irchar{\Gspin}(\gel)\irchar{\Gvib}(\gel)\conj{\irchar{\Gmol}}(\gel)\neq0 .
\end{equation}
The coupling space, of dimension $N$, is the range of the projector
$\molproj=\frac1{|\grp{G}|}\sum_{\gel}\conj{\irchar{\Gmol}}(\gel)\,\repD{\Grot}(\gel)\irrprod\repD{\Gspin}(\gel)
\irrprod\repD{\Gvib}(\gel)$. The tables give an orthonormal basis of it, its $N$ \emph{copies}. For $N=1$ the copy is the
normalised tensor $C_{\intx_{\mathrm r}\intx_{\mathrm n}\intx_{\mathrm v}}$, the Clebsch--Gordan coupling to $\Gmol$, and
the carrier state is
\begin{equation}\label{eq:carrier}
\ket{C}=\sum_{\intx_{\mathrm r},\intx_{\mathrm n},\intx_{\mathrm v}}C_{\intx_{\mathrm r}\intx_{\mathrm n}\intx_{\mathrm v}}
\irket{\Grot}{\intx_{\mathrm r}}\irket{\Gspin}{\intx_{\mathrm n}}\irket{\Gvib}{\intx_{\mathrm v}} ,
\end{equation}
with each $\intx$ running over the partners of its carrier.

Each tripartite state of the catalogue (Sec.~\ref{sec:catalogue}) is, for one of its triples, a block of a Wigner $3j$
symbol. In a group $\grp{G}$ of proper rotations, the states $\jm{j}{m}$ carry the irrep $\irrJ{j}$ of
$\grp{SO}(3)$, and $\brmult_{\Gamma}(j)$ is the multiplicity of an irrep $\Gamma$ in $\irrJ{j}$ restricted to $\grp{G}$.
When $\brmult_{\Gamma}(j)=1$, the partners of $\Gamma$ are $\irket{\Gamma}{\intx}=\sum_{m}\sdc{j}{m}{\Gamma}{\intx}\,
\jm{j}{m}$. The matrices of $\Gamma$ fix these subduction coefficients of $\grp{SO}(3)\supset\grp{G}$ up to one overall
phase. For irreps $\Gamma_{1},\Gamma_{2},\Gamma_{3}$ of $\grp{G}$ with
$\brmult_{\Gamma_{1}}(j_{1})=\brmult_{\Gamma_{2}}(j_{2})=\brmult_{\Gamma_{3}}(j_{3})=1$, the $3j(j_{1},j_{2},j_{3})$ block is
\begin{equation}\label{eq:block3j}
C^{(j_{1}j_{2}j_{3})}_{\intx_{1}\intx_{2}\intx_{3}}
=\sum_{m_{1},m_{2},m_{3}}\sdc{j_{1}}{m_{1}}{\Gamma_{1}}{\intx_{1}}^{*}\,
\sdc{j_{2}}{m_{2}}{\Gamma_{2}}{\intx_{2}}^{*}\,\sdc{j_{3}}{m_{3}}{\Gamma_{3}}{\intx_{3}}^{*}
\begin{pmatrix}j_{1}&j_{2}&j_{3}\\m_{1}&m_{2}&m_{3}\end{pmatrix} .
\end{equation}

Like $C$ in Eq.~(\ref{eq:carrier}), the block is a tensor in the partners of the three irreps. The $3j$ symbol is
invariant under $\grp{SO}(3)$, so the block is invariant under $\grp{G}$. It thus lies in the coupling space of
$(\Gamma_{1},\Gamma_{2},\Gamma_{3})$ for the trivial $\Gmol$, whose character is $1$ on every element. For $N=1$,
\begin{equation}\label{eq:racah}
C^{(j_{1}j_{2}j_{3})}=\lambda\,C ,
\end{equation}
with $C$ the coupling tensor of the triple and $\lambda$ a number, possibly zero. This is Racah's factorisation of
$\grp{SO}(3)$ coupling coefficients into point-group ones, with factors tabulated by Butler (P.~H.~Butler, \emph{Point
Group Symmetry Applications: Methods and Tables}, Plenum, New York, 1981). The $j$'s enter only through $\lambda$, so
every choice with $\lambda\neq0$ gives $C$ up to normalisation and phase. They are auxiliary labels, and none is a
quantum number of the molecule. In particular, $\grp{G}$ acts on nuclear spins only by permuting the nuclei, so the $j$
of a nuclear-spin carrier is not its total nuclear spin. For example, the three protons of
para-$\mathrm{NH_3}$ ($\Gspin=\irr{e}$) have total spin $1/2$, while every $j$ is an integer. For $N=2$, Sec.~\ref{sec:n2} writes the copies in
$@@G:T@@$ and $@@G:I@@$ as $3j$ blocks. A block is \emph{literal} when every $\Gamma_{i}$ is a whole $\irrJ{j_{i}}$.
Each matrix of subduction coefficients is then square, and the block is the $3j$ symbol in another basis.

\subsubsection*{Scope}
\begin{itemize}\setlength\itemsep{0pt}
\item \emph{Groups.} The tables cover the 75 point groups of Altmann and Herzig's \emph{Point-Group Theory Tables}. These are
$@@G:C1@@$, $@@G:Cs@@$, $@@G:Ci@@$ (also written $\grp{C_{s}}$, $\grp{C_{i}}$) and the seven axial families
$\grp{C_{\mathnormal{n}}}$, $\grp{S_{2\mathnormal{n}}}$, $\grp{C_{\mathnormal{n}h}}$, $\grp{C_{\mathnormal{n}v}}$,
$\grp{D_{\mathnormal{n}}}$, $\grp{D_{\mathnormal{n}h}}$, $\grp{D_{\mathnormal{n}d}}$ for $n=2,\dots,10$. They also include the
tetrahedral, octahedral and icosahedral groups $@@G:T@@$, $@@G:Th@@$, $@@G:Td@@$, $@@G:O@@$, $@@G:Oh@@$, $@@G:I@@$, $@@G:Ih@@$
and the linear groups $@@G:Cinfv@@$, $@@G:Dinfh@@$. The tables of the linear groups are truncated at $\Xi\le3$, $\Xi$ the
subscript of $\irr{e}_{\Xi}$ and $0$ for $\irr{a}$. The linear groups $\grp{C_{\infty}}$ and $\grp{C_{\infty h}}$ are left out, since
nuclei on a line always lie in vertical mirror planes. So are $@@G:Kh@@$, the point group of an atom, and $@@G:K@@$,
which is the point group of no configuration of nuclei.
\item \emph{$\Gmol$.} The orientation of a molecule ranges over $@@G:Kh@@$, so a rotational level of angular momentum
$\rotJ$ and parity $\pm$ carries the irrep $\irrJp{\rotJ}{\pm}$ of $@@G:Kh@@$. Its species $\Grot$ are those of this
irrep restricted to $\grp{G}$, and spin statistics then fixes $\Gmol$. If $\gel$ induces the permutation $p_a(\gel)$ of
the identical nuclei of isotope $a$, each of spin $I_a$, then
$\irchar{\Gmol}(\gel)=\prod_a\operatorname{sign}[p_a(\gel)]^{2I_a}$. This character
takes only the values $\pm1$, so $\Gmol$ is a real one-dimensional irrep. Each \emph{real one-dimensional irrep} of
$\grp{G}$ gets its own table. Complex one-dimensional irreps ($\conjpair{1}\irr{e}$, $\conjpair{2}\irr{e}$ of
$\grp{C_{\mathnormal{n}}}$, $@@G:T@@$, \dots) cannot be $\Gmol$.
\item \emph{Species.} Every irrep is allowed in every sector (pure group theory). The tables impose \emph{no}
molecule-specific restriction, such as a limit on the nuclear-spin species of a given set of identical nuclei. For
example, the four protons of methane, $\mathrm{CH_4}$, carry only the nuclear-spin species $@@L:Td:A_{1}@@$,
$@@L:Td:E@@$ and $@@L:Td:T_{2}@@$, which keep @@CH4KEEP@@ of the @@TDA2@@ $@@G:Td@@$ triples with
$\Gmol=@@L:Td:A_{2}@@$.
\end{itemize}

\paragraph{Conventions and the atlas.} Group elements act on states actively, by $U(\gel)$. Matrices act on column
vectors, and products compose from right to left. The irrep matrices of the 73 finite groups of Altmann and Herzig are
those of the Point Group Atlas (V.~V.~Albert, \url{https://github.com/valbert4/point-group-atlas}, 2026; commit
@@ATLAS_COMMIT@@), read in this convention. The atlas uses row vectors,
$\mathbf r\mapsto\mathbf r P(\gel)$, and lets $\gel$ act on a function by $U(\gel)^{-1}$, so
$\repD{\Gamma}(\gel)=\repD{\Gamma}_{\mathrm{atlas}}(\gel^{-1})=\repD{\Gamma}_{\mathrm{atlas}}(\gel)^\dagger$, with $\gel$
the same physical operation. States are identified by their wavefunctions, so the irrep names, the partners $\intx$ and
the coupling tensors are the atlas's for the same state. Irreps are irreducible over $\mathbb C$, so a complex-conjugate
pair counts as two irreps, $\conjpair{1}\irr{e}_k$ and $\conjpair{2}\irr{e}_k$. Their characters are the complex
conjugates of the atlas's values, so $\conjpair{2}\irr{e}$ has character $\iunit$ at $\elt{C_{4}}$ in $@@G:C4@@$ and at
$\elt{S_{4}}$ in $@@G:S4@@$. Principal axis $z$, $\elt{C_{2}'}$ along $x$, $\elt{\sigma_{v}}$ the $xz$ plane. The
tables of $@@G:Cinfv@@$ and $@@G:Dinfh@@$ are computed with finite stand-ins, $\grp{C_{13v}}$ and $\grp{D_{14h}}$. These
keep their own complex bases, in which $\elt{\sigma_{v}}$ (in $\grp{C_{13v}}$) or $\elt{C_{2}'}$ (in $\grp{D_{14h}}$)
acts on the two partners of a two-dimensional irrep as the plain exchange. Their partners are the atlas's, except that
partner 2 of @@STANDIN@@ is minus the atlas's partner 2. A change of carrier basis is a local unitary, so no catalogue
state depends on these choices.

\paragraph{The group sections.} Each group section gives the order and irreps of the group and its possible $\Gmol$. A
non-Abelian group then has a legend of the catalogue states that its tables meet, and one table per $\Gmol$. The legend
gives each state with a representative triple, written $(\Grot,\Gspin,\Gvib)\!\downarrow\!\Gmol$ for the triple projected
onto $\Gmol$. In $@@G:I@@$ and $@@G:Ih@@$ the legend leaves out the coupling tensors, which are in
\texttt{species\_data.json}. Elsewhere the legend also gives the coupling tensor, and letters after an id mark the
different tensors of a state in the group, up to carrier order and global phase. The kets of a tensor list the partners
of its degenerate carriers only, in the order $(\mathrm r,\mathrm n,\mathrm v)$ of the representative. They are
the atlas's partners $\intx$, namely $x,y,z$ of a $\irr{t}$ and $1,2$ of a two-dimensional irrep, up to the stand-in
signs above. The product state has no degenerate carrier, and its tensor is written as the kets of its three irreps. In
the tensors and in Table~\ref{tab:catalogue}, $\rootunity{3}=\expe^{2\pi\iunit/3}$. A table entry gives a triple, its
carrier dimensions and its state, or for an $N=2$ triple the states of its two copies (Sec.~\ref{sec:n2}). The heading
of a table gives its number of admissible triples and the number of copies of each state. An $N=2$ triple thus counts
once under each of its two states. A table with many entries repeats its columns up to three times side by side. Heavy
rules divide it into parts, and within a part the entries run down the first set of columns, then down the next. An
Abelian group has no legend, and one grid per $\Gmol$ in place of the tables. Every irrep of such a group is
one-dimensional, so every coupling tensor is the number $1$, the product state $\catP$. For given $(\Grot,\Gspin)$ the
vibrational species is then unique, $\Gvib=\Gmol\irrprod\conj{\Grot}\irrprod\conj{\Gspin}$. Each grid lists it, with
rows $\Grot$ and columns $\Gspin$, and every entry is admissible.
"""

CATALOGUE_HEAD = r"""
\section{The catalogue of AME states}\label{sec:catalogue}
Let $\ket{\psi}$ be a unit vector in the product of three carriers. For a carrier $\Gamma$, with $\Gamma'$ and
$\Gamma''$ the other two, its reduced state is $\redstate_{\Gamma}=\trace_{\Gamma'\Gamma''}\ket{\psi}\langle\psi|$. We
call $\ket{\psi}$ \emph{absolutely maximally entangled} (AME) when $\redstate_{\Gamma}=\mathbb{I}/d_{\Gamma}$ on each of
the three carriers. With three carriers, every cut separates one carrier from the other two. An AME state is thus
maximally entangled across every cut, with a flat Schmidt spectrum of rank $d_{\Gamma}$.

\begin{proposition}\label{prop:ame}
Every unit vector $\ket{\psi}$ in the coupling space of an admissible triple has
$\redstate_{\Gamma}=\mathbb{I}/d_{\Gamma}$ on each of its three carriers $\Gamma$. In particular, every copy is AME, and
so is every member of an $N=2$ space.
\end{proposition}

\begin{proof}
The irrep matrices of the tables are unitary, and so is
$U(\gel)=\repD{\Grot}(\gel)\irrprod\repD{\Gspin}(\gel)\irrprod\repD{\Gvib}(\gel)$. Since $\Gmol$ is one-dimensional,
every $U(\gel)$ multiplies each vector in the range of $\molproj$ by the phase $\irchar{\Gmol}(\gel)$. Hence
$U(\gel)\ket{\psi}\langle\psi|U(\gel)^{\dagger}=\ket{\psi}\langle\psi|$ for every $\gel\in\grp{G}$. Up to the order of
the factors, $U(\gel)=\repD{\Gamma}(\gel)\irrprod\repD{\Gamma'}(\gel)\irrprod\repD{\Gamma''}(\gel)$. The partial trace
over $\Gamma'$ and $\Gamma''$ removes the unitary $\repD{\Gamma'}(\gel)\irrprod\repD{\Gamma''}(\gel)$, so
$\repD{\Gamma}(\gel)\,\redstate_{\Gamma}\,\repD{\Gamma}(\gel)^{\dagger}=\redstate_{\Gamma}$ for every $\gel$. Thus
$\redstate_{\Gamma}$ commutes with every matrix of the irrep $\Gamma$, which is irreducible over $\mathbb{C}$. By
Schur's lemma, $\redstate_{\Gamma}=c\,\mathbb{I}$, and $c=1/d_{\Gamma}$ since
$\trace\redstate_{\Gamma}=\langle\psi|\psi\rangle=1$.
\end{proof}

Bryan, Leutheusser, Reichstein and Van Raamsdonk give this argument for a state that a group fixes up to a phase
(\href{https://arxiv.org/abs/1801.03508}{arXiv:1801.03508}, Sec.~3; Quantum \textbf{3}, 115 (2019)). AME states are
critical, so by the Kempf--Ness theorem two of them are SLOCC-equivalent exactly when they are LU-equivalent. The number
of carriers of dimension above one fixes the kind of entanglement. With none the state is the product state, and with
two it is a maximally entangled pair of equal dimensions (a single one is impossible). With three it is genuinely
tripartite. For $N=1$ the triple also fixes the reduced states on two carriers. The one on $(\Gspin,\Gvib)$ is the
projector onto the $\conj{\Grot}\irrprod\Gmol$ isotypic component of $\Gspin\irrprod\Gvib$, divided by $d_{\mathrm r}$,
and likewise for the other two pairs.

We call the sorted triple of carrier dimensions the \emph{format} of a triple. Bryan, Reichstein and Van Raamsdonk, who
call AME states locally maximally entangled, give the dimension of the space of AME states of each format up to LU
(\href{https://arxiv.org/abs/1708.01645}{arXiv:1708.01645}, Thms.~1.2 and~1.3; Ann.\ Henri Poincar\'e \textbf{19}, 2491
(2018)). In the formats $(1,d,d)$, $(2,2,2)$ and $(2,3,3)$ this space is a single point, so AME alone fixes the state up
to LU. In the other formats of Table~\ref{tab:catalogue} the AME states form a continuous family, from which the point
groups pick particular members.

Table~\ref{tab:catalogue} lists the 21 states of the coupling tensors of the 75 groups, up to local unitaries and the
order of the carriers, with each $N=2$ space in its $\pairK$ basis (Sec.~\ref{sec:n2}). They are the product state
$\catP$, the maximally entangled pairs $\catB{1}$ to $\catB{4}$ and the 16 tripartite states $\catT{1}$ to $\catT{16}$.
No two of them are related in this way, and \texttt{ame.py} checks this exactly. The same module assigns each copy its
state, and a copy that matches none of them would stop the build. Outside the single-point formats, the build also
certifies these assignments. It checks that any two copies with the same state are LU-equivalent up to the order of the
carriers.
"""

CATALOGUE_NOTES = r"""
\paragraph{Labels.} Two admissible triples that differ only in $\ptag{g}/\ptag{u}$ tags, of the carriers or of $\Gmol$,
have the same state. In $@@G:I@@$ and $@@G:Ih@@$, replacing every $\irr{t}_{1}$ of a triple by $\irr{t}_{2}$ and every
$\irr{t}_{2}$ by $\irr{t}_{1}$ keeps it admissible, with the same state. The build checks both facts on every admissible
triple, and the occurs-in column of Table~\ref{tab:catalogue} relies on them.

\paragraph{Two states in one format.} Four formats have two catalogue states each. In $(4,5,5)$ and $(5,5,5)$ they are the
states of the two copies of the $N=2$ spaces (Sec.~\ref{sec:n2}). In $(3,3,3)$ and $(3,3,5)$ the labels of an $N=1$
triple decide between them, by the rules of the occurs-in column of Table~\ref{tab:catalogue}. The $(3,3,3)$ rule holds
for all @@N333@@ admissible $(3,3,3)$ triples of $@@G:O@@$, $@@G:Td@@$ and $@@G:Oh@@$, over every $\Gmol$. For methane ($@@G:Td@@$, $\Gmol=@@L:Td:A_{2}@@$),
$(@@L:Td:T_{2}@@,@@L:Td:T_{2}@@,@@L:Td:T_{2}@@)$ gives $\epsilon$ and $(@@L:Td:T_{1}@@,@@L:Td:T_{2}@@,@@L:Td:T_{2}@@)$
gives $|\epsilon|$.

\paragraph{Explicit forms.} The named states of Table~\ref{tab:catalogue} are written in carrier bases adapted to each, so
they equal the tabulated tensors up to local unitaries. A block that is not literal keeps only part of some
$\irrJ{j}$. In $@@G:O@@$, for example, the five states of $\irrJ{2}$, read as symmetric traceless
tensors, split into the off-diagonal $\irr{t}_{2}$ and the traceless diagonal $\irr{e}$. The $3j(1,1,2)$ block on
$(\irr{t}_{1},\irr{t}_{1},\irr{t}_{2})$ is then $|\epsilon|$ ($\catT{4}$), and that on $(\irr{t}_{1},\irr{t}_{1},\irr{e})$
is trine-GHZ ($\catT{2}$). The 't~Hooft symbol is $\catT{8}$ because, in a suitable basis,
$@@L:I:G@@$ realises $@@G:I@@$ inside $\grp{SO}(4)$. Then $\Lambda^{2}@@L:I:G@@=@@L:I:T_{1}@@\oplus@@L:I:T_{2}@@$ is the
split of 2-forms into self-dual and anti-self-dual parts, which the 't~Hooft symbols embed in $@@L:I:G@@\irrprod@@L:I:G@@$.
The simplex cubic is $\catT{11}$ because $@@L:I:G@@$ is the standard representation of $\Alt{5}\subset\Sym{5}$ and
$(@@L:I:G@@,@@L:I:G@@,@@L:I:G@@)$ has $N=1$. The self-test of \texttt{ame.py} also checks that GHZ, trine-GHZ,
$\epsilon$, $|\epsilon|$, the 't~Hooft symbol and the literal $3j$ symbols land on their rows.

\paragraph{Axial groups.} In the non-Abelian axial groups and in $@@G:Cinfv@@$, $@@G:Dinfh@@$, every carrier has dimension
at most two, so every state is $\catP$, $\catB{1}$ or $\catT{1}$. Every admissible $(\irr{e}_{a},\irr{e}_{b},\irr{e}_{c})$,
of format $(2,2,2)$, thus gives $\catT{1}$. Such triples are admissible in every non-Abelian axial group except the four
of the $\catT{1}$ row of Table~\ref{tab:catalogue}, where the product of two two-dimensional irreps contains no
two-dimensional one. These statements hold at every $n$, so for the axial groups the tables below add only which
triples are admissible.
"""


def catalogue_section(nums):
    """Sec. 2: the AME paragraph, the catalogue table (ame.CATALOGUE) and its notes, through names_tex. nums: the
    numbers that check_rules returns."""
    check_formats()
    out = [CATALOGUE_HEAD]
    out.append(r"{\footnotesize\setlength\tabcolsep{3pt}\setlength\LTcapwidth{\linewidth}"
               r"\renewcommand{\arraystretch}{1.05}")
    # The table does not fit at its natural widths. So id, format and 3j block keep theirs, state and N are paragraph
    # columns, and occurs in is a paragraph column of the remaining width (\fillwidth, measured by TeX).
    natural = [["id"] + ["$%s$" % row["id_tex"] for row in ame.CATALOGUE],
               ["dims"] + ["$(%d,%d,%d)$" % tuple(row["format"]) for row in ame.CATALOGUE],
               [r"$3j$ block"] + [row["block_tex"] for row in ame.CATALOGUE]]
    out.append(r"\setlength{\fillwidth}{\dimexpr\linewidth-10\tabcolsep-0.315\linewidth\relax}%")
    for cells in natural:
        out.append(r"\setlength{\colwidth}{0pt}%s\addtolength{\fillwidth}{-\colwidth}%%"
                   % "".join(r"\cellmax{%s}" % c for c in sorted(set(cells))))
    out.append(r"\begin{longtable}{@{}l l >{\raggedright}p{0.24\linewidth} l >{\raggedright}p{\fillwidth} "
               r">{\raggedright\arraybackslash}p{0.075\linewidth}@{}}")
    out.append(r"\caption{The AME states of the coupling tensors, one row per state. Dims: the sorted carrier "
               r"dimensions (the format). $3j$ block: a Wigner $3j$ symbol of smallest $j_{1}+j_{2}+j_{3}$ whose "
               r"block, Eq.~(\ref{eq:block3j}), is the state (for $N=2$, the copy with the state's $\pairK$). It is "
               r"given for the triple shown, in $@@G:O@@$ ($\catT{1}$ to $\catT{4}$) or $@@G:I@@$ ($\catT{5}$ to "
               r"$\catT{16}$), coupled to the trivial irrep. Another triple with the same state can have another "
               r"block. Literal: each carrier is a whole $\irrJ{j}$, so the state is the $3j$ symbol in another basis. "
               r"Occurs in: the carriers up to order and without "
               r"$\ptag{g}/\ptag{u}$ tags, for every $\Gmol$ unless a condition is stated, with $\irr{t}_{a}$ for "
               r"$\irr{t}_{1}$ or $\irr{t}_{2}$ and $\irr{e}_{a}$ for a two-dimensional irrep "
               r"(Sec.~\ref{sec:catalogue}, \emph{Labels}). $N$: the dimension of the coupling space. $\pairK$ "
               r"($\catT{3}$, $\catT{4}$ and $\catT{13}$ to $\catT{16}$): the total angular momentum of the two equal "
               r"carriers, $\irr{t}$ or $\irr{h}$, in the copy of an $N=2$ space (Sec.~\ref{sec:n2}).}"
               r"\label{tab:catalogue}\\")
    head = r"\toprule id & dims & state description & $3j$ block & occurs in & $N$\\\midrule"
    out.append(head + r"\endfirsthead")
    out.append(head + r"\endhead")
    out.append(r"\bottomrule\endfoot")
    for row in ame.CATALOGUE:     # the link target raised to the top of its row, so a viewer shows the whole row
        out.append(r"\raisebox{\ht\strutbox}{\hypertarget{cat:%s}{}}$%s$ & $(%d,%d,%d)$ & %s & %s & %s & %s\\*" % (
            (row["id"], row["id_tex"]) + tuple(row["format"]) + (row["name_tex"], row["block_tex"],
                                                                    row["occurs_tex"], row["N_tex"])))
        if row["id"] in ("P", B_IDS[-1]):
            out.append(r"\keeprule")      # between P and the pairs B, and between the pairs and the states T
    out.append(r"\end{longtable}}")
    out.append(CATALOGUE_NOTES.replace("@@N333@@", str(nums["n333"])))
    tex = names_tex("\n".join(out))
    assert "@@" not in tex, "a placeholder of Sec. 2 is left"
    return tex


N2_SECTION = r"""
\section{\texorpdfstring{The $N=2$ spaces}{The N = 2 spaces}}\label{sec:n2}
\paragraph{The spaces.} Coupling spaces with $N=2$ occur only for $(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$ in $@@G:T@@$ and
$@@G:Th@@$, and for $(@@L:I:G@@,@@L:I:H@@,@@L:I:H@@)$ and $(@@L:I:H@@,@@L:I:H@@,@@L:I:H@@)$ in $@@G:I@@$ and $@@G:Ih@@$,
up to order and $\ptag{g}/\ptag{u}$ tags. The tables have @@N2@@ such triples over all $\Gmol$: @@NTTT@@ of type
$(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$, @@NGHH@@ of type $(@@L:I:G@@,@@L:I:H@@,@@L:I:H@@)$ and @@NHHH@@ of type
$(@@L:I:H@@,@@L:I:H@@,@@L:I:H@@)$. Their two copies are the $\pairK$ basis. Each triple has a pair of carriers of one
letter, $@@L:T:T@@$ or $@@L:I:H@@$ (the two $@@L:I:H@@$ carriers in $(@@L:I:G@@,@@L:I:H@@,@@L:I:H@@)$). These carriers
are the restrictions of the irreps $\irrJ{j}$ of $@@G:K@@$, with $j=1$ for $@@L:T:T@@$ and $j=2$ for $@@L:I:H@@$, to the
proper rotations of $\grp{G}$. Their subduction coefficients (Sec.~\ref{sec:tabulated}) thus fix the spin matrices
$\carJ$ and $\carJ'$ of the two carriers. As with the $j$ of a block (Sec.~\ref{sec:tabulated}), these matrices on a
nuclear-spin carrier are not its nuclear spin. The copies are the eigenvectors of the pair Casimir $(\carJ+\carJ')^{2}$,
the squared total angular momentum, with eigenvalues $\pairK(\pairK+1)$. Table~\ref{tab:catalogue} gives the values of
$\pairK$ and the states of the copies. For $(@@L:I:H@@,@@L:I:H@@,@@L:I:H@@)$, as for $(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$,
every pair of carriers gives the same copies. In $@@G:T@@$ and $@@G:I@@$ each copy is a $3j(\pairK,j,j)$ block,
Eq.~(\ref{eq:block3j}), with the other carrier taken from $\irrJ{\pairK}$. The block couples the
two equal carriers to total angular momentum $\pairK$. It is $3j(1,1,1)$ or $3j(2,1,1)$ for
$(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$, and the block of Table~\ref{tab:catalogue} for $\catT{13}$ to $\catT{16}$. Each copy is
stored with its first nonzero entry, in the lexicographic order of $(\intx_{\mathrm r},\intx_{\mathrm n},\intx_{\mathrm v})$,
real and positive. This fixes the relative phase of the two copies for the coefficients below, where
$\ket{\catT{k}}$ is the stored copy whose state is $\catT{k}$.

\paragraph{Codes.} The proof of Proposition~\ref{prop:ame} extends to two members $\ket{C}$, $\ket{C'}$ of an
$N=2$ space. For a carrier $\Gamma$, with $\Gamma'$ and $\Gamma''$ the other two, and an operator $\secop$ on $\Gamma$,
\begin{equation}\label{eq:schur}
\trace_{\Gamma'\Gamma''}\ket{C}\langle C'|=\langle C'|C\rangle\,\frac{\mathbb{I}}{d_{\Gamma}} ,\qquad
\langle C'|\secop|C\rangle=\frac{\trace\secop}{d_{\Gamma}}\,\langle C'|C\rangle .
\end{equation}
Both members transform by the same one-dimensional $\Gmol$, so $\trace_{\Gamma'\Gamma''}\ket{C}\langle C'|$ commutes
with $\repD{\Gamma}(\gel)$ for every $\gel$. Schur's lemma then gives the first equation, and the second follows from it.
Thus $\secop$ acts on the space as the number $\trace\secop/d_{\Gamma}$, whether or not it respects the symmetry. No
operator on one carrier can tell members apart or turn one into another. In coding terms, each space is a pure quantum
code of distance two on the three carriers. It is a $((3,2,2))$ code with local dimensions $(3,3,3)$, $(4,5,5)$ or
$(5,5,5)$. The second equation is its Knill--Laflamme condition. The first is Wang and Feng's criterion for a pure code
with arbitrary local dimensions (\href{https://doi.org/10.1007/s11464-012-0209-2}{Front.\ Math.\ China \textbf{7}, 573
(2012)}, Thm.~2(ii)).

An operator on two carriers that commutes with the action of $\grp{G}$ on them maps each space into itself, since it
commutes with the projector $\molproj$ of Sec.~\ref{sec:tabulated}. For each space and each pair of carriers, such operators
give every operator on the space: their matrices in the $\pairK$ basis span all $2\times2$ matrices. The part of a
$\grp{G}$-invariant rotation--vibration, rotation--spin or spin--vibration coupling that keeps the species of both
sectors acts on their carriers through such operators.

\paragraph{Couplings of two equal carriers.} For two equal carriers from $\irrJ{j}$, the coupling
$\carJ\cdot\carJ'=\tfrac12[(\carJ+\carJ')^{2}-\carJ^{2}-\carJ'^{2}]$ is diagonal in the $\pairK$ basis, with
eigenvalues $\tfrac12\pairK(\pairK+1)-j(j+1)$. These are @@JJTTT@@ for $(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$ and @@JJHHH@@
for $(@@L:I:H@@,@@L:I:H@@,@@L:I:H@@)$, for every pair of carriers, and @@JJGHH@@ for the $@@L:I:H@@$ pair of
$(@@L:I:G@@,@@L:I:H@@,@@L:I:H@@)$. First-order Coriolis coupling, proportional to the scalar product of the rotational
and vibrational angular momenta, has this form when the pair is rotation and vibration. In $@@G:T@@$ at $\rotJ=1$ and
in $@@G:I@@$ at $\rotJ=2$, the $2\rotJ+1$ rotational states form a single irrep, $@@L:T:T@@$ or $@@L:I:H@@$. Coriolis
coupling to a vibration of that irrep is then diagonal in the $\pairK$ basis.

\paragraph{Relations to known codes.} Let $\extsp$ be the orthogonal complement of all one-carrier images
$\secop\ket{C}$ of a space. A distance-two code that contains the space is pure, since its Knill--Laflamme constants
are those of Eq.~(\ref{eq:schur}). A unit vector orthogonal to the space thus extends it to a larger distance-two code
exactly when it lies in $\extsp$ and is AME.
\begin{itemize}\setlength\itemsep{0pt}
\item $(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$. The matrices of $@@L:T:T@@$ in $@@G:T@@$, and of $@@L:Th:T_{g}@@$ and
$@@L:Th:T_{u}@@$ at the proper rotations of $@@G:Th@@$, are the rotation matrices of the elements. With the partners
$x,y,z$ read as $k=0,1,2$ modulo 3, the codewords of the three-qutrit code $[[3,1,2]]_{3}$ are
$\cwket{s}=\tfrac{1}{3}\sqrt{3}\sum_{k}\ket{k,k+s,k+2s}$ for $s=0,1,2$. The copies are
$\ket{\catT{3}}=\epsilon=\tfrac{1}{2}\sqrt{2}(\cwket{1}-\cwket{2})$ and
$\ket{\catT{4}}=|\epsilon|=\tfrac{1}{2}\sqrt{2}(\cwket{1}+\cwket{2})$, so the space is $[[3,1,2]]_{3}$ without its GHZ
codeword $\cwket{0}$. Up to local unitaries it is Cleve, Gottesman and Lo's scheme for sharing a qubit among three
qutrits (\href{https://arxiv.org/abs/quant-ph/9901025}{arXiv:quant-ph/9901025}; Phys.\ Rev.\ Lett.\ \textbf{83}, 648
(1999)). Here $\extsp$ is spanned by $\ket{000}$, $\ket{111}$ and $\ket{222}$, and its AME vectors are
$\tfrac{1}{3}\sqrt{3}\sum_{k}\expe^{\iunit\extphase_{k}}\ket{kkk}$. Adding one gives $[[3,1,2]]_{3}$ up to a diagonal
local unitary.
\item $(@@L:I:G@@,@@L:I:H@@,@@L:I:H@@)$. Here $\extsp=0$, so the space lies in no larger distance-two code. Nor is
there a stabiliser code of distance two and dimension above one on
$\mathbb{C}^{4}\otimes\mathbb{C}^{5}\otimes\mathbb{C}^{5}$, in Wang, Feng and Feng's construction over Abelian groups
(\href{https://doi.org/10.1007/s11425-010-4072-x}{Sci.\ China Math.\ \textbf{53}, 2501 (2010)}). This holds for either
group of order 4 on $\mathbb{C}^{4}$, $\Cyc{4}$ or $\Cyc{2}\dirprod\Cyc{2}$. The group $\Cyc{5}\dirprod\Cyc{5}$ on
the two $@@L:I:H@@$ carriers has an order coprime to 4. Every stabiliser group thus splits into Weyl operators on the
$@@L:I:G@@$ carrier and Weyl operators on the two $@@L:I:H@@$ carriers, and its code is a product of one code on each
part. Distance two makes each factor one-dimensional. Two orthonormal codewords of a factor would have equal marginals
on one of its carriers. Since a distance-two code corrects the erasure of the rest of the factor, these marginals
would also have orthogonal supports.
\item $(@@L:I:H@@,@@L:I:H@@,@@L:I:H@@)$. The three $@@L:I:H@@$ carriers have the same matrices at the proper
rotations, and $\extsp=\Lambda^{3}\mathbb{C}^{5}$ is their totally antisymmetric subspace. The Hodge dual of a state of
$\extsp$ is a 2-vector of even rank at most 4, with a kernel vector. The state is then the wedge product of that vector
with a 2-vector, so each of its one-carrier marginals has the eigenvalue $1/3$, not $1/5$. Thus no state of $\extsp$ is
AME, and the space lies in no larger distance-two code.
\end{itemize}

\paragraph{Special members.} The members of an $N=2$ space are not all LU-equivalent, since its two copies are different
states of Table~\ref{tab:catalogue}. The members below are LU-equivalent to named states, and the build checks each of
these identifications on the stored copies of every $N=2$ triple.
\begin{itemize}\setlength\itemsep{0pt}
\item $(@@L:T:T@@,@@L:T:T@@,@@L:T:T@@)$. The members $\tfrac12(\ket{\catT{3}}\pm\iunit\sqrt3\,\ket{\catT{4}})$ are
LU-equivalent to $\epsilon$, and the codewords $\cwket{1}$ and $\cwket{2}$ to the GHZ codeword $\cwket{0}$.
\item $(@@L:I:G@@,@@L:I:H@@,@@L:I:H@@)$. The two members
$\tfrac{1}{4}\sqrt{10}\,\ket{\catT{13}}\pm\expe^{\iunit\ghhphase}\tfrac{1}{4}\sqrt{6}\,\ket{\catT{14}}$, with
$\ghhphase\approx@@GHHPHASE@@\,\pi$ (rounded to ten digits) in every such triple, are LU-equivalent to the simplex-GHZ
state $\tfrac{1}{5}\sqrt{5}\sum_{i=1}^{5}\ket{v_{i}}\ket{i}\ket{i}$. Here the $v_{i}$ are unit vectors at the vertices of a
regular simplex in the $@@L:I:G@@$ carrier, as in $\catT{11}$.
\item $(@@L:I:H@@,@@L:I:H@@,@@L:I:H@@)$. The member $\tfrac17(2\ket{\catT{15}}+3\sqrt5\,\ket{\catT{16}})$ is
LU-equivalent to the copy $\ket{\catT{15}}$. The two members
$\tfrac{2}{7}\sqrt{7}\,\ket{\catT{15}}+\tfrac{1}{56}\sqrt{21}(\sqrt{15}\pm7\iunit)\ket{\catT{16}}$ are LU-equivalent to
$\mathrm{GHZ}_{5}=\tfrac{1}{5}\sqrt{5}\sum_{i}\ket{iii}$.
\end{itemize}
"""


def n2_section(n2, delta):
    """Sec. 3: the N = 2 spaces, with the counts and eigenvalues of check_n2_spaces and the phase delta of
    check_special_members, through names_tex"""
    jdot = {typ: "$%d$ and $%d$" % v for typ, v in n2["jdot"].items()}
    tex = N2_SECTION
    for key, val in (("N2", str(sum(n2["count"].values()))), ("NTTT", str(n2["count"]["TTT"])),
                     ("NGHH", str(n2["count"]["GHH"])), ("NHHH", str(n2["count"]["HHH"])), ("JJTTT", jdot["TTT"]),
                     ("JJGHH", jdot["GHH"]), ("JJHHH", jdot["HHH"]), ("GHHPHASE", "%.10f" % (delta / np.pi))):
        tex = tex.replace("@@%s@@" % key, val)
    tex = names_tex(tex)
    assert "@@" not in tex, "a placeholder of Sec. 3 is left"
    return tex


def write_rows(fn, rows):
    """a JSON list, one element per line"""
    with open(fn, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")


def atlas_commit(blocks):
    """the atlas commit that the three atlas_*.json files record (they must agree), also named by the headers of
    the blocks atlas_macros.tex and atlas_style.tex"""
    commits = set()
    for fn in ("atlas_irreps.json", "atlas_labels.json", "atlas_print.json"):
        with open(os.path.join(HERE, fn), encoding="utf-8") as f:
            commits.add(json.load(f)["provenance"]["atlas"])
    assert len(commits) == 1, commits
    commit = commits.pop()
    assert all("at commit %s," % commit in b.split("\n")[0] for b in blocks), "an atlas_*.tex names another commit"
    return commit


def main():
    if not __debug__:
        sys.exit("make_pdf.py: run without -O or PYTHONOPTIMIZE, since its checks are assert statements")
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.pkl")
    with open(src, "rb") as f:
        results = pickle.load(f)
    digest = input_digest()
    bad = [g.get("name") for g in results if g.get("schema") != SCHEMA or g.get("inputs") != digest]
    if bad:
        sys.exit("make_pdf.py: %s has groups without schema %r or computed from other inputs (%s); rerun species.py"
                 % (src, SCHEMA, "all %d" % len(bad) if len(bad) == len(results) else ", ".join(bad)))
    if len(results) != 75:
        sys.exit("make_pdf.py: %s has %d groups, not 75; rerun species.py without arguments" % (src, len(results)))
    if ame.IDS != ["P"] + list(B_IDS) + ["T%d" % k for k in range(1, 17)]:
        sys.exit("make_pdf.py: ame.CATALOGUE is not in the order P, B1 to B4, T1 to T16")
    for g in results:          # print the atlas's forms; keep this code's labels as the JSON keys
        g["repo_labels"] = list(g["labels"])
        g["labels"] = [atlas_tex.label_tex(g["name"], lab) for lab in g["labels"]]
        g["latex"] = atlas_tex.group_tex(g["name"])
    ncopies = check_copies(results)                # every stored copy against the current irreps
    macros, style = (open(os.path.join(HERE, fn), encoding="utf-8").read().rstrip("\n")
                     for fn in ("atlas_macros.tex", "atlas_style.tex"))
    standin = [r"%s of $%s$" % (and_list(s), g["latex"]) for g in results if g["truncated"] for s in [standin_signs(g)]
               if s]
    assert standin, "no stand-in irrep with a sign: the Sec. 1 sentence on the stand-ins must change"
    intro = (INTRO.replace("@@STANDIN@@", and_list(standin)).replace("@@CITATION@@", CITATION)
             .replace("@@ATLAS_COMMIT@@", atlas_commit([macros, style])))
    td = next(g for g in results if g["name"] == "Td")
    ch4 = td["tables"][td["repo_labels"].index("A_{2}")]
    keep = sum(td["repo_labels"][row["n"]] in ("A_{1}", "E", "T_{2}") for row in ch4)
    intro = names_tex(intro.replace("@@CH4KEEP@@", str(keep)).replace("@@TDA2@@", str(len(ch4))))
    assert "@@" not in intro, "a placeholder of Sec. 1 is left"
    preamble = (names_tex(PREAMBLE).replace("@@TITLE@@", TITLE).replace("@@ATLAS_STYLE@@", style)
                .replace("@@ATLAS_MACROS@@", macros))
    nums = check_rules(results)                    # Secs. 2 and 3, row by row (the N = 2 counts first)
    check_lu_classes(results)                      # Sec. 2: the copies of each state are LU-equivalent
    n2 = check_n2_spaces(results)                  # Sec. 3; SystemExit on a mismatch
    delta = check_special_members(results)
    dev = n2["dev"]
    print("make_pdf.py: the tables list every admissible triple once; %d stored copies re-checked (norm, invariance, "
          "flat marginals, catalogue id); the copies of each "
          "state LU-equivalent; the groups of the occurs-in column of Table 1 checked" % ncopies)
    print("make_pdf.py: Sec. 3 checked on %d N = 2 triples (%s): first nonzero entries real and positive (|Im| %.1e); "
          "Schur identity %.1e; two-carrier couplings give all 2 x 2 matrices on every pair (%d commutants, leak %.1e, "
          "fourth singular value >= %.3f); J.J' and (J + J')^2 %.1e, J.J' %s; (T,T,T) against [[3,1,2]]_3 %.1e; "
          "(G,H,H) S = 0 (smallest singular value %.3f); (H,H,H) S = Lambda^3 C^5 %.1e" % (
              sum(n2["count"].values()), ", ".join("%s %d" % kv for kv in sorted(n2["count"].items())), dev["phase"],
              dev["schur"], n2["commutants"], dev["leak"], dev["rank4"], dev["jj"],
              ", ".join("%s %s" % (k, v) for k, v in sorted(n2["jdot"].items())), dev["ttt"], dev["ghh"], dev["hhh"]))
    out = [preamble, intro, catalogue_section(nums), n2_section(n2, delta)]
    # the heading kept with its three lines of text and the table's head and first group, of three rows
    out.append(r"\needspace{%dpt}" % (HEADING_ROOM["section"] + 3 * TEXT_LINE + TABLE_HEAD + 3 * TABLE_LINE))
    out.append(r"\section{Summary}\label{sec:summary}")
    out.append(names_tex(r"Admissible triples and their catalogue states, for one $\Gmol$ of each group (the counts "
                         r"are the same for every $\Gmol$). The columns $\catP$, $\catfont{B}$ and $\catfont{T}$ count "
                         r"the triples with the product state, a maximally entangled pair ($\catB{1}$ to $\catB{4}$) "
                         r"and a tripartite state ($\catT{1}$ to $\catT{16}$), and add up to the admissible triples. "
                         r"$N=2$ counts the triples whose coupling space is two-dimensional (Sec.~\ref{sec:n2}), all "
                         r"of them in $\catfont{T}$. $d_{\max}$: the largest irrep dimension. $^*$: the truncated "
                         r"linear groups (Sec.~\ref{sec:tabulated}).\par"))
    summary_table(results, out)
    data = []
    current = None
    for g in results:              # each heading kept with what follows it (\needspace), a family with its first group
        room, lines = group_section(g, data)
        sec = section_of(g)
        if sec != current:
            tex, plain = GROUP_TITLES[sec]
            out.append(r"\needspace{%dpt}" % (HEADING_ROOM["section"] + room))
            out.append(r"\section{\texorpdfstring{%s}{%s}}" % (names_tex(tex), plain))
            current = sec
        else:
            out.append(r"\needspace{%dpt}" % room)
        out += lines
    out.append(r"\end{document}")
    tex = "\n".join(out)
    with open(os.path.join(HERE, "species_tables.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    with open(os.path.join(HERE, "species_data.json"), "w") as f:
        json.dump(data, f)
    write_rows(os.path.join(HERE, "ame_catalogue.json"), ame.catalogue_json(results))


if __name__ == "__main__":
    main()
