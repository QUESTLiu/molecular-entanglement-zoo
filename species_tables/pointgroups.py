"""
Point groups of O(3): construction, irreducible representations (over C), their bases,
and Mulliken labels, matched to the Point Group Atlas (the atlas).

Groups (all_finite_groups(10)): Altmann and Herzig's 75. These are the 73 finite groups C1, Cs, Ci;
C_n, C_nv, C_nh, S_2n, D_n, D_nh, D_nd for n = 2..10; T, Th, Td, O, Oh, I, Ih, each with its
atlas record (atlas_irreps.json, 73 records), and the linear groups C_inf v and D_inf h as
finite stand-ins (C13v, D14h), valid for Xi <= 3 (Xi the subscript n of E_n, 0 for the A irreps).

Conventions:
  * elements act actively on column vectors, and products compose from right to left:
    D(g) D(h) = D(gh), gh = "h first";
  * principal axis z, C2' axis along x, sigma_v = xz plane;
  * every irrep of a finite group is the atlas's (atlas_irreps.json, from the atlas at 99b3710),
    read in this convention: D(g) = D_atlas(phys g)^dagger, where phys g is the atlas element
    with row matrix P = R^T (the same physical operation). The atlas acts on rows,
    r -> r P(g), and on functions by U(g)^{-1}, so its tabulated D(g) is ours at g^{-1}. Labels,
    partner order and coupling coefficients are therefore the atlas's for the same state;
  * complex-conjugate pairs of 1D irreps are labelled as the atlas labels the same state, and
    in this active convention their characters are the complex conjugates of the atlas's
    tabulated ones ({}^2E has character i at C_4 in C_4, and at S_4 in S_4);
  * the finite stand-ins of the linear groups keep their own matrices (complex basis, the swap
    as the plain permutation) and take the atlas's labels and ids.
"""
import json
import os
import numpy as np

PI = np.pi


# ----------------------------------------------------------------------------
# matrices
# ----------------------------------------------------------------------------
def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Raxis(axis, t):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


E3 = np.eye(3)
SH = np.diag([1.0, 1, -1])           # sigma_h
INV = -np.eye(3)                     # inversion
SV = np.diag([1.0, -1, 1])           # sigma_v (xz plane)
C2X = np.diag([1.0, -1, -1])
C2Y = np.diag([-1.0, 1, -1])
C2Z = np.diag([-1.0, -1, 1])
C3D = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])   # x->y->z  (C3 about [111])
SD_XY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])  # sigma_d swapping x<->y
C4Z = Rz(PI / 2)
GOLD = (1 + 5 ** 0.5) / 2
C5I = Raxis([0, 1, GOLD], 2 * PI / 5)  # 5-fold axis through icosahedron vertex


def mkey(M):
    return tuple(int(round(x * 1e6)) for x in np.asarray(M).flatten())


def closure(gens):
    elems = [np.eye(3)]
    keys = {mkey(np.eye(3)): 0}
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for A in frontier:
            for g in gens:
                B = g @ A
                k = mkey(B)
                if k not in keys:
                    keys[k] = len(elems)
                    elems.append(B)
                    new.append(B)
        frontier = new
    return elems, keys


# ----------------------------------------------------------------------------
# finite group with irreps from the regular representation
# ----------------------------------------------------------------------------
class FiniteGroup:
    def __init__(self, name, family, gens, probes, n=None, meta=None):
        self.name, self.family, self.n = name, family, n
        self.meta = meta or {}
        self.elems, self.keys = closure(gens)
        self.order = len(self.elems)
        self.probes = {}
        for pname, M in probes.items():
            k = mkey(M)
            if k in self.keys:
                self.probes[pname] = self.keys[k]
        self._mult_table()
        self._classes()
        self._irreps()

    def _mult_table(self):
        G = self.order
        self.mul = np.zeros((G, G), int)
        for i, A in enumerate(self.elems):
            for j, B in enumerate(self.elems):
                self.mul[i, j] = self.keys[mkey(A @ B)]
        self.inv = np.array([int(np.where(self.mul[i] == 0)[0][0]) for i in range(G)])

    def _classes(self):
        G = self.order
        seen = -np.ones(G, int)
        classes = []
        for g in range(G):
            if seen[g] >= 0:
                continue
            cl = sorted({self.mul[self.mul[h, g], self.inv[h]] for h in range(G)})
            for x in cl:
                seen[x] = len(classes)
            classes.append(cl)
        self.classes = classes

    def _irreps(self):
        G = self.order
        rng = np.random.default_rng(12345)
        # regular representation R(g) e_h = e_{gh}
        perms = self.mul  # perms[g, h] = gh
        # central Hermitian element separating all irreps (incl. complex pairs)
        Z = np.zeros((G, G), complex)
        for cl in self.classes:
            S = np.zeros((G, G))
            for g in cl:
                S[perms[g], np.arange(G)] += 1.0
            r, s = rng.normal(), rng.normal()
            Z += r * (S + S.T) + 1j * s * (S - S.T)
        w, V = np.linalg.eigh(Z)
        groups = []
        start = 0
        for i in range(1, G + 1):
            if i == G or abs(w[i] - w[i - 1]) > 1e-6 * (1 + abs(w).max()):
                groups.append(V[:, start:i])
                start = i
        irreps = []
        for Vi in groups:
            m = Vi.shape[1]
            d = int(round(m ** 0.5))
            assert d * d == m, (self.name, m)
            A = rng.normal(size=(G, G)) + 1j * rng.normal(size=(G, G))
            A = A + A.conj().T
            B = np.zeros((G, G), complex)
            for g in range(G):
                p = perms[g]
                Bg = np.zeros((G, G), complex)
                Bg[np.ix_(p, p)] = A
                B += Bg
            Bs = Vi.conj().T @ B @ Vi
            Bs = (Bs + Bs.conj().T) / 2
            ww, U = np.linalg.eigh(Bs)
            Q = Vi @ U[:, :d]
            assert d == 1 or abs(ww[d] - ww[d - 1]) > 1e-8
            D = np.zeros((G, d, d), complex)
            for g in range(G):
                Rg = np.zeros((G, G))
                Rg[perms[g], np.arange(G)] = 1.0
                D[g] = Q.conj().T @ Rg @ Q
            irreps.append(D)
        assert sum(D.shape[1] ** 2 for D in irreps) == G
        assert len(irreps) == len(self.classes)
        self.D = [self._adapt(D) for D in irreps]
        for D in self.D:
            self._check(D)
        self.chars = [np.array([np.trace(D[g]) for g in range(G)]) for D in self.D]
        self.dims = [D.shape[1] for D in self.D]
        self.labels = [self.labeler(i) for i in range(len(self.D))]
        # sort by label (label_sortkey): parity tag, primes, letter, subscript, sign of a Sigma, complex-pair prefix
        order = sorted(range(len(self.D)), key=lambda i: self._sortkey(i))
        self.D = [self.D[i] for i in order]
        self.chars = [self.chars[i] for i in order]
        self.dims = [self.dims[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        assert len(set(self.labels)) == len(self.labels), (self.name, self.labels)
        self._import_atlas()

    def _import_atlas(self):
        """replace each irrep by the atlas's for the same state: D(g) = D_atlas(phys g)^dagger"""
        rec = ATLAS_RECORD.get(self.name, self.name)
        data = atlas_data()["records"].get(rec)
        if data is None:
            if self.family not in LINEAR:
                raise LookupError("no atlas record %s for %s in atlas_irreps.json; only the linear "
                                  "stand-ins %s keep their own matrices" % (rec, self.name, ", ".join(LINEAR)))
            self.atlas_record, self.ids = None, None      # the stand-in's ids are set after the cut
            return
        self.atlas_record, self.ids = atlas_ids(self.name, self.labels)
        keys = {mkey(np.array(e["P"]).T): k for k, e in enumerate(data["elements"])}
        phys = np.array([keys[mkey(R)] for R in self.elems])
        mats = [np.array([[[complex(*z) for z in row] for row in m] for m in ir["mats"]]) for ir in data["irreps"]]
        chis = [np.einsum("gii->g", M) for M in mats]
        D, ids = [], []
        for i in range(len(self.D)):
            js = [j for j, M in enumerate(mats) if M.shape[1] == self.dims[i]
                  and np.allclose(chis[j][phys], np.conj(self.chars[i]), atol=1e-8)]
            assert len(js) == 1, (self.name, self.labels[i], js)
            D.append(np.conj(np.transpose(mats[js[0]][phys], (0, 2, 1))))
            ids.append(data["irreps"][js[0]]["id"])
        assert ids == self.ids, (self.name, ids, self.ids)
        for Dk in D:
            self._check(Dk)
        chars = [np.einsum("gii->g", Dk) for Dk in D]
        assert all(np.allclose(a, b, atol=1e-8) for a, b in zip(chars, self.chars)), self.name
        self.D = D

    def _check(self, D):
        G = self.order
        for g in range(G):
            assert np.allclose(D[g] @ D[g].conj().T, np.eye(D.shape[1]), atol=1e-8)
        for g in range(0, G, max(1, G // 12)):
            for h in range(G):
                assert np.allclose(D[g] @ D[h], D[self.mul[g, h]], atol=1e-8)

    # ------------------------------------------------------------------
    # the basis of a linear stand-in
    # ------------------------------------------------------------------
    def _adapt(self, D):
        """a two-dimensional irrep of a linear stand-in in its own complex basis: partner 1 the e^{+i theta}
        eigenvector of C_n, partner 2 its image under C2' (or sigma_v), which is then the plain exchange. Every other
        irrep is returned unchanged, and _import_atlas replaces each irrep of a finite group by the atlas's matrices."""
        if D.shape[1] != 2 or self.family not in LINEAR:
            return D
        P = self.probes["Cn"]
        s = self.probes.get("C2p", self.probes.get("sv"))
        w, U = np.linalg.eig(D[P])
        ang = np.angle(w)
        i1 = int(np.argmax(ang))  # e^{+i theta}, theta in (0, pi)
        v1 = U[:, i1] / np.linalg.norm(U[:, i1])
        v2 = D[s] @ v1
        Q = np.column_stack([v1, v2])
        Dn = np.einsum("ji,gjk,kl->gil", Q.conj(), D, Q)
        assert np.allclose(Dn[s], [[0, 1], [1, 0]], atol=1e-8)
        return Dn

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------
    def chi(self, i, probe):
        return self.chars[i][self.probes[probe]]

    def labeler(self, i):
        return LABELERS[self.family](self, i)

    def _sortkey(self, i):
        return label_sortkey(self.labels[i])


# ----------------------------------------------------------------------------
# labelling helpers.  A label is LaTeX math (without $).
# ----------------------------------------------------------------------------
MAIN_ORDER = ["A", "B", "E", "T", "G", "H", r"\Sigma", r"\Pi", r"\Delta", r"\Phi"]


def label_sortkey(lab):
    import re
    if lab.startswith("skip"):
        return ("z", 0, 0, 0, 0, 0)
    m = re.match(r"^(?:\{\}\^\{(\d)\})?(\\?[A-Za-z]+)(?:\^\{([+-])\})?(?:_\{([0-9]*)([gu]?)\})?('*)$", lab)
    assert m, lab
    pre, main, sup, num, par, pr = m.groups()
    return (par or "", len(pr), MAIN_ORDER.index(main), int(num) if num else 0,
            0 if sup != "-" else 1, int(pre) if pre else 0)


def _sign(x):
    return 1 if x.real > 0 else -1


def _frac_angle(z, n):
    """return k in 0..n-1 with z = exp(2 pi i k / n)"""
    k = int(round(np.angle(z) / (2 * PI) * n)) % n
    assert abs(z - np.exp(2j * PI * k / n)) < 1e-6, (z, n)
    return k


def mk(main, sub="", pre="", prime="", sup=""):
    s = ""
    if pre:
        s += "{}^{%s}" % pre
    s += main
    if sup:
        s += "^{%s}" % sup
    if sub:
        s += "_{%s}" % sub
    s += prime
    return s


def _parity(G, i):
    if "i" in G.probes:
        return "g" if _sign(G.chi(i, "i") / G.dims[i]) > 0 else "u"
    return ""


def _prime(G, i):
    if "i" not in G.probes and "sh" in G.probes:
        return "'" if _sign(G.chi(i, "sh") / G.dims[i]) > 0 else "''"
    return ""


def _cyclic_core(n, z):
    """1D irrep of C_n with character z on generator -> (main, pre, k)"""
    k = _frac_angle(z, n)
    if k == 0:
        return "A", "", 0
    if 2 * k == n:
        return "B", "", 0
    if k < n / 2:
        return "E", "2", k      # e^{+2 pi i k/n} on the generator: the atlas's {}^2E_k
    return "E", "1", n - k


def label_axial(G, i):
    fam, n, d = G.family, G.n, G.dims[i]
    par, pr = _parity(G, i), _prime(G, i)
    # generator for A/B/E classification
    if fam == "S2n" and n % 2 == 0:
        gen, order = "S2n", 2 * n
    elif fam == "Dnd" and n % 2 == 0:
        gen, order = "S2n", 2 * n
    else:
        gen, order = "Cn", n
    z = G.chi(i, gen)
    nE = G.meta.get("nE", 1)
    if d == 1:
        main, pre, k = _cyclic_core(order, z)
        sub = ""
        if main == "E" and nE > 1:
            sub = str(k)
        if main in ("A", "B") and fam in ("Cnv", "Dn", "Dnh", "Dnd"):
            if fam in ("Dn", "Dnh") and n == 2:
                # D2 / D2h: A, B1 (C2z-symmetric), B2 (C2y), B3 (C2x)
                sx, sy, sz = (_sign(G.chi(i, a)) for a in ("C2x", "C2y", "C2z"))
                if sx > 0 and sy > 0 and sz > 0:
                    main, sub = "A", ""
                else:
                    main = "B"
                    sub = "1" if sz > 0 else ("2" if sy > 0 else "3")
            else:
                ref = "C2p" if "C2p" in G.probes else "sv"
                sub = "1" if _sign(G.chi(i, ref)) > 0 else "2"
        return mk(main, sub + par, pre, pr)
    if d == 2:
        c = (z / 2).real
        k = int(round(np.arccos(np.clip(c, -1, 1)) / (2 * PI) * order))
        sub = str(k) if nE > 1 else ""
        return mk("E", sub + par, "", pr)
    raise ValueError


def label_linear(G, i):
    # finite stand-in of C_inf v or D_inf h
    n, d = G.n, G.dims[i]
    par = _parity(G, i)
    z = G.chi(i, "Cn")
    greek = {0: r"\Sigma", 1: r"\Pi", 2: r"\Delta", 3: r"\Phi"}
    if d == 1:
        k = _frac_angle(z, n)
        kk = min(k, n - k)
        if kk != 0:
            return "skip%d" % i
        sup = "+" if _sign(G.chi(i, "sv")) > 0 else "-"
        return mk(r"\Sigma", par, "", "", sup)
    c = (z / 2).real
    k = int(round(np.arccos(np.clip(c, -1, 1)) / (2 * PI) * n))
    if k > 3:
        return "skip%d" % i
    return mk(greek[k], par)


def label_cubic(G, i):
    fam, d = G.family, G.dims[i]
    par = _parity(G, i)
    if d == 1:
        if fam in ("T", "Th"):
            k = _frac_angle(G.chi(i, "C3"), 3)
            if k == 0:
                return mk("A", par)
            return mk("E", par, "2" if k == 1 else "1")      # the atlas's label of the state
        if fam == "Td":
            return mk("A", ("1" if _sign(G.chi(i, "sd")) > 0 else "2") + par)
        return mk("A", ("1" if _sign(G.chi(i, "C4")) > 0 else "2") + par)
    if d == 2:
        return mk("E", par)
    if d == 3:
        if fam in ("T", "Th"):
            return mk("T", par)
        if fam == "Td":
            return mk("T", "2" if _sign(G.chi(i, "sd")) > 0 else "1")
        return mk("T", ("1" if _sign(G.chi(i, "C4")) > 0 else "2") + par)
    raise ValueError


def label_icosa(G, i):
    d = G.dims[i]
    par = _parity(G, i)
    if d == 1:
        return mk("A", par)
    if d == 3:
        return mk("T", ("1" if G.chi(i, "C5").real > 0 else "2") + par)
    return mk({4: "G", 5: "H"}[d], par)


LINEAR = ("Cinfv", "Dinfh")      # the linear groups, built as finite stand-ins (linear_groups)
LABELERS = {}
for f in ("C1", "Cs", "Ci", "Cn", "Cnv", "Cnh", "S2n", "Dn", "Dnh", "Dnd"):
    LABELERS[f] = label_axial
for f in ("T", "Th", "Td", "O", "Oh"):
    LABELERS[f] = label_cubic
for f in ("I", "Ih"):
    LABELERS[f] = label_icosa
for f in LINEAR:
    LABELERS[f] = label_linear


# ----------------------------------------------------------------------------
# the atlas's data (atlas_irreps.json, atlas_labels.json)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS_RECORD = {"Cs": "C1h", "Ci": "S2"}     # every other finite group has the atlas's name
_ATLAS = {}


def atlas_data():
    if "irreps" not in _ATLAS:
        with open(os.path.join(HERE, "atlas_irreps.json")) as f:
            _ATLAS["irreps"] = json.load(f)
    return _ATLAS["irreps"]


def atlas_labels():
    if "labels" not in _ATLAS:
        with open(os.path.join(HERE, "atlas_labels.json"), encoding="utf-8") as f:
            _ATLAS["labels"] = json.load(f)["groups"]
    return _ATLAS["labels"]


def atlas_ids(name, labels):
    """the atlas's record and the id of each irrep, from the label map"""
    g = atlas_labels()[name]
    return g["atlas_record"], [g["labels"][lab]["id"] for lab in labels]


# ----------------------------------------------------------------------------
# group catalogue
# ----------------------------------------------------------------------------
def axial_group(fam, n):
    Cn = Rz(2 * PI / n)
    S2n = SH @ Rz(PI / n)
    probes = {"Cn": Cn, "C2p": C2X, "sv": SV, "sh": SH, "i": INV, "S2n": S2n,
              "C2x": C2X, "C2y": C2Y, "C2z": C2Z}
    if fam == "Cn":
        gens, name = [Cn], "C%d" % n
    elif fam == "Cnv":
        gens, name = [Cn, SV], "C%dv" % n
    elif fam == "Cnh":
        gens, name = [Cn, SH], "C%dh" % n
    elif fam == "S2n":
        gens, name = [S2n], "S%d" % (2 * n)
    elif fam == "Dn":
        gens, name = [Cn, C2X], "D%d" % n
    elif fam == "Dnh":
        gens, name = [Cn, C2X, SH], "D%dh" % n
    elif fam == "Dnd":
        gens, name = [S2n, C2X], "D%dd" % n
    # sigma_v / C2' only meaningful where present (closure decides)
    # number of distinct E_k (for subscripting)
    if fam == "S2n" and n % 2 == 0:
        nE = n - 1
    elif fam == "Dnd" and n % 2 == 0:
        nE = n - 1
    else:
        nE = (n - 1) // 2
    G = FiniteGroup(name, fam, gens, probes, n=n, meta={"nE": nE})
    return G


def special_groups():
    out = []
    out.append(FiniteGroup("C1", "C1", [E3], {"Cn": E3}, n=1))
    out.append(FiniteGroup("Cs", "Cs", [SH], {"Cn": E3, "sh": SH}, n=1))
    out.append(FiniteGroup("Ci", "Ci", [INV], {"Cn": E3, "i": INV}, n=1))
    return out


def cubic_groups():
    pr = {"C3": C3D, "C4": C4Z, "sd": SD_XY, "i": INV, "C5": C5I}
    base = [C2Z, C2X, C3D]
    defs = [("T", base), ("Th", base + [INV]), ("Td", base + [SD_XY]), ("O", base + [C4Z]),
            ("Oh", base + [C4Z, INV]), ("I", base + [C5I]), ("Ih", base + [C5I, INV])]
    return [FiniteGroup(name, name, gens, pr) for name, gens in defs]


def linear_groups():
    """finite stand-ins of C_inf v and D_inf h (C13v, D14h), valid for Xi <= 3"""
    out = []
    specs = [("Cinfv", "Cnv", 13), ("Dinfh", "Dnh", 14)]
    for name, fam, n in specs:
        Cn = Rz(2 * PI / n)
        gens = {"Cnv": [Cn, SV], "Dnh": [Cn, C2X, SH]}[fam]
        probes = {"Cn": Cn, "sv": SV, "sh": SH, "i": INV, "C2p": C2X}
        G = FiniteGroup(name, name, gens, probes, n=n)
        keep = [i for i, l in enumerate(G.labels) if not l.startswith("skip")]
        G.D = [G.D[i] for i in keep]
        G.chars = [G.chars[i] for i in keep]
        G.dims = [G.dims[i] for i in keep]
        G.labels = [G.labels[i] for i in keep]
        G.atlas_record, G.ids = atlas_ids(G.name, G.labels)
        G.truncated = True
        out.append(G)
    return out


def all_finite_groups(nmax=10):
    groups = special_groups()
    for fam in ("Cn", "Cnv", "Cnh", "S2n", "Dn", "Dnh", "Dnd"):
        for n in range(2, nmax + 1):
            groups.append(axial_group(fam, n))
    groups += cubic_groups()
    groups += linear_groups()
    return groups
