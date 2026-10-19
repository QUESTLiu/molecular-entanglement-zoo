"""
Point groups of O(3): construction, irreducible representations (over C),
standard (paper-convention) bases, and Mulliken labels.

Conventions (match main_phase_space.tex):
  * principal axis z, C2' axis along x, sigma_v = xz plane;
  * two-dimensional E-type irreps are written in the complex basis
    |1E>, |2E> with D(C_n)|1E> = e^{+i theta}|1E>, and the "swap" element
    (C2'(x), sigma_v(xz), sigma_d or C2[1-10]) acting as the plain permutation;
  * three-dimensional T irreps of the cubic/icosahedral groups are written in
    a Cartesian basis (x, y, z) adapted to {C2x, C2y, C2z} with
    C3[111]: x -> y -> z acting as a permutation matrix;
  * complex-conjugate pairs of 1D irreps are kept separate (irreducible over C)
    and labelled 1E, 2E (1E has character e^{+2 pi i k/n} on the generator).
"""
import numpy as np
from fractions import Fraction

PI = np.pi
TOL = 1e-7


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
C2_1M10 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])  # C2 about [1,-1,0]
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
    def __init__(self, name, latex, family, gens, probes, n=None, meta=None):
        self.name, self.latex, self.family, self.n = name, latex, family, n
        self.meta = meta or {}
        self.elems, self.keys = closure(gens)
        self.order = len(self.elems)
        self.probe_mats = probes
        self.probes = {}
        for pname, M in probes.items():
            k = mkey(M)
            if k in self.keys:
                self.probes[pname] = self.keys[k]
        self._mult_table()
        self._classes()
        self._irreps()

    def idx(self, M):
        return self.keys[mkey(M)]

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
        self.class_of = seen

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
        # sort: dimension, then label
        order = sorted(range(len(self.D)), key=lambda i: self._sortkey(i))
        self.D = [self.D[i] for i in order]
        self.chars = [self.chars[i] for i in order]
        self.dims = [self.dims[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        assert len(set(self.labels)) == len(self.labels), (self.name, self.labels)

    def _check(self, D):
        G = self.order
        for g in range(G):
            assert np.allclose(D[g] @ D[g].conj().T, np.eye(D.shape[1]), atol=1e-8)
        for g in range(0, G, max(1, G // 12)):
            for h in range(G):
                assert np.allclose(D[g] @ D[h], D[self.mul[g, h]], atol=1e-8)

    # ------------------------------------------------------------------
    # basis adaptation
    # ------------------------------------------------------------------
    def _adapt(self, D):
        d = D.shape[1]
        if d == 1:
            return D
        fam = self.family
        if d == 2 and fam in ("Cnv", "Dn", "Dnh", "Dnd", "Td", "O", "Oh", "Cinfv", "Dinfh"):
            if fam in ("Td", "O", "Oh"):
                P = self.probes["C3"]
                s = self.probes.get("C2_1m10", self.probes.get("sd"))
            else:
                P = self.probes["S2n"] if (fam == "Dnd" and self.n % 2 == 0) else self.probes["Cn"]
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
        if d == 3 and fam in ("T", "Th", "Td", "O", "Oh", "I", "Ih"):
            ax, ay, az = self.probes["C2x"], self.probes["C2y"], self.probes["C2z"]
            M = D[ax] + 2.0 * D[ay] + 3.7 * D[az]
            M = (M + M.conj().T) / 2
            w, U = np.linalg.eigh(M)
            # vector with C2x=+1, C2y=-1, C2z=-1 -> eigenvalue 1-2-3.7
            i = int(np.argmin(abs(w - (1 - 2 - 3.7))))
            ex = U[:, i]
            C3 = self.probes["C3"]
            ey = D[C3] @ ex
            ez = D[C3] @ ey
            Q = np.column_stack([ex, ey, ez])
            Dn = np.einsum("ji,gjk,kl->gil", Q.conj(), D, Q)
            assert np.allclose(Dn[C3], C3D, atol=1e-8)
            return Dn
        return D

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
# labeling helpers.  A label is LaTeX math (without $).
# ----------------------------------------------------------------------------
MAIN_ORDER = ["A", "B", "E", "T", "G", "H", r"\Sigma", r"\Pi", r"\Delta", r"\Phi", r"\Gamma"]


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
        return "E", "1", k
    return "E", "2", n - k


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
    # finite approximant of C_inf, C_inf h, C_inf v, D_inf h
    fam, n, d = G.family, G.n, G.dims[i]
    par = _parity(G, i)
    z = G.chi(i, "Cn")
    greek = {0: r"\Sigma", 1: r"\Pi", 2: r"\Delta", 3: r"\Phi", 4: r"\Gamma"}
    if d == 1:
        k = _frac_angle(z, n)
        kk = min(k, n - k)
        if fam in ("Cinf", "Cinfh"):
            if kk == 0:
                return mk(r"\Sigma", par)
            if kk > 3 or 2 * k == n:
                return "skip%d" % i
            pre = "1" if k < n / 2 else "2"
            return mk(greek[kk], par, pre)
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
            return mk("E", par, "1" if k == 1 else "2")
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


LABELERS = {}
for f in ("C1", "Cs", "Ci", "Cn", "Cnv", "Cnh", "S2n", "Dn", "Dnh", "Dnd"):
    LABELERS[f] = label_axial
for f in ("T", "Th", "Td", "O", "Oh"):
    LABELERS[f] = label_cubic
for f in ("I", "Ih"):
    LABELERS[f] = label_icosa
for f in ("Cinf", "Cinfh", "Cinfv", "Dinfh"):
    LABELERS[f] = label_linear


# ----------------------------------------------------------------------------
# group catalogue
# ----------------------------------------------------------------------------
def axial_group(fam, n):
    Cn = Rz(2 * PI / n)
    S2n = SH @ Rz(PI / n)
    probes = {"Cn": Cn, "C2p": C2X, "sv": SV, "sh": SH, "i": INV, "S2n": S2n,
              "C2x": C2X, "C2y": C2Y, "C2z": C2Z}
    if fam == "Cn":
        gens, name, latex = [Cn], "C%d" % n, "C_{%d}" % n
    elif fam == "Cnv":
        gens, name, latex = [Cn, SV], "C%dv" % n, "C_{%d\\mathrm{v}}" % n
    elif fam == "Cnh":
        gens, name, latex = [Cn, SH], "C%dh" % n, "C_{%d\\mathrm{h}}" % n
    elif fam == "S2n":
        gens, name, latex = [S2n], "S%d" % (2 * n), "S_{%d}" % (2 * n)
    elif fam == "Dn":
        gens, name, latex = [Cn, C2X], "D%d" % n, "D_{%d}" % n
    elif fam == "Dnh":
        gens, name, latex = [Cn, C2X, SH], "D%dh" % n, "D_{%d\\mathrm{h}}" % n
    elif fam == "Dnd":
        gens, name, latex = [S2n, C2X], "D%dd" % n, "D_{%d\\mathrm{d}}" % n
    if fam in ("Cnh",) and n % 2 == 1:
        probes.pop("i")
    # sigma_v / C2' only meaningful where present (closure decides)
    # number of distinct E_k (for subscripting)
    if fam == "S2n" and n % 2 == 0:
        nE = n - 1
    elif fam == "Dnd" and n % 2 == 0:
        nE = n - 1
    else:
        nE = (n - 1) // 2
    G = FiniteGroup(name, latex, fam, gens, probes, n=n, meta={"nE": nE})
    return G


def special_groups():
    out = []
    pr = {"sh": SH, "i": INV, "Cn": E3, "sv": SV}
    out.append(FiniteGroup("C1", "C_1", "C1", [E3], {"Cn": E3}, n=1))
    out.append(FiniteGroup("Cs", "C_{\\mathrm s}", "Cs", [SH], {"Cn": E3, "sh": SH}, n=1))
    out.append(FiniteGroup("Ci", "C_{\\mathrm i}", "Ci", [INV], {"Cn": E3, "i": INV}, n=1))
    return out


def cubic_groups():
    pr = {"C2x": C2X, "C2y": C2Y, "C2z": C2Z, "C3": C3D, "C4": C4Z, "sd": SD_XY,
          "C2_1m10": C2_1M10, "i": INV, "C5": C5I}
    base = [C2Z, C2X, C3D]
    defs = [("T", "T", base), ("Th", "T_{\\mathrm h}", base + [INV]),
            ("Td", "T_{\\mathrm d}", base + [SD_XY]), ("O", "O", base + [C4Z]),
            ("Oh", "O_{\\mathrm h}", base + [C4Z, INV])]
    out = []
    for name, latex, gens in defs:
        p = dict(pr)
        if name == "Td":
            p.pop("C2_1m10")
        out.append(FiniteGroup(name, latex, name, gens, p))
    for name, latex, gens in [("I", "I", base + [C5I]), ("Ih", "I_{\\mathrm h}", base + [C5I, INV])]:
        out.append(FiniteGroup(name, latex, name, gens, pr))
    return out


def linear_groups():
    """finite approximants valid for |Lambda| <= 3"""
    out = []
    specs = [("Cinf", "C_{\\infty}", "Cn", 13), ("Cinfh", "C_{\\infty\\mathrm h}", "Cnh", 14),
             ("Cinfv", "C_{\\infty\\mathrm v}", "Cnv", 13), ("Dinfh", "D_{\\infty\\mathrm h}", "Dnh", 14)]
    for name, latex, fam, n in specs:
        Cn = Rz(2 * PI / n)
        gens = {"Cn": [Cn], "Cnh": [Cn, SH], "Cnv": [Cn, SV], "Dnh": [Cn, C2X, SH]}[fam]
        probes = {"Cn": Cn, "sv": SV, "sh": SH, "i": INV, "C2p": C2X}
        G = FiniteGroup(name, latex, name, gens, probes, n=n)
        keep = [i for i, l in enumerate(G.labels) if not l.startswith("skip")]
        G.D = [G.D[i] for i in keep]
        G.chars = [G.chars[i] for i in keep]
        G.dims = [G.dims[i] for i in keep]
        G.labels = [G.labels[i] for i in keep]
        G.truncated = True
        out.append(G)
    return out


def all_finite_groups(nmax=8):
    groups = special_groups()
    for fam in ("Cn", "Cnv", "Cnh", "S2n", "Dn", "Dnh", "Dnd"):
        for n in range(2, nmax + 1):
            groups.append(axial_group(fam, n))
    groups += cubic_groups()
    groups += linear_groups()
    return groups


if __name__ == "__main__":
    import sys
    for G in all_finite_groups(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
        print(G.name, G.order, list(zip(G.labels, G.dims)))
