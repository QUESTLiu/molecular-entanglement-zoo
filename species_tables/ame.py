"""
The catalogue of AME states realized by the coupling tensors of the species tables, and the exact checks behind it.

By Schur's lemma every coupling tensor C of an admissible triple, and every member of an N = 2 coupling space, has
flat one-sector marginals, rho_X = I/d_X: it is an AME state of its format, the sorted carrier dimensions. With
each N = 2 space in its K basis, the coupling tensors of the 75 groups are 21 such states up to local unitaries (LU)
and the order of the carriers: the product state, four maximally entangled pairs and 16 tripartite states. A tensor
gets its catalogue id from two exact LU invariants, with no optimization:

  sym   the real dimension of {(X, Y, Z) Hermitian : (X.1.1 + 1.Y.1 + 1.1.Z) C = 0}, the kernel of a real-linear
        map (SVD threshold 1e-9), with the two trivial phase directions (scalar X, Y, Z summing to zero) included;
  p3    Tr[(rho_YZ^{T_Z})^3] for the legs (0, 1), a degree-6 LU invariant that for a pure state does not depend on
        the pair; matched to the catalogue value within 1e-9.

The 21 rows have pairwise different (format, sym, p3), which selftest() asserts, so the lookup is unique and no two
rows are related by local unitaries. sym and p3 are internal: they are this module's lookup keys and self-test checks,
and no other file (results.pkl, species_tables.pdf, species_data.json, ame_catalogue.json) holds them.

An N = 2 space is listed by its K basis. The first two legs whose labels share the letter T or H (for (G,H,H) the
two H legs) are whole SO(3) irreps D^j restricted to the proper rotations (j = 1 for T, 2 for H). Their intertwiners
U (U D_Gamma(g) = D^j(R_g) U, averaged over the proper elements, R_g = G.elems[g]) give J^Gamma = U^dag J U, and the
copies are the eigenvectors of the pair Casimir (J_Y + J_Z)^2 on the coupling space, eigenvalue K(K+1), in ascending K.

Contents:
  CATALOGUE, IDS, INDEX      the 21 rows in catalogue order, their ids, and id -> position
  invariants(T)              (sorted dims, sym, p3)
  lookup(T)                  the catalogue id of a tensor; raises LookupError when no row matches
  check_flat(T)              max_X |rho_X - I/d_X|; raises ValueError above 1e-10, else returns it
  check_pair_projector(G, idx, m, T)
                             for an N = 1 tensor, max_X |rho_YZ - P_Lambda/d_X|, P_Lambda the projector onto the
                             Lambda = conj(Gamma_X) x Gamma_mol component of Gamma_Y x Gamma_Z; raises ValueError
                             above 1e-10, else returns it
  kbasis(G, idx, m, F)       F: the orthonormal coupling-space basis, shape (d_r d_n d_v, 2) (or a list of the two
                             tensors); returns [(K, T)] in ascending K, T of shape (d_r, d_n, d_v), unit norm, NOT
                             phase-fixed (species.phase_fix is applied by the caller)
  catalogue_json(results)    the list for ame_catalogue.json, each representative tensor taken from `results`
                             (species.py's list of group dicts)
  selftest()                 the catalogue's distinct keys; each representative's id, flat marginals, pair projector
                             and invariants; the explicit forms (forms()); the 3j blocks, also of the (T,T,T) copies
                             of T; the K basis of all 45 N = 2 rows of T, Th, I, Ih; and the two GHZ_5 points of the
                             (H,H,H) space
G is a pointgroups.FiniteGroup, idx = (r, n, v) its irrep indices and m the index of Gamma_mol.

Fields of a CATALOGUE row:
  id, id_tex       "P", "B1".."B4", "T1".."T16"; the id in math mode (\\catP, \\catB{1}, \\catT{13})
  format           the sorted carrier dimensions
  N, N_text, N_tex the dimension of the representative's coupling space; the N column as plain text and LaTeX
                   (T3, T4: "1; 2 in T, Th")
  sym, p3          the row's invariants (int; fractions.Fraction), the lookup keys; internal, see above
  block3j, literal (j1, j2, j3) of a Wigner 3j of smallest j1 + j2 + j3 whose point-group block is the state (for T11
                   and T16 its permutations tie), in the representative's carrier order (r, n, v), or None; literal:
                   every carrier is a whole D^j, so the state is the 3j symbol in another basis
  block_tex        the 3j-block column in LaTeX
  name             plain-text name (the README's table, ame_catalogue.json); name_tex the state column in LaTeX
  occurs_tex       where the state occurs (carriers up to order, any Gamma_mol) in LaTeX
  rep              the representative: dict(group, labels=(r, n, v), mol, K) in this code's names, K for N = 2
  block_rep        (T1 only) the triple, in O, whose 3j block block3j and block_tex describe
The LaTeX fields use make_pdf.names_tex's placeholders @@G:<group>@@ and @@L:<group>:<label>@@, the atlas's macros
(\\irr, \\grp, \\irrprod, \\rootunity) and the PREAMBLE commands \\ket, \\Gmol, \\pairK (the SO(3) pair label K) and
\\catP, \\catB, \\catT (the catalogue ids).

numpy only. pointgroups is imported lazily, by selftest() alone; species and make_pdf are never imported, and nothing
is computed at import time.

    python -B ame.py            -> selftest()
"""
import itertools
import sys
from fractions import Fraction
from math import factorial, sqrt

import numpy as np

TOL_FLAT = 1e-10      # flat marginals and the pair-projector identity
TOL_SVD = 1e-9        # singular values counted as zero in sym
TOL_P3 = 1e-9         # p3 match
TOL_K = 1e-10         # K-basis construction (intertwiners, Casimir)
SPIN = {"T": 1, "H": 2}   # SO(3) spin of the pair legs, by letter


# ----------------------------------------------------------------------------
# the catalogue
# ----------------------------------------------------------------------------
def _rep(group, labels, mol, K=None):
    return dict(group=group, labels=tuple(labels), mol=mol, K=K)


_ICO = r"$@@G:I@@$, $@@G:Ih@@$"
_ICO_ = _ICO + ": "
_CUB = r"$@@G:Td@@$, $@@G:O@@$, $@@G:Oh@@$"
_AX = r"non-Abelian axial groups, $@@G:Cinfv@@$, $@@G:Dinfh@@$"
_T1, _T2, _G, _H = "@@L:I:T_{1}@@", "@@L:I:T_{2}@@", "@@L:I:G@@", "@@L:I:H@@"
_TA = r"\irr{t}_{a}"


def _row(id, fmt, sym, p3, N, block, name, name_tex, occurs_tex, rep, literal=False, N_text=None, N_tex=None,
         **extra):
    br = extra.get("block_rep", rep)         # the 3j symbol and the triple it is the block of (block_rep, else rep)
    block_tex = "---" if block is None else (r"$3j(%d,%d,%d)$ for $(%s)$" % (block + (",".join(
        "@@L:%s:%s@@" % (br["group"], lab) for lab in br["labels"]),)) + (", literal" if literal else ""))
    id_tex = r"\catP" if id == "P" else r"\cat%s{%s}" % (id[0], id[1:])     # \catB{1}, \catT{13}
    row = dict(id=id, id_tex=id_tex, format=fmt, N=N, N_text=N_text or str(N), N_tex=N_tex or str(N), sym=sym, p3=p3,
               block3j=block, literal=literal, block_tex=block_tex, name=name, name_tex=name_tex,
               occurs_tex=occurs_tex, rep=rep)
    row.update(extra)
    return row


_F = Fraction
CATALOGUE = [
    _row("P", (1, 1, 1), 2, _F(1), 1, None, "product", "product", "every group", _rep("C1", ("A", "A", "A"), "A")),
    _row("B1", (1, 2, 2), 5, _F(1, 4), 1, None, "maximally entangled pair", "maximally entangled pair",
         _AX + ", " + _CUB, _rep("C3v", ("A_{1}", "E", "E"), "A_{1}")),
    _row("B2", (1, 3, 3), 10, _F(1, 9), 1, None, "maximally entangled pair", "maximally entangled pair",
         r"$@@G:T@@$, $@@G:Th@@$, " + _CUB + ", " + _ICO, _rep("O", ("A_{1}", "T_{1}", "T_{1}"), "A_{1}")),
    _row("B3", (1, 4, 4), 17, _F(1, 16), 1, None, "maximally entangled pair", "maximally entangled pair", _ICO,
         _rep("I", ("A", "G", "G"), "A")),
    _row("B4", (1, 5, 5), 26, _F(1, 25), 1, None, "maximally entangled pair", "maximally entangled pair", _ICO,
         _rep("I", ("A", "H", "H"), "A")),
    _row("T1", (2, 2, 2), 4, _F(1, 4), 1, (2, 2, 2), "GHZ",
         r"GHZ, $\tfrac{1}{2}\sqrt{2}(\ket{111}+\ket{222})$",
         r"$@@G:Cinfv@@$, $@@G:Dinfh@@$ and the non-Abelian axial groups other than $@@G:C4v@@$, $@@G:D4@@$, "
         r"$@@G:D2d@@$, $@@G:D4h@@$: every admissible $(\irr{e}_{a},\irr{e}_{b},\irr{e}_{c})$; " + _CUB
         + r": $(\irr{e},\irr{e},\irr{e})$",
         _rep("C3v", ("E", "E", "E"), "A_{1}"), block_rep=_rep("O", ("E", "E", "E"), "A_{1}")),
    _row("T2", (2, 3, 3), 4, _F(1, 9), 1, (2, 1, 1), "trine-GHZ",
         r"trine-GHZ, $\tfrac{1}{3}\sqrt{3}\sum_{i}\ket{a_{i}}\ket{i}\ket{i}$, "
         r"$a_{i}=\tfrac{1}{2}\sqrt{2}(\rootunity{3}^{i},\rootunity{3}^{-i})$",
         _CUB + r": every admissible $(\irr{e},\irr{t}_{a},\irr{t}_{b})$", _rep("O", ("E", "T_{1}", "T_{1}"), "A_{1}")),
    _row("T3", (3, 3, 3), 10, _F(0), 1, (1, 1, 1), "epsilon",
         r"$\epsilon=\tfrac{1}{6}\sqrt{6}\sum\epsilon_{ijk}\ket{ijk}$",
         _CUB + r": $(\irr{t}_{a},\irr{t}_{b},\irr{t}_{c})$ when the number of $\irr{t}_{2}$ carriers, plus one if "
         r"$\Gmol$ is of $\irr{a}_{2}$ type, is even; " + _ICO_ + r"$(\irr{t}_{a},\irr{t}_{a},\irr{t}_{a})$; "
         r"$@@G:T@@$, $@@G:Th@@$: $(\irr{t},\irr{t},\irr{t})$ as $\pairK=1$, together with $\catT{4}$",
         _rep("O", ("T_{1}", "T_{1}", "T_{1}"), "A_{1}"), literal=True, N_text="1; 2 in T, Th",
         N_tex=r"1; 2~in~$@@G:T@@$,~$@@G:Th@@$"),
    _row("T4", (3, 3, 3), 4, _F(1, 18), 1, (1, 1, 2), "|epsilon|",
         r"$|\epsilon|=\tfrac{1}{6}\sqrt{6}\sum|\epsilon_{ijk}|\ket{ijk}$",
         _CUB + r": the same, when that number is odd; $@@G:T@@$, $@@G:Th@@$: $(\irr{t},\irr{t},\irr{t})$ as "
         r"$\pairK=2$, together with $\catT{3}$",
         _rep("O", ("T_{1}", "T_{1}", "T_{2}"), "A_{1}"), N_text="1; 2 in T, Th",
         N_tex=r"1; 2~in~$@@G:T@@$,~$@@G:Th@@$"),
    _row("T5", (3, 3, 4), 2, _F(1, 16), 1, (1, 3, 3), "coupling T1 x T2 > G",
         r"the coupling $%s\irrprod%s\supset%s$" % (_T1, _T2, _G), _ICO_ + "$(%s,%s,%s)$" % (_T1, _T2, _G),
         _rep("I", ("T_{1}", "T_{2}", "G"), "A")),
    _row("T6", (3, 3, 5), 5, _F(23, 450), 1, (1, 1, 2), "Wigner 3j(1,1,2)", r"Wigner $3j(1,1,2)$",
         _ICO_ + "$(%s,%s,%s)$" % (_TA, _TA, _H), _rep("I", ("T_{1}", "T_{1}", "H"), "A"), literal=True),
    _row("T7", (3, 3, 5), 2, _F(1, 25), 1, (1, 3, 2), "coupling T1 x T2 > H",
         r"the coupling $%s\irrprod%s\supset%s$" % (_T1, _T2, _H), _ICO_ + "$(%s,%s,%s)$" % (_T1, _T2, _H),
         _rep("I", ("T_{1}", "T_{2}", "H"), "A")),
    _row("T8", (3, 4, 4), 8, _F(5, 144), 1, (1, 3, 3), "'t Hooft symbol",
         r"'t~Hooft symbol, $\tfrac{1}{6}\sqrt{3}\sum\eta^{a}_{\mu\nu}\ket{a\mu\nu}$",
         _ICO_ + "$(%s,%s,%s)$" % (_TA, _G, _G), _rep("I", ("T_{1}", "G", "G"), "A")),
    _row("T9", (3, 4, 5), 2, _F(89, 3600), 1, (1, 3, 2), "coupling Ta x G > H",
         r"the coupling $%s\irrprod%s\supset%s$" % (_TA, _G, _H), _ICO_ + "$(%s,%s,%s)$" % (_TA, _G, _H),
         _rep("I", ("T_{1}", "G", "H"), "A")),
    _row("T10", (3, 5, 5), 5, _F(1, 45), 1, (1, 2, 2), "Wigner 3j(1,2,2)", r"Wigner $3j(1,2,2)$",
         _ICO_ + "$(%s,%s,%s)$" % (_TA, _H, _H), _rep("I", ("T_{1}", "H", "H"), "A"), literal=True),
    _row("T11", (4, 4, 4), 2, _F(5, 144), 1, (3, 3, 4), "simplex cubic",
         r"simplex cubic, $\propto\sum_{i=1}^{5}v_{i}\otimes v_{i}\otimes v_{i}$ ($v_{i}$ the vertices of a regular "
         r"4-simplex)", _ICO_ + "$(%s,%s,%s)$" % (_G, _G, _G), _rep("I", ("G", "G", "G"), "A")),
    _row("T12", (4, 4, 5), 2, _F(89, 3600), 1, (3, 3, 2), "coupling G x G > H",
         r"the coupling $%s\irrprod%s\supset%s$" % (_G, _G, _H), _ICO_ + "$(%s,%s,%s)$" % (_G, _G, _H),
         _rep("I", ("G", "G", "H"), "A")),
    _row("T13", (4, 5, 5), 2, _F(1, 64), 2, (3, 2, 2), "the two H carriers coupled to K = 3 (antisymmetric)",
         r"the two $%s$ carriers coupled to $\pairK=3$ (antisymmetric)" % _H,
         _ICO_ + r"$(%s,%s,%s)$ as $\pairK=3$, together with $\catT{14}$" % (_G, _H, _H),
         _rep("I", ("G", "H", "H"), "A", 3)),
    _row("T14", (4, 5, 5), 2, _F(401, 14400), 2, (4, 2, 2), "the two H carriers coupled to K = 4 (symmetric)",
         r"the two $%s$ carriers coupled to $\pairK=4$ (symmetric)" % _H,
         _ICO_ + r"$(%s,%s,%s)$ as $\pairK=4$, together with $\catT{13}$" % (_G, _H, _H),
         _rep("I", ("G", "H", "H"), "A", 4)),
    _row("T15", (5, 5, 5), 5, _F(41, 2450), 2, (2, 2, 2), "Wigner 3j(2,2,2) (K = 2)",
         r"Wigner $3j(2,2,2)$ ($\pairK=2$)",
         _ICO_ + r"$(%s,%s,%s)$ as $\pairK=2$, together with $\catT{16}$" % (_H, _H, _H),
         _rep("I", ("H", "H", "H"), "A", 2), literal=True),
    _row("T16", (5, 5, 5), 2, _F(89, 4410), 2, (4, 2, 2), "K = 4 coupling", r"the $\pairK=4$ coupling",
         _ICO_ + r"$(%s,%s,%s)$ as $\pairK=4$, together with $\catT{15}$" % (_H, _H, _H),
         _rep("I", ("H", "H", "H"), "A", 4)),
]
IDS = [row["id"] for row in CATALOGUE]
INDEX = {i: k for k, i in enumerate(IDS)}


# ----------------------------------------------------------------------------
# the invariants
# ----------------------------------------------------------------------------
def _unit(T):
    T = np.asarray(T, complex)
    return T / np.linalg.norm(T)


_HB = {}


def _herm(d):
    """a real basis of the d x d Hermitian matrices: E_ii, E_ij + E_ji, -i E_ij + i E_ji"""
    if d not in _HB:
        B = []
        for i in range(d):
            M = np.zeros((d, d), complex)
            M[i, i] = 1
            B.append(M)
            for j in range(i + 1, d):
                M = np.zeros((d, d), complex)
                M[i, j] = M[j, i] = 1
                B.append(M)
                M = np.zeros((d, d), complex)
                M[i, j], M[j, i] = -1j, 1j
                B.append(M)
        _HB[d] = np.array(B)
    return _HB[d]


def sym_svals(T):
    """singular values of the real-linear map (X, Y, Z) -> (X.1.1 + 1.Y.1 + 1.1.Z) T on Hermitian triples, padded with
    zeros to the number of real parameters (so that sym = the number of values <= TOL_SVD)"""
    T = _unit(T)
    rows = []
    for leg, d in enumerate(T.shape):
        H = _herm(d)
        A = np.moveaxis(np.tensordot(H, T, axes=([2], [leg])), 1, leg + 1)     # A[r] = (H_r on leg) T
        rows.append(A.reshape(len(H), -1))
    A = np.vstack(rows)
    s = np.linalg.svd(np.hstack([A.real, A.imag]), compute_uv=False)
    return np.concatenate([s, np.zeros(A.shape[0] - len(s))])


def sym_dim(T):
    """sym: the kernel dimension of the map of sym_svals, the two phase directions included"""
    return int((sym_svals(T) <= TOL_SVD).sum())


def p3(T):
    """Tr[(rho_01^{T_1})^3], rho_01 the marginal of legs 0 and 1"""
    T = _unit(T)
    a, b = T.shape[:2]
    rho = np.tensordot(T, T.conj(), axes=([2], [2]))                   # rho[i, j, i', j']
    pt = rho.transpose(0, 3, 2, 1).reshape(a * b, a * b)
    return float(np.trace(pt @ pt @ pt).real)


def invariants(T):
    """(sorted dims, sym, p3) of the normalized tensor T"""
    return tuple(sorted(np.shape(T))), sym_dim(T), p3(T)


def lookup(T):
    """the id of the catalogue row with the format, sym and p3 of T (|p3 difference| < 1e-9); LookupError if none"""
    fmt, s, p = invariants(T)
    hits = [row["id"] for row in CATALOGUE if row["format"] == fmt and row["sym"] == s
            and abs(p - float(row["p3"])) < TOL_P3]
    if len(hits) != 1:
        raise LookupError("no catalogue row for format %s, sym %d, p3 %.12g (%s); a state is missing from the "
                          "catalogue" % (fmt, s, p, Fraction(p).limit_denominator(20000)))
    return hits[0]


def flat_dev(T):
    """max over legs X of the largest entry of |rho_X - I/d_X| (NaN for a tensor with a NaN entry)"""
    T = _unit(T)
    devs = []
    for X, d in enumerate(T.shape):
        M = np.moveaxis(T, X, 0).reshape(d, -1)
        devs.append(np.abs(M @ M.conj().T - np.eye(d) / d).max())
    return float(np.max(devs))


def check_flat(T):
    dev = flat_dev(T)
    if not dev <= TOL_FLAT:           # NaN fails too
        raise ValueError("marginal not maximally mixed: deviation %.2e" % dev)
    return dev


def _pair_proj(G, idx, m, X):
    """the projector onto the conj(Gamma_X) x Gamma_mol isotypic component of Gamma_Y x Gamma_Z, Y < Z the other
    two legs"""
    Y, Z = [k for k in range(3) if k != X]
    DY, DZ = np.asarray(G.D[idx[Y]]), np.asarray(G.D[idx[Z]])
    w = G.chars[idx[X]] * np.conj(G.chars[m])            # conj of the character of Lambda
    P = sum(w[g] * np.kron(DY[g], DZ[g]) for g in range(G.order))
    return P * G.dims[idx[X]] / G.order


def check_pair_projector(G, idx, m, T):
    """rho_YZ = P_Lambda / d_X for every leg X of the N = 1 tensor T"""
    T = _unit(T)
    dev = 0.0
    for X, d in enumerate(T.shape):
        M = np.moveaxis(T, X, 0).reshape(d, -1)
        rho = M.T @ M.conj()                                 # sum_x |T_x><T_x|
        dev = max(dev, float(np.abs(rho - _pair_proj(G, idx, m, X) / d).max()))
    if dev > TOL_FLAT:
        raise ValueError("pair marginal is not P_Lambda/d_X: deviation %.2e" % dev)
    return dev


# ----------------------------------------------------------------------------
# SO(3): spin matrices, Wigner D, 3j symbols
# ----------------------------------------------------------------------------
def spin(j):
    """J_x, J_y, J_z of the integer spin j in the basis m = j, ..., -j"""
    m = np.arange(j, -j - 1, -1.0)
    Jp = np.zeros((2 * j + 1, 2 * j + 1))
    for k in range(1, 2 * j + 1):
        Jp[k - 1, k] = sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return (Jp + Jp.T).astype(complex) / 2, (Jp - Jp.T) / 2j, np.diag(m).astype(complex)


def axis_angle(R):
    """(theta, n) with R = exp(theta [n]_x), theta in [0, pi], for a proper rotation R"""
    R = np.asarray(R, float)
    if np.linalg.det(R) < 0:
        raise ValueError("improper matrix")
    th = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(w) > 1e-6:
        return th, w / np.linalg.norm(w)
    B = (R + np.eye(3)) / 2                                  # theta = pi: B = n n^T
    k = int(np.argmax(np.diag(B)))
    return np.pi, B[:, k] / sqrt(B[k, k])


def wignerD(R, j):
    """D^j(R) = V exp(-i theta lambda) V^dag, from eigh of n.J = V diag(lambda) V^dag (basis m = j..-j)"""
    th, n = axis_angle(R)
    Jx, Jy, Jz = spin(j)
    lam, V = np.linalg.eigh(n[0] * Jx + n[1] * Jy + n[2] * Jz)
    return (V * np.exp(-1j * th * lam)) @ V.conj().T


def w3j(j1, j2, j3, m1, m2, m3):
    """the Wigner 3j symbol (Racah's formula)"""
    if m1 + m2 + m3 != 0 or not abs(j1 - j2) <= j3 <= j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    f = factorial
    tri = f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
    pre = (-1) ** (j1 - j2 - m3) * sqrt(tri * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3)
                                        * f(j3 - m3))
    s = 0.0
    for k in range(j1 + j2 + j3 + 1):
        a = (k, j3 - j2 + k + m1, j3 - j1 + k - m2, j1 + j2 - j3 - k, j1 - k - m1, j2 - k + m2)
        if min(a) >= 0:
            s += (-1) ** k / float(np.prod([f(x) for x in a]))
    return pre * s


def tau(j1, j2, j3):
    """the 3j symbol as a normalized tensor in D^j1 x D^j2 x D^j3 (bases m = j..-j); zero off the triangle"""
    T = np.zeros((2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1))
    for a, m1 in enumerate(range(j1, -j1 - 1, -1)):
        for b, m2 in enumerate(range(j2, -j2 - 1, -1)):
            c = j3 + m1 + m2                                 # index of m3 = -(m1 + m2)
            if 0 <= c <= 2 * j3:
                T[a, b, c] = w3j(j1, j2, j3, m1, m2, -(m1 + m2))
    nrm = np.linalg.norm(T)
    return T / nrm if nrm > 0 else T


# ----------------------------------------------------------------------------
# intertwiners and the K basis
# ----------------------------------------------------------------------------
def _proper(G):
    return [g for g in range(G.order) if np.linalg.det(np.asarray(G.elems[g])) > 0]


_DJ = {}


def _dj(G, j):
    """(proper elements, [D^j(R_g)]) of G, cached; the entry keeps G alive, so id(G) is not reused"""
    key = (id(G), j)
    if key not in _DJ or _DJ[key][0] is not G:
        prop = _proper(G)
        _DJ[key] = (G, prop, [wignerD(G.elems[g], j) for g in prop])
    return _DJ[key][1], _DJ[key][2]


def intertwiner(G, i, j):
    """the isometry U from irrep i to C^{2j+1} (rows m = j..-j) with U D_i(g) = D^j(R_g) U for every proper g,
    R_g = G.elems[g]: the twirl of a random matrix over the proper elements, unique up to a phase. Irrep i must occur
    exactly once in D^j restricted to the proper rotations of G, b_Gamma(j) = 1."""
    prop, DJ = _dj(G, j)
    D = np.asarray(G.D[i])
    b = sum(np.conj(G.chars[i][g]) * np.trace(Dj) for g, Dj in zip(prop, DJ)) / len(prop)
    if abs(b - 1) > 1e-8:
        raise ValueError("irrep %s of %s occurs %.3f times in D^%d" % (G.labels[i], G.name, b.real, j))
    rng = np.random.default_rng(0)
    for _ in range(5):
        X = rng.normal(size=(2 * j + 1, D.shape[1])) + 1j * rng.normal(size=(2 * j + 1, D.shape[1]))
        U = sum(Dj @ X @ D[g].conj().T for g, Dj in zip(prop, DJ)) / len(prop)
        nrm = np.linalg.norm(U)
        if nrm > 1e-6:
            break
    U = U * sqrt(D.shape[1]) / nrm
    res = max(float(np.abs(U.conj().T @ U - np.eye(D.shape[1])).max()),
              max(float(np.abs(U @ D[g] - Dj @ U).max()) for g, Dj in zip(prop, DJ)))
    if res > TOL_K:
        raise ValueError("intertwiner %s -> D^%d of %s fails: residual %.2e" % (G.labels[i], j, G.name, res))
    return U


def coupling_projector(G, idx, m):
    """P = (1/|G|) sum_g chi*_mol(g) D_r(g) x D_n(g) x D_v(g)"""
    D = [np.asarray(G.D[i]) for i in idx]
    P = sum(np.conj(G.chars[m][g]) * np.kron(np.kron(D[0][g], D[1][g]), D[2][g]) for g in range(G.order))
    return P / G.order


def coupling_space(G, idx, m):
    """orthonormal basis (columns) of the range of P, built as species.tensors_finite builds it"""
    P = coupling_projector(G, idx, m)
    w, U = np.linalg.eigh((P + P.conj().T) / 2)
    return U[:, w > 0.5]


def pair_of(labels):
    """the first two legs whose labels share the letter T or H (g/u ignored)"""
    for a, b in itertools.combinations(range(3), 2):
        if labels[a][0] == labels[b][0] and labels[a][0] in SPIN:
            return a, b
    raise ValueError("no T or H pair in %s" % (labels,))


def _leg_op(A, leg, dims):
    ops = [np.eye(d, dtype=complex) for d in dims]
    ops[leg] = A
    return np.kron(np.kron(ops[0], ops[1]), ops[2])


def pair_casimir(G, idx, pair):
    """(J_Y + J_Z)^2 on the product space, J^Gamma = U^dag J^(j) U on the two legs of the pair"""
    dims = [G.dims[i] for i in idx]
    Js = []
    for leg in pair:
        i = idx[leg]
        j = SPIN[G.labels[i][0]]
        if dims[leg] != 2 * j + 1:
            raise ValueError("leg %s of %s is not a whole D^%d" % (G.labels[i], G.name, j))
        U = intertwiner(G, i, j)
        Js.append([U.conj().T @ Ja @ U for Ja in spin(j)])
    C = np.zeros((int(np.prod(dims)),) * 2, complex)
    for a in range(3):
        Jt = _leg_op(Js[0][a], pair[0], dims) + _leg_op(Js[1][a], pair[1], dims)
        C += Jt @ Jt
    return C


def kbasis_info(G, idx, m, F, pair=None):
    """kbasis with its residual: dict(K, T, resid) with resid = max(|C F - F M|, max_K |C T_K - K(K+1) T_K|),
    M = F^dag C F"""
    if not isinstance(F, np.ndarray):
        F = np.array([np.ravel(t) for t in F]).T
    F = np.asarray(F, complex)
    dims = tuple(G.dims[i] for i in idx)
    labels = [G.labels[i] for i in idx]
    if F.shape[0] != int(np.prod(dims)):
        raise ValueError("F has %d rows for dims %s" % (F.shape[0], dims))
    Fres = max(float(np.abs(F.conj().T @ F - np.eye(F.shape[1])).max()),
               float(np.abs(coupling_projector(G, idx, m) @ F - F).max()))
    if Fres > TOL_K:
        raise ValueError("F is not an orthonormal basis of the coupling space: residual %.2e" % Fres)
    pair = pair_of(labels) if pair is None else tuple(pair)
    C = pair_casimir(G, idx, pair)
    M = F.conj().T @ C @ F
    M = (M + M.conj().T) / 2
    lam, W = np.linalg.eigh(M)
    Ks = [int(round((-1 + sqrt(1 + 4 * max(x, 0))) / 2)) for x in lam]
    Ts = [F @ W[:, k] for k in range(F.shape[1])]
    resid = max([float(np.abs(C @ F - F @ M).max())] +
                [float(np.abs(C @ t - K * (K + 1) * t).max()) for K, t in zip(Ks, Ts)])
    if resid > TOL_K or len(set(Ks)) != len(Ks):
        raise ValueError("no K basis for %s %s: K %s, Casimir residual %.2e" % (G.name, labels, Ks, resid))
    return dict(K=Ks, T=[t.reshape(dims) for t in Ts], resid=resid)


def kbasis(G, idx, m, F):
    """the K basis of an N = 2 coupling space: [(K, T)] in ascending K, unit norm, not phase-fixed"""
    info = kbasis_info(G, idx, m, F)
    return list(zip(info["K"], info["T"]))


# ----------------------------------------------------------------------------
# ame_catalogue.json
# ----------------------------------------------------------------------------
def _entries(T):
    T = np.asarray(T)
    return [[int(i), int(j), int(k), float(T[i, j, k].real), float(T[i, j, k].imag)]
            for i, j, k in zip(*np.nonzero(abs(T) > 1e-10))]


def catalogue_json(results):
    """one dict per catalogue row, in catalogue order, for species_tables/ame_catalogue.json: id, format, N, N_text,
    name, block3j, representative, dims and tensor (sym and p3 stay internal). The representative tensor (with its
    phase as stored) is that row of `results`, checked to have the row's id"""
    groups = {g["name"]: g for g in results}
    out = []
    for row in CATALOGUE:
        rep = row["rep"]
        g = groups[rep["group"]]
        labels = list(g.get("repo_labels") or g["labels"])
        r, n, v = (labels.index(lab) for lab in rep["labels"])
        m = labels.index(rep["mol"])
        hits = [x for x in g["tables"][m] if (x["r"], x["n"], x["v"]) == (r, n, v)]
        if len(hits) != 1 or hits[0]["N"] != row["N"]:
            raise LookupError("catalogue row %s: representative not found once with N = %d (%d hits)"
                              % (row["id"], row["N"], len(hits)))
        comps = hits[0]["comps"] if row["N"] == 1 else [c for c in hits[0]["comps"] if c.get("K") == rep["K"]]
        if not comps:
            raise LookupError("catalogue row %s: no copy with K = %s" % (row["id"], rep["K"]))
        c = comps[0]
        T = np.asarray(c["T"])
        got = lookup(T)
        if got != row["id"] or c.get("state", got) != got:
            raise ValueError("catalogue row %s: representative looks up as %s, stored state %s"
                             % (row["id"], got, c.get("state")))
        out.append(dict(
            id=row["id"], format=list(row["format"]), N=row["N"], N_text=row["N_text"], name=row["name"],
            block3j=None if row["block3j"] is None else dict(
                js=list(row["block3j"]), literal=row["literal"], group=row.get("block_rep", rep)["group"]),
            representative=dict(group=rep["group"], r=rep["labels"][0], n=rep["labels"][1], v=rep["labels"][2],
                                mol=rep["mol"], K=rep["K"]),
            dims=list(T.shape), tensor=_entries(T)))
    return out


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
_GROUPS = {}


def _group(name):
    import re
    import pointgroups as pg
    if name not in _GROUPS:
        if name in ("C1", "Cs", "Ci"):
            gs = pg.special_groups()
        elif name in ("T", "Th", "Td", "O", "Oh", "I", "Ih"):
            gs = pg.cubic_groups()
        else:
            mm = re.fullmatch(r"([CDS])(\d+)([vhd]?)", name)
            k, fam = int(mm.group(2)), {("C", ""): "Cn", ("C", "v"): "Cnv", ("C", "h"): "Cnh", ("S", ""): "S2n",
                                         ("D", ""): "Dn", ("D", "h"): "Dnh", ("D", "d"): "Dnd"}[mm.group(1, 3)]
            gs = [pg.axial_group(fam, k // 2 if fam == "S2n" else k)]
        for G in gs:
            _GROUPS.setdefault(G.name, G)
    return _GROUPS[name]


def real1d(G):
    """the real one-dimensional irreps (the Gamma_mol choices)"""
    return [i for i, (d, ch) in enumerate(zip(G.dims, G.chars))
            if d == 1 and np.allclose(ch.imag, 0, atol=1e-8) and np.allclose(abs(ch.real), 1, atol=1e-8)]


def multiplicity(G, idx, m):
    val = np.sum(G.chars[idx[0]] * G.chars[idx[1]] * G.chars[idx[2]] * np.conj(G.chars[m])) / G.order
    k = int(round(val.real))
    assert abs(val - k) < 1e-6
    return k


def n2_rows():
    """every N = 2 row (group, idx, m) of T, Th, I and Ih"""
    out = []
    for name in ("T", "Th", "I", "Ih"):
        G = _group(name)
        for m in real1d(G):
            for idx in itertools.product(range(len(G.labels)), repeat=3):
                if multiplicity(G, idx, m) == 2:
                    out.append((G, idx, m))
    return out


def _resolve(G, rep):
    return tuple(G.labels.index(lab) for lab in rep["labels"]), G.labels.index(rep["mol"])


def forms():
    """the explicit forms named in Sec. 2 of the PDF (GHZ, trine-GHZ, epsilon, |epsilon|, the 't Hooft symbol eta and
    the Wigner 3j symbols of the literal rows), built here, with the row each must match"""
    w = np.exp(2j * np.pi / 3)
    ghz = np.zeros((2, 2, 2))
    ghz[0, 0, 0] = ghz[1, 1, 1] = 1
    trine = np.zeros((2, 3, 3), complex)
    for i in range(3):
        trine[:, i, i] = [w ** i, w ** -i]
    eps = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        eps[p] = np.linalg.det(np.eye(3)[list(p)])
    eta = np.zeros((3, 4, 4))
    eta[:, :3, :3] = eps
    for a in range(3):
        eta[a, a, 3], eta[a, 3, a] = 1, -1
    return [("GHZ", ghz, "T1"), ("trine-GHZ", trine, "T2"), ("epsilon", eps, "T3"), ("|epsilon|", abs(eps), "T4"),
            ("'t Hooft eta", eta, "T8"), ("3j(1,1,1)", tau(1, 1, 1), "T3"), ("3j(1,1,2)", tau(1, 1, 2), "T6"),
            ("3j(1,2,2)", tau(1, 2, 2), "T10"), ("3j(2,2,2)", tau(2, 2, 2), "T15")]


def _ray_dev(a, b):
    """max |b - <a|b> a| for normalized a, b: zero exactly on one ray, linear in the angle"""
    a, b = np.ravel(a) / np.linalg.norm(a), np.ravel(b) / np.linalg.norm(b)
    return float(np.abs(b - np.vdot(a, b) * a).max())


def _block(G, idx, js, iso):
    """the 3j(js) block on the carriers idx: (W0^dag x W1^dag x W2^dag) tau with intertwiners W (iso=True), or the
    block (P0 x P1 x P2) tau of the isotypic projectors P in D^j (iso=False, any multiplicity)"""
    t = tau(*js)
    if iso:
        W = [intertwiner(G, i, j) for i, j in zip(idx, js)]
        return np.einsum("ai,bj,ck,abc->ijk", W[0].conj(), W[1].conj(), W[2].conj(), t)
    P = []
    for i, j in zip(idx, js):
        prop, DJ = _dj(G, j)
        Pi = G.dims[i] / len(prop) * sum(np.conj(G.chars[i][g]) * Dj for g, Dj in zip(prop, DJ))
        if np.abs(Pi).max() < 1e-9:
            return np.zeros(1)
        P.append(Pi)
    return np.einsum("ia,jb,kc,abc->ijk", P[0], P[1], P[2], t)


def selftest():
    """raises AssertionError on the first failure; returns the measured residuals"""
    if not __debug__:
        sys.exit("ame.py: run without -O or PYTHONOPTIMIZE, since its checks are assert statements")
    rep = {}
    # 0. the catalogue itself
    assert len(CATALOGUE) == 21 and IDS == ["P", "B1", "B2", "B3", "B4"] + ["T%d" % k for k in range(1, 17)], IDS
    keys = [(r["format"], r["sym"], r["p3"]) for r in CATALOGUE]
    assert len(set(keys)) == 21, "two rows share (format, sym, p3)"
    assert all(isinstance(r["p3"], Fraction) and tuple(sorted(r["format"])) == r["format"] for r in CATALOGUE)
    # SO(3) machinery: D^j is a homomorphism on I, and the 3j symbols are invariant
    I = _group("I")
    g, h = 7, 41
    for j in range(5):
        dev = np.abs(wignerD(I.elems[I.mul[g, h]], j) - wignerD(I.elems[g], j) @ wignerD(I.elems[h], j)).max()
        assert dev < 1e-12, ("D^j homomorphism", j, dev)
    R = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    R = R * np.sign(np.linalg.det(R))
    for js in [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2), (4, 2, 2), (3, 3, 4), (1, 3, 3), (1, 3, 2)]:
        t = tau(*js)
        dev = np.abs(np.einsum("ia,jb,kc,abc->ijk", *[wignerD(R, j) for j in js], t) - t).max()
        assert dev < 1e-12, ("3j invariance", js, dev)
    print("ame.selftest: catalogue of %d rows; D^j homomorphism and 3j invariance to 1e-12" % len(CATALOGUE))

    # 1. every representative: project, flat marginals, pair projector, invariants
    worst = dict(flat=0.0, pair=0.0, p3=0.0, gap=np.inf)
    for row in CATALOGUE:
        G = _group(row["rep"]["group"])
        idx, m = _resolve(G, row["rep"])
        F = coupling_space(G, idx, m)
        assert F.shape[1] == row["N"], (row["id"], F.shape)
        if row["N"] == 1:
            T = F[:, 0].reshape([G.dims[i] for i in idx])
            worst["pair"] = max(worst["pair"], check_pair_projector(G, idx, m, T))
        else:
            T = dict(kbasis(G, idx, m, F))[row["rep"]["K"]]
        worst["flat"] = max(worst["flat"], check_flat(T))
        fmt, s, p = invariants(T)
        assert lookup(T) == row["id"] and (fmt, s) == (row["format"], row["sym"]), (row["id"], fmt, s, p)
        worst["p3"] = max(worst["p3"], abs(p - float(row["p3"])))
        sv = sym_svals(T)
        worst["gap"] = min(worst["gap"], sv[sv > TOL_SVD].min())
    rep["representatives"] = worst
    print("  1. 21 representatives: ids, N; flat %.1e, pair projector %.1e, |p3 - row| %.1e, smallest nonzero "
          "singular value of the sym map %.3f" % (worst["flat"], worst["pair"], worst["p3"], worst["gap"]))

    # 2. explicit forms
    for name, T, want in forms():
        check_flat(T)
        got = lookup(T)
        assert got == want, (name, got, want, invariants(T))
    ghz3 = np.zeros((3, 3, 3))
    for i in range(3):
        ghz3[i, i, i] = 1
    try:
        lookup(ghz3)
        raise AssertionError("GHZ_3 must not match a row")
    except LookupError:
        pass
    print("  2. explicit forms: " + ", ".join("%s -> %s" % (n, w) for n, _, w in forms()) + "; GHZ_3 raises")

    # 3j blocks of the representatives: the stated block is the state, and no smaller j1+j2+j3 gives a block
    nb = 0
    for row in CATALOGUE:
        if row["block3j"] is None:
            continue
        br = row.get("block_rep", row["rep"])
        G = _group(br["group"])
        assert len(_proper(G)) == G.order
        idx, m = _resolve(G, br)
        assert np.allclose(G.chars[m], 1), row["id"]
        js = row["block3j"]
        assert row["literal"] == all(G.dims[i] == 2 * j + 1 for i, j in zip(idx, js)), row["id"]
        blk = _block(G, idx, js, True)
        F = coupling_space(G, idx, m)
        T = F[:, 0] if F.shape[1] == 1 else dict(kbasis(G, idx, m, F))[br["K"]].ravel()
        ov = _ray_dev(blk, T)
        assert ov < 1e-10, (row["id"], ov)
        if row["N"] == 1:
            for js2 in itertools.product(range(sum(js)), repeat=3):
                if sum(js2) < sum(js):
                    assert np.linalg.norm(_block(G, idx, js2, False)) < 1e-8, (row["id"], js2)
        nb += 1
    # the K copies of (T,T,T) -> A in T are the 3j(K,1,1) blocks, the K leg from D^K (the PDF names both)
    G = _group("T")
    idx, m = (G.labels.index("T"),) * 3, G.labels.index("A")
    for K, T in kbasis(G, idx, m, coupling_space(G, idx, m)):
        blk = _block(G, idx, (K, 1, 1), True)
        ov = _ray_dev(blk, T)
        assert ov < 1e-10, ("(T,T,T) of T", K, ov)
    print("  3j blocks: %d rows and the K = 1, 2 copies of (T,T,T) in T; each stated block equals its state (to "
          "1e-10 in every entry, up to phase), and for N = 1 no smaller j1+j2+j3 has a nonzero block" % nb)

    # 3. the K basis of all 45 N = 2 rows
    want = {"TTT": {1: "T3", 2: "T4"}, "GHH": {3: "T13", 4: "T14"}, "HHH": {2: "T15", 4: "T16"}}
    by_group, by_type, resid, pairdev = {}, {}, 0.0, 0.0
    rng = np.random.default_rng(1)
    for G, idx, m in n2_rows():
        labels = [G.labels[i] for i in idx]
        typ = {"TTT": "TTT", "GHH": "GHH", "HGH": "GHH", "HHG": "GHH", "HHH": "HHH"}["".join(l[0] for l in labels)]
        by_group[G.name] = by_group.get(G.name, 0) + 1
        by_type[typ] = by_type.get(typ, 0) + 1
        F = coupling_space(G, idx, m)
        info = kbasis_info(G, idx, m, F)
        resid = max(resid, info["resid"])
        ids = {K: lookup(T) for K, T in zip(info["K"], info["T"])}
        assert ids == want[typ], (G.name, labels, ids)
        for T in info["T"] + [sum(c * T for c, T in zip(rng.normal(size=2) + 1j * rng.normal(size=2), info["T"]))]:
            check_flat(T)                                    # the K tensors and a random member are AME
        if typ in ("TTT", "HHH"):
            for pair in itertools.combinations(range(3), 2):
                other = kbasis_info(G, idx, m, F, pair)
                assert other["K"] == info["K"], (G.name, labels, pair, other["K"])
                resid = max(resid, other["resid"])
                for t1, t2 in zip(info["T"], other["T"]):
                    pairdev = max(pairdev, _ray_dev(t1, t2))
    assert by_group == {"T": 1, "Th": 8, "I": 4, "Ih": 32} and by_type == {"TTT": 9, "GHH": 27, "HHH": 9}, \
        (by_group, by_type)
    assert resid < 1e-12 and pairdev < 1e-12, (resid, pairdev)
    rep["kbasis"] = dict(rows=sum(by_type.values()), resid=resid, pairdev=pairdev)
    print("  3. K basis of the %d N = 2 rows (T 1, Th 8, I 4, Ih 32): K (1,2) -> (T3,T4), (3,4) -> (T13,T14), (2,4) "
          "-> (T15,T16); Casimir residual %.1e; (T,T,T), (H,H,H): same K tensors for every pair (to %.1e in every "
          "entry, up to phase); K tensors and random members flat" % (rep["kbasis"]["rows"], resid, pairdev))

    # 4. the two GHZ_5 points of the (H,H,H) space: with K_2, K_4 real and their first nonzero entry positive (the
    #    species.phase_fix rule), psi = sqrt(4/7) K_2 + sqrt(3/7) e^{i phi} K_4 has GHZ_5's sym and p3 for
    #    e^{i phi} = (s sqrt15 + 7i sigma)/8, sigma = +-1, at one sign s and not at the other
    G = _group("I")
    idx = (G.labels.index("H"),) * 3
    K = {}
    for k, T in kbasis(G, idx, 0, coupling_space(G, idx, 0)):
        f = T.ravel()[np.argmax(abs(T.ravel()) > 1e-8)]
        K[k] = T * abs(f) / f
        assert np.abs(K[k].imag).max() < 1e-12, k
    ghz5 = np.zeros((5, 5, 5))
    for i in range(5):
        ghz5[i, i, i] = 1
    _, s5, p5 = invariants(ghz5)
    hits = []
    for s, sigma in itertools.product((1, -1), repeat=2):
        _, sym_ss, p3_ss = invariants(sqrt(4 / 7) * K[2] + sqrt(3 / 7) * (s * sqrt(15) + 7j * sigma) / 8 * K[4])
        if sym_ss == s5 and abs(p3_ss - p5) < TOL_P3:
            hits.append((s, sigma))
    assert s5 == 10 and abs(p5 - 1 / 25) < TOL_P3 and len(hits) == 2 and hits[0][0] == hits[1][0], hits
    rep["ghz5"] = hits
    print("  4. (H,H,H): sqrt(4/7) K_2 + sqrt(3/7) e^{i phi} K_4 with e^{i phi} = (%ssqrt15 +- 7i)/8 has GHZ_5's sym "
          "10 and p3 1/25 (the other sign of sqrt15 does not)" % ("" if hits[0][0] > 0 else "-"))
    print("ame.selftest: all checks passed")
    return rep


if __name__ == "__main__":
    selftest()
