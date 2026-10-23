"""Symmetry-adapted Fock states (SALCs) of one degenerate vibrational mode, for every point group.

For each group of ../species_tables (same irreps, bases and Mulliken labels) and each degenerate
mode Gamma_s, the v-quantum states are organised by three structural rules, so that a finite table
gives the SALCs for every v:

  (1) parity    v and the vibrational angular momentum (l or L) have the same parity;
  (2) residues  two-dimensional modes:   the species and the SALC of |n+,n-> depend only on
                l = n+ - n- modulo a period N' (tables in l mod N');
  (3) harmonics vector-like three-dimensional modes (D(R) = eps(R) R):
                Sym^v = (+)_{L = v, v-2, ...} (a+.a+)^((v-L)/2) H_L,
                so every SALC is a normalised power of the invariant a+.a+ times a G-adapted
                harmonic SALC of degree L (tables in L).

Modes not covered by (2) or (3) (T2 of I/Ih, G and H of I/Ih) are tabulated explicitly for v <= 6 (T2) and v <= 3 (G, H).
Worked examples give explicit SALCs at small v for molecule-relevant groups.

Construction (as in salc_fock_basis.wl): the Fock action follows Lemma 2,
U(R) a+_i U(R)^+ = sum_j D_ji(R) a+_j, and the SALCs are obtained with the matrix-valued projectors
P^G_{mu nu} = (d_G/|G|) sum_R conj(D^G_{mu nu}(R)) U(R); copies by projecting occupation states in
order, partners by P^G_{mu 1}.

Checks (the script stops on failure): every rule is verified against the projector construction
for all v up to a cutoff; every listed set of states is orthonormal, complete, and transforms with
the irrep matrices of pointgroups.py; multiplicities agree with GTPack (if
multiplicities_gtpack.json is present).

Usage:  python salc_tables.py        (NumPy, SymPy; reads ../species_tables/pointgroups.py)
        pdflatex salc_tables.tex     (twice, for the table of contents)
"""
import itertools, json, math, os, sys
from fractions import Fraction
from math import factorial, comb, gcd
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'species_tables'))
import pointgroups as pg   # noqa: E402

TOL = 1e-8
V_MULT = 12          # multiplicity tables
L_HARM = 6           # explicit harmonic SALCs, L = 0..L_HARM (rule checked for v <= L_HARM)
L_BRANCH = 12        # species of H_L, L = 0..L_BRANCH
V_EXPLICIT = {3: 6, 4: 3, 5: 3}   # explicit tables for non-factorised modes, by mode dimension
V_EXAMPLE = {2: 6, 3: 6, 4: 6, 5: 6}

# ----------------------------------------------------------------------------------------
# Fock space of one mode
# ----------------------------------------------------------------------------------------

def fock_basis(d, v):
    """Occupation tuples (n_1..n_d), sum v, ordered (v,0,..) first (as ReverseSort[FrobeniusSolve])."""
    return [c for c in itertools.product(range(v, -1, -1), repeat=d) if sum(c) == v]

def _creation(d, k):
    """a+_j : Sym^(k-1) -> Sym^k for j = 0..d-1 (dense matrices)."""
    src, dst = fock_basis(d, k - 1), fock_basis(d, k)
    idx = {b: i for i, b in enumerate(dst)}
    A = np.zeros((d, len(dst), len(src)))
    for c, n in enumerate(src):
        for j in range(d):
            m = list(n); m[j] += 1
            A[j, idx[tuple(m)], c] = math.sqrt(n[j] + 1)
    return A

_CREATION = {}

def fock_matrix(D, basis):
    """U(R) on Sym^v for a+_i -> sum_j D_ji a+_j (Lemma 2), built recursively:
    U|m> = (sum_j D_ji a+_j) U|m - e_i> / sqrt(m_i)."""
    d = D.shape[0]
    v = sum(basis[0]) if basis else 0
    U = np.ones((1, 1), complex)
    for k in range(1, v + 1):
        if (d, k) not in _CREATION:
            _CREATION[(d, k)] = _creation(d, k)
        A = _CREATION[(d, k)]
        prev = {b: i for i, b in enumerate(fock_basis(d, k - 1))}
        cur = fock_basis(d, k)
        L = np.tensordot(D, A, axes=(0, 0))            # L[i] = sum_j D_ji a+_j
        Un = np.zeros((len(cur), len(cur)), complex)
        for c, m in enumerate(cur):
            i = next(t for t in range(d) if m[t] > 0)
            pm = list(m); pm[i] -= 1
            Un[:, c] = L[i] @ U[:, prev[tuple(pm)]] / math.sqrt(m[i])
        U = Un
    order = {b: i for i, b in enumerate(fock_basis(d, v))}
    perm = [order[b] for b in basis]
    return U[np.ix_(perm, perm)]

def pair_op(d, v, create=True):
    """a+.a+ : Sym^v -> Sym^(v+2)  (create)  or  a.a : Sym^v -> Sym^(v-2)."""
    src = fock_basis(d, v)
    dst = fock_basis(d, v + 2 if create else v - 2)
    idx = {b: i for i, b in enumerate(dst)}
    M = np.zeros((len(dst), len(src)))
    for c, n in enumerate(src):
        for i in range(d):
            k = list(n)
            if create:
                k[i] += 2; f = np.sqrt((n[i] + 1) * (n[i] + 2))
            else:
                if n[i] < 2:
                    continue
                k[i] -= 2; f = np.sqrt(n[i] * (n[i] - 1))
            M[idx[tuple(k)], c] += f
    return M

# ----------------------------------------------------------------------------------------
# Projectors and SALCs
# ----------------------------------------------------------------------------------------

def phase_fix(w):
    for c in w:
        if abs(c) > 1e-7:
            return w * (abs(c) / c)
    return w

def projectors(U, irreps):
    """{label: [P^G_{mu 1} for mu]} for the Fock matrices U."""
    nG = len(U)
    Us = np.array(U)
    out = {}
    for lab, DG in irreps:
        d = DG.shape[1]
        out[lab] = [np.tensordot(np.conj(DG[:, mu, 0]), Us, axes=(0, 0)) * d / nG for mu in range(d)]
    return out

def salcs(U, irreps, candidates, restrict=None, Pc=None):
    """U: list of Fock matrices over group elements. irreps: list of (label, D[g,d,d]).
    Returns [(label, [ [partner vectors] per copy ])]."""
    if Pc is None:
        Pc = projectors(U, irreps)
    out = []
    for lab, DG in irreps:
        d = DG.shape[1]
        P = Pc[lab]
        copies = []
        for c in candidates:
            w = P[0] @ c
            if restrict is not None:
                w = restrict @ w
            for q in copies:
                w = w - q[0] * (q[0].conj() @ w)
            if np.linalg.norm(w) > 1e-7:
                w = phase_fix(w / np.linalg.norm(w))
                copies.append([w] + [P[mu] @ w for mu in range(1, d)])
        if copies:
            out.append((lab, copies))
    return out

def check_set(U, irreps_by_label, sets, complete_dim=None):
    """Covariance U(g) psi_nu = sum_mu D_mu nu(g) psi_mu, orthonormality and (optional) completeness."""
    vecs = []
    for lab, copies in sets:
        DG = irreps_by_label[lab]
        for cp in copies:
            V = np.column_stack(cp)
            for g in range(len(U)):
                assert np.allclose(U[g] @ V, V @ DG[g], atol=1e-7), ('covariance', lab)
            vecs.append(V)
    if vecs:
        A = np.column_stack(vecs)
        assert np.allclose(A.conj().T @ A, np.eye(A.shape[1]), atol=1e-7), 'orthonormality'
        if complete_dim is not None:
            assert A.shape[1] == complete_dim, ('completeness', A.shape[1], complete_dim)

# ----------------------------------------------------------------------------------------
# Characters and multiplicities (Newton's identity)
# ----------------------------------------------------------------------------------------

def sym_char(G, chi, vmax):
    n = G.order
    H = np.zeros((vmax + 1, n), complex)
    for g in range(n):
        x, p = g, []
        for _ in range(vmax):
            p.append(chi[x]); x = G.mul[x, g]
        h = [1.0 + 0j]
        for v in range(1, vmax + 1):
            h.append(sum(p[k - 1] * h[v - k] for k in range(1, v + 1)) / v)
        H[:, g] = h
    return H

def mults(G, chi_rep):
    m = [np.sum(np.conj(c) * chi_rep).real / G.order for c in G.chars]
    assert np.allclose(m, np.round(m), atol=1e-6)
    return [int(round(x)) for x in m]

# ----------------------------------------------------------------------------------------
# Exact display of coefficients
# ----------------------------------------------------------------------------------------

def _sqrt_frac(F):
    p, q = F.numerator, F.denominator
    rp, rq = int(round(p ** 0.5)), int(round(q ** 0.5))
    if rp * rp == p and rq * rq == q:
        return ('%d' % rp if rq == 1 else r'\tfrac{%d}{%d}' % (rp, rq)), (rp == 1 and rq == 1)
    return r'\sqrt{\tfrac{%d}{%d}}' % (p, q) if q != 1 else r'\sqrt{%d}' % p, False

def coeff_tex(c):
    """Return (sign, body) with sign '+' or '-', body a LaTeX factor ('' for 1)."""
    mag2 = abs(c) ** 2
    F = Fraction(mag2).limit_denominator(4000)
    ph = (np.angle(c) / (2 * np.pi)) % 1.0
    Fp = Fraction(ph).limit_denominator(48) % 1
    if abs(float(F) - mag2) < 1e-9 and abs(((float(Fp) - ph + 0.5) % 1.0) - 0.5) < 1e-9:
        m, one = _sqrt_frac(F)
        m = '' if one else m
        k, N = Fp.numerator, Fp.denominator
        if k == 0:
            return '+', m
        if Fp == Fraction(1, 2):
            return '-', m
        if Fp == Fraction(1, 4):
            return '+', m + 'i'
        if Fp == Fraction(3, 4):
            return '-', m + 'i'
        ph_t = (r'\omega' if k == 1 else r'\omega^{2}') if N == 3 else r'e^{2\pi i\,%d/%d}' % (k, N)
        return '+', m + ph_t
    if abs(c.imag) < 1e-9:
        x = c.real
        e = sp.nsimplify(abs(x), [sp.sqrt(5)], tolerance=1e-10, rational=False)
        if abs(float(e) - abs(x)) < 1e-9 and len(str(e)) <= 28:
            body = sp.latex(e)
            return ('+' if x > 0 else '-'), ('' if body == '1' else '(' + body + ')' if '+' in body or '-' in body else body)
        return ('+' if x > 0 else '-'), '%.4f' % abs(x)
    return '+', '(%.4f%+.4fi)' % (c.real, c.imag)

def ket_tex(n, kind):
    if kind == '2d':          # tuple (n_1E, n_2E) = (n-, n+)  ->  |n+, n->
        return r'\ket{%d,%d}' % (n[1], n[0])
    if max(n) > 9:
        return r'\ket{' + ','.join(map(str, n)) + '}'
    return r'\ket{' + ''.join(map(str, n)) + '}'

def state_tex(vec, basis, kind):
    terms = []
    for c, n in zip(vec, basis):
        if abs(c) > 1e-7:
            s, b = coeff_tex(c)
            terms.append((s, b + ket_tex(n, kind)))
    out = ''
    for i, (s, b) in enumerate(terms):
        out += ('' if (i == 0 and s == '+') else s) + b
    return out

# ----------------------------------------------------------------------------------------
# Modes of a group
# ----------------------------------------------------------------------------------------

def proper(G, g):
    return round(np.linalg.det(G.elems[g]))

def degenerate_modes(G):
    """[(mode label, D[g,d,d], kind)] with kind in {'2d', 'pair', '3d', 'nd'}."""
    modes, used = [], set()
    for a, (lab, d) in enumerate(zip(G.labels, G.dims)):
        if d >= 2:
            modes.append((lab, G.D[a], '2d' if d == 2 else ('3d' if d == 3 else 'nd')))
        elif np.max(np.abs(G.chars[a].imag)) > 1e-6 and a not in used:
            b = next(j for j in range(len(G.labels)) if np.allclose(G.chars[j], G.chars[a].conj()))
            used |= {a, b}
            i1, i2 = sorted([a, b], key=lambda i: G.labels[i])     # {}^{1}E first
            D = np.zeros((G.order, 2, 2), complex)
            D[:, 0, 0] = G.D[i1][:, 0, 0]; D[:, 1, 1] = G.D[i2][:, 0, 0]
            modes.append((G.labels[i1] + r'\oplus ' + G.labels[i2], D, 'pair'))
    return modes

def vector_like(G, D):
    """eps(g) with D(g) = eps(g) g for all g, or None."""
    eps = []
    for g in range(G.order):
        R = G.elems[g]
        for s in (1, -1):
            if np.allclose(D[g], s * R, atol=1e-8):
                eps.append(s); break
        else:
            return None
    return np.array(eps)

def eps_label(G, eps):
    for lab, d, chi in zip(G.labels, G.dims, G.chars):
        if d == 1 and np.allclose(chi, eps):
            return lab
    raise ValueError

# ----------------------------------------------------------------------------------------
# Rule (2): residue tables for two-dimensional modes
# ----------------------------------------------------------------------------------------

def period_2d(G, D):
    """Least common multiple of the orders of the nonzero entries of D(g): the phase acquired by
    |n+,n-> under any element is (entry)^(-l) up to a v-independent factor."""
    N = 1
    for g in range(G.order):
        for z in D[g].ravel():
            if abs(z) > 1e-9:
                for o in range(1, 400):
                    if abs(z ** o - 1) < 1e-8:
                        N = N * o // gcd(N, o); break
                else:
                    raise ValueError('no period')
    return max(N, 1)

def residue_table(G, D, kind, irreps):
    """Return (N', rows, ell0) where rows[r] describes l = r (mod N'), and ell0 the l = 0 state."""
    N = period_2d(G, D)
    Ucache = {}
    def U(v):
        if v not in Ucache:
            B = fock_basis(2, v)
            Us = [fock_matrix(D[g], B) for g in range(G.order)]
            Ucache[v] = (B, Us, projectors(Us, irreps))
        return Ucache[v]
    def ket_index(B, ell, v):   # tuple (n-, n+) with n+ - n- = ell
        return B.index(((v - ell) // 2, (v + ell) // 2))
    def analyse(v, ell):
        """SALCs inside span{|l>, |-l>} (or |l> alone) at given v: [(label, [[c(+l), c(-l)] per partner])]."""
        B, Us, Pc = U(v)
        cand = []
        for e in ([ell, -ell] if (kind == '2d' and ell != 0) else [ell]):
            c = np.zeros(len(B), complex); c[ket_index(B, e, v)] = 1; cand.append(c)
        S = salcs(Us, irreps, cand, Pc=Pc)
        out = []
        for lab, copies in S:
            for cp in copies:
                coeffs = []
                for w in cp:
                    cpl = w[ket_index(B, ell, v)]
                    cmi = w[ket_index(B, -ell, v)] if (kind == '2d' and ell != 0) else 0
                    sup = {ket_index(B, ell, v)} | ({ket_index(B, -ell, v)} if (kind == '2d' and ell != 0) else set())
                    assert all(abs(w[i]) < 1e-8 for i in range(len(B)) if i not in sup), 'leaks out of +-l'
                    coeffs.append([cpl, cmi])
                out.append((lab, coeffs))
        return sorted(out, key=lambda t: (t[0], [abs(x[0]) for x in t[1]]))
    rows = {}
    rng = range(N) if kind == '2d' else range(N)
    for r in rng:
        ell = r if r > 0 else N
        if kind == 'pair':
            ell = r if r > 0 else N           # representative; negative l handled by r (python mod)
        rows[r] = analyse(ell, ell)
    ell0 = analyse(0, 0)
    # verification: every v up to the cutoff, every l in each class, against the projector
    vmax = 2 * N + 2
    for v in range(0, vmax + 1):
        B, Us, _ = U(v)
        sets = []
        for ell in range(v, -v - 1, -2):
            if kind == '2d' and ell < 0:
                continue
            if ell == 0:
                ref = ell0
            else:
                ref = rows[ell % N]
            # build formula vectors
            got = analyse(v, ell)
            def key(t):
                return [(l, [[complex(round(a.real, 7), round(a.imag, 7)), complex(round(b.real, 7), round(b.imag, 7))] for a, b in cs]) for l, cs in t]
            assert key(got) == key(ref), (G.name, v, ell)
            for lab, coeffs in ref:
                vecs = []
                for cpl, cmi in coeffs:
                    w = np.zeros(len(B), complex)
                    w[ket_index(B, ell, v)] += cpl
                    if kind == '2d' and ell != 0:
                        w[ket_index(B, -ell, v)] += cmi
                    vecs.append(w)
                sets.append((lab, [vecs]))
        check_set(Us, dict(irreps), sets, complete_dim=len(B))
    return N, rows, ell0

# ----------------------------------------------------------------------------------------
# Rule (3): harmonic tables for vector-like three-dimensional modes
# ----------------------------------------------------------------------------------------

def harmonic_tables(G, D, irreps):
    """G-adapted harmonic SALCs for L = 0..L_HARM, checked; and the rule for v <= L_HARM + 4."""
    harm = {}
    Ucache = {}
    def U(v):
        if v not in Ucache:
            B = fock_basis(3, v)
            Ucache[v] = (B, [fock_matrix(D[g], B) for g in range(G.order)])
        return Ucache[v]
    for L in range(0, L_HARM + 1):
        B, Us = U(L)
        if L >= 2:
            C = pair_op(3, L, create=False)
            Pi = np.eye(len(B)) - np.linalg.pinv(C) @ C
        else:
            Pi = np.eye(len(B))
        cand = [np.eye(len(B))[i] for i in range(len(B))]
        S = salcs(Us, irreps, cand, restrict=Pi)
        check_set(Us, dict(irreps), S)
        assert sum(len(cp) for _, copies in S for cp in copies) == 2 * L + 1, (G.name, L)
        harm[L] = S
    # the rule: SALC(v) = N (a+.a+)^k psi_L, with ||(a+.a+)^k psi_L||^2 = 4^k k! (L+3/2)_k
    for v in range(0, L_HARM + 1):
        B, Us = U(v)
        sets = []
        for L in range(v % 2, v + 1, 2):
            k = (v - L) // 2
            for lab, copies in harm[L]:
                new = []
                for cp in copies:
                    vs = []
                    for w in cp:
                        x = w
                        for j in range(k):
                            x = pair_op(3, L + 2 * j) @ x
                        nrm2 = 4 ** k * factorial(k) * np.prod([L + 1.5 + j for j in range(k)])
                        assert abs(np.linalg.norm(x) ** 2 - nrm2) < 1e-6 * max(1, nrm2)
                        vs.append(x / np.sqrt(nrm2))
                    new.append(vs)
                sets.append((lab, new))
        check_set(Us, dict(irreps), sets, complete_dim=len(B))
    return harm

def harmonic_branching(G, chi):
    """Species of H_L = Sym^L - Sym^(L-2), L = 0..L_BRANCH."""
    H = sym_char(G, chi, L_BRANCH)
    out = []
    for L in range(L_BRANCH + 1):
        out.append(mults(G, H[L] - (H[L - 2] if L >= 2 else 0)))
    return out

# ----------------------------------------------------------------------------------------
# Standard bases for the 4- and 5-dimensional icosahedral irreps
# ----------------------------------------------------------------------------------------

def icosa_bases(G):
    """Replace the G and H irrep matrices of I/Ih by ones in a polynomial basis:
    H ~ quadratic harmonics, G ~ the G-part of the cubic harmonics (orthonormal in the Fock
    inner product, chosen by projecting occupation states in order). Returns {label: (D, basis_text)}."""
    vec = np.array([E for E in G.elems])
    par = np.array([round(np.linalg.det(E)) for E in G.elems])
    out = {}
    for v, target in ((2, 'H'), (3, 'G')):
        B = fock_basis(3, v)
        Us = [fock_matrix(vec[g] * par[g], B) for g in range(G.order)]   # proper part only
        for a, lab in enumerate(G.labels):
            if not lab.startswith(target):
                continue
            chi = G.chars[a]
            d = G.dims[a]
            inv = [g for g in range(G.order) if np.allclose(G.elems[g], -np.eye(3))]
            sign = 1 if (not inv or chi[inv[0]].real > 0) else -1   # g-type: chi(-1) = +d
            # project onto the irrep with the proper-part action (characters on proper elements)
            P = sum(np.conj(chi[g]) * Us[g] * (par[g] if sign < 0 else 1) for g in range(G.order)) * d / G.order
            cols = []
            for i in range(len(B)):
                w = P[:, i].copy()
                for q in cols:
                    w = w - q * (q.conj() @ w)
                if np.linalg.norm(w) > 1e-7:
                    cols.append(phase_fix(w / np.linalg.norm(w)))
            assert len(cols) == d, (lab, len(cols))
            Q = np.column_stack(cols)
            Dn = np.array([Q.conj().T @ (Us[g] * (par[g] if sign < 0 else 1)) @ Q for g in range(G.order)])
            for g in range(G.order):
                assert np.allclose(Dn[g] @ Dn[g].conj().T, np.eye(d), atol=1e-8)
            assert np.allclose([np.trace(x) for x in Dn], chi, atol=1e-8), lab
            for g in range(0, G.order, 7):
                for h in range(G.order):
                    assert np.allclose(Dn[g] @ Dn[h], Dn[G.mul[g, h]], atol=1e-8)
            text = [state_tex(Q[:, mu], B, '3d').replace(r'\ket', r'\kett') for mu in range(d)]
            out[lab] = (Dn, text)
    return out

# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def group_irreps(G, override=None):
    irr = []
    for lab, D in zip(G.labels, G.D):
        if override and lab in override:
            D = override[lab][0]
        irr.append((lab, D))
    return irr

def explicit_sets(G, D, irreps, vmax):
    d = D.shape[1]
    out = {}
    for v in range(1, vmax + 1):
        B = fock_basis(d, v)
        Us = [fock_matrix(D[g], B) for g in range(G.order)]
        S = salcs(Us, irreps, [np.eye(len(B))[i] for i in range(len(B))])
        check_set(Us, dict(irreps), S, complete_dim=len(B))
        out[v] = (B, S)
    return out

def mult_rows(G, D):
    chi = np.array([np.trace(D[g]) for g in range(G.order)])
    H = sym_char(G, chi, V_MULT)
    return [mults(G, H[v]) for v in range(V_MULT + 1)]

EXAMPLES = [('C3v', 'E'), ('C4v', 'E'), ('D3h', "E'"), ('D6h', 'E_{1u}'), ('Td', 'E'), ('Td', 'T_{2}'),
            ('Oh', 'T_{1u}')]

def run():
    os.chdir(HERE)
    groups = [G for G in pg.all_finite_groups(8) if not getattr(G, 'truncated', False)]
    data, tex_groups = {}, []
    stats = dict(residue=0, harmonic=0, explicit=0, pair=0, law1=0, law2=0)
    examples = {}
    comb_examples = {}
    for G in groups:
        override = icosa_bases(G) if G.name in ('I', 'Ih') else None
        irreps = group_irreps(G, override)
        if override:
            G_D = {lab: D for lab, D in irreps}
        gd = {'order': G.order, 'irreps': G.labels, 'modes': []}
        sec = {'G': G, 'modes': [], 'bases': override}
        modes_all, harm_modes = [], []
        for lab, D, kind in degenerate_modes(G):
            if override and lab in override:
                D = override[lab][0]
            modes_all.append((lab, D, kind))
            entry = {'mode': lab, 'dim': D.shape[1], 'multiplicities': mult_rows(G, D)}
            if kind in ('2d', 'pair'):
                N, rows, ell0 = residue_table(G, D, kind, irreps)
                stats['pair' if kind == 'pair' else 'residue'] += 1
                entry.update(type='residue', period=N,
                             rows={str(r): [(l, [[[a.real, a.imag], [b.real, b.imag]] for a, b in cs]) for l, cs in rows[r]] for r in rows},
                             ell0=[(l, [[[a.real, a.imag]] for a, b in cs]) for l, cs in ell0])
                sec['modes'].append((lab, kind, D, dict(N=N, rows=rows, ell0=ell0)))
            elif kind == '3d' and vector_like(G, D) is not None:
                eps = vector_like(G, D)
                harm = harmonic_tables(G, D, irreps)
                chi = np.array([np.trace(D[g]) for g in range(G.order)])
                br = harmonic_branching(G, chi)
                stats['harmonic'] += 1
                harm_modes.append((lab, D, eps))
                entry.update(type='harmonic', eps=eps_label(G, eps), branching=br,
                             harmonics={str(L): [(l, [[list(map(lambda z: [z.real, z.imag], w)) for w in cp] for cp in copies]) for l, copies in harm[L]] for L in harm})
                sec['modes'].append((lab, 'harm', D, dict(eps=eps_label(G, eps), harm=harm, branching=br)))
            else:
                vmax = V_EXPLICIT[D.shape[1]]
                ex = explicit_sets(G, D, irreps, vmax)
                stats['explicit'] += 1
                entry.update(type='explicit', vmax=vmax)
                sec['modes'].append((lab, 'explicit', D, dict(sets=ex)))
            gd['modes'].append(entry)
            if (G.name, lab) in EXAMPLES:
                vmax = V_EXAMPLE[D.shape[1]]
                examples[(G.name, lab)] = (G, explicit_sets(G, D, irreps, vmax), kind if kind != '3d' else '3d')
        comb = comb_group(G, irreps, modes_all, harm_modes, stats)
        sec['comb'] = comb
        if 'law1' in comb:
            L1 = comb['law1']
            gd['combination_law1'] = {'P': L1['P'], 'N': L1['N'], 'Q': L1['Q'],
                                      'k': {lab: k for lab, k, q in L1['ks']},
                                      'parity': {lab: q for lab, k, q in L1['ks']},
                                      'table': {'%d,%d' % key: [(l, [[complex(a).real, complex(a).imag, complex(b).real, complex(b).imag] for a, b in cs]) for l, cs in ent]
                                                for key, ent in L1['table'].items() if ent}}
        if 'grid' in comb:
            gd['combination_grid'] = [{'modes': [l1, l2], 'rows': {'%d,%d' % ab: r for ab, r in rows}} for l1, l2, rows in comb['grid']]
        for gname, labs, vs in COMB_EXAMPLES:
            if gname == G.name:
                Ds = [[D for l, D, k in modes_all if l == lab][0] for lab in labs]
                comb_examples[(gname, labs, vs)] = (G, Ds, comb_example(G, irreps, Ds, vs))
        data[G.name] = gd
        tex_groups.append(sec)
        print('%-6s modes: %s' % (G.name, ', '.join(m['mode'] + ':' + m['type'] for m in gd['modes'])) if gd['modes'] else '%-6s (no degenerate modes)' % G.name)
    return groups, data, tex_groups, (examples, comb_examples), stats

def check_gtpack(groups, path='multiplicities_gtpack.json'):
    if not os.path.exists(path):
        return None
    gt = json.load(open(path))
    byname = {G.name: G for G in groups}
    for name, d in gt.items():
        G = byname[name]
        mine = []
        for lab, D, kind in degenerate_modes(G):
            rows = np.array(mult_rows(G, D))
            mine.append(tuple(sorted((G.dims[i], tuple(rows[:, i])) for i in range(len(G.dims)))))
        theirs = [tuple(sorted((d['dims'][i], tuple(np.array(m['coeffs'])[:, i])) for i in range(len(d['dims']))))
                  for m in d['modes']]
        assert sorted(mine) == sorted(theirs), name
    return len(gt)

# ----------------------------------------------------------------------------------------
# Combination bands: quanta in several distinct modes
# ----------------------------------------------------------------------------------------

V_COMB = 4                 # law-1 / law-2 verification grid, v_s = 1..V_COMB
V_GRID = {3: 3, 4: 2, 5: 2}   # multiplicity grids for cubic/icosahedral mode pairs, by max mode dim

def product_space(G, modes, vs):
    """Fock matrices on (x)_s Sym^{v_s}; modes = [D_s]. Returns (bases, U)."""
    bases = [fock_basis(D.shape[1], v) for D, v in zip(modes, vs)]
    mats = []
    for D, v, B in zip(modes, vs, bases):
        key = (id(D), v)
        if key not in _FCACHE:
            _FCACHE[key] = (D, [fock_matrix(D[g], B) for g in range(G.order)])
        mats.append(_FCACHE[key][1])
    U = []
    for g in range(G.order):
        M = np.ones((1, 1), complex)
        for m in mats:
            M = np.kron(M, m[g])
        U.append(M)
    return bases, U

_FCACHE = {}

def product_index(bases, kets):
    i = 0
    for B, k in zip(bases, kets):
        i = i * len(B) + B.index(k)
    return i

def two_d_structure(G, modes2):
    """For a list of 2D (or pair) modes: the diagonal subgroup H (elements diagonal on every mode),
    a generator P of maximal order in H, a parity element Q in H outside <P> (or None) such that
    <P, Q> = H, and a plain-swap element w outside H (or None)."""
    Hs = [g for g in range(G.order) if all(abs(D[g][0, 1]) < 1e-9 and abs(D[g][1, 0]) < 1e-9 for D in modes2)]
    def order(g):
        x, o = g, 1
        while x != 0:
            x = G.mul[x, g]; o += 1
        return o
    def gen(gs):
        S = {0}
        frontier = [0]
        while frontier:
            new = []
            for a in frontier:
                for g in gs:
                    b = G.mul[a, g]
                    if b not in S:
                        S.add(b); new.append(b)
            frontier = new
        return S
    P = max(Hs, key=lambda g: (order(g), -g))
    Q = None
    if gen([P]) != set(Hs):
        for q in Hs:
            if order(q) == 2 and gen([P, q]) == set(Hs):
                Q = q; break
        assert Q is not None, ('H not generated by P and an involution', G.name)
    w = None
    for g in range(G.order):
        if g not in Hs and all(np.allclose(D[g], [[0, 1], [1, 0]], atol=1e-9) for D in modes2):
            w = g; break
    if len(Hs) < G.order:
        assert w is not None, ('no plain swap', G.name)
    return Hs, P, order(P), Q, w

def exponent(z, N):
    """z = exp(2 pi i k / N) -> k mod N"""
    k = int(round(np.angle(z) / (2 * np.pi) * N)) % N
    assert abs(np.exp(2j * np.pi * k / N) - z) < 1e-8
    return k

def law1_key(modes2, P, N, Q, ells):
    """(l_tot mod N, parity bit) for the product state |l_1, l_2, ...>; the phase under P is
    exp(-2 pi i l_tot / N) with l_tot = sum_s k_s l_s, k_s the exponent of D_s(P)[0,0]."""
    lt = sum(exponent(D[P][0, 0], N) * l for D, l in zip(modes2, ells)) % N
    b = 0
    if Q is not None:
        b = sum((l % 2) for D, l in zip(modes2, ells) if D[Q][0, 0].real < 0) % 2
    return lt, b

def law1_table(G, irreps, P, N, Q, w):
    """For every key (t, b): the species of |l> and of the pair {|l>, |-l>}, from characters alone.
    Returns {(t, b): [(label, [[c(+l), c(-l)] per partner])]} for the pair, and the label of a
    self-conjugate single state."""
    table = {}
    for t in range(N):
        for b in ((0, 1) if Q is not None else (0,)):
            chiP = np.exp(-2j * np.pi * t / N)
            chiQ = (-1) ** b
            conj_same = abs(chiP - np.conj(chiP)) < 1e-9
            entries = []
            for lab, DG in irreps:
                d = DG.shape[1]
                if w is None:          # abelian on these modes: 1-d irreps only
                    if d == 1 and abs(DG[P][0, 0] - chiP) < 1e-8 and (Q is None or abs(DG[Q][0, 0] - chiQ) < 1e-8):
                        entries.append((lab, [[1, 0]]))
                    continue
                if conj_same and d == 1:
                    if abs(DG[P][0, 0] - chiP) < 1e-8 and (Q is None or abs(DG[Q][0, 0] - chiQ) < 1e-8):
                        sgn = DG[w][0, 0].real            # +1: (|l> + |-l>)/sqrt2
                        entries.append((lab, [[1 / np.sqrt(2), sgn / np.sqrt(2)]]))
                if not conj_same and d == 2 and abs(DG[P][0, 1]) < 1e-9:
                    q_ok = Q is None or abs(DG[Q][0, 0] - chiQ) < 1e-8
                    if q_ok and abs(DG[P][0, 0] - chiP) < 1e-8:
                        entries.append((lab, [[1, 0], [0, 1]]))      # partners |l>, |-l>
                    elif q_ok and abs(DG[P][1, 1] - chiP) < 1e-8:
                        entries.append((lab, [[0, 1], [1, 0]]))      # partners |-l>, |l>
            table[(t, b)] = entries
    return table

def ell_of(k):            # 2-mode ket (n_1E, n_2E) = (n-, n+)
    return k[1] - k[0]

def flip(k):
    return (k[1], k[0])

def check_law1(G, irreps, modes2, vs_list):
    """Build the law-1 states for each excitation pattern and verify them; also verify that
    the projector construction gives states supported on {|l>, |-l>}."""
    Hs, P, N, Q, w = two_d_structure(G, modes2)
    table = law1_table(G, irreps, P, N, Q, w)
    Dl = dict(irreps)
    count = 0
    for vs in vs_list:
        bases, U = product_space(G, modes2, vs)
        dim = int(np.prod([len(B) for B in bases]))
        sets, done = [], set()
        for kets in itertools.product(*bases):
            if kets in done:
                continue
            ells = [ell_of(k) for k in kets]
            partner = tuple(flip(k) for k in kets)
            single = (partner == kets) or w is None
            done.add(kets)
            if not single:
                done.add(partner)
            key = law1_key(modes2, P, N, Q, ells)
            e1 = np.zeros(dim, complex); e1[product_index(bases, kets)] = 1
            e2 = np.zeros(dim, complex); e2[product_index(bases, partner)] = 1
            entries = table[key]
            if single and w is not None:
                # self-conjugate state (all l_s = 0): the symmetric species of the key
                lab = [l for l, cs in entries if len(cs) == 1 and cs[0][1] > 0][0]
                sets.append((lab, [[e1]]))
            elif w is None:
                assert len(entries) == 1, (G.name, key)
                sets.append((entries[0][0], [[e1]]))
            else:
                for lab, cs in entries:
                    sets.append((lab, [[c[0] * e1 + c[1] * e2 for c in cs]]))
        check_set(U, Dl, sets, complete_dim=dim)
        count += 1
    return (P, N, Q, w, table), count

def law2_prediction(G, eps1, eps2, a, b):
    """Characters of Sym^a (x) Sym^b for vector-like modes D_s = eps_s R, from layer coupling."""
    out = np.zeros(G.order, complex)
    dets = np.array([round(np.linalg.det(G.elems[g])) for g in range(G.order)])
    def chiL(L):
        res = []
        for g in range(G.order):
            R = G.elems[g] * dets[g]
            th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            res.append(2 * L + 1 if th < 1e-9 else np.sin((L + .5) * th) / np.sin(th / 2))
        return np.array(res)
    cache = {}
    for L1 in range(a % 2, a + 1, 2):
        for L2 in range(b % 2, b + 1, 2):
            for L in range(abs(L1 - L2), L1 + L2 + 1):
                if L not in cache:
                    cache[L] = chiL(L)
                out += cache[L] * dets ** (L1 + L2) * eps1 ** L1 * eps2 ** L2
    return out

def check_law2(G, harm_modes):
    count = 0
    for (l1, D1, e1), (l2, D2, e2) in itertools.combinations_with_replacement(harm_modes, 2):
        c1 = np.array([np.trace(D1[g]) for g in range(G.order)])
        c2 = np.array([np.trace(D2[g]) for g in range(G.order)])
        H1, H2 = sym_char(G, c1, V_COMB), sym_char(G, c2, V_COMB)
        for a in range(1, V_COMB + 1):
            for b in range(1, V_COMB + 1):
                assert mults(G, H1[a] * H2[b]) == mults(G, law2_prediction(G, e1, e2, a, b)), (G.name, l1, l2, a, b)
                count += 1
    return count

def comb_grid(G, modes):
    """Multiplicity grids for every unordered pair of degenerate modes (distinct modes)."""
    out = []
    for (l1, D1), (l2, D2) in itertools.combinations_with_replacement(modes, 2):
        vmax = V_GRID[max(D1.shape[1], D2.shape[1], 3)]
        c1 = np.array([np.trace(D1[g]) for g in range(G.order)])
        c2 = np.array([np.trace(D2[g]) for g in range(G.order)])
        H1, H2 = sym_char(G, c1, vmax), sym_char(G, c2, vmax)
        rows = [((a, b), mults(G, H1[a] * H2[b])) for a in range(1, vmax + 1) for b in range(1, vmax + 1)]
        out.append((l1, l2, rows))
    return out

COMB_EXAMPLES = [('C3v', ('E', 'E'), (6, 3)), ('C3v', ('E', 'E'), (2, 2)), ('D6h', ('E_{1u}', 'E_{2g}'), (2, 2)),
                 ('Td', ('E', 'T_{2}'), (1, 1)), ('Td', ('T_{2}', 'T_{2}'), (2, 1)), ('Oh', ('T_{1u}', 'T_{2g}'), (1, 1))]

def comb_example(G, irreps, Ds, vs):
    bases, U = product_space(G, Ds, vs)
    dim = len(U[0])
    S = salcs(U, irreps, [np.eye(dim)[i] for i in range(dim)])
    check_set(U, dict(irreps), S, complete_dim=dim)
    return bases, S

def product_ket_tex(bases, i, kinds):
    parts = []
    for B, kind in zip(reversed(bases), reversed(kinds)):
        i, r = divmod(i, len(B))
        parts.append(ket_tex(B[r], kind))
    return ''.join(reversed(parts))

def comb_state_tex(vec, bases, kinds):
    terms = []
    for i, c in enumerate(vec):
        if abs(c) > 1e-7:
            s, b = coeff_tex(c)
            terms.append((s, b + product_ket_tex(bases, i, kinds)))
    out = ''
    for j, (s, b) in enumerate(terms):
        out += ('' if (j == 0 and s == '+') else s) + b
    return out


def elem_name(G, g):
    R = G.elems[g]
    d = round(np.linalg.det(R))
    if np.allclose(R, -np.eye(3)):
        return 'i'
    x, o = g, 1
    while x != 0:
        x = G.mul[x, g]; o += 1
    if d > 0:
        return 'C_{%d}' % o
    if o == 2:
        return r'\sigma'
    return 'S_{%d}' % o

def comb_group(G, irreps, modes_all, harm_modes, stats):
    """Combination-band data for one group, with all checks."""
    out = {}
    m2 = [(lab, D) for lab, D, kind in modes_all if kind in ('2d', 'pair')]
    if m2:
        Ds = [D for _, D in m2]
        n_checks = 0
        struct = None
        for i, j in itertools.combinations_with_replacement(range(len(Ds)), 2):
            for v1 in range(1, V_COMB + 1):
                for v2 in range(1, V_COMB + 1):
                    struct, c = check_law1(G, irreps, [Ds[i], Ds[j]], [(v1, v2)])
                    n_checks += c
        for i, j, k in itertools.combinations_with_replacement(range(len(Ds)), 3):
            for vs in ((1, 1, 1), (2, 1, 1), (1, 2, 3)):
                struct, c = check_law1(G, irreps, [Ds[i], Ds[j], Ds[k]], [vs])
                n_checks += c
        P, N, Q, w, table = struct
        ks = []
        for lab, D in m2:
            k = exponent(D[P][0, 0], N)
            q = None
            if Q is not None:
                assert abs(abs(D[Q][0, 0].real) - 1) < 1e-9 and abs(D[Q][0, 0] - D[Q][1, 1]) < 1e-9
                q = int(D[Q][0, 0].real < 0)
            ks.append((lab, k, q))
        out['law1'] = dict(P=elem_name(G, P), N=N, Q=(elem_name(G, Q) if Q is not None else None),
                           swap=(w is not None), ks=ks, table=table)
        stats['law1'] += n_checks
    if harm_modes:
        stats['law2'] += check_law2(G, harm_modes)
    if any(D.shape[1] >= 3 for _, D, _ in modes_all):
        out['grid'] = comb_grid(G, [(lab, D) for lab, D, _ in modes_all])
    return out

# ----------------------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------------------

def pm_formula(cpl, cmi, kind):
    """state on |n+,n-> (coefficient cpl, l) and |n-,n+> (coefficient cmi, -l)"""
    v = np.array([cpl, cmi])
    terms = []
    for c, k in ((cpl, r'\ket{n_+,n_-}'), (cmi, r'\ket{n_-,n_+}')):
        if abs(c) > 1e-7:
            s, b = coeff_tex(c)
            terms.append((s, b + k))
    out = ''
    for i, (s, b) in enumerate(terms):
        out += ('' if (i == 0 and s == '+') else s) + b
    return out

def tex_residue(lab, kind, info):
    N, rows, ell0 = info['N'], info['rows'], info['ell0']
    L = []
    if kind == 'pair':
        L.append(r'\emph{Separably degenerate mode $%s$.} Every occupation state $\ket{n_+,n_-}$ is a SALC; its species depends only on $\ell\bmod %d$ (all integers $\ell$):' % (lab, N))
        L.append(r'\begin{longtable}{c l}\hline $\ell\bmod %d$ & species of $\ket{n_+,n_-}$\\ \hline\endhead' % N)
        for r in range(N):
            L.append(r'%d & $%s$\\' % (r, rows[r][0][0]))
        L.append(r'\hline\end{longtable}')
        return '\n'.join(L)
    L.append(r'\emph{Mode $%s$, period $N^{\prime}=%d$.} $\ket{n_{\!+}\!=\!n_{\!-}}$ ($\ell=0$, even $v$): $%s$. For $\ell>0$, with $n_\pm=(v\pm\ell)/2$:' % (lab, N, ell0[0][0]))
    L.append(r'\begin{longtable}{c l l}\hline $\ell\bmod %d$ & species & SALC (partners)\\ \hline\endhead' % N)
    for r in range(N):
        first = True
        for l, coeffs in rows[r]:
            parts = [pm_formula(a, b, kind) for a, b in coeffs]
            L.append(r'%s & $%s$ & $%s$\\' % (str(r) if first else '', l, r'$,\quad $'.join(parts)))
            first = False
    L.append(r'\hline\end{longtable}')
    return '\n'.join(L)

def tex_harmonic(G, lab, info):
    eps, harm, br = info['eps'], info['harm'], info['branching']
    L = [r'\emph{Mode $%s$, vector-like: $D(R)=\varepsilon(R)\,R$ with $\varepsilon=%s$.} Every SALC is $\mathcal N_{v,L}\,(a^\dagger\!\cdot a^\dagger)^{(v-L)/2}\psi_L$ with $\psi_L$ a harmonic SALC below and $\mathcal N_{v,L}^{-2}=4^kk!\,(L+\tfrac32)_k$, $k=(v-L)/2$. Species of the harmonic layer $\mathcal H_L$:' % (lab, eps)]
    L.append(r'\begin{longtable}{c l}\hline $L$ & $\mathcal H_L$\\ \hline\endhead')
    for Lv, row in enumerate(br):
        parts = [('%d\\,' % c if c > 1 else '') + l for l, c in zip(G.labels, row) if c]
        L.append(r'%d & $%s$\\' % (Lv, r'\oplus '.join(parts)))
    L.append(r'\hline\end{longtable}')
    L.append(r'Harmonic SALCs $\psi_L$, $L\le%d$, in $\ket{n_xn_yn_z}$:' % L_HARM)
    L.append(r'\begin{longtable}{c l c p{0.62\textwidth}}\hline $L$ & species & partner & $\psi_L$\\ \hline\endhead')
    for Lv in sorted(harm):
        B = fock_basis(3, Lv)
        for l, copies in harm[Lv]:
            for ci, cp in enumerate(copies):
                for mu, w in enumerate(cp):
                    tag = l + (r'^{(%d)}' % (ci + 1) if len(copies) > 1 else '')
                    L.append(r'%d & $%s$ & %d & $%s$\\' % (Lv, tag, mu + 1, state_tex(w, B, '3d')))
        L.append(r'\hline')
    L.append(r'\end{longtable}')
    return '\n'.join(L)

def tex_explicit(G, lab, D, info, title):
    L = [title]
    L.append(r'\begin{longtable}{c l c p{0.62\textwidth}}\hline $v$ & species & partner & SALC\\ \hline\endhead')
    kind = '2d' if D.shape[1] == 2 else '3d'
    for v, (B, S) in info['sets'].items():
        for l, copies in S:
            for ci, cp in enumerate(copies):
                for mu, w in enumerate(cp):
                    tag = l + (r'^{(%d)}' % (ci + 1) if len(copies) > 1 else '')
                    L.append(r'%d & $%s$ & %d & $%s$\\' % (v, tag, mu + 1, state_tex(w, B, kind)))
        L.append(r'\hline')
    L.append(r'\end{longtable}')
    return '\n'.join(L)

def tex_mult(G, lab, D):
    rows = mult_rows(G, D)
    L = [r'Multiplicities $m_\Gamma(v)$, $v\le%d$:' % V_MULT, r'\begin{longtable}{c l}\hline $v$ & $\mathrm{Sym}^v$\\ \hline\endhead']
    for v in range(1, V_MULT + 1):
        parts = [('%d\\,' % c if c > 1 else '') + l for l, c in zip(G.labels, rows[v]) if c]
        L.append(r'%d & $%s$\\' % (v, r'\oplus '.join(parts)))
    L.append(r'\hline\end{longtable}')
    return '\n'.join(L)


def tex_comb_examples(comb_examples):
    L = [r'\section{Worked combination bands}\label{sec:combex}',
         r'Explicit SALCs of combination bands, from the projector construction on the product of the single-mode Fock spaces (Law~0); each set is orthonormal, complete, and covariant. Kets are products, first mode first. Two equal labels denote two distinct modes of the same species (e.g.\ $\nu_3$ and $\nu_4$).']
    for (gname, labs, vs), (G, Ds, (bases, S)) in comb_examples.items():
        kinds = ['2d' if D.shape[1] == 2 else '3d' for D in Ds]
        head = r'\otimes '.join(r'%s^{%d}' % (l, v) for l, v in zip(labs, vs))
        tot = []
        for lab, copies in S:
            tot.append(('%d\\,' % len(copies) if len(copies) > 1 else '') + lab)
        L.append(r'\subsection*{$%s$: $%s=%s$}' % (G.latex, head, r'\oplus '.join(tot)))
        L.append(r'\addcontentsline{toc}{subsection}{$%s$: $%s$}' % (G.latex, head))
        L.append(r'\begin{longtable}{l c p{0.7\textwidth}}\hline species & partner & SALC\\ \hline\endhead')
        for lab, copies in S:
            for ci, cp in enumerate(copies):
                for mu, w in enumerate(cp):
                    tag = lab + (r'^{(%d)}' % (ci + 1) if len(copies) > 1 else '')
                    L.append(r'$%s$ & %d & $%s$\\' % (tag, mu + 1, comb_state_tex(w, bases, kinds)))
        L.append(r'\hline\end{longtable}')
    return '\n\n'.join(L)

def key_state(cs):
    """[c(+l), c(-l)] -> text in |l>, |-l>"""
    terms = []
    for c, k in ((cs[0], r'\ket{\boldsymbol\ell}'), (cs[1], r'\ket{-\boldsymbol\ell}')):
        if abs(c) > 1e-7:
            sg, b = coeff_tex(complex(c))
            terms.append((sg, b + k))
    out = ''
    for j, (sg, b) in enumerate(terms):
        out += ('' if (j == 0 and sg == '+') else sg) + b
    return out

def tex_comb_group(G, comb):
    if not comb:
        return ''
    L = [r'\paragraph{Combination bands.}']
    if 'law1' in comb:
        d = comb['law1']
        ks = ', '.join(r'$k(%s)=%d$' % (lab, k) + (r', $q(%s)=%d$' % (lab, q) if q is not None else '') for lab, k, q in d['ks'])
        L.append(r'Law~1 with $P=%s$ ($N=%d$)%s: %s. A product state $\ket{\boldsymbol\ell}=\bigotimes_s\ket{n_+^{(s)},n_-^{(s)}}$ has $\ell_{\rm tot}=\sum_sk_s\ell_s\bmod %d$%s.' % (
            d['P'], d['N'], (r', $Q=%s$' % d['Q']) if d['Q'] else '', ks, d['N'],
            r' and parity $p=\sum_sq_s\ell_s\bmod2$' if d['Q'] else ''))
        if d['swap']:
            L.append(r'Its species, and the SALCs on $\{\ket{\boldsymbol\ell},\ket{-\boldsymbol\ell}\}$ ($\ket{-\boldsymbol\ell}$: all $n_\pm$ exchanged), are:')
        else:
            L.append(r'Every product state is a SALC, of species:')
        cols = 'c c l l' if d['Q'] else 'c l l'
        hdr = (r'$\ell_{\rm tot}$ & $p$ & species & SALC (partners)' if d['Q'] else r'$\ell_{\rm tot}$ & species & SALC (partners)')
        L.append(r'\begin{longtable}{%s}\hline %s\\ \hline\endhead' % (cols, hdr))
        for (t, b), ent in sorted(d['table'].items()):
            first = True
            for lab, cs in ent:
                state = r'$,\quad $'.join(key_state(c) for c in cs) if d['swap'] else r'\ket{\boldsymbol\ell}'
                keycells = ('%d & %d' % (t, b) if d['Q'] else '%d' % t) if first else ('&' if d['Q'] else '')
                L.append(r'%s & $%s$ & $%s$\\' % (keycells, lab, state))
                first = False
        L.append(r'\hline\end{longtable}')
        if d['swap']:
            L.append(r'If all $\ell_s=0$ the state is single and has the symmetric species of the row $\ell_{\rm tot}=0$%s.' % (', $p=0$' if d['Q'] else ''))
    if 'grid' in comb:
        L.append(r'Multiplicities of the combination bands $\mathrm{Sym}^{v_1}\Gamma_1\otimes\mathrm{Sym}^{v_2}\Gamma_2$ of two distinct modes (Law~0; for vector-like modes they follow Law~2):')
        L.append(r'\begin{longtable}{p{0.27\textwidth} c p{0.56\textwidth}}\hline modes & $(v_1,v_2)$ & decomposition\\ \hline\endhead')
        par = lambda l: ('(' + l + ')') if 'oplus' in l else l
        for l1, l2, rows in comb['grid']:
            first = True
            for (a, b), r in rows:
                parts = [('%d\\,' % c if c > 1 else '') + l for l, c in zip(G.labels, r) if c]
                L.append(r'%s & $(%d,%d)$ & $%s$\\' % (('$%s\\otimes %s$' % (par(l1), par(l2))) if first else '', a, b, r'\oplus '.join(parts)))
                first = False
            L.append(r'\hline')
        L.append(r'\end{longtable}')
    return '\n\n'.join(L)

def write_tex(tex_groups, examples_both, stats, n_gt):
    examples, comb_examples = examples_both
    out = [PREAMBLE]
    # worked examples
    out.append(r'\section{Worked examples at finite $v$}\label{sec:examples}')
    out.append(r'Explicit SALCs of one mode for $v\le6$, from the projector construction; each set is orthonormal, complete, and transforms with the irrep matrices. They are special cases of the tables of Sec.~\ref{sec:groups}.')
    for (gname, lab), (G, sets, kind) in examples.items():
        out.append(r'\subsection*{$%s$, mode $%s$}' % (G.latex, lab))
        out.append(r'\addcontentsline{toc}{subsection}{$%s$, $%s$}' % (G.latex, lab))
        D = [m for m in degenerate_modes(G) if m[0] == lab][0][1]
        out.append(tex_explicit(G, lab, D, dict(sets=sets), ''))
    out.append(tex_comb_examples(comb_examples))
    # all groups
    out.append(r'\clearpage\section{All point groups}\label{sec:groups}')
    out.append(PART2_INTRO)
    for sec in tex_groups:
        G = sec['G']
        out.append(r'\subsection*{$%s$ \quad (order %d)}' % (G.latex, G.order))
        out.append(r'\addcontentsline{toc}{subsection}{$%s$}' % G.latex)
        if not sec['modes']:
            out.append(r'All irreps are one-dimensional and real: $\ket{n}$ of a mode $\Gamma_s$ has species $\Gamma_s^{\otimes n}$.')
            continue
        if sec['bases']:
            out.append(r'Basis of the four- and five-dimensional irreps: the mode creation operators $b^\dagger_\mu$ transform like the following states of a vector-like mode ($\kett{n_xn_yn_z}$), with the $u$ irreps differing by the parity character:')
            out.append(r'\begin{longtable}{l c p{0.7\textwidth}}\hline irrep & $\mu$ & transforms like\\ \hline\endhead')
            for l, (Dn, text) in sec['bases'].items():
                for mu, t in enumerate(text):
                    out.append(r'$%s$ & %d & $%s$\\' % (l, mu + 1, t))
            out.append(r'\hline\end{longtable}')
        for lab, kind, D, info in sec['modes']:
            if kind in ('2d', 'pair'):
                out.append(tex_residue(lab, kind, info))
            elif kind == 'harm':
                out.append(tex_harmonic(G, lab, info))
            else:
                title = (r'\emph{Mode $%s$ (no residue or harmonic factorisation).} Explicit SALCs for $v\le%d$, kets $\ket{n_1\cdots n_{%d}}$ in the basis of the mode:' % (lab, max(info['sets']), D.shape[1]))
                out.append(tex_explicit(G, lab, D, info, title))
                out.append(tex_mult(G, lab, D))
        out.append(tex_comb_group(G, sec.get('comb', {})))
    out.append(r'\end{document}')
    tex = '\n\n'.join(out)
    tex = tex.replace('@@STATS@@', r'%d two-dimensional modes, %d separably degenerate pairs, %d vector-like three-dimensional modes, and %d further modes' % (stats['residue'], stats['pair'], stats['harmonic'], stats['explicit']))
    tex = tex.replace('@@LAW1@@', str(stats['law1'])).replace('@@LAW2@@', str(stats['law2']))
    tex = tex.replace('@@GT@@', ('%d crystallographic groups' % n_gt) if n_gt else 'the crystallographic groups (when the GTPack file is present)')
    open('salc_tables.tex', 'w', encoding='utf8').write(tex)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,braket,longtable}
\usepackage[hidelinks]{hyperref}
\emergencystretch=3em
\newcommand{\kett}[1]{|#1\rangle\!\rangle}
\newtheorem{proposition}{Proposition}
\title{Symmetry-Adapted Fock States of Molecular Vibrations\\ for All Point Groups}
\author{}\date{}
\begin{document}
\maketitle
\tableofcontents
\bigskip

\section{What is tabulated}

A degenerate vibrational mode $\Gamma_s$ of dimension $d$ has creation operators $a^\dagger_1,\ldots,a^\dagger_d$, and its $v$-quantum states span $\mathrm{Sym}^v\Gamma_s$. The point group acts by Lemma~2 of the paper, $U(R)\,a^\dagger_iU(R)^\dagger=\sum_jD_{ji}(R)\,a^\dagger_j$, and the symmetry-adapted states (SALCs) $\ket{{}^\mu\Gamma;k}$ are obtained with the matrix-valued projectors $P^\Gamma_{\mu\nu}=(d_\Gamma/|\mathsf G|)\sum_RD^\Gamma_{\mu\nu}(R)^*U(R)$ (Theorem~1): copies $k$ by projecting occupation states in order, partners by $P^\Gamma_{\mu1}$. This is the construction of \texttt{salc\_fock\_basis.wl}. The groups, irrep matrices, and Mulliken labels are those of \texttt{species\_tables/pointgroups.py}, so the vibrational carriers here are in the same bases as the rotational and nuclear-spin carriers of the species tables.

\emph{Kets.} Two-dimensional modes are written $\ket{n_+,n_-}$ with vibrational angular momentum $\ell=n_+-n_-$; a ${}^1E$ quantum is counted in $n_-$ ($\ell=-1$) and a ${}^2E$ quantum in $n_+$, and $U(C_n)=e^{-2\pi i\ell/n}$. Three-dimensional modes are written $\ket{n_xn_yn_z}$ in the Cartesian basis. Coefficients are exact where they are recognised ($\omega=e^{2\pi i/3}$); the remaining icosahedral ones are given to four decimals.

\section{Three classifications}

A table for every $v$ is unnecessary, because the SALCs of all $v$ follow from a finite amount of data. Three facts reduce it.

\begin{proposition}[Parity]
The vibrational angular momentum ($\ell$ for $d=2$, $L$ for vector-like $d=3$) has the parity of $v$. An inversion or $\sigma_h$ acts on $v$ quanta of a $u$ or $''$ mode as $(-1)^v$; hence $g/u$ and $'/''$ labels alternate with $v$ for such modes and are constant for $g$ and $'$ modes.
\end{proposition}

\begin{proposition}[Residues, $d=2$]
Every group element acts on a two-dimensional mode as $s\,\mathrm{diag}(e^{i\theta},e^{-i\theta})$ or $s$ times a swap, with $s=\pm1$. Hence it maps $\ket{n_+,n_-}$ to a multiple of $\ket{n_+,n_-}$ or $\ket{n_-,n_+}$, every SALC lies in $\mathrm{span}\{\ket{n_+,n_-},\ket{n_-,n_+}\}$, and its species and coefficients depend only on $\ell$ modulo a period $N'$, the least common multiple of the orders of the nonzero matrix entries of the mode: the phase of $\ket{n_+,n_-}$ under an element is such an entry to the power $-\ell$, because $v-\ell$ is even. One table with $N'$ rows therefore gives the SALCs for every $v$.
\end{proposition}

\begin{proposition}[Harmonics, vector-like $d=3$]
If $D(R)=\varepsilon(R)\,R$ for a one-dimensional character $\varepsilon$, the invariant $a^\dagger\!\cdot a^\dagger=\sum_ia^{\dagger2}_i$ and its adjoint commute with $U(R)$, and
\[
\mathrm{Sym}^v=\bigoplus_{L=v,v-2,\ldots}(a^\dagger\!\cdot a^\dagger)^{(v-L)/2}\,\mathcal H_L,\qquad \mathcal H_L=\ker(a\cdot a)\cap\mathrm{Sym}^L,\quad\dim\mathcal H_L=2L+1 .
\]
Every SALC of $v$ quanta is $\mathcal N_{v,L}(a^\dagger\!\cdot a^\dagger)^{k}\psi_L$, $k=(v-L)/2$, with $\psi_L$ a $\mathsf G$-adapted harmonic SALC and $\mathcal N_{v,L}^{-2}=4^kk!\,(L+\tfrac32)_k$. The species of $\mathcal H_L$ are the rotational branching $D^{L}|_{\mathsf G}$ with $\mathsf O(3)$ parity $(-1)^L$, times $\varepsilon^L$: the same multiplicities as the rotational sector of the paper, and the copy label of a repeated irrep becomes the layer $L$.
\end{proposition}

The first two are elementary; the third is the isotropic-oscillator decomposition restricted to $\mathsf G$. Modes covered by neither (the $T_2$, $G$, $H$ modes of $I$ and $I_h$) are listed explicitly, the $T_2$ modes for $v\le6$ and the $G$ and $H$ modes for $v\le3$ (their $v=6$ manifolds have 84 and 210 states with irrational coefficients), together with their multiplicities to $v=12$.

\emph{Checks.} \texttt{salc\_tables.py} stops unless all of the following hold. Each residue table is compared with the projector construction for every $v\le2N'+2$ and every $\ell$, and the resulting states are verified to be orthonormal, complete, and covariant. Each harmonic table is verified the same way for $L\le6$, the norm formula and the factorised SALCs are verified for $v\le6$, and the harmonic species are computed from $\mathrm{Sym}^L-\mathrm{Sym}^{L-2}$. Every explicit table is verified for orthonormality, completeness, and covariance. The multiplicities agree with GTPack for @@GT@@. In all, the tables cover @@STATS@@.

\section{Combination bands}

Quanta in several distinct modes form combination bands. The overtone tables determine them through four laws.

\begin{proposition}[Law 0: product]
For distinct modes $s=1,2,\ldots$ the manifold is the ordinary product $\mathcal H(v_1,v_2,\ldots)=\bigotimes_s\mathrm{Sym}^{v_s}\Gamma_s$; only quanta in the same mode are symmetrised, so two different modes of the same species are distinct factors. The multiplicities have the generating function
\[
\sum_{v_1,v_2,\ldots}m_\Gamma(v_1,v_2,\ldots)\,t_1^{v_1}t_2^{v_2}\cdots=\frac1{|\mathsf G|}\sum_{R\in\mathsf G}\frac{\chi_\Gamma(R)^*}{\prod_s\det\bigl[1-t_sD^{\Gamma_s}(R)\bigr]} .
\]
\end{proposition}

\begin{proposition}[Law 1: total vibrational angular momentum, $d=2$]
Let the two-dimensional modes of a group be acted on diagonally by the subgroup $\mathsf H$ (index $1$ or $2$), generated by a principal element $P$ of order $N$ and, where needed, a parity element $Q$, with $D^{\Gamma_s}(P)=\mathrm{diag}(e^{2\pi ik_s/N},e^{-2\pi ik_s/N})$ and $D^{\Gamma_s}(Q)=(-1)^{q_s}\mathbb 1$. The product state $\ket{\boldsymbol\ell}=\bigotimes_s\ket{n^{(s)}_+,n^{(s)}_-}$ transforms under $\mathsf H$ with the character fixed by
\[
\ell_{\rm tot}=\sum_sk_s\ell_s\pmod N,\qquad p=\sum_sq_s\ell_s\pmod 2 ,
\]
and the plain swap maps it to $\ket{-\boldsymbol\ell}$ (all $\ell_s\to-\ell_s$). Hence every combination SALC lies in $\mathrm{span}\{\ket{\boldsymbol\ell},\ket{-\boldsymbol\ell}\}$, and its species and coefficients depend only on $(\ell_{\rm tot},p)$: a two-dimensional irrep with partners $\ket{\pm\boldsymbol\ell}$ if the $\mathsf H$-character is complex, and two one-dimensional irreps $(\ket{\boldsymbol\ell}\pm\ket{-\boldsymbol\ell})/\sqrt2$ if it is real. Each group section lists $k_s$, $q_s$, and the table $(\ell_{\rm tot},p)\mapsto$ species, which covers every combination band of its two-dimensional modes, for any number of modes and quanta. For one mode it reduces to the residue tables.
\end{proposition}

\begin{proposition}[Law 2: coupling of harmonic layers, vector-like $d=3$]
For vector-like modes $D_s(R)=\varepsilon_s(R)R$,
\[
\mathrm{Sym}^{a}\Gamma_1\otimes\mathrm{Sym}^{b}\Gamma_2=\bigoplus_{L_1,L_2}\ \bigoplus_{L=|L_1-L_2|}^{L_1+L_2}D^{L}\big|_{\mathsf G}\otimes\bigl[\det^{L_1+L_2}\varepsilon_1^{L_1}\varepsilon_2^{L_2}\bigr],
\]
with $L_1=a,a-2,\ldots$ and $L_2=b,b-2,\ldots$: the harmonic layers of the two modes couple like ordinary angular momenta and the result is restricted to $\mathsf G$. The parity is that of $L_1+L_2$, not of $L$. The explicit states are $(a^\dagger\!\cdot a^\dagger)^{k_1}(b^\dagger\!\cdot b^\dagger)^{k_2}$ times $\mathsf{SO}(3)$ Clebsch--Gordan combinations of the harmonic SALCs, projected onto $\mathsf G$.
\end{proposition}

\begin{proposition}[Law 3: general]
For any other pair (a two- with a three-dimensional mode, or the $G$, $H$ modes of $I$, $I_h$) the species follow from the product table of $\mathsf G$ applied to the single-mode decompositions, and the SALCs from the point-group Clebsch--Gordan coefficients or, equivalently, from the projectors on the product space. Cubic $E$ modes factor through $S_3$, so their part of a mixed band still follows Law~1.
\end{proposition}

\emph{Checks.} Law~1 is verified in every group with two-dimensional modes, for every pair of modes (including two distinct modes of one species) and $1\le v_1,v_2\le4$, and for every triple of modes at $(v_1,v_2,v_3)=(1,1,1),(2,1,1),(1,2,3)$: the states it prescribes are orthonormal, complete, and covariant (@@LAW1@@ product spaces). Law~2 is verified for every pair of vector-like modes and $1\le a,b\le4$ against the product characters (@@LAW2@@ cases). The worked combination bands of Sec.~\ref{sec:combex}, among them $C_{3v}$ $E^6\otimes E^3=5A_1\oplus5A_2\oplus9E$, are produced and verified by the projector construction.
"""

PART2_INTRO = r"""For every group of the species tables: residue tables for two-dimensional modes (for $\ell>0$ each row gives the SALC on the pair $\ket{n_+,n_-}$, $\ket{n_-,n_+}$, partners separated by commas), single-row residue lists for separably degenerate pairs, harmonic tables for vector-like three-dimensional modes, and explicit tables otherwise. Linear molecules: for $C_{\infty v}$ and $D_{\infty h}$ the residue rule holds with $N'\to\infty$, the pair $\ell=\pm\Lambda$ forming the $\Lambda$ ($\Pi,\Delta,\Phi,\ldots$) species with partners $\ket{n_+,n_-}$, $\ket{n_-,n_+}$, and $\ell=0$ giving $\Sigma^+$; for $D_{\infty h}$ the parity is $(-1)^v$ for a $\Pi_u$ mode."""

def main():
    groups, data, tex_groups, examples, stats = run()
    n_gt = check_gtpack(groups)
    print('GTPack:', ('%d groups agree' % n_gt) if n_gt else 'skipped')
    json.dump({'conventions': 'kets |n+,n-> (2d; 1E quantum in n-) and |nx ny nz>; labels of species_tables/pointgroups.py',
               'groups': data}, open('salc_tables.json', 'w'), indent=1, default=str)
    write_tex(tex_groups, examples, stats, n_gt)
    print('stats', stats)
    print('wrote salc_tables.tex and salc_tables.json')

if __name__ == '__main__':
    main()
