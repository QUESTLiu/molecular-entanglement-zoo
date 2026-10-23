"""Universal overtone tables for molecular point groups.

For a single vibrational mode carrying a (real-irreducible) representation Gamma_s of a point
group G, this script decomposes the v-quantum manifold Sym^v(Gamma_s) into irreps of G,
    Sym^v Gamma_s = (+)_Gamma m_Gamma(v) Gamma ,
for every degenerate mode of every point group of a rigid molecule, and writes

    overtone_tables.json   machine-readable multiplicities, v = 0..VMAX
    overtone_tables.tex    standalone document (tables, rules, explicit states)

Method (independent of any group-theory package):
  * each group is generated from 3x3 orthogonal matrices;
  * its character table is computed from the class algebra (Burnside-Dixon);
  * chi_{Sym^v}(g) follows from the power sums p_k(g) = chi(g^k) through Newton's identity
    h_v = (1/v) sum_{k=1}^v p_k h_{v-k};
  * multiplicities are the usual character inner products.

Checks run on every invocation (the script stops if any fails):
  * every row satisfies sum_Gamma m_Gamma(v) d_Gamma = binom(v + d_s - 1, d_s - 1);
  * the closed-form vibrational-angular-momentum rule for E_k modes (Proposition 1 of the
    document) agrees with the direct computation for C_nv, D_n, D_nd (n even), 3 <= n <= 10;
  * T_d and O give identical tables with identical labels;
  * the explicit cubic-harmonic states of the document carry the stated species;
  * if overtone_tables_gtpack.json (written by overtone_tables_gtpack.wl) is present,
    the 32 crystallographic groups are compared label-free against GTPack.

Usage:  python overtone_tables.py      (requires numpy, sympy)
        pdflatex overtone_tables.tex
"""
import itertools, json, os, re, sys
from math import comb
import numpy as np
import sympy as sp

VMAX = 12          # largest number of quanta tabulated
VSHOW_AXIAL = 8    # rows shown in the axial decomposition tables

# ----------------------------------------------------------------------------------------
# 1. Groups as sets of 3x3 orthogonal matrices
# ----------------------------------------------------------------------------------------

def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

def Rax(axis, t):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K

I3 = np.eye(3)
INV = -I3                          # inversion
SH = np.diag([1., 1., -1.])        # sigma_h
SV = np.diag([1., -1., 1.])        # sigma_v, the xz plane
C2X = Rax([1, 0, 0], np.pi)        # dihedral C2'
PHI = (1 + 5 ** 0.5) / 2

def key(m):
    return tuple(np.round(m, 6).ravel() + 0.0)

def closure(gens):
    els = {key(I3): I3}
    frontier = [I3]
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                b = a @ g
                if key(b) not in els:
                    els[key(b)] = b
                    new.append(b)
        frontier = new
    return list(els.values())

def group(name):
    """Point group by Schoenflies name: C1, Ci, Cs, Cn, Cnv, Cnh, Dn, Dnh, Dnd, S2n, T, Td, Th, O, Oh, I, Ih."""
    C3d = Rax([1, 1, 1], 2 * np.pi / 3)
    special = {
        'C1': [I3], 'Ci': [INV], 'Cs': [SV],
        'T': [C3d, Rz(np.pi)],
        'Td': [C3d, Rz(np.pi), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.]])],
        'Th': [C3d, Rz(np.pi), INV],
        'O': [C3d, Rz(np.pi / 2)],
        'Oh': [C3d, Rz(np.pi / 2), INV],
        'I': [Rax([0, 1, PHI], 2 * np.pi / 5), C3d],
        'Ih': [Rax([0, 1, PHI], 2 * np.pi / 5), C3d, INV],
    }
    if name in special:
        return closure(special[name])
    t, n, s = re.fullmatch(r'([CDS])(\d+)([vhd]?)', name).groups()
    n = int(n)
    if t == 'C':
        g = [Rz(2 * np.pi / n)] + ([SV] if s == 'v' else []) + ([SH] if s == 'h' else [])
    elif t == 'D':
        g = [Rz(2 * np.pi / n), C2X]
        if s == 'h':
            g.append(SH)
        if s == 'd':
            g.append(Rz(np.pi / (2 * n)) @ SV @ Rz(-np.pi / (2 * n)))
    else:                                   # S_n with n even
        g = [SH @ Rz(2 * np.pi / n)]
    return closure(g)

# ----------------------------------------------------------------------------------------
# 2. Character table from the class algebra
# ----------------------------------------------------------------------------------------

def char_table(G):
    n = len(G)
    idx = {key(g): i for i, g in enumerate(G)}
    inv = [idx[key(g.T)] for g in G]
    mul = np.array([[idx[key(a @ b)] for b in G] for a in G])
    cls_of = [-1] * n
    classes = []
    for i in range(n):
        if cls_of[i] < 0:
            c = sorted({mul[mul[j][i]][inv[j]] for j in range(n)})
            for x in c:
                cls_of[x] = len(classes)
            classes.append(c)
    r = len(classes)
    # class constants c_{ijk} = #{a in C_i : a^{-1} z in C_j}, z a fixed element of C_k
    M = np.zeros((r, r, r))
    for j in range(r):
        for k in range(r):
            z = classes[k][0]
            for i in range(r):
                M[j, i, k] = sum(1 for a in classes[i] if cls_of[mul[inv[a]][z]] == j)
    # omega_i = |C_i| chi_i / d are common right eigenvectors of all M_j
    rng = np.random.default_rng(1)
    A = sum(rng.normal() * M[j] for j in range(r))
    _, V = np.linalg.eig(A)
    sizes = np.array([len(c) for c in classes])
    chars = []
    for a in range(r):
        x = (V[:, a] / V[0, a]) / sizes
        d = np.sqrt(n / np.sum(sizes * np.abs(x) ** 2))
        chars.append(np.round(d * x, 8))
    chars = np.array(chars)
    # sanity: orthonormality of the computed table
    gram = (chars * sizes) @ chars.conj().T / n
    assert np.allclose(gram, np.eye(r), atol=1e-6), 'character table not orthonormal'
    return classes, cls_of, mul, sizes, chars

def sym_chars(classes, mul, cls_of, chi, vmax):
    """Characters of Sym^v of the class function chi, for v = 0..vmax; array [v][class]."""
    out = []
    for c in classes:
        g = c[0]
        powers = [g]
        for _ in range(vmax - 1):
            powers.append(mul[powers[-1]][g])
        p = [chi[cls_of[x]] for x in powers]
        h = [1.0 + 0j]
        for v in range(1, vmax + 1):
            h.append(sum(p[k - 1] * h[v - k] for k in range(1, v + 1)) / v)
        out.append(h)
    return np.array(out).T

def multiplicities(chars, sizes, n, c):
    m = (chars.conj() * sizes * c).sum(1).real / n
    assert np.allclose(m, np.round(m), atol=1e-6)
    return [int(round(x)) for x in m]

# ----------------------------------------------------------------------------------------
# 3. Mulliken labels
# ----------------------------------------------------------------------------------------

def _ix(G, m):
    keys = [key(g) for g in G]
    return keys.index(key(m))

def labels_axial(name, G, cls_of, chars):
    """C_nv, D_n, D_nh, D_nd."""
    t, n, s = re.fullmatch(r'([CD])(\d+)([vhd]?)', name).groups()
    n = int(n)
    odd_d = (s == 'd' and n % 2 == 1)
    if s == 'd' and not odd_d:
        prin, N = _ix(G, SH @ Rz(np.pi / n)), 2 * n        # S_2n
    else:
        prin, N = _ix(G, Rz(2 * np.pi / n)), n
    sub = _ix(G, SV) if t == 'C' else _ix(G, C2X)
    labs = []
    for ch in chars:
        d = int(round(ch[0].real))
        cp = ch[cls_of[prin]].real
        if d == 1:
            lab = ('A' if cp > 0 else 'B') + ('1' if ch[cls_of[sub]].real > 0 else '2')
        else:
            k = int(round(np.arccos(np.clip(cp / 2, -1, 1)) * N / (2 * np.pi)))
            lab = 'E' if N in (3, 4) else 'E%d' % k
        if s == 'h' and n % 2 == 1:
            lab += "'" if ch[cls_of[_ix(G, SH)]].real > 0 else "''"
        if (s == 'h' and n % 2 == 0) or odd_d:
            lab += 'g' if ch[cls_of[_ix(G, INV)]].real > 0 else 'u'
        labs.append(lab)
    return labs

def labels_cubic(name, G, cls_of, chars):
    """T, Td, O, Th, Oh (standard settings: C3 along (1,1,1), C2/C4 along z)."""
    if name in ('Th', 'Oh'):
        base = labels_cubic(name[0], G, cls_of, chars)
        return [l + ('g' if ch[cls_of[_ix(G, INV)]].real > 0 else 'u') for l, ch in zip(base, chars)]
    C3 = Rax([1, 1, 1], 2 * np.pi / 3)
    other = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.]]) if name == 'Td' else Rz(np.pi / 2)
    labs = []
    for ch in chars:
        d = int(round(ch[0].real))
        if name == 'T':
            if d == 3:
                labs.append('T')
            else:
                c3 = ch[cls_of[_ix(G, C3)]]
                labs.append('A' if abs(c3 - 1) < 1e-6 else ('1E' if c3.imag > 0 else '2E'))
            continue
        o = ch[cls_of[_ix(G, other)]].real
        if d == 1:
            labs.append('A1' if o > 0 else 'A2')
        elif d == 2:
            labs.append('E')
        else:   # T1: chi(C4) = +1 in O; in Td, T2 is the vector rep (chi(sigma_d) = +1)
            labs.append('T1' if (o > 0) == (name == 'O') else 'T2')
    return labs

def labels_ico(name, G, cls_of, chars):
    c5 = _ix(G, Rax([0, 1, PHI], 2 * np.pi / 5))
    labs = []
    for ch in chars:
        d = int(round(ch[0].real))
        lab = {1: 'A', 4: 'G', 5: 'H'}.get(d) or ('T1' if ch[cls_of[c5]].real > 1 else 'T2')
        if name == 'Ih':
            lab += 'g' if ch[cls_of[_ix(G, INV)]].real > 0 else 'u'
        labs.append(lab)
    return labs

def labels(name, G, cls_of, chars):
    if name in ('I', 'Ih'):
        return labels_ico(name, G, cls_of, chars)
    if name in ('T', 'Td', 'O', 'Th', 'Oh'):
        return labels_cubic(name, G, cls_of, chars)
    return labels_axial(name, G, cls_of, chars)

# ----------------------------------------------------------------------------------------
# 4. Overtone tables
# ----------------------------------------------------------------------------------------

def degenerate_modes(chars):
    """Real-irreducible degenerate modes: irreps of dim >= 2, and complex-conjugate pairs."""
    dims = [int(round(c[0].real)) for c in chars]
    modes, used = [], set()
    for a, ch in enumerate(chars):
        if dims[a] >= 2:
            modes.append((a, None, ch))
        elif np.max(np.abs(ch.imag)) > 1e-6 and a not in used:
            b = next(j for j in range(len(chars)) if np.allclose(chars[j], ch.conj()))
            used |= {a, b}
            modes.append((a, b, ch + chars[b]))
    return modes

def compute(name, vmax=VMAX, with_labels=True):
    G = group(name)
    classes, cls_of, mul, sizes, chars = char_table(G)
    labs = labels(name, G, cls_of, chars) if with_labels else ['G%d' % i for i in range(len(chars))]
    assert len(set(labs)) == len(labs), (name, labs)
    dims = [int(round(c[0].real)) for c in chars]
    out = {}
    for a, b, ch in degenerate_modes(chars):
        S = sym_chars(classes, mul, cls_of, ch, vmax)
        rows = [multiplicities(chars, sizes, len(G), S[v]) for v in range(vmax + 1)]
        d_s = int(round(ch[0].real))
        for v, r in enumerate(rows):
            assert sum(m * d for m, d in zip(r, dims)) == comb(v + d_s - 1, d_s - 1), (name, v)
        ml = labs[a] if b is None else labs[a].lstrip('12')    # pair 1E(g/u) + 2E(g/u) -> E(g/u)
        out[ml] = rows
    return {'order': len(G), 'irreps': labs, 'dims': dims, 'modes': out}

# ----------------------------------------------------------------------------------------
# 5. Checks
# ----------------------------------------------------------------------------------------

def lrule(name, k, v):
    """Closed-form decomposition of v quanta in an E_k mode of C_nv, D_n, D_nd (n even)."""
    t, n, s = re.fullmatch(r'([CD])(\d+)([vhd]?)', name).groups()
    n = int(n)
    N = 2 * n if s == 'd' else n
    out = {}
    def add(l):
        out[l] = out.get(l, 0) + 1
    for l in range(v, -1, -2):
        if l == 0:
            add('A1'); continue
        r = (k * l) % N
        if r == 0:
            add('A1'); add('A2')
        elif 2 * r == N:
            add('B1'); add('B2')
        else:
            add('E' if N in (3, 4) else 'E%d' % min(r, N - r))
    return out

def check_lrule():
    count = 0
    for n in range(3, 11):
        for fam in ['C%dv' % n, 'D%d' % n] + (['D%dd' % n] if n % 2 == 0 else []):
            res = compute(fam)
            for ml, rows in res['modes'].items():
                k = 1 if ml == 'E' else int(ml[1:])
                for v in range(VMAX + 1):
                    got = {l: c for l, c in zip(res['irreps'], rows[v]) if c}
                    assert got == lrule(fam, k, v), (fam, ml, v, got)
                    count += 1
    return count

X, Y, Z = sp.symbols('x y z')
CUBIC_STATES = [   # (v, polynomials, species for: T2 mode of Td, T1 mode of Td/O, T mode of T)
    (2, [X**2 + Y**2 + Z**2], ('A1', 'A1', 'A')),
    (2, [2*Z**2 - X**2 - Y**2, X**2 - Y**2], ('E', 'E', '1E+2E')),
    (2, [Y*Z, Z*X, X*Y], ('T2', 'T2', 'T')),
    (3, [X*Y*Z], ('A1', 'A2', 'A')),
    (3, [X**3, Y**3, Z**3], ('T2', 'T1', 'T')),
    (3, [X*(Y**2 + Z**2), Y*(Z**2 + X**2), Z*(X**2 + Y**2)], ('T2', 'T1', 'T')),
    (3, [X*(Y**2 - Z**2), Y*(Z**2 - X**2), Z*(X**2 - Y**2)], ('T1', 'T2', 'T')),
]

def species_of(polys, name, pseudo):
    """Species spanned by polynomials in creation operators a^+_{x,y,z} of a vector (or, with
    pseudo=True, pseudovector) mode; returns '+'-joined labels."""
    G = group(name)
    classes, cls_of, mul, sizes, chars = char_table(G)
    labs = labels(name, G, cls_of, chars)
    mons = sorted({m for P in polys for m in sp.Poly(sp.expand(P), X, Y, Z).monoms()})
    def vec(P):
        d = sp.Poly(sp.expand(P), X, Y, Z).as_dict()
        return np.array([float(d.get(m, 0)) for m in mons])
    B = np.array([vec(P) for P in polys]).T
    Bp = np.linalg.pinv(B)
    chi = np.zeros(len(classes), complex)
    for c, members in enumerate(classes):
        g = G[members[0]] * (round(np.linalg.det(G[members[0]])) if pseudo else 1)
        M = sp.Matrix(np.round(g, 12)).T * sp.Matrix([X, Y, Z])
        img = np.array([vec(P.subs({X: M[0], Y: M[1], Z: M[2]}, simultaneous=True)) for P in polys]).T
        assert np.allclose(B @ (Bp @ img), img, atol=1e-8), 'span not invariant'
        chi[c] = np.trace(Bp @ img)
    m = multiplicities(chars, sizes, len(G), chi)
    return '+'.join(sorted(l for l, k in zip(labs, m) for _ in range(k)))

def check_cubic_states():
    for v, polys, (td2, td1, t) in CUBIC_STATES:
        assert species_of(polys, 'Td', False) == td2, (polys, 'Td T2')
        assert species_of(polys, 'Td', True) == td1, (polys, 'Td T1')
        assert species_of(polys, 'O', False) == td1, (polys, 'O T1')
        assert species_of(polys, 'T', False) == '+'.join(sorted(t.split('+'))), (polys, 'T')
    return len(CUBIC_STATES)

PRODUCT_RULES = [   # (G, G0, kind): G = G0 x Ci ('gu') or G0 x {E, sigma_h} ('prime')
    ('D3d', 'D3', 'gu'), ('D3h', 'D3', 'prime'), ('D4h', 'D4', 'gu'), ('D5d', 'D5', 'gu'),
    ('D5h', 'D5', 'prime'), ('D6h', 'D6', 'gu'), ('Th', 'T', 'gu'), ('Oh', 'O', 'gu'), ('Ih', 'I', 'gu')]

def check_product_rules():
    """Rules (R2) and (R3): the table of G0 x Ci (or x sigma_h) is that of G0 with the parity
    suffix of the mode raised to the power v."""
    count = 0
    for G, G0, kind in PRODUCT_RULES:
        big, small = compute(G), compute(G0)
        sfx = ('g', 'u') if kind == 'gu' else ("'", "''")
        for m0, rows0 in small['modes'].items():
            for p in (0, 1):
                rows = big['modes'][m0 + sfx[p]]
                for v in range(VMAX + 1):
                    want = {l + sfx[(p * v) % 2]: c for l, c in zip(small['irreps'], rows0[v]) if c}
                    got = {l: c for l, c in zip(big['irreps'], rows[v]) if c}
                    assert want == got, (G, m0, p, v, want, got)
                    count += 1
    return count

def check_gtpack(path='overtone_tables_gtpack.json'):
    """Label-free comparison with GTPack: for each group, the multiset over modes of the
    multiset over irreps of (d_Gamma, m_Gamma(0..VMAX)) must coincide."""
    if not os.path.exists(path):
        return None
    gt = json.load(open(path))
    for name, d in gt.items():
        G = group(name)
        classes, cls_of, mul, sizes, chars = char_table(G)
        dims = [int(round(c[0].real)) for c in chars]
        mine = []
        for a, b, ch in degenerate_modes(chars):
            S = sym_chars(classes, mul, cls_of, ch, VMAX)
            rows = np.array([multiplicities(chars, sizes, len(G), S[v]) for v in range(VMAX + 1)])
            mine.append(tuple(sorted((dims[i], tuple(rows[:, i])) for i in range(len(dims)))))
        theirs = []
        for m in d['modes']:
            rows = np.array(m['coeffs'])
            theirs.append(tuple(sorted((d['dims'][i], tuple(rows[:, i])) for i in range(len(d['dims'])))))
        assert sorted(mine) == sorted(theirs) and len(G) == d['order'], name
    return len(gt)

# ----------------------------------------------------------------------------------------
# 6. LaTeX output
# ----------------------------------------------------------------------------------------

def lab_tex(l):
    if l in ('1E', '2E'):
        return '{}^%sE' % l[0]
    primes = l.count("'")
    core = l.replace("'", '')
    s = core[0] + ('_{%s}' % core[1:] if core[1:] else '')
    return s + "'" * primes

def order_idx(labs, pref):
    return sorted(range(len(labs)), key=lambda i: pref.index(labs[i]) if labs[i] in pref else 99)

def decomp_tex(row, labs, idx):
    parts = [('%d' % row[i] if row[i] > 1 else '') + lab_tex(labs[i]) for i in idx if row[i]]
    return '$' + r'\oplus '.join(parts) + '$'

def axial_table(blocks, caption, label):
    L = [r'\begin{table}[H]', r'\centering\small',
         r'\begin{tabular}{c|' + 'l' * len(blocks) + '}', r'\hline',
         '$v$ & ' + ' & '.join(r'$%s$, $%s$ mode' % (g[0] + '_{' + g[1:] + '}', lab_tex(m)) for g, m, *_ in blocks) + r'\\',
         r'\hline']
    for v in range(1, VSHOW_AXIAL + 1):
        L.append('%d & ' % v + ' & '.join(decomp_tex(rows[v], labs, idx) for _, _, labs, idx, rows in blocks) + r'\\')
    L += [r'\hline', r'\end{tabular}', r'\caption{%s}' % caption, r'\label{%s}' % label, r'\end{table}', '']
    return '\n'.join(L)

def mult_table(blocks, caption, label):
    L = [r'\begin{table}[H]', r'\centering\small', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular}{c' + ''.join('|' + 'c' * len(idx) for _, _, idx, _ in blocks) + '}', r'\hline',
         ' & ' + ' & '.join(r'\multicolumn{%d}{c%s}{%s}' % (len(idx), '|' if k < len(blocks) - 1 else '', head)
                            for k, (head, _, idx, _) in enumerate(blocks)) + r'\\',
         '$v$ & ' + ' & '.join(' & '.join('$%s$' % lab_tex(labs[i]) for i in idx) for _, labs, idx, _ in blocks) + r'\\',
         r'\hline']
    for v in range(1, VMAX + 1):
        L.append('%d & ' % v + ' & '.join(' & '.join(str(rows[v][i]) for i in idx) for _, _, idx, rows in blocks) + r'\\')
    L += [r'\hline', r'\end{tabular}', r'\caption{%s}' % caption, r'\label{%s}' % label, r'\end{table}', '']
    return '\n'.join(L)

def block(res, mode, pref, head=None):
    labs = res['irreps']
    return (head, labs, order_idx(labs, pref), res['modes'][mode])

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # ---- checks
    n_l = check_lrule()
    print('l-rule: %d decompositions agree' % n_l)
    td, o = compute('Td'), compute('O')
    bylab = lambda res: {m: [dict(zip(res['irreps'], r)) for r in rows] for m, rows in res['modes'].items()}
    assert bylab(td) == bylab(o)
    print('Td and O tables identical')
    print('explicit cubic states: %d verified' % check_cubic_states())
    print('rules (R2), (R3): %d decompositions agree' % check_product_rules())
    n_gt = check_gtpack()
    print('GTPack comparison: ' + ('%d groups agree' % n_gt if n_gt else 'skipped (overtone_tables_gtpack.json not found)'))

    # ---- data
    names = ['C3v', 'D3', 'D3d', 'D3h', 'C4v', 'D4', 'D2d', 'D4h', 'C5v', 'D5', 'D5d', 'D5h',
             'C6v', 'D6', 'D6h', 'D4d', 'T', 'Td', 'O', 'Th', 'Oh', 'I', 'Ih']
    data = {n: compute(n) for n in names}
    json.dump({'vmax': VMAX, 'groups': data}, open('overtone_tables.json', 'w'), indent=1)

    A = lambda g, pref: [(g, m, data[g]['irreps'], order_idx(data[g]['irreps'], pref), data[g]['modes'][m])
                         for m in pref if m in data[g]['modes']]
    t_axial1 = axial_table(A('C3v', ['A1', 'A2', 'E']) + A('C4v', ['A1', 'A2', 'B1', 'B2', 'E'])
                           + A('C5v', ['A1', 'A2', 'E1', 'E2']),
                           r'Overtones of the degenerate modes of $C_{3v}$, $C_{4v}$, and $C_{5v}$, for $v\le%d$. Each column applies unchanged to the isomorphic groups with the same labels ($C_{3v}$: also $D_3$; $C_{4v}$: also $D_4$, $D_{2d}$; $C_{5v}$: also $D_5$), and through (R2) and (R3) to $D_{3d}$, $D_{3h}$, $D_{4h}$, $D_{5d}$, $D_{5h}$.' % VSHOW_AXIAL,
                           'tab:axial')
    t_axial2 = axial_table(A('C6v', ['A1', 'A2', 'B1', 'B2', 'E1', 'E2']),
                           r'Overtones of the $E_1$ and $E_2$ modes of $C_{6v}$ (also $D_6$; $D_{6h}$ by (R2)).', 'tab:axial2')
    t_axial3 = axial_table(A('D4d', ['A1', 'A2', 'B1', 'B2', 'E1', 'E2', 'E3']),
                           r'Overtones of the $E_1$, $E_2$, $E_3$ modes of $D_{4d}$ (e.g.\ cyclo-$\mathrm S_8$); the principal element is $S_8$ and $N=8$ in Proposition~\ref{prop:lrule}.',
                           'tab:axial3')
    cub = ['A1', 'A2', 'E', 'T1', 'T2']
    t_cubic = mult_table([block(data['Td'], 'T2', cub, r'$T_d$ (or $O$), $T_2$ mode'),
                          block(data['Td'], 'T1', cub, r'$T_d$ (or $O$), $T_1$ mode'),
                          block(data['T'], 'T', ['A', '1E', '2E', 'T'], r'$T$, $T$ mode')],
                         r'Multiplicities $m_\Gamma(v)$ for $v$ quanta in a triply degenerate mode of the cubic groups. The $T_2$ block covers $\nu_3$ and $\nu_4$ of $\mathrm{CH_4}$; for $O_h$ and $T_h$ apply (R2). In $T$ the complex species ${}^1E$ and ${}^2E$ always occur together.',
                         'tab:cubic')
    ico = ['A', 'T1', 'T2', 'G', 'H']
    t_ico = mult_table([block(data['I'], m, ico, '$%s$ mode' % lab_tex(m)) for m in ['T1', 'T2', 'G', 'H']],
                       r'Multiplicities $m_\Gamma(v)$ for $v$ quanta in each degenerate mode of $I$; for $I_h$ apply (R2).',
                       'tab:ico')
    checks = dict(NL=n_l, NGT=(n_gt or 0))
    tex = TEMPLATE
    for k, v in dict(T_AXIAL1=t_axial1, T_AXIAL2=t_axial2, T_AXIAL3=t_axial3, T_CUBIC=t_cubic, T_ICO=t_ico,
                     VMAX=str(VMAX), NL=str(n_l)).items():
        tex = tex.replace('@@%s@@' % k, v)
    open('overtone_tables.tex', 'w', encoding='utf8').write(tex)
    print('wrote overtone_tables.tex and .json')

TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,braket,float}
\usepackage[hidelinks]{hyperref}
\emergencystretch=3em
\newtheorem{proposition}{Proposition}
\title{Universal Overtone Tables for Molecular Point Groups}
\author{}
\date{}
\begin{document}
\maketitle

\section{Scope}

The species content of a vibrational manifold depends only on the point group and on the species of the modes involved, not on the molecule. This document tabulates it for every point group that occurs for rigid molecules: the 32 crystallographic groups, the fivefold groups $C_{5v}$, $D_5$, $D_{5h}$, $D_{5d}$, the group $D_{4d}$ of cyclo-$\mathrm S_8$, the icosahedral groups $I$ and $I_h$, and the linear groups $C_{\infty v}$ and $D_{\infty h}$. A specific molecule enters only through its list of mode species $\Gamma_{\rm vib}=\bigoplus_s\Gamma_s$, which follows from its equilibrium structure, and through the physical scales $\omega_s$ and $b_s$, which do not affect species. The entries are the multiplicities $m_\Gamma(v)$ in
\begin{equation}
\mathrm{Sym}^v\Gamma_s=\bigoplus_\Gamma m_\Gamma(v)\,\Gamma ,
\end{equation}
the decomposition of $v$ quanta in a single mode $s$; they are the multiplicities of the symmetry-adapted Fock encoding (Theorem~1 of the paper), and they have generating function
\begin{equation}
\sum_{v\ge0}m_\Gamma(v)\,t^v=\frac{1}{|\mathsf G|}\sum_{R\in\mathsf G}\frac{\chi^*_\Gamma(R)}{\det\bigl[1-t\,D^{\Gamma_s}(R)\bigr]} .
\label{eq:molien}
\end{equation}
Table~\ref{tab:index} lists, for each group, where its degenerate modes are treated. The same data, for $v\le@@VMAX@@$, are in \texttt{overtone\_tables.json}.

\paragraph{Verification.} The tables are produced by \texttt{overtone\_tables.py}, which generates each group from $3\times3$ orthogonal matrices, computes its character table from the class algebra, and obtains the characters of $\mathrm{Sym}^v\Gamma_s$ from the power sums $p_k(R)=\chi_{\Gamma_s}(R^k)$ through Newton's identity $h_v=v^{-1}\sum_{k=1}^v p_k\,h_{v-k}$. Independently, \texttt{overtone\_tables\_gtpack.wl} computes the same decompositions from the GTPack character tables of the 32 crystallographic groups; the two agree for every group, every degenerate mode, and every $v\le@@VMAX@@$. The explicit projector scripts of this folder, \texttt{overtones\_C3v\_E.wl} ($v\le8$) and \texttt{overtones\_Td\_T2.wl} ($v\le6$) reproduce the $C_{3v}$ and $T_d$ entries. The script also verifies rules (R2) and (R3) below against direct computation for $D_{3d}$, $D_{3h}$, $D_{4h}$, $D_{5d}$, $D_{5h}$, $D_{6h}$, $T_h$, $O_h$, and $I_h$. Every row satisfies $\sum_\Gamma m_\Gamma(v)\,d_\Gamma=\binom{v+d_s-1}{d_s-1}$.

\section{Reduction rules}

Four rules reduce the tables to a small core.

\emph{(R1) Nondegenerate modes.} If $d_{\Gamma_s}=1$, then $\mathrm{Sym}^v\Gamma_s=\Gamma_s^{\otimes v}$, a single one-dimensional species.

\emph{(R2) Inversion.} If $\mathsf G=\mathsf G_0\times C_i$, the inversion acts on $v$ quanta of a $\Gamma_{g}$ mode as $+1$ and of a $\Gamma_{u}$ mode as $(-1)^v$. The table of $\mathsf G_0$ therefore applies with the subscript $g$ appended, except for a $u$ mode with $v$ odd, where $u$ is appended. This covers $C_{2h}$, $D_{2h}$, $C_{4h}$, $D_{4h}$, $C_{6h}$, $D_{6h}$, $S_6$, $D_{3d}$, $D_{5d}$, $T_h$, $O_h$, $I_h$, and $D_{\infty h}$.

\emph{(R3) Horizontal plane.} If $n$ is odd and $\mathsf G=\mathsf G_0\times\{E,\sigma_h\}$, the same rule holds with $g,u$ replaced by $'$ and $''$. This covers $C_{3h}$, $D_{3h}$, $C_{5h}$, and $D_{5h}$.

\emph{(R4) Several modes.} Quanta in different modes combine by the ordinary tensor product, $\mathcal H(\mathbf v)\cong\bigotimes_s\mathrm{Sym}^{v_s}\Gamma_s$, so combination bands follow from the single-mode tables and the product table of $\mathsf G$. Two distinct modes of the same species are distinct factors; only quanta in the same mode are symmetrized.

A degenerate mode carries a representation that is irreducible over $\mathbb R$, which need not be irreducible over $\mathbb C$. In $C_n$ ($n\ge3$), $C_{nh}$, $S_{2n}$, $T$, and $T_h$, the separably degenerate pairs that character tables write as $E$ are sums ${}^1E\oplus{}^2E$ of complex-conjugate one-dimensional irreps, and the tables treat the mode as this sum.

\section{Axial groups: the vibrational angular-momentum rule}

All degenerate modes of the axial groups are two-dimensional, and for them the decomposition has a closed form valid for every $n$. Let a mode of species $E_k$ have circular creation operators $a^\dagger_\pm$, chosen so that the principal axis acts as $a^\dagger_\pm\mapsto e^{\mp2\pi ik/N}a^\dagger_\pm$ and a vertical reflection or dihedral $C_2'$ exchanges $a^\dagger_+\leftrightarrow a^\dagger_-$. Here $N=n$ for $C_{nv}$ and $D_n$, with the principal rotation $C_n$, and $N=2n$ for $D_{nd}$ with $n$ even, with the principal improper rotation $S_{2n}$. The $v+1$ states of $v$ quanta are labeled by the vibrational angular momentum $\ell\in\{v,v-2,\ldots,-v\}$,
\begin{equation}
\ket{v,\ell}=\frac{(a^\dagger_+)^{(v+\ell)/2}(a^\dagger_-)^{(v-\ell)/2}}{\sqrt{\bigl(\tfrac{v+\ell}{2}\bigr)!\,\bigl(\tfrac{v-\ell}{2}\bigr)!}}\ket{0},
\label{eq:vl}
\end{equation}
and the principal axis multiplies $\ket{v,\ell}$ by $e^{-2\pi ik\ell/N}$ (the convention of the paper and of \texttt{overtones\_C3v\_E.wl}, in which one ${}^1E$ quantum carries $\ell=-1$), while the reflection sends $\ket{v,\ell}\mapsto\ket{v,-\ell}$. The species follow at once.

\begin{proposition}[Overtones of an $E_k$ mode]
\label{prop:lrule}
For $v$ quanta in an $E_k$ mode of $C_{nv}$, $D_n$, or $D_{nd}$ ($n$ even), the state $\ket{v,0}$ (present for even $v$) is $A_1$, and each pair $\{\ket{v,\ell},\ket{v,-\ell}\}$ with $\ell>0$ carries, with $r\equiv k\ell\bmod N$,
\begin{equation}
\begin{cases}
A_1\oplus A_2, & r=0,\\
B_1\oplus B_2, & r=N/2,\\
E_{\min(r,\,N-r)}, & \text{otherwise}.
\end{cases}
\label{eq:lrule}
\end{equation}
In the first two cases the partners are $(\ket{v,\ell}\pm\ket{v,-\ell})/\sqrt2$, the upper sign giving the species with subscript $1$ (for $B$, with respect to the reflection that exchanges $a^\dagger_\pm$); in the third, $\ket{v,\pm\ell}$ are the two circular partners of $E_{\min(r,N-r)}$, for which $E_1\equiv E$ when $N\le4$.
\end{proposition}

The script checks Eq.~\eqref{eq:lrule} against the direct computation for $C_{nv}$, $D_n$, and $D_{nd}$ with $3\le n\le10$ and $v\le@@VMAX@@$, @@NL@@ decompositions in all, with no exception. Together with (R2) and (R3) it covers $D_{nh}$ and $D_{nd}$ for every $n$.

\emph{Cyclic and improper-cyclic groups.} In $C_n$, $S_{2n}$, and $C_{nh}$ there is no reflection to pair $\pm\ell$, and each $\ket{v,\ell}$ is separately a one-dimensional species: $A$ if $r=0$, $B$ if $r=N/2$, and the complex species ${}^1E_r$ or ${}^2E_r$ otherwise, with $r$ as above and $N=n$ for $C_n$, or $N=2n$ for $S_{2n}$ with $n$ even (principal axis $S_{2n}$). The remaining groups of this kind are direct products, $C_{nh}=C_n\times C_i$ ($n$ even) or $C_n\times\{E,\sigma_h\}$ ($n$ odd) and $S_{2n}=C_n\times C_i$ ($n$ odd), so the $C_n$ rule applies with labels appended by (R2) or (R3).

\emph{Linear molecules.} For $C_{\infty v}$ the rule holds with $N\to\infty$: a $\Pi$ bending mode with $v$ quanta gives $\Sigma^+$ for $\ell=0$ and the species $\Lambda=|\ell|=1,2,3,\ldots$ ($\Pi,\Delta,\Phi,\ldots$) for each pair $\pm\ell$, the familiar $\ell$-structure of the bending overtones of a linear triatomic. For $D_{\infty h}$, (R2) adds $g$ or $u$.

\emph{Molecule-independence of the states.} Equation~\eqref{eq:vl} and the partner combinations of Proposition~\ref{prop:lrule} are the symmetry-adapted states for every axial molecule. Their position-space wavefunctions are the Laguerre--Gaussian functions $\propto\rho^{|\ell|}L^{|\ell|}_{(v-|\ell|)/2}(\rho^2)\,e^{i\ell\varphi}e^{-\rho^2/2}$ of the dimensionless coordinates $(q_x,q_y)=\rho(\cos\varphi,\sin\varphi)$, and the momentum-space wavefunctions are the same functions of $(p_x,p_y)$ times $(-i)^v$.

@@T_AXIAL1@@
@@T_AXIAL2@@
@@T_AXIAL3@@
\section{Cubic groups}

The doubly degenerate $E$ mode of $T_d$ and $O$ factors through $S_3\cong C_{3v}$ and follows Table~\ref{tab:axial} with $A_1,A_2,E$ unchanged; the triply degenerate modes are in Table~\ref{tab:cubic}. In $T_d$, a $T_1$ mode follows from a $T_2$ mode by exchanging $A_1\leftrightarrow A_2$ and $T_1\leftrightarrow T_2$ for odd $v$, since $T_1=T_2\otimes A_2$ and $A_2$ acts on $v$ quanta as $A_2^{\,v}$. The map $R\mapsto\det(R)\,R$ is an isomorphism $T_d\to O$ under which every Mulliken label is preserved, so a $T_1$ or $T_2$ mode of $O$ has exactly the table of the same mode of $T_d$ (the script verifies this entry by entry); for $O_h$, e.g.\ the $T_{1u}$ stretching and bending modes of $\mathrm{SF_6}$, and for $T_h$ apply (R2).

@@T_CUBIC@@
\section{Icosahedral groups}

The icosahedral groups have four- and five-dimensional irreps. Together with $T$ and $T_h$ they are the only point groups in which a product of two irreps can contain an irrep more than once, so that the rotation--nuclear-spin--vibration coupling multiplicity of the paper can exceed one. Table~\ref{tab:ico} gives the four degenerate modes of $I$; for $I_h$, e.g.\ $\mathrm{C_{60}}$ and dodecahedrane, apply (R2).

@@T_ICO@@
\section{Explicit states of a triply degenerate mode}

For a triply degenerate mode the symmetry-adapted states of low $v$ are polynomials in the three creation operators. Write $P(a^\dagger_x,a^\dagger_y,a^\dagger_z)\ket0$ with $a^\dagger_{x,y,z}$ transforming as the Cartesian components of the mode; monomials are orthogonal, with $\|(a^\dagger_x)^{n_x}(a^\dagger_y)^{n_y}(a^\dagger_z)^{n_z}\ket0\|^2=n_x!\,n_y!\,n_z!$, which fixes the normalization of each entry of Table~\ref{tab:cubic_states}. The polynomials are cubic harmonics and are molecule independent: they are the same for $\nu_3$ and $\nu_4$ of methane, for $\nu_3$ and $\nu_4$ of $\mathrm{SF_6}$, and for any other triply degenerate mode of a cubic molecule. Only the species labels depend on the group and on the mode; the script verifies every entry by character projection.

\begin{table}[H]
\centering\small
\begin{tabular}{cllccc}
\hline
$v$ & polynomial $P$ & partners & $T_2$ mode of $T_d$ & $T_1$ mode of $T_d$ or $O$ & $T$ mode of $T$\\
\hline
2 & $x^2+y^2+z^2$ & 1 & $A_1$ & $A_1$ & $A$\\
2 & $2z^2-x^2-y^2,\ \sqrt3\,(x^2-y^2)$ & 2 & $E$ & $E$ & ${}^1E\oplus{}^2E$\\
2 & $yz,\ zx,\ xy$ & 3 & $T_2$ & $T_2$ & $T$\\
\hline
3 & $xyz$ & 1 & $A_1$ & $A_2$ & $A$\\
3 & $x^3,\ y^3,\ z^3$ & 3 & $T_2$ & $T_1$ & $T$\\
3 & $x(y^2+z^2),\ y(z^2+x^2),\ z(x^2+y^2)$ & 3 & $T_2$ & $T_1$ & $T$\\
3 & $x(y^2-z^2),\ y(z^2-x^2),\ z(x^2-y^2)$ & 3 & $T_1$ & $T_2$ & $T$\\
\hline
\end{tabular}
\caption{Symmetry-adapted states $P(a^\dagger_x,a^\dagger_y,a^\dagger_z)\ket0$ for two and three quanta in a triply degenerate mode, with $x\equiv a^\dagger_x$ etc. For $v=3$ the two copies of the same species (second and third rows) span a two-dimensional multiplicity space; any rotation within it is equally valid, and the Hamiltonian, not symmetry, selects the combination. In the complex basis of \texttt{overtones\_Td\_T2.wl} the $E$ pair is $3^{-1/2}\sum_i\omega^{\mp i}\ket{2_i}$, which spans the same two-dimensional space as the real pair listed here. For $O_h$ and $T_h$ append $g$ or $u$ by (R2).}
\label{tab:cubic_states}
\end{table}

\section{Index of point groups}

Table~\ref{tab:index} lists every point group of a rigid molecule with its degenerate mode species and the place where their overtones are given. Groups with only one-dimensional irreps ($C_1$, $C_i$, $C_s$, $C_2$, $C_{2v}$, $C_{2h}$, $D_2$, $D_{2h}$) need no table, by (R1).

\begin{table}[H]
\centering\small
\begin{tabular}{lll}
\hline
Group & degenerate modes & overtones\\
\hline
$C_{3v}$, $D_3$ & $E$ & Table~\ref{tab:axial}, Prop.~\ref{prop:lrule}\\
$D_{3d}$, $D_{3h}$ & $E_g,E_u$; $E',E''$ & as $D_3$, with (R2), (R3)\\
$C_{4v}$, $D_4$, $D_{2d}$ & $E$ & Table~\ref{tab:axial}, Prop.~\ref{prop:lrule}\\
$D_{4h}$ & $E_g,E_u$ & as $D_4$, with (R2)\\
$C_{5v}$, $D_5$ & $E_1,E_2$ & Table~\ref{tab:axial}, Prop.~\ref{prop:lrule}\\
$D_{5h}$, $D_{5d}$ & $E_1',E_2',E_1'',E_2''$; $E_{1g},E_{2g},E_{1u},E_{2u}$ & as $D_5$, with (R3), (R2)\\
$C_{6v}$, $D_6$ & $E_1,E_2$ & Table~\ref{tab:axial2}, Prop.~\ref{prop:lrule}\\
$D_{6h}$ & $E_{1g},E_{2g},E_{1u},E_{2u}$ & as $D_6$, with (R2)\\
$D_{nd}$, $n$ even (e.g.\ $D_{4d}$) & $E_1,\ldots,E_{n-1}$ & Prop.~\ref{prop:lrule} with $N=2n$; Table~\ref{tab:axial3} for $D_{4d}$\\
$C_{nv}$, $D_n$, $D_{nh}$, $D_{nd}$, any $n$ & $E_k$ & Prop.~\ref{prop:lrule}, (R2), (R3)\\
$C_n$, $S_{2n}$, $C_{nh}$ & ${}^1E_k\oplus{}^2E_k$ & cyclic form of Prop.~\ref{prop:lrule}, (R2), (R3)\\
$C_{\infty v}$, $D_{\infty h}$ & $\Pi$; $\Pi_g,\Pi_u$ & linear form of Prop.~\ref{prop:lrule}, (R2)\\
$T_d$, $O$ & $E$; $T_1$, $T_2$ & Table~\ref{tab:axial} ($C_{3v}$); Table~\ref{tab:cubic}\\
$O_h$ & $E_g,E_u$; $T_{1g},T_{2g},T_{1u},T_{2u}$ & as $O$, with (R2)\\
$T$ & ${}^1E\oplus{}^2E$; $T$ & cyclic rule for $C_3$; Table~\ref{tab:cubic}\\
$T_h$ & ${}^1E_g\oplus{}^2E_g$, ${}^1E_u\oplus{}^2E_u$; $T_g,T_u$ & as $T$, with (R2)\\
$I$ & $T_1,T_2,G,H$ & Table~\ref{tab:ico}\\
$I_h$ & $T_{1g},\ldots,H_u$ & as $I$, with (R2)\\
\hline
\end{tabular}
\caption{Where to find the overtone decomposition of every degenerate mode, for all point groups of rigid molecules.}
\label{tab:index}
\end{table}

\end{document}
"""

if __name__ == '__main__':
    main()
