# Second-order vibrational perturbation theory for the CaOCH3 nu8 manifold, done numerically
# (quasi-degenerate / Van Vleck form), so no closed-form x_tt, g_tt formulas are needed.
#   H = sum_i w_i (p_i^2 + q_i^2)/2 + (1/6) phi_ijk q_i q_j q_k + (1/24) phi_8888 q^4 + sum_a B_a pi_a^2   (J = 0)
# with dimensionless normal coordinates q_i, pi_a = sum_ij zeta^a_ij sqrt(w_j/w_i) q_i p_j (Watson 1968).
# Model space for v8 = n: all states with n quanta in the two nu8 components and none elsewhere.
#   H_eff = E_M + <V4 + H_cor> + sum_{k not in M} <m'|V3|k><k|V3|m> / (E_M - E_k)
# Only cubic terms with at least one nu8 index and quartic terms with four nu8 indices enter; the others
# shift every level of the manifold equally or linearly in n and so change only the effective omega.
# The levels are fitted to E(n, l) - E(0) = w~ n + x88 n(n-1) + g88 l^2 (Eq. caoch3_hamiltonian).
import sys, itertools
import numpy as np
from collections import defaultdict

EH2CM = 219474.6313705
W_CONV = np.sqrt(4.3597447222e-18/(1.66053906660e-27*1e-20))/(2*np.pi*2.99792458e10)

def run(npz, w_override=None, verbose=True, rich=True, sym=True):
    D = np.load(npz)
    w, lam, zeta, cub, qua, Ae, Be = D['w'], D['lam'], D['zeta'], D['cub'], D['qua'], float(D['Ae']), float(D['Be'])
    if rich and 'cub_h' in D:     # Richardson: steps d and d/2, error O(d^2)
        cub = (4*D['cub_h'] - cub)/3; qua = (4*D['qua_h'] - qua)/3
    n = len(w)
    g = lam*EH2CM/w                                   # q = Q sqrt(g)
    # full cubic tensor for terms with at least one nu8 index (0, 1); cub[c][j,k] = phi_{c j k}
    T3 = np.zeros((n, n, n))
    for i, j, k in itertools.product(range(n), repeat=3):
        idx = [x for x in (i, j, k) if x in (0, 1)]
        if not idx: continue
        c = idx[0]; rest = list((i, j, k)); rest.remove(c)
        T3[i, j, k] = cub[c][rest[0], rest[1]]
    # consistency of the overdetermined entries (phi_{0 1 k} from both displacements, phi_{0 0 1} etc.)
    sym_err = max(abs(cub[0][1, k] - cub[1][0, k]) for k in range(n))
    # C3v average of the tensors (removes the numerical anisotropy that splits e pairs)
    if sym:
        import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caoch3_eom_ccsd'))
        from caoch3_emodes import sym_ops
        Lm = D['L']; Dg = [Lm.T @ G @ Lm for G, _, _ in sym_ops()]
        T3 = sum(np.einsum('ia,jb,kc,abc->ijk', M, M, M, T3) for M in Dg)/len(Dg)
    T3 *= EH2CM/np.sqrt(np.einsum('i,j,k->ijk', g, g, g))
    T4 = np.zeros((2, 2, 2, 2))
    for a, b, c, d in itertools.product(range(2), repeat=4):
        idx = sorted((a, b, c, d))
        n1 = idx.count(1)
        val = {0: qua[0][0, 0], 1: qua[0][0, 1], 2: qua[0][1, 1], 3: qua[1][0, 1], 4: qua[1][1, 1]}[n1]
        T4[a, b, c, d] = val
    quart_check = (qua[0][1, 1], qua[1][0, 0])
    if sym:
        T4 = sum(np.einsum('ia,jb,kc,ld,abcd->ijkl', M[:2, :2], M[:2, :2], M[:2, :2], M[:2, :2], T4) for M in Dg)/len(Dg)
    T4 *= EH2CM/np.sqrt(np.einsum('i,j,k,l->ijkl', g[:2], g[:2], g[:2], g[:2]))
    if w_override is not None: w = np.array(w_override)
    B = [Be, Be, Ae]

    def add(dst, st, c):
        if abs(c) > 0: dst[st] += c
    def q_op(i, vec):
        out = defaultdict(complex)
        for st, c in vec.items():
            ni = st[i]
            if ni > 0: s = list(st); s[i] -= 1; add(out, tuple(s), c*np.sqrt(ni/2))
            s = list(st); s[i] += 1; add(out, tuple(s), c*np.sqrt((ni+1)/2))
        return out
    def p_op(i, vec):
        out = defaultdict(complex)
        for st, c in vec.items():
            ni = st[i]
            if ni > 0: s = list(st); s[i] -= 1; add(out, tuple(s), -1j*c*np.sqrt(ni/2))
            s = list(st); s[i] += 1; add(out, tuple(s), 1j*c*np.sqrt((ni+1)/2))
        return out
    def axpy(dst, vec, a):
        for st, c in vec.items(): dst[st] += a*c
    def E0(st): return float(np.sum(w*(np.array(st) + 0.5)))
    def V3(vec):
        out = defaultdict(complex)
        s1 = {k: q_op(k, vec) for k in range(n)}
        for j, k in itertools.product(range(n), repeat=2):
            if not np.any(T3[:, j, k]): continue
            s2 = q_op(j, s1[k])
            for i in range(n):
                if T3[i, j, k] != 0: axpy(out, q_op(i, s2), T3[i, j, k]/6)
        return out
    def V4(vec):
        out = defaultdict(complex)
        for a, b, c, d in itertools.product(range(2), repeat=4):
            if T4[a, b, c, d] != 0:
                axpy(out, q_op(a, q_op(b, q_op(c, q_op(d, vec)))), T4[a, b, c, d]/24)
        return out
    def pi_op(al, vec):
        out = defaultdict(complex)
        for i, j in itertools.product(range(n), repeat=2):
            z = zeta[al, i, j]
            if abs(z) < 1e-12: continue
            axpy(out, q_op(i, p_op(j, vec)), z*np.sqrt(w[j]/w[i]))
        return out
    def inner(u, v): return sum(np.conj(u.get(k, 0))*c for k, c in v.items())

    levels = {}
    for nv in range(4):
        M = [tuple([na, nv-na] + [0]*(n-2)) for na in range(nv, -1, -1)]
        Ms = set(M); EM = E0(M[0])
        vecs = [defaultdict(complex, {s: 1.0}) for s in M]
        w3 = [V3(v) for v in vecs]
        v4 = [V4(v) for v in vecs]
        pis = [[pi_op(al, v) for al in range(3)] for v in vecs]
        H = np.zeros((len(M), len(M)), complex)
        Hc = np.zeros_like(H)
        for a, b in itertools.product(range(len(M)), repeat=2):
            h2 = sum(np.conj(w3[a].get(k, 0))*c/(EM - E0(k)) for k, c in w3[b].items() if k not in Ms)
            h4 = v4[b].get(M[a], 0)
            hc = sum(B[al]*inner(pis[a][al], pis[b][al]) for al in range(3))
            H[a, b] = h2 + h4 + hc; Hc[a, b] = hc
        H = 0.5*(H + H.conj().T)
        ev, U = np.linalg.eigh(H)
        # l^2 of each eigenvector: l = q_a p_b - q_b p_a on the two nu8 components
        def lop(v):
            out = defaultdict(complex); axpy(out, q_op(0, p_op(1, v)), 1); axpy(out, q_op(1, p_op(0, v)), -1); return out
        for c in range(len(M)):
            vec = defaultdict(complex)
            for a in range(len(M)): vec[M[a]] += U[a, c]
            l2 = inner(lop(vec), lop(vec)).real
            levels.setdefault(nv, []).append((round(l2), EM + ev[c]))
    E00 = levels[0][0][1]
    lv = {nv: sorted((l2, e - E00) for l2, e in L) for nv, L in levels.items()}
    g88 = (np.mean([e for l2, e in lv[2] if l2 == 4]) - lv[2][0][1])/4
    g88_3 = (np.mean([e for l2, e in lv[3] if l2 == 9]) - np.mean([e for l2, e in lv[3] if l2 == 1]))/8
    wt = lv[1][0][1] - g88
    x88 = (lv[2][0][1] - 2*wt)/2
    pred31 = 3*wt + 6*x88 + g88
    split33 = np.ptp([e for l2, e in lv[3] if l2 == 9])
    out = dict(w8=w[0], zeta8=zeta[2, 0, 1], Ae=Ae, Be=Be, Azeta2=Ae*zeta[2, 0, 1]**2, wt=wt, x88=x88, g88=g88,
               g88_from_v3=g88_3, E31_pred=pred31, E31=np.mean([e for l2, e in lv[3] if l2 == 1]), split33=split33,
               phi888=T3[0, 0, 0], phi8888=T4[0, 0, 0, 0], phi8a8a8b8b=T4[0, 0, 1, 1], cub_sym_err=sym_err, quart_check=quart_check, levels=lv)
    if verbose:
        print(f"nu8 harmonic {w[0]:.2f} cm-1, zeta8 = {zeta[2,0,1]:+.4f}, A_e zeta^2 = {out['Azeta2']:.3f}")
        print(f"phi_888 = {T3[0,0,0]:+.3f}, phi_8888 = {T4[0,0,0,0]:+.3f}, 3 phi_8a8a8b8b = {3*T4[0,0,1,1]:+.3f} cm-1 (isotropy check)")
        for nv in lv: print(f"  v8={nv}: " + ", ".join(f"l^2={l2}: {e:9.3f}" for l2, e in lv[nv]))
        print(f"w~ = {wt:.3f}, x88 = {x88:+.4f}, g88 = {g88:+.4f} (from v=3: {g88_3:+.4f}); g88 - A zeta^2 = {g88-out['Azeta2']:+.4f}")
        print(f"check v=3 |l|=1: PT {out['E31']:.4f} vs fit {pred31:.4f};  |l|=3 a1/a2 splitting {split33*1e3:.3f} x 1e-3 cm-1")
    return out

if __name__ == '__main__':
    for rich, sym in ((False, False), (False, True), (True, True)):
        if rich and 'cub_h' not in np.load(sys.argv[1]): continue
        print(f"--- Richardson {rich}, C3v average {sym} ---"); run(sys.argv[1], rich=rich, sym=sym)
