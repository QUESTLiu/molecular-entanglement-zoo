#!/usr/bin/env python3
"""
CaOCH3 e-mode force field and Jahn-Teller analysis (EA-EOM-CCSD, C3v minima).

The a1 block is done in caoch3_surface.py.  Here the four e_x vibrations
(symmetric w.r.t. the xz mirror plane, built with C3v projection operators and
freed of translation/rotation) are displaced in mass-weighted Cartesians:

  X~ 2A1 : Hessian in the e block  -> 4 doubly degenerate e frequencies (ZPE)
  A~ 2E  : roots 2/3 split linearly along e (E x e Jahn-Teller).
           average (E2+E3)/2 -> e force constants / frequencies
           splitting E3-E2   -> linear JT gradient F (sign-resolved with the
                                pair displacements) -> E_JT per mode, D_k = E_JT/hw
           E x e vibronic model per mode -> Ham reduction factor p of the
           orbital angular momentum -> observable SO splitting = p * a*zeta_e

Every single point is cached (JSON), same as caoch3_surface.py.

Run:  OMP_NUM_THREADS=5 python caoch3_emodes.py caoch3_min_tz_A.json --preset tz
"""
import argparse, itertools, json, os, sys, time
import numpy as np
from pyscf import gto, scf, cc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caoch3_surface import geometry, PRESETS

MASS = {'Ca': 39.9625909, 'O': 15.9949146, 'C': 12.0, 'H': 1.00782503}
HA2CM = 219474.6313705
# sqrt(Eh / (amu * A^2)) in cm^-1
W_CONV = np.sqrt(4.3597447222e-18 / (1.66053906660e-27 * 1e-20)) / (2 * np.pi * 2.99792458e10)
H_MW = 0.02                                   # step, amu^1/2 * Angstrom


def sym_ops():
    """18x18 Cartesian representation of C3v for atoms (Ca,O,C,H1,H2,H3)."""
    def rot(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def refl(phi):                            # mirror plane containing z at azimuth phi
        c, s = np.cos(2 * phi), np.sin(2 * phi)
        return np.array([[c, s, 0], [s, -c, 0], [0, 0, 1]])

    ops = []                                  # (matrix, chi_A1, chi_A2)
    for k in range(3):                        # E, C3, C3^2 : H_i -> H_{i+k}
        R = rot(2 * np.pi * k / 3)
        perm = [0, 1, 2] + [3 + (i + k) % 3 for i in range(3)]
        ops.append((R, perm, 1, 1))
    for k in range(3):                        # sigma_v through H_k: H_i -> H_{2k - i}
        R = refl(2 * np.pi * k / 3)
        perm = [0, 1, 2] + [3 + (2 * k - i) % 3 for i in range(3)]
        ops.append((R, perm, 1, -1))
    mats = []
    for R, perm, a1, a2 in ops:
        G = np.zeros((18, 18))
        for i, j in enumerate(perm):          # atom i moves to position j
            G[3 * j:3 * j + 3, 3 * i:3 * i + 3] = R
        mats.append((G, a1, a2))
    return mats


def ex_modes(x0, m):
    """orthonormal mass-weighted e_x vibrational vectors (4 of them)."""
    ops = sym_ops()
    PA1 = sum(G for G, _, _ in ops) / 6
    PA2 = sum(a2 * G for G, _, a2 in ops) / 6
    PE = np.eye(18) - PA1 - PA2
    sig_xz = ops[3][0]                        # sigma_v through H1 (the xz plane)
    P = 0.5 * (np.eye(18) + sig_xz) @ PE
    sm = np.sqrt(np.repeat(m, 3))
    # translations x, rotation about y (both e_x-like), mass-weighted
    com = (m[:, None] * x0).sum(0) / m.sum()
    xc = x0 - com
    Tx = np.tile([1, 0, 0], 6) * sm
    Ry = np.concatenate([np.cross([0, 1, 0], r) for r in xc]) * sm
    TR = np.linalg.qr(np.stack([Tx, Ry], 1))[0]
    V = P @ np.eye(18)
    V = V - TR @ (TR.T @ V)
    u, s, _ = np.linalg.svd(V)
    U = u[:, s > 1e-6]
    assert U.shape[1] == 4, U.shape
    return U, sm


def ex_basis(x0, m, basisfile):
    """e_x displacement basis tied to a single-point cache.

    ex_modes() takes its four vectors from an SVD with four degenerate singular values, so the
    individual vectors (not the space they span) depend on the LAPACK build.  The cached energies
    are labelled by these vectors ('0+', '01++', ...).  Frequencies and Jahn-Teller energies do not
    depend on the choice, but anything that maps the modes back to Cartesian displacements does
    (Coriolis constants, cuts along a normal mode).  The basis is therefore stored next to the
    cache and reused."""
    U, sm = ex_modes(x0, m)
    if os.path.exists(basisfile):
        Us = np.array(json.load(open(basisfile))['U'])
        assert np.abs(Us @ Us.T - U @ U.T).max() < 1e-8, f'{basisfile}: stored basis spans a different space'
        return Us, sm
    json.dump({'U': U.tolist()}, open(basisfile, 'w'), indent=1)
    return U, sm


class Points:
    def __init__(self, preset, cachefile):
        self.basis = PRESETS[preset]; self.cachefile = cachefile
        self.cache = json.load(open(cachefile)) if os.path.exists(cachefile) else {}

    def roots(self, key, syms, xyz):
        if key not in self.cache:
            t = time.time()
            mol = gto.M(atom=list(zip(syms, map(tuple, xyz))), basis=self.basis, charge=1,
                        verbose=0, max_memory=int(os.environ.get('PYSCF_MAX_MEMORY', 14000)))
            mf = scf.RHF(mol); mf.conv_tol = 1e-11; mf.run()
            ncore = int(np.sum(mf.mo_energy < -5.0))
            mc = cc.CCSD(mf); mc.frozen = ncore; mc.conv_tol = 1e-9; mc.run()
            ea, _ = mc.eaccsd(nroots=3)
            self.cache[key] = [mc.e_tot + e for e in np.sort(np.atleast_1d(ea))]
            json.dump(self.cache, open(self.cachefile + '.tmp', 'w'))
            os.replace(self.cachefile + '.tmp', self.cachefile)
            print(f"    sp {key:14s} roots {np.round(self.cache[key], 7)}  [{time.time()-t:.0f} s]", flush=True)
        return np.array(self.cache[key])


def exe_reduction(D, nmax=40):
    """Ham reduction factor p of L_z for single-mode linear E x e JT, D = E_JT/hw."""
    k = np.sqrt(2 * D)
    idx = [(i, j) for i in range(nmax + 1) for j in range(nmax + 1 - i)]
    pos = {ij: n for n, ij in enumerate(idx)}
    N = len(idx)
    q = [np.zeros((N, N)), np.zeros((N, N))]
    for (i, j), n in pos.items():
        for d, (ii, jj) in enumerate(((i + 1, j), (i, j + 1))):
            if (ii, jj) in pos:
                v = np.sqrt((i + 1) if d == 0 else (j + 1)) / np.sqrt(2)
                q[d][pos[(ii, jj)], n] = q[d][n, pos[(ii, jj)]] = v
    H0 = np.diag([i + j + 1.0 for i, j in idx])
    sz, sx, sy = np.diag([1., -1.]), np.array([[0, 1.], [1, 0]]), np.array([[0, -1j], [1j, 0]])
    eps = 1e-4
    H = np.kron(np.eye(2), H0) + k * (np.kron(sz, q[0]) + np.kron(sx, q[1])) + eps * np.kron(sy, np.eye(N))
    w = np.linalg.eigvalsh(H)
    return (w[1] - w[0]) / (2 * eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('minjson', help='caoch3_min_<basis>_<X|A>.json')
    p.add_argument('--preset', default='tz', choices=list(PRESETS))
    a = p.parse_args()
    res = json.load(open(a.minjson))
    state = res['state']
    at = geometry(res['qmin'])
    syms = [s for s, _ in at]
    x0 = np.array([c for _, c in at])
    m = np.array([MASS[s] for s in syms])
    U, sm = ex_basis(x0, m, f'caoch3_emode_basis_{a.preset}_{state}.json')
    P = Points(a.preset, f'caoch3_emode_cache_{a.preset}_{state}.json')

    def pt(disp, key):                       # disp: mass-weighted 18-vector
        return P.roots(key, syms, x0 + (disp / sm).reshape(6, 3))

    h = H_MW
    r0 = pt(np.zeros(18), 'c')
    n = 4
    R = {}
    for i in range(n):
        R[(i, 1)] = pt(h * U[:, i], f'{i}+'); R[(i, -1)] = pt(-h * U[:, i], f'{i}-')
    for i, j in itertools.combinations(range(n), 2):
        R[(i, j, 1)] = pt(h * (U[:, i] + U[:, j]), f'{i}{j}++')
        R[(i, j, -1)] = pt(-h * (U[:, i] + U[:, j]), f'{i}{j}--')

    def E(r):                                 # state energy on this point
        return r[0] if state == 'X' else 0.5 * (r[1] + r[2])

    def S(r):                                 # A~ splitting
        return r[2] - r[1]

    Hs = np.zeros((n, n))
    for i in range(n):
        Hs[i, i] = (E(R[(i, 1)]) + E(R[(i, -1)]) - 2 * E(r0)) / h**2
    for i, j in itertools.combinations(range(n), 2):
        Hs[i, j] = Hs[j, i] = (E(R[(i, j, 1)]) + E(R[(i, j, -1)]) - E(R[(i, 1)]) - E(R[(i, -1)])
                               - E(R[(j, 1)]) - E(R[(j, -1)]) + 2 * E(r0)) / (2 * h * h)
    lam, L = np.linalg.eigh(Hs)
    w = np.sign(lam) * np.sqrt(np.abs(lam)) * W_CONV
    out = dict(state=state, basis=a.preset, e_freqs_cm=w.tolist(), ZPE_e_cm=float(np.sum(w)),
               split_at_ref_cm=float(S(r0) * HA2CM) if state == 'A' else None)
    print(f"\n  {state}~  e-mode harmonic frequencies (each doubly degenerate): {np.round(w, 1)} cm^-1")
    print(f"  ZPE(e modes) = {np.sum(w):.1f} cm^-1")

    if state == 'A':
        # |F_i| from single displacements, relative signs from pair displacements
        Fa = np.array([0.5 * (S(R[(i, 1)]) + S(R[(i, -1)])) / (2 * h) for i in range(n)])
        sgn = np.ones(n)
        ref = int(np.argmax(Fa))
        for j in range(n):
            if j == ref:
                continue
            i, jj = sorted((ref, j))
            s_pair = 0.5 * (S(R[(i, jj, 1)]) + S(R[(i, jj, -1)])) / (2 * h)
            sgn[j] = 1 if abs(s_pair - (Fa[ref] + Fa[j])) < abs(s_pair - abs(Fa[ref] - Fa[j])) else -1
        F = sgn * Fa                                         # Eh / (amu^1/2 A), e_x basis
        Fn = L.T @ F                                         # normal-mode basis
        EJT = Fn**2 / (2 * lam) * HA2CM                      # cm^-1
        D = EJT / w
        p_k = [exe_reduction(d) for d in D]
        p_tot = float(np.prod(p_k))
        out.update(F_ex=F.tolist(), E_JT_cm=EJT.tolist(), D=D.tolist(), ham_p=p_k, ham_p_total=p_tot,
                   E_JT_total_cm=float(EJT.sum()))
        print(f"  split of roots 2/3 at the C3v minimum: {out['split_at_ref_cm']:.3f} cm^-1 (should be ~0)")
        for k in range(n):
            print(f"   mode {k}: w = {w[k]:7.1f} cm^-1   E_JT = {EJT[k]:7.2f} cm^-1   D = {D[k]:.4f}   p = {p_k[k]:.4f}")
        print(f"  total E_JT = {EJT.sum():.2f} cm^-1,  Ham reduction factor p = {p_tot:.4f}")
    json.dump(out, open(f'caoch3_emodes_{a.preset}_{state}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
