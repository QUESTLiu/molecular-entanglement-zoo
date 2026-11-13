#!/usr/bin/env python3
"""
CaOCH3: C3v-constrained EA-EOM-CCSD geometry optimisation + a1 Hessian for
X~ 2A1 (root 1) or A~ 2E (mean of roots 2,3), from the closed-shell cation.

PySCF has no EOM gradients, so this uses energy-only finite differences in
the four C3v internal coordinates q = (r_CaO, r_OC, r_CH, a_OCH).  These four
span exactly the a1 block (CaOCH3 has 4 a1 + 4 e modes), so the Hessian is
the full a1 force field and feeds straight into the Franck-Condon step.

Each stencil = 21 single points (centre, +-h_i, +-(h_i+h_j)) -> g and H.
Newton steps until |dq| < tol.  Every single point is cached in a JSON file,
so a killed job restarts where it stopped.

Run:  OMP_NUM_THREADS=9 python caoch3_surface.py --preset tz --state A
"""
import argparse, json, os, sys, time, itertools
import numpy as np
from pyscf import gto, scf, cc

PRESETS = {
    'dz': {'Ca': 'cc-pcvdz', 'O': 'cc-pvdz', 'C': 'cc-pvdz', 'H': 'cc-pvdz'},
    'tz': {'Ca': 'cc-pcvtz', 'O': 'cc-pvtz', 'C': 'cc-pvtz', 'H': 'cc-pvtz'},
    'qz': {'Ca': 'cc-pcvqz', 'O': 'cc-pvqz', 'C': 'cc-pvqz', 'H': 'cc-pvqz'},
}
Q0 = np.array([1.9726, 1.3989, 1.090, 111.545])      # README start geometry
H_STEP = np.array([0.005, 0.005, 0.005, 0.5])        # A, A, A, deg


def geometry(q):
    rCaO, rOC, rCH, aOCH = q
    zO, zC = rCaO, rCaO + rOC
    th = np.radians(180.0 - aOCH)
    dz, rr = rCH * np.cos(th), rCH * np.sin(th)
    at = [('Ca', (0., 0., 0.)), ('O', (0., 0., zO)), ('C', (0., 0., zC))]
    for k in range(3):
        p = np.radians(120 * k)
        at.append(('H', (rr * np.cos(p), rr * np.sin(p), zC + dz)))
    return at


class Surface:
    def __init__(self, preset, x2c, cachefile, nroots=3):
        self.basis, self.x2c, self.nroots = PRESETS[preset], x2c, nroots
        self.cachefile = cachefile
        self.cache = json.load(open(cachefile)) if os.path.exists(cachefile) else {}

    def key(self, q):
        return ','.join(f'{v:.6f}' for v in q)

    def roots(self, q):
        k = self.key(q)
        if k not in self.cache:
            t = time.time()
            mol = gto.M(atom=geometry(q), basis=self.basis, charge=1, spin=0,
                        verbose=0, max_memory=int(os.environ.get('PYSCF_MAX_MEMORY', 16000)))
            mf = scf.RHF(mol)
            if self.x2c:
                mf = mf.x2c()
            mf.conv_tol = 1e-11
            mf.run()
            ncore = int(np.sum(mf.mo_energy < -5.0))          # keep Ca 3s3p correlated
            mycc = cc.CCSD(mf); mycc.frozen = ncore; mycc.conv_tol = 1e-9
            mycc.run()
            ea, _ = mycc.eaccsd(nroots=self.nroots)
            self.cache[k] = [mycc.e_tot + e for e in np.atleast_1d(ea)]
            json.dump(self.cache, open(self.cachefile, 'w'))
            print(f"    sp {k}  X {self.cache[k][0]:.8f}  A {self.cache[k][1]:.8f} "
                  f"{self.cache[k][2]:.8f}  [{time.time()-t:.0f} s]", flush=True)
        return np.array(self.cache[k])

    def energy(self, q, state):
        r = self.roots(q)
        return r[0] if state == 'X' else 0.5 * (r[1] + r[2])

    def gradient(self, q, state, h=H_STEP):
        E0 = self.energy(q, state)
        g = np.array([(self.energy(q + h[i] * np.eye(4)[i], state) -
                       self.energy(q - h[i] * np.eye(4)[i], state)) / (2 * h[i]) for i in range(4)])
        return E0, g

    def stencil(self, q, state, h=H_STEP):
        n = len(q)
        E0 = self.energy(q, state)
        ep, em = np.zeros(n), np.zeros(n)
        for i in range(n):
            d = np.zeros(n); d[i] = h[i]
            ep[i], em[i] = self.energy(q + d, state), self.energy(q - d, state)
        g = (ep - em) / (2 * h)
        H = np.diag((ep + em - 2 * E0) / h**2)
        for i, j in itertools.combinations(range(n), 2):
            d = np.zeros(n); d[i], d[j] = h[i], h[j]
            epp, emm = self.energy(q + d, state), self.energy(q - d, state)
            H[i, j] = H[j, i] = (epp + emm - ep[i] - em[i] - ep[j] - em[j] + 2 * E0) / (2 * h[i] * h[j])
        return E0, g, H


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='dz', choices=list(PRESETS))
    p.add_argument('--state', default='X', choices=['X', 'A'])
    p.add_argument('--x2c', action='store_true')
    p.add_argument('--start', type=float, nargs=4, default=None)
    p.add_argument('--maxiter', type=int, default=8)
    p.add_argument('--hess-guess', default=None,
                   help='JSON from a cheaper run (e.g. dz): quasi-Newton with gradients only '
                        '(9 points/step), full 21-point Hessian only at the end')
    a = p.parse_args()

    tag = f"{a.preset}{'_x2c' if a.x2c else ''}"
    S = Surface(a.preset, a.x2c, f'caoch3_cache_{tag}_{a.state}.json')
    q = np.array(a.start) if a.start else Q0.copy()
    print(f"CaOCH3 EA-EOM-CCSD  state={a.state}  basis={tag}  start={q}", flush=True)

    tol = np.array([2e-4, 2e-4, 2e-4, 0.02])
    cap = 0.05 * np.array([1, 1, 1, 50])
    if a.hess_guess:
        H = np.array(json.load(open(a.hess_guess))['H'])
        g_old = dq = None
        for it in range(a.maxiter):
            E0, g = S.gradient(q, a.state)
            if g_old is not None:                                   # BFGS update
                y = g - g_old
                if dq @ y > 0:
                    Hs = H @ dq
                    H = H + np.outer(y, y) / (y @ dq) - np.outer(Hs, Hs) / (dq @ Hs)
            dq = np.clip(-np.linalg.solve(H, g), -cap, cap)
            print(f"  qN iter {it}  q = {np.round(q, 5)}  E = {E0:.8f}  |g| = {np.linalg.norm(g):.2e}"
                  f"  dq = {np.round(dq, 5)}", flush=True)
            if np.all(np.abs(dq) < 2.5 * tol):
                break
            q, g_old = np.round(q + dq, 6), g
        a.maxiter = 2                                               # final Hessian (+1 polish)

    for it in range(a.maxiter):
        E0, g, H = S.stencil(q, a.state)
        dq = -np.linalg.solve(H, g)
        dq = np.clip(dq, -cap, cap)
        print(f"  iter {it}  q = {np.round(q, 5)}  E = {E0:.8f}  |g| = {np.linalg.norm(g):.2e}"
              f"  dq = {np.round(dq, 5)}", flush=True)
        if np.all(np.abs(dq) < tol):
            break
        q = np.round(q + dq, 6)

    qmin = q + dq
    Emin = E0 + 0.5 * g @ dq
    res = dict(state=a.state, basis=tag, q=q.tolist(), qmin=qmin.tolist(), E0=E0, Emin=Emin,
               g=g.tolist(), H=H.tolist(), units='Eh; q in (A, A, A, deg)',
               roots_at_q=S.roots(q).tolist())
    json.dump(res, open(f'caoch3_min_{tag}_{a.state}.json', 'w'), indent=1)
    print(f"  minimum  r(Ca-O) {qmin[0]:.4f}  r(O-C) {qmin[1]:.4f}  r(C-H) {qmin[2]:.4f} A"
          f"  a(OCH) {qmin[3]:.2f} deg   E = {Emin:.8f}")


if __name__ == '__main__':
    main()
