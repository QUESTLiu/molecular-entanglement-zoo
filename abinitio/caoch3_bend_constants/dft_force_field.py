# Anharmonic force constants of the CaOCH3 X~ 2A1 Ca-O-C bend (nu8) from UKS-B3LYP/def2-TZVP analytic Hessians.
# 1. optimize the C3v geometry in the four internal coordinates of caoch3_surface.geometry
# 2. analytic Hessian -> normal modes (translations/rotations projected out), Coriolis zeta matrices
# 3. Hessians at Q8a = +-d and Q8b = +-d (mass-weighted normal coordinates of the two nu8 components)
#    -> cubic phi_{8 j k} (all j, k) and semi-diagonal quartic phi_{8 8 j k}
# Everything is saved to dft_force_field.npz; the perturbation theory is in vpt2_bend.py.
import sys, json, time, os
import numpy as np
from scipy.optimize import minimize
from pyscf import gto, dft
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caoch3_eom_ccsd')   # EA-EOM-CCSD scripts and outputs
sys.path.insert(0, R)
from caoch3_surface import geometry
from caoch3_emodes import sym_ops, MASS

XC, BASIS = os.environ.get('XC', 'b3lyp'), os.environ.get('BASIS', 'def2-tzvp')
TAG = f"{XC}_{BASIS}".replace('-', '')
BOHR = 0.529177210903
syms = ['Ca', 'O', 'C', 'H', 'H', 'H']
m = np.array([MASS[s] for s in syms])

def make_mf(x):
    mol = gto.M(atom=list(zip(syms, map(tuple, x))), basis=BASIS, spin=1, charge=0, verbose=0,
                max_memory=16000)
    mf = dft.UKS(mol); mf.xc = XC; mf.conv_tol = 1e-11; mf.conv_tol_grad = 1e-7
    mf.grids.atom_grid = (99, 590); mf.grids.prune = None
    return mf

def cart(q): return np.array([c for _, c in geometry(q)])

dm_prev = [None]
def e_and_g(q):
    x = cart(q); mf = make_mf(x)
    e = mf.kernel(dm0=dm_prev[0]); dm_prev[0] = mf.make_rdm1()
    gx = mf.nuc_grad_method().kernel() / BOHR            # Eh / A
    J = np.zeros((18, 4)); h = 1e-5
    for i in range(4):
        dq = np.zeros(4); dq[i] = h
        J[:, i] = (cart(q + dq) - cart(q - dq)).ravel()/(2*h)
    return e, J.T @ gx.ravel()

t0 = time.time()
q0 = np.array([1.99, 1.39, 1.10, 111.0])
res = minimize(e_and_g, q0, jac=True, method='BFGS', options={'gtol': 2e-7, 'maxiter': 60})
qmin = res.x; x0 = cart(qmin)
print(f"[{time.time()-t0:.0f}s] optimized q = {np.round(qmin, 5)}, E = {res.fun:.10f}, |g| = {np.abs(res.jac).max():.1e}", flush=True)

def hessian_mw(x):
    mf = make_mf(x); mf.kernel(dm0=dm_prev[0])
    H = mf.Hessian().kernel()                            # (6, 6, 3, 3) Eh / bohr^2
    H = H.transpose(0, 2, 1, 3).reshape(18, 18)/BOHR**2  # Eh / A^2
    sm = np.sqrt(np.repeat(m, 3))
    return H/np.outer(sm, sm)

H0 = hessian_mw(x0)
print(f"[{time.time()-t0:.0f}s] reference Hessian done", flush=True)
# project translations and rotations
sm = np.sqrt(np.repeat(m, 3)); com = (m[:, None]*x0).sum(0)/m.sum(); xc = x0 - com
TR = []
for a in range(3):
    t = np.zeros((6, 3)); t[:, a] = 1; TR.append((t*np.sqrt(m)[:, None]).ravel())
for a in range(3):
    e = np.zeros(3); e[a] = 1
    TR.append((np.cross(e, xc)*np.sqrt(m)[:, None]).ravel())
TR = np.linalg.qr(np.array(TR).T)[0]
Pv = np.eye(18) - TR @ TR.T
lam, V = np.linalg.eigh(Pv @ H0 @ Pv)
keep = np.argsort(np.abs(lam))[6:]; keep = keep[np.argsort(lam[keep])]
lam, L = lam[keep], V[:, keep]
W_CONV = np.sqrt(4.3597447222e-18/(1.66053906660e-27*1e-20))/(2*np.pi*2.99792458e10)
w = np.sqrt(lam)*W_CONV
# align degenerate pairs: first member symmetric under sigma_xz (x-like), second its C3 partner (y-like)
ops = sym_ops(); S, C3, C3i = ops[3][0], ops[1][0], ops[2][0]
i = 0
while i < len(w):
    if i + 1 < len(w) and abs(w[i+1] - w[i]) < 1e-3*max(w[i], 1):
        B2 = L[:, i:i+2]; Ssub = B2.T @ S @ B2
        ev, evec = np.linalg.eigh(Ssub)
        lx = B2 @ evec[:, np.argmax(ev)]; lx /= np.linalg.norm(lx)
        ly = (C3 @ lx - C3i @ lx)/np.sqrt(3); ly /= np.linalg.norm(ly)
        L[:, i], L[:, i+1] = lx, ly; i += 2
    else:
        i += 1
print("harmonic wavenumbers (cm-1):", np.round(w, 2), flush=True)
i8 = 0   # the lowest pair is nu8
assert abs(w[0] - w[1]) < 0.5
# Coriolis zeta matrices zeta^alpha_{ij} = sum_atoms (l_{beta i} l_{gamma j} - l_{gamma i} l_{beta j})
zeta = np.zeros((3, 12, 12))
Lr = L.reshape(6, 3, 12)
for a, (b, c) in enumerate([(1, 2), (2, 0), (0, 1)]):
    zeta[a] = np.einsum('ni,nj->ij', Lr[:, b, :], Lr[:, c, :]) - np.einsum('ni,nj->ij', Lr[:, c, :], Lr[:, b, :])
print("zeta^z_{8a,8b} =", round(zeta[2, 0, 1], 4), flush=True)
Izz = np.sum(m*(xc[:, 0]**2 + xc[:, 1]**2)); Ixx = np.sum(m*(xc[:, 1]**2 + xc[:, 2]**2))
Ae, Be = 16.857629206/Izz, 16.857629206/Ixx
d = 0.10
Hd = {}
for comp in (0, 1):
    for sgn in (+1, -1):
        Hd[(comp, sgn)] = hessian_mw(x0 + (sgn*d*L[:, comp]/sm).reshape(6, 3))
        print(f"[{time.time()-t0:.0f}s] displaced Hessian comp {comp} sign {sgn} done", flush=True)
proj = lambda H: L.T @ H @ L
cub = np.array([proj(Hd[(c, 1)] - Hd[(c, -1)])/(2*d) for c in (0, 1)])          # phi_{c j k}, Eh/(amu^1.5 A^3)
qua = np.array([proj(Hd[(c, 1)] + Hd[(c, -1)] - 2*H0)/d**2 for c in (0, 1)])      # phi_{c c j k}
np.savez(f'dft_force_field_{TAG}.npz', w=w, L=L, lam=lam, zeta=zeta, cub=cub, qua=qua, Ae=Ae, Be=Be, qmin=qmin, x0=x0, d=d)
print(f"[{time.time()-t0:.0f}s] saved dft_force_field_{TAG}.npz; A_e={Ae:.4f}, B_e={Be:.5f}", flush=True)
