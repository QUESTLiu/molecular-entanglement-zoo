# Displaced Hessians at half the step of dft_force_field.py, same geometry and normal modes, for a
# Richardson extrapolation of the cubic and quartic constants (error O(d^2) from the large higher-order
# terms of a bend in rectilinear coordinates). Usage: XC=b3lyp python dft_half_step.py dft_force_field_<tag>.npz
import sys, os, time
import numpy as np
from pyscf import gto, dft
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caoch3_eom_ccsd')   # EA-EOM-CCSD scripts and outputs
sys.path.insert(0, R)
from caoch3_emodes import MASS
XC, BASIS = os.environ.get('XC', 'b3lyp'), os.environ.get('BASIS', 'def2-tzvp')
BOHR = 0.529177210903
syms = ['Ca', 'O', 'C', 'H', 'H', 'H']; m = np.array([MASS[s] for s in syms]); sm = np.sqrt(np.repeat(m, 3))
D = dict(np.load(sys.argv[1])); x0, L, d0 = D['x0'], D['L'], float(D['d'])
d = d0/2
def hessian_mw(x):
    mol = gto.M(atom=list(zip(syms, map(tuple, x))), basis=BASIS, spin=1, verbose=0, max_memory=16000)
    mf = dft.UKS(mol); mf.xc = XC; mf.conv_tol = 1e-11; mf.conv_tol_grad = 1e-7
    mf.grids.atom_grid = (99, 590); mf.grids.prune = None; mf.kernel()
    H = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(18, 18)/BOHR**2
    return H/np.outer(sm, sm)
t0 = time.time()
H0 = hessian_mw(x0); print(f"[{time.time()-t0:.0f}s] reference", flush=True)
Hd = {}
for comp in (0, 1):
    for sgn in (+1, -1):
        Hd[(comp, sgn)] = hessian_mw(x0 + (sgn*d*L[:, comp]/sm).reshape(6, 3))
        print(f"[{time.time()-t0:.0f}s] comp {comp} sign {sgn}", flush=True)
proj = lambda H: L.T @ H @ L
D['cub_h'] = np.array([proj(Hd[(c, 1)] - Hd[(c, -1)])/(2*d) for c in (0, 1)])
D['qua_h'] = np.array([proj(Hd[(c, 1)] + Hd[(c, -1)] - 2*H0)/d**2 for c in (0, 1)])
D['d_h'] = d
np.savez(sys.argv[1].replace('.npz', '_rich.npz'), **D)
print(f"[{time.time()-t0:.0f}s] saved", flush=True)
