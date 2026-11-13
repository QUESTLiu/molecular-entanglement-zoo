# Fundamental transition dipole mu8 = <1|mu_x|0> of the CaOCH3 X~ 2A1 Ca-O-C bend (nu8),
# double-harmonic approximation, UKS-B3LYP/def2-TZVP (same level, geometry and normal modes as dft_force_field.py).
# 1. dipole moments at Q8a = +-d and Q8b = +-d (mass-weighted normal coordinates of the two nu8 components)
# 2. central differences -> d mu / d Q (Debye per amu^1/2 A)
# 3. mu8 = |d mu_x / d Q8a| * <1|Q|0>,  <1|Q|0> = sqrt(hbar / (2 omega_8)), omega_8 the harmonic B3LYP wavenumber
# Usage: python bend_dipole.py [dft_force_field_b3lyp_def2tzvp.npz]   -> bend_dipole_b3lyp.out
import sys, os, time
import numpy as np
from pyscf import gto, dft
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caoch3_eom_ccsd')
sys.path.insert(0, R)
from caoch3_emodes import MASS

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'dft_force_field_b3lyp_def2tzvp.npz'
z = np.load(NPZ)
x0, L, w, d = z['x0'], z['L'], z['w'], float(z['d'])
XC, BASIS = 'b3lyp', 'def2-tzvp'
syms = ['Ca', 'O', 'C', 'H', 'H', 'H']
m = np.array([MASS[s] for s in syms]); sm = np.sqrt(np.repeat(m, 3))

dm_prev = [None]
def dipole(x):
    mol = gto.M(atom=list(zip(syms, map(tuple, x))), basis=BASIS, spin=1, charge=0, verbose=0, max_memory=16000)
    mf = dft.UKS(mol); mf.xc = XC; mf.conv_tol = 1e-11; mf.conv_tol_grad = 1e-7
    mf.grids.atom_grid = (99, 590); mf.grids.prune = None
    mf.kernel(dm0=dm_prev[0]); assert mf.converged
    dm_prev[0] = mf.make_rdm1()
    return np.asarray(mf.dip_moment(unit='Debye', verbose=0))   # neutral molecule: origin independent

t0 = time.time()
mu0 = dipole(x0)
print(f"[{time.time()-t0:.0f}s] equilibrium dipole (D): {np.round(mu0, 4)}  (|mu| = {np.linalg.norm(mu0):.3f} D, along the C3 axis z)", flush=True)
dmu = {}
for comp, name in ((0, 'Q8a (x-like)'), (1, 'Q8b (y-like)')):
    mp = dipole(x0 + (+d*L[:, comp]/sm).reshape(6, 3))
    mm = dipole(x0 + (-d*L[:, comp]/sm).reshape(6, 3))
    dmu[comp] = (mp - mm)/(2*d)
    print(f"[{time.time()-t0:.0f}s] d mu / d {name} = {np.round(dmu[comp], 5)} D/(amu^1/2 A)", flush=True)

# <1|Q|0> = sqrt(hbar/(2 omega)) in amu^1/2 A, omega = 2 pi c w
hbar, c, amu = 1.054571817e-34, 2.99792458e10, 1.66053906660e-27   # J s, cm/s, kg
w8 = 0.5*(w[0] + w[1])
Q10 = np.sqrt(hbar/(2*2*np.pi*c*w8)/amu)*1e10
mu8a, mu8b = abs(dmu[0][0])*Q10, abs(dmu[1][1])*Q10
print(f"harmonic omega_8 = {w8:.2f} cm-1, <1|Q|0> = {Q10:.4f} amu^1/2 A")
print(f"mu8 = <1|mu_x|0> = {mu8a:.4f} D (from Q8a);  <1|mu_y|0> = {mu8b:.4f} D (from Q8b)")
print(f"off-axis components (should vanish by symmetry): dmu_y/dQ8a = {dmu[0][1]:.1e}, dmu_z/dQ8a = {dmu[0][2]:.1e}, "
      f"dmu_x/dQ8b = {dmu[1][0]:.1e}, dmu_z/dQ8b = {dmu[1][2]:.1e}")
print(f"MU8_DEBYE = {0.5*(mu8a + mu8b):.4f}")
