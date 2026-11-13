# Coriolis constants zeta_t^z of the four e modes of CaOCH3 (X~ 2A1) from the EA-EOM-CCSD/TZ e-block Hessian
# cached in ../caoch3_eom_ccsd/out (caoch3_emodes.py), plus the zeta sum rule check sum_t zeta_t = B_e/(2 A_e).
import json, itertools, sys, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caoch3_eom_ccsd')   # EA-EOM-CCSD scripts and outputs
sys.path.insert(0, R)
from caoch3_emodes import ex_basis, sym_ops, MASS, H_MW, W_CONV
from caoch3_surface import geometry
res = json.load(open(f'{R}/out/caoch3_min_tz_X.json'))
at = geometry(res['qmin']); syms = [s for s, _ in at]
x0 = np.array([c for _, c in at]); m = np.array([MASS[s] for s in syms])
U, sm = ex_basis(x0, m, f'{R}/out/caoch3_emode_basis_tz_X.json')   # the basis the cached points were displaced along
cache = json.load(open(f'{R}/out/caoch3_emode_cache_tz_X.json'))
E = lambda k: cache[k][0]
h = H_MW; n = 4
Hs = np.zeros((n, n))
for i in range(n):
    Hs[i, i] = (E(f'{i}+') + E(f'{i}-') - 2*E('c'))/h**2
for i, j in itertools.combinations(range(n), 2):
    Hs[i, j] = Hs[j, i] = (E(f'{i}{j}++') + E(f'{i}{j}--') - E(f'{i}+') - E(f'{i}-') - E(f'{j}+') - E(f'{j}-') + 2*E('c'))/(2*h*h)
lam, L = np.linalg.eigh(Hs)
w = np.sqrt(lam)*W_CONV
print('harmonic e frequencies (cm-1):', np.round(w, 2))
ops = sym_ops(); C3 = ops[1][0]; C3i = ops[2][0]
Mz = np.zeros((18, 18))
for a in range(6):
    Mz[3*a, 3*a+1] = -1; Mz[3*a+1, 3*a] = 1     # (r x p)_z = x p_y - y p_x
zetas = []
for t in range(n):
    lx = U @ L[:, t]                              # mass-weighted e_x normal mode
    ly = (C3 @ lx - C3i @ lx)/np.sqrt(3)          # its e_y partner
    ly /= np.linalg.norm(ly)
    # sign convention of the partner: choose ly so that (lx, ly) is the standard pair; zeta = lx . (Mz ly) up to that sign
    z = lx @ Mz.T @ ly
    zetas.append(z)
    print(f'mode {t}: w = {w[t]:8.2f} cm-1, |lx.ly| = {abs(lx@ly):.1e}, zeta^z = {z:+.4f}')
# equilibrium rotational constants
mt = m.sum(); com = (m[:, None]*x0).sum(0)/mt; xc = x0 - com
Izz = np.sum(m*(xc[:, 0]**2 + xc[:, 1]**2)); Ixx = np.sum(m*(xc[:, 1]**2 + xc[:, 2]**2))
conv = 16.857629206   # cm-1 * amu * A^2  (h / (8 pi^2 c))
Ae, Be = conv/Izz, conv/Ixx
print(f'A_e = {Ae:.4f}, B_e = {Be:.5f} cm-1; B_e/(2A_e) = {Be/(2*Ae):.5f}; sum zeta = {sum(zetas):+.5f}; sum |zeta| sign-free check')
