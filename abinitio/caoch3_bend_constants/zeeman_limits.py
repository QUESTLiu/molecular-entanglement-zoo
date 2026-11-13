# Magnetic-field limits for the ortho preparation tree: condition (vi) of Appendix app:caoch3_pulses.
#
# Adds the electron Zeeman term g_S mu_B B_Z S_Z (laboratory Z, the field polarization) to the case (b)
# spin structure of check_tree_measured.py (spin-rotation + ortho proton hyperfine, Namiki et al. 1998,
# v=0 constants for every bend level) and reports, for the tree component (J, F) of every level:
#   1. g_F: numerical slope of the M_F = 1 component at 1 mG, next to the weak-field formula
#      g_F = g_J [F(F+1)+J(J+1)-I(I+1)] / [2F(F+1)],  g_J = g_S/(2N+1) for J = N + 1/2;
#   2. the transverse field B_perp that keeps the loss from M_F = 0 below 1e-3 if the population stays
#      0.217 ms (the longest sequence, pulse_durations.out) in that level: the sublevels rotate by
#      alpha = 2 pi g_F mu_B B_perp t / h, and the M_F = 0 population left is P_F(cos alpha)^2;
#   3. with a bias field B_Z along Z: the M_F = 0 -> +-1 splitting g_F mu_B B_Z of each level, and the
#      quadratic shift of each M_F = 0 tree component and of each tree line P1-P7.
# Rotational and nuclear Zeeman terms are omitted (moments ~1e-3 of the electron's).
import io, os, contextlib
import numpy as np
from numpy.polynomial import legendre

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, 'check_tree_measured.py')
ns = {'__file__': src, '__name__': 'zeeman_limits'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(src).read(), ns)
spin_levels, levels, qn = ns['spin_levels'], ns['levels'], ns['qn']

gS, muB = 2.00231930, 1.39962449       # electron g factor; Bohr magneton in MHz/G (CODATA)
T_SEQ = 217e-6                          # s, longest sequence (pulse_durations.out)
paper_name = {'L03_3': 'L03', 'L03_4': 'L30'}


def shifted(N, k, MF, Bz):
    """Energies (MHz) of the (J, F) components at field Bz (G), followed from zero field by overlap."""
    w, V, basis, labs = spin_levels(N, k, MF=MF)
    HZ = np.diag([gS*muB*Bz*float(MS) for (MN, MS, MI) in basis])
    w1, V1 = np.linalg.eigh(V @ np.diag(w) @ V.T + HZ)
    out = {}
    for c in range(len(w1)):
        parent = int(np.argmax(np.abs(V.T @ V1[:, c])))
        out[labs[parent][:2]] = w1[c]
    return out, {lab[:2]: w[c] for c, lab in enumerate(labs)}


def gF_formula(N, J, F, I=1.5):
    gJ = gS*(J*(J + 1) + 0.75 - N*(N + 1))/(2*J*(J + 1))
    return gJ*(F*(F + 1) + J*(J + 1) - I*(I + 1))/(2*F*(F + 1))


tree, gnum = {}, {}
print("1. Lande factors of the tree components (weak field)")
print("   level  N |k|  J    F   g_F formula  g_F numerical  rate (MHz/G)")
for nm in levels:
    v, ell, N, k, J, F = qn(nm)
    J = float(J)
    tree[nm] = (N, abs(k), J, F)
    eB, e0 = shifted(N, abs(k), 1, 1e-3)
    gnum[nm] = (eB[(J, F)] - e0[(J, F)])/(muB*1e-3)
    print(f"   {paper_name.get(nm, nm):5s}  {N}  {abs(k)}  {J:3.1f}  {F}   {gF_formula(N, J, F):.4f}       "
          f"{gnum[nm]:.4f}         {gnum[nm]*muB:.3f}")
dev = max(abs(gnum[n]/gF_formula(*[tree[n][i] for i in (0, 2, 3)]) - 1) for n in tree)
print(f"   g_F range {min(gnum.values()):.2f} to {max(gnum.values()):.2f}; largest rate "
      f"{max(gnum.values())*muB:.2f} MHz/G; largest deviation from the formula {100*dev:.1f}%")

print(f"\n2. Transverse field: B_perp for loss < 1e-3 from M_F = 0 during {T_SEQ*1e3:.3f} ms")
lim = {}
for nm, (N, k, J, F) in tree.items():
    rate = 2*np.pi*gnum[nm]*muB*1e6           # rad/s per gauss
    c = np.zeros(F + 1); c[F] = 1
    Bu = 0.0
    while 1 - legendre.legval(np.cos(rate*(Bu + 0.1)*1e-6*T_SEQ), c)**2 < 1e-3:
        Bu += 0.1
    lim[nm] = Bu
    print(f"   {paper_name.get(nm, nm):5s} F={F}: B_perp < {Bu:.1f} uG")
print(f"   smallest limit: {min(lim.values()):.1f} uG")

lines = {'P1': ('G', 'L01'), 'P2': ('L01', 'L02'), 'P3': ('L01', 'L11'), 'P4': ('L02', 'L03_3'),
         'P5': ('L02', 'L03_4'), 'P6': ('L11', 'L12'), 'P7': ('L11', 'L21')}
print("\n3. Bias field B_Z along Z")
for Bz in (0.02, 0.05, 0.1):
    sh = {}
    for nm, (N, k, J, F) in tree.items():
        eB, e0 = shifted(N, k, 0, Bz)
        sh[nm] = (eB[(J, F)] - e0[(J, F)])*1e3          # kHz
    ls = {p: sh[u] - sh[l] for p, (l, u) in lines.items()}
    split = {nm: gnum[nm]*muB*Bz*1e3 for nm in tree}
    big = max(ls, key=lambda p: abs(ls[p]))
    print(f"   B_Z = {Bz*1e3:.0f} mG: M_F = 0 -> +-1 splittings {min(split.values()):.1f} to "
          f"{max(split.values()):.1f} kHz; largest quadratic line shift {abs(ls[big]):.2f} kHz ({big})")
    print("      level shifts (kHz): " + ", ".join(f"{paper_name.get(n, n)} {s:+.2f}" for n, s in sh.items()))
    print("      line shifts (kHz):  " + ", ".join(f"{p} {s:+.2f}" for p, s in ls.items()))
