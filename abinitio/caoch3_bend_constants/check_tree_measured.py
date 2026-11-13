# Ortho CaOCH3 preparation tree: the numbers of Appendix C (app:caoch3_prep) of the paper.
#
# Level energies, line positions and rovibrational detunings: Eq. (caoch3_hamiltonian) ALONE, with the
# constants of the upper block of Table caoch3_constants (measured nu8, A, B; computed zeta8, x88, g88).
# Terms beyond that equation enter none of these numbers. They are evaluated separately:
#   - centrifugal distortion (Crozet et al. 2002, Table 1): only its size is reported
#     (largest shift of a tree level and of a tree line; splitting of the line that coincides with P4);
#   - case (b) spin-rotation + proton hyperfine structure (Namiki et al. 1998), assumed equal in every
#     v8 level (no data for the excited bend levels): used only for the Rabi-frequency limit.
from sympy import Rational as R, sqrt as ssqrt, N as num
from sympy.physics.wigner import wigner_3j, wigner_6j
import numpy as np, itertools

C_MHZ = 29979.2458                       # MHz per cm^-1
# --- Eq. (caoch3_hamiltonian), measured (X~ 2A1) ---
A, B = 5.448303, 0.11626491              # cm^-1, Crozet 2002 Table 1 (B = 3485.53416 MHz, Namiki 1998)
nu8 = 144.0                              # cm^-1, v8 = 1 term value (Kozyryev 2019, Jacox 2003), +-5; central value used
# --- terms beyond Eq. (caoch3_hamiltonian), measured; never added to the energies ---
DN, DNK, DK = 2.6885e-8, 2.3580e-6, 7.03e-5   # cm^-1, Crozet 2002 Table 1 (DK: CaCH3 value); size estimate only
EPS_BC, EPS_AA = 12.4505, 0.0            # MHz, Namiki 1998 / Crozet 2002; Rabi-frequency limit only
AF, TAA = -0.421, 1.070                  # MHz, proton Fermi contact and T_aa, Namiki 1998; Rabi-frequency limit only
# --- Eq. (caoch3_hamiltonian), computed / not measured ---
z8 = 0.926                               # EA-EOM-CCSD/cc-pCVTZ harmonic force field (zeta_from_cc.py)
import json, os
_anh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anharmonic.json')
if os.path.exists(_anh) and not os.environ.get('HARMONIC'):
    _a = json.load(open(_anh)); g_anh, x88 = _a['g_anh'], _a['x88']   # DFT force field + VPT2 (vpt2_bend.py)
else:
    g_anh, x88 = 0.0, 0.0                # not measured; set to zero
g88 = A*z8**2 + g_anh                    # the l^2 term includes A zeta^2 from the vibrational angular momentum
w8 = nu8 - g88                           # so that E(v=1, l=+-1) - E(0) = nu8

S, I = R(1, 2), R(3, 2)

def E(v, ell, N, k, x=x88, g=g88):
    """Eq. (caoch3_hamiltonian) in cm^-1. Every tabulated energy and line position uses this alone."""
    w = nu8 - g
    return w*v + x*v*(v-1) + g*ell**2 + B*N*(N+1) + (A-B)*k**2 - 2*A*z8*k*ell

def E_dist(v, ell, N, k):
    """Centrifugal distortion in cm^-1: a term beyond Eq. (caoch3_hamiltonian), used only to report its size."""
    return -DN*(N*(N+1))**2 - DNK*N*(N+1)*k**2 - DK*k**4

def E_with_dist(v, ell, N, k):
    return E(v, ell, N, k) + E_dist(v, ell, N, k)

levels = {   # name: (n-, n+, N, J, F); branch k = ell - 3
    'G':     (0, 0, 3, R(7, 2), 2),
    'L01':   (0, 1, 3, R(7, 2), 3),
    'L02':   (0, 2, 3, R(7, 2), 4),
    'L11':   (1, 1, 4, R(9, 2), 4),
    'L03_3': (0, 3, 3, R(7, 2), 5),
    'L03_4': (0, 3, 4, R(9, 2), 5),
    'L12':   (1, 2, 4, R(9, 2), 5),
    'L21':   (2, 1, 4, R(9, 2), 5),
}
links = [('P1', 'G', 'L01'), ('P2', 'L01', 'L02'), ('P3', 'L01', 'L11'), ('P4', 'L02', 'L03_3'),
         ('P5', 'L02', 'L03_4'), ('P6', 'L11', 'L12'), ('P7', 'L11', 'L21')]
def qn(n):
    nm, npl, N, J, F = levels[n]; v = nm + npl; ell = npl - nm
    return v, ell, N, ell - 3, J, F
EG = E(*qn('G')[:4])

print(f"constants: nu8={nu8}, w8={w8:.3f}, g88={g88:.3f} (A z^2={A*z8**2:.3f}), x88={x88}, A={A}, B={B}, zeta8={z8}")
print("Energies and line positions: Eq. (caoch3_hamiltonian) alone (no centrifugal distortion, no spin structure).")
print("Levels:")
for n in levels:
    v, ell, N, k, J, F = qn(n)
    print(f"  {n:6s} v={v} l={ell:+d} k={k:+d} N={N} J={J} F={F}  E-EG = {E(v,ell,N,k)-EG:10.5f} cm-1")

# ---------- couplings (pure angular momentum; independent of the constants) ----------
def fM(F, Fu, MF=0): return (-1)**(Fu-MF)*wigner_3j(Fu, 1, F, -MF, 0, MF)
def fI(J, F, Ju, Fu): return (-1)**(Ju+I+F+1)*ssqrt((2*Fu+1)*(2*F+1))*wigner_6j(Ju, Fu, I, F, J, 1)
def fS(N, J, Nu, Ju): return (-1)**(Nu+S+J+1)*ssqrt((2*Ju+1)*(2*J+1))*wigner_6j(Nu, Ju, S, J, N, 1)
def fN(N, k, Nu, ku, q): return (-1)**(Nu-ku)*ssqrt((2*Nu+1)*(2*N+1))*wigner_3j(Nu, 1, N, -ku, q, k)
rows = []
print("\nTree lines:")
for P, lo, up in links:
    v, ell, N, k, J, F = qn(lo); vu, ellu, Nu, ku, Ju, Fu = qn(up); q = ku - k
    nm, npl = levels[lo][:2]; nq = npl if q == 1 else nm
    d = float(num(fM(F, Fu)*fI(J, F, Ju, Fu)*fS(N, J, Nu, Ju)*fN(N, k, Nu, ku, q)*(-q)*ssqrt(nq+1)))
    nu = E(vu, ellu, Nu, ku) - E(v, ell, N, k)
    rows.append((P, lo, up, q, Nu-N, d, nu))
    print(f"  {P}: {lo:6s}->{up:6s} q={q:+d} dN={Nu-N:+d}  d={d:+.4f} mu8  nu={nu:10.5f} cm-1")
nus = {r[0]: r[-1] for r in rows}
pairs = sorted(((abs(nus[a]-nus[b]), a, b) for a, b in itertools.combinations(nus, 2)))
print("closest pairs of tree lines (cm-1, GHz):", [(a, b, round(d, 4), round(d*C_MHZ/1e3, 2)) for d, a, b in pairs[:4]])
chain = 2*x88 + 2*g_anh + 2*A*(1-z8)**2 - 2*B
print(f"q=+1, dN=0 chain (P1, P2, P4): equal spacing 2x+2g_anh+2A(1-z)^2-2B = {chain:.4f} cm-1 = {chain*C_MHZ/1e3:.2f} GHz"
      f"  [anharmonic 2x+2g_anh = {2*x88 + 2*g_anh:+.2f}, rotation and Coriolis 2A(1-z)^2-2B = {2*A*(1-z8)**2 - 2*B:+.2f} cm-1]")

# ---------- rovibrational selectivity, window by window ----------
# windows of App. C: P1 | P2,P3 | P4,P6,P7 | P5 ; populated = levels holding population before or after the window
windows = [['P1'], ['P2', 'P3'], ['P4', 'P6', 'P7'], ['P5']]
pop_sets = [{'G', 'L01'}, {'L01', 'L02', 'L11'}, {'L02', 'L11', 'L03_3', 'L12', 'L21'},
            {'L02', 'L03_4', 'L03_3', 'L12', 'L21'}]
fam = [(v, ell, N, ell-3) for v in range(5) for ell in range(-v, v+1, 2) for N in range(max(abs(ell-3), 0), 8)]
treeq = {qn(n)[:4]: n for n in levels}
def nearest_competing(Efun, line_nu):
    """Per pulse: the nearest other rovibrational line touching a level populated in the pulse's window."""
    out = []
    for win, pops in zip(windows, pop_sets):
        pop = {qn(n)[:4] for n in pops}
        for P in win:
            nu = line_nu[P]
            best = (1e9, None)
            for (v, ell, N, k) in fam:
                for qq in (1, -1):
                    for dn in (-1, 0, 1):
                        u = (v+1, ell+qq, N+dn, k+qq)
                        if u[2] < abs(u[3]) or u[0] > 4: continue
                        if (v, ell, N, k) not in pop and u not in pop: continue
                        if any((v, ell, N, k) == qn(l)[:4] and u == qn(uu)[:4] for p_, l, uu in links if p_ in win): continue
                        nu2 = Efun(*u) - Efun(v, ell, N, k)
                        if abs(nu2 - nu) < best[0]: best = (abs(nu2 - nu), ((v, ell, N, k), u))
            out.append((best[0], P, best[1]))
    return out

print("\nNearest other rovibrational line touching a level populated in the pulse's window:")
worst = nearest_competing(E, nus)
for dmin_, P, pair in worst:
    print(f"  {P}: {dmin_:.4f} cm-1 = {dmin_*C_MHZ/1e3:7.2f} GHz  {pair}  touches {[treeq.get(x) for x in pair]}")
mn = min(worst)
print(f"overall closest: {mn[1]} {mn[0]:.4f} cm-1 = {mn[0]*C_MHZ/1e3:.2f} GHz; (Omega/Delta)^2 at 10 kHz = {(1e4/(mn[0]*C_MHZ*1e6))**2:.1e}")
# tones of the same window must also be far apart
for win in windows:
    for a_, b_ in itertools.combinations(win, 2):
        print(f"  window tones {a_},{b_}: {abs(nus[a_]-nus[b_])*C_MHZ/1e3:.2f} GHz apart")

# all lines touching any tree level (not time-ordered), to list exact/near coincidences
print("\nLines within 1 GHz of a tree line that touch some tree level:")
for P, lo, up, q, dN, d, nu in rows:
    for (v, ell, N, k) in fam:
        for qq in (1, -1):
            for dn in (-1, 0, 1):
                u = (v+1, ell+qq, N+dn, k+qq)
                if u[2] < abs(u[3]) or u[0] > 4: continue
                if (v, ell, N, k) not in treeq and u not in treeq: continue
                if (v, ell, N, k) == qn(lo)[:4] and u == qn(up)[:4]: continue
                nu2 = E(*u) - E(v, ell, N, k)
                if abs(nu2-nu)*C_MHZ < 1000:
                    print(f"  {P}: {(v,ell,N,k)}->{u}  offset {(nu2-nu)*C_MHZ:+9.3f} MHz  touches {[treeq.get(x) for x in ((v,ell,N,k), u)]}")

# ---------- term beyond Eq. (caoch3_hamiltonian): size of centrifugal distortion ----------
# Nothing printed above contains distortion. This block only reports how much it would change.
print("\nCentrifugal distortion (a term beyond Eq. (caoch3_hamiltonian); in none of the numbers above):")
dG = E_dist(*qn('G')[:4])
lev_shift = {n: E_dist(*qn(n)[:4]) - dG for n in levels}
n_max = max(lev_shift, key=lambda n: abs(lev_shift[n]))
print(f"  largest shift of a tree level relative to G: {abs(lev_shift[n_max]):.4f} cm-1 ({n_max})")
line_shift = {P: E_dist(*qn(up)[:4]) - E_dist(*qn(lo)[:4]) for P, lo, up in links}
P_max = max(line_shift, key=lambda P: abs(line_shift[P]))
print(f"  largest shift of a tree line: {abs(line_shift[P_max]):.4f} cm-1 = {abs(line_shift[P_max])*C_MHZ:.0f} MHz ({P_max})")
nus_d = {P: nus[P] + line_shift[P] for P in nus}
pairs_d = sorted(((abs(nus_d[a]-nus_d[b]), a, b) for a, b in itertools.combinations(nus_d, 2)))
mn_d = min(nearest_competing(E_with_dist, nus_d))
print(f"  with distortion the closest tree lines would be {pairs_d[0][1]}, {pairs_d[0][2]} at {pairs_d[0][0]*C_MHZ/1e3:.2f} GHz "
      f"(without: {pairs[0][0]*C_MHZ/1e3:.2f} GHz)")
print(f"  with distortion the nearest competing line would be {mn_d[0]*C_MHZ/1e3:.2f} GHz from {mn_d[1]} "
      f"(without: {mn[0]*C_MHZ/1e3:.2f} GHz from {mn[1]})")
lo4, up4 = (2, 2, 4, -1), (3, 3, 4, 0)      # the N = 4 counterpart of P4; its upper level is L30
off0 = (E(*up4) - E(*lo4)) - nus['P4']
offd = (E_with_dist(*up4) - E_with_dist(*lo4)) - nus_d['P4']
print(f"  line {lo4}->{up4} (touches L03_4 = L30): offset from P4 {off0*C_MHZ:+.3f} MHz in Eq. (caoch3_hamiltonian); "
      f"distortion alone splits it by {offd*C_MHZ:+.3f} MHz (= 8 D_NK)")

# ---------- spin structure: case (b), spin-rotation + ortho proton hyperfine ----------
# H = eps_bc N.S + (eps_aa - eps_bc) k S_z' + a_F I.S + (T_aa/2)(3 S_z' I_z' - S.I),
# where ' marks the molecule frame. Within ortho spin (I = 3/2) the three proton tensors average to an
# axial tensor (zz = T_aa, xx = yy = -T_aa/2), and sum_i I_i -> I for the contact term.
# Molecule-frame components: X'_q = sum_p D^j_{pq}(Omega) X_p (inverse of mu_Z = sum_q D^1_{0q}* mu_q).
def sph(j, m, p):
    """(m', c): spherical component p of a spin-j vector operator, Edmonds phases."""
    if p == 0: return [(m, float(m))]
    if p == 1: return [] if m + 1 > j else [(m+1, -float(ssqrt(j*(j+1)-m*(m+1)))/np.sqrt(2))]
    return [] if m - 1 < -j else [(m-1, float(ssqrt(j*(j+1)-m*(m-1)))/np.sqrt(2))]

def Dnc(N, k, rank, p, MN):
    """(MN', <N k MN'| D^rank_{p0}(Omega) |N k MN>) for fixed N, k (D not conjugated)."""
    MNp = MN - p
    if abs(MNp) > N: return None
    val = (-1)**p*(-1)**(MNp-k)*(2*N+1)*wigner_3j(N, rank, N, -MNp, -p, MN)*wigner_3j(N, rank, N, -k, 0, k)
    return MNp, float(num(val))

def spin_levels(N, k, eps_bc=EPS_BC, eps_aa=EPS_AA, aF=AF, taa=TAA, MF=0):
    basis = [(MN, MS, MI) for MN in range(-N, N+1) for MS in (R(1, 2), -R(1, 2))
             for MI in (R(3, 2), R(1, 2), -R(1, 2), -R(3, 2)) if MN + MS + MI == MF]
    ix = {b: i for i, b in enumerate(basis)}; n = len(basis)
    H = np.zeros((n, n)); J2 = np.zeros((n, n)); F2 = np.zeros((n, n))
    def add(M, i, key, c):
        j = ix.get(key)
        if j is not None: M[j, i] += c
    for (MN, MS, MI), i in ix.items():
        for p in (-1, 0, 1):
            sgn = (-1)**p
            for MNp, cN in sph(N, MN, p):
                for MSp, cS in sph(S, MS, -p):
                    add(H, i, (MNp, MSp, MI), eps_bc*sgn*cN*cS)               # eps_bc N.S
                    add(J2, i, (MNp, MSp, MI), 2*sgn*cN*cS); add(F2, i, (MNp, MSp, MI), 2*sgn*cN*cS)
            for MIp, cI in sph(I, MI, p):
                for MSp, cS in sph(S, MS, -p):
                    add(H, i, (MN, MSp, MIp), (aF - taa/2)*sgn*cI*cS)          # a_F I.S - (T_aa/2) S.I
                    add(F2, i, (MN, MSp, MIp), 2*sgn*cI*cS)
            for MIp, cI in sph(I, MI, p):
                for MNp, cN in sph(N, MN, -p):
                    add(F2, i, (MNp, MS, MIp), 2*sgn*cI*cN)
            # (eps_aa - eps_bc) k S_z',  S_z' = sum_p D^1_{p0} S_p
            r = Dnc(N, k, 1, p, MN)
            if r is not None and r[1] != 0:
                for MSp, cS in sph(S, MS, p):
                    add(H, i, (r[0], MSp, MI), (eps_aa - eps_bc)*k*r[1]*cS)
        # (T_aa/2) sqrt6 sum_p D^2_{p0} [S x I]^2_p
        for ps in (-1, 0, 1):
            for pi in (-1, 0, 1):
                p = ps + pi
                cg = float(num((-1)**p*np.sqrt(5)*wigner_3j(1, 1, 2, ps, pi, -p)))   # <1 ps 1 pi | 2 p>
                if cg == 0: continue
                r = Dnc(N, k, 2, p, MN)
                if r is None or r[1] == 0: continue
                for MSp, cS in sph(S, MS, ps):
                    for MIp, cI in sph(I, MI, pi):
                        add(H, i, (r[0], MSp, MIp), (taa/2)*np.sqrt(6)*r[1]*cg*cS*cI)
        J2[i, i] += N*(N+1) + 0.75; F2[i, i] += N*(N+1) + 0.75 + 3.75
    assert np.allclose(H, H.T, atol=1e-9), np.abs(H - H.T).max()
    w, V = np.linalg.eigh(H)
    labs = []
    for c in range(n):
        Jv = V[:, c] @ J2 @ V[:, c]; Fv = V[:, c] @ F2 @ V[:, c]
        labs.append((round((-1 + np.sqrt(1 + 4*Jv)))/2, round((-1 + np.sqrt(1 + 4*Fv))/2), w[c]))
    return w, V, basis, labs

# check 1: T_aa = 0, a_F = 0 -> analytic spin-rotation [eps_aa k^2 + eps_bc (N(N+1)-k^2)]/(N(N+1)) * (J(J+1)-N(N+1)-3/4)/2
for (N, k) in [(3, 3), (3, 1), (4, 2)]:
    w, V, b, labs = spin_levels(N, k, aF=0, taa=0)
    for J in (N + 0.5, N - 0.5):
        ana = (EPS_AA*k**2 + EPS_BC*(N*(N+1) - k**2))/(N*(N+1))*(J*(J+1) - N*(N+1) - 0.75)/2
        got = sorted({round(e, 4) for (JJ, F, e) in labs if JJ == J})
        assert len(got) == 1 and abs(got[0] - ana) < 1e-3, (N, k, J, got, ana)
# check 2: a_F only, J kept pure by large eps -> first-order contact shift a_F <I.S> projected on J
for (N, k) in [(3, 3), (4, 2)]:
    w, V, b, labs = spin_levels(N, k, eps_bc=1e6, aF=1.0, taa=0)
    for (J, F, e) in labs:
        cJ = (J*(J+1) + 0.75 - N*(N+1))/(2*J*(J+1))
        eSR = (EPS_AA*k**2 + 1e6*(N*(N+1) - k**2))/(N*(N+1))*(J*(J+1) - N*(N+1) - 0.75)/2
        ana = cJ*(F*(F+1) - J*(J+1) - 3.75)/2
        assert abs((e - eSR) - ana) < 2e-3, (N, k, J, F, e - eSR, ana)
print("\nspin-structure checks passed (spin-rotation exact; contact term = first-order case (b) formula)")
print("\nSpin structure (MHz), ortho, M_F = 0, v=0 constants used for every level:")
spin = {}
for nm in levels:
    v, ell, N, k, J, F = qn(nm)
    key = (N, abs(k))
    if key not in spin:
        spin[key] = spin_levels(N, abs(k))
        w, V, b, labs = spin[key]
        print(f"  N={N} |k|={abs(k)}: " + ", ".join(f"({J},{F}) {e:+.3f}" for J, F, e in sorted(labs, key=lambda t: t[2])))

own = {}
for nm in levels:
    v, ell, N, k, J, F = qn(nm)
    own[nm] = next(e for JJ, FF, e in spin[(N, abs(k))][3] if JJ == float(J) and FF == F)
nm_max = max(own, key=lambda n: abs(own[n]))
print(f"largest spin shift of a tree level (its own J, F component): {abs(own[nm_max]):.1f} MHz = "
      f"{abs(own[nm_max])/C_MHZ:.4f} cm-1 ({nm_max}); not included in the energies above")

# ---------- hyperfine-resolved selectivity of each pulse ----------
def rot_me(N, k, M, Nu, ku, Mu, q):   # <Nu ku Mu| D^1_{0q}* |N k M>, Eq. (caoch3_rotation_me)
    return float(num((-1)**(Mu-ku)*ssqrt((2*Nu+1)*(2*N+1))*wigner_3j(Nu, 1, N, -Mu, 0, M)*wigner_3j(Nu, 1, N, -ku, q, k)))
def comp(nm):
    v, ell, N, k, J, F = qn(nm)
    w, V, b, labs = spin[(N, abs(k))]
    return N, k, w, V, b, labs
def coupling(lo, up, ilo, iup):
    N, k, wl, Vl, bl, _ = comp(lo); Nu, ku, wu, Vu, bu, _ = comp(up)
    q = ku - k; nm, npl = levels[lo][:2]; nq = npl if q == 1 else nm
    Vq = -q*np.sqrt(nq + 1)
    iu = {x: i for i, x in enumerate(bu)}
    tot = 0.0
    for a, (MN, MS, MI) in enumerate(bl):
        j = iu.get((MN, MS, MI))
        if j is None or Vl[a, ilo] == 0: continue
        tot += Vu[j, iup]*Vl[a, ilo]*rot_me(N, k, MN, Nu, ku, MN, q)
    return tot*Vq
def find(nm):
    v, ell, N, k, J, F = qn(nm); _, _, w, V, b, labs = comp(nm)
    return next(i for i, (JJ, FF, e) in enumerate(labs) if JJ == float(J) and FF == F)
print("\nHyperfine-resolved check (spin constants of v=0 in every level):")
worst_lim = []
for P, lo, up, q, dN, d, nu in rows:
    il, iu = find(lo), find(up)
    _, _, wl, _, _, labl = comp(lo); _, _, wu, _, _, labu = comp(up)
    d0 = coupling(lo, up, il, iu)
    f0 = wu[iu] - wl[il]
    comps = []
    for jl in range(len(wl)):           # from the populated lower state to other upper components, and back
        for ju in range(len(wu)):
            if (jl, ju) == (il, iu) or (jl != il and ju != iu): continue
            c = coupling(lo, up, jl, ju)
            if abs(c) < 1e-6: continue
            comps.append((abs((wu[ju] - wl[jl]) - f0), abs(c/d0), labl[jl][:2], labu[ju][:2]))
    comps.sort()
    dmin, ratio, a, b_ = comps[0]
    # leakage ~ (ratio * Omega / Delta)^2 ; Omega such that every leakage <= 1e-3
    lim = min(dm/(r*np.sqrt(1e3)) for dm, r, _, _ in comps)
    worst_lim.append(lim)
    print(f"  {P}: |d|={abs(d0):.4f} (pure {abs(d):.4f}); nearest competing component {a}->{b_}: {dmin*1e3:7.1f} kHz away, "
          f"strength ratio {ratio:.2f};  Omega/2pi <= {lim*1e3:6.1f} kHz for leakage <= 1e-3")
Om = min(worst_lim)
print(f"common Omega/2pi <= {Om*1e3:.1f} kHz -> pi pulse {0.5/Om:.1f} us; sequence of 7 pulses (at most) {7*0.5/Om:.0f} us")
