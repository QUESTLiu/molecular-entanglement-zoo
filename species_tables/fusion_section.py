"""
Multiplicity-two coupling (fusion) spaces: analysis, figures, and the
dodecahedrane / 13C60 examples.  Writes

    fusion_section.tex       (\\input by species_tables.tex)
    fig_fusion_qubit.pdf    (entanglement across the fusion Bloch sphere)
    fusion_data.pkl          (all numbers used in the section)

    python fusion_section.py
"""
import itertools
import os
import pickle
import numpy as np
import mpmath as mp

import pointgroups as pg
from species import tensors_finite, phase_fix
from entanglement import classify, lu_fidelity, RNG

HERE = os.path.dirname(os.path.abspath(__file__))
groups = {G.name: G for G in pg.cubic_groups()}
Ih, Tg = groups["Ih"], groups["T"]
DATA = {}


def idx(G, lab):
    return G.labels.index(lab)


# ============================================================================
# 1. permutation structure of the N = 2 spaces
# ============================================================================
def powchar(G, i, k):
    out = []
    for g in range(G.order):
        h = 0
        for _ in range(k):
            h = G.mul[g, h]
        out.append(G.chars[i][h])
    return np.array(out)


def mult(G, chi, j):
    return int(round((np.sum(chi * G.chars[j].conj()) / G.order).real))


def dec(G, chi):
    return {G.labels[j]: mult(G, chi, j) for j in range(len(G.labels)) if mult(G, chi, j)}


def sym_powers(G, lab, mol):
    i, m = idx(G, lab), idx(G, mol)
    c1, c2, c3 = (powchar(G, i, k) for k in (1, 2, 3))
    s2, a2 = (c1 ** 2 + c2) / 2, (c1 ** 2 - c2) / 2
    s3 = (c1 ** 3 + 3 * c2 * c1 + 2 * c3) / 6
    a3 = (c1 ** 3 - 3 * c2 * c1 + 2 * c3) / 6
    return dict(sym2=dec(G, s2), alt2=dec(G, a2), sym3=mult(G, s3, m), alt3=mult(G, a3, m),
                mixed=mult(G, c1 ** 3, m) - mult(G, s3, m) - mult(G, a3, m))


DATA["sym"] = {"H": sym_powers(Ih, "H_{g}", "A_{g}"), "G": sym_powers(Ih, "G_{g}", "A_{g}"),
               "T": sym_powers(Tg, "T", "A")}


# ============================================================================
# 2. fusion spaces, natural bases
# ============================================================================
def onb(vecs):
    V = np.array([v.ravel() for v in vecs]).T
    q, _ = np.linalg.qr(V)
    return [q[:, k].reshape(vecs[0].shape) for k in range(q.shape[1])]


def fusion(G, labs, mol):
    return onb(tensors_finite(G, *[idx(G, l) for l in labs], idx(G, mol)))


F_HHH = fusion(Ih, ["H_{g}"] * 3, "A_{g}")
F_GHH = fusion(Ih, ["G_{g}", "H_{g}", "H_{g}"], "A_{g}")
F_TTT = fusion(Tg, ["T"] * 3, "A")


def vector_operators(G, lab, t1):
    """invariant tensors X[i,j,a] of D x conj(D) x conj(D^{T1}) -> all vector operators on the carrier"""
    i, v = idx(G, lab), idx(G, t1)
    D, R = G.D[i], G.D[v]
    d = D.shape[1]
    P = np.zeros((d * d * 3,) * 2, complex)
    for g in range(G.order):
        P += np.kron(np.kron(D[g], D[g].conj()), R[g].conj())
    P /= G.order
    w, U = np.linalg.eigh((P + P.conj().T) / 2)
    ks = np.where(w > 0.5)[0]
    if len(ks) > 1:
        # eigh returns an arbitrary basis of a degenerate space, which a phase alone need not make
        # Hermitian (it changes with rounding noise). The space is closed under X -> X^dagger, since
        # the vector irrep is real, so take a Hermitian basis of it from the parts (X +- X^dagger).
        V = [U[:, k].reshape(d, d, 3).transpose(2, 0, 1) for k in ks]
        basis = []
        for X in V:
            Xd = np.conj(np.transpose(X, (0, 2, 1)))
            for C in ((X + Xd) / 2, (X - Xd) / 2j):
                for B in basis:
                    C = C - B * np.vdot(B, C).real      # real for Hermitian tensors
                if np.linalg.norm(C) > 1e-6 and len(basis) < len(V):
                    basis.append(C / np.linalg.norm(C))
        assert len(basis) == len(V)
        for X in basis:
            assert all(np.allclose(X[a], X[a].conj().T, atol=1e-8) for a in range(3))
        return basis
    out = []
    for k in ks:
        X = U[:, k].reshape(d, d, 3).transpose(2, 0, 1)
        # make Hermitian
        ph = 1.0
        for a in range(3):
            j = np.unravel_index(np.argmax(abs(X[a])), X[a].shape)
            if abs(X[a][j]) > 1e-8 and abs(X[a][j[::-1]]) > 1e-8:
                ph = np.sqrt(X[a][j[::-1]].conj() / X[a][j])
                break
        X = X * ph
        assert all(np.allclose(X[a], X[a].conj().T, atol=1e-8) for a in range(3))
        out.append(X)
    return out


def su2_normalise(X):
    """scale/sign so that [X_x, X_y] = i X_z (exact for angular-momentum operators)"""
    C = X[0] @ X[1] - X[1] @ X[0]
    j = np.unravel_index(np.argmax(abs(X[2])), X[2].shape)
    c = (1j * X[2][j] / C[j]).real
    Y = c * X
    is_su2 = all(np.allclose(Y[a] @ Y[b] - Y[b] @ Y[a], 1j * Y[c_], atol=1e-8)
                 for a, b, c_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    return Y, is_su2


XH = vector_operators(Ih, "H_{g}", "T_{1g}")
XG = vector_operators(Ih, "G_{g}", "T_{1g}")
assert len(XH) == 1 and len(XG) == 1
LH, H_su2 = su2_normalise(XH[0])
LG, G_su2 = su2_normalise(XG[0])
if not G_su2:  # no su(2) structure on G: normalise to the same Frobenius norm per component as J=3
    LG = XG[0] / np.sqrt(np.trace(XG[0][0] @ XG[0][0]).real) * np.sqrt(np.trace(LH[0] @ LH[0]).real)
casH = sum(LH[a] @ LH[a] for a in range(3))
DATA["vecop"] = dict(H_su2=H_su2, H_casimir=float(np.trace(casH).real / 5), G_su2=G_su2)
# T group: angular momentum L_a = -i eps_a in the Cartesian basis
LT = np.zeros((3, 3, 3), complex)
for a, b, c in itertools.permutations(range(3)):
    LT[a, b, c] = -1j * np.linalg.det(np.eye(3)[[a, b, c]])
DATA["vecop"]["T_count"] = len(vector_operators(Tg, "T", "T"))


def pair_op(Ls, p, q):
    """sum_a L^p_a L^q_a on the three-carrier space"""
    dims = [L.shape[1] for L in Ls]
    out = 0
    for a in range(3):
        ops = [np.eye(d) for d in dims]
        ops[p], ops[q] = Ls[p][a], Ls[q][a]
        out = out + np.kron(np.kron(ops[0], ops[1]), ops[2])
    return out


def restrict(O, basis):
    B = np.array([b.ravel() for b in basis]).T
    return B.conj().T @ O @ B


def eig_states(M, basis):
    w, V = np.linalg.eigh((M + M.conj().T) / 2)
    return w, [phase_fix(sum(V[k, j] * basis[k] for k in range(len(basis)))) for j in range(len(w))]


# ---- (H,H,H): SO(3) 3j(222) direction and the vector-coupling eigenbasis
QB = []
for (a, b) in [(0, 1), (0, 2), (1, 2)]:
    M = np.zeros((3, 3)); M[a, b] = M[b, a] = 1 / np.sqrt(2); QB.append(M)
QB += [np.diag([1, -1, 0]) / np.sqrt(2), np.diag([1, 1, -2]) / np.sqrt(6)]
DQ = np.zeros((Ih.order, 5, 5))
for g, M in enumerate(Ih.elems):
    R = M * np.linalg.det(M)
    for j, Qj in enumerate(QB):
        Y = R @ Qj @ R.T
        DQ[g, :, j] = [np.sum(Qi * Y) for Qi in QB]
iH = idx(Ih, "H_{g}")
X0 = RNG.normal(size=(5, 5)) + 1j * RNG.normal(size=(5, 5))
U = sum(Ih.D[iH][g] @ X0 @ DQ[g].T for g in range(Ih.order))
U /= np.sqrt(np.trace(U.conj().T @ U).real / 5)
t3j = np.einsum("aij,bjk,cki->abc", np.array(QB), np.array(QB), np.array(QB))
C_3j = phase_fix(np.einsum("ia,jb,kc,abc->ijk", U, U, U, t3j))
w3j = float(np.linalg.norm([np.vdot(b, C_3j) for b in F_HHH]))

Ls = [LH, LH, LH]
M_rn, M_rv, M_nv = (restrict(pair_op(Ls, p, q), F_HHH) for p, q in ((0, 1), (0, 2), (1, 2)))
wH, (C_K2, C_K4) = eig_states(M_rn, F_HHH)
same_pairs = np.allclose(M_rn, M_rv, atol=1e-9) and np.allclose(M_rn, M_nv, atol=1e-9)
fid_K2_3j = abs(np.vdot(C_K2, C_3j))
DATA["HHH"] = dict(w3j=w3j, LdotL_eigs=[float(x) for x in wH], same_for_all_pairs=bool(same_pairs),
                   overlap_K2_with_3j=float(fid_K2_3j))
C_ico = C_K4

# ---- (G,H,H): S/A basis and the three pairwise vector couplings
P = lambda X: np.transpose(X, (0, 2, 1))
C_S = onb([phase_fix(x + P(x)) for x in F_GHH if np.linalg.norm(x + P(x)) > 1e-6][:1])[0]
C_A = onb([phase_fix(x - P(x)) for x in F_GHH if np.linalg.norm(x - P(x)) > 1e-6][:1])[0]
basisGHH = [C_S, C_A]
Ls = [LG, LH, LH]
MG = {k: restrict(pair_op(Ls, p, q), basisGHH) for k, (p, q) in
      {"rn": (0, 1), "rv": (0, 2), "nv": (1, 2)}.items()}
eigG = {k: eig_states(M, basisGHH) for k, M in MG.items()}


def comm_norm(A, B):
    return float(np.linalg.norm(A @ B - B @ A))


DATA["GHH"] = dict(
    M={k: np.round(v, 6).tolist() for k, v in MG.items()},
    eigs={k: [float(x) for x in v[0]] for k, v in eigG.items()},
    nv_diag_in_SA=bool(abs(MG["nv"][0, 1]) < 1e-9),
    comm_rn_rv=comm_norm(MG["rn"], MG["rv"]), comm_rn_nv=comm_norm(MG["rn"], MG["nv"]),
    # overlap between Coriolis (r-v) eigenstates and spin-rotation (r-n) eigenstates
    overlap_rv_rn=float(abs(np.vdot(eigG["rv"][1][0], eigG["rn"][1][0]))),
)

# ---- T group: (T,T,T) with L = -i eps
Cp = np.zeros((3, 3, 3), complex); Cm = np.zeros((3, 3, 3), complex)
for p_, s in zip(itertools.permutations(range(3)), [1, -1, -1, 1, 1, -1]):
    (Cp if s > 0 else Cm)[p_] = 1 / np.sqrt(3)
eps = (Cp - Cm) / np.sqrt(2)
aeps = (Cp + Cm) / np.sqrt(2)
MT = restrict(pair_op([LT, LT, LT], 0, 1), [aeps, eps])
DATA["TTT"] = dict(M=np.round(MT, 6).tolist(), in_span=[float(np.linalg.norm([np.vdot(b, x) for b in F_TTT]))
                                                         for x in (Cp, Cm)])


# ============================================================================
# 3. LU invariants over the fusion Bloch sphere
# ============================================================================
def negativity(C, keep):
    d = C.shape
    out = [p for p in range(3) if p not in keep][0]
    X = np.moveaxis(C, out, 2).reshape(d[keep[0]] * d[keep[1]], d[out])
    rho = X @ X.conj().T
    a, b = d[keep[0]], d[keep[1]]
    rt = rho.reshape(a, b, a, b).transpose(0, 3, 2, 1).reshape(a * b, a * b)
    ev = np.linalg.eigvalsh((rt + rt.conj().T) / 2)
    return float(-ev[ev < 0].sum())


def geometric(C, starts=30, iters=200):
    """Lambda^2 = max |<abc|C>|^2 over product states (higher-order power method)"""
    best = 0.0
    for s in range(starts):
        vs = [RNG.normal(size=d) + 1j * RNG.normal(size=d) for d in C.shape]
        vs = [x / np.linalg.norm(x) for x in vs]
        for it in range(iters):
            vs[0] = np.einsum("abc,b,c->a", C, vs[1].conj(), vs[2].conj()); vs[0] /= np.linalg.norm(vs[0])
            vs[1] = np.einsum("abc,a,c->b", C, vs[0].conj(), vs[2].conj()); vs[1] /= np.linalg.norm(vs[1])
            vs[2] = np.einsum("abc,a,b->c", C, vs[0].conj(), vs[1].conj()); vs[2] /= np.linalg.norm(vs[2])
        best = max(best, abs(np.einsum("abc,a,b,c->", C, *[v.conj() for v in vs])) ** 2)
    return float(best)


def invariants(C):
    C = C / np.linalg.norm(C)
    return dict(N_rn=negativity(C, (0, 1)), N_rv=negativity(C, (0, 2)), N_nv=negativity(C, (1, 2)),
                L2=geometric(C))


NTH, NPH = 19, 36
TH = np.linspace(0, np.pi, NTH)
PH = np.linspace(0, 2 * np.pi, NPH, endpoint=False)


def scan(C1, C2):
    grid = {k: np.zeros((NTH, NPH)) for k in ("N_rn", "N_rv", "N_nv", "L2")}
    for i, t in enumerate(TH):
        for j, p in enumerate(PH):
            if (i == 0 or i == NTH - 1) and j > 0:
                for k in grid:
                    grid[k][i, j] = grid[k][i, 0]
                continue
            inv = invariants(np.cos(t / 2) * C1 + np.exp(1j * p) * np.sin(t / 2) * C2)
            for k in grid:
                grid[k][i, j] = inv[k]
    return grid


families = {
    "T": dict(title=r"$T$: $(T,T,T)\to A$", C1=aeps, C2=eps, l1=r"$|\epsilon|$", l2=r"$\epsilon$"),
    "HHH": dict(title=r"$I_{\rm h}$: $(H_g,H_g,H_g)\to A_g$", C1=C_K2, C2=C_K4, l1=r"$K{=}2$ (3j)", l2=r"$K{=}4$"),
    "GHH": dict(title=r"$I_{\rm h}$: $(G_g,H_g,H_g)\to A_g$", C1=C_S, C2=C_A, l1="S", l2="A"),
}
cache = os.path.join(HERE, "fusion_scan.pkl")
if os.path.exists(cache):
    scans = pickle.load(open(cache, "rb"))
else:
    scans = {}
    for k, f in families.items():
        print("scanning", k, flush=True)
        scans[k] = scan(f["C1"], f["C2"])
    pickle.dump(scans, open(cache, "wb"))
DATA["ranges"] = {k: {q: (float(v.min()), float(v.max())) for q, v in s.items()} for k, s in scans.items()}

# distinguished points
points = {
    "T": [("GHZ (even cyclic)", Cp), ("GHZ (odd cyclic)", Cm), (r"$|\epsilon|$", aeps), (r"$\epsilon$", eps),
          (r"$(|\epsilon|+i\epsilon)/\sqrt2$", (aeps + 1j * eps) / np.sqrt(2))],
    "HHH": [(r"$K{=}2$ (= SO(3) $3j$)", C_K2), (r"$K{=}4$ (icosahedral)", C_K4),
            (r"$(K2+K4)/\sqrt2$", (C_K2 + C_K4) / np.sqrt(2)), (r"$(K2+iK4)/\sqrt2$", (C_K2 + 1j * C_K4) / np.sqrt(2))],
    "GHH": [("S", C_S), ("A", C_A), (r"$(S+A)/\sqrt2$", (C_S + C_A) / np.sqrt(2)),
            (r"$(S+iA)/\sqrt2$", (C_S + 1j * C_A) / np.sqrt(2)),
            ("Coriolis eigenstate 1", eigG["rv"][1][0]), ("Coriolis eigenstate 2", eigG["rv"][1][1])],
}
DATA["points"] = {}
for k, lst in points.items():
    DATA["points"][k] = []
    for name, C in lst:
        C = C / np.linalg.norm(C)
        c = classify(C)
        inv = invariants(C)
        inv["L2"] = geometric(C, starts=120, iters=300)
        DATA["points"][k].append(dict(name=name, rank=c["rank"], cls=c["cls"], **inv))
        print(k, name, c, {q: round(v, 4) for q, v in inv.items()}, flush=True)


# ============================================================================
# 4. figure
# ============================================================================
def sphere_coords(C, C1, C2):
    """(theta/pi, phi/pi) of C = cos(t/2) C1 + e^{i p} sin(t/2) C2"""
    a, b = np.vdot(C1, C), np.vdot(C2, C)
    t = 2 * np.arccos(min(1.0, abs(a)))
    p = (np.angle(b) - np.angle(a)) % (2 * np.pi) if abs(b) > 1e-9 and abs(a) > 1e-9 else 0.0
    return t / np.pi, p / np.pi


MARKERS = {
    "T": [(*sphere_coords(Cp, aeps, eps), "o"), (*sphere_coords(Cm, aeps, eps), "o")],
    "GHH": [(*sphere_coords(v, C_S, C_A), "o") for v in eigG["rv"][1]]
    + [(*sphere_coords(v, C_S, C_A), "^") for v in eigG["rn"][1]],
}
GHZ_FILE = os.path.join(HERE, "ghz_points.pkl")
DATA["ghz"] = {}
if os.path.exists(GHZ_FILE):  # written by ghz_points.py (needs fusion_data.pkl from a previous run)
    ghz = pickle.load(open(GHZ_FILE, "rb"))
    for k, (C1, C2) in (("HHH", (C_K2, C_K4)), ("GHH", (C_S, C_A))):
        DATA["ghz"][k] = []
        for sol in ghz.get(k, []):
            t, p = sol["x"] * np.pi
            C = np.cos(t / 2) * C1 + np.exp(1j * p) * np.sin(t / 2) * C2
            C /= np.linalg.norm(C)
            DATA["ghz"][k].append(dict(theta=sol["x"][0], phi=sol["x"][1], residual=sol["residual"], L2=sol["L2"],
                                       fid=sol["fid_ghz5"], N_rn=negativity(C, (0, 1)), N_rv=negativity(C, (0, 2)),
                                       N_nv=negativity(C, (1, 2)), pop1=float(np.cos(t / 2) ** 2)))
            MARKERS.setdefault(k, []).append((sol["x"][0], sol["x"][1], "*"))
DATA["markers"] = MARKERS


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seqblue", ramp)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                         "xtick.color": "#555555", "ytick.color": "#555555", "mathtext.fontset": "cm"})
    cols = [("EG", r"$E_G=-\log_2\Lambda^2$"), ("N_rn", r"negativity $\mathcal{N}(\rho_{\rm rn})$"),
            ("N_nv", r"negativity $\mathcal{N}(\rho_{\rm nv})$")]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.3), constrained_layout=True)
    for r, (k, f) in enumerate(families.items()):
        s = scans[k]
        for c, (q, lab) in enumerate(cols):
            ax = axes[r, c]
            Z = -np.log2(s["L2"]) if q == "EG" else s[q]
            Zp = np.hstack([Z, Z[:, :1]])
            php = np.append(PH, 2 * np.pi)
            im = ax.pcolormesh(php / np.pi, TH / np.pi, Zp, cmap=cmap, shading="gouraud", rasterized=True)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(labelsize=6.5)
            ax.set_ylim(1, 0)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=6.5)
            ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"], fontsize=6.5)
            # theta = 0 (top) is C1, theta = pi (bottom) is C2
            ax.text(0.03, 0.96, f["l1"], transform=ax.transAxes, fontsize=6.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), va="top")
            ax.text(0.03, 0.04, f["l2"], transform=ax.transAxes, fontsize=6.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), va="bottom")
            for (tt, pp, mk) in MARKERS.get(k, []):
                for pshift in ((pp, pp + 2) if pp < 1e-9 else (pp,)):
                    ax.plot([pshift], [tt], marker=mk, ms=4.5, mfc="white", mec="#222222", mew=0.8,
                            clip_on=False, ls="none")
            if r == 0:
                ax.set_title(lab, fontsize=8)
            if c == 0:
                ax.set_ylabel(f["title"] + "\n" + r"$\theta$", fontsize=7.5)
            if r == 2:
                ax.set_xlabel(r"$\varphi$", fontsize=7.5)
    fig.savefig(os.path.join(HERE, "fig_fusion_qubit.pdf"))
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "build", "fig_fusion_qubit.png"), dpi=110)


make_figure()


# ============================================================================
# 5. molecules
# ============================================================================
phi = (1 + 5 ** 0.5) / 2


def cyc(v):
    return [v, (v[1], v[2], v[0]), (v[2], v[0], v[1])]


def signs(v):
    return {tuple(s[i] * v[i] for i in range(3)) for s in itertools.product([1, -1], repeat=3)}


def polyhedron(bases, cube=False):
    s = set(itertools.product([1, -1], repeat=3)) if cube else set()
    for b in bases:
        for c in cyc(b):
            s |= signs(c)
    return np.array(sorted(s), float)


dodeca = polyhedron([(0, phi, 1 / phi)], cube=True)       # 20 vertices
c60 = polyhedron([(0, 1, 3 * phi), (1, 2 + phi, 2 * phi), (phi, 2, phi ** 3)])  # 60 vertices


def perm_of(pts, M):
    Y = pts @ M.T
    p = [int(np.argmin(np.linalg.norm(pts - y, axis=1))) for y in Y]
    assert np.allclose(pts[p], Y, atol=1e-8)
    return p


def cycle_type(p):
    seen, lens = set(), []
    for i in range(len(p)):
        if i not in seen:
            L, j = 0, i
            while j not in seen:
                seen.add(j); j = p[j]; L += 1
            lens.append(L)
    return sorted(lens)


def class_name(M):
    det = np.linalg.det(M)
    R = M * det
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if det > 0:
        return {0: "E", 72: "C_5", 144: "C_5^2", 120: "C_3", 180: "C_2"}[int(round(ang))]
    return {0: "i", 72: "S_{10}^3", 144: "S_{10}", 120: "S_6", 180: r"\sigma"}[int(round(ang))]


CLASS_ORDER = ["E", "C_5", "C_5^2", "C_3", "C_2", "i", "S_{10}^3", "S_{10}", "S_6", r"\sigma"]
mp.mp.dps = 50
chiI = {lab: Ih.chars[k] for k, lab in enumerate(Ih.labels)}


def exact_char(x):
    """I_h character values are a + b*phi with small integers a, b"""
    for a in range(-5, 6):
        for b in range(-2, 3):
            if abs(x - (a + b * phi)) < 1e-8:
                return mp.mpf(a) + b * mp.phi
    raise ValueError(x)


def decompose_exact(chi):
    out = {}
    for lab, ch in chiI.items():
        s = mp.mpf(0)
        for g in range(Ih.order):
            s += chi[g] * exact_char(ch[g].real)
        k = int(mp.nint(s / Ih.order))
        if k:
            out[lab] = k
    return out


def rot_char(J, sigma, M):
    det = np.linalg.det(M)
    R = M * det
    w = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    chi = (2 * J + 1) if abs(w) < 1e-9 else np.sin((2 * J + 1) * w / 2) / np.sin(w / 2)
    return chi * (sigma if det < 0 else 1)


DATA["rot"] = {}
for J in range(0, 7):
    for sig, s in ((1, "g"), (-1, "u")):
        chi = [rot_char(J, sig, M) for M in Ih.elems]
        DATA["rot"][(J, s)] = {k: v for k, v in decompose_exact([mp.mpf(float(x)) for x in chi]).items()}


def molecule(name, shells, masses_radii, spin_shell):
    """shells: list of (points) sharing one permutation action; spin_shell: index of permuted spin-1/2 nuclei"""
    pts_spin = shells[spin_shell]
    rows = {}
    sgn, spin = [], []
    for M in Ih.elems:
        p = perm_of(pts_spin, M)
        ct = cycle_type(p)
        cyc_n = len(ct)
        par = (-1) ** (len(p) - cyc_n)
        sgn.append(par)
        spin.append(mp.mpf(2) ** cyc_n)
        cn = class_name(M)
        if cn not in rows:
            from collections import Counter
            rows[cn] = dict(cycles=dict(Counter(ct)), ncyc=cyc_n, parity=par)
    mol = [lab for k, lab in enumerate(Ih.labels) if Ih.dims[k] == 1 and np.allclose(chiI[lab].real, sgn)][0]
    nat = sum(len(s) for s in shells)
    vib = []
    for M in Ih.elems:
        fixed = sum(sum(1 for i, j in enumerate(perm_of(s, M)) if i == j) for s in shells)
        tr, det = np.trace(M), np.linalg.det(M)
        vib.append(mp.mpf(float(fixed * tr - tr - det * tr)))
    I = 2 / 3 * sum(m * r ** 2 for m, r in masses_radii)       # amu A^2
    B = 16.8576 / I                                             # cm^-1
    return dict(name=name, natoms=nat, nspin=len(pts_spin), classes=rows, mol=mol,
                spin=decompose_exact(spin), vib=decompose_exact(vib), I=I, B=B)


# dodecahedrane: C on a dodecahedron with C-C = 1.54 A, C-H = 1.10 A radially outward
edge_unit = 2 / phi                      # edge of the (+-1,+-1,+-1)/(0,phi,1/phi) dodecahedron
rC = np.linalg.norm(dodeca[0]) * 1.54 / edge_unit
rH = rC + 1.10
mol_dod = molecule("C20H20", [dodeca, dodeca], [(12.0, rC)] * 20 + [(1.00783, rH)] * 20, 1)
# 13C60: cage radius 3.55 A
mol_c60 = molecule("13C60", [c60], [(13.00335, 3.55)] * 60, 0)
mol_c60["B12"] = mol_c60["B"] * 13.00335 / 12.0
DATA["molecules"] = [mol_dod, mol_c60]
DATA["rC"], DATA["rH"] = rC, rH

# N = 2 species for Gamma_mol = A_g, from the Ih table
res = {g["name"]: g for g in pickle.load(open(os.path.join(HERE, "results.pkl"), "rb"))}["Ih"]
L = res["labels"]
DATA["N2_rows"] = []
for m in res["mols"]:
    for row in res["tables"][m]:
        if row["N"] == 2:
            DATA["N2_rows"].append(dict(mol=L[m], r=L[row["r"]], n=L[row["n"]], v=L[row["v"]],
                                        gen=row["generic"], comps=[(c["cls"], c["rank"], c["tag"]) for c in row["comps"]]))
DATA["bases"] = dict(K2=C_K2, K4=C_K4, S=C_S, A=C_A, abs_eps=aeps, eps=eps)
pickle.dump(DATA, open(os.path.join(HERE, "fusion_data.pkl"), "wb"))
print("HHH:", DATA["HHH"])
print("GHH:", {k: v for k, v in DATA["GHH"].items() if k != "M"})
print("TTT:", DATA["TTT"])
print("vecop:", DATA["vecop"])
for m in DATA["molecules"]:
    print(m["name"], m["mol"], m["vib"], "B=%.4g" % m["B"], m["classes"])
