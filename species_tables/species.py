"""
Admissible rotation--nuclear-spin--vibration species (Gamma_r, Gamma_n, Gamma_v)
for every point group of O(3) and every one-dimensional real Gamma_mol, with
coupling tensors and their entanglement class.

    python species.py            -> results.pkl, species_data.json
"""
import itertools
import json
import os
import pickle
import sys
import numpy as np

import pointgroups as pg
from entanglement import classify, CLASS_ORDER, RNG

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "class_cache.pkl")


# ----------------------------------------------------------------------------
# SO(3) and O(3) (truncated at J <= JMAX)
# ----------------------------------------------------------------------------
class RotationGroup:
    def __init__(self, improper, jmax=3):
        self.name = "Kh" if improper else "K"
        self.latex = r"K_{\mathrm h}=\mathsf O(3)" if improper else r"K=\mathsf{SO}(3)"
        self.family = self.name
        self.truncated = True
        self.jmax = jmax
        self.irr = []
        for par in (["g", "u"] if improper else [""]):
            for J in range(jmax + 1):
                self.irr.append((J, par))
        self.labels = [("\\mathcal D^{(%d)}_{%s}" % (J, p)) if p else ("\\mathcal D^{(%d)}" % J) for J, p in self.irr]
        self.dims = [2 * J + 1 for J, p in self.irr]
        self.order = None

    def real1d(self):
        return [i for i, (J, p) in enumerate(self.irr) if J == 0]

    @staticmethod
    def gens(J):
        m = np.arange(J, -J - 1, -1)
        Jz = np.diag(m).astype(complex)
        Jp = np.zeros((2 * J + 1, 2 * J + 1), complex)
        for k in range(1, 2 * J + 1):
            mm = m[k]
            Jp[k - 1, k] = np.sqrt(J * (J + 1) - mm * (mm + 1))
        return [Jz, Jp, Jp.conj().T]

    def N(self, r, n, v, m):
        (J1, p1), (J2, p2), (J3, p3), (J0, p0) = (self.irr[x] for x in (r, n, v, m))
        if p0:
            par = ((p1 == "u") + (p2 == "u") + (p3 == "u") + (p0 == "u")) % 2
            if par:
                return 0
        return 1 if abs(J1 - J2) <= J3 <= J1 + J2 else 0

    def tensors(self, r, n, v, m):
        J = [self.irr[x][0] for x in (r, n, v)]
        d = [2 * j + 1 for j in J]
        g = [self.gens(j) for j in J]
        rows = []
        for a in range(3):
            I = [np.eye(x) for x in d]
            op = (np.kron(np.kron(g[0][a], I[1]), I[2]) + np.kron(np.kron(I[0], g[1][a]), I[2])
                  + np.kron(np.kron(I[0], I[1]), g[2][a]))
            rows.append(op)
        M = np.vstack(rows)
        u, s, vh = np.linalg.svd(M)
        null = vh[np.sum(s > 1e-9):].conj()
        return [x.reshape(d) for x in null]


# ----------------------------------------------------------------------------
# finite groups
# ----------------------------------------------------------------------------
def real1d(G):
    if isinstance(G, RotationGroup):
        return G.real1d()
    out = []
    for i, (d, ch) in enumerate(zip(G.dims, G.chars)):
        if d == 1 and np.allclose(ch.imag, 0, atol=1e-8) and np.allclose(abs(ch.real), 1, atol=1e-8):
            out.append(i)
    return out


def N_finite(G, r, n, v, m):
    val = np.sum(G.chars[r] * G.chars[n] * G.chars[v] * G.chars[m].conj()) / G.order
    k = int(round(val.real))
    assert abs(val - k) < 1e-6
    return k


def tensors_finite(G, r, n, v, m):
    d = (G.dims[r], G.dims[n], G.dims[v])
    P = np.zeros((np.prod(d),) * 2, complex)
    for g in range(G.order):
        P += np.conj(G.chars[m][g]) * np.kron(np.kron(G.D[r][g], G.D[n][g]), G.D[v][g])
    P /= G.order
    P = (P + P.conj().T) / 2
    w, U = np.linalg.eigh(P)
    vecs = U[:, w > 0.5]
    return [vecs[:, k].reshape(d) for k in range(vecs.shape[1])]


def N_of(G, r, n, v, m):
    return G.N(r, n, v, m) if isinstance(G, RotationGroup) else N_finite(G, r, n, v, m)


def tensors_of(G, r, n, v, m):
    return G.tensors(r, n, v, m) if isinstance(G, RotationGroup) else tensors_finite(G, r, n, v, m)


# ----------------------------------------------------------------------------
# tensor utilities
# ----------------------------------------------------------------------------
def phase_fix(T):
    T = np.asarray(T, complex)
    T = T / np.linalg.norm(T)
    flat = T.ravel()
    k = np.argmax(abs(flat) > 1e-8)
    T = T * abs(flat[k]) / flat[k]
    T[abs(T) < 1e-12] = 0
    return T


def tkey(T, absval=False):
    """canonical key up to permutation of parties and global phase"""
    best = None
    for perm in itertools.permutations(range(3)):
        X = np.transpose(T, perm)
        X = abs(phase_fix(X)) if absval else phase_fix(X)
        k = (X.shape, tuple(int(round(x * 1e5)) for x in X.real.ravel()),
             tuple(int(round(x * 1e5)) for x in X.imag.ravel()))
        if best is None or k < best:
            best = k
    return best


def sym_adapt(tensors, labs):
    """orthonormal basis of the coupling space adapted to permutations of equal carriers"""
    nT = len(tensors)
    if nT == 1:
        return [(phase_fix(tensors[0]), "")]
    V = np.array([t.ravel() for t in tensors]).T  # columns
    shape = tensors[0].shape
    eq = [(a, b) for a, b in itertools.combinations(range(3), 2) if labs[a] == labs[b]]

    def permop(perm):
        dim = V.shape[0]
        M = np.zeros((dim, dim))
        for idx in itertools.product(*[range(s) for s in shape]):
            src = np.ravel_multi_index(idx, shape)
            dst = np.ravel_multi_index(tuple(idx[perm[i]] for i in range(3)), shape)
            M[dst, src] = 1
        return M

    sectors = []
    if len(eq) == 3:
        perms = list(itertools.permutations(range(3)))
        sgn = [1, -1, -1, 1, 1, -1]  # parity of permutations in lexicographic order
        sgn = [int(round(np.linalg.det(np.eye(3)[list(p)]))) for p in perms]
        S = sum(permop(p) for p in perms) / 6
        A = sum(s * permop(p) for s, p in zip(sgn, perms)) / 6
        sectors = [("S", S), ("A", A)]
    elif len(eq) == 1:
        a, b = eq[0]
        perm = [0, 1, 2]
        perm[a], perm[b] = b, a
        Pm = permop(perm)
        I = np.eye(V.shape[0])
        sectors = [("S", (I + Pm) / 2), ("A", (I - Pm) / 2)]
    out = []
    used = np.zeros((V.shape[0], 0), complex)
    for tag, Pr in sectors:
        W = Pr @ V
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        W = u[:, s > 1e-8]
        for k in range(W.shape[1]):
            out.append((W[:, k], tag))
        used = np.hstack([used, W])
    # remaining (mixed symmetry or no symmetry)
    Q = V - used @ (used.conj().T @ V)
    u, s, vh = np.linalg.svd(Q, full_matrices=False)
    for k in range(int(np.sum(s > 1e-8))):
        out.append((u[:, k], "M" if sectors else ""))
    # make basis vectors as sparse as possible inside each sector is not attempted
    assert len(out) == nT
    return [(phase_fix(v.reshape(shape)), tag) for v, tag in out]


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(c):
    with open(CACHE, "wb") as f:
        pickle.dump(c, f)


def classify_cached(T, cache):
    k = tkey(T)
    if k not in cache:
        cache[k] = classify(T)
    return cache[k]


def analyse_group(G, cache, verbose=True):
    nirr = len(G.labels)
    mols = real1d(G)
    out = dict(name=G.name, latex=G.latex, family=G.family, order=G.order,
               labels=list(G.labels), dims=list(G.dims), mols=mols,
               truncated=getattr(G, "truncated", False), tables={})
    abelian = max(G.dims) == 1
    out["abelian"] = abelian
    for m in mols:
        rows = []
        for r, n, v in itertools.product(range(nirr), repeat=3):
            N = N_of(G, r, n, v, m)
            if N == 0:
                continue
            row = dict(r=r, n=n, v=v, N=N, dims=(G.dims[r], G.dims[n], G.dims[v]))
            if abelian:
                row["comps"] = [dict(T=np.ones((1, 1, 1), complex), tag="", cls="sep", rank=1)]
                rows.append(row)
                continue
            Ts = tensors_of(G, r, n, v, m)
            assert len(Ts) == N
            comps = []
            for T, tag in sym_adapt(Ts, (G.labels[r], G.labels[n], G.labels[v])):
                c = classify_cached(T, cache)
                comps.append(dict(T=T, tag=tag, cls=c["cls"], rank=c["rank"]))
            if N > 1:
                coef = RNG.normal(size=N) + 1j * RNG.normal(size=N)
                Tg = phase_fix(sum(cf * T for cf, T in zip(coef, Ts)))
                c = classify(Tg)
                row["generic"] = dict(cls=c["cls"], rank=c["rank"])
            row["comps"] = comps
            rows.append(row)
        out["tables"][m] = rows
        if verbose:
            print("  ", G.name, G.labels[m], len(rows), "admissible triples", flush=True)
    return out


def all_groups(nmax):
    gs = pg.all_finite_groups(nmax)
    gs.append(RotationGroup(False))
    gs.append(RotationGroup(True))
    return gs


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    cache = load_cache()
    results = []
    gdir = os.path.join(HERE, "results")
    os.makedirs(gdir, exist_ok=True)
    for G in all_groups(nmax):
        if only and G.name not in only:
            continue
        fn = os.path.join(gdir, G.name + ".pkl")
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                results.append(pickle.load(f))
            continue
        print(G.name, flush=True)
        res = analyse_group(G, cache)
        save_cache(cache)
        with open(fn, "wb") as f:
            pickle.dump(res, f)
        results.append(res)
    with open(os.path.join(HERE, "results.pkl" if not only else "results_partial.pkl"), "wb") as f:
        pickle.dump(results, f)
