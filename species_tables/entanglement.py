"""
Entanglement classification of the (locally maximally mixed) coupling tensors
C_{abc} of an admissible triple (Gamma_r, Gamma_n, Gamma_v) -> Gamma_mol (1D).

Classes
  sep       : all carriers one-dimensional (product state)
  bip       : exactly one carrier one-dimensional; the other two carry a
              maximally entangled d x d pair
  GHZ       : genuinely tripartite, all dims equal to d, tensor rank d
              (LU-equivalent to the d-level GHZ state, verified by fidelity)
  GHZ-type  : genuinely tripartite with unequal dims, tensor rank = max dim
              (a GHZ_dmax state with the smaller carriers' levels compressed)
  LC-eps    : (3,3,3), LU-equivalent to the Levi-Civita state eps_ijk/sqrt6
              (antisymmetric, tensor rank 5)
  LC-|eps|  : (3,3,3), LU-equivalent to |eps_ijk|/sqrt6
              (symmetric "permanent" partner, tensor rank 4)
  higher    : genuinely tripartite, tensor rank > max dim, not of LC form
"""
import numpy as np
from scipy.optimize import least_squares

RNG = np.random.default_rng(2026)


def unfold_ranks(T, tol=1e-8):
    a, b, c = T.shape
    r1 = np.linalg.matrix_rank(T.reshape(a, b * c), tol)
    r2 = np.linalg.matrix_rank(np.transpose(T, (1, 0, 2)).reshape(b, a * c), tol)
    r3 = np.linalg.matrix_rank(np.transpose(T, (2, 0, 1)).reshape(c, a * b), tol)
    return (r1, r2, r3)


def _model(A, B, C):
    return np.einsum("ir,jr,kr->ijk", A, B, C)


def _als(T, R, iters, A, B, C):
    lam = 1e-6
    prev = np.inf
    for it in range(iters):
        W = (B.T @ B.conj()) * (C.T @ C.conj())
        A = np.einsum("ijk,jr,kr->ir", T, B.conj(), C.conj()) @ np.linalg.inv(W + lam * np.eye(R))
        W = (A.T @ A.conj()) * (C.T @ C.conj())
        B = np.einsum("ijk,ir,kr->jr", T, A.conj(), C.conj()) @ np.linalg.inv(W + lam * np.eye(R))
        W = (A.T @ A.conj()) * (B.T @ B.conj())
        C = np.einsum("ijk,ir,jr->kr", T, A.conj(), B.conj()) @ np.linalg.inv(W + lam * np.eye(R))
        if it % 50 == 49:
            res = np.linalg.norm(T - _model(A, B, C))
            if res < 1e-11:
                break
            if it > 300 and res > 1e-3 and prev - res < 1e-4 * prev:
                break  # stagnated far from an exact decomposition
            prev = res
            lam = max(lam * 0.5, 1e-14)
    return A, B, C


def _lm(T, R, A, B, C):
    a, b, c = T.shape

    def unpack(x):
        z = x[: len(x) // 2] + 1j * x[len(x) // 2:]
        return z[: a * R].reshape(a, R), z[a * R: (a + b) * R].reshape(b, R), z[(a + b) * R:].reshape(c, R)

    def fun(x):
        r = (_model(*unpack(x)) - T).ravel()
        return np.concatenate([r.real, r.imag])

    Ia, Ib, Ic = np.eye(a), np.eye(b), np.eye(c)
    n = a * b * c

    def jac(x):
        A_, B_, C_ = unpack(x)
        JA = np.einsum("il,js,ks->ijkls", Ia, B_, C_).reshape(n, a * R)
        JB = np.einsum("is,jl,ks->ijkls", A_, Ib, C_).reshape(n, b * R)
        JC = np.einsum("is,js,kl->ijkls", A_, B_, Ic).reshape(n, c * R)
        J = np.hstack([JA, JB, JC])
        return np.block([[J.real, -J.imag], [J.imag, J.real]])

    z = np.concatenate([A.ravel(), B.ravel(), C.ravel()])
    x0 = np.concatenate([z.real, z.imag])
    try:
        sol = least_squares(fun, x0, jac=jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400)
        return unpack(sol.x)
    except Exception:
        return A, B, C


def cp_decomposable(T, R, starts=24, iters=1500):
    """True if a well-conditioned rank-R CP decomposition with residual ~0 is found."""
    T = T / np.linalg.norm(T)
    a, b, c = T.shape
    best = np.inf
    for s in range(starts):
        A = RNG.normal(size=(a, R)) + 1j * RNG.normal(size=(a, R))
        B = RNG.normal(size=(b, R)) + 1j * RNG.normal(size=(b, R))
        C = RNG.normal(size=(c, R)) + 1j * RNG.normal(size=(c, R))
        A, B, C = _als(T, R, iters, A, B, C)
        res = np.linalg.norm(T - _model(A, B, C))
        if res < 1e-3:
            A, B, C = _lm(T, R, A, B, C)
            res = np.linalg.norm(T - _model(A, B, C))
        best = min(best, res)
        if res < 1e-9:
            norms = np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0) * np.linalg.norm(C, axis=0)
            if norms.max() < 1e3:
                return True, res
    return False, best


def lu_fidelity(T, S, starts=12, iters=400):
    """max over local unitaries |<S| U1 x U2 x U3 |T>| (both normalised)."""
    T = T / np.linalg.norm(T)
    S = S / np.linalg.norm(S)
    if T.shape != S.shape:
        return 0.0
    dims = T.shape
    best = 0.0
    for s in range(starts):
        U = []
        for d in dims:
            q, _ = np.linalg.qr(RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d)))
            U.append(q)
        val = 0
        for it in range(iters):
            for p in range(3):
                X = T
                # apply the other two unitaries
                for q in range(3):
                    if q != p:
                        X = np.moveaxis(np.tensordot(U[q], X, axes=([1], [q])), 0, q)
                # K[i,a] = sum_{others} X[..i..] conj(S[..a..])
                Xp = np.moveaxis(X, p, 0).reshape(dims[p], -1)
                Sp = np.moveaxis(S, p, 0).reshape(dims[p], -1)
                K = Xp @ Sp.conj().T          # (i, a)
                W, sv, Vh = np.linalg.svd(K)
                U[p] = (W @ Vh).conj().T      # maximises Re Tr(U K)
                val = sv.sum()
            if 1 - val < 1e-12:
                break
        best = max(best, val)
        if 1 - best < 1e-10:
            break
    return best


def eps_tensor(sym=False):
    E = np.zeros((3, 3, 3))
    for (i, j, k), s in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1,
                         (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
        E[i, j, k] = 1 if sym else s
    return E / np.sqrt(6)


def ghz_tensor(d):
    G = np.zeros((d, d, d))
    for i in range(d):
        G[i, i, i] = 1
    return G / np.sqrt(d)


CLASS_ORDER = ["sep", "bip", "GHZ", "GHZ-type", "LC-eps", "LC-|eps|", "higher"]


def classify(T, rmax_extra=8):
    """returns dict(cls, rank, dims, note)"""
    T = np.asarray(T, complex)
    T = T / np.linalg.norm(T)
    dims = T.shape
    ur = unfold_ranks(T)
    assert ur == dims, (ur, dims)       # irreducibility => full local ranks
    nt = sum(d > 1 for d in dims)
    if nt == 0:
        return dict(cls="sep", rank=1, dims=dims)
    if nt == 2:
        return dict(cls="bip", rank=max(dims), dims=dims)
    assert nt == 3
    R0 = max(dims)
    rank = None
    for R in range(R0, R0 + rmax_extra + 1):
        ok, res = cp_decomposable(T, R)
        if ok:
            rank = R
            break
    out = dict(rank=rank if rank is not None else -(R0 + rmax_extra), dims=dims)
    if rank == R0:
        if len(set(dims)) == 1:
            f = lu_fidelity(T, ghz_tensor(R0))
            assert f > 1 - 1e-6, f
            out["cls"] = "GHZ"
        else:
            out["cls"] = "GHZ-type"
        return out
    if dims == (3, 3, 3):
        if lu_fidelity(T, eps_tensor(False)) > 1 - 1e-7:
            out["cls"] = "LC-eps"
            return out
        if lu_fidelity(T, eps_tensor(True)) > 1 - 1e-7:
            out["cls"] = "LC-|eps|"
            return out
    out["cls"] = "higher"
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    print(classify(eps_tensor()), classify(eps_tensor(True)), classify(ghz_tensor(2)))
    w = np.exp(2j * np.pi / 3)
    e = np.zeros((3, 2, 3), complex)
    for i in range(3):
        e[i, 0, i] = w ** (i + 1)
        e[i, 1, i] = -w ** (-(i + 1))
    print(classify(e))
    print(time.time() - t)
