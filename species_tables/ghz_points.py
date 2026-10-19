"""
GHZ-type members of the I_h multiplicity-two coupling spaces.

For (H_g,H_g,H_g) and (G_g,H_g,H_g) -> A_g, find the points of the fusion Bloch sphere
    |C(theta,phi)> = cos(theta/2)|C1> + e^{i phi} sin(theta/2)|C2>
whose tensor rank equals d_max = 5 (the least possible), by minimising the residual of a
rank-5 CP decomposition over (theta, phi) with warm-started ALS/LM.  At each solution the
LU fidelity with GHZ_5 (5x5x5 case) and Lambda^2 are evaluated.

    python ghz_points.py      -> ghz_points.pkl
"""
import pickle
import numpy as np
from scipy.optimize import minimize
from entanglement import _als, _lm, _model, lu_fidelity, ghz_tensor, RNG

D = pickle.load(open("fusion_data.pkl", "rb"))
B = D["bases"]


def state(x, C1, C2):
    t, p = x
    C = np.cos(t / 2) * C1 + np.exp(1j * p) * np.sin(t / 2) * C2
    return C / np.linalg.norm(C)


class CPResidual:
    def __init__(self, R, shape):
        self.R, self.shape, self.f = R, shape, None

    def fit(self, C, starts=1):
        best = (np.inf, None)
        inits = [self.f] if self.f is not None else []
        inits += [None] * starts
        for f0 in inits:
            if f0 is None:
                f0 = [RNG.normal(size=(d, self.R)) + 1j * RNG.normal(size=(d, self.R)) for d in self.shape]
            A, Bm, Cc = _als(C, self.R, 400, *[x.copy() for x in f0])
            A, Bm, Cc = _lm(C, self.R, A, Bm, Cc)
            res = np.linalg.norm(C - _model(A, Bm, Cc))
            if res < best[0]:
                best = (res, [A, Bm, Cc])
        self.f = best[1]
        return best[0]


def geometric(C, starts=200, iters=400):
    best = 0.0
    for s in range(starts):
        vs = [RNG.normal(size=d) + 1j * RNG.normal(size=d) for d in C.shape]
        vs = [x / np.linalg.norm(x) for x in vs]
        for it in range(iters):
            vs[0] = np.einsum("abc,b,c->a", C, vs[1].conj(), vs[2].conj()); vs[0] /= np.linalg.norm(vs[0])
            vs[1] = np.einsum("abc,a,c->b", C, vs[0].conj(), vs[2].conj()); vs[1] /= np.linalg.norm(vs[1])
            vs[2] = np.einsum("abc,a,b->c", C, vs[0].conj(), vs[1].conj()); vs[2] /= np.linalg.norm(vs[2])
        best = max(best, abs(np.einsum("abc,a,b,c->", C, *[v.conj() for v in vs])) ** 2)
    return best


def refine(C1, C2, x0):
    cp = CPResidual(5, C1.shape)
    cp.fit(state(x0, C1, C2), starts=40)
    res = minimize(lambda x: cp.fit(state(x, C1, C2)), x0, method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-15, maxiter=600))
    C = state(res.x, C1, C2)
    r = cp.fit(C, starts=10)
    norms = np.linalg.norm(cp.f[0], axis=0) * np.linalg.norm(cp.f[1], axis=0) * np.linalg.norm(cp.f[2], axis=0)
    return res.x, r, norms.max(), C


out = {}
seeds = {"HHH": [(0.454, 0.388), (0.454, 1.055), (0.454, 1.721), (0.546, 0.388), (0.546, 1.055), (0.546, 1.721)],
         "GHH": [(0.580, 0.747), (0.580, 1.747), (0.420, 0.747), (0.420, 1.747), (0.580, 0.253), (0.580, 1.253)]}
pairs = {"HHH": (B["K2"], B["K4"]), "GHH": (B["S"], B["A"])}
for name, (C1, C2) in pairs.items():
    sols = []
    for s in seeds[name]:
        x, r, nmax, C = refine(C1, C2, np.array(s) * np.pi)
        x = np.array([x[0] % (2 * np.pi), x[1] % (2 * np.pi)]) / np.pi
        if r < 1e-9 and nmax < 1e3 and not any(np.allclose(x, y["x"], atol=1e-5) for y in sols):
            fid = lu_fidelity(C, ghz_tensor(5)) if C.shape == (5, 5, 5) else None
            sols.append(dict(x=x, residual=r, L2=geometric(C), fid_ghz5=fid))
            print(name, "theta/pi=%.6f phi/pi=%.6f" % tuple(x), "rank-5 residual=%.1e" % r,
                  "Lambda^2=%.6f" % sols[-1]["L2"], "F(GHZ5)=%s" % fid, flush=True)
        else:
            print(name, "seed", s, "-> no exact rank-5 point (residual %.1e)" % r, flush=True)
    out[name] = sols
pickle.dump(out, open("ghz_points.pkl", "wb"))
