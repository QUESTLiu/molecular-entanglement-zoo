"""
Admissible rotation--nuclear-spin--vibration species (Gamma_r, Gamma_n, Gamma_v) for Altmann and Herzig's 75 point
groups (pointgroups.all_finite_groups: the 73 finite groups and the stand-ins of C_inf v and D_inf h) and every
one-dimensional real Gamma_mol, with the coupling tensors and their rows in the catalogue of AME states (ame.py).

Every coupling tensor is an AME state of its format and gets its catalogue id from ame.lookup. A row (an admissible
triple) has a coupling space of dimension N = 1 or 2, and one copy (comp) per basis tensor, dict(T, K, state):
  N = 1   T = phase_fix(tensor), K None; checked: flat marginals, rho_YZ = P_Lambda/d_X;
  N = 2   the K basis (ame.kbasis) in ascending K, each copy phase-fixed, K = k; checked: flat marginals;
state is the catalogue id (ame.lookup). The rows of an Abelian group (every irrep one-dimensional) all have the 1x1x1
tensor 1, the product state "P".

    python -B species.py [nmax [G1,G2,...]]   -> results/<G>.pkl per group, then results.pkl (results_partial.pkl
                                                 with a group list or nmax other than 10); nmax defaults to 10

A cached results/<G>.pkl is reused only when its "schema" is SCHEMA and its "inputs" is input_digest(), the sha256 of
the files it depends on (INPUTS); any other is recomputed. make_pdf.py renders results.pkl into species_tables.tex,
species_data.json and ame_catalogue.json, and it refuses a results.pkl of another schema or other inputs, or with a
number of groups other than 75.
"""
import functools
import hashlib
import itertools
import os
import pickle
import sys
import numpy as np

import pointgroups as pg
import ame

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "ame-4"      # the group dict of results/<G>.pkl and results.pkl, comps dict(T, K, state), state an id of
                      # ame.CATALOGUE (P, B1..B4, T1..T16)
# the files that a group's result depends on, hashed by input_digest
INPUTS = ("pointgroups.py", "ame.py", "species.py", "atlas_irreps.json", "atlas_labels.json")


@functools.lru_cache(maxsize=None)
def input_digest():
    """sha256 of the files that a group's result depends on (INPUTS)"""
    h = hashlib.sha256()
    for fn in INPUTS:
        with open(os.path.join(HERE, fn), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# finite groups
# ----------------------------------------------------------------------------
def real1d(G):
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


def tkey(T):
    """canonical key up to permutation of parties and global phase"""
    best = None
    for perm in itertools.permutations(range(3)):
        X = phase_fix(np.transpose(T, perm))
        k = (X.shape, tuple(int(round(x * 1e5)) for x in X.real.ravel()),
             tuple(int(round(x * 1e5)) for x in X.imag.ravel()))
        if best is None or k < best:
            best = k
    return best


# ----------------------------------------------------------------------------
# copies and their catalogue ids
# ----------------------------------------------------------------------------
def comp(T, K=None):
    """one copy: the phase-fixed tensor T, K (None for N = 1) and its catalogue id"""
    return dict(T=T, K=K, state=ame.lookup(T))


_abelian_checked = []


def abelian_comp():
    """the copy of every row of an Abelian group: the 1x1x1 tensor 1, the product state P (checked once per run)"""
    T = np.ones((1, 1, 1), complex)
    if not _abelian_checked:
        sid = ame.lookup(T)
        if sid != "P":
            raise AssertionError("ame gives %r for the 1x1x1 tensor 1, not 'P'" % sid)
        _abelian_checked.append(True)
    return dict(T=T, K=None, state="P")


def copies(G, idx, m, Ts):
    """the comps of the row idx = (r, n, v) -> m, from the orthonormal basis Ts of its coupling space
    (tensors_finite)"""
    if len(Ts) == 1:
        T = phase_fix(Ts[0])
        ame.check_flat(T)
        ame.check_pair_projector(G, idx, m, T)
        return [comp(T)]
    if len(Ts) == 2:
        kb = ame.kbasis(G, idx, m, Ts)
        Ks = [K for K, _ in kb]
        if len(Ks) != 2 or Ks != sorted(set(Ks)):
            raise ValueError("%s %s: K basis %s is not two distinct K in ascending order"
                             % (G.name, [G.labels[i] for i in idx], Ks))
        out = []
        for K, T in kb:
            T = phase_fix(T)
            ame.check_flat(T)
            out.append(comp(T, K))
        return out
    raise ValueError("%s (%s) -> %s: N = %d; no coupling space with N > 2 occurs in these groups"
                     % (G.name, ",".join(G.labels[i] for i in idx), G.labels[m], len(Ts)))


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def analyse_group(G):
    nirr = len(G.labels)
    mols = real1d(G)
    out = dict(name=G.name, family=G.family, order=G.order,
               labels=list(G.labels), dims=list(G.dims), mols=mols,
               atlas_record=getattr(G, "atlas_record", None), ids=list(getattr(G, "ids", None) or []),
               truncated=getattr(G, "truncated", False), tables={})
    abelian = max(G.dims) == 1
    out["abelian"] = abelian
    out["schema"] = SCHEMA
    out["inputs"] = input_digest()
    for m in mols:
        rows = []
        for r, n, v in itertools.product(range(nirr), repeat=3):
            N = N_finite(G, r, n, v, m)
            if N == 0:
                continue
            row = dict(r=r, n=n, v=v, N=N, dims=(G.dims[r], G.dims[n], G.dims[v]))
            if abelian:
                row["comps"] = [abelian_comp()]
                rows.append(row)
                continue
            Ts = tensors_finite(G, r, n, v, m)
            assert len(Ts) == N
            row["comps"] = copies(G, (r, n, v), m, Ts)
            rows.append(row)
        out["tables"][m] = rows
        print("  ", G.name, G.labels[m], len(rows), "admissible triples", flush=True)
    return out


def all_groups(nmax):
    return pg.all_finite_groups(nmax)


def load_group(fn):
    """the group dict cached in fn, or None (with a note) when it is unreadable, its schema is not SCHEMA or it was
    computed from other inputs (input_digest)"""
    try:
        with open(fn, "rb") as f:
            res = pickle.load(f)
    except Exception as e:
        print("  %s is unreadable (%s); recomputing" % (fn, e), flush=True)
        return None
    schema = res.get("schema") if isinstance(res, dict) else None
    if schema != SCHEMA:
        print("  %s has schema %r, not %r; recomputing" % (fn, schema, SCHEMA), flush=True)
        return None
    if res.get("inputs") != input_digest():
        print("  %s was computed from other inputs; recomputing" % fn, flush=True)
        return None
    return res


def dump(obj, fn):
    """pickle obj to fn through a temporary file, so an interrupted run leaves no truncated pickle"""
    tmp = "%s.%d.tmp" % (fn, os.getpid())
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, fn)


if __name__ == "__main__":
    if not __debug__:
        sys.exit("species.py: run without -O or PYTHONOPTIMIZE, since its checks are assert statements")
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    groups = all_groups(nmax)
    if only and set(only) - {G.name for G in groups}:
        sys.exit("species.py: no group %s for nmax = %d" % (", ".join(sorted(set(only) - {G.name for G in groups})),
                                                            nmax))
    results = []
    gdir = os.path.join(HERE, "results")
    os.makedirs(gdir, exist_ok=True)
    for G in groups:
        if only and G.name not in only:
            continue
        fn = os.path.join(gdir, G.name + ".pkl")
        res = load_group(fn) if os.path.exists(fn) else None
        if res is None:
            print(G.name, flush=True)
            res = analyse_group(G)
            dump(res, fn)
        results.append(res)
    dump(results, os.path.join(HERE, "results.pkl" if not only and nmax == 10 else "results_partial.pkl"))
