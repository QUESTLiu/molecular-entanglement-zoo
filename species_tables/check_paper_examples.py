"""
Compare the generated tables with the molecular examples of main_phase_space.tex
(H2O, NH3, CaOCH3, CH4): admissible rows, carrier dimensions, entanglement class,
and the explicit coupling tensors (equal up to a global phase).
"""
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = {g["name"]: g for g in pickle.load(open(os.path.join(HERE, "results.pkl"), "rb"))}
w = np.exp(2j * np.pi / 3)


def table(gname, mol, spins):
    g = res[gname]
    L = g["labels"]
    m = L.index(mol)
    out = {}
    for row in g["tables"][m]:
        if L[row["n"]] in spins:
            out[(L[row["r"]], L[row["n"]], L[row["v"]])] = row
    return out


def same_up_to_phase(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    return abs(abs(np.vdot(A, B)) - 1) < 1e-9


def ket(dims, terms):
    T = np.zeros(dims, complex)
    for idx, c in terms:
        T[idx] += c
    return T


# ---------------------------------------------------------------- H2O
print("H2O  (C2v, Gamma_mol = B2, spin species A1 ortho / B2 para)")
paper = {("B_{2}", "A_{1}", "A_{1}"), ("A_{1}", "B_{2}", "A_{1}"), ("A_{1}", "A_{1}", "B_{2}"), ("B_{2}", "B_{2}", "B_{2}")}
gen = table("C2v", "B_{2}", {"A_{1}", "B_{2}"})
print("  generated rows with Gamma_v in {A1,B2}:", sorted(k for k in gen if k[2] in ("A_{1}", "B_{2}")))
print("  all generated rows:", len(gen), "| paper rows:", len(paper))
print("  paper rows all present & separable:", all(k in gen and gen[k]["comps"][0]["cls"] == "sep" for k in paper))
print("  generated rows not in paper:", sorted(set(gen) - paper))

# ---------------------------------------------------------------- NH3 / CaOCH3
print("\nNH3 / CaOCH3  (C3v, Gamma_mol = A2, spin species A1 ortho / E para)")
gen = table("C3v", "A_{2}", {"A_{1}", "E"})
A = (1,)
s2 = 1 / np.sqrt(2)
pp = {}
pp[("A_{2}", "A_{1}", "A_{1}")] = ("sep", ket((1, 1, 1), [((0, 0, 0), 1)]))
pp[("A_{1}", "A_{1}", "A_{2}")] = ("sep", ket((1, 1, 1), [((0, 0, 0), 1)]))
for j, lab in ((1, "A_{1}"), (2, "A_{2}")):
    pp[(lab, "E", "E")] = ("bip", ket((1, 2, 2), [((0, 0, 1), 1), ((0, 1, 0), (-1) ** j)]))
    pp[("E", "E", lab)] = ("bip", ket((2, 2, 1), [((0, 1, 0), 1), ((1, 0, 0), (-1) ** j)]))
pp[("E", "A_{1}", "E")] = ("bip", ket((2, 1, 2), [((0, 0, 1), 1), ((1, 0, 0), -1)]))
pp[("E", "E", "E")] = ("GHZ", ket((2, 2, 2), [((0, 0, 0), 1), ((1, 1, 1), -1)]))
ok = True
for k, (cls, T) in pp.items():
    row = gen.get(k)
    good = row is not None and row["comps"][0]["cls"] == cls and same_up_to_phase(row["comps"][0]["T"], T)
    ok &= good
    print("  %-26s paper %-4s generated %-4s tensor match: %s" % (k, cls, row["comps"][0]["cls"] if row else "-", good))
print("  all paper rows reproduced:", ok, "| generated rows:", len(gen), "| paper rows:", len(pp))
print("  generated rows not in paper:", sorted(set(gen) - set(pp)))

# ---------------------------------------------------------------- CH4
print("\nCH4  (Td, Gamma_mol = A2, spin species A1 meta / E para / T2 ortho)")
gen = table("Td", "A_{2}", {"A_{1}", "E", "T_{2}"})
T1, T2, A1, A2, E = "T_{1}", "T_{2}", "A_{1}", "A_{2}", "E"
fam = {  # from the paper's table (left and right halves)
    (A1, A1, A2): "a", (A2, A1, A1): "a",
    (A1, E, E): "b", (A2, E, E): "b", (E, A1, E): "b", (E, E, A1): "b", (E, E, A2): "b",
    (E, E, E): "c",
    (A1, T2, T1): "d", (A2, T2, T2): "d", (T1, A1, T2): "d", (T2, A1, T1): "d", (T1, T2, A1): "d", (T2, T2, A2): "d",
    (T1, E, T1): "e-", (T1, E, T2): "e+", (T2, E, T1): "e+", (T2, E, T2): "e-",
    (T1, T2, E): "e+", (T2, T2, E): "e-", (E, T2, T1): "e+", (E, T2, T2): "e-",
    (T2, T2, T2): "f", (T1, T2, T1): "f", (T1, T2, T2): "g", (T2, T2, T1): "g",
}
cls_of = {"a": "sep", "b": "bip", "c": "GHZ", "d": "bip", "e+": "GHZ-type", "e-": "GHZ-type", "f": "LC-eps", "g": "LC-|eps|"}


def paper_tensor(k, f):
    dims = tuple({A1: 1, A2: 1, E: 2, T1: 3, T2: 3}[x] for x in k)
    if f == "a":
        return ket(dims, [((0, 0, 0), 1)])
    if f == "b":
        sgn = 1 if k in ((A2, E, E), (E, E, A2)) else -1
        slots = [p for p in range(3) if dims[p] == 2]
        t = []
        for a, b, c in ((0, 1, 1), (1, 0, sgn)):
            idx = [0, 0, 0]
            idx[slots[0]], idx[slots[1]] = a, b
            t.append((tuple(idx), c))
        return ket(dims, t)
    if f == "c":
        return ket(dims, [((0, 0, 0), 1), ((1, 1, 1), -1)])
    if f == "d":
        slots = [p for p in range(3) if dims[p] == 3]
        t = []
        for i in range(3):
            idx = [0, 0, 0]
            idx[slots[0]] = idx[slots[1]] = i
            t.append((tuple(idx), 1))
        return ket(dims, t)
    if f in ("e+", "e-"):
        sg = 1 if f == "e+" else -1
        pe = dims.index(2)
        pt = [p for p in range(3) if p != pe]
        t = []
        for i in range(3):
            for e, c in ((0, w ** (i + 1)), (1, sg * w ** (-(i + 1)))):
                idx = [0, 0, 0]
                idx[pe] = e
                idx[pt[0]] = idx[pt[1]] = i
                t.append((tuple(idx), c))
        return ket(dims, t)
    if f in ("f", "g"):
        t = []
        for (i, j, l), s in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
            t.append(((i, j, l), s if f == "f" else 1))
        return ket(dims, t)


ok = True
bad = []
for k, f in fam.items():
    row = gen.get(k)
    gcls = row["comps"][0]["cls"] if row else "-"
    tm = row is not None and same_up_to_phase(row["comps"][0]["T"], paper_tensor(k, f))
    good = row is not None and gcls == cls_of[f] and tm
    ok &= good
    if not good:
        bad.append((k, f, gcls, tm))
print("  paper rows:", len(fam), "| generated rows:", len(gen), "| same set:", set(fam) == set(gen))
print("  all 26 classes and tensors reproduced:", ok)
for b in bad:
    print("   MISMATCH", b)
