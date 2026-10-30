"""
Check the generated tables (results.pkl) on three worked molecules against values written out by hand: H2O (C2v,
Gamma_mol = B2), NH3 / CaOCH3 (C3v, Gamma_mol = A2) and CH4 (Td, Gamma_mol = A2), each with the nuclear-spin species
of its identical nuclei. For each molecule it checks the admissible rows (for H2O, those with Gamma_v in {A1, B2}, its
fundamentals) and the catalogue id (ame.CATALOGUE) of each, and, for NH3 / CaOCH3 and CH4, the coupling tensors, equal
up to a global phase. It prints PASS or FAIL last and exits 1 on any failure.

    python -B check_examples.py [results.pkl]
"""
import os
import pickle
import sys
import numpy as np

import ame

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.pkl")
res = {g["name"]: g for g in pickle.load(open(PATH, "rb"))}
w = np.exp(2j * np.pi / 3)
failures = []

if any(c.get("state") not in ame.IDS for g in res.values() for rows in g["tables"].values() for row in rows
       for c in row["comps"]):
    print("FAIL: %s has rows without a catalogue id of ame.CATALOGUE; rerun species.py" % PATH)
    sys.exit(1)


def check(name, ok):
    if not ok:
        failures.append(name)
    return ok


def table(gname, mol, spins):
    g = res[gname]
    L = g["labels"]
    m = L.index(mol)
    out = {}
    for row in g["tables"][m]:
        if L[row["n"]] in spins:
            out[(L[row["r"]], L[row["n"]], L[row["v"]])] = row
    return out


def state(row):
    return row["comps"][0]["state"] if row else "-"


def same_up_to_phase(A, B):
    if np.shape(A) != np.shape(B):
        return False
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    return float(np.abs(B - np.vdot(A, B) * A).max()) < 1e-10


def ket(dims, terms):
    T = np.zeros(dims, complex)
    for idx, c in terms:
        T[idx] += c
    return T


# ---------------------------------------------------------------- H2O
print("H2O  (C2v, Gamma_mol = B2, spin species A1 ortho / B2 para)")
hand = {("B_{2}", "A_{1}", "A_{1}"), ("A_{1}", "B_{2}", "A_{1}"), ("A_{1}", "A_{1}", "B_{2}"), ("B_{2}", "B_{2}", "B_{2}")}
gen = table("C2v", "B_{2}", {"A_{1}", "B_{2}"})
fund = {k for k in gen if k[2] in ("A_{1}", "B_{2}")}
print("  generated rows with Gamma_v in {A1,B2}:", sorted(fund))
print("  all generated rows:", len(gen), "| rows by hand:", len(hand))
print("  rows with Gamma_v in {A1,B2} = rows by hand:", check("H2O rows", fund == hand))
print("  rows by hand all present, catalogue id P:",
      check("H2O catalogue ids", all(k in gen and state(gen[k]) == "P" for k in hand)))
print("  generated rows not written by hand:", sorted(set(gen) - hand))

# ---------------------------------------------------------------- NH3 / CaOCH3
print("\nNH3 / CaOCH3  (C3v, Gamma_mol = A2, spin species A1 ortho / E para)")
gen = table("C3v", "A_{2}", {"A_{1}", "E"})
pp = {}
pp[("A_{2}", "A_{1}", "A_{1}")] = ("P", ket((1, 1, 1), [((0, 0, 0), 1)]))
pp[("A_{1}", "A_{1}", "A_{2}")] = ("P", ket((1, 1, 1), [((0, 0, 0), 1)]))
for j, lab in ((1, "A_{1}"), (2, "A_{2}")):
    pp[(lab, "E", "E")] = ("B1", ket((1, 2, 2), [((0, 0, 1), 1), ((0, 1, 0), (-1) ** j)]))
    pp[("E", "E", lab)] = ("B1", ket((2, 2, 1), [((0, 1, 0), 1), ((1, 0, 0), (-1) ** j)]))
pp[("E", "A_{1}", "E")] = ("B1", ket((2, 1, 2), [((0, 0, 1), 1), ((1, 0, 0), -1)]))
pp[("E", "E", "E")] = ("T1", ket((2, 2, 2), [((0, 0, 0), 1), ((1, 1, 1), -1)]))
ok = True
for k, (sid, T) in pp.items():
    row = gen.get(k)
    tm = row is not None and same_up_to_phase(row["comps"][0]["T"], T)
    good = state(row) == sid and tm
    ok &= good
    print("  %-26s by hand %-3s generated %-3s tensor match: %s%s"
          % (k, sid, state(row), tm, "" if good else "  MISMATCH"))
check("NH3/CaOCH3 catalogue ids and tensors", ok)
print("  all rows by hand reproduced (catalogue ids and tensors):", ok, "| generated rows:", len(gen),
      "| rows by hand:", len(pp))
print("  generated rows not written by hand:", sorted(set(gen) - set(pp)))
check("NH3/CaOCH3 rows", set(gen) == set(pp))

# ---------------------------------------------------------------- CH4
print("\nCH4  (Td, Gamma_mol = A2, spin species A1 meta / E para / T2 ortho)")
gen = table("Td", "A_{2}", {"A_{1}", "E", "T_{2}"})
T1, T2, A1, A2, E = "T_{1}", "T_{2}", "A_{1}", "A_{2}", "E"
form = {  # the 26 rows, each with the form a to g of its tensor (hand_tensor)
    (A1, A1, A2): "a", (A2, A1, A1): "a",
    (A1, E, E): "b", (A2, E, E): "b", (E, A1, E): "b", (E, E, A1): "b", (E, E, A2): "b",
    (E, E, E): "c",
    (A1, T2, T1): "d", (A2, T2, T2): "d", (T1, A1, T2): "d", (T2, A1, T1): "d", (T1, T2, A1): "d", (T2, T2, A2): "d",
    (T1, E, T1): "e-", (T1, E, T2): "e+", (T2, E, T1): "e+", (T2, E, T2): "e-",
    (T1, T2, E): "e+", (T2, T2, E): "e-", (E, T2, T1): "e+", (E, T2, T2): "e-",
    (T2, T2, T2): "f", (T1, T2, T1): "f", (T1, T2, T2): "g", (T2, T2, T1): "g",
}
id_of = {"a": "P", "b": "B1", "c": "T1", "d": "B2", "e+": "T2", "e-": "T2", "f": "T3", "g": "T4"}   # form -> id


def hand_tensor(k, f):
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
for k, f in form.items():
    row = gen.get(k)
    tm = row is not None and same_up_to_phase(row["comps"][0]["T"], hand_tensor(k, f))
    good = state(row) == id_of[f] and tm
    ok &= good
    if not good:
        bad.append((k, f, id_of[f], state(row), tm))
print("  rows by hand:", len(form), "| generated rows:", len(gen), "| same set:",
      check("CH4 rows", set(form) == set(gen)))
print("  all 26 catalogue ids and tensors reproduced:", check("CH4 catalogue ids and tensors", ok))
for k in ((T2, T2, T2), (T1, T2, T2)):
    print("  %s (form %s): catalogue id %s" % (k, form[k], state(gen.get(k))))
for b in bad:
    print("   MISMATCH (row, form, id by hand, generated id, tensor match):", b)

print()
if failures:
    print("FAIL:", "; ".join(failures))
    sys.exit(1)
print("PASS: the H2O, NH3/CaOCH3 and CH4 rows, catalogue ids and tensors match the values written out by hand")
