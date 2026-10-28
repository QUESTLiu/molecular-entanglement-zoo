"""How the REPO's outputs print irreps, groups, elements and occupation kets, as o3harmonics' documents
write them (SCOPE-notation.md §4; data: o3h_labels.json, o3h_print.json, o3h_irreps.json).

- label_tex(G, label): the irrep, \\irr{a}_{1\\ptag{g}}, \\conjpair{2}\\irr{e}_{1}, \\irrJp{1}{+};
- group_tex(G): the group, \\grp{C_{3v}}, \\grp{C_{1h}}, \\grp{SO}(3);
- element_tex(G, R): the element of the 3x3 column matrix R, \\elt{S_{3}^{2}}, by o3harmonics' symbol;
- occ(n, partner_map): the occupation ket of a level, o3harmonics' \\occ{v_1 v_2 ...} (v_mu quanta in
  its partner mu), with the sign a ket of the REPO's own basis picks up (partner_map from the
  crosswalk, S identification): identity +; sign on mu=2 (-1)^{n_2}; exchange with sign (-1)^{n_1}
  and the two digits exchanged.
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_D = {}


def _load(name):
    if name not in _D:
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            _D[name] = json.load(f)
    return _D[name]


def label_tex(group, label):
    return _load("o3h_labels.json")["groups"][group]["labels"][label]["doc"]


def group_tex(group):
    return _load("o3h_print.json")["groups"][group]["group_tex"]


def partner_map(group, label):
    return _load("o3h_print.json")["groups"][group]["partner_maps"].get(label)


def _sym_tex(symbol):
    """o3harmonics' plain element symbol (C_3^2, sigma_v, S_6^5, i) as \\elt{...}"""
    m = re.fullmatch(r"(sigma|[A-Za-z])(?:_([0-9A-Za-z]+))?(?:\^([0-9]+))?('*)", symbol)
    assert m, symbol
    base, sub, sup, primes = m.groups()
    out = r"\sigma" if base == "sigma" else base
    if sub:
        out += "_{%s}" % sub
    if sup and sup != "1":
        out += "^{%s}" % sup
    return r"\elt{%s}" % (out + primes)


def element_tex(group, R):
    rec = _load("o3h_print.json")["groups"][group]["o3h_record"]
    els = _load("o3h_irreps.json")["records"][rec]["elements"]
    for e in els:
        if np.allclose(np.array(e["P"]).T, R, atol=1e-6):
            return _sym_tex(e["symbol"])
    raise KeyError((group, R))


def occ(n, pm=None):
    """(sign, ket) for the occupation tuple n of a level in the REPO's basis"""
    n = list(n)
    s = 1
    if pm == "sign2":
        s = -1 if n[1] % 2 else 1
    elif pm == "exchange_sign":
        s = -1 if n[0] % 2 else 1
        n = [n[1], n[0]]
    body = ",".join(map(str, n)) if max(n) > 9 else "".join(map(str, n))
    return s, r"\occ{%s}" % body
