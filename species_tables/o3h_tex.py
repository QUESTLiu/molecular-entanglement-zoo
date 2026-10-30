"""How species_tables.tex prints irreps and groups, as o3harmonics' documents write them (data: o3h_labels.json,
o3h_print.json).

- label_tex(G, label): the irrep, \\irr{a}_{1\\ptag{g}}, \\conjpair{2}\\irr{e}_{1};
- group_tex(G): the group, \\grp{C_{3v}}, \\grp{C_{1h}}, \\grp{SO}(3);
- partner_map(G, label): the map from the partners of a linear stand-in's own basis (Cinfv, Dinfh) to o3harmonics'
  partners, "identity" or "sign2" (minus partner 2), or None for a one-dimensional irrep.
"""
import json
import os

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
