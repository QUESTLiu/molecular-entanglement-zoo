"""
Render results.pkl (from species.py) into species_tables.tex / .pdf and
species_data.json.
"""
import json
import os
import pickle
import subprocess
from collections import OrderedDict
from fractions import Fraction

import numpy as np

from entanglement import CLASS_ORDER
from species import tkey

HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_TEX = {
    "sep": "separable",
    "bip": "bipartite",
    "GHZ": "GHZ",
    "GHZ-type": "GHZ-type",
    "LC-eps": r"LC($\epsilon$)",
    "LC-|eps|": r"LC($|\epsilon|$)",
}
COARSE = {
    "sep": "separable",
    "bip": "bipartite",
    "GHZ": "GHZ-type tripartite",
    "GHZ-type": "GHZ-type tripartite",
    "LC-eps": "Levi-Civita-type tripartite",
    "LC-|eps|": "Levi-Civita-type tripartite",
    "higher": "tripartite, beyond GHZ/LC",
}
PRINTABLE = {"I", "Ih", "K", "Kh"}  # groups whose tensors are NOT written out


def class_tex(cls, rank):
    if cls == "higher":
        return r"tri.\ ($r{=}%s$)" % (rank if rank > 0 else r"{>}%d" % (-rank))
    return CLASS_TEX[cls]


def letters(k):
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(97 + r) + s
    return s


# ----------------------------------------------------------------------------
# coefficient / tensor rendering
# ----------------------------------------------------------------------------
def phase_tex(z):
    ph = np.angle(z) / (2 * np.pi) % 1.0
    f = Fraction(ph).limit_denominator(48)
    if abs(float(f) - ph) > 1e-6 and abs(float(f) - ph - 1) > 1e-6:
        return "+e^{i%.4f}" % (2 * np.pi * ph), False
    f = f % 1
    table = {Fraction(0): "+", Fraction(1, 2): "-", Fraction(1, 4): "+i", Fraction(3, 4): "-i",
             Fraction(1, 3): r"+\omega", Fraction(2, 3): r"+\omega^{2}",
             Fraction(1, 6): r"-\omega^{2}", Fraction(5, 6): r"-\omega"}
    if f in table:
        return table[f], True
    p, q = f.numerator, f.denominator
    if p * 2 > q:
        return r"+e^{-2\pi i\cdot %d/%d}" % (q - p, q), True
    return r"+e^{2\pi i\cdot %d/%d}" % (p, q), True


def mag_tex(x):
    f = Fraction(x * x).limit_denominator(400)
    if abs(float(f) - x * x) > 1e-7:
        return "%.4f" % x
    if f == 1:
        return ""
    p, q = f.numerator, f.denominator
    if p == 1:
        return r"\tfrac{1}{\sqrt{%d}}" % q
    return r"\sqrt{\tfrac{%d}{%d}}" % (p, q)


def ket_labels(d, cubic):
    if d == 1:
        return [None]
    if d == 2:
        return ["1", "2"]
    if d == 3 and cubic:
        return ["x", "y", "z"]
    return [str(i + 1) for i in range(d)]


def tensor_tex(T, cubic):
    dims = T.shape
    labs = [ket_labels(d, cubic) for d in dims]
    terms = []
    for idx in np.ndindex(*dims):
        c = T[idx]
        if abs(c) > 1e-8:
            ket = r"\,".join(labs[p][idx[p]] for p in range(3) if labs[p][0] is not None)
            terms.append((c, ket))
    if all(d == 1 for d in dims):
        return r"$\ket{A}\ket{A}\ket{A}$"
    mags = np.array([abs(c) for c, _ in terms])
    common = np.allclose(mags, mags[0], atol=1e-7)
    s = ""
    for c, ket in terms:
        ph, _ = phase_tex(c)
        coef = ph
        if not common:
            m = mag_tex(abs(c))
            coef = ph + m if m else ph
        s += r"%s|%s\rangle" % (coef, ket)
    s = s.lstrip("+")
    if common:
        pre = mag_tex(mags[0])
        if pre:
            s = pre + r"\bigl(" + s + r"\bigr)"
    return "$" + s + "$"


# ----------------------------------------------------------------------------
def sort_rows(rows):
    def key(r):
        c0 = r["comps"][0]["cls"]
        return (max(r["dims"]), CLASS_ORDER.index(c0), r["dims"], r["r"], r["n"], r["v"])
    return sorted(rows, key=key)


def assign_families(g):
    letter_of, variants = OrderedDict(), {}
    fam = {}
    for m in g["mols"]:
        for row in sort_rows(g["tables"][m]):
            for c in row["comps"]:
                lk = (tkey(c["T"], absval=True), c["cls"])
                vk = tkey(c["T"])
                if lk not in letter_of:
                    letter_of[lk] = letters(len(letter_of))
                    variants[lk] = OrderedDict()
                if vk not in variants[lk]:
                    variants[lk][vk] = dict(T=c["T"], cls=c["cls"], rank=c["rank"],
                                            rep=(row["r"], row["n"], row["v"]), mol=m)
                c["vk"], c["lk"] = vk, lk
    legend = []
    for lk, L in letter_of.items():
        vs = list(variants[lk].items())
        for j, (vk, info) in enumerate(vs):
            name = L if len(vs) == 1 else "%s_{%d}" % (L, j + 1)
            fam[vk] = name
            legend.append((name, info))
    return fam, legend


def esc(s):
    return s


def lab(g, i):
    return "$" + g["labels"][i] + "$"


def dims_tex(d):
    return "$(%d,%d,%d)$" % d


def nontrivial_table(g, m, fam, has_tau, out):
    rows = sort_rows(g["tables"][m])
    lines = []
    for row in rows:
        N = row["N"]
        for t, c in enumerate(row["comps"]):
            cells = [lab(g, row["r"]), lab(g, row["n"]), lab(g, row["v"]), dims_tex(row["dims"]),
                     class_tex(c["cls"], c["rank"]), "(%s)" % ("$" + fam[c["vk"]] + "$" if "_" in fam[c["vk"]] else fam[c["vk"]])]
            if has_tau:
                cells.insert(4, "%d/%d%s" % (t + 1, N, ("\\," + c["tag"]) if c["tag"] else "") if N > 1 else "")
            lines.append(cells)
    ncol = len(lines[0]) if lines else (7 if has_tau else 6)
    head = [r"$\Gamma_{\rm r}$", r"$\Gamma_{\rm n}$", r"$\Gamma_{\rm v}$", r"$(d_{\rm r},d_{\rm n},d_{\rm v})$",
            "class", "fam."]
    if has_tau:
        head.insert(4, r"$\tau/N$")
    K = 52
    blank = [""] * ncol
    for start in range(0, len(lines), 2 * K):
        chunk = lines[start:start + 2 * K]
        left = chunk[:K]
        right = chunk[K:]
        if not right and len(left) > 18:
            h = (len(left) + 1) // 2
            left, right = left[:h], left[h:]
        nr = max(len(left), len(right))
        colspec = ("c" * 4 + ("c" if has_tau else "") + "ll")
        if right:
            out.append(r"\begin{center}\footnotesize\setlength\tabcolsep{2.6pt}")
            out.append(r"\begin{tabular}{%s@{\qquad}%s}" % (colspec, colspec))
            out.append(r"\toprule")
            out.append(" & ".join(head + head) + r"\\")
        else:
            out.append(r"\begin{center}\footnotesize\setlength\tabcolsep{3pt}")
            out.append(r"\begin{tabular}{%s}" % colspec)
            out.append(r"\toprule")
            out.append(" & ".join(head) + r"\\")
        out.append(r"\midrule")
        for i in range(nr):
            L = left[i] if i < len(left) else blank
            if right:
                R = right[i] if i < len(right) else blank
                out.append(" & ".join(L + R) + r"\\")
            else:
                out.append(" & ".join(L) + r"\\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}\end{center}")


def abelian_grid(g, m, out):
    labs = g["labels"]
    n = len(labs)
    grid = {}
    for row in g["tables"][m]:
        grid[(row["r"], row["n"])] = row["v"]
    size = r"\footnotesize" if n <= 8 else r"\scriptsize"
    out.append(r"\begin{center}%s\setlength\tabcolsep{3pt}" % size)
    wide = n > 12
    if wide:
        out.append(r"\resizebox{\linewidth}{!}{%")
    out.append(r"\begin{tabular}{c|%s}" % ("c" * n))
    out.append(r"\toprule")
    out.append(r"$\Gamma_{\rm r}\backslash\Gamma_{\rm n}$ & " + " & ".join("$" + l + "$" for l in labs) + r"\\")
    out.append(r"\midrule")
    for r in range(n):
        out.append("$" + labs[r] + "$ & " + " & ".join(("$" + labs[grid[(r, k)]] + "$") if (r, k) in grid else r"($|\Lambda|{>}3$)"
                                                for k in range(n)) + r"\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}" + ("}" if wide else ""))
    out.append(r"\end{center}")


GROUP_TITLES = OrderedDict([
    ("nonaxial", "Non-axial groups $C_1$, $C_{\\mathrm s}$, $C_{\\mathrm i}$"),
    ("Cn", "Cyclic groups $C_n$"),
    ("S2n", "Improper cyclic groups $S_{2n}$"),
    ("Cnh", "Groups $C_{n\\mathrm h}$"),
    ("Cnv", "Groups $C_{n\\mathrm v}$"),
    ("Dn", "Dihedral groups $D_n$"),
    ("Dnh", "Groups $D_{n\\mathrm h}$"),
    ("Dnd", "Groups $D_{n\\mathrm d}$"),
    ("cubic", "Tetrahedral and octahedral groups"),
    ("icosa", "Icosahedral groups"),
    ("linear", "Linear (continuous axial) groups"),
    ("rot", "The full rotation groups"),
])


def section_of(g):
    f = g["family"]
    if f in ("C1", "Cs", "Ci"):
        return "nonaxial"
    if f in ("T", "Th", "Td", "O", "Oh"):
        return "cubic"
    if f in ("I", "Ih"):
        return "icosa"
    if f in ("Cinf", "Cinfh", "Cinfv", "Dinfh"):
        return "linear"
    if f in ("K", "Kh"):
        return "rot"
    return f


def class_counts(g, m):
    cnt = OrderedDict((c, 0) for c in CLASS_ORDER)
    for row in g["tables"][m]:
        for c in row["comps"]:
            cnt[c["cls"]] += 1
    return cnt


def group_section(g, out, data):
    labs = g["labels"]
    out.append(r"\subsection{$%s$}" % g["latex"])
    out.append(r"\label{sec:%s}" % g["name"])
    order = ("$|\\mathsf G|=%d$" % g["order"]) if g["order"] else "continuous"
    irr = ", ".join("$%s$\\,(%d)" % (l, d) for l, d in zip(labs, g["dims"]))
    out.append(r"\noindent %s; irreps (dimension): %s." % (order, irr))
    if g["truncated"]:
        if g["family"] in ("K", "Kh"):
            out.append(r" Truncated at $J\le 3$; coupling tensors are Wigner $3j$ symbols in the $|J m\rangle$ basis.")
        else:
            out.append(r" Truncated at $|\Lambda|\le 3$ (exact for these irreps: computed with the finite subgroup "
                       r"$C_{13}$, $C_{14\mathrm h}$, $C_{13\mathrm v}$ or $D_{14\mathrm h}$, whose invariants coincide "
                       r"with those of the continuous group when $|\Lambda_1|+|\Lambda_2|+|\Lambda_3|<13$).")
    mols = ", ".join("$%s$" % labs[m] for m in g["mols"])
    out.append(r" Possible $\Gamma_{\rm mol}$ (real one-dimensional irreps): %s." % mols)
    gd = dict(group=g["name"], order=g["order"], irreps=[dict(label=l, dim=d) for l, d in zip(labs, g["dims"])],
              tables={})
    data.append(gd)
    if g["abelian"]:
        out.append(r" All irreps are one-dimensional, so every admissible triple is \textbf{separable}, "
                   r"the coupling tensor is the single number $1$, and for given $(\Gamma_{\rm r},\Gamma_{\rm n})$ "
                   r"the vibrational species is unique, $\Gamma_{\rm v}=\Gamma_{\rm mol}\otimes\bar\Gamma_{\rm r}"
                   r"\otimes\bar\Gamma_{\rm n}$. Each grid below lists that $\Gamma_{\rm v}$ "
                   r"(rows $\Gamma_{\rm r}$, columns $\Gamma_{\rm n}$)%s.\par"
                   % ("" if g["truncated"] else "; all %d entries per grid are admissible" % (len(labs) ** 2)))
        for m in g["mols"]:
            out.append(r"\paragraph{$\Gamma_{\rm mol}=%s$} (%d admissible triples, all separable)" %
                       (labs[m], len(g["tables"][m])))
            abelian_grid(g, m, out)
            gd["tables"][labs[m]] = [dict(r=labs[x["r"]], n=labs[x["n"]], v=labs[x["v"]], dims=[1, 1, 1], N=1,
                                          cls="sep") for x in g["tables"][m]]
        return
    fam, legend = assign_families(g)
    has_tau = any(row["N"] > 1 for m in g["mols"] for row in g["tables"][m])
    printable = g["name"] not in PRINTABLE
    cubic = g["family"] in ("T", "Th", "Td", "O", "Oh")
    # legend
    out.append(r"\par\smallskip\noindent\textit{Coupling-tensor families.} ")
    if printable:
        out.append(r"Kets list the indices of the degenerate carriers in the order $(\mathrm r,\mathrm n,\mathrm v)$ of "
                   r"the representative triple (one-dimensional carriers omitted); "
                   + (r"$x,y,z$ are Cartesian $T$ components, $1,2$ stand for ${}^1E,{}^2E$; $\omega=e^{2\pi i/3}$."
                      if cubic else r"$1,2$ stand for the components ${}^1E_k,{}^2E_k$ (or ${}^{1,2}\Pi$, \dots).")
                   )
    else:
        out.append(r"Tensors of dimension $>3$ are not written out; they are given in \texttt{species\_data.json}.")
    out.append(r"{\footnotesize\setlength\tabcolsep{4pt}")
    if printable:
        out.append(r"\begin{longtable}{l l c l p{0.46\linewidth}}")
        out.append(r"\hline fam. & class (nature) & rank & representative & tensor\\\hline\endhead")
    else:
        out.append(r"\begin{longtable}{l l c l}")
        out.append(r"\hline fam. & class (nature) & rank & representative \\\hline\endhead")
    for name, info in legend:
        rep = info["rep"]
        reps = r"$(%s,%s,%s)\!\to\!%s$" % (labs[rep[0]], labs[rep[1]], labs[rep[2]], labs[info["mol"]])
        nm = "$" + name + "$" if "_" in name else name
        cl = class_tex(info["cls"], info["rank"])
        if info["cls"] not in ("sep", "bip"):
            cl += r" \emph{(%s)}" % COARSE[info["cls"]].replace("tripartite", "trip.")
        rk = str(info["rank"]) if info["rank"] > 0 else r"$>%d$" % (-info["rank"])
        if printable:
            out.append("(%s) & %s & %s & %s & %s\\\\" % (nm, cl, rk, reps, tensor_tex(info["T"], cubic)))
        else:
            out.append("(%s) & %s & %s & %s\\\\" % (nm, cl, rk, reps))
    out.append(r"\hline\end{longtable}}")
    # multiplicity notes
    if has_tau:
        notes = []
        seen = set()
        for m in g["mols"]:
            for row in g["tables"][m]:
                if row["N"] > 1:
                    key = (row["r"], row["n"], row["v"], m)
                    gen = row["generic"]
                    notes.append(r"$(%s,%s,%s)\!\to\!%s$: $N=%d$, basis %s; generic member: %s"
                                 % (labs[row["r"]], labs[row["n"]], labs[row["v"]], labs[m], row["N"],
                                    ", ".join(class_tex(c["cls"], c["rank"]) + ("[%s]" % c["tag"] if c["tag"] else "")
                                              for c in row["comps"]),
                                    class_tex(gen["cls"], gen["rank"])))
        out.append(r"\noindent\textit{Multiplicity $N>1$.} The coupling space is $N$-dimensional; the table lists an "
                   r"orthonormal basis adapted to the permutation symmetry of equal carriers "
                   r"(S symmetric, A antisymmetric, M mixed/none), and the entanglement class depends on the "
                   r"chosen vector. Generic (random) members of each space:")
        out.append(r"{\footnotesize\begin{itemize}\setlength\itemsep{0pt}")
        for nt in notes:
            out.append(r"\item " + nt)
        out.append(r"\end{itemize}}")
    for m in g["mols"]:
        cnt = class_counts(g, m)
        cs = ", ".join("%d %s" % (v, class_tex(k, 0) if k != "higher" else "beyond GHZ/LC") for k, v in cnt.items() if v)
        out.append(r"\subsubsection*{$%s$, $\Gamma_{\rm mol}=%s$ \normalfont\small(%d admissible triples: %s)}"
                   % (g["latex"], labs[m], len(g["tables"][m]), cs))
        nontrivial_table(g, m, fam, has_tau, out)
        rows_json = []
        for row in sort_rows(g["tables"][m]):
            for t, c in enumerate(row["comps"]):
                T = c["T"]
                ent = [[int(i), int(j), int(k), float(T[i, j, k].real), float(T[i, j, k].imag)]
                       for i, j, k in zip(*np.nonzero(abs(T) > 1e-10))]
                rows_json.append(dict(r=labs[row["r"]], n=labs[row["n"]], v=labs[row["v"]], dims=list(row["dims"]),
                                      N=row["N"], tau=t + 1, sym=c["tag"], cls=c["cls"], nature=COARSE[c["cls"]],
                                      rank=c["rank"], family=fam[c["vk"]], tensor=ent))
        gd["tables"][labs[m]] = rows_json


def summary_table(results, out):
    out.append(r"{\footnotesize\setlength\tabcolsep{3.5pt}")
    out.append(r"\begin{longtable}{l r r r l r r r r r r r}")
    hdr = (r"\hline $\mathsf G$ & $|\mathsf G|$ & \#irr & $d_{\max}$ & $\Gamma_{\rm mol}$ choices & \#adm. & sep & bip & "
           r"GHZ & GHZ-t. & LC & beyond\\\hline")
    out.append(hdr + r"\endfirsthead")
    out.append(hdr + r"\endhead")
    for g in results:
        m = g["mols"][0]
        cnt = class_counts(g, m)
        mols = ", ".join("$%s$" % g["labels"][x] for x in g["mols"])
        if len(g["mols"]) > 4:
            mols = r"%d (all real 1D)" % len(g["mols"])
        out.append(r"\hyperref[sec:%s]{$%s$} & %s & %d%s & %d & %s & %d & %d & %d & %d & %d & %d & %d\\" % (
            g["name"], g["latex"].split("=")[0], g["order"] if g["order"] else r"$\infty$", len(g["labels"]),
            "$^*$" if g["truncated"] else "", max(g["dims"]), mols, len(g["tables"][m]),
            cnt["sep"], cnt["bip"], cnt["GHZ"], cnt["GHZ-type"], cnt["LC-eps"] + cnt["LC-|eps|"], cnt["higher"]))
    out.append(r"\hline\end{longtable}}")


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx}
\usepackage[dvipsnames]{xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[colorlinks=true,linkcolor=blue!50!black]{hyperref}
\setcounter{tocdepth}{2}
\newcommand{\ket}[1]{|#1\rangle}
\title{Admissible rotation--nuclear-spin--vibration species and their\\ symmetry-enforced entanglement
for the point groups of $\mathsf O(3)$}
\author{Generated with \texttt{species\_tables/} (Python)}
\date{\today}
\begin{document}
\maketitle
"""

INTRO = r"""
\section{What is tabulated}
For a point group $\mathsf G\subset\mathsf O(3)$ and a required molecular species $\Gamma_{\rm mol}$, a triple
$(\Gamma_{\rm r},\Gamma_{\rm n},\Gamma_{\rm v})$ of rotational, nuclear-spin and vibrational species is admissible when
\begin{equation}
N^{\Gamma_{\rm mol}}_{\Gamma_{\rm r}\Gamma_{\rm n}\Gamma_{\rm v}}=\frac1{|\mathsf G|}\sum_{R\in\mathsf G}
\chi_{\Gamma_{\rm r}}(R)\chi_{\Gamma_{\rm n}}(R)\chi_{\Gamma_{\rm v}}(R)\chi^*_{\Gamma_{\rm mol}}(R)\neq0 .
\end{equation}
The coupling (fusion) space is the range of the projector
$P=\frac1{|\mathsf G|}\sum_R\chi^*_{\Gamma_{\rm mol}}(R)\,D^{\Gamma_{\rm r}}(R)\otimes D^{\Gamma_{\rm n}}(R)\otimes D^{\Gamma_{\rm v}}(R)$;
for $N=1$ it is spanned by one normalised tensor $C_{abc}$ (the Clebsch--Gordan coupling to $\Gamma_{\rm mol}$), and the carrier state is
$\ket{C}=\sum_{abc}C_{abc}\ket{a}_{\rm r}\ket{b}_{\rm n}\ket{c}_{\rm v}$.

\paragraph{Scope.}
\begin{itemize}\setlength\itemsep{0pt}
\item \emph{Groups.} All closed subgroups of $\mathsf O(3)$: $C_1,C_{\rm s},C_{\rm i}$; the seven axial families
$C_n,S_{2n},C_{n\rm h},C_{n\rm v},D_n,D_{n\rm h},D_{n\rm d}$ for $n=2,\dots,8$; $T,T_{\rm h},T_{\rm d},O,O_{\rm h},I,I_{\rm h}$;
the linear groups $C_\infty,C_{\infty\rm h},C_{\infty\rm v},D_{\infty\rm h}$ (irreps with $|\Lambda|\le3$);
and $\mathsf{SO}(3)$, $\mathsf O(3)$ (irreps $J\le3$). ($D_\infty\cong C_{\infty\rm v}$ has the same table as
$C_{\infty\rm v}$ with $\Sigma^\pm\to A_{1,2}$.)
\item \emph{$\Gamma_{\rm mol}$.} Eq.~(gamma\_mol) of the main text, $\chi_{\Gamma_{\rm mol}}(R)=\prod_a{\rm sgn}[p_a(R)]^{2I_a}$,
only produces characters $\pm1$, so $\Gamma_{\rm mol}$ runs over \emph{every real one-dimensional irrep} of $\mathsf G$; each gets its own table.
Complex one-dimensional irreps (${}^{1,2}E$ of $C_n$, $T$, \dots) cannot be $\Gamma_{\rm mol}$.
Tables for different $\Gamma_{\rm mol}$ are related by $\Gamma_{\rm v}\mapsto\Gamma_{\rm v}\otimes\Gamma_{\rm mol}\otimes\Gamma'_{\rm mol}$
and contain the same entanglement classes.
\item \emph{Species.} Every irrep is allowed in every sector (pure group theory). Molecule-specific restrictions, e.g.\ the
nuclear-spin species that a given set of identical nuclei can actually carry, are \emph{not} imposed: e.g.\ the 43 $T_{\rm d}$,
$\Gamma_{\rm mol}=A_2$ triples reduce to the 26 rows of the $\mathrm{CH_4}$ table when $\Gamma_{\rm n}\in\{A_1,E,T_2\}$.
\item Irreps are irreducible over $\mathbb C$; complex-conjugate pairs are listed separately as ${}^1E,{}^2E$
(${}^1E_k$ has character $e^{+2\pi ik/n}$ on the generating rotation/rotoreflection).
\end{itemize}

\paragraph{Basis conventions} (those of the main text). Principal axis $z$, $C_2'$ along $x$, $\sigma_{\rm v}=xz$.
Two-dimensional irreps use the complex basis $\ket{1}\equiv\ket{{}^1E}$, $\ket{2}\equiv\ket{{}^2E}$ with
$D(C_n)={\rm diag}(e^{i\theta},e^{-i\theta})$ (for $D_{n\rm d}$, $n$ even: $S_{2n}$) and the swapping element
($C_2'(x)$, else $\sigma_{\rm v}(xz)$; in the cubic groups $\sigma_{\rm d}(x\leftrightarrow y)$ for $T_{\rm d}$ and $C_2[1\bar10]$ for $O,O_{\rm h}$)
represented by $\bigl(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\bigr)$; for cubic $E$ the diagonal element is $C_3[111]$ with $\ket{{}^1E}\mapsto\omega\ket{{}^1E}$.
Three-dimensional cubic irreps are Cartesian: $\ket{x},\ket{y},\ket{z}$ diagonalise $C_2^{x,y,z}$ and $C_3[111]$ permutes them
cyclically. These reproduce exactly the tensors (a)--(g) of the $T_{\rm d}$ example (up to the family letters, which are
assigned automatically here). Entanglement classes do not depend on these choices.

\section{Entanglement classes}
By Schur's lemma every reduced state of $\ket{C}$ on one carrier is $\mathbb{I}/d_\Gamma$ (each flattening of $C$ has full rank
$d_\Gamma$). Such \emph{locally maximally mixed} states are critical; by the Kempf--Ness theorem two of them are equivalent under
SLOCC iff they are equivalent under local unitaries, so the following labels are SLOCC (and LU) classes:
\begin{center}\small
\begin{tabular}{l l p{0.62\linewidth}}
\toprule
label & nature & criterion\\\midrule
separable & separable & $d_{\rm r}=d_{\rm n}=d_{\rm v}=1$.\\
bipartite & bipartite & exactly one carrier one-dimensional; the other two form a maximally entangled $d\times d$ pair
(Schmidt rank $d$, flat spectrum).\\
GHZ & GHZ-type tripartite & $d_{\rm r}=d_{\rm n}=d_{\rm v}=d>1$ and tensor rank $d$; LU-equivalent to
$d^{-1/2}\sum_i\ket{iii}$ (verified by maximising the LU fidelity).\\
GHZ-type & GHZ-type tripartite & all $d>1$, unequal, tensor rank $=d_{\max}$: $\ket C=\sum_{i=1}^{d_{\max}}\ket{a_i}\ket{b_i}\ket{c_i}$, i.e.\ a
$\mathrm{GHZ}_{d_{\max}}$ state with the smaller carriers' levels compressed (e.g.\ $T_{\rm d}$ family (e$_\pm$), $(3,2,3)$).\\
LC($\epsilon$) & Levi-Civita-type tripartite & $(3,3,3)$, LU-equivalent to $6^{-1/2}\sum\epsilon_{ijk}\ket{ijk}$ (tensor rank 5).\\
LC($|\epsilon|$) & Levi-Civita-type tripartite & $(3,3,3)$, LU-equivalent to $6^{-1/2}\sum|\epsilon_{ijk}|\ket{ijk}$ (the symmetric,
permanent-type partner; tensor rank 4, hence SLOCC-inequivalent to $\epsilon$).\\
tri.\ ($r{=}R$) & tripartite, beyond GHZ/LC & genuinely tripartite with tensor rank $R>d_{\max}$ and not of LC form
(only in $I$, $I_{\rm h}$, $\mathsf{SO}(3)$, $\mathsf O(3)$ and for generic members of $N>1$ spaces).\\
\bottomrule
\end{tabular}
\end{center}
\noindent\emph{Method.} Tensor ranks are obtained numerically: the lower bound $d_{\max}$ is the flattening rank; an explicit,
well-conditioned complex CP decomposition with residual $<10^{-9}$ (alternating least squares + Levenberg--Marquardt, many random
starts) certifies the upper bound. Hence the GHZ/GHZ-type assignments ($r=d_{\max}$) are rigorous; ranks reported above
$d_{\max}$ are the smallest $R$ for which a decomposition was found (for $\epsilon$ and $|\epsilon|$ they agree with the known values
5 and 4). LU equivalence to $\mathrm{GHZ}_d$, $\epsilon$, $|\epsilon|$ is tested by maximising $|\langle S|U_1\otimes U_2\otimes U_3|C\rangle|$
(fidelity $1$ to $10^{-7}$).

\paragraph{Families.} Within each group, tensors are given a letter by their modulus pattern (up to permutation of the three
sectors) and class; subscripts distinguish phase variants (e.g.\ the $\pm$ of family (b), or (e$_\pm$), of the $T_{\rm d}$
example). The legend at the start of each group writes the tensor of one representative triple.

\paragraph{Observations.}
\begin{enumerate}\setlength\itemsep{0pt}
\item Abelian groups ($C_n,S_{2n},C_{n\rm h},C_1,C_{\rm s},C_{\rm i},D_2,D_{2\rm h},C_{2\rm v},C_\infty,C_{\infty\rm h}$): separable only.
\item Axial non-Abelian groups and $C_{\infty\rm v},D_{\infty\rm h}$ (all irreps of dimension $\le2$): only separable, bipartite
($2\times2$ Bell pairs) and GHZ $(2,2,2)$. For $E_a\otimes E_b\otimes E_c$ the tensor is always the two-term GHZ state
because a single pair of sign patterns $\pm a\pm b\pm c\equiv0\pmod n$ survives.
\item Cubic groups add $(1,3,3)$-type bipartite states, GHZ-type $(2,3,3)$ states (rank 3), and both Levi-Civita types.
In $T_{\rm d}$, $O$, $O_{\rm h}$ both $\epsilon$ and $|\epsilon|$ occur, distinguished by whether the $T\otimes T\to T$
step is antisymmetric or symmetric.
\item \textbf{Multiplicity two outside the icosahedral groups.} In $T$ and $T_{\rm h}$ one has $T\otimes T=A\oplus{}^1E\oplus{}^2E\oplus2T$,
so $(T,T,T)$ (and its $g/u$ versions in $T_{\rm h}$) has $N=2$: the space is spanned by $|\epsilon|$ (symmetric) and
$\epsilon$ (antisymmetric). Its members range over different classes: $\epsilon$ and $|\epsilon|$ themselves, the combination
$(|\epsilon|+\epsilon)/\sqrt2\propto\ket{xyz}+\ket{yzx}+\ket{zxy}$ which is GHZ, and generic members of rank 5.
This is an exception to the statement that $N\le1$ for all non-icosahedral molecular point groups.
\item Icosahedral groups: $N=2$ occurs exactly for $(G,H,H)$ and $(H,H,H)$ up to permutation (and their $g/u$ versions),
confirming Eq.~(icosahedral\_products) of the main text. Besides bipartite, $\epsilon$-type $(T_1,T_1,T_1)$, $(T_2,T_2,T_2)$ and
GHZ-type $(3,3,5)$ couplings (rank 5), most tripartite triples are of a new kind: rank 5 for $(3,3,4)$ and $(4,4,4)$, 6 for $(3,4,4)$
and $(3,4,5)$, 7 for $(4,4,5)$, 8 for $(3,5,5)$, $(4,5,5)$, $(5,5,5)$.
\item $\mathsf{SO}(3)$/$\mathsf O(3)$ ($3j$ tensors): $(1,1,1)$ is exactly $\epsilon$; $(1,1,2)$ and $(1,2,3)$ are GHZ-type (rank $=d_{\max}$);
all others are beyond GHZ/LC. For the $7$-dimensional cases $(2,3,3)$ and $(3,3,3)$ no decomposition was found up to rank 15
(reported as $r>15$).
\end{enumerate}
"""


def main():
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "results.pkl"
    with open(os.path.join(HERE, src), "rb") as f:
        results = pickle.load(f)
    from fusion_tex import fusion_tex
    out = [PREAMBLE, INTRO, fusion_tex()]
    out.append(r"\section{Summary}")
    out.append(r"Counts are for one $\Gamma_{\rm mol}$ (identical for every choice); for $N>1$ each basis tensor is counted. "
               r"$^*$: truncated continuous group.")
    summary_table(results, out)
    out.append(r"\tableofcontents")
    out.append(r"\clearpage")
    data = []
    current = None
    for g in results:
        sec = section_of(g)
        if sec != current:
            out.append(r"\clearpage\section{%s}" % GROUP_TITLES[sec])
            current = sec
        group_section(g, out, data)
    out.append(r"\end{document}")
    tex = "\n".join(out)
    with open(os.path.join(HERE, "species_tables.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    with open(os.path.join(HERE, "species_data.json"), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
