"""
LaTeX for the section on multiplicity-two coupling spaces (the "fusion qubit"),
generated from fusion_data.pkl (written by fusion_section.py).
"""
import os
import pickle

import o3h_tex

HERE = os.path.dirname(os.path.abspath(__file__))

CLS = {"GHZ": "GHZ", "LC-eps": r"LC($\epsilon$)", "LC-|eps|": r"LC($|\epsilon|$)", "higher": "beyond GHZ/LC",
       "GHZ-type": "GHZ-type"}


def f4(x):
    return "%.3f" % x


def irr(lab):
    """an icosahedral irrep as o3harmonics prints it (the fusion section is about I_h)"""
    try:
        return "$" + o3h_tex.label_tex("Ih", lab) + "$"
    except KeyError:
        return "$" + lab + "$"


# o3harmonics' class symbols, the count in front: \cls{12C_{5}}
CLASS_SYMBOL = {"E": "E", "C_5": "C_{5}", "C_5^2": "C_{5}^{2}", "C_3": "C_{3}", "C_2": "C_{2}", "i": "i",
                "S_{10}^3": "S_{10}^{3}", "S_{10}": "S_{10}", "S_6": "S_{6}", r"\sigma": r"\sigma"}


def lowest_J(D, lab):
    for J in range(0, 7):
        for s in ("g", "u"):
            if lab in D["rot"][(J, s)]:
                return r"$\irrJp{%d}{%s}$" % (J, "+" if s == "g" else "-")
    return "--"


def big(n):
    if n < 10 ** 7:
        return "$%s$" % "{:,}".format(n).replace(",", r"\,")
    m, e = ("%.3e" % n).split("e")
    return r"$%s\times10^{%d}$" % (m, int(e))


def points_table(D, key, caption):
    rows = D["points"][key]
    out = [r"\begin{center}\small\setlength\tabcolsep{5pt}",
           r"\begin{tabular}{l c l c c c c}",
           r"\toprule state & rank & class & $\mathcal N(\rho_{\rm rn})$ & $\mathcal N(\rho_{\rm rv})$ & "
           r"$\mathcal N(\rho_{\rm nv})$ & $\Lambda^2$\\\midrule"]
    for p in rows:
        out.append(r"%s & %d & %s & %s & %s & %s & %s\\" % (p["name"], p["rank"], CLS[p["cls"]], f4(abs(p["N_rn"])),
                                                         f4(abs(p["N_rv"])), f4(abs(p["N_nv"])), f4(p["L2"])))
    dmax = {"HHH": "GHZ", "GHH": "GHZ-type"}
    for g in D.get("ghz", {}).get(key, []):
        out.append(r"$(\theta,\varphi)=(%.4f\pi,\,%.4f\pi)$ & 5 & %s & %s & %s & %s & %s\\" % (
            g["theta"], g["phi"], CLS[dmax[key]], f4(abs(g["N_rn"])), f4(abs(g["N_rv"])), f4(abs(g["N_nv"])), f4(g["L2"])))
    out.append(r"\bottomrule\end{tabular}\\[2pt]{\footnotesize %s}\end{center}" % caption)
    return "\n".join(out)


def ghz_paragraph(D):
    G = D.get("ghz", {})
    if not G.get("HHH"):
        return ""
    h, g = G["HHH"], G.get("GHH", [])
    pts = lambda L: r",\ ".join(r"$(%.4f\pi,\,%.4f\pi)$" % (x["theta"], x["phi"]) for x in L)
    from fractions import Fraction

    def frac(x):
        f = Fraction(x).limit_denominator(64)
        return (r"\tfrac{%d}{%d}" % (f.numerator, f.denominator)) if abs(float(f) - x) < 1e-9 else "%.4f" % x

    pop = h[0]["pop1"]
    dphi = (h[1]["phi"] - h[0]["phi"]) if len(h) > 1 else 0
    gpop = g[0]["pop1"] if g else None
    return r"""
\paragraph{GHZ states inside the icosahedral fusion spaces.} Maximising $\Lambda^2$ and then minimising the residual of a rank-5
CP decomposition over the sphere (\texttt{ghz\_points.py}) gives, for $(H_g,H_g,H_g)$, %d points %s with residual
$<10^{-14}$ and LU fidelity $1$ (to $10^{-15}$) with $\mathrm{GHZ}_5=5^{-1/2}\sum_{k}\ket{kkk}$. Both have populations exactly
$%s$ in $\ket{K{=}2}$ and $%s$ in $\ket{K{=}4}$:
\[\ket{\mathrm{GHZ}_5}\simeq\sqrt{%s}\,\ket{K{=}2}+e^{i\varphi_\pm}\sqrt{%s}\,\ket{K{=}4}\quad(\text{LU-equivalent}).\]
The fusion space of three five-dimensional carriers thus contains genuine five-level GHZ states, with every pairwise marginal
unentangled ($\mathcal N=0$). For $(G_g,H_g,H_g)$ there are %d rank-5 GHZ-type points %s, with populations $%s$ (S) and $%s$ (A):
a $\mathrm{GHZ}_5$ with the rotational $G$ carrier compressed to four levels, in which rotation is pairwise unentangled with spin and
vibration ($\mathcal N_{\rm rn}=\mathcal N_{\rm rv}=0$) while the spin--vibration pair keeps $\mathcal N_{\rm nv}=\tfrac12$.

These states are \emph{not} eigenstates of the vector couplings. In $(H,H,H)$, however, the vector couplings act as
$(c_{\rm rn}+c_{\rm rv}+c_{\rm nv})\,{\rm diag}(-3,4)$, so free evolution inside the doublet only advances $\varphi$ at fixed
$\theta$, with beat period $2\pi\hbar/[7(c_{\rm rn}+c_{\rm rv}+c_{\rm nv})]$. Any state with populations $%s$ / $%s$ in
$\ket{K{=}2}$ / $\ket{K{=}4}$ therefore passes through both GHZ$_5$ points (separated by $\Delta\varphi=%.4f\pi$) once per beat:
the five-level GHZ state is a symmetry-allowed, dynamically recurring state of a single molecule.
""" % (len(h), pts(h), frac(pop), frac(1 - pop), frac(pop), frac(1 - pop), len(g), pts(g),
       frac(gpop) if g else "--", frac(1 - gpop) if g else "--", frac(pop), frac(1 - pop), dphi)


def molecule_block(D, m, other_note):
    labs = ["A_{g}", "T_{1g}", "T_{2g}", "G_{g}", "H_{g}", "A_{u}", "T_{1u}", "T_{2u}", "G_{u}", "H_{u}"]
    cls_order = ["E", "C_5", "C_5^2", "C_3", "C_2", "i", "S_{10}^3", "S_{10}", "S_6", r"\sigma"]
    sizes = {"E": 1, "C_5": 12, "C_5^2": 12, "C_3": 20, "C_2": 15, "i": 1, "S_{10}^3": 12, "S_{10}": 12,
             "S_6": 20, r"\sigma": 15}
    out = []
    # Gamma_mol derivation
    out.append(r"\begin{center}\small\setlength\tabcolsep{3.5pt}")
    out.append(r"\begin{tabular}{l" + "c" * len(cls_order) + "}")
    out.append(r"\toprule class & " + " & ".join(r"$\cls{%d%s}$" % (sizes[c], CLASS_SYMBOL[c]) if sizes[c] > 1
                                                  else r"$\cls{%s}$" % CLASS_SYMBOL[c] for c in cls_order) + r"\\\midrule")
    ct = []
    for c in cls_order:
        cyc = m["classes"][c]["cycles"]
        ct.append("$" + r"\,".join("%d^{%d}" % (L, k) for L, k in sorted(cyc.items())) + "$")
    out.append(r"cycle type $\ell^{\,\#}$ & " + " & ".join(ct) + r"\\")
    out.append(r"cycles $c[p(\gel)]$ & " + " & ".join(str(m["classes"][c]["ncyc"]) for c in cls_order) + r"\\")
    out.append(r"$\operatorname{sign}p(\gel)=\chi_{\Gamma_{\rm mol}}$ & " +
               " & ".join("$%+d$" % m["classes"][c]["parity"] for c in cls_order) + r"\\")
    out.append(r"$\chi^{\rm(n)}=2^{c[p(\gel)]}$ & " +
               " & ".join("$2^{%d}$" % m["classes"][c]["ncyc"] for c in cls_order) + r"\\")
    out.append(r"\bottomrule\end{tabular}\end{center}")
    # spin and vibrational content
    out.append(r"\begin{center}\small\setlength\tabcolsep{6pt}")
    out.append(r"\begin{tabular}{c r c @{\qquad} c r c}")
    hdr = r"$\Gamma$ & spin multiplets $m^{\rm(n)}_\Gamma$ & vib.\ fund."
    out.append(r"\toprule " + hdr + " & " + hdr + r"\\\midrule")
    for lg, lu in zip(labs[:5], labs[5:]):
        out.append(" & ".join([irr(lg), big(m["spin"].get(lg, 0)), str(m["vib"].get(lg, 0)),
                               irr(lu), big(m["spin"].get(lu, 0)), str(m["vib"].get(lu, 0))]) + r"\\")
    out.append(r"\bottomrule\end{tabular}\end{center}")
    # N = 2 species
    rows = [r for r in D["N2_rows"] if r["mol"] == m["mol"]]
    out.append(r"\begin{center}\small\setlength\tabcolsep{5pt}")
    out.append(r"\begin{tabular}{ccc l l l}")
    out.append(r"\toprule $\Gamma_{\rm r}$ & $\Gamma_{\rm n}$ & $\Gamma_{\rm v}$ & lowest $\irrJp{J}{p}$ & "
               r"spin multiplets $m^{\rm(n)}_{\Gamma_{\rm n}}$ & vibration\\\midrule")
    for r in rows:
        nv = m["vib"].get(r["v"], 0)
        vib = ("one quantum in any of %d modes" % nv) if nv else "combination"
        out.append(r"%s & %s & %s & %s & %s & %s\\" % (
            irr(r["r"]), irr(r["n"]), irr(r["v"]), lowest_J(D, r["r"]), big(m["spin"].get(r["n"], 0)), vib))
    out.append(r"\bottomrule\end{tabular}\end{center}")
    return "\n".join(out)


def fusion_tex():
    D = pickle.load(open(os.path.join(HERE, "fusion_data.pkl"), "rb"))
    H, G, T = D["HHH"], D["GHH"], D["TTT"]
    sy = D["sym"]
    rng = D["ranges"]
    dod, c60 = D["molecules"]
    eH = H["LdotL_eigs"]
    eG = G["eigs"]

    def rng_s(k, q):
        a, b = rng[k][q]
        return r"%s\text{--}%s" % (f4(abs(a)), f4(b))

    s = []
    s.append(r"""
\clearpage
\section{Multiplicity-two coupling spaces: the fusion qubit}
\label{sec:fusion}
For every molecular point group except four, an admissible triple has a \emph{unique} coupling tensor. The exceptions are
\begin{center}
$T,\,T_{\rm h}$: $(T,T,T)$ (and its $g/u$ versions); \qquad $I,\,I_{\rm h}$: $(G,H,H)$ and $(H,H,H)$ in every order and $g/u$ assignment,
\end{center}
for which $N^{\Gamma_{\rm mol}}_{\Gamma_{\rm r}\Gamma_{\rm n}\Gamma_{\rm v}}=2$: the one-dimensional species $\Gamma_{\rm mol}$ occurs
\emph{twice} in $\Gamma_{\rm r}\otimes\Gamma_{\rm n}\otimes\Gamma_{\rm v}$. This section works out what that means, which states
nature selects inside the two-dimensional coupling space $\mathcal F$, and how entangled they are, and then realises the
icosahedral case in two molecules, dodecahedrane $\mathrm{C_{20}H_{20}}$ and fully $^{13}$C-substituted buckminsterfullerene
$^{13}\mathrm{C}_{60}$. All numbers are computed by \texttt{fusion\_section.py}.

\subsection{What a two-dimensional coupling space means}
Fix the rotational, nuclear-spin and vibrational copies (the multiplicity labels $k_{\rm r}$, $(I_{\rm tot},M_I)$, $k_{\rm v}$). The
physical states with these labels and species $(\Gamma_{\rm r},\Gamma_{\rm n},\Gamma_{\rm v})$ form the two-dimensional space
$\mathcal F$ = range of the $\Gamma_{\rm mol}$-projector on $D^{\Gamma_{\rm r}}\otimes D^{\Gamma_{\rm n}}\otimes D^{\Gamma_{\rm v}}$.
\begin{enumerate}\setlength\itemsep{1pt}
\item \emph{Every} $\ket C\in\mathcal F$ is locally maximally mixed: each one-sector reduced state commutes with an irreducible
$D^\Gamma$ and is $\mathbb{I}/d_\Gamma$. Hence (Kempf--Ness) two vectors of $\mathcal F$ are SLOCC-equivalent iff they are
LU-equivalent, and any LU invariant that varies over $\mathcal F$ separates SLOCC classes.
\item $\mathsf G$ acts on $\mathcal F$ as the scalar $\chi_{\Gamma_{\rm mol}}$, and every $\mathsf G$-invariant operator acting on
\emph{one} sector is, on that sector's carrier, a multiple of $\mathbb{I}$ (Schur). The rigid-rotor, harmonic-vibrational and
nuclear-Zeeman/spin--spin parts of the Hamiltonian, and every one-sector measurement, therefore act on $\mathcal F$ as numbers.
$\mathcal F$ is a two-level system stored entirely in the \emph{correlations} between the sectors, invisible to any one of them ---
the molecular analogue of the three-spin noiseless subsystem, in which total spin $\tfrac12$ occurs twice in
$(\tfrac12)^{\otimes3}$ and collective rotations cannot touch the extra qubit.
\item Only inter-sector couplings $V=\sum_{p<q}V_{pq}$ (rotation--spin, rotation--vibration, spin--vibration) can act on
$\mathcal F$. Being $\mathsf G$-invariant they map the $\Gamma_{\rm mol}$-isotypic subspace, i.e.\ $\mathcal F$, to itself, so
$V|_{\mathcal F}$ is a $2\times2$ Hermitian matrix: every $N=2$ level is a \emph{fusion doublet}, degenerate until the
sectors are coupled. Its two eigenvectors are the physical states, so --- unlike the $N=1$ case, where the state is fixed by
$\mathsf G$ and $\Gamma_{\rm mol}$ alone --- their entanglement depends on the \emph{ratios} of the inter-sector couplings.
\item Coupling the sectors pairwise in different orders gives different bases of $\mathcal F$, related by a non-trivial
$2\times2$ unitary: the point-group recoupling ($6j$) coefficients, trivial for $N=1$, become physical.
\end{enumerate}
""")
    # natural bases table
    def d2s(d):
        return r"$\oplus$".join(("%d" % v if v > 1 else "") + "$" + k + "$" for k, v in d.items())
    s.append(r"""
\subsection{Natural bases of $\mathcal F$}
\begin{center}\small\setlength\tabcolsep{4pt}
\begin{tabular}{l l l c c c p{0.27\linewidth}}
\toprule
triple & ${\rm Sym}^2\Gamma$ & ${\rm Alt}^2\Gamma$ & ${\rm Sym}^3$ & ${\rm Alt}^3$ & mixed & natural basis of $\mathcal F$\\\midrule
""")
    s.append(r"$T$: $(T,T,T)\to A$ & %s & %s & %d & %d & %d & $|\epsilon|$ (S), $\epsilon$ (A)\\" % (
        d2s(sy["T"]["sym2"]), d2s(sy["T"]["alt2"]), sy["T"]["sym3"], sy["T"]["alt3"], sy["T"]["mixed"]))
    s.append(r"$I_{\rm h}$: $(G_g,H_g,H_g)\to A_g$ & %s & %s & -- & -- & -- & n$\leftrightarrow$v symmetric (S) / antisymmetric (A)\\" % (
        d2s(sy["H"]["sym2"]), d2s(sy["H"]["alt2"])))
    s.append(r"$I_{\rm h}$: $(H_g,H_g,H_g)\to A_g$ & %s & %s & %d & %d & %d & SO(3) rank $K\in\{2,4\}$ of any pair\\" % (
        d2s(sy["H"]["sym2"]), d2s(sy["H"]["alt2"]), sy["H"]["sym3"], sy["H"]["alt3"], sy["H"]["mixed"]))
    s.append(r"\bottomrule\end{tabular}\end{center}")
    s.append(r"""
\noindent\emph{$T$.} ${\rm Sym}^3T$ and ${\rm Alt}^3T$ each contain $A$ once, so $\mathcal F={\rm span}\{|\epsilon|,\epsilon\}$.
The two GHZ states $(|\epsilon|\pm\epsilon)/\sqrt2\propto\ket{xyz}+\ket{yzx}+\ket{zxy}$ and $\ket{xzy}+\ket{zyx}+\ket{yxz}$
(even and odd cyclic) also lie in $\mathcal F$.

\noindent\emph{$(G,H,H)$.} $G$ occurs once in ${\rm Sym}^2H$ and once in ${\rm Alt}^2H$, so the two states are distinguished
by whether the nuclear-spin and vibrational carriers first form their intermediate $G$ symmetrically or antisymmetrically.

\noindent\emph{$(H,H,H)$.} Both copies lie in ${\rm Sym}^3H$ (${\rm Sym}^3H\supset2A_g$; ${\rm Alt}^3H$ and the mixed part contain
none), so \emph{no permutation label} distinguishes them. The distinction comes instead from the chain
$\grp{SO}(3)\supset I$: $H=D^{(2)}|_I$, and a pair of $H$ carriers couples to $\grp{SO}(3)$ rank $K=0,\dots,4$, of which only
$K=2$ ($\to H$) and $K=4$ ($\to G\oplus H$) contain $H$. Therefore $\mathcal F={\rm span}\{\ket{K{=}2},\ket{K{=}4}\}$. The state
$\ket{K{=}2}$ is exactly the Wigner $3j$ coupling $(2\,2\,2)$, the one coupling a spherical body would also allow (computed weight of
the $3j$ tensor in $\mathcal F$: %s; overlap with $\ket{K{=}2}$: %s); $\ket{K{=}4}$ exists only because the symmetry is
icosahedral. The label $K$ is the same whichever pair of sectors is used (see below), so this basis is canonical.
""" % ("%.6f" % H["w3j"], "%.6f" % H["overlap_K2_with_3j"]))
    s.append(r"""
\subsection{Which states do the physical couplings select?}
The leading inter-sector couplings of a spherical top are products of vector ($T_{1g}$) operators: the Coriolis term
$-2B\zeta\,\mathbf J\cdot\boldsymbol\ell$ (r--v), nuclear spin--rotation $\sum c\,\mathbf J\cdot\mathbf I$ (r--n), and the vector part of
the spin--vibration coupling $\mathbf I\cdot\boldsymbol\ell$ (n--v). On an $H_g$ or $G_g$ carrier of $I_{\rm h}$ the vector operator is
unique up to a factor (${\rm Alt}^2H\supset T_{1g}$ once, $G\otimes G\supset T_{1g}$ once), so each of these is
$c_{pq}\,\mathbf L^{(p)}\!\cdot\mathbf L^{(q)}$ with $[L_x,L_y]=iL_z$ fixing the normalisation and all molecular detail in $c_{pq}$.
(In $T$ there are two vector operators on $T$; the angular-momentum one is $L_a=-i\epsilon_a$.) Restricting to $\mathcal F$:
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{$(H,H,H)$:} $\mathbf L^{(p)}\!\cdot\mathbf L^{(q)}|_{\mathcal F}={\rm diag}(%s,\,%s)$ in the basis
$\{\ket{K{=}2},\ket{K{=}4}\}$, \emph{identical for all three pairs} (%s), i.e.\ $\tfrac12[K(K{+}1)-12]$. Hence
\[V|_{\mathcal F}=(c_{\rm rn}+c_{\rm rv}+c_{\rm nv})\,{\rm diag}(-3,\,4):\]
an interval rule with splitting $7(c_{\rm rn}+c_{\rm rv}+c_{\rm nv})$, and eigenstates fixed independently of the coupling
ratios: the $3j$ state and the icosahedral $K{=}4$ state. They are SLOCC-inequivalent (second table in Sec.~\ref{sec:fusion_sphere}: tensor
ranks 8 and 7, different negativities and $\Lambda^2$). Vector couplings can split this doublet but never mix its members.
\item \textbf{$(G,H,H)$:} $\mathbf L^{(\rm n)}\!\cdot\mathbf L^{(\rm v)}$ is diagonal in the S/A basis with eigenvalues %s
($K=4$ for S, $K=3$ for A); $\mathbf L^{(\rm r)}\!\cdot\mathbf L^{(\rm n)}$ and $\mathbf L^{(\rm r)}\!\cdot\mathbf L^{(\rm v)}$ both have
eigenvalues %s but \emph{different} eigenvectors ($|\langle\cdot|\cdot\rangle|=%s$), and neither commutes with
$\mathbf L^{(\rm n)}\!\cdot\mathbf L^{(\rm v)}$ ($\|[\,\cdot,\cdot\,]\|=%s$). Coriolis, spin--rotation and spin--vibration coupling
therefore each prefer a different orientation of the fusion qubit, and the actual eigenstates rotate with the ratios
$c_{\rm rn}:c_{\rm rv}:c_{\rm nv}$. In a free molecule the Coriolis term ($\sim B\zeta$) dwarfs nuclear spin--rotation
(kHz), so the states are close to the Coriolis eigenstates (open circles in Fig.~\ref{fig:fusion}).
\item \textbf{$T$, $(T,T,T)$:} $\mathbf L\cdot\mathbf L|_{\mathcal F}={\rm diag}(+1,-1)$ on $\{|\epsilon|,\epsilon\}$ for every pair
($K=2$ and $K=1$): vector couplings select the two Levi-Civita states, never the GHZ states, with splitting
$2(c_{\rm rn}+c_{\rm rv}+c_{\rm nv})$.
\end{itemize}
""" % (f"{eH[0]:.0f}", f"{eH[1]:.0f}", "verified numerically" if H["same_for_all_pairs"] else "NOT equal",
       r"$\{%s,\,%s\}$" % (f"{eG['nv'][1]:.0f}", f"{abs(eG['nv'][0]):.0f}"),
       r"$\{%s,\,%s\}$" % (f"{eG['rv'][0]:.1f}", f"{eG['rv'][1]:.0f}"), "%.2f" % G["overlap_rv_rn"],
       "%.2f" % G["comm_rn_nv"]))
    s.append(r"""
\subsection{Entanglement across the fusion Bloch sphere}
\label{sec:fusion_sphere}
Write $\ket{C(\theta,\varphi)}=\cos\tfrac\theta2\ket{C_1}+e^{i\varphi}\sin\tfrac\theta2\ket{C_2}$ in the natural basis and evaluate
two LU invariants: the negativity $\mathcal N$ of each two-sector marginal (entanglement left between two sectors after the third
is discarded) and the geometric measure $E_{\rm geo}=-\log_2\Lambda^2$, $\Lambda^2=\max|\langle abc|C\rangle|^2$ over product states
(GHZ$_d$: $\log_2 d$; $\epsilon$: $\log_26$; $|\epsilon|$: $\log_2\tfrac92$). Both vary continuously over the sphere
(Fig.~\ref{fig:fusion}), so by item 1 each $N=2$ species contains a \emph{continuum of SLOCC-inequivalent} genuinely tripartite
states, all symmetry-allowed; which one the molecule is in is decided by the inter-sector couplings.
\begin{itemize}\setlength\itemsep{1pt}
\item $T$: $\mathcal N=%s$, $\Lambda^2=%s$. GHZ points on the equator ($\varphi=0,\pi$, circles) are the only points with
unentangled pairwise marginals; $\epsilon$ is the most entangled point ($\Lambda^2=1/6$).
\item $(H_g,H_g,H_g)$: on the grid $\mathcal N=%s$ (equal for all pairs, as both basis states are fully symmetric) and
$\Lambda^2=%s$. The maximum $\Lambda^2=\tfrac15$ is attained at isolated points that are exact GHZ$_5$ states (below).
\item $(G_g,H_g,H_g)$: on the grid $\mathcal N_{\rm rn}=\mathcal N_{\rm rv}=%s$, $\mathcal N_{\rm nv}=%s$ (the two equal $H$ carriers are
the most strongly paired almost everywhere), $\Lambda^2=%s$; again $\Lambda^2=\tfrac15$ is reached at isolated rank-5 points.
\end{itemize}
GHZPARAGRAPH
\begin{figure}[!tp]
\centering\includegraphics[width=\linewidth]{fig_fusion_qubit.pdf}
\caption{LU invariants over the fusion Bloch sphere $\ket{C(\theta,\varphi)}$ ($\theta=0$ top, $=\ket{C_1}$; $\theta=\pi$ bottom,
$=\ket{C_2}$; labels in the corners). Rows: $T$ with $C_{1,2}=|\epsilon|,\epsilon$ (circles: the two GHZ states);
$I_{\rm h}$ $(H_g,H_g,H_g)$ with $C_{1,2}=\ket{K{=}2}$ (the $3j$ state), $\ket{K{=}4}$; $I_{\rm h}$ $(G_g,H_g,H_g)$ with $C_{1,2}$ = S, A
(circles: Coriolis $\mathbf L^{\rm(r)}\!\cdot\mathbf L^{\rm(v)}$ eigenstates; triangles: spin--rotation $\mathbf L^{\rm(r)}\!\cdot\mathbf L^{\rm(n)}$
eigenstates; spin--vibration eigenstates are the poles). Stars: exact rank-5 states (GHZ$_5$ for $(H,H,H)$, GHZ-type for
$(G,H,H)$). Columns: geometric entanglement $E_{\rm geo}$, and the negativities of the
r--n and n--v marginals (r--v equals r--n in every row). Darker = more entangled. Grid $19\times36$; $\Lambda^2$ from 30
power-method restarts per point.}
\label{fig:fusion}
\end{figure}
""" % (rng_s("T", "N_rn"), rng_s("T", "L2"), rng_s("HHH", "N_rn"),
       rng_s("HHH", "L2"), rng_s("GHH", "N_rn"), rng_s("GHH", "N_nv"),
       rng_s("GHH", "L2")))
    s[-1] = s[-1].replace("GHZPARAGRAPH", ghz_paragraph(D))
    s.append(points_table(D, "T", r"Distinguished states of the $T$ fusion space $(T,T,T)\to A$."))
    s.append(points_table(D, "HHH", r"Distinguished states of $(H_g,H_g,H_g)\to A_g$ in $I_{\rm h}$; $K{=}2$ and $K{=}4$ are the "
                          r"eigenstates of every vector coupling.").replace(r"\end{center}", r"\label{tab:fusion_points_H}\end{center}", 1))
    s.append(points_table(D, "GHH", r"Distinguished states of $(G_g,H_g,H_g)\to A_g$ in $I_{\rm h}$. Ranks are the smallest found "
                          r"numerically; $\Lambda^2$ from 120 restarts."))
    # rotational subduction
    s.append(r"""
\subsection{Icosahedral molecules: common data}
Rotational species follow from $D^{(J)}|_{I}$ with parity $\sigma$ carried by the improper elements; $J^g$ gives the $g$ irreps
below and $J^u$ the same with $g\to u$:
\begin{center}\small
\begin{tabular}{c l}
\toprule $J$ & $\Gamma_{\rm r}$ ($\sigma=g$)\\\midrule
""")
    for J in range(0, 7):
        d = D["rot"][(J, "g")]
        s.append(r"%d & %s\\" % (J, r" $\oplus$ ".join(("%d" % v if v > 1 else "") + "$" + k + "$" for k, v in d.items())))
    s.append(r"\bottomrule\end{tabular}\end{center}")
    s.append(r"""
\noindent So $H_g$ first appears at $J=2$, $G_g$ at $J=3$. For identical spin-$\tfrac12$ nuclei on an orbit of $I_{\rm h}$,
$\chi_{\Gamma_{\rm mol}}(R)={\rm sgn}\,p(R)$ and $\chi^{\rm(n)}(R)=2^{c[p(R)]}$ (main text, Eqs.~(gamma\_mol) and
(nuclear\_spin\_character)). The $N=2$ species with $\Gamma_{\rm mol}=A_g$ are the %d triples below; each is realised by the lowest
$J^\sigma$ listed, by any of the $m^{\rm(n)}_{\Gamma_{\rm n}}$ nuclear-spin multiplets of that species, and by one quantum of a
fundamental of species $\Gamma_{\rm v}$.
""" % sum(1 for r in D["N2_rows"] if r["mol"] == "A_{g}"))
    s.append(r"""
\subsection{Example 1: dodecahedrane $\mathrm{C_{20}H_{20}}$}
Twenty CH units on a regular dodecahedron; $^{12}$C has $I=0$, so the permuted spins are the 20 protons. With C--C $=1.54$\,\AA\ and
C--H $=1.10$\,\AA\ (radii %s and %s\,\AA), the rigid spherical-top estimate $I=\tfrac23\sum m r^2$ gives $B\approx%s\ {\rm cm^{-1}}$.
""" % ("%.2f" % D["rC"], "%.2f" % D["rH"], "%.3f" % dod["B"]))
    s.append(molecule_block(D, dod, ""))
    s.append(r"""
\noindent Every operation is an even permutation of the protons ($\cls{12C_{5}}$, $\cls{12C_{5}^{2}}$: four 5-cycles; $\cls{20C_{3}}$: two fixed protons and six 3-cycles;
$\cls{15C_{2}}$, $\cls{i}$: ten transpositions; $\cls{15\sigma}$: four protons in the plane and eight transpositions; $\cls{12S_{10}}$,
$\cls{12S_{10}^{3}}$: two 10-cycles; $\cls{20S_{6}}$: one transposition and three 6-cycles), so $\Gamma_{\rm mol}=A_g$. The spin multiplicities sum to $\sum_\Gamma d_\Gamma m_\Gamma=2^{20}$.
The 114 vibrations include six $H_g$ and four each of $G_g$, $G_u$, $H_u$.

\emph{The fusion doublet.} The cleanest case is $(H_g,H_g,H_g)$: $J=2$, $\sigma=g$ (about $6B\approx%s\ {\rm cm^{-1}}$ above
$J=0$), any of the %s $H_g$ proton-spin multiplets, and one quantum in any of the six $H_g$ modes. For each such choice there are
two states. The Coriolis coupling of the excited $H_g$ mode, together with spin--rotation and spin--vibration, splits them by
$7(c_{\rm rn}+c_{\rm rv}+c_{\rm nv})$ into the $3j$ state $\ket{K{=}2}$ (lower if the sum is positive) and the icosahedral
state $\ket{K{=}4}$, which are SLOCC-inequivalent tripartite states. A coherent superposition of the two with the populations
given above becomes a five-level GHZ state of rotation, proton spin and the $H_g$ vibration twice per beat of the doublet. Adding $G_g$ rotation ($J=3$) or a $G_g$ spin species gives the
$(G,H,H)$ doublets, whose composition is tuned by the ratio of Coriolis to spin couplings.
""" % ("%.2f" % (6 * dod["B"]), "{:,}".format(dod["spin"]["H_{g}"]).replace(",", "\\,")))
    s.append(r"""
\subsection{Example 2: $^{13}\mathrm{C}_{60}$}
Sixty $^{13}$C ($I=\tfrac12$) on the truncated icosahedron (cage radius $3.55$\,\AA), giving $B\approx%s\ {\rm cm^{-1}}$
($%s\ {\rm cm^{-1}}$ for $^{12}$C$_{60}$ with the same estimate).
""" % ("%.5f" % c60["B"], "%.5f" % c60["B12"]))
    s.append(molecule_block(D, c60, ""))
    s.append(r"""
\noindent Again every operation is an even permutation ($\cls{12C_{5}}$, $\cls{12C_{5}^{2}}$: twelve 5-cycles; $\cls{20C_{3}}$: twenty 3-cycles; $\cls{15C_{2}}$, $\cls{i}$: thirty
transpositions; $\cls{15\sigma}$: four atoms in the plane and 28 transpositions; $\cls{12S_{10}}$, $\cls{12S_{10}^{3}}$: six 10-cycles;
$\cls{20S_{6}}$: ten 6-cycles), so
$\Gamma_{\rm mol}=A_g$, and $\sum_\Gamma d_\Gamma m_\Gamma=2^{60}$. The vibrational content is the familiar
$2A_g+3T_{1g}+4T_{2g}+6G_g+8H_g+A_u+4T_{1u}+5T_{2u}+6G_u+7H_u$, which serves as a check on the construction.

\emph{The fusion doublet.} $(H_g,H_g,H_g)$ is realised at $J=2^g$ (only $\approx%s\ {\rm cm^{-1}}$ above $J=0$), with any of the
$\approx%s$ $H_g$ nuclear-spin multiplets, and one quantum of any of the eight $H_g$ modes (e.g.\ the lowest Raman-active $H_g(1)$
squashing mode near $270\ {\rm cm^{-1}}$). The same interval rule, the same pair of eigenstates ($3j$ and $K{=}4$) and the same GHZ$_5$ recurrences
apply, now with the $^{13}$C nuclear spins as the spin sector.
\emph{Contrast with $^{12}$C$_{60}$}: with $I=0$ the only nuclear-spin species is $A_g$, every carrier triple has $d_{\rm n}=1$,
and nothing beyond rotation--vibration pairing exists; $^{13}$C enrichment is what opens the fusion space.

\subsection{Caveats}
Tensor ranks above $d_{\max}$ are the smallest found numerically. $\Lambda^2$ comes from a multi-start power method and can only
under-estimate the true maximum. The molecular geometries are idealised and the rotational constants are rigid-sphere estimates; the
statements about which states are selected cover vector-type ($T_{1g}$) couplings. Higher-rank inter-sector couplings (built from
the $G$ or $H$ operators on $H$ carriers, of which there are two each) are also allowed and can select other directions in
$\mathcal F$.
""" % ("%.3f" % (6 * c60["B"]), big(c60["spin"]["H_{g}"]).strip("$")))
    return "\n".join(s)


if __name__ == "__main__":
    print(fusion_tex()[:3000])
