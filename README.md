# Quantum theory of molecular phase space: species and entanglement tables

Code and tables for **symmetry-enforced entanglement between molecular rotation, nuclear spin and vibration**, for every point group of O(3).

A molecule with point group **G** has its total wavefunction fixed by spin statistics to a one-dimensional species Γ_mol. A rotational, nuclear-spin and vibrational species triple (Γ_r, Γ_n, Γ_v) is **admissible** when

N = (1/|G|) Σ_{g∈G} χ_r(g) χ_n(g) χ_v(g) χ*_mol(g) ≠ 0.

The admissible state is then fixed by the point-group Clebsch–Gordan tensor

|C⟩ = Σ C_{μ_r μ_n μ_v} |μ_r⟩_r |μ_n⟩_n |μ_v⟩_v,

and its entanglement across the three sectors is a property of the triple. This repository lists every admissible triple, its coupling tensor and its entanglement class. The irreps, their labels and the printed notation are those of o3harmonics (see [Conventions and o3harmonics](#conventions-and-o3harmonics)).

## Contents

| Path | What it is |
|---|---|
| [`species_tables/species_tables.pdf`](species_tables/species_tables.pdf) | The full tables (177 pages), with a worked analysis of the multiplicity-two cases |
| [`species_tables/species_data.json`](species_tables/species_data.json) | Every admissible triple with its class, family and nonzero coupling-tensor entries, keyed by this code's irrep labels; each irrep also carries its o3harmonics id, and each group its o3harmonics record (see [below](#species_datajson)) |
| `species_tables/pointgroups.py` | Builds the point groups and their irreps over ℂ. Every irrep of the 59 finite groups is o3harmonics', read in this code's convention and checked to be unitary at every element and multiplicative on a sample of rows of the multiplication table; the finite stand-ins of the linear groups keep their own matrices. Each irrep is labelled as o3harmonics labels the same state (the stand-ins by Σ, Π, …, which `o3h_labels.json` maps to o3harmonics' labels) and carries its o3harmonics id |
| `species_tables/entanglement.py` | Entanglement classifier: flattening ranks, numerical CP tensor rank, local-unitary fidelity |
| `species_tables/species.py` | Admissibility, coupling tensors, classification, also for SO(3) and O(3) (writes `results.pkl`) |
| `species_tables/make_pdf.py` | Builds the LaTeX tables and `species_data.json`, printing o3harmonics' irrep labels and group names through `o3h_tex.py` |
| `species_tables/o3h_irreps.json` | Exported from o3harmonics (commit 10dab5c): for each of the 59 finite groups, the row matrix P and the symbol of every element, and the id, label and matrices of every irrep |
| `species_tables/o3h_labels.json` | For each of the 65 groups, its o3harmonics record and, for each label of this code, o3harmonics' id, display name and LaTeX for the same state |
| `species_tables/o3h_print.json` | For each group, its printed name, its o3harmonics record, and the map of each irrep's partners to o3harmonics' where that map is a signed permutation (the identity, a sign on partner 2, or an exchange with a sign) |
| `species_tables/o3h_tex.py` | The printed forms: `label_tex` (an irrep, such as `\irr{a}_{1\ptag{g}}` or `\conjpair{2}\irr{e}_{1}`), `group_tex` (`\grp{C_{3v}}`), `element_tex` (an element by its o3harmonics symbol, such as `\elt{S_{3}}`), `partner_map`, and `occ` (an occupation ket in o3harmonics' partners) |
| `species_tables/o3h_macros.tex` | o3harmonics' LaTeX macros (10dab5c) without its `\ket` line and its six element-letter lines, followed by letter commands such as `\gel` (a group element) and `\gmat` (an element as its 3×3 matrix). Both PDFs load it |
| `species_tables/fusion_section.py` | Analysis of the N = 2 coupling spaces, the figure `fig_fusion_qubit.pdf`, and the C₂₀H₂₀ / ¹³C₆₀ examples |
| `species_tables/fusion_tex.py` | LaTeX for that section |
| `species_tables/ghz_points.py` | Searches the N = 2 spaces for GHZ-type members |
| `species_tables/check_paper_examples.py` | Regression check against the H₂O, NH₃/CaOCH₃ and CH₄ tables of the accompanying paper |
| [`vibrational_fock_basis/`](vibrational_fock_basis) | The vibrational sector: symmetry-adapted Fock states for any point group and excitation pattern (Wolfram Language with GTPack), a GTPack check of the multiplicities, and `salc_tables.py`, which tabulates the states of one degenerate mode for every finite group. See its [README](vibrational_fock_basis/README.md) |
| [`vibrational_fock_basis/salc_tables.pdf`](vibrational_fock_basis/salc_tables.pdf) | Symmetry-adapted Fock states (SALCs) of one degenerate vibrational mode, for the 59 finite groups of the species tables, with the same labels and in this code's own two-dimensional bases, whose map to o3harmonics' partners the PDF states. Organised by parity, residues of the vibrational angular momentum ℓ (2D modes), and harmonic layers H_ℓ (vector-like 3D modes), with worked examples at finite v, and the laws for combination bands (total ℓ, coupling of harmonic layers) with per-group tables. Generated and checked by `salc_tables.py` |

## Scope

**Groups.** All closed subgroups of O(3):
- C₁, C₁ₕ (also written C_s), S₂ (also written C_i);
- the seven axial families C_n, S₂ₙ, C_nh, C_nv, D_n, D_nh, D_nd for n = 2…8;
- T, T_h, T_d, O, O_h, I, I_h;
- C∞, C∞h, C∞v, D∞h, truncated at Ξ ≤ 3, with Ξ the subscript n of E_n, ¹E_n and ²E_n, and 0 for the A irreps (computed exactly with C₁₃ / C₁₄h / C₁₃v / D₁₄h, whose invariants coincide with the continuous groups' in that range);
- SO(3) and O(3), truncated at J ≤ 3 (coupling tensors equal to the Wigner 3j symbols up to an overall sign).

**Γ_mol.** The spin-statistics character χ_mol(g) = Π_a sign[p_a(g)]^{2I_a} only takes the values ±1. Each group therefore gets one table for every **real one-dimensional irrep**.

**Species.** Every irrep is allowed in every sector, which makes the tables pure group theory. Molecule-specific restrictions, such as which nuclear-spin species a given set of nuclei can carry, are not imposed.

## Conventions and o3harmonics

The irreps, their labels and the printed notation follow o3harmonics, a catalogue of the closed subgroups of O(3) and their irreps, at commit 10dab5c. The two codes write the action of an element differently, so the tables state a dictionary.

- **Action.** Here group elements act on states actively, by U(g); matrices act on column vectors, and products compose from right to left. o3harmonics uses row vectors, r ↦ r P(g), composes gh as "g first", and lets g act on a function by (g f)(r) = f(r P(g)), that is by U(g)⁻¹. Here the same physical operation acts on column vectors by P(g)ᵀ, the transpose of its o3harmonics row matrix.
- **Irreps.** Every irrep of the 59 finite groups is imported from o3harmonics and read as D(g) := D_o3h(phys g)^† = D_o3h((phys g)⁻¹), where phys g is the o3harmonics element of the same physical operation. States are identified by their wavefunctions, so the irrep names, the partners μ and the coupling tensors are o3harmonics' for the same state, while the matrices are its matrices at the inverse element. For example, U(C₃) acts on the two partners of E in C₃ᵥ as diag(ζ₃, ζ₃²), ζ₃ = e^{2πi/3}, which is o3harmonics' matrix at C₃⁻; at C₃⁺ it tabulates diag(ζ₃², ζ₃).
- **Complex pairs.** Irreps are irreducible over ℂ, so complex-conjugate pairs of one-dimensional irreps are listed separately, as ¹E_k and ²E_k, named as o3harmonics names the same state: ²E_k is the member whose character on the generating rotation or rotoreflection is e^{+2πik/n} under U(g), the complex conjugate of the value o3harmonics tabulates. Earlier versions of these tables called that member ¹E_k, so every complex-pair label has changed places, in the PDFs and in `species_data.json`.
- **Partners.** The two partners of a real two-dimensional irrep are μ = 1, 2 (earlier versions wrote them ¹E, ²E, the symbols o3harmonics keeps for complex pairs). The partners of a three-dimensional irrep are μ = 1, 2, 3.
- **Bases.**
  - The two-dimensional irreps of the axial groups are in o3harmonics' basis of monomials (x − iy)^ℓ z^δ, (x + iy)^ℓ z^δ (δ = 0 or 1), in which the principal rotation (S₂ₙ for D_nd with n even) is diagonal and C₂′ or σ_v exchanges the partners. For 52 of them partner 1 has the eigenvalue e^{+iθ} with 0 < θ < π, and the exchange is plain, as in earlier versions. The 18 irreps with δ = 1, such as E″ of D₃ₕ and E_g of D₃d, have a minus sign on the exchange, and in E₃ of D₄d, E₄ and E₅ of D₆d and E₅ to E₇ of D₈d the two partners are exchanged, with a sign, relative to that rule.
  - In the cubic E, C₃ about [111] acts as diag(ζ₃, ζ₃²), and σ_d (x ↔ y) as the plain exchange (in O and O_h, the C₂ about [1 −1 0]).
  - Cubic T irreps are Cartesian (x, y, z = μ = 1, 2, 3), with C₃ about [111] cyclic (x → y → z).
  - The icosahedral G and H are in o3harmonics' basis.
  - SO(3) and O(3) use |J m⟩, m = J, …, −J, o3harmonics' partner order.
- **What the import changed.** Of the 455 irreps of the 59 finite groups and the linear stand-ins, 425 keep the matrices of earlier versions, and 30 changed: the 24 two-dimensional irreps above and the six icosahedral G and H. The rows with those carriers are re-expressed in o3harmonics' basis. The admissible triples and their classes are unchanged, and the tensors of the paper's T_d example are reproduced term for term.
- **Linear groups.** The finite stand-ins keep their own complex bases, with the exchange as the plain permutation, and take o3harmonics' labels and ids. The tables print their irreps by o3harmonics' names: A₁, A₂, E₁, E₂, E₃ for the Σ⁺, Σ⁻, Π, Δ, Φ of C∞v, with the parity tags and the complex-pair prefixes where they apply (in D∞h, Σ⁺_u is A₂ᵤ). `species_data.json` keeps the Greek labels as keys.
- **The SALC tables** keep this code's own two-dimensional bases, because their residue rules and Law 1 need an exchange that acts as the plain permutation. They print o3harmonics' labels and give the map of their kets to o3harmonics' partners (see the [vibrational README](vibrational_fock_basis/README.md#salc-tables-for-all-point-groups)).
- **Notation in the PDFs.** Both PDFs print as o3harmonics' documents do: irrep labels in small capitals (`\irr{a}_{1}`, `\conjpair{2}\irr{e}_{1}`, `\irr{t}_{1\ptag{u}}`), with the parity tags upright; groups in sans serif (`\grp{C_{3v}}`, `\grp{SO}(3)`); elements and classes in a slanted sans serif, the element font (`\elt{C_{3}}`, `\cls{12C_{5}}`); the irreps of SO(3) and O(3) as D^{(J)} and D^{(J,±)}; roots of unity as ζ_n = e^{2πi/n}; and i and e upright. The macros are in `species_tables/o3h_macros.tex`. A group element is written by the letter command `\gel`, so its letter and font are set in one place; it prints g in the element font.

### species_data.json

Every key of earlier versions is kept. Each group has `group`, `order` (for a linear group the order of its finite stand-in; null for SO(3) and O(3)), `irreps` and `tables`, with one table per Γ_mol, keyed by its label. Each row has `r`, `n`, `v`, `dims`, `N` and `cls`; the rows of the non-abelian groups also have `tau` (the copy, counted from 1, which the PDF prints as κ), `sym`, `nature`, `rank`, `family` and `tensor`. Tensor entries are [a, b, c, Re, Im], with indices from 0, so (a, b, c) = (μ_r − 1, μ_n − 1, μ_v − 1).

The labels are this code's strings: o3harmonics' names for the finite groups (such as `A_{1g}`, `E''`, `{}^{2}E_{1}`), with the complex-pair prefixes changed as described above, so that `{}^{2}E_{k}` is the member with character e^{+2πik/n}; `\Sigma^{+}`, `\Pi`, … for the linear groups; `\mathcal D^{(J)}` for SO(3), and the same with a subscript g or u for O(3). Two fields are new. Each irrep has its o3harmonics `id` (such as `A1`, `2E1`, `E1g`, `D2u`), and each group has its `o3h_record`, the name of its o3harmonics record (`C1h` for C₁ₕ, `S2` for S₂, `SO3` and `O3` for SO(3) and O(3), and otherwise the group's own name).

## Entanglement classes

Every reduced state of |C⟩ on a single sector is 𝟙/d (Schur's lemma). These states are therefore critical, so SLOCC equivalence coincides with LU equivalence (Kempf–Ness).

| Label | Nature | Criterion |
|---|---|---|
| separable | separable | all carriers one-dimensional |
| bipartite | bipartite | exactly one carrier one-dimensional; the other two form a maximally entangled d×d pair |
| GHZ | GHZ-type tripartite | equal dims d, tensor rank d, LU-equivalent to GHZ_d |
| GHZ-type | GHZ-type tripartite | unequal dims, tensor rank = d_max (a compressed GHZ state) |
| LC(ε) | Levi-Civita-type tripartite | (3,3,3), LU-equivalent to ε_ijk/√6 (rank 5) |
| LC(\|ε\|) | Levi-Civita-type tripartite | (3,3,3), LU-equivalent to \|ε_ijk\|/√6 (rank 4; inequivalent to ε) |
| tri (r = R) | beyond GHZ/LC | genuinely tripartite, rank R > d_max |

The method is as follows:
- The lower bound d_max on tensor rank is the flattening rank.
- The upper bound is certified by an explicit, well-conditioned CP decomposition with residual < 10⁻⁹ (ALS + Levenberg–Marquardt, many random starts). GHZ-type assignments are therefore exact.
- Ranks above d_max are the smallest found numerically.
- LU equivalence is tested by maximising |⟨S|U₁⊗U₂⊗U₃|C⟩|.

## Main results

1. **Abelian groups** are separable only. **Axial non-Abelian groups** (all irreps of dimension ≤ 2) give only separable, Bell-pair bipartite and GHZ (2,2,2) states.
2. **Cubic groups** add (1,3,3)-type bipartite states, GHZ-type (2,3,3) states of rank 3 (the paper's family (e±)), and both Levi-Civita types. ε and |ε| are SLOCC-inequivalent (ranks 5 and 4).
3. **Multiplicity two.** N = 2 occurs for (T,T,T) in T and T_h (because T⊗T ⊃ 2T), and for (G,H,H) and (H,H,H) in I and I_h. The coupling space is then a two-dimensional *fusion qubit*:
   - every single-sector symmetric operator acts on it as a scalar;
   - only inter-sector couplings split it.
4. **Which fusion states the couplings select.** Vector couplings (Coriolis, spin–rotation, spin–vibration) behave differently in each N = 2 case:
   - **(H,H,H):** they act on the fusion qubit as diag(−3, 4) for every pair of sectors. This is an interval rule whose eigenstates are the SO(3) 3j state (K = 2) and an icosahedral K = 4 state, with tensor ranks 8 and 7.
   - **(G,H,H):** the three pairwise couplings do not commute, so the physical state depends on the ratios of the coupling constants.
   - **T:** they select ε and |ε|, never the GHZ members of the space.
5. **Entanglement varies continuously across each fusion space**, as measured by pairwise negativities and the geometric measure. Each N = 2 species therefore contains a continuum of SLOCC-inequivalent tripartite states.
6. **Exact five-level GHZ states in I_h.** The (H_g,H_g,H_g) fusion space contains exactly two states LU-equivalent to GHZ₅ (rank-5 residual ~10⁻¹⁵, fidelity 1). Both have populations 4/7 in K = 2 and 3/7 in K = 4. Free evolution under the vector couplings only advances the relative phase, so a superposition with those populations passes through GHZ₅ twice per beat of the doublet. (G_g,H_g,H_g) has two rank-5 GHZ-type points (populations 3/8 S, 5/8 A).
7. **Molecular realisations.** Dodecahedrane C₂₀H₂₀ and ¹³C₆₀ both have Γ_mol = A_g. Each carries H_g and G_g nuclear-spin species and H_g, G_g, H_u, G_u vibrational fundamentals. The (H_g,H_g,H_g) fusion doublet is reached at J = 2 with one H_g vibrational quantum.

## Running

```bash
pip install -r requirements.txt
cd species_tables
python species.py 8              # all groups, n <= 8; writes results.pkl (cached per group in results/)
python fusion_section.py         # N = 2 analysis, figure, molecules; writes fusion_data.pkl
python ghz_points.py             # locates the exact GHZ-type points (reads fusion_data.pkl)
python fusion_section.py         # second pass adds those points to the figure and data
python make_pdf.py results.pkl   # writes species_tables.tex and species_data.json
pdflatex species_tables.tex      # run twice for the table of contents
python check_paper_examples.py   # regression check against the paper's molecular tables
```

From a cold cache, `species.py` takes about 6½ minutes for all 65 groups, and `fusion_section.py`, `ghz_points.py` and `fusion_section.py` again about 5 minutes together, most of it in `ghz_points.py` (measured on one workstation). `vibrational_fock_basis/salc_tables.py` takes about 1 minute. Each group's result is cached in `results/<group>.pkl` and the classifications in `class_cache.pkl`, so later runs are fast; delete `results/` and the `.pkl` files to regenerate from scratch.

## Caveats

- Tensor ranks above d_max are numerical upper bounds. Permuted copies of the same 4×5×5 triple sometimes come out one apart, so treat those values as ±1.
- The largest product-state overlap Λ², from which the geometric measure E_geo = −log₂Λ² is computed, comes from a multi-start power method. It can only underestimate the true maximum.
- The molecular geometries are idealised, and the rotational constants are rigid-sphere estimates.
- The statement about which fusion states are selected covers only vector-type couplings. Higher-rank inter-sector couplings can select other states.

## Requirements

Python ≥ 3.10 with NumPy, SciPy, mpmath and Matplotlib (and SymPy for `vibrational_fock_basis/salc_tables.py`); a LaTeX installation (pdflatex, booktabs, longtable) to build the PDFs. `fusion_section.py` sets the text of its figure with LaTeX through Matplotlib, so it also needs the lmodern and pgf packages, and dvipng for the preview `build/fig_fusion_qubit.png`.

The Wolfram Language scripts in `vibrational_fock_basis/` need Mathematica or the Wolfram Engine (tested 14.3), and GTPack (tested 1.4).
