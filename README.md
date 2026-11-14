# Molecular Entanglement Zoo: Admissible rovibrational species and their symmetry-enforced entanglement

Identical-particle statistics selects the physical states of a molecule. These states factor into multiplicity spaces that symmetry leaves free and coupling tensors that it fixes, independently of the molecular Hamiltonian (E. R. Das, J.-C. Zhang, V. V. Albert and Y. Liu, *Symmetry classification of hyperfine rovibrational molecular phase space*, in preparation). This repository computes these coupling tensors for the 75 point groups of Altmann and Herzig's *Point-Group Theory Tables*. It also gives the **symmetry-enforced entanglement** that each tensor carries between rotation, nuclear spin and vibration.

A molecule with point group **G** has its total wavefunction fixed by spin statistics to a one-dimensional species $`\Gamma_{\rm mol}`$. A triple $`(\Gamma_r,\Gamma_n,\Gamma_v)`$ of rotational, nuclear-spin and vibrational species is **admissible** when

```math
N=\frac{1}{|G|}\sum_{g\in G}\chi_r(g)\,\chi_n(g)\,\chi_v(g)\,\chi^*_{\rm mol}(g)\neq0 .
```

For N = 1 the coupling tensor of the triple is then the point-group Clebsch–Gordan tensor

```math
|C\rangle=\sum_{\mu_r\mu_n\mu_v}C_{\mu_r\mu_n\mu_v}\,|{}^{\Gamma_r}{}_{\,\mu_r}\rangle|{}^{\Gamma_n}{}_{\,\mu_n}\rangle|{}^{\Gamma_v}{}_{\,\mu_v}\rangle ,
```

and its entanglement across the three sectors is a property of the triple. For N = 2 the tables list a basis of two such tensors, and the members of the coupling space are not all equivalent under local unitaries. The tensors that the tables list for a triple are its copies. Each coupling tensor is an absolutely maximally entangled (AME) state. With each N = 2 space given by its two copies, the coupling tensors of the 75 groups are 21 such states up to local unitaries and the order of the carriers. This repository lists every admissible triple, its coupling tensor and its AME state, given by a catalogue id (two ids for the 45 triples with N = 2). The irreps, their labels and the printed notation are those of o3harmonics (see [Conventions and o3harmonics](#conventions-and-o3harmonics)).

## Contents

| Path | What it is |
|---|---|
| [`species_tables/species_tables.pdf`](species_tables/species_tables.pdf) | The full tables (138 pages), with the catalogue of the 21 AME states (Sec. 2) |
| [`species_tables/species_data.json`](species_tables/species_data.json) | Every admissible triple with its catalogue id (`P`, `B1` to `B4`, `T1` to `T16`) and, outside the 33 Abelian groups, its nonzero coupling-tensor entries, keyed by this code's irrep labels. Each irrep also carries its o3harmonics id, and each group its o3harmonics record (see [below](#species_datajson)) |
| [`species_tables/ame_catalogue.json`](species_tables/ame_catalogue.json) | The catalogue: for each of the 21 AME states its format, name, N, 3j block and a representative triple with its tensor (see [below](#ame_cataloguejson)) |
| `species_tables/pointgroups.py` | Builds the point groups and their irreps over ℂ. Every irrep of the 73 finite groups is o3harmonics', read in this code's convention and checked to be unitary at every element and multiplicative on a sample of rows of the multiplication table. The finite stand-ins of the linear groups keep their own matrices. Each irrep is labelled as o3harmonics labels the same state (the stand-ins by Σ, Π, …, which `o3h_labels.json` maps to o3harmonics' labels) and carries its o3harmonics id |
| `species_tables/ame.py` | The catalogue of AME states (`CATALOGUE`) and its exact checks: flat single-carrier marginals, the pair marginals of the N = 1 tensors, the K basis of the N = 2 spaces, and each tensor's catalogue id (`lookup`), from exact LU invariants internal to the code. NumPy only. `python ame.py` runs its self-test |
| `species_tables/species.py` | Admissibility, coupling tensors and catalogue ids for the 75 groups (writes `results.pkl`) |
| `species_tables/make_pdf.py` | Builds the LaTeX tables, `species_data.json` and `ame_catalogue.json`, printing o3harmonics' irrep labels and group names through `o3h_tex.py`. On `results.pkl` it re-checks every copy (its norm, invariance, flat marginals and catalogue id) and certifies that the copies of each state are LU-equivalent up to the order of the carriers. It also re-checks that the tables list every admissible triple once, the rules of Sec. 2, the groups of the occurs-in column of Table 1 and the claims of Sec. 3, and it stops on a mismatch |
| `species_tables/o3h_irreps.json` | Exported from o3harmonics (commit 99b3710): for each of the 73 finite groups, the row matrix P of every element, and the id and matrices of every irrep |
| `species_tables/o3h_labels.json` | For each of the 75 groups, its o3harmonics record and, for each label of this code, o3harmonics' id and LaTeX for the same state |
| `species_tables/o3h_print.json` | For each of the 75 groups, its printed name. For each two-dimensional irrep of the linear stand-ins, which keep their own basis, the map from its partners to o3harmonics': the identity, or minus partner 2 (see *Linear groups* [below](#conventions-and-o3harmonics)). The print-only entries `K` and `Kh` are the printed names of SO(3) and O(3), which have no tables |
| `species_tables/o3h_tex.py` | The printed forms `label_tex` (an irrep, such as `\irr{a}_{1\ptag{g}}` or `\conjpair{2}\irr{e}_{1}`) and `group_tex` (`\grp{C_{3v}}`), and `partner_map` (the partner map of a stand-in's irrep in `o3h_print.json`) |
| `species_tables/o3h_macros.tex` | o3harmonics' LaTeX macros (99b3710), without its `\ket` line and its six element-letter lines. `make_pdf.py` puts them in the preamble of the PDF |
| `species_tables/o3h_style.tex` | o3harmonics' layout (99b3710): the headings, contents, running heads, link colour and equation numbers of its documents, with a margin of 1.6 cm (1.8 cm there). `make_pdf.py` puts it in the preamble of the PDF |
| `species_tables/check_examples.py` | Checks three worked molecules, H₂O, NH₃/CaOCH₃ and CH₄, against values written out by hand: admissible rows, catalogue ids and coupling tensors, with exit status 1 on any mismatch |
| [`abinitio/`](abinitio/README.md) | Scripts, inputs and raw outputs of the ab initio calculations behind the calcium monomethoxide (CaOCH₃) example of the paper: the Coriolis constant ζ₈ (EA-EOM-CCSD), the anharmonic constants x₈,₈ and g₈,₈ and the bend transition dipole μ₈ (B3LYP), and the level, line, pulse and magnetic-field numbers of its appendix. [Below](#ab-initio-constants-for-calcium-monomethoxide) says how each number was computed and which script and stored output gives it |

## Scope

**Groups.** The 75 groups of Altmann and Herzig's *Point-Group Theory Tables* (2011):
- C₁, C₁ₕ (also written $`C_s`$), S₂ (also written $`C_i`$);
- the seven axial families $`C_n`$, $`S_{2n}`$, $`C_{nh}`$, $`C_{nv}`$, $`D_n`$, $`D_{nh}`$, $`D_{nd}`$ for n = 2…10;
- T, $`T_h`$, $`T_d`$, O, $`O_h`$, I, $`I_h`$;
- C∞v and D∞h, truncated at Ξ ≤ 3, with Ξ the subscript of $`E_\Xi`$ and 0 for the A irreps (computed exactly with C₁₃v and D₁₄h, whose invariants coincide with the continuous groups' in that range).

C∞ and C∞h are left out because nuclei on a line always lie in vertical mirror planes. So are O(3), the point group of an atom, and SO(3), which is the point group of no configuration of nuclei.

**$`\Gamma_{\rm mol}`$.** The orientation of a molecule ranges over O(3), so a rotational level of angular momentum J and parity ± carries the irrep $`D^{(J,\pm)}`$ of O(3). Its species $`\Gamma_r`$ are those of this irrep restricted to G, and spin statistics then fixes $`\Gamma_{\rm mol}`$. If g induces the permutation $`p_a(g)`$ of the identical nuclei of isotope a, each of spin $`I_a`$, then $`\chi_{\rm mol}(g)=\prod_a\mathrm{sign}[p_a(g)]^{2I_a}`$. This character only takes the values ±1, so each group gets one table for every **real one-dimensional irrep**.

**Species.** Every irrep is allowed in every sector, which makes the tables pure group theory. The tables impose no molecule-specific restriction, such as a limit on the nuclear-spin species of a given set of nuclei. For example, the four protons of methane carry only the nuclear-spin species A₁, E and T₂, which keep 26 of the 43 $`T_d`$ triples with $`\Gamma_{\rm mol}=A_2`$.

## Conventions and o3harmonics

The irreps, their labels and the printed notation follow o3harmonics, a catalogue of the closed subgroups of O(3) and their irreps, at commit 99b3710. The two codes write the action of an element differently, so the tables state a dictionary.

- **Action.** Here group elements act on states actively, by U(g). Matrices act on column vectors, and products compose from right to left. o3harmonics uses row vectors, r ↦ r P(g), composes gh as "g first", and lets g act on a function by (g f)(r) = f(r P(g)), that is by U(g)⁻¹. Here the same physical operation acts on column vectors by P(g)ᵀ, the transpose of its o3harmonics row matrix.
- **Irreps.** Every irrep of the 73 finite groups is imported from o3harmonics and read as $`D(g):=D_{\rm o3h}(\mathrm{phys}\,g)^\dagger=D_{\rm o3h}\big((\mathrm{phys}\,g)^{-1}\big)`$, where phys g is the o3harmonics element of the same physical operation. States are identified by their wavefunctions, so the irrep names, the partners μ and the coupling tensors are o3harmonics' for the same state, while the matrices are its matrices at the inverse element. For example, U(C₃) acts on the two partners of E in C₃ᵥ as diag(ζ₃, ζ₃²), $`\zeta_3=e^{2\pi i/3}`$, which is o3harmonics' matrix at C₃². At C₃ o3harmonics tabulates diag(ζ₃², ζ₃).
- **Complex pairs.** Irreps are irreducible over ℂ, so complex-conjugate pairs of one-dimensional irreps are listed separately, as $`{}^1E_k`$ and $`{}^2E_k`$, named as o3harmonics names the same state. Under U(g) their characters are the complex conjugates of o3harmonics' values, so ²E has character i at C₄ in C₄ and at S₄ in S₄.
- **Partners.** A real two-dimensional irrep has the partners μ = 1, 2, and a three-dimensional irrep has the partners μ = 1, 2, 3.
- **Bases.**
  - The two-dimensional irreps of the axial groups are in o3harmonics' basis of monomials $`(x-iy)^\ell z^\delta`$, $`(x+iy)^\ell z^\delta`$ (δ = 0 or 1), in which the principal rotation ($`S_{2n}`$ for $`D_{nd}`$ with n even) is diagonal and C₂′ or $`\sigma_v`$ exchanges the partners, up to a sign.
  - In the cubic E, C₃ about [111] acts as diag(ζ₃, ζ₃²), and $`\sigma_d`$ (x ↔ y) as the plain exchange (in O and $`O_h`$, the C₂ about [1 −1 0]).
  - Cubic T irreps are Cartesian (x, y, z = μ = 1, 2, 3), with C₃ about [111] cyclic (x → y → z).
  - The icosahedral G and H are in o3harmonics' basis.
- **Linear groups.** The finite stand-ins keep their own complex bases, with the exchange as the plain permutation, and take o3harmonics' labels and ids. Their partners are o3harmonics', except that partner 2 of $`E_{1g}`$, $`E_{3g}`$ and $`E_{2u}`$ of D∞h ($`\Pi_g`$, $`\Phi_g`$ and $`\Delta_u`$) is minus o3harmonics' partner 2. The tables print their irreps by o3harmonics' names: A₁, A₂, E₁, E₂, E₃ for the Σ⁺, Σ⁻, Π, Δ, Φ of C∞v. D∞h has the same names with parity tags, except that $`\Sigma^+_u`$ is $`A_{2u}`$ and $`\Sigma^-_u`$ is $`A_{1u}`$. `species_data.json` keeps the Greek labels as keys.
- **Notation in the PDF.** The PDF prints as o3harmonics' documents do: irrep labels in small capitals (`\irr{a}_{1}`, `\conjpair{2}\irr{e}_{1}`, `\irr{t}_{1\ptag{u}}`), with the parity tags upright; groups in sans serif (`\grp{C_{3v}}`, `\grp{SO}(3)`); elements in a slanted sans serif, the element font (`\elt{C_{3}}`); roots of unity as $`\zeta_n=e^{2\pi i/n}`$; and i and e upright. Three letters differ from o3harmonics', because the PDF gives o3harmonics' letters other roles. The SO(3) label is j (o3harmonics' J, here the angular momentum of a rotational level). The parameter of an axial family is n (o3harmonics' N, here the dimension of a coupling space). Ξ is the index of an irrep of a linear group (o3harmonics' n, here the family parameter). The macros are in `species_tables/o3h_macros.tex`, and the layout of the headings and contents is o3harmonics' too (`species_tables/o3h_style.tex`). Many symbols are commands defined once in the preamble that `make_pdf.py` writes, so their letter and font are set in one place. For example, a group element is written `\gel`, which prints g in the element font, and the catalogue ids are written `\catP`, `\catB{1}` and `\catT{7}`, in the calligraphic font of `\catfont`.

### species_data.json

The top level is a list of groups. Each group has `group`, `o3h_record`, `order` (for a linear group, the order of its finite stand-in), `irreps` and `tables`. Each irrep has `label`, `id` and `dim`, and `tables` has one table per $`\Gamma_{\rm mol}`$, keyed by its label.

Each row has `r`, `n`, `v`, `dims`, `N` and `state`, the catalogue id (`P`, `B1` to `B4` or `T1` to `T16`). In the non-Abelian groups each copy has its own row, which also has `tau` (the copy, counted from 1) and `tensor`. The copies of an N = 2 triple are its K basis in ascending K, and their rows also carry `K`. Tensor entries are [a, b, c, Re, Im], with indices from 0, so $`(a,b,c)=(\mu_r-1,\mu_n-1,\mu_v-1)`$. The first nonzero entry of every tensor is real and positive (to rounding).

The labels are this code's strings. For the finite groups they are o3harmonics' names, such as `A_{1g}`, `E''` and `{}^{2}E_{1}`, with complex pairs named as in *Complex pairs* [above](#conventions-and-o3harmonics). The linear groups have `\Sigma^{+}`, `\Pi`, and so on. Each irrep also has its o3harmonics `id`, such as `A1`, `2E1` or `E1g`. Each group has its `o3h_record`, the name of its o3harmonics record: `C1h` for C₁ₕ, `S2` for S₂, and otherwise the group's own name.

### ame_catalogue.json

This file lists the 21 catalogue states in the order of the table [below](#the-catalogue-of-ame-states). Each entry has:
- `id` (`P`, `B1` to `B4`, `T1` to `T16`), `format` (the sorted carrier dimensions) and `name`;
- `N`, the dimension of the representative's coupling space, and `N_text`, the N column as printed (`1; 2 in T, Th` for `T3` and `T4`);
- `block3j`, null for `P` and `B1` to `B4`, and otherwise an object with
  - `js`, the (j₁, j₂, j₃) of a Wigner 3j symbol of smallest j₁ + j₂ + j₃ whose block is the state, in the representative's carrier order (r, n, v);
  - `literal`, true when every carrier is a whole $`D^{(j)}`$;
  - `group`, the group of the block (O for `T1`, otherwise that of the representative);
- `representative`, a triple of the tables with this state: `group`, `r`, `n`, `v`, `mol` and `K` (null for N = 1);
- `dims` and `tensor`, the representative's tensor in the form of `species_data.json` (for `P`, whose C₁ row stores none, the number 1).

## The catalogue of AME states

Every reduced state of |C⟩ on a single carrier X is $`\mathbb{1}/d_X`$ (Schur's lemma; Proposition 2.1 of the PDF), and so is that of every member of an N = 2 coupling space. Each coupling tensor is therefore an absolutely maximally entangled (AME) state of its format, the sorted carrier dimensions. Every cut $`X\vert YZ`$ has a flat Schmidt spectrum of rank $`d_X`$. With no multidimensional carrier the state is a product, with two it is a maximally entangled pair (a single one cannot occur), and with three it has genuine tripartite entanglement. AME states are critical, so SLOCC equivalence coincides with LU equivalence (Kempf–Ness). For N = 1, the state $`\rho_{YZ}`$ of two carriers Y, Z is the projector onto the $`\Gamma^*_X\otimes\Gamma_{\rm mol}`$ isotypic component of $`\Gamma_Y\otimes\Gamma_Z`$, divided by $`d_X`$, with X the third carrier.

Bryan, Reichstein and Van Raamsdonk, who call AME states locally maximally entangled, give the dimension of the space of AME states of each format up to LU ([arXiv:1708.01645](https://arxiv.org/abs/1708.01645), Thms. 1.2 and 1.3; Ann. Henri Poincaré 19, 2491 (2018)). In the formats (1,d,d), (2,2,2) and (2,3,3) this space is a single point, so AME alone fixes the state up to LU. In the other formats of the catalogue the AME states form a continuous family, from which the point groups pick particular members.

In all, the coupling tensors of the 75 groups, with each N = 2 space in its K basis, are 21 states up to LU and the order of the carriers. They are the product state `P`, the four maximally entangled pairs `B1` to `B4` and the 16 tripartite states `T1` to `T16`. No two of them are related in this way, and `ame.py` checks this exactly. A tensor that matches none of them stops the build. For `T3`, `T4` and `T13` to `T16`, K labels the two copies of an N = 2 space. Its triple has two equal carriers, T or H, and K is their total angular momentum (item 4 below). The 21 states are those of `ame.CATALOGUE`:

| id | dims | state description | N |
|---|---|---|---|
| `P` | (1,1,1) | product | 1 |
| `B1` | (1,2,2) | maximally entangled pair | 1 |
| `B2` | (1,3,3) | maximally entangled pair | 1 |
| `B3` | (1,4,4) | maximally entangled pair | 1 |
| `B4` | (1,5,5) | maximally entangled pair | 1 |
| `T1` | (2,2,2) | GHZ | 1 |
| `T2` | (2,3,3) | trine-GHZ | 1 |
| `T3` | (3,3,3) | ε | 1; 2 in T, $`T_h`$ |
| `T4` | (3,3,3) | \|ε\| | 1; 2 in T, $`T_h`$ |
| `T5` | (3,3,4) | coupling T₁ ⊗ T₂ ⊃ G | 1 |
| `T6` | (3,3,5) | Wigner 3j(1,1,2) | 1 |
| `T7` | (3,3,5) | coupling T₁ ⊗ T₂ ⊃ H | 1 |
| `T8` | (3,4,4) | 't Hooft symbol | 1 |
| `T9` | (3,4,5) | coupling $`T_a\otimes G\supset H`$ | 1 |
| `T10` | (3,5,5) | Wigner 3j(1,2,2) | 1 |
| `T11` | (4,4,4) | simplex cubic | 1 |
| `T12` | (4,4,5) | coupling G ⊗ G ⊃ H | 1 |
| `T13` | (4,5,5) | the two H carriers coupled to K = 3 (antisymmetric) | 2 |
| `T14` | (4,5,5) | the two H carriers coupled to K = 4 (symmetric) | 2 |
| `T15` | (5,5,5) | Wigner 3j(2,2,2) (K = 2) | 2 |
| `T16` | (5,5,5) | K = 4 coupling | 2 |

$`T_a`$ stands for T₁ or T₂. Two admissible triples that differ only in their g and u tags have the same state. In I and $`I_h`$, replacing every T₁ of a triple by T₂ and every T₂ by T₁ keeps it admissible, with the same state. `make_pdf.py` checks both facts on every admissible triple.

Every tripartite state is, for one of its triples in O or I, the block of a Wigner 3j symbol (the 3j block of `ame_catalogue.json`). In a group of proper rotations, such as O and I, let $`b_\Gamma(j)`$ be the multiplicity of Γ in the irrep $`D^{(j)}`$ of SO(3) restricted to the group. When $`b_\Gamma(j)=1`$, the partners of Γ are $`|\Gamma\,\mu\rangle=\sum_m|j\,m\rangle\langle j\,m|\Gamma\,\mu\rangle`$, with the subduction coefficients of SO(3) ⊃ G fixed up to one phase. For irreps Γ₁, Γ₂, Γ₃ with b = 1, the block is

```math
C^{(j_1j_2j_3)}_{\mu_1\mu_2\mu_3}=\sum_{m_1m_2m_3}\langle\Gamma_1\mu_1|j_1m_1\rangle\langle\Gamma_2\mu_2|j_2m_2\rangle\langle\Gamma_3\mu_3|j_3m_3\rangle\begin{pmatrix}j_1&j_2&j_3\\m_1&m_2&m_3\end{pmatrix}.
```

The block is G-invariant because the 3j symbol is SO(3)-invariant, so it lies in the coupling space of (Γ₁, Γ₂, Γ₃) for the trivial $`\Gamma_{\rm mol}`$. For N = 1 it is $`C^{(j_1j_2j_3)}=\lambda C`$, with C the coupling tensor and λ a number, possibly zero. This is Racah's factorisation, with factors tabulated by P. H. Butler (*Point Group Symmetry Applications: Methods and Tables*, Plenum, New York, 1981). The j's enter only through λ, and none is a quantum number of the molecule. For example, G acts on nuclear spins only by permuting the nuclei, so the j of a nuclear-spin carrier is not its total nuclear spin. The three protons of para-NH₃, with $`\Gamma_n=E`$, have total spin 1/2, while every j is an integer. `T3`, `T6`, `T10` and `T15` are literal: their carriers are whole $`D^{(j)}`$, and the block is the 3j symbol in another basis. In O, for example, $`D^{(2)}`$ splits into the off-diagonal T₂ and the diagonal E, so the 3j(1,1,2) block on (T₁,T₁,T₂) is |ε| (`T4`) and that on (T₁,T₁,E) is trine-GHZ (`T2`).

## Main results

1. **Abelian and axial groups.** The Abelian groups give only the product state `P`. Apart from `P`, the admissible triples of the non-Abelian axial groups and of C∞v and D∞h are all `B1` or `T1`, at every n. `B1` is a maximally entangled pair of E irreps, and `T1` is GHZ on three E irreps. C₄ᵥ, D₄, D₄h and D₂d have no admissible triple of three E irreps.
2. **Cubic groups.** Besides `P`, `B1` and `T1`, $`T_d`$, O and $`O_h`$ give `B2`, the trine-GHZ state (`T2`), ε (`T3`) and |ε| (`T4`). Every admissible triple of E, $`T_a`$ and $`T_b`$, in any order, is the trine-GHZ state. A $`(T_a,T_b,T_c)`$ triple is ε when the number of T₂ carriers, plus one if $`\Gamma_{\rm mol}`$ is of A₂ type, is even, and |ε| when it is odd. For methane ($`\Gamma_{\rm mol}=A_2`$), this puts (T₂,T₂,T₂) on `T3` and (T₁,T₂,T₂) on `T4`. T and $`T_h`$ give `P`, `B2` and the N = 2 space of (T,T,T), whose K basis is ε and |ε|.
3. **Icosahedral groups.** I and $`I_h`$ give `P`, `B2` to `B4`, `T3` and `T5` to `T16`. Their only (3,3,3) triples are $`(T_a,T_a,T_a)`$, all `T3`. At (3,3,5), two T₁'s or two T₂'s give the Wigner 3j(1,1,2) (`T6`), and T₁ with T₂ gives the coupling T₁ ⊗ T₂ ⊃ H (`T7`). Each other N = 1 format has a single state.
4. **Multiplicity two and the K basis.** N = 2 occurs for 45 triples: (T,T,T) in T and $`T_h`$ (because T⊗T ⊃ 2T), and (G,H,H) and (H,H,H) in I and $`I_h`$. Their T and H carriers are the restrictions of the SO(3) irreps $`D^{(j)}`$ to the proper rotations, with j = 1 for T and j = 2 for H. As with the j of a 3j block, the angular momentum of these $`D^{(j)}`$ on a nuclear-spin carrier is not its nuclear spin. The tables list each coupling space by its K basis, in ascending K. Here K is the total angular momentum of two such carriers, which in (G,H,H) are the two H carriers:
   - (T,T,T): K = 1 and 2 give ε and |ε| (`T3` and `T4`);
   - (G,H,H): K = 3 and 4 give `T13` and `T14`;
   - (H,H,H): K = 2 and 4 give `T15` and `T16`, with K = 2 the Wigner 3j(2,2,2).

   For (T,T,T) and (H,H,H) the K basis is the same for every choice of the two carriers. In T and I each copy is the 3j(K, j, j) block, with the other carrier taken from $`D^{(K)}`$: 3j(1,1,1) and 3j(2,1,1) for (T,T,T). Every member of the space is AME, and the space is a distance-2 code: $`\langle\varphi|O_X|\psi\rangle=(\mathrm{tr}\,O_X/d_X)\,\langle\varphi|\psi\rangle`$ for every operator $`O_X`$ on one carrier X. Projected onto the space, every single-carrier operator is therefore the scalar $`\mathrm{tr}\,O_X/d_X`$. Sec. 3 of the PDF treats the three spaces as codes: (T,T,T) is the three-qutrit code [[3,1,2]]₃ without its GHZ codeword, and neither icosahedral space lies in a larger distance-2 code.
5. **Coriolis coupling.** When rotation and vibration are the pair of carriers that defines K, the Coriolis coupling at a J level that is a single irrep (J = 1 in T, J = 2 in I) selects the K basis. The levels are K = 1 and 2 in T, and in I they are K = 2 and 4 for (H,H,H) and K = 3 and 4 for (H,G,H).
6. **Special members.** Sec. 3 of the PDF lists the members of the N = 2 spaces that are LU-equivalent to named states. Here |T₃⟩, |T₄⟩, … are the stored copies whose states are `T3`, `T4`, ….
   - (T,T,T): the copies |T₃⟩ and |T₄⟩ are ε and |ε| in the partners x, y, z. The members (|T₃⟩ ± i√3 |T₄⟩)/2 are LU-equivalent to ε. The members (|T₃⟩ + |T₄⟩)/√2 and (|T₄⟩ − |T₃⟩)/√2 are codewords of the [[3,1,2]]₃ qutrit code, and each is LU-equivalent to the three-qutrit GHZ state (|000⟩ + |111⟩ + |222⟩)/√3.
   - (G,H,H): the two members $`\sqrt{5/8}\,|T_{13}\rangle\pm e^{i\delta}\sqrt{3/8}\,|T_{14}\rangle`$, with δ ≈ 0.1651502732 π, are LU-equivalent to the simplex-GHZ state $`5^{-1/2}\sum_i|v_i\rangle|i\rangle|i\rangle`$, with the $`v_i`$ at the vertices of a regular simplex.
   - (H,H,H): |T₁₅⟩ is the Wigner 3j(2,2,2), and (2|T₁₅⟩ + 3√5 |T₁₆⟩)/7 is LU-equivalent to it. The two members $`\sqrt{4/7}\,|T_{15}\rangle+\sqrt{3/7}\,e^{i\varphi}|T_{16}\rangle`$ with $`e^{i\varphi}=(\sqrt{15}\pm7i)/8`$ are LU-equivalent to GHZ₅.

   `make_pdf.py` checks each identification on the stored copies of every N = 2 triple, and it stops on a mismatch.

## Running

```bash
pip install -r requirements.txt
cd species_tables
python species.py                # all 75 groups (n <= 10): writes results.pkl (cached per group in results/)
python ame.py                    # self-test of the catalogue, the 3j blocks and the K basis
python check_examples.py         # H2O, NH3/CaOCH3 and CH4 against values written out by hand (exit status 1 on a mismatch)
python make_pdf.py results.pkl   # writes species_tables.tex, species_data.json and ame_catalogue.json
pdflatex species_tables.tex      # run until the .aux and .toc stop changing (three runs from a clean directory)
```

From a cold cache, `species.py` takes about 30 seconds for all 75 groups, and the self-test of `ame.py` about 2 seconds (measured on one workstation). Each group's result is cached in `results/<group>.pkl`, so later runs are fast. `species.py` recomputes a cached group whose schema or inputs differ from the current ones. The inputs are `pointgroups.py`, `ame.py`, `species.py`, `o3h_irreps.json` and `o3h_labels.json`. `make_pdf.py` refuses a `results.pkl` computed from other inputs or without the 75 groups in the order of `pointgroups.py`. Delete `results/` and the `.pkl` files to regenerate from scratch.

## Caveats

The Coriolis statement (item 5, and Sec. 3 of the PDF) uses that the angular-momentum coupling of the two carriers is a function of K. Other couplings, such as higher-rank ones or those of the G carrier in (G,H,H), can select other members of the space.

## Ab initio constants for calcium monomethoxide

The folder `abinitio/` holds the scripts, inputs and raw outputs behind the calcium monomethoxide (CaOCH₃) example of the paper: its section "Preparing symmetry-adapted bend states of calcium monomethoxide" and the appendix "Experimental preparation of SALCs in calcium monomethoxide".

The ν₈ fundamental and the rotational constants A and B are measured. We estimate the remaining constants of the energy formula (Eq. `eq:caoch3_hamiltonian`) with PySCF. Coupled-cluster normal modes give the Coriolis constant ζ₈ = 0.926, and second-order vibrational perturbation theory on a B3LYP force field gives x₈,₈ = −0.53 cm⁻¹ and the anharmonic part of g₈,₈, 0.63 cm⁻¹. The anharmonic estimates vary by up to 0.11 cm⁻¹ with finite-difference step size and density functional. The same force field gives the bend transition dipole μ₈, which only converts Rabi frequencies into a laser intensity.

Every level energy, line position and detuning of the appendix follows from the energy formula alone with these constants. Centrifugal distortion and the spin structure are not added to them; the scripts report their sizes separately. The spin structure sets the Rabi frequency, and with the electron Zeeman term added it gives the magnetic-field limits.

### Where each number in the paper comes from

Labels are the LaTeX labels of the manuscript. Paths are relative to `abinitio/`.

| Paper | Quantity | Value | Script | Stored output |
|---|---|---|---|---|
| `app:caoch3_ham`, `tab:caoch3_constants` | ζ₈ | 0.926 | `caoch3_bend_constants/zeta_from_cc.py` | `caoch3_bend_constants/zeta_from_cc.out` (mode 0: 0.9259) |
| same | x₈,₈ | −0.53 cm⁻¹ | `caoch3_bend_constants/dft_force_field.py`, then `vpt2_bend.py` | `caoch3_bend_constants/vpt2_bend_b3lyp.out` (C₃ᵥ-averaged block: −0.5289) |
| same | g₈,₈ anharmonic part | 0.63 cm⁻¹ | same | same block (+0.6270) |
| `app:caoch3_ham` | change of x₈,₈ and g₈,₈ with step size and functional | ≤ 0.11 cm⁻¹ | `dft_half_step.py` (half step), `XC=pbe0 dft_force_field.py` | `vpt2_bend_b3lyp.out` (Richardson block), `vpt2_bend_pbe0.out` |
| `tab:caoch3_constants` | Aζ₈², g₈,₈, ω̃₈ | 4.672, 5.30, 138.70 cm⁻¹ | arithmetic with the measured A and ν₈ | first line of `caoch3_bend_constants/check_tree_measured.out` |
| `tab:caoch3_levels` | tree level energies | 0 to 416.110 cm⁻¹ | `caoch3_bend_constants/check_tree_measured.py` | `caoch3_bend_constants/check_tree_measured.out`, "Levels" |
| `eq:caoch3_lines`, `tab:caoch3_pulses` | line positions, couplings d/μ₈ | 135.400 to 139.746 cm⁻¹ | same | same file, "Tree lines" |
| `app:caoch3_ham`, `app:caoch3_pulses` | from Eq. `eq:caoch3_hamiltonian`: line spacing 0.027 cm⁻¹ (0.81 GHz), nearest competing line 0.73 GHz | | same | same file |
| `app:caoch3_ham`, `app:caoch3_pulses` | terms beyond the equation, estimates only: distortion ≤ 0.013 cm⁻¹ and 0.57 MHz; spin shifts ≤ 25 MHz, 0.36 MHz, Ω/2π ≤ 12 kHz | | same | same file, "Centrifugal distortion" and spin blocks |
| `app:caoch3_pulses`, `tab:caoch3_constants` | $`\mu_8=\langle1\vert\mu_x\vert0\rangle`$ (B3LYP, double harmonic); check: permanent dipole, harmonic ν₈ | 0.150 D; 1.45 D, 150.7 cm⁻¹ | `caoch3_bend_constants/bend_dipole.py` | `caoch3_bend_constants/bend_dipole_b3lyp.out` (0.1504; 1.447; 150.66) |
| `app:caoch3_pulses`, `tab:caoch3_pulses`, `sec:experiment` | common intensity (P7 at Ω/2π = 10 kHz), field, Ω/2π range, π-pulse durations $`\tau_\pi`$, sequence totals, error bounds | 1.5 mW/cm², 107 V/m, 9.8–44 kHz, 11.4–50.9 µs, 91–217 µs, < 7×10⁻⁴ and < 4×10⁻⁹ | `caoch3_bend_constants/pulse_durations.py` (reads the two outputs above and `check_tree_measured.out`) | `caoch3_bend_constants/pulse_durations.out` |
| `app:caoch3_pulses` (condition (vi)) | Landé $`g_F`$ of the tree levels, largest precession rate, transverse-field limit, largest quadratic line shift at $`B_Z`$ = 50 mG | 0.17–0.39, 0.54 MHz/G, ≲ 20 µG, 4 kHz (P1) | `caoch3_bend_constants/zeeman_limits.py` (reads `check_tree_measured.py`) | `caoch3_bend_constants/zeeman_limits.out` |

### ζ₈ (EA-EOM-CCSD)

Electron-attachment EOM-CCSD from the closed-shell CaOCH₃⁺ reference, with PySCF. Basis: cc-pCVTZ on Ca with the 3s3p shell correlated, cc-pVTZ on O, C and H.

1. `caoch3_eom_ccsd/caoch3_surface.py --preset tz --state X --hess-guess caoch3_min_dz_X.json` optimises the X̃ ²A₁ geometry in C₃ᵥ by finite differences → `caoch3_eom_ccsd/out/caoch3_min_tz_X.json`. The double-zeta run (`--preset dz`) supplies the starting Hessian.
2. `caoch3_eom_ccsd/caoch3_emodes.py caoch3_min_tz_X.json --preset tz` builds the force field of the four e modes → `out/caoch3_emode_cache_tz_X.json` (single points), `out/caoch3_emode_basis_tz_X.json` (displacement vectors), `out/caoch3_emodes_tz_X.json`.
3. `caoch3_bend_constants/zeta_from_cc.py` forms the normal modes and $`\zeta=l_x\cdot(M_z\,l_y)`$. The four ζ sum to $`1+B_e/2A_e`$ to 10⁻⁵, as the sum rule requires.

Run steps 1 and 2 from `abinitio/caoch3_eom_ccsd/out/` (`python ../caoch3_surface.py ...`). Every single point is cached in a JSON file, so the commands replay the stored points and compute nothing new.

Keep `caoch3_emode_basis_tz_X.json` with its cache. The four displacement vectors come from a singular-value decomposition with degenerate singular values, so another LAPACK build returns a different set spanning the same space, and the cached energies are labelled by these vectors.

### x₈,₈ and g₈,₈ (B3LYP force field, second-order perturbation theory)

1. `dft_force_field.py` (UKS-B3LYP/def2-TZVP): C₃ᵥ geometry, analytic Hessian, and Hessians displaced by ±0.10 $`\mathrm{amu}^{1/2}\,\mathrm{\AA}`$ along both ν₈ components → cubic constants with a ν₈ index and the ν₈ quartic constants (`dft_force_field_b3lyp_def2tzvp.npz`).
2. `vpt2_bend.py <npz>` applies second-order perturbation theory numerically in each v₈ manifold and fits the levels to Eq. `eq:caoch3_hamiltonian`.
3. `dft_half_step.py` (half the step) and `XC=pbe0 python dft_force_field.py` give the variation quoted in the appendix.

Run these from `abinitio/caoch3_bend_constants/`. The same B3LYP and PBE0 force fields give ζ₈ = 0.9267 and 0.9260, in agreement with the coupled-cluster value.

### μ₈ (B3LYP, double harmonic)

`bend_dipole.py dft_force_field_b3lyp_def2tzvp.npz` evaluates the dipole moment at ±0.10 $`\mathrm{amu}^{1/2}\,\mathrm{\AA}`$ along both ν₈ components and gives $`\mu_8=\langle1|\mu_x|0\rangle=(\hbar/2\omega)^{1/2}\,\partial\mu_x/\partial Q`$ = 0.150 D. As a check, it also gives the permanent dipole, 1.45 D, against the measured 1.58(8) D.

### Levels, lines, pulses and magnetic-field limits

All three scripts run from `abinitio/caoch3_bend_constants/` with plain Python and take seconds.

- `check_tree_measured.py` evaluates Eq. `eq:caoch3_hamiltonian` alone for the eight tree levels, the seven transitions and their competing lines. It then reports, separately, the size of centrifugal distortion (never added to the energies) and the spin-rotation and proton hyperfine structure. Measured constants are set at the top of the script; x₈,₈ and the anharmonic part of g₈,₈ are read from `anharmonic.json`. The script calls the two k = 0 targets `L03_3` (N = 3) and `L03_4` (N = 4); the paper calls them L₀₃ and L₃₀.
- `pulse_durations.py` sets one intensity for all seven tones, so that P7 has Ω/2π = 10 kHz, and gives the Rabi frequency of each pulse against its hyperfine bound, the π-pulse durations and the sequence totals.
- `zeeman_limits.py` adds the electron Zeeman term to the spin structure of `check_tree_measured.py` and gives the Landé factors and magnetic-field limits of condition (vi) in the appendix.

### Software

Python 3.12–3.13, PySCF 2.14.0, NumPy 2.5.3, SciPy 1.18.1, and SymPy for `check_tree_measured.py` and `zeeman_limits.py` (`abinitio/requirements.txt`). On an 18-core laptop the triple-zeta geometry optimisation took 1.5 h (39 single points), the e-mode force field 1 h (21 points) and one B3LYP force field 9 min.

## Requirements

Python ≥ 3.10 with NumPy. To build the PDF: pdflatex with the packages of `make_pdf.py`'s preamble (cmap, geometry, amsmath, amssymb, amsthm, booktabs, longtable, array, ragged2e, multicol, multirow, xcolor, fancyhdr, tikz, hyperref, fontenc and lmodern), and sansmathfonts for o3harmonics' element font. A full TeX Live installation has them all. The calculations in `abinitio/` also need PySCF, SciPy and SymPy, at the versions in `abinitio/requirements.txt`.
