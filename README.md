# Quantum theory of molecular phase space: species and entanglement tables

Code and tables for **symmetry-enforced entanglement between molecular rotation, nuclear spin and vibration**, for every point group of O(3).

A molecule with point group **G** has its total wavefunction fixed by spin statistics to a one-dimensional species Γ_mol. A rotational, nuclear-spin and vibrational species triple (Γ_r, Γ_n, Γ_v) is **admissible** when

N = (1/|G|) Σ_R χ_r(R) χ_n(R) χ_v(R) χ*_mol(R) ≠ 0.

The admissible state is then fixed by the point-group Clebsch–Gordan tensor

|C⟩ = Σ C_abc |a⟩_r |b⟩_n |c⟩_v,

and its entanglement across the three sectors is a property of the triple. This repository lists every admissible triple, its coupling tensor and its entanglement class.

## Contents

| Path | What it is |
|---|---|
| [`pdf/species_tables.pdf`](pdf/species_tables.pdf) | The full tables (177 pages), with a worked analysis of the multiplicity-two cases |
| [`data/species_data.json`](data/species_data.json) | Every admissible triple with its class, family and nonzero coupling-tensor entries |
| `species_tables/pointgroups.py` | Builds the point groups, their irreps over ℂ in the standard basis, and Mulliken labels |
| `species_tables/entanglement.py` | Entanglement classifier: flattening ranks, numerical CP tensor rank, local-unitary fidelity |
| `species_tables/species.py` | Admissibility, coupling tensors, classification (writes `results.pkl`) |
| `species_tables/make_pdf.py` | Builds the LaTeX tables and `species_data.json` |
| `species_tables/fusion_section.py` | Analysis of the N = 2 coupling spaces, the figure `fig_fusion_qubit.pdf`, and the C₂₀H₂₀ / ¹³C₆₀ examples |
| `species_tables/fusion_tex.py` | LaTeX for that section |
| `species_tables/ghz_points.py` | Searches the N = 2 spaces for GHZ-type members |
| `species_tables/check_paper_examples.py` | Regression check against the H₂O, NH₃/CaOCH₃ and CH₄ tables of the accompanying paper |

## Scope

**Groups.** All closed subgroups of O(3):
- C₁, C_s, C_i;
- the seven axial families C_n, S₂ₙ, C_nh, C_nv, D_n, D_nh, D_nd for n = 2…8;
- T, T_h, T_d, O, O_h, I, I_h;
- C∞, C∞h, C∞v, D∞h, truncated at |Λ| ≤ 3 (computed exactly with C₁₃ / C₁₄h / C₁₃v / D₁₄h, whose invariants coincide with the continuous groups' in that range);
- SO(3) and O(3), truncated at J ≤ 3 (Wigner 3j tensors).

**Γ_mol.** The spin-statistics character χ_mol(R) = Π_a sgn[p_a(R)]^{2I_a} only takes the values ±1. Each group therefore gets one table for every **real one-dimensional irrep**.

**Species.** Every irrep is allowed in every sector, which makes the tables pure group theory. Molecule-specific restrictions, such as which nuclear-spin species a given set of nuclei can carry, are not imposed.

**Conventions.** Irreps are irreducible over ℂ, so complex-conjugate pairs are listed separately as ¹E, ²E. The basis conventions are:
- 2D irreps use the complex basis |¹E⟩, |²E⟩ that diagonalises C_n, with C₂′ or σ_v acting as a plain swap;
- cubic T irreps are Cartesian (x, y, z), with C₃[111] cyclic.

These reproduce the tensors of the paper's T_d example term for term.

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

The first full run takes a few hours, mostly on the numerical tensor-rank searches for I_h and SO(3)/O(3). Classifications are cached in `class_cache.pkl`, so later runs are fast.

## Caveats

- Tensor ranks above d_max are numerical upper bounds. Permuted copies of the same 4×5×5 triple sometimes come out one apart, so treat those values as ±1.
- The geometric measure Λ² comes from a multi-start power method. It can only underestimate the true maximum.
- The molecular geometries are idealised, and the rotational constants are rigid-sphere estimates.
- The statement about which fusion states are selected covers only vector-type couplings. Higher-rank inter-sector couplings can select other states.

## Requirements

Python ≥ 3.10 with NumPy, SciPy, mpmath and Matplotlib; a LaTeX installation (pdflatex, booktabs, longtable) to build the PDF.
