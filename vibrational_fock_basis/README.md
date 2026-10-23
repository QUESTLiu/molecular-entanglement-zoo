# Vibrational Fock basis (Wolfram Language)

Mathematica code for the **vibrational sector**. It builds the symmetry-adapted occupation-number (Fock) states of vibrational manifolds, which are the vibrational carriers |μΓ; k⟩_v that enter every coupled rotation–nuclear-spin–vibration state.

The scripts implement two results of the main text:
- **Lemma 2 (Fock-space action).** U(R) a†ᵢ U(R)† = Σⱼ D_ji(R) a†ⱼ, so U_v(R) = ⊗_s Sym^{v_s}[D^{Γ_s}(R)].
- **Theorem 1 (Symmetry-adapted Fock encoding).** The matrix-valued projectors
  P^Γ_μν = (d_Γ/|G|) Σ_R D^Γ_μν(R)* U_v(R)   (Eq. 25)
  split every manifold H(**v**) into copies k = 1…m_Γ(**v**) with partners μ = 1…d_Γ. The multiplicities follow from the characters (Eq. S5).

Every script uses the main text's basis conventions, the same ones the rotational and nuclear-spin carriers use in [`../species_tables`](../species_tables):

| Irrep | Basis |
|---|---|
| 2D (E-type) | Complex partners ¹E, ²E. The lowest proper rotation is diagonal (C₃ → diag(ω, ω²), ω = e^{2πi/3}) and a vertical reflection is a pure swap. One ¹E quantum carries vibrational angular momentum ℓ = −1. States are written \|n₊, n₋⟩ with ℓ = n₊ − n₋ and U(C_n) = e^{−2πiℓ/n}. |
| Cubic T₂ | Cartesian components x, y, z (single quantum \|1ᵢ⟩ = a†ᵢ\|0⟩). T₁ = det R · R. |
| Cubic E | \|¹,²E⟩ = 3^{−1/2} Σᵢ ω^{∓i} \|2ᵢ⟩ on two quanta. C₃[111] (x→y→z) is diag(ω, ω²) and σ_d(x↔y) is a pure swap. |

## Files

| File | What it computes | Needs |
|---|---|---|
| [`salc_fock_basis.wl`](salc_fock_basis.wl) | **General case.** Any point group, any irrep per normal mode, any quanta vector **v**: characters χ_v(R), decomposition Γ_red = ⊕ m_Γ Γ, and every SALC \|μΓ; k⟩. Default input: CaOCH₃ (C₃ᵥ, four A₁ and four E modes, one quantum each). | GTPack |
| [`overtones_C3v_E.wl`](overtones_C3v_E.wl) | The n-quantum overtone of one E mode in C₃ᵥ (ν₄ of NH₃, ν₈ of CaOCH₃): A₁/A₂ states and partner-resolved ¹E/²E pairs, \|¹E⟩ = \|ℓ⟩, \|²E⟩ = \|−ℓ⟩ with ℓ ≡ 2 (mod 3). Self-contained (explicit C₃ᵥ matrices). | — |
| [`overtones_Td_T2.wl`](overtones_Td_T2.wl) | The N-quantum overtone Sym^N(T₂) of one T₂ mode in T_d (ν₃, ν₄ of CH₄): Sym²T₂ = A₁ ⊕ E ⊕ T₂, Sym³T₂ = A₁ ⊕ T₁ ⊕ 2T₂, …, with repeated irreps split into individual partner-resolved copies. Self-contained (T_d built as signed permutation matrices). | — |
| [`example_outputs/`](example_outputs) | Console output of each script: C₃ᵥ n = 6, T_d N = 2 and N = 3, CaOCH₃ one quantum per mode | — |
| [`salc_tables.pdf`](salc_tables.pdf) | **SALC tables for every point group.** The symmetry-adapted Fock states of one degenerate mode for all groups of [`../species_tables`](../species_tables), in the same bases and labels, organised so that finite tables give every v. Also worked examples at finite v. See [below](#salc-tables-for-all-point-groups). | — |
| [`salc_tables.py`](salc_tables.py) | Generates `salc_tables.tex` and `salc_tables.json`. It uses the same projector construction as `salc_fock_basis.wl` and checks every rule against it. Uses [`../species_tables/pointgroups.py`](../species_tables/pointgroups.py) | NumPy, SymPy |
| [`salc_tables.json`](salc_tables.json) | Machine-readable residue tables, harmonic SALCs and branching, and multiplicities (v ≤ 12) | — |
| [`multiplicities_gtpack.wl`](multiplicities_gtpack.wl) | Independent GTPack computation of the multiplicities m_Γ(v) for the 32 crystallographic groups. It writes `multiplicities_gtpack.json`, which `salc_tables.py` compares against | GTPack |

Each script checks its own result, and these checks print `True` in the example outputs:
- **Both overtone scripts:** U(R) is a representation, and the irrep matrices respect the same group products.
- **T_d script, in addition:** the SALCs are orthonormal and complete, and U(R)|νΓ;k⟩ = Σ_μ D_μν(R)|μΓ;k⟩ for every R.

### Usage

The inputs sit at the top of each file:

```wolfram
(* salc_fock_basis.wl *)
groupName  = "C3v";
modeIrreps = {"A1", "A1", "A1", "A1", "E", "E", "E", "E"};   (* one entry per degenerate set *)
vQuanta    = {1, 1, 1, 1, 1, 1, 1, 1};                       (* quanta in each *)
circular   = True;    (* main-text basis for 2D irreps; False keeps GTPack's basis *)
numeric    = False;   (* True for large manifolds *)

(* overtones_C3v_E.wl *)   n = 6;
(* overtones_Td_T2.wl *)   degree = 2;
```

```bash
wolframscript -file salc_fock_basis.wl
wolframscript -file overtones_C3v_E.wl
wolframscript -file overtones_Td_T2.wl
```

Each script also leaves its full result in an association for further use:
- `salc_fock_basis.wl` → `SALCresult` (irrep matrices, characters, multiplicities, SALC vectors);
- `overtones_Td_T2.wl` → `TdOvertoneResult`.

For repeated irreps the copy index k is fixed only up to a U(m_Γ) rotation (Theorem 1 (iii)). The scripts pick one orthonormal choice by projecting occupation states in order.

## Checks against the main text

| Case | Output | Main text |
|---|---|---|
| One E quantum (C₃ᵥ) | ¹E = \|0,1⟩ (ℓ = −1), ²E = \|1,0⟩ | ¹E_v = \|ℓ₄ = −1⟩, ²E_v = \|ℓ₄ = +1⟩ |
| Two E quanta | A₁ = \|1,1⟩ (ℓ = 0); ¹E = \|2,0⟩ (ℓ = +2) | [E²] = A₁ ⊕ E; C₃ = diag(ω, 1, ω²) on \|+2⟩, \|0⟩, \|−2⟩ |
| Two T₂ quanta (T_d) | ¹E ∝ Σᵢ ω^{−i}\|2ᵢ⟩, ²E ∝ Σᵢ ω^{+i}\|2ᵢ⟩ (one common phase) | \|¹,²E⟩_{2ν₄} = 3^{−1/2} Σᵢ ω^{∓i}\|2ᵢ⟩ |

## SALC tables for all point groups

[`salc_tables.pdf`](salc_tables.pdf) tabulates the symmetry-adapted occupation-number states |μΓ; k⟩ of one degenerate vibrational mode, for every group of the species tables (C₁, C_s, C_i; C_n, C_nv, C_nh, S₂ₙ, D_n, D_nh, D_nd for n = 2…8; T, T_h, T_d, O, O_h, I, I_h; C∞ᵥ, D∞h). The states are built exactly as in `salc_fock_basis.wl`: Lemma 2 action, matrix-valued projectors, copies by projecting occupation states in order. The irrep bases and labels are those of `species_tables/pointgroups.py`, and the kets are |n₊,n₋⟩ (a ¹E quantum is counted in n₋, so ℓ = −1) and |n_x n_y n_z⟩.

Instead of one table per v, three classifications reduce the states for every v to finite data:

- **Parity.** ℓ (or L) has the parity of v. Inversion and σ_h act as (parity)^v, so g/u and ′/″ alternate with v for u- and ″-type modes.
- **Residues (two-dimensional modes).** Each SALC lies in span{|n₊,n₋⟩, |n₋,n₊⟩}. Its species and coefficients depend only on ℓ = n₊ − n₋ modulo a period N′. One table of N′ rows therefore gives the SALCs for all v. Separably degenerate pairs (C_n, S₂ₙ, C_nh, T, T_h) have each |n₊,n₋⟩ as a SALC, so the table reduces to a list of species against ℓ mod N′.
- **Harmonics (vector-like three-dimensional modes, D(R) = ε(R)R).** These are the T modes of T, T_h, T_d, O and O_h, and T₁ of I and I_h. The manifold decomposes as

  Sym^v = ⊕_{L=v,v−2,…} (a†·a†)^{(v−L)/2} 𝓗_L.

  Every SALC is 𝒩 (a†·a†)^k ψ_L, with ψ_L a G-adapted harmonic SALC and 𝒩⁻² = 4^k k! (L+3/2)_k. The species of 𝓗_L are the rotational branching D^L|_G, with parity (−1)^L, times ε^L. So the vibrational overtones follow the same branching rules as the rotational sector, and the copy label of a repeated irrep becomes the layer L.

The T₂, G and H modes of I and I_h fit neither rule, so they are listed explicitly (T₂ for v ≤ 6; G and H for v ≤ 3, since their v = 6 manifolds have 84 and 210 states), with multiplicities up to v = 12. Their G and H irrep bases are fixed by polynomials, which the tables list.

**Worked examples** give explicit SALCs for v ≤ 6 for C₃ᵥ E, C₄ᵥ E, D₃ₕ E′, D₆ₕ E₁ᵤ, T_d E and T₂, and O_h T₁ᵤ. They reproduce the conventions of `overtones_C3v_E.wl`: for example, ¹E = |0,1⟩ for one E quantum and ¹E = |2,0⟩ for two.

**Checks.** `python salc_tables.py` stops if any of these fails:
- **Residue tables:** each one is compared with the projector construction for every v ≤ 2N′ + 2 and every ℓ.
- **Harmonic tables:** checked for L ≤ 6 (explicit harmonic SALCs are listed up to L = 6). The factorised SALCs and the norm formula are checked for v ≤ 6.
- **Every listed set of states** is orthonormal, complete, and transforms with the irrep matrices of `pointgroups.py`.
- **Multiplicities** agree with GTPack for the 32 crystallographic groups.

```bash
python salc_tables.py                        # writes salc_tables.tex and .json (about 30 s)
pdflatex salc_tables.tex                     # run twice for the table of contents
wolframscript -file multiplicities_gtpack.wl # optional: regenerates multiplicities_gtpack.json
```

## Requirements

- Mathematica or the Wolfram Engine with `wolframscript` (tested with 14.3).
- [GTPack](https://gtpack.org) (tested with 1.4) for `salc_fock_basis.wl` and `multiplicities_gtpack.wl` only, loaded with ``Needs["GroupTheory`"]``.
- Python ≥ 3.10 with NumPy and SymPy for `salc_tables.py`.

## Changes from earlier versions

- **Convention fix (C₃ᵥ script).** C₃ now acts as ω^{−ℓ}, as in the main text, instead of ω^{+ℓ}. The E states are printed as ¹E/²E partner pairs.
- **Representation fix (T_d script).** The Fock matrices now follow Lemma 2 (a†ᵢ → Σⱼ D_ji a†ⱼ). The previous q → Rq rule gave an anti-representation, which only character projectors could use.
- **New output (T_d script).** E is in the complex ¹E/²E basis, and repeated irreps are split with matrix-valued projectors. Previously each irrep's whole isotypic subspace came out as a single block.
- **SALC script.**
  - Kets are printed in the main text's |n₊, n₋⟩ order (a ¹E quantum has ℓ = −1).
  - Equation references are updated to Lemma 2, Theorem 1, Eq. (25) and Eq. (S5).
  - Single-mode inputs no longer fail.
