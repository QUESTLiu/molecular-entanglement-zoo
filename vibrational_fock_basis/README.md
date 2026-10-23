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
| [`overtone_tables.pdf`](overtone_tables.pdf) | **Overtone tables for every point group** (5 pages): which irreps v quanta of one degenerate mode contain, for all point groups of rigid molecules. See [below](#overtone-tables-for-all-point-groups). | — |
| [`overtone_tables.py`](overtone_tables.py) | Generates `overtone_tables.tex` and `overtone_tables.json` and runs all checks. Self-contained Python: builds each group from 3×3 matrices and computes its character table itself. | NumPy, SymPy |
| [`overtone_tables.json`](overtone_tables.json) | Multiplicities m_Γ(v), v ≤ 12, for every degenerate mode of 23 representative groups | — |
| [`overtone_tables_gtpack.wl`](overtone_tables_gtpack.wl) | Independent computation of the same decompositions from GTPack character tables (32 crystallographic groups); writes `overtone_tables_gtpack.json`, which `overtone_tables.py` compares against | GTPack |

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

## Overtone tables for all point groups

The species content of a vibrational manifold depends only on the point group and on the mode species, not on the molecule. [`overtone_tables.pdf`](overtone_tables.pdf) gives the decomposition

Sym^v Γ_s = ⊕_Γ m_Γ(v) Γ

of v quanta in one degenerate mode for every point group of a rigid molecule. That covers the 32 crystallographic groups, C₅ᵥ, D₅, D₅ₕ, D₅d, D₄d (cyclo-S₈), I, I_h, and the linear groups. A molecule enters only through its list of mode species. The document contains:
- **Reduction rules (R1)–(R4):** nondegenerate modes; inversion (g/u) and σ_h (′/″) factors, applied as (parity)^v; combination bands as tensor products.
- **One closed-form rule for every E_k mode of C_nv, D_n, D_nh, D_nd, for any n.** In terms of the vibrational angular momentum ℓ: with r = kℓ mod N, each ±ℓ pair gives A₁⊕A₂ (r = 0), B₁⊕B₂ (r = N/2) or E_min(r, N−r). Cyclic and linear-molecule variants are included.
- **Decomposition tables** for C₃ᵥ, C₄ᵥ, C₅ᵥ, C₆ᵥ and D₄d up to v = 8, and **multiplicity tables** for the cubic (T_d, O, T) and icosahedral (I) modes up to v = 12.
- **Explicit v = 2, 3 states** of a triply degenerate mode (cubic harmonics), with their species in T_d, O and T.
- **An index** saying where each group's degenerate modes are covered.

The phase conventions match the rest of this folder: U(C_n) = e^{−2πiℓ/n}, a ¹E quantum carries ℓ = −1, and T₁ = det R · R.

**Checks.** `python overtone_tables.py` stops if any of these fails:
- **Dimensions:** every row satisfies Σ_Γ m_Γ(v) d_Γ = C(v+d_s−1, d_s−1).
- **The ℓ rule:** agrees with direct computation for C_nv, D_n and D_nd, 3 ≤ n ≤ 10, v ≤ 12 (832 decompositions).
- **Inversion and σ_h rules:** agree for D₃d, D₃ₕ, D₄ₕ, D₅d, D₅ₕ, D₆ₕ, T_h, O_h and I_h (468 decompositions).
- **T_d and O:** give identical tables with identical labels.
- **The explicit cubic states:** carry the stated species (character projection).
- **GTPack:** agreement for all 32 crystallographic groups, every degenerate mode, v ≤ 12, label-free.

In addition, `overtones_C3v_E.wl` (v ≤ 8) and `overtones_Td_T2.wl` (v ≤ 6) reproduce the C₃ᵥ and T_d entries.

```bash
python overtone_tables.py                  # writes overtone_tables.tex and .json (a few seconds)
pdflatex overtone_tables.tex
wolframscript -file overtone_tables_gtpack.wl   # optional: regenerates overtone_tables_gtpack.json
```

## Requirements

- Mathematica or the Wolfram Engine with `wolframscript` (tested with 14.3).
- [GTPack](https://gtpack.org) (tested with 1.4) for `salc_fock_basis.wl` and `overtone_tables_gtpack.wl` only, loaded with ``Needs["GroupTheory`"]``.
- Python ≥ 3.10 with NumPy and SymPy for `overtone_tables.py`.

## Changes from earlier versions

- **Convention fix (C₃ᵥ script).** C₃ now acts as ω^{−ℓ}, as in the main text, instead of ω^{+ℓ}. The E states are printed as ¹E/²E partner pairs.
- **Representation fix (T_d script).** The Fock matrices now follow Lemma 2 (a†ᵢ → Σⱼ D_ji a†ⱼ). The previous q → Rq rule gave an anti-representation, which only character projectors could use.
- **New output (T_d script).** E is in the complex ¹E/²E basis, and repeated irreps are split with matrix-valued projectors. Previously each irrep's whole isotypic subspace came out as a single block.
- **SALC script.**
  - Kets are printed in the main text's |n₊, n₋⟩ order (a ¹E quantum has ℓ = −1).
  - Equation references are updated to Lemma 2, Theorem 1, Eq. (25) and Eq. (S5).
  - Single-mode inputs no longer fail.
