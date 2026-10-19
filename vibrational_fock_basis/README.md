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

## Requirements

- Mathematica or the Wolfram Engine with `wolframscript` (tested with 14.3).
- [GTPack](https://gtpack.org) (tested with 1.4) for `salc_fock_basis.wl` only, loaded with ``Needs["GroupTheory`"]``.

## Changes from earlier versions

- **Convention fix (C₃ᵥ script).** C₃ now acts as ω^{−ℓ}, as in the main text, instead of ω^{+ℓ}. The E states are printed as ¹E/²E partner pairs.
- **Representation fix (T_d script).** The Fock matrices now follow Lemma 2 (a†ᵢ → Σⱼ D_ji a†ⱼ). The previous q → Rq rule gave an anti-representation, which only character projectors could use.
- **New output (T_d script).** E is in the complex ¹E/²E basis, and repeated irreps are split with matrix-valued projectors. Previously each irrep's whole isotypic subspace came out as a single block.
- **SALC script.**
  - Kets are printed in the main text's |n₊, n₋⟩ order (a ¹E quantum has ℓ = −1).
  - Equation references are updated to Lemma 2, Theorem 1, Eq. (25) and Eq. (S5).
  - Single-mode inputs no longer fail.
