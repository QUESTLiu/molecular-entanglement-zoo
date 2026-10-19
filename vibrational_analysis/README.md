# Vibrational analysis (Wolfram Language)

Mathematica code for the vibrational sector of the phase-space construction:
- symmetry-adapted (SALC) Fock states of vibrational manifolds;
- overtone projectors for degenerate modes;
- a character-table check of which rotation–nuclear-spin–vibration species are admissible.

The `.wl` files are plain-text Wolfram Language scripts. They read and diff as text on GitHub, and they run without opening a notebook:

```bash
wolframscript -file C3v_E_overtone_projector.wl
```

The notebook is included with its outputs cleared, next to a plain-text `.wl` export of the same code.

## Files

| File | What it computes | Dependencies |
|---|---|---|
| [`SALC_vibrational_basis_GTPack.wl`](SALC_vibrational_basis_GTPack.wl) | General SALC builder (Theorem 2 / Eqs. S1–S5 of the paper) for any point group, any assignment of mode irreps and any quanta vector **v** | GTPack |
| [`C3v_E_overtone_projector.wl`](C3v_E_overtone_projector.wl) | Symmetry-adapted basis of the n-quantum overtone of one doubly degenerate E mode in C₃ᵥ | none |
| [`Generic_Td_T2_orthogonal_overtone_projectors.wl`](Generic_Td_T2_orthogonal_overtone_projectors.wl) | Projectors and orthonormal symmetry-adapted states for the Sym^N(T₂) overtone manifold of a triply degenerate T₂ mode in T_d (e.g. ν₃, ν₄ of CH₄) | none |
| [`allowed_species_gamma_mol.nb`](allowed_species_gamma_mol.nb) | Admissible (Γ_rot, Γ_nuc, Γ_vib) triples containing Γ_mol, by character decomposition, for C₂ᵥ, C₃ᵥ, T_d and I_h (outputs cleared) | GTPack |
| [`allowed_species_gamma_mol.wl`](allowed_species_gamma_mol.wl) | Plain-text export of the notebook (same code) | GTPack |
| [`example_outputs/`](example_outputs) | Console output of each script with its default settings | — |

### `SALC_vibrational_basis_GTPack.wl`

The inputs are set at the top of the file:

```wolfram
groupName  = "C3v";
modeIrreps = {"A1", "A1", "A1", "A1", "E", "E", "E", "E"};   (* CaOCH3: nu1 ... nu8 *)
vQuanta    = {1, 1, 1, 1, 1, 1, 1, 1};
circular   = True;    (* 2D irreps in the circular basis |n+,n-> *)
numeric    = False;   (* True for large manifolds *)
```

The script then:
1. builds U_v(R) on the Fock space H(**v**), as the Kronecker product over modes of each mode's symmetric-power representation;
2. computes the characters χ_v(R) and the decomposition Γ_red = ⊕ m_Γ(**v**) Γ (Eq. S4);
3. constructs every SALC |Γ μ, k⟩ (copy k = 1…m_Γ, partner μ = 1…d_Γ) with the matrix-valued projectors P^Γ_μν (Eq. S3).

It prints the character table with the Γ_red row, the decomposition, and the SALC states. All results are also collected in the association `SALCresult`.

The convention is R a_i† R⁻¹ = Σ_j D_ji(R) a_j†. With `circular = True`, 2D irreps are rotated into the basis where the smallest proper rotation is diagonal (first partner has the positive phase) and one reflection is a pure swap n₊ ↔ n₋. This is the same convention as the main text and `species_tables/`.

### `C3v_E_overtone_projector.wl`

Change `userOvertone` (default 6). The basis is |n₊, n₋⟩ with n₊ + n₋ = n. C₃ acts as ω^(n₊−n₋), and the three σ_v act as the swap n₊ ↔ n₋ with phases 1, ω^(n₊−n₋) and ω^(2(n₊−n₋)). The script prints the A₁, A₂ and E adapted states, i.e. the familiar vibrational-angular-momentum labels ℓ = n₊ − n₋ (mod 3). For n = 6 it gives 2 A₁, 1 A₂ and 2 E copies.

### `Generic_Td_T2_orthogonal_overtone_projectors.wl`

Change `degree` (default 2). The script:
- generates T_d as the 24 signed permutation matrices with sign product +1 (the T₂ vector representation on x, y, z);
- induces Sym^N(T₂) on the orthonormal Fock basis;
- decomposes it (Sym²T₂ = A₁ ⊕ E ⊕ T₂, Sym³T₂ = A₁ ⊕ T₁ ⊕ 2T₂, …);
- builds the character projectors and checks P² = P, symmetry, rank, completeness and mutual orthogonality;
- extracts orthonormal symmetry-adapted states and checks that they block-diagonalise the representation.

For multiplicity > 1 the character projector returns the whole isotypic subspace. The script warns about this; separating individual copies needs the matrix-valued projectors of the SALC script.

### `allowed_species_gamma_mol.nb` / `.wl`

`MakeTableData` reads a GTPack character table and `DecomposeProduct` reduces Γ_r ⊗ Γ_n ⊗ Γ_v. `AllowedTriples` keeps the triples that contain Γ_mol and lists their multiplicity. Worked cases:

| Group | Γ_mol | Species used |
|---|---|---|
| C₂ᵥ (H₂O) | B₂ | A₁, B₂ |
| C₃ᵥ (NH₃, CaOCH₃) | A₂ | all; with the explicit coupled states as LaTeX |
| T_d (CH₄) | A₂ | all |
| I_h | A_g | hard-coded characters; Γ_r ∈ g irreps |

These results are a cross-check of the Python tables in [`../species_tables`](../species_tables), which extend the same calculation to every point group, every real Γ_mol and the coupling tensors. For example, the I_h block returns 130 triples, which is the g-rotational half of the 260 listed in the PDF.

## Requirements

- Mathematica / Wolfram Engine with `wolframscript` (tested with 14.3).
- [GTPack](https://gtpack.org) (tested with 1.4) for `SALC_vibrational_basis_GTPack.wl` and the species notebook, loaded with ``Needs["GroupTheory`"]``. The two overtone-projector scripts are self-contained.
