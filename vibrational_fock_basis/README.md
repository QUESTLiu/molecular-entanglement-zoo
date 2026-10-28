# Vibrational Fock basis

Code for the **vibrational sector**. It builds the symmetry-adapted occupation-number (Fock) states of vibrational manifolds, which are the vibrational carriers |μΓ; k⟩_v that enter every coupled rotation–nuclear-spin–vibration state. A Wolfram Language script does this for any point group and excitation pattern, and `salc_tables.py` tabulates the states of one degenerate mode for every finite group of [`../species_tables`](../species_tables).

The code implements two results of the main text:
- **Fock-space action (lemma).** U(g) a†ᵢ U(g)† = Σⱼ D_ji(g) a†ⱼ, so U_v(g) = ⊗_s Sym^{v_s}[D^{Γ_s}(g)].
- **Symmetry-adapted Fock encoding (theorem).** The matrix-valued projectors
  P^Γ_μν = (d_Γ/|G|) Σ_{g∈G} D^Γ_μν(g)* U_v(g)
  split every manifold H(**v**) into copies k = 1…m_Γ(**v**) with partners μ = 1…d_Γ. The multiplicities follow from the characters, m_Γ(**v**) = |G|⁻¹ Σ_{g∈G} χ*_Γ(g) χ_**v**(g).

The irrep labels of `salc_tables.pdf` are o3harmonics', as in [`../species_tables`](../species_tables) (see the [conventions](../README.md#conventions-and-o3harmonics)). The bases are the main text's:

| Irrep | Basis |
|---|---|
| 2D (E-type) | Complex partners μ = 1, 2. The principal rotation (S₂ₙ in D_nd with n even) is diagonal, with e^{+iθ}, 0 < θ < π, on partner 1 (C₃ → diag(ζ₃, ζ₃²), ζ₃ = e^{2πi/3}, which is o3harmonics' matrix at C₃⁻), and a vertical reflection is a pure swap. A quantum in partner 1 carries vibrational angular momentum ℓ = −1. States are written \|n₊, n₋⟩, with n₋ quanta in partner 1, ℓ = n₊ − n₋, and, for a mode of species E_q, U(C_n)\|n₊, n₋⟩ = e^{−2πiqℓ/n}\|n₊, n₋⟩. |
| Cubic T₂ | Cartesian components x, y, z, the partners μ = 1, 2, 3 (single quantum \|1ᵢ⟩ = a†ᵢ\|0⟩). In T_d, D^{T₂}(g) = P(g)ᵀ, the 3×3 matrix of g acting on column vectors, and D^{T₁}(g) = det(g) P(g)ᵀ. |
| Cubic E | Partners μ = 1, 2 = 3^{−1/2} Σᵢ ζ₃^{∓i} \|2ᵢ⟩ on two quanta (i = x, y, z = 1, 2, 3), up to a common phase. C₃ about [111] (x → y → z) is diag(ζ₃, ζ₃²) and σ_d (x ↔ y) is a pure swap. |

`salc_tables.py` keeps this code's own bases, the ones the species tables used before they imported o3harmonics' matrices, and puts the icosahedral G and H in polynomial bases. These coincide with o3harmonics' bases except for 24 two-dimensional irreps (a sign on partner 2, or the partners exchanged) and the icosahedral G and H, so the PDF states the map of its kets to o3harmonics' partners ([below](#salc-tables-for-all-point-groups)). `salc_fock_basis.wl` with `circular = True` rotates GTPack's two-dimensional irreps into the same kind of basis; it prints GTPack's irrep names and calls the partners 1E and 2E.

## Files

| File | What it computes | Needs |
|---|---|---|
| [`salc_fock_basis.wl`](salc_fock_basis.wl) | **General case.** Any point group, any irrep per normal mode, any quanta vector **v**: characters χ_v(g), decomposition Γ_red = ⊕ m_Γ Γ, and every SALC \|μΓ; k⟩. Default input: CaOCH₃ (C₃ᵥ, four A₁ and four E modes, one quantum each). | GTPack |
| [`salc_tables.pdf`](salc_tables.pdf) | **SALC tables for every finite point group.** The symmetry-adapted Fock states of one degenerate mode for the 59 finite groups of [`../species_tables`](../species_tables), with the same labels, organised so that finite tables give every v. Also worked examples at finite v. See [below](#salc-tables-for-all-point-groups). | — |
| [`salc_tables.py`](salc_tables.py) | Generates `salc_tables.tex` and `salc_tables.json`. It uses the same projector construction as `salc_fock_basis.wl` and checks every rule against it. Uses [`../species_tables/pointgroups.py`](../species_tables/pointgroups.py) for the groups and its own bases, and [`../species_tables/o3h_tex.py`](../species_tables/o3h_tex.py) and [`../species_tables/o3h_macros.tex`](../species_tables/o3h_macros.tex) for o3harmonics' printed forms | NumPy, SymPy |
| [`salc_tables.json`](salc_tables.json) | Machine-readable tables. For each group: the irreps as the PDF prints them, this code's labels (`irreps_repo`), o3harmonics' record and ids; for each mode its residue table, harmonic SALCs and branching, or explicit type, and its multiplicities (v ≤ 12); the Law-1 tables and the combination grids; and a `conventions` string | — |
| [`multiplicities_gtpack.wl`](multiplicities_gtpack.wl) | Independent GTPack computation of the multiplicities m_Γ(v) for the 32 crystallographic groups. It writes [`multiplicities_gtpack.json`](multiplicities_gtpack.json), which `salc_tables.py` compares against | GTPack |
| [`example_outputs/`](example_outputs) | Console output of `salc_fock_basis.wl` for CaOCH₃, one quantum per mode, and of the two overtone scripts that this folder no longer has (C₃ᵥ E for n = 6, T_d T₂ for N = 2 and N = 3). Those three files predate the current tables: their T_d states for N = 2 and N = 3 agree with `salc_tables.pdf`, but at C₃ᵥ n = 6 the sign of A₂ and the order of the two E copies differ, and they name the E partners 1E, 2E | — |

`salc_fock_basis.wl` warns if GTPack's irrep matrices are not unitary or if the multiplicities do not fill H(**v**). `salc_tables.py` stops on any failed check (see [below](#combination-bands)).

### Usage

The inputs sit at the top of the file:

```wolfram
(* salc_fock_basis.wl *)
groupName  = "C3v";
modeIrreps = {"A1", "A1", "A1", "A1", "E", "E", "E", "E"};   (* one entry per degenerate set *)
vQuanta    = {1, 1, 1, 1, 1, 1, 1, 1};                       (* quanta in each *)
circular   = True;    (* main-text basis for 2D irreps; False keeps GTPack's basis *)
numeric    = False;   (* True for large manifolds *)
```

```bash
wolframscript -file salc_fock_basis.wl
```

The script also leaves its full result in the association `SALCresult` (irrep matrices, characters, multiplicities, SALC vectors) for further use.

For repeated irreps the copy index k is fixed only up to a U(m_Γ) rotation (part (iii) of the symmetry-adapted Fock encoding theorem). The scripts pick one orthonormal choice by projecting occupation states in order; the harmonic tables of `salc_tables.pdf` label the copies by layer ℓ instead.

## Checks against the main text

From the worked examples of `salc_tables.pdf`:

| Case | `salc_tables.pdf` | Main text |
|---|---|---|
| One E quantum (C₃ᵥ) | μ = 1: \|0,1⟩ (ℓ = −1); μ = 2: \|1,0⟩ | partners 1, 2 = \|ℓ₄ = −1⟩, \|ℓ₄ = +1⟩ |
| Two E quanta | A₁ = \|1,1⟩ (ℓ = 0); E, μ = 1: \|2,0⟩ (ℓ = +2) | Sym²E = A₁ ⊕ E; U(C₃) = diag(ζ₃, 1, ζ₃²) on \|+2⟩, \|0⟩, \|−2⟩ |
| Two T₂ quanta (T_d) | E, μ = 1, 2: ζ₃ · 3^{−1/2} Σᵢ ζ₃^{∓i}\|2ᵢ⟩ (one common phase) | 3^{−1/2} Σᵢ ζ₃^{∓i}\|2ᵢ⟩ |

## SALC tables for all point groups

[`salc_tables.pdf`](salc_tables.pdf) tabulates the symmetry-adapted occupation-number states |μΓ; k⟩ of one degenerate vibrational mode (the PDF writes them as o3harmonics writes a coupled ket, with Γ above the partner μ, and names the copy κ) for the 59 finite groups of the species tables (C₁, C₁ₕ, S₂; C_n, C_nv, C_nh, S₂ₙ, D_n, D_nh, D_nd for n = 2…8; T, T_h, T_d, O, O_h, I, I_h). The linear groups are not tabulated; for C∞ᵥ and D∞h the PDF states the residue rule in the limit a₂ → ∞. The states are built exactly as in `salc_fock_basis.wl`: the Fock-space action, matrix-valued projectors, copies by projecting occupation states in order.

Labels, groups and elements print as in the species tables, in o3harmonics' notation. The irrep matrices are this code's own (those of `species_tables/pointgroups.py` before the import), because the residue rules and Law 1 need a swap that acts as the plain permutation. The kets are |n₊,n₋⟩, with a quantum in partner 1 counted in n₋ (so ℓ = −1), and |n_x n_y n_z⟩ in o3harmonics' partners x, y, z. In o3harmonics' partners |n₊,n₋⟩ is the occupation ket |n₋ n₊⟩; it is (−1)^{n₊}|n₋ n₊⟩ for the 18 irreps whose partner 2 has the opposite sign there, and (−1)^{n₋}|n₊ n₋⟩ for E₃ of D₄d, E₄ and E₅ of D₆d and E₅ to E₇ of D₈d (the map is in [`../species_tables/o3h_print.json`](../species_tables/o3h_print.json)). A complex pair ²E_k ⊕ ¹E_k is one mode; its partner 1 is ²E_k, the member with character e^{+2πik/n} on the generator.

Instead of one table per v, three classifications reduce the states for every v to finite data:

- **Parity.** ℓ (the vibrational angular momentum of a 2D mode, or the harmonic degree of a vector-like 3D mode) has the parity of v. Inversion and σ_h act as (parity)^v, so g/u and ′/″ alternate with v for u- and ″-type modes.
- **Residues (two-dimensional modes).** Each SALC lies in span{|n₊,n₋⟩, |n₋,n₊⟩}. Its species and coefficients depend only on ℓ = n₊ − n₋ modulo a₂, the least common multiple of the orders of the nonzero matrix entries of the mode (for an irreducible mode, o3harmonics' a₂ of its generating function). One table of a₂ rows therefore gives the SALCs for all v. Separably degenerate pairs (C_n, S₂ₙ, C_nh, T, T_h) have each |n₊,n₋⟩ as a SALC, so the table reduces to a list of species against ℓ mod a₂.
- **Harmonics (vector-like three-dimensional modes, D(g) = λ(g) P(g)ᵀ with λ one-dimensional).** These are the T modes of T, T_h, T_d, O and O_h, and T₁ of I and I_h. The manifold decomposes as

  Sym^v Γ = ⊕_{ℓ=v,v−2,…} r^{v−ℓ} H_ℓ, with r² = a†·a†.

  Every SALC is r^{v−ℓ} ψ_ℓ / ‖r^{v−ℓ} ψ_ℓ‖, with ψ_ℓ a G-adapted harmonic SALC and ‖r^{2k} ψ_ℓ‖² = 4^k k! (ℓ+3/2)_k, k = (v−ℓ)/2. The species of H_ℓ are the restriction D^{(ℓ,(−1)^ℓ)}↓G of the O(3) irrep with parity (−1)^ℓ, times λ^ℓ. So the vibrational overtones follow the same branching rules as the rotational sector, and the copy label of a repeated irrep becomes the layer ℓ.

The T₂, G and H modes of I and I_h fit neither rule, so they are listed explicitly (T₂ for v ≤ 6; G and H for v ≤ 3, since their v = 6 manifolds have 84 and 210 states), with multiplicities up to v = 12. Their G and H irrep bases are fixed by polynomials, which the tables list; they are not the o3harmonics bases that the species tables use.

**Worked examples** give explicit SALCs for v ≤ 6 for C₃ᵥ E, C₄ᵥ E, D₃ₕ E′, D₆ₕ E₁ᵤ, T_d E and T₂, and O_h T₁ᵤ. They follow the main text's conventions: for example, partner 1 of E is |0,1⟩ for one E quantum and |2,0⟩ for two.

### Combination bands

Quanta in several distinct modes, such as ν₃ + 2ν₄ or Sym⁶E ⊗ Sym³E, follow from the overtone tables through four laws. Section 3 of the PDF states them, and every group with degenerate modes has a combination paragraph.

- **Law 0: product.** H(v₁, v₂, …) = ⊗_s Sym^{v_s} Γ_s. Only quanta in the same mode are symmetrised, and two different modes of one species are distinct factors. The multiplicities have the generating function (1/|G|) Σ_{g∈G} χ_Γ(g)* / Π_s det[1 − t_s D^{Γ_s}(g)].
- **Law 1: total vibrational angular momentum (2D modes).** With a principal element P of order N, D_s(P) = diag(e^{2πik_s/N}, e^{−2πik_s/N}), and a parity element Q with D_s(Q) = (−1)^{q_s}, the product state |**ℓ**⟩ = ⊗_s |n₊⁽ˢ⁾, n₋⁽ˢ⁾⟩ has species fixed by ℓ_tot = Σ_s k_s ℓ_s mod N and p = Σ_s q_s ℓ_s mod 2. Each SALC lies in span{|**ℓ**⟩, |−**ℓ**⟩}. One table per group therefore covers every combination band of its 2D modes, for any number of modes and quanta. The PDF names P and Q by their o3harmonics symbols, such as S₃ for the principal element of C₃ₕ and D₃ₕ. Example, C₃ᵥ: Sym⁶E ⊗ Sym³E = 5A₁ ⊕ 5A₂ ⊕ 9E. The ℓ₁ + ℓ₂ ≡ 0 (mod 3) pairs give (|**ℓ**⟩ ± |−**ℓ**⟩)/√2, and the rest are E pairs whose partner 1 is the ℓ_tot ≡ 2 (mod 3) member.
- **Law 2: coupling of harmonic layers (vector-like 3D modes).** Sym^a Γ₁ ⊗ Sym^b Γ₂ = ⊕_{ℓ₁, ℓ₂} ⊕_{ℓ=|ℓ₁−ℓ₂|}^{ℓ₁+ℓ₂} D^{(ℓ,+)}↓G ⊗ [det_G^{ℓ₁+ℓ₂} λ₁^{ℓ₁} λ₂^{ℓ₂}]. The layers couple like ordinary angular momenta, with parity set by ℓ₁ + ℓ₂.
- **Law 3: general.** For other pairs, such as E ⊗ T₂ in T_d or the icosahedral G and H modes, use the product table of G and point-group Clebsch–Gordan coefficients. The PDF gives multiplicity grids for every pair of modes of the cubic and icosahedral groups.

Worked combination bands with explicit states:
- C₃ᵥ Sym⁶E ⊗ Sym³E and Sym²E ⊗ Sym²E;
- D₆ₕ Sym²E₁ᵤ ⊗ Sym²E₂g;
- T_d E ⊗ T₂ and Sym²T₂ ⊗ T₂;
- O_h T₁ᵤ ⊗ T₂g.

**Checks.** `python salc_tables.py` stops if any of these fails:
- **Residue tables:** each one is compared with the projector construction for every v ≤ 2a₂ + 2 and every ℓ.
- **Harmonic tables:** checked for ℓ ≤ 6 (explicit harmonic SALCs are listed up to ℓ = 6). The factorised SALCs and the norm formula are checked for v ≤ 6.
- **Every listed set of states** is orthonormal, complete, and transforms with the irrep matrices the tables use.
- **Multiplicities** agree with GTPack for the 32 crystallographic groups.
- **Law 1** is verified in every group with 2D modes, for every pair of modes and 1 ≤ v₁, v₂ ≤ 4, and for every triple at three patterns. The states it prescribes are checked to be orthonormal, complete and covariant.
- **Law 2** is verified for every pair of vector-like modes, 1 ≤ a, b ≤ 4.

```bash
python salc_tables.py                        # writes salc_tables.tex and .json (about a minute)
pdflatex salc_tables.tex                     # run twice for the table of contents
wolframscript -file multiplicities_gtpack.wl # optional: regenerates multiplicities_gtpack.json
```

## Requirements

- Mathematica or the Wolfram Engine with `wolframscript` (tested with 14.3), for the `.wl` scripts.
- [GTPack](https://gtpack.org) (tested with 1.4) for `salc_fock_basis.wl` and `multiplicities_gtpack.wl`, loaded with ``Needs["GroupTheory`"]``.
- Python ≥ 3.10 with NumPy and SymPy for `salc_tables.py`.

## Changes from earlier versions

- **o3harmonics notation.** `salc_tables.pdf` prints o3harmonics' irrep labels, groups and element symbols, with the complex pairs named as o3harmonics names the same state (the earlier ¹E_k is now ²E_k). The Law-1 elements are named by their symbols, so C₃ₕ and D₃ₕ have S₃ where the tables printed S₆, and likewise S₅ and S₇ for S₁₀ and S₁₄. `salc_tables.json` keeps its keys, holds the printed labels, and adds this code's labels and o3harmonics' record and ids.
- **Overtone scripts removed.** `overtones_C3v_E.wl` and `overtones_Td_T2.wl` were deleted (commits 3443275 and eb01dbf); `salc_tables.pdf` covers both cases. Their outputs remain in `example_outputs/`.
- **SALC script** (`salc_fock_basis.wl`).
  - Kets are printed in the main text's |n₊, n₋⟩ order (a quantum in partner 1 has ℓ = −1).
  - Its comments cite the main text's lemma and theorem with equation numbers, which no longer match the main text.
  - Single-mode inputs no longer fail.
