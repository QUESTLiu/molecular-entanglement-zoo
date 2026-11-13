# Ab initio constants for the CaOCH₃ state-preparation example

Scripts, inputs and raw outputs behind the constants marked "this work" in the calcium
monomethoxide example of the paper *Molecular Entanglement in symmetry-classified phase space* (E. R. Das, J.-C. Zhang,
V. V. Albert and Y. Liu, in preparation): the section "Preparing symmetry-adapted bend states of
calcium monomethoxide" and its appendix (LaTeX labels `sec:experiment` and `app:caoch3_prep`).

The paper takes four constants of the Ca–O–C bend ν₈ from these calculations: the Coriolis
constant ζ₈, the anharmonic constants x₈,₈ and g₈,₈, and the fundamental transition dipole μ₈,
which only converts Rabi frequencies into a laser intensity. All other constants are measured. The
level energies, line positions and detunings of the appendix follow from
Eq. `eq:caoch3_hamiltonian` alone with these constants; centrifugal distortion and the spin structure
are not added to them (their sizes are reported separately). The same spin structure, with the
electron Zeeman term added, gives the magnetic-field limits of the appendix. This folder holds
nothing else. The species tables of the same paper are in [`../species_tables/`](../species_tables/).

## Where each number in the paper comes from

Labels are the LaTeX labels of the manuscript. Paths are relative to this folder.

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
| `app:caoch3_pulses`, `tab:caoch3_constants` | μ₈ = ⟨1\|μ_x\|0⟩ (B3LYP, double harmonic); check: permanent dipole, harmonic ν₈ | 0.150 D; 1.45 D, 150.7 cm⁻¹ | `caoch3_bend_constants/bend_dipole.py` | `caoch3_bend_constants/bend_dipole_b3lyp.out` (0.1504; 1.447; 150.66) |
| `app:caoch3_pulses`, `tab:caoch3_pulses`, `sec:experiment` | common intensity (P7 at Ω/2π = 10 kHz), field, Ω/2π range, π-pulse durations τ_π, sequence totals, error bounds | 1.5 mW/cm², 107 V/m, 9.8–44 kHz, 11.4–50.9 µs, 91–217 µs, < 7×10⁻⁴ and < 4×10⁻⁹ | `caoch3_bend_constants/pulse_durations.py` (reads the two outputs above and `check_tree_measured.out`) | `caoch3_bend_constants/pulse_durations.out` |
| `app:caoch3_pulses` (condition (vi)) | Landé g_F of the tree levels, largest precession rate, transverse-field limit, largest quadratic line shift at B_Z = 50 mG | 0.17–0.39, 0.54 MHz/G, ≲ 20 µG, 4 kHz (P1) | `caoch3_bend_constants/zeeman_limits.py` (reads `check_tree_measured.py`) | `caoch3_bend_constants/zeeman_limits.out` |

`paper_links.json` holds this table in machine-readable form, with the exact LaTeX strings of
the manuscript next to the output file and pattern that give each number.

## ζ₈ (EA-EOM-CCSD)

Electron-attachment EOM-CCSD from the closed-shell CaOCH₃⁺ reference, with PySCF. Basis:
cc-pCVTZ on Ca with the 3s3p shell correlated, cc-pVTZ on O, C and H.

1. `caoch3_eom_ccsd/caoch3_surface.py --preset tz --state X --hess-guess caoch3_min_dz_X.json`
   optimises the X̃ ²A₁ geometry in C₃ᵥ by finite differences
   → `caoch3_eom_ccsd/out/caoch3_min_tz_X.json`. The double-zeta run (`--preset dz`) supplies the
   starting Hessian.
2. `caoch3_eom_ccsd/caoch3_emodes.py caoch3_min_tz_X.json --preset tz` builds the force field of
   the four e modes → `out/caoch3_emode_cache_tz_X.json` (single points),
   `out/caoch3_emode_basis_tz_X.json` (displacement vectors), `out/caoch3_emodes_tz_X.json`.
3. `caoch3_bend_constants/zeta_from_cc.py` forms the normal modes and ζ = l_x · (M_z l_y). The
   four ζ sum to 1 + B_e/2A_e to 10⁻⁵, as the sum rule requires.

Run steps 1 and 2 from `caoch3_eom_ccsd/out/` (`python ../caoch3_surface.py ...`). Every single
point is cached in a JSON file, so the commands replay the stored points and compute nothing new.

Keep `caoch3_emode_basis_tz_X.json` with its cache. The four displacement vectors come from a
singular-value decomposition with degenerate singular values, so another LAPACK build returns a
different set spanning the same space, and the cached energies are labelled by these vectors.

## x₈,₈ and g₈,₈ (B3LYP force field, second-order perturbation theory)

1. `dft_force_field.py` (UKS-B3LYP/def2-TZVP): C₃ᵥ geometry, analytic Hessian, and Hessians
   displaced by ±0.10 amu^½ Å along both ν₈ components → cubic constants with a ν₈ index and the
   ν₈ quartic constants (`dft_force_field_b3lyp_def2tzvp.npz`).
2. `vpt2_bend.py <npz>` applies second-order perturbation theory numerically in each v₈ manifold
   and fits the levels to Eq. `eq:caoch3_hamiltonian`.
3. `dft_half_step.py` (half the step) and `XC=pbe0 python dft_force_field.py` give the variation
   quoted in the appendix.

Run these from `caoch3_bend_constants/`. The same B3LYP and PBE0 force fields give ζ₈ = 0.9267
and 0.9260, in agreement with the coupled-cluster value.

## Level table

`python check_tree_measured.py` evaluates Eq. `eq:caoch3_hamiltonian` alone for the eight tree
levels, the seven transitions and their competing lines. It then reports, separately, the size of
centrifugal distortion (never added to the energies) and the spin-rotation and proton hyperfine
structure (used only for the Rabi-frequency limit). `python zeeman_limits.py` adds the electron
Zeeman term to that spin structure and gives the Landé factors and magnetic-field limits of
condition (vi) in Appendix `app:caoch3_pulses`. Measured constants are set at the top of
`check_tree_measured.py`; x₈,₈ and the anharmonic part of g₈,₈ are read from `anharmonic.json`. The script calls
the two k = 0 targets `L03_3` (N = 3) and `L03_4` (N = 4); the paper calls them L₀₃ and L₃₀.

## Software

Python 3.12–3.13, PySCF 2.14.0, NumPy 2.5.3, SciPy 1.18.1; SymPy for `check_tree_measured.py` and
`zeeman_limits.py` (`requirements.txt`). On an 18-core laptop the triple-zeta geometry optimisation took 1.5 h
(39 single points), the e-mode force field 1 h (21 points) and one B3LYP force field 9 min.
