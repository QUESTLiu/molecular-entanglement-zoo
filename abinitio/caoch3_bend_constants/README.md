# CaOCH₃ ν₈ bend constants (used in the paper)

Sources of the "this work" entries of Table `tab:caoch3_constants` and of the numbers derived
from them in Appendix `app:caoch3_prep`. The table of paper numbers is in the
[repository README](../../README.md#ab-initio-constants-for-calcium-monomethoxide).

Run the scripts from this folder. `zeta_from_cc.py`, `dft_force_field.py`, `dft_half_step.py`,
`vpt2_bend.py` and `bend_dipole.py` import the geometry and symmetry routines from `../caoch3_eom_ccsd/`.

| File | What it does | Output |
|---|---|---|
| `zeta_from_cc.py` | Coriolis constants of the four e modes from the EA-EOM-CCSD/TZ e-block force field in `../caoch3_eom_ccsd/out/`; sum-rule check | `zeta_from_cc.out` |
| `dft_force_field.py` | UKS/def2-TZVP geometry, analytic Hessian, Hessians displaced along ν₈ → cubic and quartic constants. `XC=b3lyp` (default) or `XC=pbe0` | `dft_force_field_<xc>_def2tzvp.npz`, `dft_force_field_<xc>.log` |
| `dft_half_step.py <npz>` | the same displaced Hessians at half the step, for Richardson extrapolation | `dft_force_field_b3lyp_def2tzvp_rich.npz`, `dft_half_step_b3lyp.log` |
| `vpt2_bend.py <npz>` | numerical second-order perturbation theory in each v₈ manifold; fit to Eq. `eq:caoch3_hamiltonian` → x₈,₈, g₈,₈ | `vpt2_bend_b3lyp.out`, `vpt2_bend_pbe0.out` |
| `anharmonic.json` | the two values the paper uses, x₈,₈ = −0.53 and g₈,₈ anharmonic part = 0.63 cm⁻¹ (B3LYP, C₃ᵥ averaged) | read by `check_tree_measured.py` |
| `check_tree_measured.py` | from Eq. `eq:caoch3_hamiltonian` alone: level energies (Table `tab:caoch3_levels`), line positions and couplings (Eq. `eq:caoch3_lines`, Table `tab:caoch3_pulses`), competing lines. Reported separately: size of centrifugal distortion, spin structure and the Rabi-frequency limit (Appendix `app:caoch3_pulses`) | `check_tree_measured.out` |
| `bend_dipole.py <npz>` | UKS-B3LYP/def2-TZVP dipole moments at ±0.10 amu^½ Å along both ν₈ components of `dft_force_field_b3lyp_def2tzvp.npz` → ∂μ/∂Q and μ₈ = ⟨1\|μ_x\|0⟩ = (ħ/2ω)^½ ∂μ_x/∂Q (double harmonic); also the equilibrium dipole as a check | `bend_dipole_b3lyp.out` |
| `pulse_durations.py` | one intensity for all seven tones, set so that P7 has Ω/2π = 10 kHz; field, intensity, Ω of each pulse against its hyperfine bound, π-pulse durations (last column of Table `tab:caoch3_pulses`) and sequence totals. Plain Python, no ab initio step | `pulse_durations.out` |
| `zeeman_limits.py` | adds the electron Zeeman term to the spin structure of `check_tree_measured.py`: g_F of each tree level (numerical and weak-field formula), transverse-field limit for 10⁻³ loss from M_F = 0 over 0.217 ms, and, with a bias field along Z, the M_F = 0 → ±1 splittings and quadratic line shifts (condition (vi) of Appendix `app:caoch3_pulses`). Plain Python, no ab initio step | `zeeman_limits.out` |

Which block of `vpt2_bend_b3lyp.out` the paper quotes: "Richardson False, C3v average True".
The "Richardson True" block and `vpt2_bend_pbe0.out` give the spread of 0.11 cm⁻¹ stated in the
appendix.

Level names: the script calls the two k = 0 targets `L03_3` (N = 3) and `L03_4` (N = 4); the
paper calls them L₀₃ and L₃₀.

`HARMONIC=1 python check_tree_measured.py` sets x₈,₈ and the anharmonic part of g₈,₈ to zero.
