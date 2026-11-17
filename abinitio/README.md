# Ab initio constants for the CaOCH₃ example

Scripts, inputs and raw outputs behind the calcium monomethoxide example of the paper
*Symmetry classification of hyperfine rovibrational molecular phase space* (E. R. Das,
J.-C. Zhang, V. V. Albert and Y. Liu, in preparation). The section
[Ab initio constants for calcium monomethoxide](../README.md#ab-initio-constants-for-calcium-monomethoxide)
of the repository README says how each number of the paper was computed, which script and stored
output gives it, and how to rerun the scripts.

| Path | Contents |
|---|---|
| `caoch3_eom_ccsd/` | EA-EOM-CCSD geometry optimization and e-mode force field; cached single points and logs in `out/` |
| `caoch3_bend_constants/` | ζ₈, the B3LYP force field and second-order perturbation theory, μ₈, the level, line and pulse checks, and the magnetic-field limits; its README lists every file |
| `requirements.txt` | package versions used for the stored outputs |
