# LT Model

This folder contains a Jupyter notebook for the line-tension (LT) model of Dorn and Rajnak, used to predict the stress-dependent kink-pair nucleation enthalpy ΔH*(τ) of screw dislocations in bcc transition metals.

## Files

- `LT_model.ipynb`: defines the LT-model helper functions, determines the line-energy ratio `R` and the Peierls-potential shape parameter `alpha` from atomistic inputs (zero-stress kink-pair formation enthalpy ΔH*(0), Peierls barrier ΔE_P, lattice parameter a0, and Peierls stress τ_P), and plots the predicted ΔH*(τ) curves for Nb, Mo, and W.
  It also fits Kocks' law, ΔH*(τ) = ΔH*(0)[1 − (τ/τ_P)^p]^q, to the 3D NEB enthalpies and reproduces Fig. 11 and Table VI of the paper.

## Inputs (ACE-2025, T = 0 K)

| Element | ΔH*(0) (eV) | ΔE_P (eV/b) | a0 (Å) | τ_P (GPa) |
|---|---|---|---|---|
| Nb | 0.722 | 0.0449 | 3.308165 | 1.375 |
| Mo | 1.073 | 0.0576 | 3.163423 | 1.525 |
| W  | 1.488 | 0.0986 | 3.186619 | 2.875 |

## Results (Table VI)

| Element | p | q | α | Γ_0 (eV/Å) |
|---|---|---|---|---|
| Nb | 0.865 | 1.613 | −1.114 | 2.534 |
| Mo | 0.742 | 1.149 | −0.203 | 2.712 |
| W  | 0.854 | 1.633 | −0.631 | 3.621 |

The 3D NEB data are computed with the LAMMPS workflow in [`../Kink_pair_NEB_LAMMPS_example`](../Kink_pair_NEB_LAMMPS_example).

## Running the notebook

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook LT_model.ipynb
```
