# Kink-pair NEB calculation: LAMMPS example

Example LAMMPS workflow for the stress-dependent kink-pair nucleation enthalpy ΔH(τ) of a 1/2⟨111⟩ screw dislocation computed with the ACE-2025 potentials (Sec. III D of the paper). The example is for Mo at an applied shear stress τ = 200 MPa, in a cell of (l_x = 60a) × (l_y = 40b) × (l_z = 40c) containing 96 000 atoms.

## Method

- Periodic array of dislocations (PAD): periodic along x = [12̄1] (glide direction) and y = [1̄11] (dislocation line), free surfaces along z = [101].
- The shear stress is applied with `fix addforce` to the top and bottom boundary slabs. The force per atom is f(τ) = τ l_x l_y / N_atoms, where N_atoms is the number of atoms in the slab.
- With `fix_modify ... energy yes`, the work of the applied forces is subtracted from the potential energy. The reported `pe` is therefore the enthalpy H = E_p − W_ext [Eqs. (6) and (7) of the paper].
- NEB: 32 replicas, FIRE minimizer, force tolerance 5 × 10⁻³ eV/Å.
- The per-atom extrapolation grade γ of the ACE potential is computed with `pace/extrapolation` and written to the dump files.

## Files

| Folder | Input | Purpose |
|---|---|---|
| `0_relax_initial/` | `input_BCC`, `initial.data` | Relax the straight dislocation in the initial Peierls valley under the applied stress |
| `0_relax_final/` | `input_BCC`, `final.data` | Relax the dislocation in the neighbouring Peierls valley |
| `1_NEB_configs/` | `run.sh`, `input_restart2data*`, `data2neb_input.py` | Convert the relaxed restart files into the NEB initial and final configurations |
| `2_NEB_calc/` | `inputBCC` | Run the NEB and write the enthalpy of each replica |
| `potential.in` | | Points to `ACE2025_potential/Mo/output_potential.{yaml,asi}` in this repository |

## Running

Requires LAMMPS built with the ML-PACE package (`pair_style pace/extrapolation`) and the REPLICA package (`neb`).

```bash
cd 0_relax_initial && mpirun -np 64 lmp -in input_BCC && cd ..
cd 0_relax_final   && mpirun -np 64 lmp -in input_BCC && cd ..
cd 1_NEB_configs   && bash run.sh && cd ..
cd 2_NEB_calc      && mpirun -np 256 lmp -partition 32x8 -in inputBCC -l log.neb.log
```

The enthalpy of replica *i* is the third line of `2_NEB_calc/data.i`. The kink-pair nucleation enthalpy is the maximum along the path.

To change the applied stress, edit `variable tau_app` (in MPa) in `0_relax_*/input_BCC` and `2_NEB_calc/inputBCC`. For another element, change `potential.in` and the `mass` lines, and generate the initial configuration for that element.

The ΔH(τ) results for Nb, Mo, and W, together with the Kocks' law and line-tension fits (Fig. 11 and Table VI), are in [`../LT_model/LT_model.ipynb`](../LT_model/LT_model.ipynb).
