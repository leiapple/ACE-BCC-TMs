# BCC screw dislocation scripts

Python tools (ASE, ATOMMAN, matscipy, pyace) for 1/2⟨111⟩ screw dislocations in bcc metals.

## Files

- `bcc_dislocation.py`: `BCCScrewDislocation` — equilibrium lattice parameter, cubic elastic constants, construction and relaxation of the 135-atom screw-dislocation dipole (in the initial or a neighbouring Peierls valley via `disloc_center`), and differential displacement (DD) maps.
- `peierls_neb.py`:
  - `BCCScrewDislocPeierlsBarrier` — relaxes the end points and runs a climbing-image NEB; writes the Peierls barrier in meV/b.
  - `core_trajectory` — dislocation core trajectory from the stress variation of the NEB images (stress method) and the deviation angle α.
  - `rotated_elastic_constants` — elastic constants in the dislocation frame.
- `dislocation_analysis.py`: drivers for several elements.
  - `calculate_dislocation_properties` — a0, Cij, and DD maps (Fig. 5(a)).
  - `calculate_peierls_barrier_and_trajectory` — Peierls barrier (Fig. 6, Table VII), core trajectory, and deviation angle (Fig. 5(b), Table II).
- `DD_maps.ipynb`: runs both drivers for V, Nb, Ta, Mo, and W with the ACE-2025 potentials.

## NEB settings

| Parameter | Value |
|---|---|
| Images | 11 (9 intermediate replicas from linear interpolation) |
| Optimizer | FIRE, climbing-image NEB |
| Force tolerance | 10⁻³ eV/Å |
| Spring constant | 0.1 eV/Å² |

## Requirements

```bash
pip install ase matscipy atomman numpy pandas matplotlib tqdm
```

and `pyace` from [ICAMS/python-ace](https://github.com/ICAMS/python-ace) for the ACE potentials.
