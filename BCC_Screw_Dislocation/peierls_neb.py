"""
peierls_neb.py

Peierls barrier and core trajectory of 1/2<111> screw dislocations in bcc metals.

- ``BCCScrewDislocPeierlsBarrier``: NEB calculation of the Peierls barrier with ASE
  for the 135-atom screw-dislocation dipole built with ``bcc_dislocation.py``
  (Sec. II C of the paper).
- ``core_trajectory``: dislocation core trajectory along the NEB path from the
  stress variation of the replicas (stress method, Refs. [12,15] of the paper),
  and the deviation angle alpha (Table II).
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from ase.optimize import BFGS, FIRE, LBFGS
from ase.filters import FrechetCellFilter
from ase.mep import NEB


class BCCScrewDislocPeierlsBarrier:
    """Peierls barrier of a screw-dislocation dipole in a bcc metal from NEB."""

    def __init__(self, initial_config, final_config, calc=None, element='',
                 model_name='ACE-2025', Nreplica=11, optimizer='FIRE', working_dir='.'):
        """
        Args:
            initial_config, final_config (ase.Atoms): dipole in the initial and the
                neighbouring Peierls valley
            calc (ase Calculator): interatomic potential
            element (str): chemical symbol (used for labels only)
            model_name (str): label used in the plot
            Nreplica (int): total number of images, including the two end points
            optimizer (str): 'FIRE', 'BFGS' or 'LBFGS'
            working_dir (str): output directory
        """
        self.initial_config = initial_config
        self.final_config = final_config
        self.calc = calc
        self.element = element
        self.model_name = model_name
        self.Nreplica = Nreplica
        self.optimizer = optimizer.upper()
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        if self.optimizer not in ['FIRE', 'BFGS', 'LBFGS']:
            raise ValueError(f"Optimizer must be 'FIRE', 'BFGS', or 'LBFGS', not {optimizer}")

    def _get_optimizer(self, atoms, trajectory=None):
        if trajectory:
            trajectory = os.path.join(self.working_dir, trajectory)
        opt = {'FIRE': FIRE, 'BFGS': BFGS, 'LBFGS': LBFGS}[self.optimizer]
        return opt(atoms, trajectory=trajectory)

    def relax_initial_final(self, fmax=1e-3, steps=10000):
        """Fully relax (positions and cell) the initial and final configurations."""
        for config in [self.initial_config, self.final_config]:
            config.calc = self.calc
            self._get_optimizer(FrechetCellFilter(config)).run(fmax=fmax, steps=steps)
        E_diff = (self.final_config.get_potential_energy()
                  - self.initial_config.get_potential_energy())
        if abs(E_diff) > 1e-3:
            print(f'WARNING: initial and final energies differ by {E_diff*1000:.2f} meV')
        return E_diff

    @staticmethod
    def _fix_pbc_issues(initial, final):
        """Apply the minimum-image convention to the final positions."""
        cell = final.get_cell()
        pos_i = initial.get_positions()
        pos_f = final.get_positions()
        for i, vec in enumerate(pos_f - pos_i):
            min_dist, best_shift = np.inf, np.zeros(3)
            for a, b, c in np.ndindex(3, 3, 3):
                shift = (a - 1) * cell[0] + (b - 1) * cell[1] + (c - 1) * cell[2]
                dist = np.linalg.norm(vec + shift)
                if dist < min_dist:
                    min_dist, best_shift = dist, shift
            pos_f[i] += best_shift
        return pos_f

    @staticmethod
    def _interpolate(initial, final, ratio):
        """Linear interpolation between the end points (centre-of-mass motion removed)."""
        for config in [initial, final]:
            if config.constraints:
                del config.constraints
        delta = final.get_positions() - initial.get_positions()
        image = initial.copy()
        image.set_positions(initial.get_positions() + ratio * (delta - delta.mean(axis=0)))
        return image

    def run_neb(self, fmax=1e-3, steps=10000, spring_constant=0.1):
        """Climbing-image NEB between the relaxed end points."""
        initial = self.initial_config.copy()
        final = self.final_config.copy()
        final.set_cell(initial.get_cell(), scale_atoms=True)
        final.set_positions(self._fix_pbc_issues(initial, final))
        images = [self._interpolate(initial, final, r)
                  for r in np.linspace(0, 1, self.Nreplica)]
        for image in images:
            image.calc = self.calc
        neb = NEB(images, k=spring_constant, allow_shared_calculator=True, climb=True)
        self._get_optimizer(neb, trajectory='neb.traj').run(fmax=fmax, steps=steps)

    def get_images(self, trajectory='neb.traj', write_poscar=True):
        """Return the converged NEB images (and optionally write them as POSCAR files)."""
        images = read(os.path.join(self.working_dir, f'{trajectory}@-{self.Nreplica}:'))
        if write_poscar:
            for i, image in enumerate(images):
                image.write(os.path.join(self.working_dir, f'neb_{i}.poscar'), format='vasp')
        return images

    def plot_barrier(self, trajectory='neb.traj', save_csv=True, vasp_data_file=None):
        """
        Peierls barrier in meV/b. The cell is one Burgers vector long and contains two
        dislocations, so the energy is divided by 2.
        """
        images = self.get_images(trajectory)
        reaction_coords = np.linspace(0, 1, self.Nreplica)
        energies = []
        for image in images:
            image.calc = self.calc
            energies.append(image.get_potential_energy())
        energies = 1000 * (np.array(energies) - min(energies)) / 2

        if save_csv:
            pd.DataFrame({'Reaction_Coordinate': reaction_coords,
                          'Energy_meV': energies}).to_csv(
                os.path.join(self.working_dir, 'peierls_barrier_data.csv'), index=False)

        plt.figure(figsize=(4, 3.5))
        plt.plot(reaction_coords, energies, '-o', label=self.model_name)
        if vasp_data_file is not None and os.path.isfile(vasp_data_file):
            vasp = np.loadtxt(vasp_data_file)
            plt.plot(vasp[:, 1] / vasp[:, 1].max(), vasp[:, 2] * 1000 / 2, '-s', label='DFT-VASP')
        plt.xlabel('Reaction coordinate')
        plt.ylabel('Energy (meV/b)')
        plt.title(f'{self.element}: Peierls barrier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.working_dir, 'peierls_barrier.png'), dpi=300)
        plt.show()
        return reaction_coords, energies


def rotated_elastic_constants(Cij):
    """
    Rotate the cubic elastic constants (6x6, any units) to the dislocation frame used by
    atomman: x = [-1-12] (glide direction), y = [1-10] (slip-plane normal),
    z = [111] (dislocation line).
    """
    from matscipy.elasticity import rotate_elastic_constants
    slip_plane = np.array([1, -1, 0], dtype=float)
    glide_direction = np.array([-1, -1, 2], dtype=float)
    line_direction = np.cross(slip_plane, glide_direction)
    A = np.array([v / np.linalg.norm(v) for v in (glide_direction, slip_plane, line_direction)])
    return rotate_elastic_constants(Cij, A)


def core_trajectory(images, calc, C6, a0):
    """
    Dislocation core trajectory along the NEB path from the stress method.

    The position along the glide direction is assumed to vary linearly between the
    initial and the final (neighbouring) Peierls valley, i.e. from 0 to a0*sqrt(6)/3.
    The position normal to the glide plane follows from the stress variation of each
    replica relative to the initial one. Both dislocations of the dipole move, so the
    displacement is divided by 2.

    Args:
        images (list of ase.Atoms): NEB images (initial ... final)
        calc: ASE calculator
        C6 (6x6 array): elastic constants in the dislocation frame (eV/A^3)
        a0 (float): lattice parameter (A)

    Returns:
        x, y (arrays, A) and the deviation angle alpha (deg)
    """
    b = a0 * np.sqrt(3) / 2
    Ypos = []
    stress_ref = None
    for image in images:
        image.calc = calc
        stress = image.get_stress(voigt=True)
        if stress_ref is None:
            stress_ref = stress
        S = np.linalg.norm(np.cross(image.get_cell()[0], image.get_cell()[1]))
        XX = np.linalg.solve(C6 / b / S, (stress - stress_ref).T)
        Ypos.append(-XX[4] / b**2)

    n = len(images)
    x = np.linspace(0.0, a0 * np.sqrt(6) / 3, n)   # initial to final Peierls valley
    y = (np.array(Ypos) - Ypos[0]) / 2

    # deviation angle: from the initial point to the point of largest deviation
    i_min = np.argmin(y)
    alpha = np.degrees(np.arctan2(y[i_min] - y[0], x[i_min] - x[0]))
    return x, y, alpha
