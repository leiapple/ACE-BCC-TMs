import os
import ase.io
from ase.calculators.vasp import Vasp

# Equilibrium lattice constants
Nb_a0 = 3.307
Fe_a0 = 2.834
scale_factor = Nb_a0 / Fe_a0

# SLURM job script template
slurm_template = """#!/bin/bash
#SBATCH --job-name=Nb_pv_{i}
#SBATCH --output=Nb_pv_{i}.out
#SBATCH --error=Nb_pv_{i}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=parallelshort
#SBATCH --mem=8G

# Load necessary modules (if required)
module load VASP
module load Python

# Run the calculation
python run_vasp.py
"""

# Python script template for running VASP
run_vasp_template = """
from ase import Atoms
from ase.calculators.vasp import Vasp
import os

# Load the structure
from ase.io import read
atom_obj = read('structure_{i}.xyz')

# Set up the VASP calculator
calc = Vasp(
    prec='Accurate',
    algo='fast',
    xc="PBE", setups={{'Nb': '_sv'}},
    kspacing=0.15,
    ispin=2,
    isif=0,
    istart=0,
    icharg=2,
    lorbit=10,
    nelm=360,
    encut=500,
    ediff=1e-6,
    ismear=1,
    sigma=0.1,
    lasph=True,
    lreal=False,
    ldiag='T',
    lwave=False,
    npar=2
)

# Attach the calculator to the structure
atom_obj.calc = calc

# Run the calculation
atom_obj.get_potential_energy()
write('Nb_prim.xyz', atom_obj, format='extxyz')

"""

primitive_xyz = ase.io.read('representative_data.xyz',':')

# Loop through the first 1000 items in primitive_xyz
for i, item in enumerate(primitive_xyz[1:]):
    direc = 'Nb_prim_' + str(i)
    os.makedirs(direc, exist_ok=True)
    os.chdir(direc)

    # Modify the structure
    Nb_prim = item.copy()
    for atom in Nb_prim:
        atom.symbol = 'Nb'
    Nb_prim.set_cell(Nb_prim.cell * scale_factor, scale_atoms=True)

    # Save the structure to a file
    Nb_prim.write('structure_{}.xyz'.format(i))

    # Write the SLURM job script
    with open('job_{}.sh'.format(i), 'w') as f:
        f.write(slurm_template.format(i=i))

    # Write the Python script for running VASP
    with open('run_vasp.py', 'w') as f:
        f.write(run_vasp_template.format(i=i))

    # Submit the SLURM job
    os.system('sbatch job_{}.sh'.format(i))

    print('job {} submitted!'.format(i))
    os.chdir("../")
