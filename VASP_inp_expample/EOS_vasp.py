from ase import Atoms
from ase.calculators.vasp import Vasp
import numpy as np
import matplotlib.pyplot as plt
import os
import json


# Vanadium parameters
symbol = 'W'
# BCC
#alat = 3.0
# FCC
#alat = 3.79  # Initial guess for lattice parameter in Å (BCC experimental value)
# HCP
# alat = 2.60 
# Define structures
structures = {
    'bcc': {'structure': 'bcc', 'kpoints': [16, 16, 16]},
    'fcc': {'structure': 'fcc', 'kpoints': [16, 16, 16]},
   'hcp': {'structure': 'hcp', 'kpoints': [12, 12, 8], 'c/a': 1.780575}  # Ideal c/a ratio
}

# Volume scaling factors (typically 0.90 to 1.10 in steps of 0.02)
volumes = np.linspace(0.85, 1.15, 30)

def save_results(results, filename='results.json'):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, data in results.items():
            json_results[name] = {
                'volumes': [float(v) for v in data['volumes']],
                'energies': [float(e) for e in data['energies']]
            }
        json.dump(json_results, f, indent=4)

def get_vasp_calculator(structure_type, kpts):
    return Vasp(
        prec='Accurate',
        xc='PBE',
        algo='Fast',
        kpts=kpts,
        encut=500,  # High cutoff energy for V (d-electron system)
        nelm=360,   # Max electronic steps
        nelmin=5,
        ediff=1e-6, # Energy convergence criterion
        ibrion=2,
        isif=0,
        istart=0,
        icharg=2,   # Read charge density from WAVECAR        
        ispin=1,    # Spin-polarized calculation (V is magnetic)
        ismear=1,   # Methfessel-Paxton smearing
        sigma=0.1,  # Smearing width in eV
        ldiag=True,  # Use diagonalization for electronic structure
        lasph=True,  # Use ASR for better convergence
        lreal=False,
        lwave=False,
        LPLANE =True,
        LSCALU =False,    
        npar=4,
        setups={'W': '_sv'}  # Use pseudo-potential with p valence electrons
    )

results = {key: {'volumes': [], 'energies': []} for key in structures}

for name, params in structures.items():
    print(f"\nCalculating {name.upper()} structure")
    
    # Create directory for calculations
    os.makedirs(f'calculations/{name}', exist_ok=True)
    os.chdir(f'calculations/{name}')
    
    for scale in volumes:
        # 
        os.makedirs(f'{scale}')
        os.chdir(f'{scale}')
        
        # Create atoms object
        if name == 'bcc':
            alat = 3.17
            vol_scale = scale ** 3
            a = alat * scale
            atoms = Atoms(symbol, positions=[(0, 0, 0)], cell=[a, a, a], pbc=True)
            atoms.set_cell([[a, 0, 0], [0, a, 0], [a/2, a/2, a/2]], scale_atoms=True)
        elif name == 'fcc':
            alat = 3.99 
            vol_scale = scale ** 3
            a = alat * scale
            atoms = Atoms(symbol, positions=[(0, 0, 0)], cell=[a, a, a], pbc=True)
            atoms.set_cell([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]], scale_atoms=True)
        elif name == 'hcp':
            alat = 2.78
            a = alat * scale
            c = a * params['c/a']
            vol_scale = a * a * c
            atoms = Atoms(2*symbol, 
                         positions=[(0, 0, 0), 
                                   (a/3, a*2/3, c/2)],
                         cell=[[a/2, -a*np.sqrt(3)/2, 0], 
                               [a/2, a*np.sqrt(3)/2, 0], 
                               [0, 0, c]],
                         pbc=True)
        
        # Set calculator
        atoms.calc = get_vasp_calculator(name, params['kpoints'])

        # Calculate energy
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        
        # Store results
        results[name]['volumes'].append(volume)
        results[name]['energies'].append(energy)        
        os.chdir('../')
        
        # Save incremental results
        save_results(results, f'../results.json')

        print(f"Scale: {scale:.3f}, Volume: {volume:.3f} Å³, Energy: {energy:.6f} eV")
    
    os.chdir('../..')  # Return to main directory
