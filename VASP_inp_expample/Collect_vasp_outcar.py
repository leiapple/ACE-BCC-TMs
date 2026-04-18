import numpy as np
import os
from ase.io import read, write

def extract_atoms(outcar_path):
    """
    Extracts atomic structure from a VASP OUTCAR file using ASE.

    Args:
        outcar_path (str): Path to the OUTCAR file.

    Returns:
        ase.Atoms: Atomic structure extracted from the OUTCAR file.
    """
    try:
        # Read the OUTCAR file using ASE
        atoms = read(outcar_path, format='vasp-out')
        print(f"Successfully processed {outcar_path}")
        return atoms
    except Exception as e:
        print(f"Error processing {outcar_path}: {e}")



def collect_outcar_data(main_folder):
    """
    Collects OUTCAR data from all subfolders of a main folder.

    Args:
        main_folder (str): Path to the main folder containing subfolders with OUTCAR files.

    Returns:
        dict: A dictionary with folder paths as keys and ASE Atoms objects as values.
    """
    outcar_data = []

    for root, dirs, files in os.walk(main_folder):
        if 'OUTCAR' in files:
            outcar_path = os.path.join(root, 'OUTCAR')
            atoms = extract_atoms(outcar_path)
            if atoms is not None:
                outcar_data.append(atoms)

    return outcar_data

main_folder = './calculations'
outcar_data = collect_outcar_data(main_folder)
output_file = 'v_eos.xyz'  # Output file name
write(output_file, outcar_data, format='extxyz')
