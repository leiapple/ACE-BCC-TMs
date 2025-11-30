# dislocation_analysis.py

import pandas as pd
from tqdm import tqdm
from bcc_dislocation import BCCScrewDislocation  # Import the class

def calculate_dislocation_properties(elements, a0_list, calculators, fmax=0.005, optimizer='BFGS'):
    """
    Calculate dislocation properties for multiple BCC elements.
    """
    # Your function implementation here (same as your original code)
    columns = [
        'Element', 'initial_a0', 'equilibrium_a0',
        'C11_GPa', 'C12_GPa', 'C44_GPa',
        'anisotropy_ratio', 'bulk_modulus', 'shear_modulus',
        'status'
    ]
    results_df = pd.DataFrame(columns=columns)
    
    if len(elements) != len(a0_list):
        raise ValueError("Length of elements and a0_list must match")
    
    for element, a0, calculator in tqdm(zip(elements, a0_list, calculators), total=len(elements), desc="Processing elements"):
        try:
            dislocation_system = BCCScrewDislocation(element, a0, calculator)
            
            a0_eq = dislocation_system.find_equilibrium_lattice_constant()
            Cij = dislocation_system.calculate_elastic_constants()
            
            conversion = 160.21743091427834
            C11 = Cij[0,0] * conversion
            C12 = Cij[0,1] * conversion
            C44 = Cij[3,3] * conversion
            
            B = (C11 + 2*C12)/3
            G = (C11 - C12 + 3*C44)/5
            A = 2*C44/(C11 - C12)
            
            bcc_disl = dislocation_system.create_dislocation_object(a0_eq, C11, C12, C44)
            base_system, disl_system = dislocation_system.relax_dislocation_dipole(
                bcc_disl, fmax=fmax, optimizer=optimizer
            )
            
            output_file = f"{element}_screw_DD.png"
            dislocation_system.plot_differential_displacement_map(
                bcc_disl, base_system, disl_system, filename=output_file
            )
            
            new_row = {
                'Element': element,
                'initial_a0': a0,
                'equilibrium_a0': a0_eq,
                'C11_GPa': C11,
                'C12_GPa': C12,
                'C44_GPa': C44,
                'anisotropy_ratio': A,
                'bulk_modulus': B,
                'shear_modulus': G,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"\nError processing {element}: {str(e)}")
            new_row = {
                'Element': element,
                'initial_a0': a0,
                'status': f"failed: {str(e)}"
            }
        
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return results_df