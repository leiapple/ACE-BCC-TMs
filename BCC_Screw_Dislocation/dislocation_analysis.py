# dislocation_analysis.py

import pandas as pd
from tqdm import tqdm
from bcc_dislocation import BCCScrewDislocation  # Import the class

def calculate_dislocation_properties(elements, a0_list, calculators, fmax=0.005, optimizer='BFGS'):
    """
    Calculate dislocation properties for multiple BCC elements.
    """
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



def calculate_peierls_barrier_and_trajectory(elements, a0_list, calculators,
                                             fmax=1e-3, spring_constant=0.1,
                                             Nreplica=11, optimizer='BFGS'):
    """
    Peierls barrier, core trajectory, and deviation angle for multiple BCC elements.

    For each element, a screw-dislocation dipole is built in the initial and in the
    neighbouring Peierls valley, relaxed, and connected by a climbing-image NEB
    (FIRE, ``Nreplica`` images including the end points). The core trajectory is
    obtained from the stress variation of the NEB images (stress method).

    Outputs per element (in the folder ``<element>/``): ``neb.traj``, ``neb_<i>.poscar``,
    ``peierls_barrier_data.csv``, ``peierls_barrier.png``, and ``<element>_traj_ACE.dat``.

    Returns:
        pandas.DataFrame with the Peierls barrier (meV/b) and the deviation angle (deg)
    """
    import numpy as np
    from peierls_neb import BCCScrewDislocPeierlsBarrier, rotated_elastic_constants, core_trajectory

    conversion = 160.21743091427834  # eV/A^3 to GPa
    rows = []
    for element, a0, calculator in tqdm(zip(elements, a0_list, calculators),
                                        total=len(elements), desc="Peierls NEB"):
        system = BCCScrewDislocation(element, a0, calculator)
        a0_eq = system.find_equilibrium_lattice_constant()
        Cij = system.calculate_elastic_constants()
        C11, C12, C44 = Cij[0, 0] * conversion, Cij[0, 1] * conversion, Cij[3, 3] * conversion

        # dipole in the initial and in the neighbouring Peierls valley
        disl = system.create_dislocation_object(a0_eq, C11, C12, C44)
        _, sys_initial = system.relax_dislocation_dipole(
            disl, disloc_center=[0, 0, 0], fmax=fmax, optimizer=optimizer)
        _, sys_final = system.relax_dislocation_dipole(
            disl, disloc_center=[a0_eq * np.sqrt(6) / 3, 0, 0], fmax=fmax, optimizer=optimizer)

        neb = BCCScrewDislocPeierlsBarrier(
            sys_initial.dump('ase_Atoms'), sys_final.dump('ase_Atoms'), calc=calculator,
            element=element, Nreplica=Nreplica, optimizer='FIRE', working_dir=element)
        neb.relax_initial_final(fmax=fmax)
        neb.run_neb(fmax=fmax, spring_constant=spring_constant)
        _, energies = neb.plot_barrier()

        x, y, alpha = core_trajectory(neb.get_images(write_poscar=False), calculator,
                                      rotated_elastic_constants(Cij), a0_eq)
        np.savetxt(f"{element}/{element}_traj_ACE.dat", np.column_stack((x, y)),
                   header="x (A)  y (A)", fmt="%.8f")

        rows.append({'Element': element, 'equilibrium_a0': a0_eq,
                     'Peierls_barrier_meV_per_b': energies.max(),
                     'deviation_angle_deg': alpha})

    return pd.DataFrame(rows)
