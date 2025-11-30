# bcc_dislocation.py

# general imports
import numpy as np
import matplotlib.pyplot as plt

# ASE imports
from ase.optimize import BFGS, LBFGS, FIRE
from ase.filters import FrechetCellFilter
from ase.build import bulk

# atomman packages
import atomman as am
import atomman.unitconvert as uc

# matscipy imports
from matscipy.elasticity import fit_elastic_constants

class BCCScrewDislocation:
    """
    A class to create, relax, and analyze screw dislocation dipoles in BCC metals.
    """
    
    def __init__(self, element, initial_guess_lattice_constant, calculator):
        """
        Initialize the BCC screw dislocation dipole system.
        """
        self.element = element
        self.initial_guess_lattice_constant = initial_guess_lattice_constant
        self.calculator = calculator
        
        # Initialize attributes that will be set later
        self.structure = None
        self.relaxed_structure = None
        self.dd_map = None
        self.dd_map_relaxed = None
    
    def find_equilibrium_lattice_constant(self):
        """
        Calculate the equilibrium lattice constant for the BCC metal.
        """
        # Your implementation here
        bcc_unit_cell = bulk(self.element, 'bcc', 
                            a=self.initial_guess_lattice_constant, 
                            cubic=True)
        bcc_unit_cell.calc = self.calculator
        
        optimizer = BFGS(FrechetCellFilter(bcc_unit_cell))
        optimizer.run(fmax=0.001)
        
        equilibrium_lattice_constant = bcc_unit_cell.get_cell()[0][0]
        return equilibrium_lattice_constant

    def calculate_elastic_constants(self):
        """
        Calculate elastic constants for a given crystal structure.
        """
        # Your implementation here
        bcc_unit_cell = bulk(self.element, 'bcc', 
                            a=self.initial_guess_lattice_constant, 
                            cubic=True)
        bcc_unit_cell.calc = self.calculator
        
        optimizer = FIRE(FrechetCellFilter(bcc_unit_cell))
        optimizer.run(fmax=0.001)
        print(bcc_unit_cell.get_cell())

        Cij, _ = fit_elastic_constants(bcc_unit_cell, symmetry='cubic', verbose=True)
        return Cij
    
    def create_dislocation_object(self, lattice_constant, C11, C12, C44):
        """
        Create an atomman Dislocation object for BCC screw dislocation.
        """
        # Your implementation here
        alat = uc.set_in_units(lattice_constant, 'angstrom')
        C11 = uc.set_in_units(C11, 'GPa')
        C12 = uc.set_in_units(C12, 'GPa')
        C44 = uc.set_in_units(C44, 'GPa')
        
        conventional_setting = 'i'
        unit_cell = am.load('prototype', 'A2--W--bcc', 
                           a=alat, 
                           symbols=self.element)
        
        elastic_constants = am.ElasticConstants(C11=C11, C12=C12, C44=C44)
        
        burgers_vector = np.array([0.5, 0.5, 0.5])
        slip_plane = np.array([1, -1, 0])
        line_direction = np.array([0.5, 0.5, 0.5])
        
        shift_vector = np.array([0.0, 0.0, 0.66666666666667])
        shift_scale = True
        
        dislocation = am.defect.Dislocation(
            unit_cell, 
            elastic_constants, 
            burgers_vector, 
            line_direction, 
            slip_plane,
            conventional_setting=conventional_setting,
            shift=shift_vector, 
            shiftscale=shift_scale
        )
        
        return dislocation
    
    def relax_dislocation_dipole(self, dislocation, fmax=0.01, optimizer='BFGS'):
        """
        Relax the dislocation dipole structure using ASE.
        """
        # Your implementation here
        base_system, dislocation_system = dislocation.dipole(
            sizemults=[1, 7, 5.5],
            boxtilt=True,
            return_base_system=True
        )
        
        dislocation_dipole_ase, properties = dislocation_system.dump(
            'ase_Atoms', 
            return_prop=True
        )
        
        dislocation_dipole_ase.calc = self.calculator
        
        if optimizer == 'BFGS':
            opt = BFGS(dislocation_dipole_ase)
        elif optimizer == 'LBFGS':
            opt = LBFGS(dislocation_dipole_ase)
        elif optimizer == 'FIRE':
            opt = FIRE(dislocation_dipole_ase)
        else:
            raise ValueError("Optimizer must be 'BFGS', 'LBFGS', or 'FIRE'")
        
        opt.run(fmax=fmax)
        
        relaxed_system = am.load(
            'ase_Atoms', 
            dislocation_dipole_ase, 
            prop=properties
        )
        
        return base_system, relaxed_system
    
    def plot_differential_displacement_map(
        self, 
        dislocation, 
        base_system, 
        dislocation_system, 
        filename='dislocation.png'
    ):
        """
        Generate and save a differential displacement (DD) map.
        """
        # Your implementation here
        lattice_constant = dislocation.ucell.box.a
        burgers_vector = dislocation.dislsol.burgers
        
        big_base_system = base_system.supersize(3, 1, 1)
        big_dislocation_system = dislocation_system.supersize(3, 1, 1)
        
        neighbor_cutoff = 0.9 * lattice_constant
        neighbors = big_dislocation_system.neighborlist(cutoff=neighbor_cutoff)
        
        dd = am.defect.DifferentialDisplacement(
            big_base_system, 
            big_dislocation_system, 
            neighbors=neighbors, 
            reference=1
        )
        
        plot_params = {
            'ddmax': np.linalg.norm(burgers_vector) / 2,
            'plotxaxis': 'y',
            'plotyaxis': 'z',
            'xlim': (0, dislocation_system.box.bvect[1] + dislocation_system.box.cvect[2] + self.initial_guess_lattice_constant),
            'ylim': (0, dislocation_system.box.cvect[2] + 1.0),
            'zlim': (lattice_constant * 3**0.5 / 2 - 0.01, 
                     2 * lattice_constant * 3**0.5 / 2 + 0.01),
            'figsize': 14,
            'arrowwidth': 1/100,
            'arrowscale': 2.5
        }
        
        dd.plot('x', use0z=True, atomcmap='rainbow', **plot_params)
        plt.title('DD map: '+self.element)
        plt.savefig(filename, dpi=300)
        plt.close()