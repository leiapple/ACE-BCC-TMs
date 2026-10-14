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
    
    Attributes:
        element (str): Chemical symbol of the BCC element
        initial_guess_lattice_constant (float): Initial guess for lattice constant (Å)
        calculator (ASE Calculator): Calculator for energy/force computations
        structure (ASE Atoms): Initial dislocation structure
        relaxed_structure (ASE Atoms): Relaxed dislocation structure
        dd_map (matplotlib.figure): Differential displacement map of initial structure
        dd_map_relaxed (matplotlib.figure): DD map of relaxed structure
    """
    
    def __init__(self, element, initial_guess_lattice_constant, calculator):
        """
        Initialize the BCC screw dislocation dipole system.
        
        Args:
            element (str): Chemical symbol of the BCC metal (e.g., 'Fe', 'W')
            initial_guess_lattice_constant (float): Initial guess for lattice constant in Å
            calculator (ASE Calculator): Calculator for energy/force computations
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
        
        Returns:
            float: Equilibrium lattice constant in Å
        """
        # Create a BCC unit cell with initial guess lattice constant
        bcc_unit_cell = bulk(self.element, 'bcc', 
                            a=self.initial_guess_lattice_constant, 
                            cubic=True)
        bcc_unit_cell.calc = self.calculator
        
        # Optimize the cell using BFGS with cell filter
        optimizer = BFGS(FrechetCellFilter(bcc_unit_cell))
        optimizer.run(fmax=0.001)  # Relax until max force < 0.001 eV/Å
        
        # Extract the optimized lattice constant
        equilibrium_lattice_constant = bcc_unit_cell.get_cell()[0][0]
        
        return equilibrium_lattice_constant

    def calculate_elastic_constants(self):
        """
        Calculate elastic constants for a given crystal structure.
        :param structure: ASE Atoms object representing the crystal structure.
        :return: Elastic constants matrix (6x6).
        """
        # Create a BCC unit cell with initial guess lattice constant
        bcc_unit_cell = bulk(self.element, 'bcc', 
                            a=self.initial_guess_lattice_constant, 
                            cubic=True)
        bcc_unit_cell.calc = self.calculator
        
        # Optimize the cell using BFGS with cell filter
        optimizer = FIRE(FrechetCellFilter(bcc_unit_cell))
        optimizer.run(fmax=0.001)  # Relax until max force < 0.001 eV/Å
        print(bcc_unit_cell.get_cell())

        Cij, _ = fit_elastic_constants(bcc_unit_cell, symmetry='cubic', verbose=True)

        return Cij
    
    def create_dislocation_object(self, lattice_constant, C11, C12, C44, final_config='False'):
        """
        Create an atomman Dislocation object for BCC screw dislocation.
        
        Args:
            lattice_constant (float): Equilibrium lattice constant in Å
            C11, C12, C44 (float): Elastic constants in GPa
            
        Returns:
            atomman.defect.Dislocation: Configured dislocation object
        """
        # Convert units for atomman
        alat = uc.set_in_units(lattice_constant, 'angstrom')
        C11 = uc.set_in_units(C11, 'GPa')
        C12 = uc.set_in_units(C12, 'GPa')
        C44 = uc.set_in_units(C44, 'GPa')
        
        # Load BCC unit cell from prototype
        conventional_setting = 'i'  # Standard setting for BCC
        # (built locally instead of am.load('prototype', 'A2--W--bcc'), which needs
        # the online NIST potentials database)
        unit_cell = am.load('ase_Atoms',
                            bulk(self.element, 'bcc', a=lattice_constant, cubic=True))
        
        # Create elastic constants tensor
        elastic_constants = am.ElasticConstants(C11=C11, C12=C12, C44=C44)
        
        # Define dislocation properties for BCC a/2<111>{110} screw dislocation
        burgers_vector = np.array([0.5, 0.5, 0.5])  # a/2<111> Burgers vector
        slip_plane = np.array([1, -1, 0])           # {110} slip plane
        line_direction = np.array([0.5, 0.5, 0.5])  # Screw dislocation along <111>
        
        # Shift to center dislocation at easy core site
        shift_vector = np.array([0.0, 0.66666666666667, 0.0])
        shift_scale = True
        
        # Create dislocation object
        dislocation = am.defect.Dislocation(
            unit_cell, 
            elastic_constants, 
            burgers_vector, 
            line_direction, 
            slip_plane,
            m='x',
            n='y',
            conventional_setting=conventional_setting,
            shift=shift_vector, 
            shiftscale=shift_scale
        )
        
        return dislocation
    
    def relax_dislocation_dipole(self, dislocation, disloc_center = [0,0,0], fmax=0.01, optimizer='BFGS'):
        """
        Relax the dislocation dipole structure using ASE.
        
        Args:
            dislocation (atomman.defect.Dislocation): Dislocation object
            fmax (float): Convergence criterion for max force (eV/Å)
            optimizer (str): Optimization algorithm ('BFGS', 'LBFGS', or 'FIRE')
            
        Returns:
            tuple: (base_system, relaxed_dislocation_system)
        """
        # Create dislocation dipole system
        base_system, dislocation_system = dislocation.dipole(
            sizemults=[5, 4.5, 1],  # System size multipliers, 135 atoms
            #sizemults=[7, 5.5, 1], # 231 atoms
            center = disloc_center,  # Dislocation positions
            centerscale = False,
            boxtilt=True,            # Allow non-orthogonal box
            return_base_system=True  # Return reference system
        )
        
        # Convert to ASE Atoms object
        dislocation_dipole_ase, properties = dislocation_system.dump(
            'ase_Atoms', 
            return_prop=True
        )
        
        # Attach calculator and relax structure
        dislocation_dipole_ase.calc = self.calculator
        
        # Select optimizer
        if optimizer == 'BFGS':
            opt = BFGS(dislocation_dipole_ase)
        elif optimizer == 'LBFGS':
            opt = LBFGS(dislocation_dipole_ase)
        elif optimizer == 'FIRE':
            opt = FIRE(dislocation_dipole_ase)
        else:
            raise ValueError("Optimizer must be 'BFGS', 'LBFGS', or 'FIRE'")
        
        # Run relaxation
        opt.run(fmax=fmax)
        
        # Convert back to atomman system with original properties
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
        
        Args:
            dislocation (atomman.defect.Dislocation): Dislocation object
            base_system (atomman.System): Reference perfect crystal system
            dislocation_system (atomman.System): Dislocation system
            filename (str): Output filename for the plot
        """
        # Get lattice constant and Burgers vector
        lattice_constant = dislocation.ucell.box.a
        burgers_vector = dislocation.dislsol.burgers
        
        # Supersize systems along line direction for better visualization
        big_base_system = base_system.supersize(1, 1, 3)
        big_dislocation_system = dislocation_system.supersize(1, 1, 3)
        
        # Compute neighbor list and differential displacements
        neighbor_cutoff = 0.9 * lattice_constant
        neighbors = big_dislocation_system.neighborlist(cutoff=neighbor_cutoff)
        
        dd = am.defect.DifferentialDisplacement(
            big_base_system, 
            big_dislocation_system, 
            neighbors=neighbors, 
            reference=1
        )
        
        plot_params = {
            'ddmax': np.linalg.norm(burgers_vector) / 2,  # a/2<111> dislocations
            'plotxaxis': 'x',
            'plotyaxis': 'y',
            'xlim': (0, dislocation_system.box.avect[0] + dislocation_system.box.bvect[1] + self.initial_guess_lattice_constant),
            'ylim': (0, dislocation_system.box.bvect[1] + 1.0),
            'zlim': (lattice_constant * 3**0.5 / 2 - 0.01, 
                     2 * lattice_constant * 3**0.5 / 2 + 0.01),
            'figsize': 14,
            'arrowwidth': 1/100,
            'arrowscale': 2.5
        }
        
        dd.plot('z', use0z=True, atomcmap='rainbow', **plot_params)
        plt.title('DD map: '+self.element)

        # Save the plot
        plt.savefig(filename, dpi=300)
        plt.show()
        #plt.close()
