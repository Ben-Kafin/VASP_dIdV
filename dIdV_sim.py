from numpy import array,dot,exp,zeros
from numpy.linalg import norm
import numpy as np
import sys
import matplotlib.pyplot as plt
import getopt
from os.path import exists,getsize
from os import getcwd,chdir
from time import time
from pathos.multiprocessing import ProcessPool
sys.path.append('C:/Users/Benjamin Kafin/Documents/GitHub/VASP_dIdV/')
from lib import parse_doscar,parse_poscar,tunneling_factor
from scipy.ndimage import gaussian_filter
from pymatgen.io.vasp import Locpot
import copy
from scipy.integrate import simpson
from scipy.interpolate import interp1d
    




class dIdV_sim:
    
    def parse_VASP_output(self, **args):
        doscar = args.get("doscar", "./DOSCAR")
        poscar = args.get("poscar", "./POSCAR")
    
        try:
            if self.is_tip:
                self.tip_lv, self.tip_coord, self.tip_atomtypes, self.tip_atomnums = parse_poscar(poscar)[:4]
                self.tip_dos, self.tip_energies, self.tip_ef, self.tip_orbitals = parse_doscar(doscar)

            else:
                self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar)[:4]
                self.dos, self.energies, self.ef, self.orbitals = parse_doscar(doscar)

        except Exception as e:
            print(f"Error reading input files: {e}")
            sys.exit()
            


    def __init__(self, filepath, is_tip=False):
        """
        Initialize the dIdV_sim class.
    
        Parameters:
        - filepath (str): Path to the directory containing VASP output files.
        - is_tip (bool): Whether this instance is for the tip LDOS.
        """
        self.npts = 64
        self.emax = 0
        self.emin = 0
        self.estart = 0
        self.eend = 0
        self.is_tip = is_tip  # Flag to differentiate tip vs. system
        self.energies = None
        self.tip_energies = None  # Separate grid for tip LDOS
        self.dos = None
        self.tip_dos = None
        self.coord = None
        self.tip_coord = None
        self.ldos_at_position = None
        self.orbitals = []
    
        # Change working directory
        chdir(filepath)
    
        # Parse POSCAR file and handle dynamic return values
        poscar = "./POSCAR"
        if not exists(poscar):
            raise FileNotFoundError(f"POSCAR file not found at {poscar}")
    
        poscar_output = parse_poscar(poscar)
    
        if len(poscar_output) == 4:
            self.lv, self.coord, self.atomtypes, self.atomnums = poscar_output
            self.seldyn = None  # No `seldyn` data in POSCAR
        elif len(poscar_output) == 5:
            self.lv, self.coord, self.atomtypes, self.atomnums, self.seldyn = poscar_output
        else:
            raise ValueError("Unexpected number of outputs from parse_poscar. Expected 4 or 5.")
    
        # Parse DOSCAR file to initialize attributes
        self.dos, self.energies, self.ef, self.orbitals = parse_doscar('./DOSCAR')
    
        # Load LOCPOT file and calculate the work function
        #self.locpot = dIdV_sim.load_locpot("./LOCPOT")
        #self.work_function = dIdV_sim.calculate_work_function(self.locpot, self.ef, tolerance=0.05)




    def write_ldos(self):
        filename = './line_E{}to{}V_D{}'.format(self.emin, self.emax, self.npts)
        with open(filename, 'w') as file:
            file.write('Integration performed from {} to {} V over {} energy points\n'.format(self.emin, self.emax, self.eend - self.estart))
            file.write('Orbital contributions to LDOS: {}'.format(', '.join(self.orbitals)))
            file.write('\n\n')
            for axes in [self.path_distance, self.energies]:
                for i in range(self.npts):
                    for j in range(self.eend - self.estart):
                        file.write(str(axes[i][j]) + ' ')
                    file.write('\n')
                file.write('\n')
            for projection in self.ldos:
                for i in range(self.npts):
                    for j in range(self.eend - self.estart):
                        file.write(str(projection[i][j]) + ' ')
                    file.write('\n')
                file.write('\n')

                
    #reads in the ldos file created by self.write_ldos()
    def parse_ldos(self, filepath):
        header = filepath.split('_')
        if header[0][-3:] != 'line':
            print('Not an LDOS line file. Exiting...')
            sys.exit()
    
        erange = header[1][1:-1].split('to')
        self.emin = float(erange[0])
        self.emax = float(erange[1])
        self.npts = int(header[4][1:])
        self.phi = float(header[5][1:])
        self.unit_cell_num = int(header[6][1:])
    
        with open(filepath, 'r') as file:
            lines = file.readlines()
            nedos = int(lines[0].split()[8])
            self.orbitals = lines[2].split(', ')
            self.path_distance = array([[0.0 for _ in range(nedos)] for _ in range(self.npts)])
            self.energies = array([[0.0 for _ in range(nedos)] for _ in range(self.npts)])
            self.ldos = array([[[0.0 for _ in range(nedos)] for _ in range(self.npts)] for _ in range(len(self.orbitals))])
            for i in range(self.npts):
                for j in range(nedos):
                    self.path_distance[i][j] = lines[4 + i].split()[j]
                    self.energies[i][j] = lines[5 + self.npts + i].split()[j]
                    for k in range(len(self.orbitals)):
                        self.ldos[k][i][j] = lines[6 + k + (2 + k) * self.npts + i].split()[j]

    
    def calculate_ldos_at_position(self, position, **args):
        """
        Calculate the LDOS at a given position, dynamically determining the energy range
        from the available energy grid.
    
        Parameters:
        - position: np.array, 3D coordinates at which LDOS is calculated.
        """
        # Determine energy range dynamically
        self.emin = self.energies.min()
        self.emax = self.energies.max()
    
        # Determine energy range indices
        self.estart, self.eend = 0, len(self.energies)
        for i in range(len(self.energies)):
            if self.energies[i] < self.emin:
                self.estart = i
            if self.energies[i] > self.emax:
                self.eend = i
                break
    
        # Initialize LDOS for the specific position
        self.ldos_at_position = np.zeros((len(self.orbitals), self.eend - self.estart))
    
        # Perform LDOS integration over atomic contributions
        for atom_idx, atom_coord in enumerate(self.coord):
            posdiff = norm(position - atom_coord)
            sf = np.exp(-posdiff * 1.0e-1)  # Exponential decay factor based on distance
    
            for orbital_idx in range(len(self.orbitals)):
                # Safeguard to avoid out-of-bound access
                if atom_idx < len(self.dos) and orbital_idx < len(self.dos[atom_idx]):
                    dos_array = np.array(self.dos[atom_idx][orbital_idx][self.estart:self.eend])
                    self.ldos_at_position[orbital_idx] += dos_array * sf


    
    
    #performs integration at single point of the x,y grid when run in parallel
    def integrator(self, i):
        from numpy import array
        pos = array([self.x[i], self.y[i], self.z[i]])
        temp_ldos = zeros((len(self.orbitals), self.npts, self.eend - self.estart))
        counter = 1  # Properly initialize counter here
        for k in self.periodic_coord:
            if counter == sum(self.atomnums) + 1:
                counter = 1
            if counter - 1 not in self.exclude:
                posdiff = norm(pos - k)
                sf = exp(-1.0 * posdiff * self.K * 1.0e-10)
                for l in range(len(self.dos[counter])):
                    temp_ldos[l][i] += self.dos[counter][l][self.estart:self.eend] * sf
            counter += 1
        return temp_ldos


    

    def smear_spatial(self,dx):
        dx/=self.path_distance[1]
        for i in range(self.eend-self.estart):
            self.ldos[:,i]=gaussian_filter(self.ldos[:,i],dx,mode='constant')
        


    def plot_total_ldos(self, position):
        """
        Plot the total LDOS (summed over all orbitals) at the specified position.
        """
        if not hasattr(self, 'ldos_at_position'):
            raise AttributeError("LDOS at the specified position has not been calculated yet. Call 'calculate_ldos_at_position' first.")
    
        # Sum LDOS across all orbitals
        total_ldos = np.sum(self.ldos_at_position, axis=0)
    
        # Create a figure for the plot
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # Plot the total LDOS
        ax.plot(
            self.energies[self.estart:self.eend],  # Energy range
            total_ldos,                            # Total LDOS values
            label="Total LDOS",
            color="blue",
            linewidth=2
        )
    
        # Add labels, title, and legend
        ax.set_xlabel("Energy - $E_f$ (eV)")
        ax.set_ylabel("Total LDOS (arbitrary units)")
        ax.set_title(f"Total LDOS at Position {position}")
        ax.legend(loc="upper right")
        ax.grid(True)
    
        # Show the plot
        plt.show()
    

    # Function to interpolate, shift, and slice
    def interpolate_shift_and_slice(self, ldos, energies, shift, num_points):
        shifted_energies = energies + shift
        interp_func = interp1d(shifted_energies, ldos, kind='cubic', fill_value="extrapolate")
        
        new_energies = np.linspace(min(shifted_energies), max(shifted_energies), num_points)
        new_ldos = interp_func(new_energies)
        
        valid_indices = (new_energies >= energies[0]) & (new_energies <= energies[-1])
        return new_ldos[valid_indices], new_energies[valid_indices]
    
    def zero_bias_shift(self, sample_fermi, tip_fermi, tip_ldos, tip_energies):
        shift = sample_fermi - tip_fermi
        zero_bias_tip_ldos, zero_bias_tip_energies = self.interpolate_shift_and_slice(tip_ldos, tip_energies, shift, len(tip_energies))
        return shift, zero_bias_tip_ldos, zero_bias_tip_energies
    
    def calculate_didv(self, sample_ldos, sample_energies, zero_bias_tip_ldos, zero_bias_tip_energies, vmin, vmax, npts, work_function):
        biases = np.linspace(vmin, vmax, npts)
    
        def slice_and_resample(ldos, energies, vmin, vmax, npts):
            """
            Slice the LDOS curve to the range [vmin, vmax] and resample to match the shape of `biases`.
            """
            target_energies = np.linspace(vmin, vmax, npts)
            interp_func = interp1d(energies, ldos, kind='cubic', bounds_error=False, fill_value=0)
            resampled_ldos = interp_func(target_energies)
            return resampled_ldos, target_energies
    
        def normalize_curve(ldos):
            """
            Normalize the LDOS curve.
            """
            norm_factor = np.trapz(ldos, biases)
            if norm_factor < 1e-10:  # Avoid division by a very small number
                print("Warning: Norm factor is very small, skipping normalization.")
                return ldos
            return ldos / norm_factor
    
        tunneling_current = []
        for bias in biases:
            if bias == 0:
                tunneling_current.append(0.0)  # Zero bias produces zero tunneling current
                continue

            if bias < 0:
                # Handle negative biases
                shifted_sample_ldos, shifted_energies = self.interpolate_shift_and_slice(
                    sample_ldos, sample_energies, -bias, len(sample_energies)
                )
                valid_tip_ldos = zero_bias_tip_ldos
                valid_tip_energies = zero_bias_tip_energies
            
                # Slice and resample both shifted_sample_ldos and valid_tip_ldos to align with biases
                sample_interp_func = interp1d(shifted_energies, shifted_sample_ldos, kind='cubic', bounds_error=False, fill_value=0)
                sliced_sample_ldos = sample_interp_func(biases)
            
                tip_interp_func = interp1d(valid_tip_energies, valid_tip_ldos, kind='cubic', bounds_error=False, fill_value=0)
                sliced_tip_ldos = tip_interp_func(biases)
                
                sliced_tip_ldos /= np.trapz(sliced_tip_ldos, biases); sliced_sample_ldos /= np.trapz(sliced_sample_ldos, biases)
            
                # Compute overlap after slicing and resampling
                overlap = sliced_tip_ldos * sliced_sample_ldos
            
                # Define integration range directly using biases
                integration_range = biases[(biases >= 0) & (biases <= -bias)]
            
                # Ensure the integration range is valid
                if integration_range.size == 0:
                    raise ValueError(f"Integration range is empty for bias {bias}. Check energy grid and slicing conditions.")
            
                # Slice the overlap curve using the integration range
                overlap_sliced = overlap[(biases >= 0) & (biases <= -bias)]
            
                # Get tunneling factors for the integration range
                tunneling_factors_sliced = tunneling_factor(-bias, integration_range, work_function)
                
                # Compute the weighted overlap
                weighted_overlap = overlap_sliced * tunneling_factors_sliced
            
                # Integrate the weighted overlap over the integration range
                current = simpson(weighted_overlap, integration_range)
                tunneling_current.append(-current)  # Negative bias produces negative current

            else:
                # Handle positive biases
                shifted_tip_ldos, shifted_energies = self.interpolate_shift_and_slice(
                    zero_bias_tip_ldos, zero_bias_tip_energies, bias, len(zero_bias_tip_energies)
                )
                valid_sample_ldos = sample_ldos
                valid_sample_energies= sample_energies
            
                # Slice and resample both shifted_tip_ldos and valid_sample_ldos to align with biases
                tip_interp_func = interp1d(shifted_energies, shifted_tip_ldos, kind='cubic', bounds_error=False, fill_value=0)
                sliced_tip_ldos = tip_interp_func(biases)
            
                sample_interp_func = interp1d(valid_sample_energies, valid_sample_ldos, kind='cubic', bounds_error=False, fill_value=0)
                sliced_sample_ldos = sample_interp_func(biases)
                
                sliced_tip_ldos /= np.trapz(sliced_tip_ldos, biases); sliced_sample_ldos /= np.trapz(sliced_sample_ldos, biases)
            
                # Compute overlap after slicing and resampling
                overlap = sliced_tip_ldos * sliced_sample_ldos
            
                # Define integration range directly using biases
                integration_range = biases[(biases >= 0) & (biases <= bias)]
            
                # Ensure the integration range is valid
                if integration_range.size == 0:
                    raise ValueError(f"Integration range is empty for bias {bias}. Check energy grid and slicing conditions.")
            
                # Slice the overlap curve using the integration range
                overlap_sliced = overlap[(biases >= 0) & (biases <= bias)]
            
                # Get tunneling factors for the integration range
                tunneling_factors_sliced = tunneling_factor(bias, integration_range, work_function)
            
                # Compute the weighted overlap
                weighted_overlap = overlap_sliced * tunneling_factors_sliced
            
                # Integrate the weighted overlap over the integration range
                current = simpson(weighted_overlap, integration_range)
                tunneling_current.append(current)  # Positive bias produces positive current
            '''    
            if np.isclose(bias, vmin, atol=1e-6) or np.isclose(bias, vmax, atol=1e-6):
                plot_label = "vmin" if np.isclose(bias, vmin, atol=1e-6) else "vmax"
            
                # Plot normalized LDOS curves
                plt.figure(figsize=(10, 6))
                plt.plot(biases, sliced_tip_ldos, label=f"Normalized Tip LDOS ({plot_label})", color='blue')
                plt.plot(biases, sliced_sample_ldos, label=f"Normalized Sample LDOS ({plot_label})", color='orange')
                plt.xlabel("Bias Voltage (V)")
                plt.ylabel("Normalized LDOS")
                plt.title(f"LDOS at {plot_label}")
                plt.legend()
                plt.grid()
                plt.show()
            
                # Plot overlap curve with shaded area restricted to the integration range
                plt.figure(figsize=(10, 6))
            
                # Mask for restricting the shading to the integration range
                mask = (biases >= integration_range.min()) & (biases <= integration_range.max())
            
                plt.plot(biases, overlap, label=f"Overlap Curve ({plot_label})", color='green')
            
                # Shaded area under the overlap curve using tunneling weights for shading intensity
                plt.fill_between(
                    biases[mask],  # Biases within the integration range
                    0, 
                    overlap[mask],  # Overlap values within the integration range
                    color='gray', 
                    alpha=1,  # Normalize tunneling weights for shading
                    edgecolor='none'
                )
            
                plt.xlabel("Bias Voltage (V)")
                plt.ylabel("Overlap")
                plt.title(f"Overlap Curve with Weighted Shading ({plot_label})")
                plt.legend()
                plt.grid()
                plt.show()
'''


    
        tunneling_current = np.array(tunneling_current)
        #tunneling_current_smoothed = gaussian_filter(tunneling_current, sigma=0)
        didv = np.gradient(tunneling_current, biases)

    
        # Plot Tunneling Current
        plt.figure(figsize=(10, 6))
        plt.plot(biases, tunneling_current, label='Tunneling Current')
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Tunneling Current (A)")
        plt.title("Tunneling Current vs Bias Voltage")
        plt.legend()
        plt.grid()
        plt.show()
    
        # Plot dI/dV Curve
        plt.figure(figsize=(10, 6))
        plt.plot(biases, didv, label='dI/dV Curve')
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("dI/dV (a.u.)")
        plt.title("dI/dV Curve vs Bias Voltage")
        plt.legend()
        plt.grid()
        plt.show()
    
        return didv


    
   
    







# Main script
filepath_sys = 'system filepath'
position_sys = np.array([0,   0, 28.34])

# Initialize the system LDOS
system_ldos = dIdV_sim(filepath_sys, is_tip=False)

# Use attributes extracted and calculated during initialization
system_ef = system_ldos.ef

# Debug outputs
print(f"System Fermi Level (system_ef): {system_ef}")


# Calculate and plot LDOS
system_ldos.calculate_ldos_at_position(position_sys)
sigma = 1.0
smoothed_system_ldos = gaussian_filter(np.sum(system_ldos.ldos_at_position, axis=0), sigma)
system_ldos.plot_total_ldos(position_sys)


filepath_tip = 'tip filepath'
position_tip = np.array([5.87769, 10.18070, 23.59848])

# Initialize the tip LDOS
tip_ldos = dIdV_sim(filepath_tip, is_tip=True)

# Use attributes extracted and calculated during initialization
tip_ef = tip_ldos.ef

# Debug outputs
print(f"Tip Fermi Level (tip_ef): {tip_ef}")


# Calculate and plot LDOS
tip_ldos.calculate_ldos_at_position(position_tip)
smoothed_tip_ldos = gaussian_filter(np.sum(tip_ldos.ldos_at_position, axis=0), sigma=1.0)
tip_ldos.plot_total_ldos(position_tip)





if __name__ == '__main__':
    # Define bias voltage range for the simulation
    vmin = -2.25  # Minimum bias voltage (eV)
    vmax = 2.25   # Maximum bias voltage (eV)
    npts = 501   # Number of steps in the bias voltage range

    shift, zero_bias_tip_ldos, zero_bias_tip_energies = tip_ldos.zero_bias_shift(
        sample_fermi=system_ef,
        tip_fermi=tip_ef,
        tip_ldos=smoothed_tip_ldos,
        tip_energies=tip_ldos.energies[tip_ldos.estart:tip_ldos.eend]
    )

    print(shift)

    # Run the dI/dV calculation
    # Correctly call calculate_didv as an instance method
    didv_curve = tip_ldos.calculate_didv(
        sample_ldos=smoothed_system_ldos,
        sample_energies=system_ldos.energies[system_ldos.estart:system_ldos.eend],
        zero_bias_tip_ldos=zero_bias_tip_ldos,
        zero_bias_tip_energies=zero_bias_tip_energies,
        vmin=vmin,
        vmax=vmax,
        npts=npts,
        work_function=shift  # Using shift as the effective work function
    )


    # Optionally, save the dI/dV data to a file
    output_file = 'didv_curve.txt'
    with open(output_file, 'w') as f:
        f.write('Bias Voltage (V), dI/dV (A/V)\n')
        for voltage, didv_value in zip(np.linspace(vmin, vmax, npts), didv_curve):
            f.write(f"{voltage:.6f}, {didv_value:.6e}\n")

    print(f"dI/dV curve saved to {output_file}")
