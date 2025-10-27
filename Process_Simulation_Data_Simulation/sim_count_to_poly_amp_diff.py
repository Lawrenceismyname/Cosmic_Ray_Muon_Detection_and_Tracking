import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

def read_detector_count_data(file_pattern="DetCount_0_nt_Hits_t*.csv"):
    """
    Read detector energy data from CSV files and organize into arrays
    """
    # Initialize lists to store data from all files
    det1_energy_all = []
    det1_count_all = []
    det2_energy_all = []
    det2_count_all = []
    mu_offset = []
    file_names = []
    
    # Get all matching files
    files = glob.glob(file_pattern)
    files.sort()  # Sort to ensure consistent order
    
    for file_path in files:
        try:
            # Read CSV file, skipping header lines starting with #
            df = pd.read_csv(file_path, comment='#', header=None, 
                           names=['Det_1_Photon_Energy_eV', 'Det_1_PhotonCount', 
                                  'Det_2_Photon_Energy_eV', 'Det_2_PhotonCount', 'Muon_Offset_[m]', 'Det_1_FirstHitTime_[ns]', 'Det_2_FirstHitTime_[ns]'])
            
            # Extract data
            det1_energy_all.extend(df['Det_1_Photon_Energy_eV'].values)
            det1_count_all.extend(df['Det_1_PhotonCount'].values)
            det2_energy_all.extend(df['Det_2_Photon_Energy_eV'].values)
            det2_count_all.extend(df['Det_2_PhotonCount'].values)
            mu_offset.extend(df['Muon_Offset_[m]'].values)
            
            file_names.append(file_path)
            print(f"Read {len(df)} entries from {file_path}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Convert to numpy arrays
    det1_energy_array = np.array(det1_energy_all)
    det1_count_array = np.array(det1_count_all)
    det2_energy_array = np.array(det2_energy_all)
    det2_count_array = np.array(det2_count_all)
    mu_offset_array = np.array(mu_offset)

    return {
        'Det_1_Energy': det1_energy_array,
        'Det_1_Count': det1_count_array,
        'Det_2_Energy': det2_energy_array,
        'Det_2_Count': det2_count_array,
        'Mu_Offset' : mu_offset_array,
        'file_names': file_names
    }

# Main execution
if __name__ == "__main__":
    # Read all detector data
    data = read_detector_count_data()
    
    # Now you can access the data
    print(f"Total entries: {len(data['Det_1_Count'])}")
    print(f"Detector 1 Count array shape: {data['Det_1_Count'].shape}")
    print(f"Detector 2 Count array shape: {data['Det_2_Count'].shape}")
    
    # # Print some statistics
    # print(f"\nDetector 1 Energy - Min: {data['Det_1_Energy'].min():.2f} eV, "
    #       f"Max: {data['Det_1_Energy'].max():.2f} eV, "
    #       f"Mean: {data['Det_1_Energy'].mean():.2f} eV")
    
    # print(f"Detector 2 Energy - Min: {data['Det_2_Energy'].min():.2f} eV, "
    #       f"Max: {data['Det_2_Energy'].max():.2f} eV, "
    #       f"Mean: {data['Det_2_Energy'].mean():.2f} eV")
    
    # # Show first few values
    # print(f"\nFirst 5 Detector 1 energies: {data['Det_1_Energy'][:5]}")
    # print(f"First 5 Detector 2 energies: {data['Det_2_Energy'][:5]}")


det1 = data['Det_1_Count']
det2 = data['Det_2_Count']
mu_offset = data['Mu_Offset']

flux = (np.sum(det1)+np.sum(det2))/(2*1024)
print(flux)

# offset to cm
mu_offset = 100*mu_offset

remove_zeros = (det1>0)&(det2>0)
det1 = det1[remove_zeros]
det2 = det2[remove_zeros]
mu_offset= mu_offset[remove_zeros]


ratios = (det1 - det2) / (det1 + det2)
# ratios = np.log(det1 / det2)
# ratios = det1 / det2

# # Plot
# plt.scatter(ratios, mu_offset, label='Data points')
# plt.xlabel('Log Ratio')
# plt.ylabel('Position along bar [m]')
# plt.title('SIMULATION: Natural Log Ratios ln(1_amps / 2_amps) vs Known Position:')
# plt.show()


# end of the bar would give bad ratios
one_percent = int(round(0.01*len(ratios), 0))

cleaned_ratios = ratios #[one_percent:len(ratios)-one_percent]
cleaned_offsets = mu_offset #[one_percent:len(ratios)-one_percent]

#scale to 1
scaled_cleaned_offsets = cleaned_offsets/12.5


# # Fit 3rd degree polynomial
# poly = Polynomial.fit(ratios, scaled_cleaned_offsets, deg=3)


# # Get the polynomial coefficients
# coefficients = poly.convert().coef
# print("Polynomial coefficients (constant term first):", coefficients)

# # Evaluate the polynomial at new points
# x_new = np.linspace(min(ratios), max(ratios), len(ratios))
# y_new = poly(x_new)

# residuals = scaled_cleaned_offsets - y_new
# std_residuals = np.std(residuals)

# print(std_residuals)

# # Plot original data and fitted curve
# plt.scatter(ratios, scaled_cleaned_offsets, alpha=0.8, label='Data Scatter Plot')
# plt.plot(x_new, y_new, 'r-', linewidth = 2 ,label='3rd degree polynomial fit')

# # Add stats as text box
# stats_text = f"""Polynomial Statistics:
# r = (a1 - a2) / (a1 + a2)
# x = {coefficients[3]:.3f}*r^3 + {coefficients[2]:.3f}*r^2 + {coefficients[1]:.3f}*r + {coefficients[0]:.3f}
# Std Dev Error = {std_residuals:.3f}
# n = {len(ratios)}"""

# plt.text(0.02, 0.25, stats_text, transform=plt.gca().transAxes, 
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#          fontfamily='monospace')

# plt.legend()
# plt.show()




# Define the tangent function to fit
def tan_func(x, a, b, c, d):
    """
    a * tan(b * x + c) + d
    """
    return a * np.tan(b * x + c) + d


# Fit tangent function
try:
    # Initial parameter guess [a, b, c, d]
    # You may need to adjust these based on your data characteristics
    p0 = [1.0, 1.0, 0.0, 0.0]
    
    # Perform the fit
    popt, pcov = curve_fit(tan_func, ratios, scaled_cleaned_offsets, p0=p0)
    
    # Extract optimized parameters
    a, b, c, d = popt
    
    # Evaluate the tangent function at new points
    x_new = np.linspace(min(ratios), max(ratios), len(ratios))
    y_new = tan_func(x_new, a, b, c, d)
    
    # Calculate residuals and standard deviation
    residuals = scaled_cleaned_offsets - tan_func(ratios, a, b, c, d)
    std_residuals = np.std(residuals)
    
    print(f"Tangent function parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
    print(f"Standard deviation of residuals: {std_residuals:.3f}")
    
    # Plot original data and fitted curve
    plt.scatter(ratios, scaled_cleaned_offsets, alpha=0.8, label='Data Scatter Plot')
    plt.plot(x_new, y_new, 'r-', linewidth=2, label='Tangent function fit')

    plt.plot(x_new, y_new + std_residuals, 'pink', linewidth=2, label='±1 std dev')
    plt.plot(x_new, y_new - std_residuals, 'pink', linewidth=2)
    
    # Add stats as text box
    stats_text = f"""Tangent Function Statistics:
r = (a1 - a2) / (a1 + a2)
x = {a:.3f} * tan({b:.3f}*r + {c:.3f}) + {d:.3f}
Std Dev Error = {std_residuals:.3f}
n = {len(ratios)}"""
    
    plt.text(0.02, 0.25, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace')
    
    plt.xlabel('Amplitude Difference Ratio of Photon Count Detected: (a1 - a2)/(a1 - a2)')
    plt.ylabel('Position')
    plt.title('Simulated Photon Count Detected vs Muon Hit Position')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
except Exception as e:
    print(f"Fit failed: {e}")
    print("Try adjusting the initial parameter guesses (p0)")