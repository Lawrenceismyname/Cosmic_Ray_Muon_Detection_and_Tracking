import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def read_detector_data(file_pattern="DetCount_0_nt_Hits_t*.csv"):
    """
    Read detector data from CSV files and organize into arrays
    """
    # Initialize lists to store data from all files
    det1_count_all = []
    det2_count_all = []
    det1_time_all = []
    det2_time_all = []
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
            det1_count_all.extend(df['Det_1_PhotonCount'].values)
            det2_count_all.extend(df['Det_2_PhotonCount'].values)
            det1_time_all.extend(df['Det_1_FirstHitTime_[ns]'].values)
            det2_time_all.extend(df['Det_2_FirstHitTime_[ns]'].values)
            mu_offset.extend(df['Muon_Offset_[m]'].values)
            
            file_names.append(file_path)
            print(f"Read {len(df)} entries from {file_path}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Convert to numpy arrays
    det1_count_array = np.array(det1_count_all)
    det2_count_array = np.array(det2_count_all)
    det1_time_array = np.array(det1_time_all)
    det2_time_array = np.array(det2_time_all)
    mu_offset_array = np.array(mu_offset)

    return {
        'Det_1_Count': det1_count_array,
        'Det_2_Count': det2_count_array,
        'Det_1_time': det1_time_array,
        'Det_2_time': det2_time_array,
        'Mu_Offset' : mu_offset_array,
        'file_names': file_names
    }

# Main execution
if __name__ == "__main__":
    # Read all detector data
    data = read_detector_data()
    
    # Now you can access the data
    print(f"Total entries: {len(data['Det_1_time'])}")
    
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


det1 = data['Det_1_time']
det2 = data['Det_2_time']
mu_offset = data['Mu_Offset']

# offset to cm
mu_offset = 100*mu_offset

remove_zeros = (det1>0)&(det2>0)
det1 = det1[remove_zeros]
det2 = det2[remove_zeros]
mu_offset= mu_offset[remove_zeros]

diff = det1 - det2

clean = (diff**2 < 81)
diff = diff[clean]
#scale to 1
mu_offset = mu_offset[clean]/12.5



from scipy import stats

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(diff, mu_offset)

# Calculate residuals and their standard deviation
y_pred = slope * diff + intercept
residuals = mu_offset - y_pred
std_residuals = np.std(residuals)
r_value = r_value**2

print(f"Standard deviation of residuals: {std_residuals:.3f}")
print(f"Standard error of slope estimate: {std_err:.3f}")

# PLOT
plt.scatter(diff, mu_offset, alpha=0.8, label='Data Scatter Plot')
plt.plot(diff, y_pred, 'red', linewidth=2, label='Regression')
# Add upper and lower bounds as dashed lines
plt.plot(diff, y_pred + std_residuals, 'pink', linewidth=2, label='±1 std dev')
plt.plot(diff, y_pred - std_residuals, 'pink' , linewidth=2)


# Add stats as text box
stats_text = f"""Regression Statistics:
x = {slope:.3f}*(t1 - t2) + {intercept:.3f}
R^2 = {r_value:.3f}
Std Dev Error = {std_residuals:.3f}
n = {len(diff)}"""

plt.text(0.5, 0.32, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontfamily='monospace')

plt.xlabel('ToA Difference of Photon Detected: t1 - t2 [ns]')
plt.ylabel('Position')
plt.title('Simulated ToA Difference vs Muon Hit Position')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
