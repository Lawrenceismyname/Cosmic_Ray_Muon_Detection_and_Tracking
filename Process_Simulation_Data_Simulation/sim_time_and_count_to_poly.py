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

det1 = data['Det_1_Count']
det2 = data['Det_2_Count']
det1_t = data['Det_1_time']
det2_t = data['Det_2_time']
mu_offset = data['Mu_Offset']

# offset to cm
mu_offset = 100*mu_offset

remove_zeros = (det1>0)&(det2>0)
det1 = det1[remove_zeros]
det2 = det2[remove_zeros]
det1_t = det1_t[remove_zeros]
det2_t = det2_t[remove_zeros]
mu_offset= mu_offset[remove_zeros]



ratio = np.log(det1/det2)

diff = det1_t - det2_t


X = np.column_stack([np.ones(len(ratio)),ratio, diff])

beta = np.linalg.pinv(X) @ mu_offset

# Extract coefficients
intercept = beta[0]
coef_x1 = beta[1]
coef_x2 = beta[2]


print(f"\nRegression Coefficients:")
print(f"Intercept (beat0): {intercept:.4f}")
print(f"Coefficient for x1 (beta1): {coef_x1:.4f}")
print(f"Coefficient for x2 (beta2): {coef_x2:.4f}")

# Make predictions
y_pred = X @ beta

print(f"\nModel: y = {intercept:.4f} + {coef_x1:.4f}*x1 + {coef_x2:.4f}*x2")

# Calculate R-squared
ss_res = np.sum((mu_offset - y_pred) ** 2)
ss_tot = np.sum((mu_offset - np.mean(mu_offset)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.4f}")


# Calculate residuals (errors)
residuals = mu_offset - y_pred

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean(residuals**2))

print(f"RMSE: {rmse:.6f}")



# # Optional: Print first few predictions vs actual
# print("\nFirst 5 predictions vs actual:")
# for i in range(min(5, len(mu_offset))):
#     print(f"  Pred: {y_pred[i]:.4f}, Actual: {mu_offset[i]:.4f}, Error: {residuals[i]:.4f}")



# Create the plot
plt.figure(figsize=(10, 6))
# Plot actual vs predicted
plt.scatter(mu_offset, y_pred, alpha=0.8)

# Perfect prediction line (y=x)
min_val = min(mu_offset.min(), y_pred.min())
max_val = max(mu_offset.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'red', linewidth=2, label='Perfect Prediction')
plt.plot([min_val, max_val], [min_val+rmse, max_val+rmse], 'pink', linewidth=2, label='±1 std dev')
plt.plot([min_val, max_val], [min_val-rmse, max_val-rmse], 'pink', linewidth=2)

# Add stats as text box
stats_text = f"""Regression Statistics:
R^2 = {r_squared:.3f}
Std Dev Error = {rmse:.3f}
n = {len(diff)}"""

plt.text(0.01, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontfamily='monospace')

plt.xlabel('Actual Position')
plt.ylabel('Predicted Predicted')
plt.title('Predicted Values vs Muon Hit Position')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

