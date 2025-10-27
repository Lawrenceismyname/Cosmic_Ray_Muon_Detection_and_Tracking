#ch17 closest to you, ch 1 furthest from you

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def process_timing_data(input_file='processed_timing_data.csv'):
    # Read the processed timing data
    df = pd.read_csv(input_file)
    
    # Group by index and extract required channels
    processed_data = []
    
    for index_num, group in df.groupby('Index'):
        # Get ToA_ns values for channels 17 and 18
        tot_17 = group[group['CH_Id'] == 17]['ToT_ns'].values
        tot_18 = group[group['CH_Id'] == 18]['ToT_ns'].values
        
        # Get ToA_ns values for channels 17 and 18
        toa_17 = group[group['CH_Id'] == 17]['ToA_ns'].values
        toa_18 = group[group['CH_Id'] == 18]['ToA_ns'].values
        

        # Get the other channel (not 16, 17, or 18)
        other_channels = group[(group['CH_Id'] != 16) & 
                             (group['CH_Id'] != 17) & 
                             (group['CH_Id'] != 18)]
        
        if len(tot_17) == 1 and len(tot_18) == 1 and len(other_channels) == 1:
            other_ch_id = other_channels['CH_Id'].iloc[0]
            
            processed_data.append({
                'Index': index_num,
                'Other_Ch_Id': other_ch_id,
                'ToT_17': tot_17[0],
                'ToT_18': tot_18[0],
                'ToA_17': toa_17[0],
                'ToA_18': toa_18[0]
            })
    
    result_df = pd.DataFrame(processed_data)
    return result_df

# Process the timing data
result = process_timing_data()

# Display results
print("Processed Timing Data:")
print("=" * 60)
print(f"Total events processed: {len(result)}")
print(f"Other channels found: {sorted(result['Other_Ch_Id'].unique())}")

# Extract data for analysis
ch17_tot = result['ToT_17'].values
ch18_tot = result['ToT_18'].values

ch17_toa = result['ToA_17'].values
ch18_toa = result['ToA_18'].values

pos = result['Other_Ch_Id'].values

# Calculate ToA difference
toa_diff = ch17_toa - ch18_toa

# TOT as Time
tot_diff = (ch17_tot - ch18_tot)

# Convert position to physical distance
pos = pos.astype(float)
for i in range(len(pos)):
    pos[i] = 0.5 + pos[i] * 1.6


# Create blue to green color gradient based on position
positions = sorted(result['Other_Ch_Id'].unique())
n_positions = len(positions)

# Create blue to green colormap
colors = plt.cm.Blues_r(np.linspace(0, 0.9, n_positions))  # Blue shades
green_colors = plt.cm.Greens(np.linspace(0, 0.9, n_positions))  # Green shades

# Blend from blue to green
blended_colors = []
for i in range(n_positions):
    # Transition from blue to green
    blend_ratio = i / (n_positions - 1) if n_positions > 1 else 0
    color = (1 - blend_ratio) * np.array(colors[i][:3]) + blend_ratio * np.array(green_colors[i][:3])
    blended_colors.append(tuple(color) + (1,))  # Add alpha channel

# Create position to color mapping
pos_to_color = {pos: blended_colors[i] for i, pos in enumerate(positions)}

X = np.column_stack([np.ones(len(toa_diff)),toa_diff, tot_diff])

beta = np.linalg.pinv(X) @ pos

# Extract coefficients
intercept = beta[0]
coef_x1 = beta[1]
coef_x2 = beta[2]

print(f"\nRegression Coefficients:")
print(f"Intercept (beat0): {intercept:.4f}")
print(f"Coefficient for ToA_diff (beta1): {coef_x1:.4f}")
print(f"Coefficient for ToT_diff (beta2): {coef_x2:.4f}")

# Make predictions
y_pred = X @ beta

print(f"\nModel: y = {intercept:.4f} + {coef_x1:.4f}*ToA_diff + {coef_x2:.4f}*ToT_diff")

# Calculate R-squared
ss_res = np.sum((pos - y_pred) ** 2)
ss_tot = np.sum((pos - np.mean(pos)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.4f}")


# Calculate residuals (errors)
residuals = pos - y_pred



# Calculate uncertainty in coefficients
n = len(pos)  # number of observations
p = len(beta)  # number of parameters
dof = n - p   # degrees of freedom

# Calculate residual standard error
rse = np.sqrt(np.sum(residuals**2) / dof)

# Calculate covariance matrix of coefficients
try:
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_beta = XtX_inv * (rse**2)
    
    # Standard errors are the square root of diagonal elements
    std_errors = np.sqrt(np.diag(cov_beta))
    intercept_se = std_errors[0]
    coef_x1_se = std_errors[1]
    coef_x2_se = std_errors[2]
    
    # Calculate t-statistics and p-values
    t_stats = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    
except np.linalg.LinAlgError:
    print("Warning: Matrix inversion failed, using alternative method")
    # Alternative method using pseudoinverse
    std_errors = rse * np.sqrt(np.diag(np.linalg.pinv(X.T @ X)))
    intercept_se = std_errors[0]
    coef_x1_se = std_errors[1]
    coef_x2_se = std_errors[2]
    t_stats = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))


print(f"\nRegression Coefficients with Uncertainties:")
print(f"Intercept (beta0): {intercept:.4f} ± {intercept_se:.4f}")
print(f"Coefficient for ToA_diff (beta1): {coef_x1:.4f} ± {coef_x1_se:.4f}")
print(f"Coefficient for ToT_diff (beta2): {coef_x2:.4f} ± {coef_x2_se:.4f}")

# NEW: COVARIANCE AND ERROR PROPAGATION ANALYSIS
print(f"\n" + "="*50)
print("COVARIANCE AND ERROR PROPAGATION ANALYSIS")
print("="*50)

# Extract covariance between coefficients
cov_coef1_coef2 = cov_beta[1,2]  # Covariance between coef_x1 and coef_x2
corr_coef1_coef2 = cov_coef1_coef2 / (coef_x1_se * coef_x2_se)  # Correlation

print(f"Covariance between coefficients: {cov_coef1_coef2:.6f}")
print(f"Correlation between coefficients: {corr_coef1_coef2:.4f}")

# Calculate combined uncertainty using error propagation formula
print(f"\nError Propagation Analysis:")
print(f"Individual uncertainties: unc_ToA = {coef_x1_se:.4f}, unc_ToT = {coef_x2_se:.4f}")

# Case 2: Independent case (ρ = 0)
independent_combined = np.sqrt(coef_x1_se**2 + coef_x2_se**2)
print(f"Independent case : {independent_combined:.4f}")

# Case 3: Actual combined with covariance
# For Z = aX + bY, σ_Z² = a²σ_X² + b²σ_Y² + 2ab·Cov(X,Y)
# Here we're looking at the uncertainty in the linear combination of coefficients
var_combined = coef_x1_se**2 + coef_x2_se**2 + 2 * cov_coef1_coef2
actual_combined = np.sqrt(var_combined)
print(f"Actual with covariance: {actual_combined:.4f}")

# Calculate prediction uncertainty for typical values
print(f"\nPrediction Uncertainty for Typical Values:")
x1_typical = np.mean(toa_diff)
x2_typical = np.mean(tot_diff)

# Uncertainty in prediction: σ_pred² = σ_intercept² + x1²σ_coef1² + x2²σ_coef2² + 2*x1*x2*cov_coef1_coef2
var_pred = (intercept_se**2 + 
           (x1_typical**2 * coef_x1_se**2) + 
           (x2_typical**2 * coef_x2_se**2) + 
           2 * x1_typical * x2_typical * cov_coef1_coef2)

std_pred = np.sqrt(var_pred)
print(f"Typical ToA_diff: {x1_typical:.1f} ns")
print(f"Typical ToT_diff: {x2_typical:.1f} ns") 
print(f"Prediction uncertainty: ±{std_pred:.4f} mm")

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean(residuals**2))

print(f"\nRMSE: {rmse:.6f}")



# # Optional: Print first few predictions vs actual
# print("\nFirst 5 predictions vs actual:")
# for i in range(min(5, len(pos))):
#     print(f"  Pred: {y_pred[i]:.4f}, Actual: {pos[i]:.4f}, Error: {residuals[i]:.4f}")



# Create the plot
plt.figure(figsize=(10, 6))
# Plot actual vs predicted
plt.scatter(pos, y_pred, alpha=0.8)

# Perfect prediction line (y=x)
min_val = min(pos.min(), y_pred.min())
max_val = max(pos.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'red', linewidth=2, label='Perfect Prediction')
plt.plot([min_val, max_val], [min_val+rmse, max_val+rmse], 'pink', linewidth=2, label='±1 std dev')
plt.plot([min_val, max_val], [min_val-rmse, max_val-rmse], 'pink', linewidth=2)

# Add stats as text box
stats_text = f"""Regression Statistics:
R^2 = {r_squared:.3f}
Std Dev Error = {rmse:.3f}
n = {len(tot_diff)}"""

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