import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the datasets
timing_data = pd.read_csv('processed_timing_data.csv')
pha_data = pd.read_csv('processed_validated_positions_PHA.csv')

print(f"Original timing data shape: {timing_data.shape}")
print(f"Original PHA data shape: {pha_data.shape}")

# Merge datasets on Index and CH_Id
merged_data = pd.merge(timing_data, pha_data, on=['Index', 'CH_Id'], how='inner')
print(f"Merged data shape: {merged_data.shape}")

# Pivot the data to have channels as columns for each position
# Create a dataset where each row represents a position with data from multiple channels
pivot_timing = timing_data.pivot_table(index='Index', columns='CH_Id', 
                                    values=['ToA_ns', 'ToT_ns'], 
                                    aggfunc='first')

pivot_pha = pha_data.pivot_table(index='Index', columns='CH_Id', 
                               values='PHA_LG', aggfunc='first')

# Flatten column names
pivot_timing.columns = [f'{col[0]}_{col[1]}' for col in pivot_timing.columns]
pivot_pha.columns = [f'PHA_LG_{col}' for col in pivot_pha.columns]

# Combine the pivoted data
result = pd.concat([pivot_timing, pivot_pha], axis=1)

# Get the Other_Ch_Id from PHA data (assuming it's the non-16 channel)
other_ch_data = pha_data[pha_data['CH_Id'] != 16].groupby('Index')['CH_Id'].first()
result['Other_Ch_Id'] = other_ch_data

# Remove rows with missing data for channels 17 and 18 (key channels from your example)
result = result.dropna(subset=['ToA_ns_17', 'ToA_ns_18', 'ToT_ns_17', 'ToT_ns_18', 
                            'PHA_LG_17', 'PHA_LG_18', 'Other_Ch_Id'])

print(f"Aligned data shape: {result.shape}")
print(f"Available channels in aligned data: {[col for col in result.columns if 'PHA_LG' in col]}")

# Extract data for analysis using the same structure as your example
ch17_tot = result['ToT_ns_17'].values
ch18_tot = result['ToT_ns_18'].values

ch17_toa = result['ToA_ns_17'].values
ch18_toa = result['ToA_ns_18'].values

pos = result['Other_Ch_Id'].values

ch17 = result['PHA_LG_17'].values
ch18 = result['PHA_LG_18'].values

# Calculate the features as in your example
log_amp_ratio = np.log(ch17/ch18)
amp_diff_ratio = (ch17-ch18)/(ch17+ch18)

# Calculate ToA difference
toa_diff = ch17_toa - ch18_toa

# TOT as Time
tot_diff = (ch17_tot - ch18_tot)

# Convert position to physical distance
pos = pos.astype(float)
for i in range(len(pos)):
    pos[i] = 0.5 + pos[i] * 1.6

# print(f"\nData summary:")
# print(f"Number of positions: {len(pos)}")
# print(f"Position range: {pos.min():.2f} to {pos.max():.2f}")
# print(f"ToA difference range: {toa_diff.min():.2f} to {toa_diff.max():.2f}")
# print(f"ToT difference range: {tot_diff.min():.2f} to {tot_diff.max():.2f}")
# print(f"Log amplitude ratio range: {log_amp_ratio.min():.2f} to {log_amp_ratio.max():.2f}")

# Perform multiple linear regression using the matrix approach
X = np.column_stack([np.ones(len(toa_diff)), toa_diff, tot_diff, log_amp_ratio, amp_diff_ratio])
#X = np.column_stack([np.ones(len(toa_diff)), tot_diff, log_amp_ratio])
#X = np.column_stack([np.ones(len(toa_diff)), log_amp_ratio, amp_diff_ratio])


# Calculate beta coefficients using pseudo-inverse
beta = np.linalg.pinv(X) @ pos

# print(f"\n=== Multiple Linear Regression Results ===")
# print(f"Intercept (beta0): {beta[0]:.4f}")
# print(f"ToA difference coefficient (beta1): {beta[1]:.4f}")
# print(f"ToT difference coefficient (beta2): {beta[2]:.4f}")
# print(f"Log amplitude ratio coefficient (beta3): {beta[3]:.4f}")
# print(f"Amplitude difference ratio coefficient (beta4): {beta[4]:.4f}")

# Calculate predicted positions
pos_pred = X @ beta

# Calculate R-squared
ss_res = np.sum((pos - pos_pred) ** 2)
ss_tot = np.sum((pos - np.mean(pos)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.4f}")

# Calculate residuals
residuals = pos - pos_pred
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE: {rmse:.4f}")

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted positions
plt.subplot(1, 3, 1)
plt.scatter(pos, pos_pred, alpha=0.6)
plt.plot([pos.min(), pos.max()], [pos.min(), pos.max()], 'r--', lw=2)
plt.xlabel('Actual Position')
plt.ylabel('Predicted Position')
plt.title(f'Actual vs Predicted Position\n(R² = {r_squared:.4f})')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(pos_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Position')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.grid(True, alpha=0.3)

# Plot 3: Feature importance (coefficients magnitude)
plt.subplot(1, 3, 3)
features = ['Intercept', 'ToA Diff', 'ToT Diff', 'Log Amp Ratio', 'Amp Diff Ratio']
#features = ['Intercept', 'Log Amp Ratio', 'Amp Diff Ratio']
coefficients = np.abs(beta[1:])  # Exclude intercept for relative importance
plt.bar(features[1:], coefficients)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance (Coefficient Magnitude)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# # Additional analysis: Check correlation between features
# feature_matrix = np.column_stack([toa_diff, tot_diff, log_amp_ratio])
# correlation_matrix = np.corrcoef(feature_matrix.T)

# print(f"\n=== Feature Correlation Matrix ===")
# feature_names = ['ToT Diff', 'Log Amp Ratio']
# #feature_names = ['ToA Diff', 'ToT Diff', 'Log Amp Ratio']
# print("          " + "   ".join(f"{name:>12}" for name in feature_names))
# for i, row in enumerate(correlation_matrix):
#     print(f"{feature_names[i]:>10} " + " ".join(f"{val:12.4f}" for val in row))

# # Using scikit-learn for comparison
# from sklearn.linear_model import LinearRegression

# X_sklearn = np.column_stack([toa_diff, tot_diff, log_amp_ratio])
# model = LinearRegression()
# model.fit(X_sklearn, pos)
# pos_pred_sklearn = model.predict(X_sklearn)
# r2_sklearn = r2_score(pos, pos_pred_sklearn)

# print(f"\n=== Scikit-learn Comparison ===")
# print(f"R-squared: {r2_sklearn:.4f}")
# print(f"Coefficients: {model.coef_}")
# print(f"Intercept: {model.intercept_:.4f}")

# # Print some example predictions
# print(f"\n=== Example Predictions ===")
# print("Index | Actual Pos | Predicted Pos | Residual")
# print("-" * 50)
# for i in range(min(10, len(pos))):
#     print(f"{i:5} | {pos[i]:10.4f} | {pos_pred[i]:13.4f} | {residuals[i]:8.4f}")