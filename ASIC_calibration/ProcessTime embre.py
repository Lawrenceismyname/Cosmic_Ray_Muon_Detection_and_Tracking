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
        toa_17 = group[group['CH_Id'] == 17]['ToA_ns'].values
        toa_18 = group[group['CH_Id'] == 18]['ToA_ns'].values
        
        # Get the other channel (not 16, 17, or 18)
        other_channels = group[(group['CH_Id'] != 16) & 
                             (group['CH_Id'] != 17) & 
                             (group['CH_Id'] != 18)]
        
        if len(toa_17) == 1 and len(toa_18) == 1 and len(other_channels) == 1:
            other_ch_id = other_channels['CH_Id'].iloc[0]
            
            processed_data.append({
                'Index': index_num,
                'Other_Ch_Id': other_ch_id,
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
ch17_toa = result['ToA_17'].values
ch18_toa = result['ToA_18'].values
pos = result['Other_Ch_Id'].values

# Calculate ToA difference
toa_diff = ch17_toa - ch18_toa

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

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(toa_diff, pos)

y_pred = slope * toa_diff + intercept
residuals = pos - y_pred
std_residuals = np.std(residuals)
r_squared = r_value**2

print(f"\nRegression Statistics:")
print(f"Standard deviation of residuals: {std_residuals:.3f}")
print(f"Standard error of slope estimate: {std_err:.3f}")
print(f"R-squared: {r_squared:.3f}")

# Create detailed plot with embre coloring
plt.figure(figsize=(12, 8))

# Plot each position with its corresponding color
for position in positions:
    mask = result['Other_Ch_Id'] == position
    plt.scatter(toa_diff[mask], pos[mask], 
                color=pos_to_color[position], 
                s=60, alpha=1, 
                edgecolors='black', linewidth=0.5)

# Plot regression line
plt.plot(toa_diff, y_pred, 'r-', linewidth=3, label='Regression Line', alpha=0.8)

# Add confidence intervals
plt.plot(toa_diff, y_pred + std_residuals, 'pink', linewidth=2, 
         label='±1 std dev')
plt.plot(toa_diff, y_pred - std_residuals, 'pink', linewidth=2)

# # Add stats as text box
# stats_text = f"""Regression Statistics:
# y = {slope:.3f}x + {intercept:.3f}
# R² = {r_squared:.3f}
# Std Dev Error = {std_residuals:.3f}
# n = {len(toa_diff)}"""

# plt.text(0.65, 0.15, stats_text, transform=plt.gca().transAxes, 
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
#          fontfamily='monospace', fontsize=10)

plt.xlabel('ToA Difference: ToA_1 - ToA_2 (ns)', fontsize =15)
plt.ylabel('Position of Perpendicular Bar', fontsize =15)
plt.ylim(-2, 27)

# Add stats as a custom legend entry
stats_text = f"n = {len(result)}"
plt.plot([], [], ' ', label=stats_text)  # Empty plot with text as label



plt.title('Time of Arrival (ToA) Difference vs Position', fontsize =20)
plt.legend(fontsize = 10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





# Print summary statistics
print(f"\nSummary Statistics for ToA Difference:")
print(f"Mean: {np.mean(toa_diff):.3f} ns")
print(f"Std Dev: {np.std(toa_diff):.3f} ns")
print(f"Min: {np.min(toa_diff):.3f} ns")
print(f"Max: {np.max(toa_diff):.3f} ns")
print(f"Range: {np.max(toa_diff) - np.min(toa_diff):.3f} ns")