#ch17 closest to you, ch 1 furthest from you


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def process_pha_data(input_file='processed_validated_positions_PHA.csv'):
    # Read the processed data
    df = pd.read_csv(input_file)
    
    # Group by index and extract required channels
    processed_data = []
    
    for index_num, group in df.groupby('Index'):
        # Get PHA_LG values for channels 17 and 18
        pha_17 = group[group['CH_Id'] == 17]['PHA_LG'].values
        pha_18 = group[group['CH_Id'] == 18]['PHA_LG'].values
        
        # Get the other channel (not 16, 17, or 18)
        other_channels = group[(group['CH_Id'] != 16) & 
                             (group['CH_Id'] != 17) & 
                             (group['CH_Id'] != 18)]
        
        if len(pha_17) == 1 and len(pha_18) == 1 and len(other_channels) == 1:
            other_ch_id = other_channels['CH_Id'].iloc[0]
            other_pha = other_channels['PHA_LG'].iloc[0]
            
            processed_data.append({
                'Index': index_num,
                'Other_Ch_Id': other_ch_id,
                'Other_PHA_LG': other_pha,
                'PHA_LG_17': pha_17[0],
                'PHA_LG_18': pha_18[0]
            })
    
    result_df = pd.DataFrame(processed_data)
    return result_df

# Process the data
result = process_pha_data()

# Display results
print("Processed PHA Data:")
print("=" * 60)
print(f"Total events processed: {len(result)}")
print(f"Other channels found: {sorted(result['Other_Ch_Id'].unique())}")


# Method 1: Direct column access

ch17 = result['PHA_LG_17'].values
ch18 = result['PHA_LG_18'].values
pos = result['Other_Ch_Id'].values


log_amp_ratio = np.log(ch17/ch18)
# log_amp_ratio = (ch17 - ch18)/(ch17 + ch18)


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


slope, intercept, r_value, p_value, std_err = stats.linregress(log_amp_ratio, pos)

y_pred = slope * log_amp_ratio + intercept
residuals = pos - y_pred
std_residuals = np.std(residuals)
r_value = r_value**2

print(f"Standard deviation of LOG residuals: {std_residuals:.3f}")
print(f"Standard error of LOG slope estimate: {std_err:.3f}")


# # Residuals plot - very important for checking model assumptions
# plt.figure(figsize=(12, 4))

# Plot each position with its corresponding color
for position in positions:
    mask = result['Other_Ch_Id'] == position
    plt.scatter(log_amp_ratio[mask], pos[mask], 
                color=pos_to_color[position], 
                s=60, alpha=1, 
                edgecolors='black', linewidth=0.5)



plt.plot(log_amp_ratio, y_pred, 'r-', linewidth=2, label='Regression')
# Add upper and lower bounds as dashed lines
plt.plot(log_amp_ratio, y_pred + std_residuals, 'pink', linewidth=2, label='±1 std dev')
plt.plot(log_amp_ratio, y_pred - std_residuals, 'pink', linewidth=2)



# # Add stats as text box
# stats_text = f"""Regression Statistics:
# y = {slope:.3f}x + {intercept:.3f}
# R^2= {r_value:.3f}
# Std Dev Error = {std_residuals:.3f}
# n = {len(log_amp_ratio)}"""

# plt.text(0.6, 0.32, stats_text, transform=plt.gca().transAxes, 
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#          fontfamily='monospace')

plt.xlabel('Amplitude Log Ratio: ln(a1/a2)', fontsize =12)
plt.ylabel('Positon of Perpendicular Bar', fontsize =12)
plt.ylim(-2, 27)

# Add stats as a custom legend entry
stats_text = f"n = {len(result)}"
plt.plot([], [], ' ', label=stats_text)  # Empty plot with text as label



plt.title('Amplitude Detected vs Position of Perpendicular Bars', fontsize =15)
plt.legend()
plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# plt.plot(residuals, alpha=0.7)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Data point')
# plt.ylabel('Prediction Error [L]')
# plt.title('Prediction Error')
# plt.legend
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
plt.show()





log_amp_ratio = (ch17 - ch18)/(ch17 + ch18)


slope, intercept, r_value, p_value, std_err = stats.linregress(log_amp_ratio, pos)

y_pred = slope * log_amp_ratio + intercept
residuals = pos - y_pred
std_residuals = np.std(residuals)
r_value = r_value**2

print(f"Standard deviation of AMP DIFF residuals: {std_residuals:.3f}")
print(f"Standard error of slope AMP DIFF estimate: {std_err:.3f}")


# Residuals plot - very important for checking model assumptions
# plt.figure(figsize=(12, 4))

# Plot each position with its corresponding color
for position in positions:
    mask = result['Other_Ch_Id'] == position
    plt.scatter(log_amp_ratio[mask], pos[mask], 
                color=pos_to_color[position], 
                s=60, alpha=1, 
                edgecolors='black', linewidth=0.5)


plt.plot(log_amp_ratio, y_pred, 'r-', linewidth=2, label='Regression')
# Add upper and lower bounds as dashed lines
plt.plot(log_amp_ratio, y_pred + std_residuals, 'pink', linewidth=2, label='±1 std dev')
plt.plot(log_amp_ratio, y_pred - std_residuals, 'pink', linewidth=2)


# # Add stats as text box
# stats_text = f"""Regression Statistics:
# y = {slope:.3f}x + {intercept:.3f}
# R^2= {r_value:.3f}
# Std Dev Error = {std_residuals:.3f}
# n = {len(log_amp_ratio)}"""

# plt.text(0.6, 0.32, stats_text, transform=plt.gca().transAxes, 
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#          fontfamily='monospace')

plt.xlabel('Amplitude Log Ratio: (a1 - a2)/(a1 + a2)', fontsize =12)
plt.ylabel('Positon of Perpendicular Bar', fontsize =12)
plt.ylim(-2, 27)

# Add stats as a custom legend entry
stats_text = f"n = {len(result)}"
plt.plot([], [], ' ', label=stats_text)  # Empty plot with text as label



plt.title('Amplitude Detected vs Position of Perpendicular Bars', fontsize =15)
plt.legend()
plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# plt.plot(residuals, alpha=0.7)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('Data point')
# plt.ylabel('Prediction Error [L]')
# plt.title('Prediction Error')
# plt.legend
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
plt.show()
