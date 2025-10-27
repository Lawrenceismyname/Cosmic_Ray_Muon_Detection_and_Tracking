#ch17 closest to you, ch 1 furthest from you


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

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


ratios = (ch17 - ch18)/(ch17 + ch18)


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
    popt, pcov = curve_fit(tan_func, ratios, pos, p0=p0)
    
    # Extract optimized parameters
    a, b, c, d = popt
    
    # Evaluate the tangent function at new points
    x_new = np.linspace(min(ratios), max(ratios), len(ratios))
    y_new = tan_func(x_new, a, b, c, d)
    
    # Calculate residuals and standard deviation
    residuals = pos - tan_func(ratios, a, b, c, d)
    std_residuals = np.std(residuals)
    
    print(f"Tangent function parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
    print(f"Standard deviation of residuals: {std_residuals:.3f}")
    
    # Plot original data and fitted curve
    # plt.scatter(ratios, pos, alpha=0.8, label='Data Scatter Plot')

    for position in positions:
        mask = result['Other_Ch_Id'] == position
        plt.scatter(ratios[mask], pos[mask], 
            color=pos_to_color[position], 
            s=60, alpha=1, 
            edgecolors='black', linewidth=0.5)
        

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



