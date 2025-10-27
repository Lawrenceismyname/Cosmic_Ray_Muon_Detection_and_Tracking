import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


file_path = 'and-16mV-big_ALL.csv'

#PERHAPS APPLY MOVING AVERAGE TO CAPTURE FALLING EDGE IF SIGNAL NOISY
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    # Create result array with zeros at the beginning
    result = np.zeros_like(a, dtype=float)
    result[n-1:] = ret[n-1:] / n
    
    return result



def analyze_fastframe_data(file_path):
    # Read the CSV file, handling the metadata properly
    df = pd.read_csv(file_path, skiprows=10)  # Skip metadata rows
    
    # Extract metadata from header
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Find FastFrame Count in the metadata
    fastframe_count = 0  # Default
    
    for i, line in enumerate(lines):
        if 'FastFrame Count' in line:
            parts = line.strip().split(',')
            for part in parts:
                if part.strip().isdigit():
                    fastframe_count = int(part.strip())
                    break
            break
    
    print(f"Detected {fastframe_count} frames")
    
    # Calculate frames
    total_rows = len(df)
    rows_per_frame = total_rows // fastframe_count
    
    print(f"Total rows: {total_rows}, Rows per frame: {rows_per_frame}")
    
    # Initialize lists for output CSV
    frame_numbers = []
    ch3_max_amps = []
    ch3_10p_times = []
    ch3_time_over_50p = []
    ch3_integrals = []
    ch4_max_amps = []
    ch4_10p_times = []
    ch4_time_over_50p = []
    ch4_integrals = []

    # NOISY FRAMES
    range1 = list(range(101, 108))     
    range2 = list(range(2540, 2554))
    range3 = list(range(2850, 2859))
    range4 = list(range(2860, 2871))


    noisy_frames = range1 + range2 + range3 + range4
    # print(noisy_frames)
    
    # Store data for plotting validation
    validation_frame_data = None
    validation_frame_number = 300
    validation_ch3_data = None
    validation_ch4_data = None
    
    # Main processing loop
    for frame in range(fastframe_count):
        
        #For noisy Frames
        skip_frame = False

        # Skip the entire frame if noisy HARD CODED
        for i in noisy_frames:
            if frame + 1 == i:
                print(f"Frame {frame + 1}: Signal noisy, skipping entire frame")
                skip_frame = True
                break # Move to next frame
        
        # Skip the entire frame if any channel has issues
        if skip_frame:
            continue  # Move to next frame
        
        # Extract data for current frame
        start_idx = frame * rows_per_frame
        end_idx = (frame + 1) * rows_per_frame
        frame_data = df.iloc[start_idx:end_idx].copy()
        
        # Reset index for this frame
        frame_data = frame_data.reset_index(drop=True)
        
        skip_frame = False
        # Check both channels for signal quality BEFORE processing
        for channel in ['CH3', 'CH4']:
            amplitude = frame_data[channel].values
            
            # Check for infinite values (clipped signal)
            if np.any(np.isinf(amplitude)):
                print(f"Frame {frame + 1}, {channel}: Signal clipped (infinite values detected), skipping entire frame")
                skip_frame = True
                break  # No need to check the other channel
            
            # Check for NaN values as well
            if np.any(np.isnan(amplitude)):
                print(f"Frame {frame + 1}, {channel}: NaN values detected, skipping entire frame")
                skip_frame = True
                break  # No need to check the other channel
        
        # Skip the entire frame if any channel has issues
        if skip_frame:
            continue  # Move to next frame
        
        # Initialize variables for this frame
        ch3_max_amp = None
        ch3_10p_time = None
        ch3_time_over_50p_amp = None
        ch3_integral = None
        ch4_max_amp = None
        ch4_10p_time = None
        ch4_time_over_50p_amp = None
        ch4_integral = None
        
        # Process both channels for valid frames
        for channel in ['CH3', 'CH4']:
            time = frame_data['TIME'].values
            amplitude = frame_data[channel].values

            amplitude = moving_average(amplitude, 10)
            
            # Find 50% amplitude time
            max_amp = np.max(amplitude)
            min_amp = np.min(amplitude)
            amplitude_range = max_amp - min_amp
            
            # Calculate integral of the signal (trapezoidal rule)
            signal_integral = np.trapezoid(amplitude, time)
            
            if amplitude_range > 0:  # Only calculate if there's a meaningful range
                ten_percent_level = min_amp + 0.1 * amplitude_range
                fifty_percent_level = min_amp + 0.5 * amplitude_range
                
                # Find when signal crosses 50% level (rising edge)
                ten_percent_time = None
                fifty_percent_time = None
                rising_edge_idx = None
                
                for i in range(1, len(amplitude)):
                    if (amplitude[i-1] < fifty_percent_level and 
                        amplitude[i] >= fifty_percent_level):
                        t1, t2 = time[i-1], time[i]
                        a1, a2 = amplitude[i-1], amplitude[i]
                        if a2 != a1:  # Avoid division by zero
                            fifty_percent_time = (t1 + (fifty_percent_level - a1) * (t2 - t1) / (a2 - a1)) - time[0]
                        else:
                            fifty_percent_time = t1 - time[0]
                        rising_edge_idx = i
                        
                        # Find 10% level before the 50% crossing (going backwards)
                        for j in range(350):
                            if i-j-1 >= 0:  # Ensure we don't go out of bounds
                                if (amplitude[i-j-1] < ten_percent_level and 
                                    amplitude[i-j] >= ten_percent_level):
                                    t1, t2 = time[i-j-1], time[i-j]
                                    a1, a2 = amplitude[i-j-1], amplitude[i-j]
                                    if a2 != a1:  # Avoid division by zero
                                        ten_percent_time = (t1 + (ten_percent_level - a1) * (t2 - t1) / (a2 - a1)) - time[0]
                                    else:
                                        ten_percent_time = t1 - time[0]
                                    break
                        break
                
                # Find when signal crosses back below 50% level (falling edge)
                falling_edge_time = None
                if rising_edge_idx is not None:
                    # Start searching 20 values after the rising edge to avoid noise near the edge
                    start_search_idx = rising_edge_idx + 20
                    
                    # Make sure we have enough data for the search
                    if start_search_idx < len(amplitude):
                        for i in range(start_search_idx, len(amplitude)):
                            if (amplitude[i-1] >= fifty_percent_level and 
                                amplitude[i] < fifty_percent_level):
                                # Found falling edge - use linear interpolation between current and previous point
                                t1, t2 = time[i-1], time[i]
                                a1, a2 = amplitude[i-1], amplitude[i]
                                if a2 != a1:  # Avoid division by zero
                                    falling_edge_time = (t1 + (fifty_percent_level - a1) * (t2 - t1) / (a2 - a1)) - time[0]
                                else:
                                    falling_edge_time = t1 - time[0]
                                break
                
                # Calculate time over 50% amplitude
                if fifty_percent_time is not None and falling_edge_time is not None:
                    time_over_50p = falling_edge_time - fifty_percent_time
                else:
                    time_over_50p = None
                    
            else:
                ten_percent_level = min_amp
                fifty_percent_level = min_amp
                ten_percent_time = None
                fifty_percent_time = None
                time_over_50p = None
            
            # Store values for output CSV
            if channel == 'CH3':
                # process into mV
                ch3_max_amp = max_amp*1000
                # process into ns
                ch3_10p_time = ten_percent_time*(1000000000) if ten_percent_time is not None else None
                # process into ns
                ch3_time_over_50p_amp = time_over_50p*(1000000000) if time_over_50p is not None else None
                # Store integral (V*s)
                ch3_integral = signal_integral
                
            else:  # CH4
                # process into mV
                ch4_max_amp = max_amp*1000
                # process into ns
                ch4_10p_time = ten_percent_time*(1000000000) if ten_percent_time is not None else None
                # process into ns
                ch4_time_over_50p_amp = time_over_50p*(1000000000) if time_over_50p is not None else None
                # Store integral (V*s)
                ch4_integral = signal_integral
        
        # Store data for validation plot (use first valid frame)
        if frame + 1 == validation_frame_data:
            validation_frame_data = frame_data.copy()
            validation_ch3_data = {
                'time': time,
                'amplitude': frame_data['CH3'].values,
                'max_amp': ch3_max_amp,
                'ten_percent_time': ch3_10p_time,
                'time_over_50p': ch3_time_over_50p_amp,
                'integral': ch3_integral
            }
            validation_ch4_data = {
                'time': time,
                'amplitude': frame_data['CH4'].values,
                'max_amp': ch4_max_amp,
                'ten_percent_time': ch4_10p_time,
                'time_over_50p': ch4_time_over_50p_amp,
                'integral': ch4_integral
            }
        
        # Append to output lists
        frame_numbers.append(frame + 1)
        ch3_max_amps.append(ch3_max_amp)
        ch3_10p_times.append(ch3_10p_time)
        ch3_time_over_50p.append(ch3_time_over_50p_amp)
        ch3_integrals.append(ch3_integral)
        ch4_max_amps.append(ch4_max_amp)
        ch4_10p_times.append(ch4_10p_time)
        ch4_time_over_50p.append(ch4_time_over_50p_amp)
        ch4_integrals.append(ch4_integral)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Frame': frame_numbers,
        'Ch3_max_amp[mV]': ch3_max_amps,
        'Ch3_10%_amp_time[ns]': ch3_10p_times,
        'Ch3_time_over_50%_amp[ns]': ch3_time_over_50p,
        'Ch3_integral[V*s]': ch3_integrals,
        'Ch4_max_amp[mV]': ch4_max_amps,
        'Ch4_10%_amp_time[ns]': ch4_10p_times,
        'Ch4_time_over_50%_amp[ns]': ch4_time_over_50p,
        'Ch4_integral[V*s]': ch4_integrals
    })
    
    # Create output filename
    base_name = os.path.splitext(file_path)[0]
    output_filename = f"{base_name}_processed.csv"
    
    # Save output CSV
    output_df.to_csv(output_filename, index=False)
    print(f"Output saved to: {output_filename}")
    
    # Plot validation figure if we have data
    if validation_frame_data is not None:
        plot_validation_figure(validation_frame_number, validation_ch3_data, validation_ch4_data)
    
    return output_df

def plot_validation_figure(frame_number, ch3_data, ch4_data):
    """Plot one frame with calculations annotated for validation"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    time_ns = ch3_data['time'] * 1e9  # Convert to nanoseconds
    ch3_amplitude_mv = ch3_data['amplitude'] * 1000  # Convert to mV
    ch4_amplitude_mv = ch4_data['amplitude'] * 1000  # Convert to mV
    
    ch3_amplitude_mv = moving_average(ch3_amplitude_mv, 20)
    ch4_amplitude_mv = moving_average(ch4_amplitude_mv, 20)

    # Plot CH3
    ax1.plot(time_ns, ch3_amplitude_mv, 'b-', linewidth=1.5, label='CH3 Signal')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title(f'Frame {frame_number} - CH3 Validation')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for CH3
    if ch3_data['ten_percent_time'] is not None:
        ax1.axvline(x=ch3_data['ten_percent_time'], color='r', linestyle='--', 
                   label=f'10% Time: {ch3_data["ten_percent_time"]:.1f} ns')
        
        # Calculate and plot 50% level
        min_amp = np.min(ch3_amplitude_mv)
        max_amp = np.max(ch3_amplitude_mv)
        ten_percent_level = min_amp + 0.5 * (max_amp - min_amp)
        ax1.axhline(y=ten_percent_level, color='g', linestyle='--', 
                   label=f'10% Level: {ten_percent_level:.1f} mV')
    
    ax1.legend()
    
    # Plot CH4
    ax2.plot(time_ns, ch4_amplitude_mv, 'g-', linewidth=1.5, label='CH4 Signal')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_title(f'Frame {frame_number} - CH4 Validation')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for CH4
    if ch4_data['ten_percent_time'] is not None:
        ax2.axvline(x=ch4_data['ten_percent_time'], color='r', linestyle='--', 
                   label=f'50% Time: {ch4_data["ten_percent_time"]:.1f} ns')
        
        # Calculate and plot 50% level
        min_amp = np.min(ch4_amplitude_mv)
        max_amp = np.max(ch4_amplitude_mv)
        ten_percent_level = min_amp + 0.5 * (max_amp - min_amp)
        ax2.axhline(y=ten_percent_level, color='g', linestyle='--', 
                   label=f'50% Level: {ten_percent_level:.1f} mV')
    
    ax2.legend()
    
#     # Add text box with calculated values
#     text_str = f"""Calculated Values:
# CH3:
#   Max Amp: {ch3_data['max_amp']:.1f} mV
#   10% Time: {ch3_data['ten_percent_time']:.1f} ns
#   Time over 50%: {ch3_data['time_over_50p']:.1f} ns
#   Integral: {ch3_data['integral']:.2e} V·s

# CH4:
#   Max Amp: {ch4_data['max_amp']:.1f} mV
#   10% Time: {ch4_data['ten_percent_time']:.1f} ns
#   Time over 50%: {ch4_data['time_over_50p']:.1f} ns
#   Integral: {ch4_data['integral']:.2e} V·s"""
    
#     plt.figtext(0.02, 0.02, text_str, fontfamily='monospace', fontsize=10, 
#                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'validation_frame_{frame_number}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the analysis
output_df = analyze_fastframe_data(file_path)