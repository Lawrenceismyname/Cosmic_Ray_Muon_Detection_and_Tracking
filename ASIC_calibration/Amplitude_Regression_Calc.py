import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Read the data
df = pd.read_csv('Run7_list.csv', skiprows=12)

# Clean up column names by stripping any extra whitespace
df.columns = df.columns.str.strip()

# Get the start and end timestamps ITS IN MICROSECONDS
start_timestamp = df['TStamp'].iloc[0]
end_timestamp = df['TStamp'].iloc[-1]

print(f"\nStart timestamp: {start_timestamp}")
print(f"End timestamp: {end_timestamp}")

# Calculate the difference
time_difference = (end_timestamp - start_timestamp)/1e6
print(f"Time Difference [s]: {time_difference}")


# print("Cleaned column names:")
# print(df.columns.tolist())

# Create arrays for each channel
ch0_data = df[df['CH_Id'] == 0][['PHA_LG', 'ToT_ns', 'ToA_ns']].values
ch1_data = df[df['CH_Id'] == 1][['PHA_LG', 'ToT_ns', 'ToA_ns']].values

# Or as separate arrays
ch0_amps = df[df['CH_Id'] == 0]['PHA_LG'].values
ch0_tot = df[df['CH_Id'] == 0]['ToT_ns'].values
ch0_toa = df[df['CH_Id'] == 0]['ToA_ns'].values

ch1_amps = df[df['CH_Id'] == 1]['PHA_LG'].values
ch1_tot = df[df['CH_Id'] == 1]['ToT_ns'].values
ch1_toa = df[df['CH_Id'] == 1]['ToA_ns'].values

# Calculate ratios = Amp_ch3 / Amp_ch4
# ratios = (ch0_amps - ch1_amps) / (ch0_amps + ch1_amps)

log_amp_ratios = np.log(ch0_amps / ch1_amps)
# ratios = (ch0_tot - ch1_tot)

print("Data points")
print(len(log_amp_ratios))

muons_per_minute = len(log_amp_ratios)/(time_difference/60)
print(f"Muons Detected Per Minute [mu/min]: {muons_per_minute}")


positions_amp = -7.875*log_amp_ratios

min_pos_amp = np.min(positions_amp)
max_pos_amp = np.max(positions_amp)
print(f"min position from amp = {min_pos_amp}")
print(f"max position from amp = {max_pos_amp}")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(log_amp_ratios, positions_amp)
plt.xlabel('Log Ratio ln(SiPM_1 Amplitude / SiPM_2 Amplitude)')
plt.ylabel('Position along bar [cm]')
plt.title('Amplitude log_ratios and Corresponding Predicted Positions')
plt.grid(True, alpha=0.3)
plt.show()