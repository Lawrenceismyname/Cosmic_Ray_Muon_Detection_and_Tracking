import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('Run0_list.csv', skiprows=12)

# Clean up column names by stripping any extra whitespace
df.columns = df.columns.str.strip()

print("Cleaned column names:")
print(df.columns.tolist())

# Create arrays for each channel
ch0_data = df[df['CH_Id'] == 0][['PHA_LG', 'ToT_ns', 'ToA_ns']].values
ch1_data = df[df['CH_Id'] == 1][['PHA_LG', 'ToT_ns', 'ToA_ns']].values

# Or as separate arrays
ch0_pha = df[df['CH_Id'] == 0]['PHA_LG'].values
ch0_tot = df[df['CH_Id'] == 0]['ToT_ns'].values
ch0_toa = df[df['CH_Id'] == 0]['ToA_ns'].values

ch1_pha = df[df['CH_Id'] == 1]['PHA_LG'].values
ch1_tot = df[df['CH_Id'] == 1]['ToT_ns'].values
ch1_toa = df[df['CH_Id'] == 1]['ToA_ns'].values


