import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d, CubicSpline
from sympy import symbols, solve, Eq



# Read the CSV file
df = pd.read_csv('and-16mV-big_ALL_processed.csv')

# Extract amplitude data from each channel
ch3_amps = df['Ch3_max_amp[mV]'].values
ch4_amps = df['Ch4_max_amp[mV]'].values


ch3_time = df['Ch3_10%_amp_time[ns]'].values
ch4_time = df['Ch4_10%_amp_time[ns]'].values



# Read the CSV file
df = pd.read_csv('or-cal_ALL_processed.csv')

# Extract amplitude data from each channel
ch3_amps_or = df['Ch3_max_amp[mV]'].values
ch4_amps_or = df['Ch4_max_amp[mV]'].values

ch3_time_or = df['Ch3_10%_amp_time[ns]'].values
ch4_time_or = df['Ch4_10%_amp_time[ns]'].values


ch3_amps = np.concatenate([ch3_amps, ch3_amps_or])
ch4_amps = np.concatenate([ch4_amps, ch4_amps_or])

ch3_time = np.concatenate([ch3_time, ch3_time_or])
ch4_time = np.concatenate([ch4_time, ch4_time_or])

print(len(ch3_amps))

valid_mask = (ch3_amps > 15) & (ch4_amps > 15) & (ch3_time > 0) & (ch4_time > 0)

ch3_amps = ch3_amps[valid_mask]
ch4_amps = ch4_amps[valid_mask]

amp_mean_ratio = np.mean(ch3_amps)/np.mean(ch4_amps)

print(amp_mean_ratio)

ch3_time = ch3_time[valid_mask]
ch4_time = ch4_time[valid_mask]

print(len(ch3_amps))


#AMPLITUDE
ratios = np.log(ch3_amps / ch4_amps)

###TIME
t_diff = ch3_time - ch4_time

#normalize
mean_ratios = np.mean(ratios)
ratios = ratios - mean_ratios
print(f"Ratio mean: {mean_ratios}")

ratios = ratios/(np.sqrt(np.cov(ratios)))

mean_t_diff = np.mean(t_diff)
print(f"T Diff mean: {mean_t_diff}")
t_diff = t_diff - mean_t_diff

t_diff = t_diff/(np.sqrt(np.cov(t_diff)))


# mask = (ratios>-2.5) & (ratios<2.5) & (t_diff < 2.5) & (t_diff > -2.5)

# ratios = ratios[mask]
# t_diff = t_diff[mask]

print(len(ratios))




plt.subplot(1,2,1)
plt.hist(ratios, 20)
plt.title("1. Natural Log Amplitude Ratio")
plt.xlabel("ln( a1 / a2 )")
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist(t_diff, 20)
plt.title("2. Time Difference")
plt.xlabel("( t1 - t2 )")
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# plt.subplot(1,2,1)
# plt.hist(ratios, 250)
# plt.title("Natural Log Amplitude Ratio")
# plt.xlabel("ln( a1 / a2 )")

# #AMPLITUDE
# ratios = (ch3_amps - ch4_amps)/(ch3_amps + ch4_amps)

# plt.subplot(1,2,2)
# plt.hist(ratios, 250)
# plt.title("Amplitude Difference Ratio")
# plt.xlabel("(a1 - a2)/(a1 - a2)")
# plt.show()
