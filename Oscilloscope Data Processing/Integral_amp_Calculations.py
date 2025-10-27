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


ch3_int = df['Ch3_integral[V*s]'].values
ch4_int = df['Ch4_integral[V*s]'].values



# Read the CSV file
df = pd.read_csv('or-cal_ALL_processed.csv')

# Extract amplitude data from each channel
ch3_amps_or = df['Ch3_max_amp[mV]'].values
ch4_amps_or = df['Ch4_max_amp[mV]'].values

ch3_int_or = df['Ch3_integral[V*s]'].values
ch4_int_or = df['Ch4_integral[V*s]'].values


ch3_amps = np.concatenate([ch3_amps, ch3_amps_or])
ch4_amps = np.concatenate([ch4_amps, ch4_amps_or])

ch3_int = np.concatenate([ch3_int, ch3_int_or])
ch4_int = np.concatenate([ch4_int, ch4_int_or])

print(len(ch3_amps))

valid_mask = (ch3_amps > 15) & (ch4_amps > 15) & (ch3_int > 0) & (ch4_int > 0)

ch3_amps = ch3_amps[valid_mask]
ch4_amps = ch4_amps[valid_mask]

amp_mean_ratio = np.mean(ch3_amps)/np.mean(ch4_amps)

print(amp_mean_ratio)

ch3_int = ch3_int[valid_mask]
ch4_int = ch4_int[valid_mask]

print(len(ch3_amps))


#AMPLITUDE
ratios = (ch3_amps - ch4_amps)/(ch3_amps + ch4_amps)

#normalize
ratios = ratios - np.mean(ratios)

ratios = ratios/np.sqrt(np.cov(ratios))

###int
int_ratios = np.log( np.sqrt(ch3_int / ch4_int))

int_ratios = int_ratios - np.mean(int_ratios)

int_ratios = int_ratios/np.sqrt(np.cov(int_ratios))


# # mask = (ratios>-2) & (ratios<2)

# ratios = ratios[mask]
# int_ratios = int_ratios[mask]

print(len(ratios))


plt.subplot(1,2,1)
plt.hist(ratios, 20)
plt.title("3. Amplitude Difference Ratio")
plt.xlabel("(a1 - a2)/(a1 + a2)")
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist(int_ratios, 20)
plt.title("4. Integral Method")
plt.xlabel("ln( sqrt( q1 / q2 ) )")
plt.ylabel('Count')


plt.tight_layout()
plt.show()
