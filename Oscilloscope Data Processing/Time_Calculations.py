import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d, CubicSpline
from sympy import symbols, solve, Eq



# Read the CSV file
df = pd.read_csv('and-16mV-big_ALL_processed.csv')

# Extract amplitude data from each channel
ch3_tocf = df['Ch3_time_over_50%_amp[ns]'].values
ch4_tocf = df['Ch4_time_over_50%_amp[ns]'].values


ch3_time = df['Ch3_10%_amp_time[ns]'].values
ch4_time = df['Ch4_10%_amp_time[ns]'].values



# Read the CSV file
df = pd.read_csv('or-cal_ALL_processed.csv')

# Extract amplitude data from each channel
ch3_tocf_or = df['Ch3_time_over_50%_amp[ns]'].values
ch4_tocf_or = df['Ch4_time_over_50%_amp[ns]'].values

ch3_time_or = df['Ch3_10%_amp_time[ns]'].values
ch4_time_or = df['Ch4_10%_amp_time[ns]'].values


ch3_tocf = np.concatenate([ch3_tocf, ch3_tocf_or])
ch4_tocf = np.concatenate([ch4_tocf, ch4_tocf_or])

ch3_time = np.concatenate([ch3_time, ch3_time_or])
ch4_time = np.concatenate([ch4_time, ch4_time_or])

print(len(ch3_tocf))

valid_mask = (ch3_tocf > 10) & (ch4_tocf > 10) & (ch3_time > 0) & (ch4_time > 0)

ch3_tocf = ch3_tocf[valid_mask]
ch4_tocf = ch4_tocf[valid_mask]

amp_mean_ratio = np.mean(ch3_tocf)/np.mean(ch4_tocf)

print(amp_mean_ratio)

ch3_time = ch3_time[valid_mask]
ch4_time = ch4_time[valid_mask]

print(len(ch3_tocf))


#TOCF
tocf_diff = ch3_tocf - ch4_tocf
tocf_log_sqrt_ratio = np.log( np.sqrt( ch3_tocf / ch4_tocf) )

###TIME
t_diff = ch3_time - ch4_time

#normalize
tocf_diff = tocf_diff - np.mean(tocf_diff)

t_diff = t_diff - np.mean(t_diff)


mask = (t_diff < 3) & (t_diff > -3) & (tocf_diff < 50) & (tocf_diff > -50) & (tocf_log_sqrt_ratio < 0.25) & (tocf_log_sqrt_ratio > -0.25)

tocf_diff = tocf_diff[mask]
tocf_log_sqrt_ratio = tocf_log_sqrt_ratio[mask]
t_diff = t_diff[mask]

print(len(tocf_diff))



# # mask = (tocf_log_sqrt_ratio < 0.25) & (tocf_log_sqrt_ratio > -0.25)
# tocf_log_sqrt_ratio = tocf_log_sqrt_ratio[mask]
print(len(tocf_log_sqrt_ratio))



plt.subplot(1,3,1)
plt.hist(t_diff, 250)
plt.title("Time Difference")
plt.xlabel("( t1 - t2 ) [ns]")



plt.subplot(1,3,2)
plt.hist(tocf_diff, 250)
plt.title("Time Over constant Fraction Difference")
plt.xlabel("( ToCF1 - ToCF2 ) [ns]")


plt.subplot(1,3,3)
plt.hist(tocf_log_sqrt_ratio, 250)
plt.title("Time Over constant Fraction ln(sqrt(ratio))")
plt.xlabel("ln( sqrt( ToCF1 / ToCF2) )")


plt.tight_layout()
plt.show()

