""" 
This library is used to make plots to help with the signal processing and 
repetition detection stage of modeling process.  They are intended to help 
with data analysis, and are not the visualizations used in the webapp. The 
figures are automatically saved in a figure directory and are not shown.
"""

import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import brewer2mpl
from matplotlib import rcParams

# Set defaults for matplotlib
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors
Idx = [0, 1, 2, 3, 6, 4, 5]
dark2_colors = [dark2_colors[i] for i in Idx]
pair_colors = brewer2mpl.get_map('Paired', 'Qualitative', 8).mpl_colors
RYG_colors = brewer2mpl.get_map('RdYlGn', 'Diverging', 10).mpl_colors
spectral_colors = brewer2mpl.get_map('Spectral', 'Diverging', 10).mpl_colors
rcParams['figure.figsize'] = (8, 5)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 1.5
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

def plot3_acceleration(data, freq):
    """Plot the raw acceleration data. Split the three components into different plots."""

    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationX)
    plt.title('X-component Acceleration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0, time[-1])
    plt.savefig('../figures//acceleration/pu_accelX-' + sample + '.png');
    plt.close(fig1)
    fig2 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationY)
    plt.title('Y-component Acceleration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0, time[-1])
    plt.savefig('../figures//acceleration/pu_accelY-' + sample + '.png');
    plt.close(fig2)
    fig3 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationZ)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0, time[-1])
    plt.title('Z-component Acceleration');
    plt.savefig('../figures/acceleration/pu_accelZ-' + sample + '.png');
    plt.close(fig3)
    
def plot1_acceleration(data, freq, sample):
    """Plot the raw acceleration data."""

    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationX, label='X' )
    plt.plot(time, data.accelerometerAccelerationY, label='Y')
    plt.plot(time, data.accelerometerAccelerationZ, label='Z')
    plt.title('Acceleration')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0, time[-1])
    ymin, ymax = plt.ylim()
    ydiff = ymax - ymin
    plt.ylim(ymin - 0.1*ydiff, ymax)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures//acceleration/pu_accel-' + sample + '.png');
    plt.close(fig1)
    
def plot1_quaternion(data, freq, sample):
    """Plot the raw quaternion data."""

    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.motionQuaternionX, label='X' )
    plt.plot(time, data.motionQuaternionY, label='Y')
    plt.plot(time, data.motionQuaternionZ, label='Z')
    plt.plot(time, data.motionQuaternionW, label='W')
    plt.title('Quaternions')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Radians')
    plt.xlim(0,time[-1])
    ymin, ymax = plt.ylim()
    ydiff = ymax - ymin
    plt.ylim(ymin - 0.1*ydiff, ymax)
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/quaternion/pu_quat-' + sample + '.png');
    plt.close(fig1)

def plot1_gyro(data, freq, sample):
    """Plot the raw gyro rotation rate data."""

    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.gyroRotationX, label='X')
    plt.plot(time, data.gyroRotationY, label='Y')
    plt.plot(time, data.gyroRotationZ, label='Z')
    plt.title('Gyro Rotation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Radians / Second')
    plt.xlim(0,time[-1])
    ymin, ymax = plt.ylim()
    ydiff = ymax - ymin
    plt.ylim(ymin - 0.1*ydiff, ymax)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/gyro/pu_gyro-' + sample + '.png');
    plt.close(fig1)
    
def plot1_motion(data, freq, sample):
    """Plot the raw phone attitude data."""

    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.motionPitch, label='Pitch')
    plt.plot(time, data.motionRoll, label='Roll')
    plt.plot(time, data.motionYaw, label='Yaw')
    plt.title('Phone Attitude')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Radians')
    plt.xlim(0,time[-1])
    ymin, ymax = plt.ylim()
    ydiff = ymax - ymin
    plt.ylim(ymin - 0.1*ydiff, ymax)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/motion/pu_attitude-' + sample + '.png');
    plt.close(fig1)

def plot_corr(data, correls, freq, sample):
    """Plot the feature correlations for each data type. There will be a 
    separate plot for accelerartion, gyro rotation rate, phone attitude, and 
    quaternions."""

    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, correls.ix[:, 'accelerometerAccelerationX', 'accelerometerAccelerationY'], label='X-Y')
    plt.plot(time, correls.ix[:, 'accelerometerAccelerationX', 'accelerometerAccelerationZ'], label='X-Z')
    plt.plot(time, correls.ix[:, 'accelerometerAccelerationY', 'accelerometerAccelerationZ'], label='Y-Z')
    plt.title('Acceleration Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0, time[-1])
    plt.ylim(-1.5, 1.25)
    plt.yticks(np.arange(-1.0, 1.5, 0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/correlation/pu_corr_acc-' + sample + '.png');
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(time, correls.ix[:, 'gyroRotationX', 'gyroRotationY'], label='X-Y')
    plt.plot(time, correls.ix[:, 'gyroRotationX', 'gyroRotationZ'], label='X-Z')
    plt.plot(time, correls.ix[:, 'gyroRotationY', 'gyroRotationZ'], label='Y-Z')
    plt.title('Gyro Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0,time[-1])
    plt.ylim(-1.5, 1.25)
    plt.yticks(np.arange(-1.0, 1.5, 0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/correlation/pu_corr_gyro-' + sample + '.png');
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(time, correls.ix[:, 'motionPitch', 'motionRoll'], label='Pitch-Roll')
    plt.plot(time, correls.ix[:, 'motionPitch', 'motionYaw'], label='Pitch-Yaw')
    plt.plot(time, correls.ix[:, 'motionRoll', 'motionYaw'], label='Roll-Yaw')
    plt.title('Attitude Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0,time[-1])
    plt.ylim(-1.5, 1.25)
    plt.yticks(np.arange(-1.0, 1.5, 0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/correlation/pu_corr_att-' + sample + '.png');
    plt.close(fig3)

    fig4 = plt.figure()
    plt.plot(time, correls.ix[:, 'motionQuaternionX', 'motionQuaternionY'], label='X-Y')
    plt.plot(time, correls.ix[:, 'motionQuaternionX', 'motionQuaternionZ'], label='X-Z')
    plt.plot(time, correls.ix[:, 'motionQuaternionX', 'motionQuaternionW'], label='X-W')
    plt.plot(time, correls.ix[:, 'motionQuaternionY', 'motionQuaternionZ'], label='Y-Z')
    plt.plot(time, correls.ix[:, 'motionQuaternionY', 'motionQuaternionW'], label='Y-W')
    plt.plot(time, correls.ix[:, 'motionQuaternionZ', 'motionQuaternionW'], label='Z-W')
    plt.title('Quarternion Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0, time[-1])
    plt.ylim(-1.75, 1.25)
    plt.yticks(np.arange(-1.0, 1.5, 0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/correlation/pu_corr_quat-' + sample + '.png');
    plt.close(fig4)

def plot_bandpass(data, filtered_data, freq, lowcut, highcut, sample):
    """Plot the bandpass filtered data. They'll be a separate plot for 
    filtered accelerartion, gyro rotation rate, phone attitude, and quaternions."""

    rcParams['axes.color_cycle'] = pair_colors
    time = data.index.values / freq
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(time, data.accelerometerAccelerationX, label='X')
    plt.plot(time, filtered_data.accelerometerAccelerationX, lw=2, label='filtered X')
    plt.plot(time, data.accelerometerAccelerationY, label='Y')
    plt.plot(time, filtered_data.accelerometerAccelerationY, lw=2, label='filtered Y')
    plt.plot(time, data.accelerometerAccelerationZ, label='Z')
    plt.plot(time, filtered_data.accelerometerAccelerationZ, lw=2, label='filtered Z')
    plt.title('Acceleration bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.ylim(-2.0, 1.5)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/bp_filter/pu_bp_acc-' + sample + '.png');
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(time, data.gyroRotationX, label='X')
    plt.plot(time, filtered_data.gyroRotationX, lw=2, label='filtered X')
    plt.plot(time, data.gyroRotationY, label='Y')
    plt.plot(time, filtered_data.gyroRotationY, lw=2, label='filtered Y')
    plt.plot(time, data.gyroRotationZ, label='Z')
    plt.plot(time, filtered_data.gyroRotationZ, lw=2, label='filtered Z')
    plt.title('Gyro bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radian / Second')
    plt.ylim(-5.0, 5.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/bp_filter/pu_bp_gyr-' + sample + '.png');
    plt.close(fig2)

    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(time, data.motionPitch, label='Pitch')
    plt.plot(time, filtered_data.motionPitch, lw=2, label='filtered Pitch')
    plt.plot(time, data.motionRoll, label='Roll')
    plt.plot(time, filtered_data.motionRoll, lw=2, label='filtered Roll')
    plt.plot(time, data.motionYaw, label='Yaw')
    plt.plot(time, filtered_data.motionYaw, lw=2, label='filtered Yaw')
    plt.title('Attitude Motion bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radians')
    plt.ylim(-2.5, 3.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/bp_filter/pu_bp_att-' + sample + '.png');
    plt.close(fig3)

    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(time, data.motionQuaternionX, label='X')
    plt.plot(time, filtered_data.motionQuaternionX, lw=2, label='filtered X')
    plt.plot(time, data.motionQuaternionY, label='Y')
    plt.plot(time, filtered_data.motionQuaternionY, lw=2, label='filtered Y')
    plt.plot(time, data.motionQuaternionZ, label='Z')
    plt.plot(time, filtered_data.motionQuaternionZ, lw=2, label='filtered Z')
    plt.plot(time, data.motionQuaternionW, label='W')
    plt.plot(time, filtered_data.motionQuaternionW, lw=2, label='filtered W')
    plt.title('Quaternions bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radians')
    plt.ylim(-0.75, 1.0)
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/bp_filter/pu_bp_quat-' + sample + '.png');
    plt.close(fig4)

def plot_pushups(data, pushup_data, window_ind, peakmax, feature, freq, sample):
    """Plot the raw pitch time series overlain with the pushup duration window
    and markings for the individual repetitions."""

    rcParams['axes.color_cycle'] = dark2_colors
    
    # separate out pushup data and convert everything to seconds (from frequency)
    time = data.index.values / freq
    pushup_start = window_ind[0] / freq # for marking the peaks
    pushup_time = time[window_ind[0]:(window_ind[1]+1)] # for plotting the duration
    data_arr = data[feature].values
    data_lst = [data_arr[r]*180.0 / np.pi for r in xrange(len(data_arr))]
    pushup_arr = data.ix[window_ind[0]:window_ind[1]][feature].values
    pushup_lst = [pushup_arr[r]*180.0 / np.pi for r in xrange(len(pushup_arr))]
    plotting_amp = np.min(pushup_lst) - 5.0
    
    # Plot complete time series and push-up duration overlay
    fig1 = plt.figure()
    plt.plot(time, data_lst, label='Full time series')
    plt.plot(pushup_time, pushup_lst, label='Pushup duration')
    
    # Mark push-up repetitions
    plt.scatter([pushup_start + p for p in peakmax], np.linspace(plotting_amp, plotting_amp, num=len(peakmax)), color='r',marker='x', lw=2, label='Pushups')
    plt.title('Pushup Repetitions')
    plt.ylabel('Phone Pitch (degrees)')
    plt.xlabel('Time (Seconds)')
    plt.xlim(0, time[-1])
    ymin, ymax = plt.ylim()
    plt.ylim(-40.0, 100.0) # keep the ylim the same for all samples so you can easily compare amplitudes
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pushup_reps/pu_reps-' + sample + '.png');
    plt.close(fig1)