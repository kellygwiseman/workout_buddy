""" Plot the Exercise data """

import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
from matplotlib import rcParams

# Set defaults for matplotlib
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors
Idx = [0,1,2,3,6,4,5]
dark2_colors = [dark2_colors[i] for i in Idx]
pair_colors = brewer2mpl.get_map('Paired', 'Qualitative', 8).mpl_colors
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
    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationX)
    plt.title('X-component Acceleration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0,time[-1])
    plt.savefig('../figures/pu_accelX-'+sample+'.png');
    fig2 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationY)
    plt.title('Y-component Acceleration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0,time[-1])
    plt.savefig('../figures/pu_accelY-'+sample+'.png');
    fig3 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationZ)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0,time[-1])
    plt.title('Z-component Acceleration');
    plt.savefig('../figures/pu_accelZ-'+sample+'.png');
    
def plot1_acceleration(data, freq, sample):
    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.accelerometerAccelerationX, label='X' )
    plt.plot(time, data.accelerometerAccelerationY, label='Y')
    plt.plot(time, data.accelerometerAccelerationZ, label='Z')
    plt.title('Acceleration')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0,time[-1])
    plt.ylim(-1.5,1.5)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_accel-'+sample+'.png');
    
def plot1_quaternion(data, freq, sample):
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
    plt.ylim(-0.5, 1.0)
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_quat-'+sample+'.png');

def plot1_Uacceleration(data, freq, sample):
    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, data.motionUserAccelerationX, label='X' )
    plt.plot(time, data.motionUserAccelerationY, label='Y')
    plt.plot(time, data.motionUserAccelerationZ, label='Z')
    plt.title('User Acceleration')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.xlim(0,time[-1])
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_Uaccel-'+sample+'.png');

def plot1_gyro(data, freq, sample):
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
    plt.ylim(-5.0, 5.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_gyro-'+sample+'.png');
    
def plot1_motion(data, freq, sample):
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
    plt.ylim(-2.0,3.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_attitude-'+sample+'.png');

def plot_corr(data, correls, freq, sample):
    rcParams['axes.color_cycle'] = dark2_colors
    time = data.index.values / freq
    fig1 = plt.figure()
    plt.plot(time, correls.ix[:, 'accelerometerAccelerationX', 'accelerometerAccelerationY'], label='X-Y')
    plt.plot(time, correls.ix[:, 'accelerometerAccelerationX', 'accelerometerAccelerationZ'], label='X-Z')
    plt.plot(time, correls.ix[:, 'accelerometerAccelerationY', 'accelerometerAccelerationZ'], label='Y-Z')
    plt.title('Acceleration Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0,time[-1])
    plt.ylim(-1.5,1.25)
    plt.yticks(np.arange(-1.0,1.5,0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_corr_acc-'+sample+'.png');

    fig2 = plt.figure()
    plt.plot(time, correls.ix[:, 'gyroRotationX', 'gyroRotationY'], label='X-Y')
    plt.plot(time, correls.ix[:, 'gyroRotationX', 'gyroRotationZ'], label='X-Z')
    plt.plot(time, correls.ix[:, 'gyroRotationY', 'gyroRotationZ'], label='Y-Z')
    plt.title('Gyro Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0,time[-1])
    plt.ylim(-1.5,1.25)
    plt.yticks(np.arange(-1.0,1.5,0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_corr_gyro-'+sample+'.png');

    fig3 = plt.figure()
    plt.plot(time, correls.ix[:, 'motionPitch', 'motionRoll'], label='Pitch-Roll')
    plt.plot(time, correls.ix[:, 'motionPitch', 'motionYaw'], label='Pitch-Yaw')
    plt.plot(time, correls.ix[:, 'motionRoll', 'motionYaw'], label='Roll-Yaw')
    plt.title('Attitude Feature Correlation')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.xlim(0,time[-1])
    plt.ylim(-1.5,1.25)
    plt.yticks(np.arange(-1.0,1.5,0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_corr_att-'+sample+'.png');

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
    plt.xlim(0,time[-1])
    plt.ylim(-1.75,1.25)
    plt.yticks(np.arange(-1.0,1.5,0.5))
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/pu_corr_quat-'+sample+'.png');

def plot_bandpass(data, filtered_data, freq, lowcut, highcut, sample):
    rcParams['axes.color_cycle'] = pair_colors
    time = data.index.values / freq
    fig1 = plt.figure(figsize=(10,6))
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
    plt.savefig('../figures/pu_bp_acc-'+sample+'.png');

    fig2 = plt.figure(figsize=(10,6))
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
    plt.savefig('../figures/pu_bp_gyr-'+sample+'.png');

    fig3 = plt.figure(figsize=(10,6))
    plt.plot(time, data.motionPitch, label='Pitch')
    plt.plot(time, filtered_data.motionPitch, lw=2, label='filtered Pitch')
    plt.plot(time, data.motionRoll, label='Roll')
    plt.plot(time, filtered_data.motionRoll, lw=2, label='filtered Roll')
    plt.plot(time, data.motionYaw, label='Yaw')
    plt.plot(time, filtered_data.motionYaw, lw=2, label='filtered Yaw')
    plt.title('Attitude Motion bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radians')
    plt.ylim(-2.5,3.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/pu_bp_att-'+sample+'.png');

    fig4 = plt.figure(figsize=(10,6))
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
    plt.savefig('../figures/pu_bp_quat-'+sample+'.png');

def plot_pushups(data, pushup_window, peakind, freq, sample):
    # separate out pushup data and convert everything to seconds (from frequency)
    time = data.index.values / freq
    pushup_start = pushup_window[0] / freq
    pushup_time = time[pushup_window[0]:(pushup_window[-1]+1)]
    pushup_data = data.ix[pushup_window[0]:pushup_window[-1]]
    
    # Plot complete time series and push-up duration overlay
    fig1 = plt.figure()
    plt.plot(time, data.motionPitch, label='Raw data')
    plt.plot(pushup_time, pushup_data['motionPitch'], label='Push-up duration')
    
    # Mark push-up repetitions
    plt.scatter(pushup_start + peakind,np.linspace(0.0,0.0,num=len(peakind)), color='r',marker='x', lw=2, label='Push-ups')
    plt.title('Push-up Repetitions')
    plt.ylabel('BP Filtered Pitch (Radians)')
    plt.xlabel('Time (Seconds)')
    plt.xlim(0,time[-1])
    ymin, ymax = plt.ylim()
    plt.ylim(-1.0, 1.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_reps-f'+sample+'.png');