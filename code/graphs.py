
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import brewer2mpl
from matplotlib import rcParams
from filter import *

# Set defaults for matplotlib
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors
Idx = [0,1,2,3,6,4,5]
dark2_colors = [dark2_colors[i] for i in Idx]
pair_colors = brewer2mpl.get_map('Paired', 'Qualitative', 8).mpl_colors
rcParams['figure.figsize'] = (8, 4)
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

def plot_bandpass(data, fs, lowcut, highcut, order, sample):
    rcParams['axes.color_cycle'] = pair_colors
    time = data.index.values / fs
    df_filt = data.copy()
    X = butter_bandpass_filter(data.accelerometerAccelerationX, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['accelerometerAccelerationX'] = X
    Y = butter_bandpass_filter(data.accelerometerAccelerationY, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['accelerometerAccelerationY'] = Y
    Z = butter_bandpass_filter(data.accelerometerAccelerationZ, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['accelerometerAccelerationZ'] = Z
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(time, data.accelerometerAccelerationX, label='X')
    plt.plot(time, X, lw=2, label='filtered X')
    plt.plot(time, data.accelerometerAccelerationY, label='Y')
    plt.plot(time, Y, lw=2, label='filtered Y')
    plt.plot(time, data.accelerometerAccelerationZ, label='Z')
    plt.plot(time, Z, lw=2, label='filtered Z')
    plt.title('Acceleration bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Meter / Second^2 (in G)')
    plt.ylim(-2.0, 1.5)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/pu_bp_acc-'+sample+'.png');

    fig2 = plt.figure(figsize=(10,6))
    X = butter_bandpass_filter(data.gyroRotationX, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['gyroRotationX'] = X
    Y = butter_bandpass_filter(data.gyroRotationY, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['gyroRotationY'] = Y
    Z = butter_bandpass_filter(data.gyroRotationZ, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['gyroRotationZ'] = Z
    fig = plt.figure(figsize=(10,6))
    plt.plot(time, data.gyroRotationX, label='X')
    plt.plot(time, X, lw=2, label='filtered X')
    plt.plot(time, data.gyroRotationY, label='Y')
    plt.plot(time, Y, lw=2, label='filtered Y')
    plt.plot(time, data.gyroRotationZ, label='Z')
    plt.plot(time, Z, lw=2, label='filtered Z')
    plt.title('Gyro bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radian / Second')
    plt.ylim(-5.0, 5.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/pu_bp_gyr-'+sample+'.png');

    fig3 = plt.figure(figsize=(10,6))
    X = butter_bandpass_filter(data.motionPitch, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionPitch'] = X
    Y = butter_bandpass_filter(data.motionRoll, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionRoll'] = Y
    Z = butter_bandpass_filter(data.motionYaw, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionYaw'] = Z
    fig = plt.figure(figsize=(10,6))
    plt.plot(time, data.motionPitch, label='Pitch')
    plt.plot(time, X, lw=2, label='filtered Pitch')
    plt.plot(time, data.motionRoll, label='Roll')
    plt.plot(time, Y, lw=2, label='filtered Roll')
    plt.plot(time, data.motionYaw, label='Yaw')
    plt.plot(time, Z, lw=2, label='filtered Yaw')
    plt.title('Attitude Motion bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radians')
    plt.ylim(-2.5,3.0)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/pu_bp_att-'+sample+'.png');

    fig4 = plt.figure(figsize=(10,6))
    X = butter_bandpass_filter(data.motionQuaternionX, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionQuaternionX'] = X
    Y = butter_bandpass_filter(data.motionQuaternionY, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionQuaternionY'] = Y
    Z = butter_bandpass_filter(data.motionQuaternionZ, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionQuaternionZ'] = Z
    W = butter_bandpass_filter(data.motionQuaternionW, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    df_filt['motionQuaternionW'] = W
    fig = plt.figure(figsize=(10,6))
    plt.plot(time, data.motionQuaternionX, label='X')
    plt.plot(time, X, lw=2, label='filtered X')
    plt.plot(time, data.motionQuaternionY, label='Y')
    plt.plot(time, Y, lw=2, label='filtered Y')
    plt.plot(time, data.motionQuaternionZ, label='Z')
    plt.plot(time, Z, lw=2, label='filtered Z')
    plt.plot(time, data.motionQuaternionW, label='W')
    plt.plot(time, W, lw=2, label='filtered W')
    plt.title('Quaternions bandpass filtered between %g and %g Hz' %(lowcut, highcut))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Radians')
    plt.ylim(-0.75, 1.0)
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.05)
    plt.savefig('../figures/pu_bp_quat-'+sample+'.png');
    return df_filt

def plot_pushups(data, pushup_window, widths, max_dist, gap_thresh, freq, sample):
    time = data.index.values / freq
    pushup_start = pushup_window.index[0] / freq
    pushup_time = pushup_window.index.values / freq
    
    # calculate push-up reps
    peakind = signal.find_peaks_cwt(pushup_window['motionPitch'], widths = widths,
                                    max_distances = max_dist, gap_thresh = gap_thresh)
    peakind = [x / freq for x in peakind]
    
    # Plot complete time series and push-up duration overlay
    fig1 = plt.figure()
    plt.plot(time, data.motionPitch, label='Raw data')
    plt.plot(pushup_time, pushup_window['motionPitch'], label='Push-up duration')
    
    # Plot push-ups
    plt.scatter(pushup_start + peakind,np.linspace(0.0,0.0,num=len(peakind)), color='r',marker='x', lw=2, label='Push-ups')
    plt.title('Push-up Repetitions')
    plt.ylabel('Pitch (Radians)')
    plt.xlabel('Time (Seconds)')
    plt.xlim(0,time[-1])
    ymin, ymax = plt.ylim()
    #plt.ylim((ymin-0.2), ymax)
    plt.ylim(-0.75, 1.75)
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1, borderpad=0.1)
    plt.savefig('../figures/pu_reps-'+sample+'.png')
    return peakind;