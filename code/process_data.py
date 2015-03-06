import pandas as pd
import numpy as np
import graphs as gr
from filter import filter_features
import detect_peaks as dp

class ProcessData(object):
	def __init__(self, info, pushup_type='all', plot=False):
		'''
		INPUT:
		- info: dataframe with user and sample data
		- pushup_type: 'all', '10_count', normal', 'narrow'
		- plot: create plots of different stages of processed data

		'''
		self.info = info
		self.pushup_type = pushup_type
		self.plot = plot

	def _process_info(self):
		'''
		Process info dataframe and initialize exercise statistics
		''' 

		self.info['file'] = self.info['timestamp'].copy()
		self.info['timestamp'] = pd.to_datetime(self.info['timestamp'],format='%Y-%m-%d_%H-%M-%S')
		if self.pushup_type == '10_count':
			cond = self.info['count'] == 10
			self.info = self.info[cond]
			self.info.reset_index(inplace=True)
		if self.pushup_type == 'normal':
			cond = self.info['stance'] == 'normal'
			self.info = self.info[cond]
			self.info.reset_index(inplace=True)
		if self.pushup_type == 'narrow':
			cond = self.info['stance'] == 'narrow'
			self.info = self.info[cond]
			self.info.reset_index(inplace=True)
		self.info.drop(['phone','comments'], axis=1, inplace=True)
		self.info['Pcount'] = 0
		self.info['avg_amp'] = 0.0
		self.info['avg_dur'] = 0.0

	def batch_process_samples(self):
		self._process_info()
		numeric_features = ['motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ',
                    'motionRotationRateX', 'motionRotationRateY', 'motionRotationRateZ',
                    'accelerometerAccelerationX','accelerometerAccelerationY','accelerometerAccelerationZ',
                    'gyroRotationX','gyroRotationY','gyroRotationZ','motionYaw','motionRoll','motionPitch',
                    'motionQuaternionX','motionQuaternionY','motionQuaternionZ','motionQuaternionW']
		# Initialize metrics lists
		rep_metrics_list = []
		avg_metrics_list = []

		# Iterate through all the samples in info
		for i in xrange(self.info.shape[0]):
		    df = pd.read_csv('../data/sensor/'+ self.info.loc[i,'file'] + '.csv')
		    sample = self.info.loc[i,'name'] + '_' + self.info.loc[i,'stance']
		    female = self.info.loc[i,'female']
		    freq = float(self.info.loc[i,'hertz'])
		    height = self.info.loc[i,'height (inches)']
		    form = self.info.loc[i,'form']
		    df_num = df[numeric_features]
		    
		    # Bandpass filter the data to separate the noise from the pushup signal
		    lowcut = 0.5
		    highcut = 2.0
		    order = 1
		    df_filt = filter_features(df_num, numeric_features, lowcut, highcut, freq, order)

		    # Calculate feature correlations
		    rolling_window = 30
		    correls = pd.rolling_corr(df_filt, rolling_window)

		    # Filter based on strength of feature correlations
		    cond1 = np.abs(correls.ix[:, 'accelerometerAccelerationX', 'accelerometerAccelerationY']) > 0.6
		    cond2 = (np.abs(correls.ix[:, 'motionQuaternionY', 'motionQuaternionW'])+
		             np.abs(correls.ix[:, 'motionQuaternionX', 'motionQuaternionY'])+
		             np.abs(correls.ix[:, 'motionQuaternionY', 'motionQuaternionZ'])+
		             np.abs(correls.ix[:, 'gyroRotationX', 'gyroRotationZ'])) > 2.5

		    # Trim the noisy first 2.5 seconds and last 1 seconds of sample record
		    cond3 = df_filt.index > int(freq*2.5) 
		    cond4 = df_filt.index < (df_filt.shape[0] - int(freq)) 

		    # Initial pushup repetition window
		    pushup_window = df_filt[cond1 & cond2 & cond3 & cond4].index

		    ## Count the number of pushup repetitions ##
		    # Peak parameters
		    mph = 0.2 - (0.025 * female) # minimum peak height
		    mpd = (0.5 * freq) + (0.25 * freq * female) # minimum peak separation distance
		    feature = 'motionPitch'
		    peakind, count = dp.count_peaks(df_filt, pushup_window, feature, mph, mpd, freq)
		    avg_amp = dp.average_amplitude(df_filt, peakind, pushup_window, feature, freq)
		    avg_dur = dp.average_duration(peakind, count)

		    # Final tight pushup repetition window
		    window_ind = dp.calculate_rep_window(peakind, pushup_window, avg_dur, freq)

		    # add repetition metrics to list
		    sample_metrics = dp.rep_metrics(df_filt, peakind, pushup_window, feature, freq, female, height, form)
		    rep_metrics_list.append(sample_metrics)
		    avg_metrics = dp.avg_rep_metrics(df_filt, peakind, pushup_window, feature, freq, female, height, form)
		    avg_metrics_list.append(avg_metrics)

		    # add results to dataframe
		    self.info.loc[i, 'Pcount'] = count
		    self.info.loc[i, 'avg_amp'] = avg_amp
		    self.info.loc[i, 'avg_dur'] = avg_dur

		    if self.plot:
		    	# Plot raw data #
		    	gr.plot1_acceleration(df_num, freq, sample)
		    	gr.plot1_gyro(df_num, freq, sample)
		    	gr.plot1_motion(df_num, freq, sample)
		    	gr.plot1_quaternion(df_num, freq, sample)
		    	# Plot the filtered data
		    	gr.plot_bandpass(df_num, df_filt, freq, lowcut, highcut, sample)
		    	# Plot the feature correlations
		    	gr.plot_corr(df_filt, correls, freq, sample)
		    	# Plot pushup repetitions
		    	gr.plot_pushups(df_filt, pushup_window, window_ind, peakind, feature, freq, sample)

	    ## Write processed data to files ##
	    # write processed data to csv file
		self.info.drop('file',axis=1, inplace=True)
		self.info.to_csv('../processed/processed_pushup_'+self.pushup_type+'.csv')

		# write rep_metrics to file
		rep_data = []
		for i in xrange(len(rep_metrics_list)):
		    for j in rep_metrics_list[i]:
		        rep_data.append(j)
		data_arr = np.array(rep_data)
		np.save('../processed/pushup_rep_metrics_'+self.pushup_type+'.npy', data_arr)

		# write avg_metrics to file
		data_arr = np.array(avg_metrics_list)
		np.save('../processed/pushup_avg_metrics_'+self.pushup_type+'.npy', data_arr)

	def _process_one_sample(self, file):
		pass