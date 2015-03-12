import pandas as pd
import numpy as np
import plotly_graphs as pg
from filter import filter_features
import detect_peaks as dp
from classify import ClassifyRep

class UserPrediction(object):
	def __init__(self, info, user=6):
		'''
		INPUT:
		- info: dataframe with user and sample data
		- pushup_type: 'all', '10_count', normal', 'narrow'
		- plot: create plots of different stages of processed data

		'''
		self.info = info
		self.user = user

	def _process_info(self):
		'''
		Process info dataframe and initialize exercise statistics
		''' 

		self.info['file'] = self.info['timestamp'].copy()
		self.info['timestamp'] = pd.to_datetime(self.info['timestamp'],format='%Y-%m-%d_%H-%M-%S')
		cond = self.info['user_id'] == self.user
		self.info = self.info[cond]
		self.info.reset_index(inplace=True)
		self.info['form'] = 'unknown'
		self.info.drop(['phone','comments'], axis=1, inplace=True)
		self.info['Pcount'] = 0
		self.info['avg_amp'] = 0.0
		self.info['avg_dur'] = 0.0
		self.info['amp_std'] = 0.0
		self.info['dur_std'] = 0.0

	def batch_process_user_samples(self):
		self._process_info()
		numeric_features = ['motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ',
                    'motionRotationRateX', 'motionRotationRateY', 'motionRotationRateZ',
                    'accelerometerAccelerationX','accelerometerAccelerationY','accelerometerAccelerationZ',
                    'gyroRotationX','gyroRotationY','gyroRotationZ','motionYaw','motionRoll','motionPitch',
                    'motionQuaternionX','motionQuaternionY','motionQuaternionZ','motionQuaternionW']

		# Iterate through all the samples in info
		for i in xrange(self.info.shape[0]):
			# Initialize metrics lists
			rep_metrics_list = []

			# Initialize time series lists
			pitch_ts = []
			accY_ts = []
			accZ_ts = []
			quatY_ts = []

			df = pd.read_csv('../data/sensor/'+ self.info.loc[i,'file'] + '.csv')
			sample = 'user_'+str(self.user) + '_' + str(self.info.loc[i,'file'])
			female = self.info.loc[i,'female']
			freq = float(self.info.loc[i,'hertz'])
			height = self.info.loc[i,'height (inches)']
			timestamp = self.info.loc[i, 'timestamp']
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
			# Calculate initial peak parameters using filtered data
			mph = 0.2 - (0.025 * female) # minimum peak height
			mpd = (0.5 * freq) + (0.25 * freq * female) # minimum peak separation distance
			feature = 'motionPitch'
			peakind, count = dp.count_peaks_initial(df_filt, pushup_window, feature, mph, mpd, freq)
			avg_dur = dp.average_duration(peakind, count)
			avg_amp_initial = dp.average_amplitude_initial(df_filt, peakind, pushup_window, feature, freq)

			# Final tight pushup repetition window
			window_ind = dp.calculate_total_rep_window(peakind, pushup_window, avg_dur, freq)

			# Calculate final peak parameters using unfiltered data
			mph = avg_amp_initial # minimum peak height
			mpd = min((avg_dur*freq - 0.4*freq), 1.5*freq) # minimum peak separation distance
			peakmin, count_min, pushup_data = dp.count_peak_min(df_num, window_ind, feature, mph, mpd, freq, valley=True)
			print count_min
			mph = -0.8 # minimum peak height
			mpd = min((avg_dur*freq - 0.4*freq), 1.5*freq) # minimum peak separation distance
			peakmax, count_max, pushup_data = dp.count_peak_max(df_num, count_min, window_ind, feature, mph, mpd, freq, valley=False)
			print count_max

			# add repetition metrics to list
			avg_metrics = dp.avg_rep_metrics(df_num, peakmin, peakmax, window_ind, feature, freq, female, height, form)
			sample_metrics = dp.rep_metrics(df_num, peakmin, peakmax, window_ind, feature, freq, female, height)
			sample_metrics = np.array(sample_metrics)

			# add results to dataframe
			self.info.loc[i, 'Pcount'] = count_min
			self.info.loc[i, 'avg_amp'] = avg_metrics[2]
			self.info.loc[i, 'avg_dur'] = avg_metrics[3]
			self.info.loc[i, 'amp_std'] = avg_metrics[4]
			self.info.loc[i, 'dur_std'] = avg_metrics[5]

			# Repetition windows
			multiple_rep_windows = dp.calculate_multiple_rep_window(peakmax, window_ind, freq)
			# add repetition time series to feature lists
			for i in xrange(len(multiple_rep_windows)):
			    pushup_data = df_num.ix[multiple_rep_windows[i][0]:multiple_rep_windows[i][1]]
			    pitch_ts.append(pushup_data['motionPitch'].tolist())
			    accY_ts.append(pushup_data['accelerometerAccelerationY'].tolist())
			    accZ_ts.append(pushup_data['accelerometerAccelerationZ'].tolist())
			    quatY_ts.append(pushup_data['motionQuaternionY'].tolist())
			pitch_ts = np.array(pitch_ts)
			accY_ts = np.array(accY_ts)
			accZ_ts = np.array(accZ_ts)

			# classify sequence of pushup repetitions
			X = sample_metrics[:,[2,3]]
			c = ClassifyRep()
			pred_rf, prob_rf = c.predict('../models/rf_n50_mf2_md2_all.pkl', X)
			pred_svm, prob_svm = c.predict('../models/svm_C10_g1.0_all.pkl', X)
			p, pred_tsP, prob_tsP = c.predict_ts('../models/dtw_kNNpitch_n4_w10_all.pkl', pitch_ts, component='pitch', avg_length=34)
			y, pred_tsY, prob_tsY = c.predict_ts('../models/dtw_kNNaccY_n4_w10_all.pkl', accY_ts, component='accY', avg_length=34)
			z, pred_tsZ, prob_tsZ  = c.predict_ts('../models/dtw_kNNaccZ_n4_w10_all.pkl', accZ_ts, component='accZ', avg_length=34)
			ensemble_arr = np.array([pred_rf, pred_svm, pred_tsP, pred_tsY, pred_tsZ])
			ensemble_prob_arr = np.array([prob_rf[:,1], prob_svm[:,1], prob_tsP, prob_tsY, prob_tsZ])

			# use weighted ensemble probability model 
			weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
			w_prob = np.dot(weights,ensemble_prob_arr)
			w_prob_true = (w_prob > 0.5)* 1.0
			print 'ensemble probability:', w_prob
			print 'ensemble prediction:', w_prob_true

			pg.daily_reps(timestamp.hour, w_prob, sample)
			pg.plot_ts(p, sample, freq=20.0)
