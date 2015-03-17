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
		- user: user_id
		'''
		self.info = info
		self.user = user
		self.female = 0
		self.form = 'unknown'
		self.sample = user
		self.w_prob = []
		self.p = []
		self.timestamp = None

	def _process_info(self):
		'''
		Process info dataframe and initialize exercise statistics
		''' 
		self.info['file'] = self.info['timestamp'].copy()
		self.info['timestamp'] = pd.to_datetime(self.info['timestamp'], format='%Y-%m-%d_%H-%M-%S')
		cond = self.info['user_id'] == self.user
		self.info = self.info[cond]
		self.info.reset_index(inplace=True)

	def batch_process_user_samples(self):
		self._process_info()
		numeric_features = ['motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ',
		            'motionRotationRateX', 'motionRotationRateY', 'motionRotationRateZ',
		            'accelerometerAccelerationX','accelerometerAccelerationY','accelerometerAccelerationZ',
		            'gyroRotationX','gyroRotationY','gyroRotationZ','motionYaw','motionRoll','motionPitch',
		            'motionQuaternionX','motionQuaternionY','motionQuaternionZ','motionQuaternionW']

		# Initiate rep_history list
		rep_bin_history = []

		# Iterate through all the samples in info
		for i in xrange(self.info.shape[0]):

			# Initialize time series lists
			pitch_ts = []
			accY_ts = []
			accZ_ts = []
			quatY_ts = []

			df = pd.read_csv('../data/sensor/'+ self.info.loc[i,'file'] + '.csv')
			self.sample = 'user_'+str(self.user) + '_' + str(self.info.loc[i,'file'])
			self.female = self.info.loc[i,'female']
			freq = float(self.info.loc[i,'hertz'])
			height = self.info.loc[i,'height (inches)']
			self.timestamp = self.info.loc[i, 'timestamp']
			df_num = df[numeric_features]
			exercise = self.info.loc[i, 'exercise']

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
			cond2 = (np.abs(correls.ix[:, 'motionQuaternionY', 'motionQuaternionW']) +
			         np.abs(correls.ix[:, 'motionQuaternionX', 'motionQuaternionY']) +
			         np.abs(correls.ix[:, 'motionQuaternionY', 'motionQuaternionZ']) +
			         np.abs(correls.ix[:, 'gyroRotationX', 'gyroRotationZ'])) > 2.5

			# Trim the noisy first 2.5 seconds and last 1 seconds of sample record
			cond3 = df_filt.index > int(freq * 2.5) 
			cond4 = df_filt.index < (df_filt.shape[0] - int(freq)) 

			# Initial pushup repetition window
			pushup_window = df_filt[cond1 & cond2 & cond3 & cond4].index

			## Count the number of pushup repetitions ##
			# Calculate initial peak parameters using filtered data
			mph = 0.18 # minimum peak height
			mpd = (0.5 * freq)  # minimum peak separation distance
			feature = 'motionPitch'
			peakind, count = dp.count_peaks_initial(df_filt, pushup_window, feature, mph, mpd, freq)
			avg_dur = dp.average_duration(peakind, count)
			avg_amp_initial = dp.average_amplitude_initial(df_filt, peakind, pushup_window, feature, freq)

			# Final tight pushup repetition window
			window_ind = dp.calculate_total_rep_window(peakind, pushup_window, avg_dur, freq)

			# Calculate final peak parameters using unfiltered data
			# min peaks (middle of the rep when you reach lowest press-down)
			mph = avg_amp_initial # minimum peak height
			mpd = min((avg_dur * freq - 0.45 * freq), 1.5 * freq) # minimum peak separation distance
			peakmin, count_min, pushup_data = dp.count_peak_min(df_num, window_ind, feature, mph, mpd, freq, valley=True)
			print count_min
			# max peaks (start and end of rep)
			mph = -0.8 # minimum peak height
			mpd = min((avg_dur * freq - 0.45 * freq), 1.5 * freq) # minimum peak separation distance
			peakmax, count_max, pushup_data = dp.count_peak_max(df_num, count_min, window_ind, feature, mph, mpd, freq, valley=False)
			print count_max

			# add repetition metrics to list
			avg_metrics = dp.avg_rep_metrics(df_num, peakmin, peakmax, window_ind, feature, freq, self.female, height, self.form)
			sample_metrics = dp.rep_metrics(df_num, peakmin, peakmax, window_ind, feature, freq, self.female, height)
			sample_metrics = np.array(sample_metrics)

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
			pred_svm, prob_svm = c.predict('../models/svm_C10.0_g1.0_all.pkl', X)
			self.p, pred_tsP, prob_tsP = c.predict_ts('../models/dtw_kNNpitch_n4_w10_all.pkl', pitch_ts, component='pitch', avg_length=34)
			y, pred_tsY, prob_tsY = c.predict_ts('../models/dtw_kNNaccY_n4_w10_all.pkl', accY_ts, component='accY', avg_length=34)
			z, pred_tsZ, prob_tsZ  = c.predict_ts('../models/dtw_kNNaccZ_n4_w10_all.pkl', accZ_ts, component='accZ', avg_length=34)
			ensemble_arr = np.array([pred_rf, pred_svm, pred_tsP, pred_tsY, pred_tsZ])
			ensemble_prob_arr = np.array([prob_rf[:,1], prob_svm[:,1], prob_tsP, prob_tsY, prob_tsZ])

			# use weighted ensemble probability model 
			weights = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
			self.w_prob = np.dot(weights,ensemble_prob_arr)
			w_prob_true = (self.w_prob > 0.5) * 1.0
			print 'ensemble probability:', self.w_prob
			print 'ensemble prediction:', w_prob_true
			good = self.w_prob[(self.w_prob > 0.5)]
			ok = self.w_prob[(self.w_prob <= 0.5)]
			overall = len(good) > len(ok)
			if overall == True:
				self.form = 'good'
			else:
				self.form = 'ok'

			# Determine appropriate tip message
			if avg_metrics[4] > 0.1:
				tip =  "You're doing " + self.form + ". Next time try to have more consistent press-downs depths."
			elif avg_metrics[5] > 0.2:
				tip =  "You're doing " + self.form + ". Next time try to keep an even pace throughout your set."
			elif avg_metrics[2] < 1.0:
				tip = "You're doing " + self.form + ". Next time try to go lower."
			elif (avg_metrics[2] > 0.75) and (exercise == 'Kpushup'):
				tip = "You're doing " + self.form + ". Try to switch to regular pushups next time."
			elif (avg_metrics[2] > 1.0) and (avg_metrics[2] < 1.4):
				tip = "You're doing " + self.form + ". Next time try pressing down even lower."
			elif avg_metrics[2] > 1.4:
				tip = "Great form! Next time add more reps or try a different pushup stance."

			print tip

			# append rep ratings to rep history list
			rep_bin_history.append(([len(ok),len(good)], self.timestamp))

		# make plotly figures of lastest reps
		daily_url = pg.daily_reps(self.timestamp.hour, self.w_prob, self.sample)
		ts_url = pg.plot_ts(self.p, self.sample, freq=20.0)
		bar_url = pg.reps_bar_chart(self.w_prob, self.sample)

		# make plotly aggregate monthly figure
		monthly_url = pg.monthly_reps(rep_bin_history, self.user)

		return tip, daily_url, ts_url, bar_url, monthly_url
