import pandas as pd
import numpy as np
import plotly_graphs as pg
from filter import filter_features
import detect_peaks as dp
from classify import ClassifyRep

class AnonPrediction(object):
	"""
	This class contains methods to process, classify, and visualize anonymous
	sensor data.
	"""
	def __init__(self, filename):
		'''
		INPUT:
		- filename: name of txt file containing all the raw pushup sensor data
		'''
		self.filename = filename

	def process_user_sample(self):
		numeric_features = ['accelerometerAccelerationX','accelerometerAccelerationY','accelerometerAccelerationZ',
		            'gyroRotationX','gyroRotationY','gyroRotationZ','motionYaw','motionRoll','motionPitch',
		            'motionQuaternionX','motionQuaternionY','motionQuaternionZ','motionQuaternionW']

		# Initiate rep_history list
		rep_bin_history = []

		# Initialize time series lists
		pitch_ts = []
		accY_ts = []
		accZ_ts = []
		quatY_ts = []

		df = pd.read_table(self.filename, sep=',', skipinitialspace=True)
		sample = 'anon_user'
		female = 0.0 # put in dummy sex, it's collected for the training data but currently not used in algorithms
		freq = 20.0
		height = 69.0 # put in dummy height, it's collected for the training data but currently not used in algorithms
		form = 'unknown'
		timestamp = pd.to_datetime(df.loc[1, 'loggingTime'], format='%Y-%m-%d %H:%M:%S.%f')
		df_num = df[numeric_features]

		# Bandpass filter the data to help separate the noise from the pushup signal
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

		## Calculate pushup repetition parameters ##

		# Calculate initial peak parameters using filtered data
		mph = 0.18 # minimum peak height
		mpd = (0.5 * freq) # minimum peak separation distance
		feature = 'motionPitch'
		peakind, count = dp.count_peaks_initial(df_filt, pushup_window, feature, mph, mpd, freq)
		avg_dur = dp.average_duration(peakind, count)
		avg_amp_initial = dp.average_amplitude_initial(df_filt, peakind, pushup_window, feature, freq)

		# Final tight pushup repetition window
		window_ind = dp.calculate_total_rep_window(peakind, pushup_window, avg_dur, freq)

		# Calculate final peak parameters using raw data
		# min peaks (middle of the rep when you reach lowest press-down)
		mph = avg_amp_initial # minimum peak height
		mpd = min((avg_dur*freq - 0.45*freq), 1.5*freq) # minimum peak separation distance
		peakmin, count_min, pushup_data = dp.count_peak_min(df_num, window_ind, feature, mph, mpd, freq, valley=True)
		# max peaks (start and end of rep)
		mph = -0.8 # minimum peak height
		mpd = min((avg_dur*freq - 0.45*freq), 1.5*freq) # minimum peak separation distance
		peakmax, count_max, pushup_data = dp.count_peak_max(df_num, count_min, window_ind, feature, mph, mpd, freq, valley=False)

		# Add repetition metrics to list
		avg_metrics = dp.avg_rep_metrics(df_num, peakmin, peakmax, window_ind, feature, freq, female, height, form)
		sample_metrics = dp.rep_metrics(df_num, peakmin, peakmax, window_ind, feature, freq, female, height)
		sample_metrics = np.array(sample_metrics)

		# Repetition windows
		multiple_rep_windows = dp.calculate_multiple_rep_window(peakmax, window_ind, freq)
		# Add repetition time series to feature lists
		for i in xrange(len(multiple_rep_windows)):
		    pushup_data = df_num.ix[multiple_rep_windows[i][0]:multiple_rep_windows[i][1]]
		    pitch_ts.append(pushup_data['motionPitch'].tolist())
		    accY_ts.append(pushup_data['accelerometerAccelerationY'].tolist())
		    accZ_ts.append(pushup_data['accelerometerAccelerationZ'].tolist())
		    quatY_ts.append(pushup_data['motionQuaternionY'].tolist())
		pitch_ts = np.array(pitch_ts)
		accY_ts = np.array(accY_ts)
		accZ_ts = np.array(accZ_ts)

		## Classify sequence of pushup repetitions ##
		X = sample_metrics[:,[2,3]]
		c = ClassifyRep()
		pred_rf, prob_rf = c.predict('../models/rf_n50_mf2_md2_all.pkl', X)
		pred_svm, prob_svm = c.predict('../models/svm_C10.0_g1.0_all.pkl', X)
		p, pred_tsP, prob_tsP = c.predict_ts('../models/dtw_kNNpitch_n4_w10_all.pkl', pitch_ts, component='pitch', avg_length=34)
		y, pred_tsY, prob_tsY = c.predict_ts('../models/dtw_kNNaccY_n4_w10_all.pkl', accY_ts, component='accY', avg_length=34)
		z, pred_tsZ, prob_tsZ  = c.predict_ts('../models/dtw_kNNaccZ_n4_w10_all.pkl', accZ_ts, component='accZ', avg_length=34)
		ensemble_prob_arr = np.array([prob_rf[:,1], prob_svm[:,1], prob_tsP, prob_tsY, prob_tsZ])

		# Use weighted ensemble probability model 
		weights = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
		w_prob = np.dot(weights,ensemble_prob_arr)
		good = w_prob[(w_prob > 0.5)]
		ok = w_prob[(w_prob <= 0.5)]
		overall = len(good) > len(ok)
		if overall == True:
			form = 'good'
		else:
			form = 'ok'

		# Determine appropriate tip message
		if avg_metrics[4] > 0.1:
			tip =  "You're doing "+form+". Next time try to have more consistent press-downs depths."
		elif avg_metrics[5] > 0.2:
			tip =  "You're doing "+form+". Next time try to keep an even pace throughout your set."
		elif avg_metrics[2] < 1.0:
			tip = "You're doing "+form+". Next time try to go lower."
		elif (avg_metrics[2] > 1.0) and (avg_metrics[2] < 1.4):
			tip = "You're doing "+form+". Next time try pressing down even lower."
		elif avg_metrics[2] > 1.4:
			tip = "Great form! Next time add more reps or try a different pushup stance."

		## Make Interactive plots for webapp ##
		# Make aggregate monthly figure
		rep_bin_history.append(([len(ok),len(good)], timestamp))
		monthly_url = pg.monthly_reps(rep_bin_history, sample+'_monthly')

		# Make figures of lastest set of reps
		ts_url = pg.plot_ts(p, sample, freq=20.0)
		bar_url = pg.reps_bar_chart(w_prob, sample)

		return tip, ts_url, bar_url, monthly_url
