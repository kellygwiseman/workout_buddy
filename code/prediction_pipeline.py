from process_data import ProcessData
from classify import ClassifyRep
import graphs as gr
import pandas as pd
import numpy as np
import plotly_graphs as pg
from scipy import signal

if __name__ == '__main__':
	# process one sample
	info = pd.read_csv('../data/test_sample.csv', skipinitialspace=True)
	p = ProcessData(info,'all',plot=False)
	data_arr, ts_arr, timestamp, sample = p.process_one_sample()

	# optimal pitch
	example_ts = np.load('../processed/pushup_raw_ts_one_all_70h.npy')[0,34]
	B = [signal.resample(example_ts, 34)]
	# initialize rep to 0
	B = np.array([xi - xi[0] for xi in B])

	# select features to include in prediction model
	X = data_arr[:,[2,3]].astype(float) # just the amplitude and duration
	labels = data_arr[:,-1]
	labels[labels =='excellent'] = 1
	labels[labels == 'good'] = 1
	labels[labels == 'ok'] = 0
	labels = labels.astype(int)

	# classify sequence of pushup repetitions
	c = ClassifyRep(X, labels)
	pred_rf, prob_rf = c.predict('../models/rf_n50_mf2_md2_all.pkl', X)
	pred_svm, prob_svm = c.predict('../models/svm_C10_g1.0_all.pkl', X)
	p, pred_tsP, prob_tsP = c.predict_ts('../models/dtw_kNNpitch_n4_w10_all.pkl', ts_arr[0,:], component='pitch', avg_length=34)
	y, pred_tsY, prob_tsY = c.predict_ts('../models/dtw_kNNaccY_n4_w10_all.pkl', ts_arr[1,:], component='accY', avg_length=34)
	z, pred_tsZ, prob_tsZ  = c.predict_ts('../models/dtw_kNNaccZ_n4_w10_all.pkl', ts_arr[2,:], component='accZ', avg_length=34)
	ensemble_arr = np.array([pred_rf, pred_svm, pred_tsP, pred_tsY, pred_tsZ])
	ensemble_prob_arr = np.array([prob_rf[:,1], prob_svm[:,1], prob_tsP, prob_tsY, prob_tsZ])

	#print out predictions and probabilities for each model
	#print 'RF:', pred_rf
	#print 'RF prob:', prob_rf
	#print 'SVM:', pred_svm
	#print 'SVM prob:', prob_svm
	#print 'ts pitch:', pred_tsP
	#print 'ts pitch prob:', prob_tsP
	#print 'ts accY:', pred_tsY
	#print 'ts accY prob:', prob_tsY
	#print 'ts accZ:', pred_tsZ
	#print 'ts accZ prob:', prob_tsZ
	#print ''

	# use weighted ensemble probability model 
	weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
	w_prob = np.dot(weights,ensemble_prob_arr)
	w_prob_true = (w_prob > 0.5)* 1.0
	print 'ensemble probability:', w_prob
	print 'ensemble prediction:', w_prob_true
	pg.daily_reps(timestamp[0].hour, w_prob, sample)
	pg.plot_ts(B, p, sample, avg_length=34, freq=20.0)
