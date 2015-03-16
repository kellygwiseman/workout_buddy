from process_data import ProcessData
from classify import ClassifyRep
import pandas as pd
import numpy as np
import plotly_graphs as pg

if __name__ == '__main__':
	# process one sample
	info = pd.read_csv('../data/test_sample.csv', skipinitialspace=True)
	p = ProcessData(info,'all',plot=True)
	rep_arr, ts_arr, timestamp, sample = p.process_one_sample()

	# select features to include in prediction model
	X = rep_arr[:,[2,3]].astype(float) # just the amplitude and duration

	# classify sequence of pushup repetitions
	c = ClassifyRep()
	pred_rf, prob_rf = c.predict('../models/rf_n50_mf2_md2_all.pkl', X)
	pred_svm, prob_svm = c.predict('../models/svm_C10_g1.0_all.pkl', X)
	p, pred_tsP, prob_tsP = c.predict_ts('../models/dtw_kNNpitch_n4_w10_all.pkl', ts_arr[0,:], component='pitch', avg_length=34)
	y, pred_tsY, prob_tsY = c.predict_ts('../models/dtw_kNNaccY_n4_w10_all.pkl', ts_arr[1,:], component='accY', avg_length=34)
	z, pred_tsZ, prob_tsZ  = c.predict_ts('../models/dtw_kNNaccZ_n4_w10_all.pkl', ts_arr[2,:], component='accZ', avg_length=34)
	ensemble_arr = np.array([pred_rf, pred_svm, pred_tsP, pred_tsY, pred_tsZ])
	ensemble_prob_arr = np.array([prob_rf[:,1], prob_svm[:,1], prob_tsP, prob_tsY, prob_tsZ])

	# use weighted ensemble probability model 
	weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
	w_prob = np.dot(weights,ensemble_prob_arr)
	w_prob_true = (w_prob > 0.5)* 1.0
	print 'ensemble probability:', w_prob
	print 'ensemble prediction:', w_prob_true
	pg.daily_reps(timestamp[0].hour, w_prob, sample)
	pg.plot_ts(p, sample, freq=20.0)
