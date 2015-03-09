from process_data import ProcessData
from classify import ClassifyRep
import pandas as pd
import numpy as np

if __name__ == '__main__':
	# process one sample
	info = pd.read_csv('../data/test_sample.csv', skipinitialspace=True)
	p = ProcessData(info,'all',plot=False)
	data_arr, ts_arr = p.process_one_sample()

	# select features to include in prediction model
	X = data_arr[:,[2,3]].astype(float) # just the amplitude and duration
	labels = data_arr[:,-1]
	labels[labels =='excellent'] = 1
	labels[labels == 'good'] = 1
	labels[labels == 'ok'] = 0
	labels = labels.astype(int)

	# classify sequence of pushup repetitions
	c = ClassifyRep(X, labels)
	pred_rf = c.predict('../models/rf_n50_mf2_md2_normal.pkl', X)
	pred_svm = c.predict('../models/svm_C10_g1.0_normal.pkl', X)
	pred_tsP = c.predict_ts('../models/dtw_kNNpitch_n4_w10_normal.pkl', ts_arr[0], component='pitch', avg_length=34)
	pred_tsY = c.predict_ts('../models/dtw_kNNaccY_n4_w10_normal.pkl', ts_arr[0], component='accY', avg_length=34)
	pred_tsZ = c.predict_ts('../models/dtw_kNNaccZ_n4_w10_normal.pkl', ts_arr[0], component='accZ', avg_length=34)
	ensemble_arr = np.array([pred_rf, pred_svm, pred_tsP, pred_tsY, pred_tsZ])
	print 'RF:', pred_rf
	print 'SVM:', pred_svm
	print 'ts pitch:', pred_tsP
	print 'ts accY:', pred_tsY
	print 'ts accZ:', pred_tsZ
	print ''
	weights = np.array([0.33, 0.34, 0.0, 0.33, 0.0])
	w_pred = np.dot(weights,ensemble_arr) 
	w_pred = (w_pred > 0.5)* 1.0
	print 'weighted ensemble:', w_pred
