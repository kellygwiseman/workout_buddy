from process_data import ProcessData
from classify import ClassifyRep
import pandas as pd
import numpy as np

if __name__ == '__main__':
	# process samples
	#info = pd.read_csv('../data/pushup_info.csv', skipinitialspace=True)
	#p = ProcessData(info,'normal',plot=False)
	#data_arr, ts_arr = p.batch_process_samples()

	# select features to include in to training model
	data_arr = np.load('../processed/pushup_avg_metrics_normal.npy')
	ts_arr = np.load('../processed/pushup_raw_ts_one_normal.npy')
	X = data_arr[:,[2,3]].astype(float) # just the amplitude and duration
	labels = data_arr[:,-1]
	labels[labels =='excellent'] = 1
	labels[labels == 'good'] = 1
	labels[labels == 'ok'] = 0
	labels = labels.astype(int)

	# train classifier
	c = ClassifyRep(X, labels)
	sss = c.split_data(n_iter=5, test_size = 0.3, random_state=100)
	c.random_forest(sss, stance='normal', n_est=50, max_feat=2, max_depth=2, pickle=True)
	c.support_vector_machine(sss, stance='normal', C=10, gamma=1.0, pickle=True)
	ts = ts_arr[0]
	c.dtw_kNN(sss, ts, component='pitch', stance='normal', avg_length=34, n_neighbors=4, max_warping_window=10, pickle=True)
	ts = ts_arr[1]
	c.dtw_kNN(sss, ts, component='accY', stance = 'normal', avg_length=34, n_neighbors=4, max_warping_window=10, pickle=True)
	ts = ts_arr[2]
	c.dtw_kNN(sss, ts, component='accZ', stance = 'normal', avg_length=34, n_neighbors=4, max_warping_window=10, pickle=True)
	weights = np.array([0.33, 0.34, 0, 0.33, 0.0])
	c.ensemble(sss, weights)