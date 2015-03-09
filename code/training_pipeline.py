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
	data_arr = np.load('../processed/pushup_avg_metrics_all_70h.npy')
	ts_arr = np.load('../processed/pushup_raw_ts_one_all_70h.npy')
	X = data_arr[:,[2,3]].astype(float) # just the amplitude and duration
	labels = data_arr[:,-1]
	labels[labels =='excellent'] = 1
	labels[labels == 'good'] = 1
	labels[labels == 'ok'] = 0
	labels = labels.astype(int)

	# train classifier
	stance = 'all'
	c = ClassifyRep(X, labels)
	sss = c.split_data(n_iter=5, test_size = 0.3, random_state=100)
	c.random_forest(sss, stance=stance, n_est=50, max_feat=2, max_depth=2, pickle=True)
	c.support_vector_machine(sss, stance=stance, C=10, gamma=1.0, pickle=True)
	ts = ts_arr[0]
	c.dtw_kNN(sss, ts, component='pitch', stance=stance, avg_length=34, n_neighbors=4, max_warping_window=10, pickle=True)
	ts = ts_arr[1]
	c.dtw_kNN(sss, ts, component='accY', stance = stance, avg_length=34, n_neighbors=4, max_warping_window=10, pickle=True)
	ts = ts_arr[2]
	c.dtw_kNN(sss, ts, component='accZ', stance = stance, avg_length=34, n_neighbors=4, max_warping_window=10, pickle=True)
	weights = np.array([0.25, 0.25, 0.0, 0.25, 0.25])
	c.ensemble(sss, weights)