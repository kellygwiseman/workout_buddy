import numpy as np
from dtw import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.svm import SVC
from scipy import signal
import cPickle as pickle

class ClassifyRep(object):
	"""
	Add class description
	"""
	def __init__(self):
		self.pred_list = []
		self.ensemble_pred = []
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		self.n_iter = 5

	def split_data(self, labels, n_iter=5, test_size=0.3, random_state=100):
		self.n_iter = n_iter
		sss = StratifiedShuffleSplit(labels, n_iter=n_iter, test_size=test_size, random_state=random_state)
		return sss

	def random_forest(self, sss, X, labels, stance, n_est=50, max_feat=2, max_depth=2, prob=False, pickle=False):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_list = []
		i = 1
		for train_index, test_index in sss:
		    X_train, X_test = X[train_index], X[test_index]
		    y_train, y_test = labels[train_index], labels[test_index]
		    rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, max_depth=max_depth)
		    rf.fit(X_train, y_train)
		    y_pred = rf.predict(X_test)
		    y_prob = rf.predict_proba(X_test)
		    self._print_iteration_metrics(y_test, y_pred, i)
		    if prob:
		    	pred_list.append(y_prob[:,1])
		    else:
		    	pred_list.append(y_pred)
		    i += 1

		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)
		self.pred_list.append(pred_list)
		if pickle:      
			rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, max_depth=max_depth)
			rf.fit(X, labels)
			save_model(rf, "../models/rf" + '_n' + str(n_est) + '_mf' + str(max_feat) + '_md' + str(max_depth) + '_' + stance + '.pkl' )

	def support_vector_machine(self, sss, X, labels, stance, C=10, gamma=0.1, prob=False, pickle=False):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_list = []
		i = 1
		for train_index, test_index in sss:
		    X_train, X_test = X[train_index], X[test_index]
		    y_train, y_test = labels[train_index], labels[test_index]
		    svm = SVC(class_weight='auto', C=C, gamma=gamma, probability=True)
		    svm.fit(X_train, y_train)
		    y_pred = svm.predict(X_test)
		    y_prob = svm.predict_proba(X_test)
		    self._print_iteration_metrics(y_test, y_pred, i)
		    if prob:
		    	pred_list.append(y_prob[:,1])
		    else:
		    	pred_list.append(y_pred)
		    i += 1

		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)
		self.pred_list.append(pred_list)
		if pickle:      
			svm = SVC(class_weight='auto', C=C, gamma=gamma, probability=True)
			svm.fit(X, labels)
			save_model(svm, "../models/svm" + '_C' + str(C) + '_g' + str(gamma) + '_' + stance + '.pkl')

	def dtw_kNN(self, sss, ts, labels, stance, component, avg_length=34, n_neighbors=4, max_warping_window=10, prob=False, pickle=False):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_list = []
		# resample to average length
		X = [signal.resample(xi, avg_length) for xi in ts]
		# initialize rep to 0
		X = np.array([xi - xi[0] for xi in X]) 
		i = 1
		for train_index, test_index in sss:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			m = KnnDtw(n_neighbors=n_neighbors, max_warping_window=max_warping_window)
			m.fit(X_train, y_train)
			y_pred, y_prob = m.predict(X_test)
			for i in xrange(len(y_pred)):
				if y_pred[i] == 0:
					y_prob[i] = 1 - y_prob[i]
			if prob:
				pred_list.append(y_prob)
			else:
				pred_list.append(y_pred)
			self._print_iteration_metrics(y_test, y_pred, i)
			i += 1

		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)
		self.pred_list.append(pred_list)
		if pickle:      
			m = KnnDtw(n_neighbors=n_neighbors, max_warping_window=max_warping_window)
			m.fit(X, labels)
			save_model(m, "../models/dtw_kNN" + component + '_n' + str(n_neighbors) + '_w' + str(max_warping_window) + '_' + stance + '.pkl' )

	def ensemble(self, sss, labels, weights):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_arr = np.array(self.pred_list)
		i = 1
		for train_index, test_index in sss:
			temp_arr = pred_arr[:,i-1,:]
			w_pred = np.dot(weights,temp_arr) 
			w_pred = w_pred > 0.5
			y_test = labels[test_index]
			self._print_iteration_metrics(y_test, w_pred, i)
			i += 1
		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)

	def predict(self, pickle_mdl, X):
		m = get_model(pickle_mdl)
		y_pred = m.predict(X)
		y_prob = m.predict_proba(X)
		return y_pred, y_prob

	def predict_ts(self, pickle_mdl, ts, component, avg_length=34):
		# resample to average length
		Xnorm = [signal.resample(xi, avg_length) for xi in ts]
		# initialize rep to 0
		Xnorm = np.array([xi - xi[0] for xi in Xnorm])
		# non-resampled data for plotting
		X = np.array([xi - xi[0] for xi in ts])
		m = get_model(pickle_mdl)
		y_pred, y_prob = m.predict(Xnorm)
		for i in xrange(len(y_pred)):
			if y_pred[i] == 0:
				y_prob[i] = 1 - y_prob[i]
		return X, y_pred, y_prob
		
	def _print_iteration_metrics(self, y_test, y_pred, i):
		acc = metrics.accuracy_score(y_test, y_pred)
		self.acc_list.append(acc)
		recall = metrics.recall_score(y_test, y_pred)
		self.recall_list.append(recall)
		precision = metrics.precision_score(y_test, y_pred)
		self.prec_list.append(precision)
		cm = metrics.confusion_matrix(y_test, y_pred)
		print'Iteration',i,'confusion matrix:'
		print cm[0]
		print cm[1]
		print ''

def get_model(pickle_mdl):
	with open(pickle_mdl, 'r') as f:
		_model = pickle.load(f)
	return _model

def save_model(mdl, path):
    with open(path, 'w') as f:
        pickle.dump(mdl, f)