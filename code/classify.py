import pandas as pd
import numpy as np
from dtw import *
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.svm import SVC
from scipy import signal
import cPickle as pickle

class ClassifyRep(object):
	def __init__(self, X, labels):
		'''
		INPUT:
		- X: pushup feature array
		- labels: pushup form labels

		'''
		self.X = X
		self.labels = labels
		self.pred_list = []
		self.ensemble_pred = []
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		self.n_iter = 5

	def split_data(self, n_iter=5, test_size=0.3, random_state=100):
		self.n_iter = n_iter
		sss = StratifiedShuffleSplit(self.labels, n_iter = n_iter, test_size = test_size, random_state=random_state)
		return sss

	def random_forest(self, sss, stance, n_est=50, max_feat=2, max_depth=2, pickle=False):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_list = []
		i=1
		for train_index, test_index in sss:
		    X_train, X_test = self.X[train_index], self.X[test_index]
		    y_train, y_test = self.labels[train_index], self.labels[test_index]
		    rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, max_depth=max_depth)
		    rf.fit(X_train, y_train)
		    y_pred = rf.predict(X_test)
		    y_prob = rf.predict_proba(X_test)
		    self._print_iteration_metrics(y_test, y_pred, i)
		    #pred_list.append(y_prob[:,1])
		    pred_list.append(y_pred)
		    i+=1

		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)
		self.pred_list.append(pred_list)
		if pickle:      
			rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, max_depth=max_depth)
			rf.fit(self.X, self.labels)
			save_model(rf, "../models/rf"+'_n'+str(n_est)+'_mf'+str(max_feat)+'_md'+str(max_depth)+'_'+stance+'.pkl' )

	def support_vector_machine(self, sss, stance, C = 20, gamma = 0.1, pickle=False):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_list = []
		i = 1
		for train_index, test_index in sss:
		    X_train, X_test = self.X[train_index], self.X[test_index]
		    y_train, y_test = self.labels[train_index], self.labels[test_index]
		    svm = SVC(C=C, gamma=gamma)
		    svm.fit(X_train, y_train)
		    y_pred = svm.predict(X_test)
		    self._print_iteration_metrics(y_test, y_pred, i)
		    pred_list.append(y_pred)
		    i+=1

		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)
		self.pred_list.append(pred_list)
		if pickle:      
			svm = SVC(C=C, gamma=gamma)
			svm.fit(self.X, self.labels)
			save_model(svm, "../models/svm"+'_C'+str(C)+'_g'+str(gamma)+'_'+stance+'.pkl')

	def dtw_kNN(self, sss, ts, stance, component, avg_length=34, n_neighbors=4, max_warping_window=10, pickle=False):
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
			y_train, y_test = self.labels[train_index], self.labels[test_index]
			m = KnnDtw(n_neighbors=n_neighbors, max_warping_window=max_warping_window)
			m.fit(X_train, y_train)
			y_pred, proba = m.predict(X_test)
			pred_list.append(y_pred)
			self._print_iteration_metrics(y_test, y_pred, i)
			i+=1

		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)
		self.pred_list.append(pred_list)
		if pickle:      
			m = KnnDtw(n_neighbors=n_neighbors, max_warping_window=max_warping_window)
			m.fit(X, self.labels)
			save_model(m, "../models/dtw_kNN"+component+'_n'+str(n_neighbors)+'_w'+str(max_warping_window)+'_'+stance+'.pkl' )

	def ensemble(self, sss, weights):
		self.acc_list = []
		self.recall_list = []
		self.prec_list = []
		pred_arr = np.array(self.pred_list)
		i = 1
		for train_index, test_index in sss:
			temp_arr = pred_arr[:,i-1,:]
			w_pred = np.dot(weights,temp_arr) 
			w_pred = w_pred > 0.5
			y_test = self.labels[test_index]
			self._print_iteration_metrics(y_test, w_pred, i)
			i+=1
		print 'Average accuracy and std:', np.mean(self.acc_list), np.std(self.acc_list)
		print 'Average precison and std:', np.mean(self.prec_list), np.std(self.prec_list)
		print 'Average recall and std:', np.mean(self.recall_list), np.std(self.recall_list)

	def predict(self, pickle_mdl, X):
		m = get_model(pickle_mdl)
		y_pred = m.predict(X)
		return y_pred

	def predict_ts(self, pickle_mdl, ts, component, avg_length=34):
		# resample to average length
		X = [signal.resample(xi, avg_length) for xi in ts]
		# initialize rep to 0
		X = np.array([xi - xi[0] for xi in X])
		m = get_model(pickle_mdl)
		y_pred, proba = m.predict(X)
		return y_pred
		
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