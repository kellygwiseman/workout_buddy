import pandas as pd
import numpy as np
from dtw import *
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.svm import SVC

class ClassifyRep(object):
	def __init__(self, X, labels):
		'''
		INPUT:
		- X: pushup feature array
		- labels: pushup form labels

		'''
		self.X = X
		self.labels = labels

	def split_data(self, n_iter=10, test_size = 0.3, random_state=100):
		sss = StratifiedShuffleSplit(self.labels, n_iter = 10, test_size = 0.3, random_state=100)
		return sss

	def random_forest(self, sss, n_est=50, max_feat=2, max_depth=2):
		acc_list = []
		recall_list = []
		prec_list = []
		i=1
		for train_index, test_index in sss:
		    X_train, X_test = self.X[train_index], self.X[test_index]
		    y_train, y_test = self.labels[train_index], self.labels[test_index]
		    rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat, max_depth=max_depth)
		    rf.fit(X_train, y_train)
		    y_pred = rf.predict(X_test)
		    y_prob = rf.predict_proba(X_test)
		    acc = metrics.accuracy_score(y_test, y_pred)
		    acc_list.append(acc)
		    recall = metrics.recall_score(y_test, y_pred)
		    recall_list.append(recall)
		    precision = metrics.precision_score(y_test, y_pred)
		    prec_list.append(precision)
		    cm = metrics.confusion_matrix(y_test, y_pred)
		    print'Iteration',i,'confusion matrix',cm
		    print ''
		    i+=1

		print 'Average accuracy and std:', np.mean(acc_list), np.std(acc_list)
		print 'Average precison and std:', np.mean(prec_list), np.std(prec_list)
		print 'Average recall and std:', np.mean(recall_list), np.std(recall_list)

	def support_vector_machine(self):
		pass

	def dynamic_time_warping(self):
		pass

	def ensemble(self, models, weights):
		pass