""" 
Pipeline used to classify all the repetitions for one user. Three example
user types are included. 
"""

from user_prediction import UserPrediction
import pandas as pd
import plotly_graphs as pg

if __name__ == '__main__':
	# Novice user
	user = 1
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user=user)
	tip, ts_url, bar_url, monthly_url = p.batch_process_user_samples()
	print ts_url, bar_url, monthly_url

	# Good user
	user = 6
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user=user)
	tip, ts_url, bar_url, monthly_url = p.batch_process_user_samples()
	print ts_url, bar_url, monthly_url

	# Expert user
	user = 7
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user=user)
	tip, ts_url, bar_url, monthly_url = p.batch_process_user_samples()
	print ts_url, bar_url, monthly_url

