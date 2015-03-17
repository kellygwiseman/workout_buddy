from user_prediction import UserPrediction
import pandas as pd
import plotly_graphs as pg

""" Used to classify all the repetitions for one user """

if __name__ == '__main__':
	user = 1
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user=user)
	tip, daily_url, ts_url, bar_url, monthly_url = p.batch_process_user_samples()
	print daily_url, ts_url, bar_url, monthly_url

	user = 6
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user=user)
	tip, daily_url, ts_url, bar_url, monthly_url = p.batch_process_user_samples()
	print daily_url, ts_url, bar_url, monthly_url


	user = 7
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user=user)
	tip, daily_url, ts_url, bar_url, monthly_url = p.batch_process_user_samples()
	print daily_url, ts_url, bar_url, monthly_url

