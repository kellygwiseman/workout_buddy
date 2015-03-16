from user_prediction import UserPrediction
import pandas as pd
import plotly_graphs as pg

if __name__ == '__main__':
	user = 1
	#info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user = user)
	prob_history, bin_history, tip, daily_url, ts_url, bar_url = p.batch_process_user_samples()
	monthly_url = pg.monthly_reps(bin_history, user)

	user = 6
	#info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user = user)
	prob_history, bin_history, tip, daily_url, ts_url, bar_url = p.batch_process_user_samples()
	monthly_url = pg.monthly_reps(bin_history, user)

	user = 7
	#info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user = user)
	prob_history, bin_history, tip, daily_url, ts_url, bar_url = p.batch_process_user_samples()
	monthly_url = pg.monthly_reps(bin_history, user)
