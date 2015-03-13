from user_prediction import UserPrediction
import pandas as pd
import plotly_graphs as pg

if __name__ == '__main__':
	# process one sample
	user = 2
	#info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user = user)
	prob_history, bin_history, tip, daily_url, ts_url = p.batch_process_user_samples()
	monthly_url = pg.monthly_reps(bin_history, user)
	
	# write out tip to dashboard based on last set of reps
	print tip

