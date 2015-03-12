from user_prediction import UserPrediction
import graphs as gr
import pandas as pd
import numpy as np
import plotly_graphs as pg
from scipy import signal

if __name__ == '__main__':
	# process one sample
	user = 21
	info = pd.read_csv('../data/full_dataset_info.csv', skipinitialspace=True)
	p = UserPrediction(info, user = user)
	prob_history, bin_history = p.batch_process_user_samples()
	pg.monthly_reps(bin_history, user)



