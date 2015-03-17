from anonymous_user_prediction import AnonPrediction

if __name__ == '__main__':
	p = AnonPrediction('../data/test_samples/test_expert_male_basic.txt')
	tip, daily_url, ts_url, bar_url, monthly_url = p.process_user_sample()
	print daily_url, ts_url, bar_url, monthly_url