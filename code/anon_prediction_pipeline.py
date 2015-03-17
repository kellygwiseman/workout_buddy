""" 
Pipeline used to classify one anonymous set of pushup repetitions. No user info
is provided, only the sensor data is processed.
"""

from anonymous_user_prediction import AnonPrediction

if __name__ == '__main__':
	# An example of an expert user doing basic pushups is provided
	p = AnonPrediction('../data/test_samples/test_expert_male_basic.txt')
	tip, ts_url, bar_url, monthly_url = p.process_user_sample()
	print ts_url, bar_url, monthly_url