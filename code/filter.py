""" Bandpass filter the Exercise data """

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=1):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def filter_features(data, features, lowcut, highcut, fs, order=1):
	df_filt = data.copy()
	for f in features:
		filtered = butter_bandpass_filter(data[f], lowcut=lowcut, highcut=highcut, fs=fs, order=order)
		df_filt[f] = filtered
	return df_filt