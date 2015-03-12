"""Determine Attributes about Exercise Repetitions"""

from __future__ import division, print_function
import numpy as np

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Code from Marcos Duarte, https://github.com/demotu/BMC, version = 1.0.4

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 
"""
    

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def count_peaks_initial(data, pushup_window, feature, mph, mpd, freq, valley=False, edge='falling'):
    ''' Calculate initial timing of the press-up position since start of pushup window. 
    Because the data is filtered, this won't include the starting up position. '''
    pushup_data = data.ix[pushup_window[0]:pushup_window[-1]]
    peakind = detect_peaks(pushup_data[feature], mph = mph, mpd = mpd, valley=valley)
    peakind = [x / freq for x in peakind] # convert to seconds instead of frequency
    count = len(peakind)
    return peakind, count

def count_peak_min(data, window_ind, feature, mph, mpd, freq, valley=False, edge='falling'):
    ''' Calculate timing of the press-down position since start of pushup window'''
    pushup_data = data.ix[window_ind[0]:window_ind[-1]][feature].values
    # force the push-up reps to start at about 0 pitch amplitude
    pushup_data = [pushup_data[i] - pushup_data[0] for i in xrange(len(pushup_data))]
    peakind = detect_peaks(pushup_data, mph = mph, mpd = mpd, valley=valley)
    peakind = [x / freq for x in peakind] # convert to seconds instead of frequency
    # check to max sure there are no local minimums on the edges
    if (peakind[0] < 0.2):
        peakind = peakind[1:]
    if (window_ind[-1] - window_ind[0])/freq - peakind[-1] < 0.3:
        peakind = peakind[:-1]
    count = len(peakind)
    return peakind, count, pushup_data

def count_peak_max(data, peakmin_count, window_ind, feature, mph, mpd, freq, valley=False, edge='falling'):
    ''' Calculate timing of the press-up position since start of pushup window.  This also include the starting
    up position, so there is one more peak_max than peak_min.'''
    pushup_data = data.ix[window_ind[0]:window_ind[-1]][feature].values
    # force the push-up reps to start at about 0 pitch amplitude
    pushup_data = [pushup_data[i] - pushup_data[0] for i in xrange(len(pushup_data))]
    peakind = detect_peaks(pushup_data, mph = mph, mpd = mpd, valley=valley)
    count = len(peakind)
    peakind = [x / freq for x in peakind] # convert to seconds instead of frequency
    if (peakind[0] > 1.0) and (count <= peakmin_count):
        peakind.insert(0, 0)
    count = len(peakind)
    return peakind, count, pushup_data

def average_duration(peakind, count):
    # don't include the first repetition duration because we are counting peaks (the end of the pushup)
    # for this initial average duration before we have the final pushup window
    duration = peakind[-1] - peakind[0]
    avg_dur = duration / (count - 1)
    return avg_dur

def average_amplitude_initial(data, peakind, pushup_window, feature, freq):
    ind = [int(x*freq) for x in peakind]
    ind = [pushup_window[0] + x for x in ind]
    amps = data.ix[ind][feature]
    avg_amp = amps.mean()
    return avg_amp

def average_amplitude(data, peakmin, peakmax, window_ind, feature, freq):
    min_ind = [window_ind[0] + int(x*freq) for x in peakmin]
    max_ind = [window_ind[0] + int(x*freq) for x in peakmax]
    amps = [data.ix[max_ind[i]][feature] - data.ix[min_ind[i]][feature] for i in xrange(len(min_ind))]
    avg_amp = np.mean(amps)
    return avg_amp

def rep_metrics(data, peakmin, peakmax, window_ind, feature, freq, female, height):
    min_ind = [window_ind[0] + int(x*freq) for x in peakmin]
    max_ind = [window_ind[0] + int(x*freq) for x in peakmax]
    amps = [data.ix[max_ind[i]][feature] - data.ix[min_ind[i]][feature] for i in xrange(len(min_ind))]
    durations = [peakmax[n+1] - peakmax[n] for n in xrange(len(peakmax) - 1)]
    sample_metrics = [[female, height, amps[n], durations[n]] for n in xrange(len(amps))]
    return sample_metrics

def avg_rep_metrics(data, peakmin, peakmax, window_ind, feature, freq, female, height, form):
    min_ind = [window_ind[0] + int(x*freq) for x in peakmin]
    max_ind = [window_ind[0] + int(x*freq) for x in peakmax]
    amps = [data.ix[max_ind[i]][feature] - data.ix[min_ind[i]][feature] for i in xrange(len(min_ind))]
    avg_amps = np.mean(amps)
    amp_std = np.std(amps)
    durations = [peakmax[n+1] - peakmax[n] for n in xrange(len(peakmax) - 1)]
    avg_dur = np.mean(durations)
    dur_std = np.std(durations)
    sample_metrics = [female, height, avg_amps, avg_dur, amp_std, dur_std, form]
    return sample_metrics

def one_rep_window(peakmax, window_ind, freq):
    # use middle rep for example rep
    middle_rep_ind = int(len(peakmax) / 2)
    rep_duration = peakmax[middle_rep_ind] - peakmax[middle_rep_ind-1] 
    start = window_ind[0]/freq + peakmax[middle_rep_ind] - rep_duration
    end = window_ind[0]/freq + peakmax[middle_rep_ind]
    window_sec = (start, end)
    window_ind = (int(start*freq), int(end*freq))
    return window_ind

def calculate_total_rep_window(peakind, pushup_window, avg_duration, freq):
    start = pushup_window[0]/freq + peakind[0] - avg_duration - 0.6
    end = pushup_window[0]/freq + peakind[-1] + (avg_duration / 3)
    window_sec = (start, end)
    window_ind = (int(start*freq), int(end*freq))
    return window_ind

def calculate_multiple_rep_window(peakmax, window_ind, freq):
    multiple_window = [(window_ind[0]+int(peakmax[n]*freq), window_ind[0]+int(peakmax[n+1]*freq)) for n in xrange(len(peakmax)-1)]
    return multiple_window