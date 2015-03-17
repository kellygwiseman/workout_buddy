""" 
This library is used to make interactive plots for the webapp. The figures are
stored at https:plot.ly and the functions return the unique url. 
"""

import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
from datetime import date, timedelta
import pandas as pd
import brewer2mpl

spectral_colors = brewer2mpl.get_map('Spectral', 'Diverging', 10).colors

def reps_bar_chart(w_prob, sample):
	"""Plot the latest set of reps ratings in a bar chart and returns url."""

	w_prob_true = (w_prob > 0.5)*1.0
	w_prob_false = (w_prob <= 0.5)*1.0
	trace1 = Scatter(
		x = np.arange(1, len(w_prob) + 1, 1.0),
		y = np.linspace(50,50,len(w_prob)),
		name='Good threshold',
		mode = 'lines',
		marker = Marker(
			color = 'grey',
			line = Line(
				color = 'grey',
				width = 1.0
				)
			)
		)
	trace2 = dict(
		x = np.arange(1, len(w_prob) + 1, 1.0),
		y = np.multiply(w_prob_false, w_prob)*100,
		type = 'bar',
		name = 'Ok',
		marker = Marker(
			color = 'rgb(204, 0, 0)'
			)
		)
	trace3 = dict(
		x = np.arange(1, len(w_prob) + 1, 1.0),
		y = np.multiply(w_prob_true, w_prob)*100,
		type = 'bar',
		name = 'Good',
		marker = Marker(
			color = 'rgb(0, 204, 102)'
			)
		)
	
	data = Data([trace1, trace2, trace3])
	layout = Layout(
	    title = 'Form Breakdown of Last Set of Reps',
        autosize = False,
    	width = 400,
    	height = 400,
    	margin = Margin(
        	l = 50,
        	r = 50,
        	b = 50,
        	t = 70,
        	pad = 4
    	),
    	yaxis = YAxis(
    		title = 'Repetition Rating (percent)',
    		titlefont = Font(
            	size = 16,
            	color = 'rgb(107, 107, 107)'
        		),
        	range = [0.0,100.0]
    	),
    	xaxis = XAxis(
    		title = 'Repetition',
    		titlefont = Font(
            	size = 16,
            	color = 'rgb(107, 107, 107)'
        	),
    		range = [0, len(w_prob) + 1]
    	)
	)
	fig = Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='rep_rating' + str(sample), file_opt='new', world_readable=True, auto_open=False)
	return plot_url

def reps_polar_chart(w_prob, sample):
	"""Plots the latest set of reps ratings in a polar chart and returns the url."""

	w_prob_true = (w_prob > 0.5)*1.0
	w_prob_false = (w_prob <= 0.5)*1.0
	trace1 = dict(
		t = np.arange(0.5, len(w_prob) + 0.5, 1.0),
		r = np.multiply(w_prob_false, w_prob)*100,
		type = 'area',
		name = 'Ok',
		marker = Marker(
			color = 'rgb(204, 0, 0)'
			)
		)
	trace2 = dict(
		t = np.arange(0.5, len(w_prob) + 0.5, 1.0),
		r = np.multiply(w_prob_true, w_prob)*100,
		type = 'area',
		name = 'Good',
		marker = Marker(
			color = 'rgb(0, 204, 102)'
			)
		)
	data = Data([trace1, trace2])
	layout = Layout(
	    title = 'Form Breakdown of Latest Reps',
        autosize = False,
    	width = 400,
    	height = 400,
    	margin = Margin(
        	l = 50,
        	r = 50,
        	b = 50,
        	t = 50,
        	pad = 4
    	),
    	radialaxis = RadialAxis(
        	ticksuffix = '%',
        	range = [0.0, 100.0]
    	),
    	angularaxis = AngularAxis(
    		range = [0,len(w_prob)]
    	)
	)
	fig = Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='polar_rep_' + str(sample), file_opt='new', world_readable=True, auto_open=False)
	return plot_url

def monthly_reps(bin_history, sample):
	"""Plots the past month's aggregate number of reps in a bar chart and returns the url."""

	start = date.today() - timedelta(days=29)
	end = date.today()
	x = pd.date_range(start, end)
	x = [y.strftime("%m/%d") for y in x]
	dates = [bin_history[i][1].strftime("%m/%d") for i in range(len(bin_history))]
	ok = [bin_history[i][0][0] for i in range(len(bin_history))]
	good = [bin_history[i][0][1] for i in range(len(bin_history))]
	y1 = np.linspace(0, 0, 30)
	y2 = np.linspace(0, 0, 30)
	for i, d in enumerate(dates):
		ind = x.index(d)
		y1[ind] += ok[i]
		y2[ind] += good[i]

	trace1 = Bar(
	    x = x,
	    y = y1,
	    name = 'Ok',
	    marker = Marker(
        	color = 'rgb(204, 0, 0)'
    	)
	)
	trace2 = Bar(
	    x = x,
	    y = y2,
	    name = 'Good',
	    marker = Marker(
        	color = 'rgb(0, 204, 102)'
    	)
	)
	data = Data([trace1, trace2])
	layout = Layout(
	    barmode = 'stack',
	    title = '30 Day Activity',
	    yaxis = YAxis(
        	title = 'Number of Repetitions',
        	titlefont = Font(
            	size = 16,
            	color = 'rgb(107, 107, 107)'
        	)
    	),
        autosize = False,
    	width = 600,
    	height = 400,
    	margin = Margin(
        	l = 50,
        	r = 50,
        	b = 50,
        	t = 70,
        	pad = 4
    	)
	)
	fig = Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='daily_user' + str(sample), file_opt='new', world_readable=True, auto_open=False)
	return plot_url

def make_trace(x, y, name, color):
	"""Creates a repetition trace for the time series plot."""

    return Scatter(
        x = x,     
        y = y,
        name = name,
        mode = 'lines',
        marker = Marker(
        	color = 'rgb('+str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')',
        	line = Line(
        		color = 'rgb(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')',
        		width = 1.0
        		)
        	)
    )

def rad_to_degree(rad):
	"""Converts radians to degrees."""
	return [rad[r]*180.0 / np.pi for r in xrange(len(rad))]

def calculate_time_axis(ts, freq):
	"""Creates the repetiton time axis for the time series plot."""
	return np.arange(0, len(ts), 1) / freq

def plot_ts(ts, sample, freq=20.0):
	"""Plots the pitch data for the latest set of repetitions in a time series
	format and returns the url. Each rep is plotted individually so you can 
	easily compare the reps in your set."""

	# expert user optimal pitch for basic pushup stance
	example_ts = np.load('../processed/pushup_raw_ts_expert_basic.npy')[0,4]
	# initialize rep to 0
	B = np.array([xi - xi[0] for xi in [example_ts]])
	traceB = make_trace(calculate_time_axis(B[0], freq), rad_to_degree(B[0]), 'optimal rep', [150, 150, 150])

	# user pitch
	trace = [make_trace(calculate_time_axis(ts[i], freq), rad_to_degree(ts[i]), str(i + 1), spectral_colors[i % 10]) for i in xrange(len(ts))]
	trace.insert(0, traceB)
	data = Data(trace)
	layout = Layout(
	    title = 'Time Series of Last Set of Reps',
	    yaxis = YAxis(
        	title = 'Pitch (degrees)',
        	range = [-100, 10],
        	titlefont = Font(
            	size = 16,
            	color = 'rgb(107, 107, 107)'
        	)
    	),
    	xaxis = XAxis(
    		title = 'Rep Duration (seconds)',
    		titlefont = Font(
            	size = 16,
            	color = 'rgb(107, 107, 107)'
        	)
        ),
        autosize = False,
    	width = 600,
    	height = 400,
    	margin = Margin(
        	l = 50,
        	r = 50,
        	b = 50,
        	t = 60,
        	pad = 4
    	)
	)
	fig = Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='Pitch_user' + str(sample), file_opt='new', world_readable=True, auto_open=False)
	return plot_url