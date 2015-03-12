import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
from datetime import date, timedelta
import pandas as pd
import brewer2mpl

spectral_colors = brewer2mpl.get_map('Spectral', 'Diverging', 10).colors

def daily_reps(timestamp, w_prob, sample):
	good = w_prob[(w_prob > 0.5)]
	ok = w_prob[(w_prob <= 0.5)]
	x = np.arange(0,24,1)
	y1 = np.linspace(0,0,24)
	y1[timestamp] = len(ok)
	trace1 = Bar(
	    x=x,
	    y=y1,
	    name='Ok',
	    marker=Marker(
        	color='rgb(204, 0, 0)'
    	)
	)
	y2 = np.linspace(0,0,24)
	y2[timestamp] = len(good)
	trace2 = Bar(
	    x=x,
	    y=y2,
	    name='Good',
	    marker=Marker(
        	color='rgb(0, 204, 102)'
    	)
	)
	data = Data([trace1, trace2])
	layout = Layout(
	    barmode='stack',
	    title='Daily Activity',
	    yaxis=YAxis(
        	title='Reps',
        	titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
    	),
    	xaxis=XAxis(
    		title='Hour',
    		titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
        ),
        autosize=False,
    	width=400,
    	height=300,
    	margin=Margin(
        	l=40,
        	r=40,
        	b=40,
        	t=30,
        	pad=4
    	)
	)
	fig = Figure(data=data, layout=layout)
	py.image.save_as(fig, '../figures/daily/daily_'+sample+'.svg')
	#plot_url = py.plot(fig, filename='daily_'+sample)

def monthly_reps(bin_history, sample):
	start = date.today() - timedelta(days=29)
	end = date.today()
	x = pd.date_range(start, end)
	x = [y.strftime("%m/%d") for y in x]
	dates = [bin_history[i][1].strftime("%m/%d") for i in range(len(bin_history))]
	ok = [bin_history[i][0][0] for i in range(len(bin_history))]
	good = [bin_history[i][0][1] for i in range(len(bin_history))]
	y1 = np.linspace(0,0,30)
	y2 = np.linspace(0,0,30)
	for i, d in enumerate(dates):
		ind = x.index(d)
		y1[ind] += ok[i]
		y2[ind] += good[i]

	trace1 = Bar(
	    x=x,
	    y=y1,
	    name='Ok',
	    marker=Marker(
        	color='rgb(204, 0, 0)'
    	)
	)
	trace2 = Bar(
	    x=x,
	    y=y2,
	    name='Good',
	    marker=Marker(
        	color='rgb(0, 204, 102)'
    	)
	)
	data = Data([trace1, trace2])
	layout = Layout(
	    barmode='stack',
	    title='30 Day Activity',
	    yaxis=YAxis(
        	title='Reps',
        	titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
    	),
        autosize=False,
    	width=600,
    	height=300,
    	margin=Margin(
        	l=40,
        	r=40,
        	b=60,
        	t=30,
        	pad=4
    	)
	)
	fig = Figure(data=data, layout=layout)
	py.image.save_as(fig, '../figures/monthly/monthly_user'+str(sample)+'.svg')
	#plot_url = py.plot(fig, filename='daily_'+sample)

def make_trace(x, y, name, color):  
    return dict(
        x=x,     
        y=y,
        name=name,
        line=dict(color='rgb('+str(color[0])+','+str(color[1])+','+str(color[2])+')')
    )

def rad_to_degree(rad):
	return [rad[r]*180.0 / np.pi for r in xrange(len(rad))]

def calculate_time_axis(ts, freq):
	return np.arange(0, len(ts), 1) / freq

def plot_ts(ts, sample, freq=20.0):
	# optimal pitch
	example_ts = np.load('../processed/pushup_raw_ts_one_all.npy')[0,31] #normal stance Beau
	# initialize rep to 0
	B = np.array([xi - xi[0] for xi in [example_ts]])
	traceB = make_trace(calculate_time_axis(B[0], freq), rad_to_degree(B[0]), 'optimal', [150,150,150])
	
	# user pitch
	trace = [make_trace(calculate_time_axis(ts[i], freq), rad_to_degree(ts[i]), str(i+1), spectral_colors[i%10]) for i in xrange(len(ts))]
	trace.insert(0, traceB)
	data = Data(trace)
	layout = Layout(
	    title='Last Set of Reps',
	    yaxis=YAxis(
        	title='Pitch (degrees)',
        	range = [-90, 10],
        	titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
    	),
    	xaxis=XAxis(
    		title='Duration (seconds)',
    		titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
        ),
        autosize=False,
    	width=400,
    	height=300,
    	margin=Margin(
        	l=55,
        	r=45,
        	b=45,
        	t=30,
        	pad=4
    	)
	)
	fig = Figure(data=data, layout=layout)
	py.image.save_as(fig, '../figures/pushup_ts/Pitch_'+sample+'.svg')