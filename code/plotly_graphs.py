import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
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
	py.image.save_as(fig, '../figures/daily/daily_'+sample+'.png')
	#plot_url = py.plot(fig, filename='daily_'+sample)

def make_trace(x, y, name, color):  
    return dict(
        x=x,     
        y=y,
        name=name,
        line=dict(color='rgb('+str(color[0])+','+str(color[1])+','+str(color[2])+')')
    )

def plot_ts(B, ts, sample, avg_length=34, freq=20.0):
	time = np.arange(0,avg_length,1)/freq
	traceB = make_trace(time, B[0], 'optimal', [200,200,200])
	trace = [make_trace(time, ts[i], str(i+1), spectral_colors[i]) for i in xrange(len(ts))]
	trace.insert(0, traceB)
	data = Data(trace)
	layout = Layout(
	    barmode='stack',
	    title='Latest Reps',
	    yaxis=YAxis(
        	title='Pitch (radians)',
        	titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
    	),
    	xaxis=XAxis(
    		title='Duration (Seconds)',
    		titlefont=Font(
            	size=16,
            	color='rgb(107, 107, 107)'
        	)
        ),
        autosize=False,
    	width=400,
    	height=300,
    	margin=Margin(
        	l=45,
        	r=45,
        	b=45,
        	t=30,
        	pad=4
    	)
	)
	fig = Figure(data=data, layout=layout)
	py.image.save_as(fig, '../figures/pushup_ts/pu_ts-Pitch_'+sample+'.png')