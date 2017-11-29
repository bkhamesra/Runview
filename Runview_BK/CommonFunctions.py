import plotly.offline as py
import plotly.graph_objs as go
from inspect import getframeinfo, stack
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
#mpl.rcParams['lines.linewidth']=2

def output_data(parfile, data):
	
	datafile = file(parfile)	
	datafile.seek(0)
	for line in datafile:
		if data in line:
			break
	line = line.split()
	data_value = float(line[-1])
	datafile.close()
	return data_value

def debuginfo(message):

	caller = getframeinfo(stack()[1][0])
	filename = caller.filename.split("/")[-1]

	print "Warning: %s:%d - %s" % (filename, caller.lineno, message)

def DataDir(dirpath, outdir):

	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outdir, filename)
	datadir = os.path.join(outputdir,'data')

	if not os.path.exists(datadir):
		os.makedirs(datadir)
	return datadir


def FigDir(dirpath, outdir):

	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outdir, filename)
	statfigdir = os.path.join(outputdir,'figures/static/')
	dynfigdir = os.path.join(outputdir,'figures/dynamic/')

	if not os.path.exists(statfigdir):
		os.makedirs(statfigdir)
	if not os.path.exists(dynfigdir):
		os.makedirs(dynfigdir)
	return [statfigdir, dynfigdir]


def norm(vec, axis):

	return np.apply_along_axis(np.linalg.norm, axis, vec)	


def plot1(x,y,xlabel, ylabel, plotname, outdir):

        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	ax.plot(x,y, 'b', linewidth=1)

	ax.set_ylabel(ylabel, fontsize = 14)
	ax.set_xlabel(xlabel, fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4), fontsize=200)
	startx,endx = ax.get_xlim()
	#plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10. )))
	
	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(outdir,(plotname+'.png')), dpi = 500)
	plt.close()

def plot2(x1,y1, x2, y2, xlabel, ylabel, plotname, outdir):
        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	bh1, = ax.plot(x1, y1, 'b', linewidth=1, label="BH1")
	bh2, = ax.plot(x2, y2, 'k--', linewidth=1, label = "BH2")

	ax.set_ylabel(ylabel, fontsize = 14)
	ax.set_xlabel(xlabel, fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4), fontsize=20)
	startx,endx = ax.get_xlim()
	#plt.xticks(np.arange(startx, endx, int(endx/10.-startx/10.)))
	
	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(outdir,(plotname+'.png')), dpi = 500)
	plt.close()

#Generic Functions for 1 and 2 object plots. Adaptations (read: basically copies) of trajectories plots, comments and all. Go to those to figure out how they work more clearly

#1 variable plot, with added log toggle
def plyplot1(x,y1,x_label,y_label,title, Ly2=None, locate_merger=False, point_hrzn=0, point_maxamp=0): 
	trace1 = go.Scatter(
	  x = x, 
	  y = y1,
	  mode = "lines",
	  name = y_label
	)
	if Ly2 != None:
	  trace2 = go.Scatter( #logarithmic data can be plotted on the same and toggled, as performed below
	    x = x, 
	    y = Ly2,
	    mode = "lines",
	    visible=False, #makes this data not load on startup
	    name = "Log " + y_label
	  )
	  
	  data = [trace1,trace2]
	  
	  updatemenus = list([
	    dict(type="buttons",
		active=-1,
		buttons=list([
		  dict(label="Regular",
			method='update',
			args=[{'visible': [True,False]},
			      {'title':title}]),
		  dict(label="Log",
			method='update',
			args=[{'visible':[False,True]},
			      {'title':"Log "+title}])]))])
	  
	  layout = go.Layout(
	    title = title,
	    hovermode = "closest",
	    xaxis = dict(
	      title = x_label
	    ),
	    yaxis = dict(
	      title = y_label
	    ),
	    updatemenus=updatemenus
	  )
	    
	else:
	  data = [trace1]
	  
	if locate_merger==True:
	  layout = go.Layout(
	    title = title,
	    hovermode = "closest",
	    xaxis = dict(
	      title = x_label
	    ),
	    yaxis = dict(
	      title = y_label
	    ),
	    shapes = [
	    {
            'type': 'line',
            'x0': point_hrzn,
            'y0': 0,
            'x1': point_hrzn,
            'y1': y1[int(point_hrzn)],
            'line': {
                'color': 'rgb(255, 0, 0)',
                'width': 1,
	      },
	    },
	    {
            'type': 'line',
            'x0': point_maxamp,
            'y0': 0,
            'x1': point_maxamp,
            'y1': y1[int(point_maxamp)],
            'line': {
                'color': 'rgb(0, 255, 0)',
                'width': 1,
	      },
	    }]
	    )
	else:
	  layout = go.Layout(
	    title = title,
	    hovermode = "closest",
	    xaxis = dict(
	      title = x_label
	    ),
	    yaxis = dict(
	      title = y_label
	    )
	  )

	  
	plot = go.Figure(data=data, layout=layout)
	return plot
	
def plyplot2(x1, x2, y1, y2, name1, name2, x_label, y_label, title, locate_merger=False, point_hrzn=0, point_maxamp=0 ): #for 2 objects; directly from trajectories, with comments included
	trace1= go.Scatter( #scatter is standard data type, accomodates discrete points and lines, the latter used here
	  x = x1, 
	  y = y1,
	  mode = "lines",
	  name = name1 #variables and labels should be fairly intuitive
	)
	trace2 = go.Scatter( #I call them traces because that's what plotly calls them
	  x = x2, 
	  y = y2,
	  mode = "lines",
	  name = name2
	)
	
	data = [trace1, trace2] #data is a list containing all the graph objects. It could be initialized with the object initializations inside, but that quickly gets ugly
	if locate_merger==True:
	  layout = go.Layout( #layout objects do exactly what you think they do
	    title = title, #obvious
	    hovermode = "closest", #sets what data point the hover info will display for
	    xaxis = dict( #obvious, but note use of dict for these, although it doesn't follow dictionary notation. If in doubt, read the syntax errors
	      title = x_label
	    ),
	    yaxis = dict(
	      title = y_label
	    ),
	    shapes = [
	  {
	      'type': 'line',
	      'x0': point_hrzn,
	      'y0': 0,
	      'x1': point_hrzn,
	      'y1': y1[int(point_hrzn)],
	      'line': {
		  'color': 'rgb(255,0 ,0)',
		  'width': 1,
	      },
	  },
	  {
	      'type': 'line',
	      'x0': point_maxamp,
	      'y0': 0,
	      'x1': point_maxamp,
	      'y1': y1[int(point_maxamp)],
	      'line': {
		  'color': 'rgb(0, 255, 0)',
		  'width': 1,
	      },
	  }]
	  )
	else:
	  layout = go.Layout( #layout objects do exactly what you think they do
	    title = title, #obvious
	    hovermode = "closest", #sets what data point the hover info will display for
	    xaxis = dict( #obvious, but note use of dict for these, although it doesn't follow dictionary notation. If in doubt, read the syntax errors
	      title = x_label
	    ),
	    yaxis = dict(
	      title = y_label
	    ),
	  )
	
	
	  
	
	plot = go.Figure(data=data, layout=layout) #creates the figure object		
	return plot #do the final step of actually plotting inside the appropriate file, with the appropriate folder paths
      
def func_t_hrzn(datadir, locate_merger):

	if locate_merger==True:
		bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
		t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
		return t_hrzn3
	else:
		return 
