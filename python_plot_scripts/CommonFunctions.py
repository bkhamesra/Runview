from inspect import getframeinfo, stack
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
rc('font', **{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
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
	figdir = os.path.join(outputdir,'figures')

	if not os.path.exists(figdir):
		os.makedirs(figdir)
	return figdir

def plot1(x,y,xlabel, ylabel, plotname, outdir):
        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	ax.plot(x,y, 'b', linewidth=2)

	ax.set_ylabel(ylabel, fontsize = 14)
	ax.set_xlabel(xlabel, fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	startx,endx = ax.get_xlim()
	plt.xticks(np.arange(startx, endx, 100))
	
	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(outdir,(plotname+'.png')), dpi = 500)
	plt.close()
