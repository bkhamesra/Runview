###############################################################################
# Script - CommonFunctions.py
# Author - Bhavesh Khamesra
# Purpose -  Contains common functions used frequently in this package. 
###############################################################################

from inspect import getframeinfo, stack
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
#mpl.rcParams['lines.linewidth']=2

def output_data(parfile, data):
	
    '''Check if file is present in Stitched output directory/waveform directoryi
	
       ----------------- Parameters -------------------
       parfile (type String) - path of parfile
       data (type String) - parameter to be searched
    '''		

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
	
    '''Output debugging informationi
	
       ----------------- Parameters -------------------
       message (type string) - debugging message to user
    '''		

	caller = getframeinfo(stack()[1][0])
	filename = caller.filename.split("/")[-1]

	print("Warning: %s:%d - %s" % (filename, caller.lineno, message))

def DataDir(dirpath, outdir):
    '''Create data directory to store important files inside Summary
	
       ----------------- Parameters -------------------
       dirpath (type string) - simulation directory path
       outdir (type string) - output Summary directory parent path
    '''		

	filename = dirpath.split("/")[-1]
	outputdir = os.path.join(outdir, filename)
	datadir = os.path.join(outputdir,'data')

	if not os.path.exists(datadir):
		os.makedirs(datadir)
	return datadir


def FigDir(dirpath, outdir):

    '''Create figures directory to store important files inside Summary
	
       ----------------- Parameters -------------------
       dirpath (type string) - simulation directory path
       outdir (type string) - output Summary directory parent path
    '''		
	filename = dirpath.split("/")[-1]
	outputdir = os.path.join(outdir, filename)
	figdir = os.path.join(outputdir,'figures')
	
	if not os.path.exists(figdir):
		os.makedirs(figdir)
	return figdir


def norm(vec, axis):
      ''' Compute norm of the vector

       ----------------- Parameters -------------------
       vec (type np.array) - vector whose norm to be computed
       axis (type int) - axis along which to compute the norm
    '''		
	return np.apply_along_axis(np.linalg.norm, axis, vec)	


def plot1(x,y,xlabel, ylabel, plotname, outdir):
      ''' Create y vs x plot

       ----------------- Parameters -------------------
       x (type np.array) - x data 
       y (type np.array) - y data 
       xlabel/ylabel (type string) - label for x/y axis
       plotname (type string) - Title of the plot
       outdir (type string) - Directory in which to save the plot
    '''		

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

      ''' Create comparison y vs x plot

       ----------------- Parameters -------------------
       x1, x2 (type np.array) - x axis datasets
       y1, y2 (type np.array) - y axis datasets 
       xlabel/ylabel (type string) - label for x/y axis
       plotname (type string) - Title of the plot
       outdir (type string) - Directory in which to save the plot
    '''		
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


def func_t_hrzn(datadir, locate_merger):

      ''' Find first appearance of final black hole

       ----------------- Parameters -------------------
       datadir (type string) - Directory with BH_diagnostics data
       locate_merger (type Boolean) - Find time of BH3 if set to True 
    '''		
	if locate_merger==True:
		bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
		t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
		return t_hrzn3
	else:
		return 
