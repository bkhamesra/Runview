import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
rc('font', **{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['lines.linewidth']=2

def spinplots(file1, file2, figdir):
	ihspin_0 = open(file1)
	ihspin_1 = open(file2)
	time_bh1, sx1, sy1, sz1 = np.loadtxt(ihspin_0, unpack=True, usecols=(0,1,2,3))
	time_bh2, sx2, sy2, sz2 = np.loadtxt(ihspin_1, unpack=True, usecols=(0,1,2,3))
	
	s1 = np.sqrt(sx1**2. + sy1**2. + sz1**2.)
	s2 = np.sqrt(sx2**2. + sy2**2. + sz2**2.)
	
	sxplot = plot2(time_bh1, sx1, time_bh2, sx2, 'Time', r'$S_x$', 'Spinx', figdir)
	syplot = plot2(time_bh1, sy1, time_bh2, sy2, 'Time', r'$S_y$', 'Spiny', figdir)
	szplot = plot2(time_bh1, sz1, time_bh2, sz2, 'Time', r'$S_z$', 'Spinz', figdir)
	smag_plot = plot2(time_bh1, s1, time_bh2, s2, 'Time', r'$|S|$', 'Spinmag', figdir)

def Spins(wfdir, outdir):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
	ihspin0 = os.path.join(datadir,"ihspin_hn_0.asc")
	ihspin1 = os.path.join(datadir,"ihspin_hn_1.asc")
	
	if not os.path.isfile(ihspin0):
		ihspin0 = os.path.join(datadir,"ihspin_hn_3.asc")
		ihspin1 = os.path.join(datadir,"ihspin_hn_4.asc")
		
	if os.path.isfile(ihspin0):
		spinplots(ihspin0, ihspin1, figdir)
		
		
		
