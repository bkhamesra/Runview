#ihspin0 and ihpsin3 both have the spin data but they differ in numbers and total time. Guess is one of them is using AHF while other is using sphererad (approx). So would be a good idea to check which file has more data. 


import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
#rc('font', **{'family':'serif','serif':['Computer Modern']})
#rc('text', usetex=True)
#mpl.rcParams['lines.linewidth']=2

def spinplots(file1, file2, file3, wfdir, outdir, locate_merger):
	
	figdir = FigDir(wfdir, outdir)

	ihspin_0 = open(file1)
	ihspin_1 = open(file2)
	ihspin_2 = open(file3)
	time_bh1, sx1, sy1, sz1 = np.loadtxt(ihspin_0, unpack=True, usecols=(0,1,2,3))
	time_bh2, sx2, sy2, sz2 = np.loadtxt(ihspin_1, unpack=True, usecols=(0,1,2,3))
	time_bh3, sx3, sy3, sz3 = np.loadtxt(ihspin_2, unpack=True, usecols=(0,1,2,3))
	
	s1 = np.sqrt(sx1**2. + sy1**2. + sz1**2.)
	s2 = np.sqrt(sx2**2. + sy2**2. + sz2**2.)
	s3 = np.sqrt(sx3**2. + sy3**2. + sz3**2.)
	
	#sx_dict = merger_info_plot3(wfdir, outdir, time_bh1, sx1, time_bh2, sx2, time_bh3, sx3,locate_merger, r=0)
	#sy_dict = merger_info_plot3(wfdir, outdir, time_bh1, sy1, time_bh2, sy2, time_bh3, sy3,locate_merger, r=0)
	#sz_dict = merger_info_plot3(wfdir, outdir, time_bh1, sz1, time_bh2, sz2, time_bh3, sz3,locate_merger, r=0)
	#s_dict = merger_info_plot3(wfdir, outdir, time_bh1, s1, time_bh2, s2, time_bh3, s3,locate_merger, r=0)


	sxplot = plot3(time_bh1, sx1, time_bh2, sx2, time_bh3, sx3, 'Time', r'$S_x$', 'Spinx', figdir)	#, **sx_dict)
	syplot = plot3(time_bh1, sy1, time_bh2, sy2, time_bh3, sy3, 'Time', r'$S_y$', 'Spiny', figdir)	#, **sy_dict)
	szplot = plot3(time_bh1, sz1, time_bh2, sz2, time_bh3, sz3, 'Time', r'$S_z$', 'Spinz', figdir)	#, **sz_dict)
	smag_plot = plot3(time_bh1, s1, time_bh2, s2, time_bh3, s3, 'Time', 'mag(S)', 'Spinmag', figdir)	#, **s_dict)

def spinplots_bh12(file1, file2,  wfdir, outdir, locate_merger):
	
	figdir = FigDir(wfdir, outdir)

	ihspin_0 = open(file1)
	ihspin_1 = open(file2)
	time_bh1, sx1, sy1, sz1 = np.loadtxt(ihspin_0, unpack=True, usecols=(0,1,2,3))
	time_bh2, sx2, sy2, sz2 = np.loadtxt(ihspin_1, unpack=True, usecols=(0,1,2,3))
	
	s1 = np.sqrt(sx1**2. + sy1**2. + sz1**2.)
	s2 = np.sqrt(sx2**2. + sy2**2. + sz2**2.)
	
	#sx_dict = merger_info_plot2(wfdir, outdir, time_bh1, sx1, time_bh2, sx2, locate_merger, r=0)
	#sy_dict = merger_info_plot2(wfdir, outdir, time_bh1, sy1, time_bh2, sy2, locate_merger, r=0)
	#sz_dict = merger_info_plot2(wfdir, outdir, time_bh1, sz1, time_bh2, sz2, locate_merger, r=0)
	#s_dict = merger_info_plot2(wfdir, outdir, time_bh1, s1, time_bh2, s2,locate_merger, r=0)


	sxplot = plot2(time_bh1, sx1, time_bh2, sx2, 'Time', r'$S_x$', 'Spinx', figdir)	# , **sx_dict)
	syplot = plot2(time_bh1, sy1, time_bh2, sy2, 'Time', r'$S_y$', 'Spiny', figdir)	#, **sy_dict)
	szplot = plot2(time_bh1, sz1, time_bh2, sz2, 'Time', r'$S_z$', 'Spinz', figdir)	#, **sz_dict)
	smag_plot = plot2(time_bh1, s1, time_bh2, s2, 'Time', 'mag(S)', 'Spinmag', figdir) #, **s_dict)


def Spins(wfdir, outdir, locate_merger=False):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
	ihspin0 = os.path.join(datadir,"ihspin_hn_0.asc")
	ihspin1 = os.path.join(datadir,"ihspin_hn_1.asc")
	ihspin2 = os.path.join(datadir,"ihspin_hn_2.asc")
	ihspin3 = os.path.join(datadir,"ihspin_hn_3.asc")
	ihspin4 = os.path.join(datadir,"ihspin_hn_4.asc")
	
	if os.path.isfile(ihspin0) and os.path.isfile(ihspin3):
		time1 = np.loadtxt(ihspin0, unpack=True, usecols=(0,))
		time3 = np.loadtxt(ihspin3, unpack=True, usecols=(0,))
		if time3[-1]>=time1[-1]:
			#os.remove(ihspin0)
			#os.remove(ihspin1)
			ihspin0 = ihspin3			#os.rename(ihspin3, ihspin0)
			ihspin1 = ihspin4			#os.rename(ihspin4, ihspin1)		
	elif not os.path.isfile(ihspin0):
		ihspin0 = ihspin3						#os.rename(ihspin3, ihspin0)
		ihspin1 = ihspin4						#os.rename(ihspin1, ihspin1)
		
	if os.path.isfile(ihspin0) and os.path.isfile(ihspin2):
		spinplots(ihspin0, ihspin1, ihspin2, wfdir, outdir, locate_merger)
	elif os.path.isfile(ihspin0) and not os.path.isfile(ihspin2):
		spinplots_bh12(ihspin0, ihspin1, wfdir, outdir, locate_merger)
		
		
