#ihspin0 and ihpsin3 both have the spin data but they differ in numbers and total time. Guess is one of them is using AHF while other is using sphererad (approx). So would be a good idea to check which file has more data. 


import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
from Psi4 import maxamp_time
#rc('font', **{'family':'serif','serif':['Computer Modern']})
#rc('text', usetex=True)
#mpl.rcParams['lines.linewidth']=2



def spinplots(file1, file2, statfigdir, dynfigdir):
	ihspin_0 = open(file1)
	ihspin_1 = open(file2)
	time_bh1, sx1, sy1, sz1 = np.loadtxt(ihspin_0, unpack=True, usecols=(0,1,2,3))
	time_bh2, sx2, sy2, sz2 = np.loadtxt(ihspin_1, unpack=True, usecols=(0,1,2,3))
	
	s1 = np.sqrt(sx1**2. + sy1**2. + sz1**2.)
	s2 = np.sqrt(sx2**2. + sy2**2. + sz2**2.)
	
	sxplot = plot2(time_bh1, sx1, time_bh2, sx2, 'Time', r'$S_x$', 'Spinx', statfigdir)
	syplot = plot2(time_bh1, sy1, time_bh2, sy2, 'Time', r'$S_y$', 'Spiny', statfigdir)
	szplot = plot2(time_bh1, sz1, time_bh2, sz2, 'Time', r'$S_z$', 'Spinz', statfigdir)
	smag_plot = plot2(time_bh1, s1, time_bh2, s2, 'Time', 'mag(S)', 'Spinmag', statfigdir)
	plysxplot = plyplot2(time_bh1, sx1, time_bh2, sx2, "BH1", "BH2",'Time', r'$S_x$', 'Spinx.html', locate_merger=locate_merger, time_hrzn=t_hrzn3, time_maxamp=time_maxamp, idx_hrzn=hrzn_idx, idx_maxamp=maxamp_idx)#see common functions for details, RU
	plysyplot = plyplot2(time_bh1, sy1, time_bh2, sy2,"BH1", "BH2",'Time', r'$S_y$', 'Spiny.html', locate_merger=locate_merger, time_hrzn=t_hrzn3, time_maxamp=time_maxamp, idx_hrzn=hrzn_idx, idx_maxamp=maxamp_idx)
	plyszplot = plyplot2(time_bh1, sz1, time_bh2, sz2, "BH1", "BH2", 'Time', r'$S_z$', 'Spinz.html', locate_merger=locate_merger, time_hrzn=t_hrzn3, time_maxamp=time_maxamp, idx_hrzn=hrzn_idx, idx_maxamp=maxamp_idx)
	plysmag_plot = plyplot2(time_bh1, s1, time_bh2, s2, "BH1", "BH2", 'Time', 'mag(S)', 'Spinmag.html', locate_merger=locate_merger, time_hrzn=t_hrzn3, time_maxamp=time_maxamp, idx_hrzn=hrzn_idx, idx_maxamp=maxamp_idx)
	py.plot(plysxplot,filename = dynfigdir + "Spinx") #standard plot function, object + path/filename
	py.plot(plysyplot,filename = dynfigdir + "Spiny")
	py.plot(plyszplot,filename = dynfigdir + "Spinz")
	py.plot(plysmag_plot,filename = dynfigdir + "Spinmag")
	
def Spins(wfdir, outdir,locate_merger=False):

	statfigdir, dynfigdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
  	#Horizon Location
	
	ihspin0 = os.path.join(datadir,"ihspin_hn_0.asc")
	ihspin1 = os.path.join(datadir,"ihspin_hn_1.asc")
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
	
	print(time1)
	
	if locate_merger==True:
		bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
		t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
		maxamp, t_maxamp = maxamp_time(wfdir, outdir)	
		t_maxamp = t_maxamp-75.		
		hrzn_idx = np.amin(np.where(time1>=t_hrzn3))
		maxamp_idx = np.amin(np.where(time1>=t_maxamp))
		time_maxamp=t_maxamp[0]

		time_arr = np.around(np.array((t_hrzn3, t_maxamp)),2)
		print("Final Horizon Detected at %f and Max Amplitude at %f"%(t_hrzn3, t_maxamp))
 
   
	
	if os.path.isfile(ihspin0):
		spinplots(ihspin0, ihspin1, statfigdir, dynfigdir)
		

outDirSO = "/home/rudall/Runview/TestCase/OutputDirectory/SOetc_2/"
binSO = "/home/rudall/Runview/TestCase/BBH/SO_D9_q1.5_th2_135_ph1_90_m140/"
binQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_l11_M192-all/"
outDirQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_2/"

Spins(binSO, outDirSO, locate_merger=True)