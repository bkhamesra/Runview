import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import  matplotlib.pyplot as plt
import os,glob
from CommonFunctions import *

def runstats(wfdir, outdir):

  	statfigdir,dynfigdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
	
	
	runstat_file = os.path.join(datadir, 'runstats.asc')
	if not os.path.exists(runstat_file):
		debuginfo("%s file not found" %runstat_file)
		return

	runstat = open(runstat_file)
	
	
	iteration,coord_time, walltime, speed, period, cputime = np.loadtxt(runstat, unpack=True, usecols=(0,1,2,3,4,5))
	
	walltime_hrs = walltime/3600.
	day = 0	
	for i in range(len(walltime_hrs)):
		if walltime_hrs[i]<walltime_hrs[i-1]:
			day=day+walltime_hrs[i-1]/24.


	avg_speed = np.mean(speed)
	cputime_total = cputime[-1]

	plot1(coord_time, speed, 'Time (M)', 'Speed (M/hour)', 'Runstats', statfigdir)
	plystatplot = plyplot1(coord_time, speed, 'Time (M)', 'Speed (M/hour)', 'Runstats') #see common functions for details, RU
	py.plot(plystatplot, filename=dynfigdir+"Runstats.html") #standard plot, object + path/filename

outDirSO = "/home/rudall/Runview/TestCase/OutputDirectory/SOetc_2/"
binSO = "/home/rudall/Runview/TestCase/BBH/SO_D9_q1.5_th2_135_ph1_90_m140/"
binQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_l11_M192-all/"
outDirQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_2/"

runstats(binQC,outDirQC)