from Collect_Files import *
from Trajectory import *
from Energy_momentum import Energy_Momentum
from Runstats import runstats
from Psi4 import Psi4_Plots



dirpath ='/nethome/numrel/datafiles/Waveforms/NG-series/D09_q1.00_a0.35_Q20'
outdir = '/Users/Bhavesh/Documents/ResearchWork/Runview/testing_python'
#CollectFiles(dirpath, outdir)	
#Trajectory(dirpath, outdir)
#Energy_Momentum(dirpath, outdir)
runstats(dirpath, outdir)
Psi4_Plots(dirpath, outdir)

