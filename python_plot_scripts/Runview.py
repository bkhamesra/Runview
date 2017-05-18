from Collect_Files import *
from Trajectory import *
from Energy_momentum import Energy_Momentum
from Runstats import runstats
from Psi4 import Psi4_Plots
from Spins import Spins
from webpage import webpage 

dirpath ='/nethome/numrel/datafiles/Waveforms/NG-series/D10_q4.00_a0.8_th174_Q20'
#outdir = '/Users/Bhavesh/Documents/ResearchWork/Runview/testing_python'
outdir = '/nethome/bkhamesra3/Desktop/testing_python'
#CollectFiles(dirpath, outdir)	
#Trajectory(dirpath, outdir)
#Energy_Momentum(dirpath, outdir)
#runstats(dirpath, outdir)
#Psi4_Plots(dirpath, outdir)
#Spins(dirpath, outdir)
webpage(dirpath, outdir)
