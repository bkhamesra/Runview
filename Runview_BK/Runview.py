from Collect_Files import *
from Trajectory import *
from Energy_momentum import Energy_Momentum
from Runstats import runstats
from Psi4 import Psi4_Plots
from Spins import Spins
from webpage import webpage 
from webpage_puncdyn import webpage_pd 
from hnmass import Mass_Plots
from StitchFiles import StitchData
from PunctureDynamics import PunctureDynamics

from Animate_Trajectories import *

import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument( "system-type", help='Define the system type. Possible options are BBH, NSBH, BNS. \n Usage: ./Runview BBH ...', type=str)
parser.add_argument("source_dir", help='Specify the location of source directory inside which all the outputs of simulation exists. \n Usage: <path to source directory>', type=str)
parser.add_argument( "output", help='Specify the location of output directory where the webpage will be created. \n Usage:<path to output directory>', type=str)
parser.add_argument("-v", "--verbosity", help='Add this option to print details about running processes.\n Usage: -v or --verbosity', action="store_true")
parser.add_argument( "-ah","--include_horizon", help='Add this option to include the horizon outputs in results. \n Usage: -ah or --include_horizon', action="store_true")
parser.add_argument("--stitch_data", help='Use this option to stitch multiple outputs. \n Usage: --stitch_data', action="store_true")
parser.add_argument("--find_merger", help='Use this option to track the merger and final black hole. \n Usage: --find_merger', action="store_true")
parser.add_argument("--find_qnm", help='Use this option to find the quasinormal modes using Psi4. \n Usage: --find_qnm', action="store_true")
parser.add_argument("-pd","--puncture_dynamics", help='Creates the necessary plots for puncture dynamics project. \n Usage: --puncture_dynamics', action="store_true")


args = parser.parse_args()

AHF = args.include_horizon
stitchdata = args.stitch_data
verbose = args.verbosity
findmerger = args.find_merger
findqnm = args.find_qnm
puncdyn = args.puncture_dynamics


dirpath = args.source_dir
outdir =  args.output


if stitchdata:
	StitchData(dirpath, save_hrzn=AHF)
	dirpath=os.path.join(dirpath,(os.path.basename(dirpath)+'-all'))

CollectFiles(dirpath, outdir)	

if puncdyn==True:
	PunctureDynamics(dirpath, outdir, locate_merger=findmerger)
	runstats(dirpath, outdir)
	webpage_pd(dirpath, outdir, locate_merger=findmerger)
else:
	Trajectory(dirpath, outdir, locate_merger=findmerger)
	Energy_Momentum(dirpath, outdir, locate_merger=findmerger)
	runstats(dirpath, outdir)
	Psi4_Plots(dirpath, outdir, locate_merger=findmerger, locate_qnm=findqnm)
	Spins(dirpath, outdir, locate_merger=findmerger)
	Mass_Plots(dirpath, outdir, locate_merger=findmerger)
	webpage(dirpath, outdir, locate_merger=findmerger)

#if AHF:
#	animate_trajectories(dirpath, outdir)
