###############################################################################
# Script - Runview.py
# Author - Bhavesh Khamesra
# Purpose - Main code which provides all arguments and calls relevant analysis functions 
###############################################################################

from Collect_Files import *
from Trajectory import *
from Energy_momentum import Energy_Momentum
from Runstats import runstats
from Psi4 import Psi4_Plots
from Spins import Spins
from webpage import webpage 
from Mass import Mass_Plots
from StitchFiles import StitchFiles
#from RemoteSync import sync 

#from Animate_Trajectories import *

import os, argparse

#Declare arguments
parser = argparse.ArgumentParser()
parser.add_argument( "system-type", help='Define the system type. Possible options are BBH, NSBH, BNS. \n Usage: ./Runview BBH ...', type=str)
parser.add_argument("source_dir", help='Specify the location of source directory inside which all the outputs of simulation exists. \n Usage: <path to source directory>', type=str)
parser.add_argument( "output", help='Specify the location of output directory where the webpage will be created. \n Usage:<path to output directory>', type=str)
parser.add_argument("-v", "--verbosity", help='Add this option to print details about running processes.\n Usage: -v or --verbosity', action="store_true")
parser.add_argument("-s", "--sync", help='Add this option to synchronise from remote machine to local machine at source path.\n Usage: -s user@machine_address:path or --sync user@machine_address:path')
parser.add_argument( "-ah","--include_horizon", help='Add this option to include the horizon outputs in results. \n Usage: -ah or --include_horizon', action="store_true")
parser.add_argument("--stitch_data", help='Use this option to stitch multiple outputs. \n Usage: --stitch_data', action="store_true")
parser.add_argument("--find_merger", help='Use this option to track the merger and final black hole. \n Usage: --find_merger', action="store_true")
parser.add_argument("--find_qnm", help='Use this option to find the quasinormal modes using Psi4. \n Usage: --find_qnm', action="store_true")
parser.add_argument("-pd","--puncture_dynamics", help='Creates the necessary plots for puncture dynamics project. \n Usage: --puncture_dynamics', action="store_true")
#parser.add_argument( "--extra_surface", help='Specify the number of extra surfaces apart. . \n Usage:--extra_surface <number of extra surface>', type=int)
#parser.add_argument( "extra_surf", help='Specify the number of extra surfaces apart. . \n Usage:--extra_surface <number>', type=int)


args = parser.parse_args()

AHF = args.include_horizon
stitchdata = args.stitch_data
sync       = args.sync
verbose    = args.verbosity
findmerger = args.find_merger
findqnm    = args.find_qnm
#extra_surf = args.extra_surface
remotepath = args.sync
dirpath    = args.source_dir
outdir     =  args.output


#remote syncing is temporarily deactivated 
if sync==True:
     raise NameError ("Remote Sync is not supported currently. Please use rsync before using Runview to sync the data.")
#    sync(remotepath, dirpath)	

#Stitch Data
if stitchdata:
    StitchFiles(dirpath, save_hrzn=AHF) #, extra_surf=extra_surf)
    dirpath=os.path.join(dirpath,(os.path.basename(dirpath)+'-all'))

#Collect necessary files in Summary - data directory
CollectFiles(dirpath, outdir)	

Trajectory(dirpath, outdir, locate_merger=findmerger)
Energy_Momentum(dirpath, outdir)
runstats(dirpath, outdir)
Psi4_Plots(dirpath, outdir, locate_merger=findmerger, locate_qnm=findqnm)
Spins(dirpath, outdir)
Mass_Plots(dirpath, outdir)
webpage(dirpath, outdir, locate_merger=findmerger)

#if AHF:
#	animate_trajectories(dirpath, outdir)
