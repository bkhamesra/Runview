###############################################################################
# Script - CollectFiles.py
# Author - Bhavesh Khamesra
# Purpose -  Create a Summary Data Directory with collection of relevant files. 
###############################################################################


from shutil import copy
import os
from CommonFunctions import *
import glob


def checkfile(wfdir, filename, pathcheck = 'Mandatory'):
    '''Check if file is present in Stitched output directory/waveform directoryi
	
       ----------------- Parameters -------------------
       wfdir (type 'String') - path of directory which has the relevant files
       filename (type 'String') - Name of file to be checked
       pathcheck (type 'String') - 'Mandatory' if filel is definitely required else 'Optional' 
    '''		
    
    path = os.path.join(wfdir, filename)
    message = "%s missing in the Waveform Directory. Please Check again" %(filename)	
    
    if (not(os.path.isfile(path))) and (pathcheck=='Mandatory'):
        raise ValueError(message)
    elif not(os.path.isfile(path)) and pathcheck=='Optional':
        debuginfo(message)


def copyfiles(wfdir, outdir, filename, pathcheck = 'Mandatory'):
    '''Copy files from stitched output directory to Summary Data directory

       ----------------- Parameters -------------------
       wfdir (type 'String') - path of directory with relevant files (stitched files directory)
       outdir (type 'String') - path where you wish to copy the files
       filename (type 'String') - Name of file to be checked
       pathcheck (type 'String') - 'Mandatory' if filel is definitely required else 'Optional' 
    '''			
    
    try:
        outpath = os.path.join(outdir, filename)
        filepath = os.path.join(wfdir, filename)
    
        #if not os.path.exists(outpath):
        copy(filepath, outpath)
    except IOError:
        checkfile(wfdir, filename, pathcheck)


def CollectFiles(dirpath, outdir):
    ''' Collect Relevant files from Stitched output directory to Summary Data directory

       ----------------- Parameters -------------------
       dirpath - Path of waveform/simulation directory
       outdir - Path of Final Summary directory'''

    datadir  = DataDir(dirpath, outdir)
    print("Relevant data will be saved at - {} \n".format(datadir))
    parfile = os.path.basename(sorted(glob.glob(os.path.join(dirpath, '*.par')))[0])		#(dirpath.split('/')[-1]) + ('-1.par')
    shifttracker0 = 'ShiftTracker0.asc'
    shifttracker1 = 'ShiftTracker1.asc'
    propdist = 'ProperDistance.asc'
    
    ihspin0 = 'ihspin_hn_0.asc'
    ihspin1 = 'ihspin_hn_1.asc'
    ihspin2 = 'ihspin_hn_2.asc'
    ihspin3 = 'ihspin_hn_3.asc'
    ihspin4 = 'ihspin_hn_4.asc'
    
    psi4_ylm = "Ylm_WEYLSCAL4::Psi4_l2_m2_r80.00.asc"
    psi4r_ylm = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r80.00.asc"
    psi4_mp = "mp_WeylScal4::Psi4i_l2_m2_r80.00.asc"
    psi4_anal = "psi4analysis_r80.00.asc"
    runstat = "runstats.asc"
    bhdiag1 = "BH_diagnostics.ah1.gp"
    bhdiag2 = "BH_diagnostics.ah2.gp"
    bhdiag3 = "BH_diagnostics.ah3.gp"
    bhdiag4 = "BH_diagnostics.ah4.gp"
    bhdiag5 = "BH_diagnostics.ah5.gp"
    bhdiag6 = "BH_diagnostics.ah6.gp"
    filelist_mand = [parfile, shifttracker0, shifttracker1]
    filelist_opt = [ihspin0, ihspin1, ihspin2, ihspin3, ihspin4, runstat, psi4_ylm,psi4r_ylm,psi4_mp, psi4_anal, bhdiag1, bhdiag2, bhdiag3, bhdiag4, bhdiag5, bhdiag6, propdist]
    
    for mfile in filelist_mand:
    	copyfiles( dirpath,datadir, mfile )
    
    for ofile in filelist_opt:
    	copyfiles( dirpath, datadir, ofile, pathcheck = 'Optional')
    
    qlm_files = glob.glob(os.path.join(dirpath, 'qlm_*'))
    
    for files in qlm_files:
        f = files.split('/')[-1]	
        copyfiles(dirpath, datadir, f, pathcheck = 'Optional')
