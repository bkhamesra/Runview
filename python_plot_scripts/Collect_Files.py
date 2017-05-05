# Create a directory with all the relevant files - ShiftTracker, IHSpin, HNMAss, YLM_22_75, Psi4Analysis, Runstat
from shutil import copyfile
import os
from CommonFunctions import debuginfo

def checkfile(wfdir, filename, pathcheck = 'Mandatory'):
	
	path = os.path.join(wfdir, filename)
	message = "%s missing in the Waveform Directory. Please Check again" %(filename)	

	if (not(os.path.isfile(path))) and (pathcheck=='Mandatory'):
		raise ValueError(message)
	elif not(os.path.isfile(path)) and pathcheck=='Optional':
		debuginfo(message)


def copy(wfdir, outdir, filename, pathcheck = 'Mandatory'):
	
	try:
		outpath = os.path.join(outdir, filename)
		filepath = os.path.join(wfdir, filename)

		if not os.path.exists(outpath):
			copyfile(filepath, outpath)
	except IOError:
		checkfile(wfdir, filename, pathcheck)


def CollectFiles(dirpath, outdir):
	
	datadir  = DataDir(dirpath, outdir)
	print("Output will be saved at - {} \n".format(datadir))
    	parfile = (dirpath.split('/')[-1]) + ('.par')
	shifttracker0 = 'ShiftTracker0.asc'
	shifttracker1 = 'ShiftTracker1.asc'
	
	ihspin0 = 'ihspin_hn_0.asc'
	ihspin1 = 'ihspin_hn_1.asc'
	ihspin3 = 'ihspin_hn_3.asc'
	ihspin4 = 'ihspin_hn_4.asc'

	psi4_ylm = ("Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc")
	psi4_Weyl = ("Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc")
	runstat = ("runstats.asc")
	
	filelist_mand = [parfile, shifttracker0, shifttracker1]
	filelist_opt = [ihspin0, ihspin1, ihspin3, ihspin4, runstat, psi4_ylm, psi4_Weyl]
	
	for mfile in filelist_mand:
		copy( dirpath,datadir, mfile )
	
	for ofile in filelist_opt:
		copy( dirpath, datadir, ofile, pathcheck = 'Optional')


#dirpath ='/nethome/numrel/datafiles/Waveforms/TP-series/D8_q3.00_a0.5_0.0_th000_ph000_m120'
#outdir = '/nethome/bkhamesra3/Desktop/testing_python'
#CollectFiles(dirpath, outdir)	
