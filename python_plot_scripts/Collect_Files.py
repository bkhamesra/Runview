# Create a directory with all the relevant files - ShiftTracker, IHSpin, HNMAss, YLM_22_75, Psi4Analysis, Runstat

import os
import CommonFunctions

def checkfile(wfdir, filename, pathcheck = 'Mandatory'):
	
	path = os.path.join(wfdir, filename)
	message = "%s missing in the Waveform Directory. Please Check again" %(filename)	

	if not os.path.isfile(path) and pathcheck='Mandatory':
		raise ValueError(message)
	else if os.path.isfile(path) and pathcheck='Optional':
		debuginfo(message)


def copy(wfdir, outdir, filename, pathcheck = 'Mandatory'):
	
	try:
		outpath = os.path.join(outdir, filename)
		filepath = os.path.join(wfdir, filename)

		if not os.path.exists(path):
			copyfile(filepath, outpath)
	except IOError:
		checkfile(wfdir, filename, pathcheck)


def CollectFiles(dirpath, outdir):
	
	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outputdir, filepath)
	datadir = os.path.join(outputdir,'data')
	print("Output will be saved at - {} \n".format(datadir))

	if not os.path.exists(datadir):
		os.makedirs(datadir)

    	parfile = (dirpath.split('/')[-1]) + ('.par')
	shifttracker0 = 'ShiftTracker0.asc'
	shifttracker1 = 'ShiftTracker1.asc'
	
	ihspin0 = 'ihspin_hn_0.asc'
	ihspin1 = 'ihspin_hn_1.asc'
	ihspin3 = 'ihspin_hn_3.asc'
	ihspin4 = 'ihspin_hn_4.asc'

	psi4 = ("Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc")
	runstat = ("runstats.asc")
	
	filelist_mand = [parfile, shifttracker0, shifttracker1, psi4]
	filelist_opt = [ihspin0, ihspin1, ihspin3, ihspin4, runstat]
	
	for mfile in filelist_mand:
		copyfile( dirpath,datadir, mfile )
	
	for ofile in filelist_opt:
		copyfile( dirpath, datadir, ofil, pathcheck = 'Optional')

	
