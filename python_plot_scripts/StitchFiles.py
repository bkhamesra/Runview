import numpy as np
import os
import glob
from shutil import copy, copyfile 

def copyfiles(filepath, dest, parfile=False):
	i=1
	for files in glob.glob(filepath):
		destfile = os.path.join(dest,os.path.basename(files))
		if parfile==False:
			destfile = destfile+'-%d'%i
			copyfile(files, destfile)
		else:
			destfile = destfile.split('.par')[0]+'-%d'%i +'.par'	
			copyfile(files,destfile)
		
		i = i+1



def CombineData(wfdir, outdir, filename, time_clm):
	
	hdr = '############################################################################### \n'
	filepath = sorted(glob.glob(wfdir + '/output-0???/*/' + filename))
	for files in filepath:
	    print ("Stitching file - {}".format(os.path.basename(files)))
	    if files==filepath[0]:
		data = np.genfromtxt(files)
		time = data[:,time_clm]
		data_save = data
		for line in open(files,'r'):
			if line[0]=='#':
				hdr = hdr+ line
	    else:
		data = np.genfromtxt(files)
		idx = np.where(data[:,time_clm]>time[-1])
		data = data[idx]
		time = np.append(time, data[:,time_clm])
		data_save = np.vstack((data_save, data))

	try:
		shtr_output = open(os.path.join(outdir, '%s'%filename),'w')
		np.savetxt(shtr_output, data_save, header=hdr, delimiter='\t', newline='\n')
		shtr_output.close()
	except NameError:
		return

def StitchData(wfdir, save_hrzn=True):

	combine_dir = os.path.join(wfdir,(os.path.basename(wfdir)+"-all"))
	if not os.path.exists(combine_dir):
		os.mkdir(combine_dir)
	
	parfile = os.path.join(wfdir, 'output-0???/*/*.par')
	stdout = os.path.join(wfdir, 'output-0???/stdout')
	
	copyfiles(parfile, combine_dir, parfile=True)
	copyfiles(stdout, combine_dir)
	
	if save_hrzn==True:
		hrzn_files = os.path.join(wfdir, 'output-0???/*/h.*.ah*')
		dest = os.path.join(combine_dir,"HorizonData")
		if not os.path.exists(dest):
			os.mkdir(dest)
		for files in sorted(glob.glob(hrzn_files)):
			copyfile(files, os.path.join(dest, os.path.basename(files)))



	CombineData(wfdir, combine_dir, 'ShiftTracker0.asc', 1)
	CombineData(wfdir, combine_dir, 'ShiftTracker1.asc', 1)
	CombineData(wfdir, combine_dir, 'runstats.asc', 1)
	CombineData(wfdir, combine_dir, 'BH_diagnostics.ah1.gp', 1)
	CombineData(wfdir, combine_dir, 'BH_diagnostics.ah2.gp', 1)
	CombineData(wfdir, combine_dir, 'BH_diagnostics.ah3.gp', 1)	


	for k in np.array((0,1,3,4)):
		if not glob.glob(wfdir+'/output-000?/*/ihspin_hn_%d.asc'%k)==[]:
			CombineData(wfdir, combine_dir, 'ihspin_hn_%d.asc'%k,0)
	
	for f in glob.glob(wfdir+'/output-0000/*/Ylm_*'):
		filename = (os.path.basename(f))
		CombineData(wfdir, combine_dir, filename, 0)

	for f in glob.glob(wfdir+'/output-0000/*/psi4analysis*'):
		filename = os.path.basename(f)
		CombineData(wfdir, combine_dir, filename, 0)
