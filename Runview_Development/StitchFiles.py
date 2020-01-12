import numpy as np
import os, glob
import multiprocessing as mp
from functools import partial
from shutil import copy, copyfile 

#Copy Files - parfiles, stdout, horizon data, h5
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


#Stitch files from multiple outputs into single file
def CombineData(wfdir, outdir, filename, time_clm):
	
	hdr = '############################################################################### \n'
	filepath = sorted(glob.glob(os.path.join(wfdir,  ('output-0???/*/' + filename))))
	

	print('\n Stiching File: {} \n '.format(filename))
	for files in (filepath):
	
	    print ("Output - {}".format('out'+files.split('out')[-1]))	#(os.path.basename(files)))
	    # Add the header in the beginning of the file
	    if files==filepath[0]: 
		for line in open(files,'r'):
		    if line[0]=='#':
		        hdr = hdr+ line

	    # Read and save the data from multiple outputs checking repeatition of time
	    if files==filepath[0]:
		data = np.genfromtxt(files)
		time = data[:,time_clm]
		data_save = data
	    else:
		data = np.genfromtxt(files)
		idx = np.where(data[:,time_clm]>t)
		data = data[idx]
		time = np.append(time, data[:,time_clm])
		data_save = np.vstack((data_save, data))
	   
	    if time.size>1:
		t = np.unique(time)[-1]
	    else:
		t = time

	if np.size(filepath)>0:
	    shtr_output = open(os.path.join(outdir, '%s'%filename),'w')
	    np.savetxt(shtr_output, data_save, header=hdr, delimiter='\t', newline='\n')
	    shtr_output.close()
	else:
	    print("File not found. \n")

def StitchData(wfdir, save_hrzn=True, extra_surf=0):

	combine_dir = os.path.join(wfdir,(os.path.basename(wfdir)+"-all"))
	if not os.path.exists(combine_dir):
		os.mkdir(combine_dir)
	
        #Copy Parfile and Standard Output
	parfile = os.path.join(wfdir, 'output-0???/*/*.par')
	stdout = os.path.join(wfdir, 'output-0???/stdout')
	
	copyfiles(parfile, combine_dir, parfile=True)
	copyfiles(stdout, combine_dir)
	
	#Copy Position data
	CombineData(wfdir, combine_dir, 'ShiftTracker0.asc', 1)
	CombineData(wfdir, combine_dir, 'ShiftTracker1.asc', 1)
	CombineData(wfdir, combine_dir, 'puncturetracker-pt_loc..asc', 1)
	CombineData(wfdir, combine_dir, 'ProperDistance.asc', 1)	

	#Copy Runstats
	CombineData(wfdir, combine_dir, 'runstats.asc', 1)

	
	#results = [pool.apply(CombineData(wfdir, combine_dir, 'ihspin_hn_%d.asc'%i) for i in range(9)]
	
	#Copy Spin Data
	for k in range(9):
	    if not glob.glob(wfdir+'/output-0???/*/ihspin_hn_%d.asc'%k)==[]:
		CombineData(wfdir, combine_dir, 'ihspin_hn_%d.asc'%k,0)
	
	#Copy horizon data and BH diagnostics file
	for k in np.array((0,1,2,3,4,5,6,7,8)):
	    if not glob.glob(wfdir+'/output-0???/*/BH_diagnostics.ah%d.gp'%k)==[]:
	 	CombineData(wfdir, combine_dir, 'BH_diagnostics.ah%d.gp'%k,1)

	if save_hrzn==True:
		print("SitchData: Collecting horizon files. ")
		hrzn_files = os.path.join(wfdir, 'output-0???/*/h.*.ah*')
		hrzn_data = glob.glob(os.path.join(wfdir, 'output-0???/*/h.*.ah*'))
		if not(len(hrzn_data)>1):
		    hrzn_data = glob.glob(os.path.join(wfdir, 'output-0???/*/*/h.*.ah*'))

		dest = os.path.join(combine_dir,"HorizonData")
		if not os.path.exists(dest):
			os.mkdir(dest)
		for files in sorted(hrzn_data):
			copyfile(files, os.path.join(dest, os.path.basename(files)))


	#Copy Psi4 data
	for f in glob.glob(wfdir+'/output-0000/*/Ylm_*'):
		filename = (os.path.basename(f))
		CombineData(wfdir, combine_dir, filename, 0)

	for f in glob.glob(wfdir+'/output-0000/*/psi4analysis*'):
		filename = os.path.basename(f)
		CombineData(wfdir, combine_dir, filename, 0)
	
	#Copy quasi local measures data - Currently not working, files cannot be found due to some bug, fix this!
	quasilocalmeasures = []
        extra_hrzn = extra_surf
	for i in range(3+extra_surf):
	    qlm = ['qlm_3det[%d]..asc'%i, 'qlm_3det[%d].x.asc'%i,'qlm_adm_angular_momentum_x[%d]..asc'%i, 'qlm_adm_angular_momentum_y[%d]..asc'%i, 'qlm_adm_angular_momentum_z[%d]..asc'%i, 'qlm_adm_energy[%d]..asc'%i , 'qlm_adm_momentum_x[%d]..asc'%i, 'qlm_adm_momentum_y[%d]..asc'%i, 'qlm_adm_momentum_z[%d]..asc'%i  ]
	    quasilocalmeasures = quasilocalmeasures + qlm

	for filename in quasilocalmeasures:
	    CombineData(wfdir, combine_dir, filename, 8)
	
	#pool = mp.Pool(processes=8)
	#Copy  Lapse, Shift, K etc
        #lapse_shift = ['alpha.d.asc', 'alpha.x.asc', 'alpha.y.asc', 'alpha.z.asc', 'betax.d.asc', 'betax.x.asc', 'betax.y.asc', 'betax.z.asc','betay.d.asc', 'betay.x.asc', 'betay.y.asc', 'betay.z.asc', 'betaz.d.asc', 'betaz.x.asc', 'betaz.y.asc', 'betaz.z.asc']


	#results = [pool.apply(CombineData, args=(wfdir, combine_dir, f, 8,)) for f in lapse_shift]
	#for f in lapse_shift:
	#	CombineData(wfdir, combine_dir, f, 8)
	#
	#data_2d = os.path.join(combine_dir, "HDF5_2D")
	#data_3d = os.path.join(combine_dir, "HDF5_3D")
	#
	#if os.path.exists(data_2d)==False:
	#    os.mkdir(data_2d)

	#if os.path.exists(data_3d)==False:
	#    os.mkdir(data_3d)


		
