###############################################################################
# Script - StitchFiles.py
# Author - Bhavesh Khamesra
# Purpose - Stitch data from multiple outputs of simulation 
###############################################################################


import numpy as np
import os
import glob
from shutil import copy, copyfile 

def copyfiles(filepath, dest, parfile=False):
    '''Copy the relevant files
    ----------Parameters-----------
    filepath - path of files
    dest - path of destination directory
    '''
    i=0
    for files in sorted(glob.glob(filepath)):
        destfile = os.path.join(dest,os.path.basename(files))
        if parfile==False:
            if os.path.basename(destfile) =='stdout':
                destfile = os.path.join(dest,'stdout-%d'%i)
            elif destfile[-3:] =='out':
                destfile = os.path.join(dest,'stdout-%d'%i)
            else:
                destfile = destfile+'-%d'%i
  
            copyfile(files, destfile)
        else:
            destfile = destfile.split('.par')[0]+'-%d'%i +'.par'    
            copyfile(files,destfile)
        
        i = i+1



def StitchData(wfdir, outdir, filename, time_clm):
    

    '''Stitch 0d and 1d datafiles from multiple outputs

    ----------------- Parameters -------------------
    wfdir (type 'String') - path of Directory with relevant files
    outdir (type 'String') - path of Output Summary directory
    '''		
    hdr = '############################################################################### \n'
    filepath = sorted(glob.glob(wfdir + '/output-0???/*/' + filename))

    if len(filepath)==0: return

    with open(filepath[0], 'r') as hdr_file:
        for line in hdr_file:
            if line[0]=='#':
                hdr = hdr+ line

    for files in filepath:
        print ("Stitching file - {}".format(os.path.basename(files)))
        if files==filepath[0]:
            data = np.genfromtxt(files)
            time = data[:,time_clm]
            data_save = data
        
        else:
            data = np.genfromtxt(files)
            if data.ndim==1:
                 data = np.reshape(data,(1, len(data)))

            idx = np.where(data[:,time_clm]>time[-1])
            if len(idx)>0:
               	data = data[idx]
                time = np.append(time, data[:,time_clm])
                data_save = np.vstack((data_save, data))
		

    try:
        if len(filepath)>0:
            shtr_output = open(os.path.join(outdir, '%s'%filename),'w')
            np.savetxt(shtr_output, data_save, header=hdr, delimiter='\t', newline='\n')
            shtr_output.close()
    except (NameError, IndexError) as error:
        print(error)
        return

def StitchFiles(wfdir, save_hrzn=True):
    
    combine_dir = os.path.join(wfdir,(os.path.basename(wfdir)+"-all"))
    if not os.path.exists(combine_dir):
        os.mkdir(combine_dir)
    
    parfile = os.path.join(wfdir, 'output-0???/*/*.par')
    stdout = os.path.join(wfdir, 'output-0???/*out')
    
    copyfiles(parfile, combine_dir, parfile=True)
    copyfiles(stdout, combine_dir)
    
    if save_hrzn==True:
        print("SitchData: Collecting horizon files. ")
        hrzn_files = os.path.join(wfdir, 'output-0???/*/h.*.ah*')
        dest = os.path.join(combine_dir,"HorizonData")
        if not os.path.exists(dest):
            os.mkdir(dest)
        for files in sorted(glob.glob(hrzn_files)):
            copyfile(files, os.path.join(dest, os.path.basename(files)))



    StitchData(wfdir, combine_dir, 'ShiftTracker0.asc', 1)
    StitchData(wfdir, combine_dir, 'ShiftTracker1.asc', 1)
    StitchData(wfdir, combine_dir, 'ProperDistance.asc', 1)    
    StitchData(wfdir, combine_dir, 'runstats.asc', 1)
    StitchData(wfdir, combine_dir, 'BH_diagnostics.ah1.gp', 1)
    StitchData(wfdir, combine_dir, 'BH_diagnostics.ah2.gp', 1)
    StitchData(wfdir, combine_dir, 'BH_diagnostics.ah3.gp', 1) 
    StitchData(wfdir, combine_dir, 'BH_diagnostics.ah4.gp', 1) 
    StitchData(wfdir, combine_dir, 'BH_diagnostics.ah5.gp', 1) 

    for k in np.array((1,2,3,4,5,6)):
        if not glob.glob(wfdir+'/output-0???/*/BH_diagnostics.ah%d.gp'%k)==[]:
            StitchData(wfdir, combine_dir, 'ihspin_hn_%d.asc'%k,0)

    for k in np.array((0,1,2,3,4)):
        if not glob.glob(wfdir+'/output-0???/*/ihspin_hn_%d.asc'%k)==[]:
            StitchData(wfdir, combine_dir, 'ihspin_hn_%d.asc'%k,0)
    
    for f in glob.glob(wfdir+'/output-0000/*/Ylm_*'):
        filename = (os.path.basename(f))
        StitchData(wfdir, combine_dir, filename, 0)

    for f in glob.glob(wfdir+'/output-0000/*/psi4analysis*'):
        filename = os.path.basename(f)
        StitchData(wfdir, combine_dir, filename, 0)
