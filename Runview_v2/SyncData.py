from CommonFunctions import *
import subprocess as sp
import os

#To Do  - add options to exclude certain files like GW data, carpet memory and timing files or horizon data or h5 data etc. 
def download_simdata(simname, machine, userid, localpath, include_checks=False, sim_remotepath=None, additional_args=None, verbose=0):
    '''Downloads the data from the given machine. 
       simname - name of the simulation on the remote cluster 
       machine - remote machine name (comet, stampede2, cygnus, hive currently supported. Update the source code to add more.)
       userid  - Your username on remote machine
       localpath - Where you want to download the simulation data
       include_checks - Set to true if you want to download checkpoints
       sim_remotepath - Add the simulation path on remote machine/cluster if it is not the default path as in cactus machine files.
       additional_args - Can add additional flags or exclude statements (eg. --exclude=TimerReport*) ''' 
       

    usersimpath = sim_remotepath
    	
    #set machine hostname and simulation directory path - Ideal way would be to import it from simfactory machine files
    if machine == "comet":
    	hostname = 'comet.sdsc.xsede.org'
    	simpath_default = '/oasis/scratch/comet/%s/temp_project/simulations/'%userid
    elif machine == "stampede2":
    	hostname = 'stampede2.tacc.utexas.edu'
    	simpath_default = '/scratch/00507/%s/simulations/'%userid
    elif machine == "cygnus":
    	hostname = 'login-s2.pace.gatech.edu'
    	simpath_default = '/nv/hp11/%s/scratch/simulations/'%userid
    elif machine == "hive":
    	hostname = 'login-hive.pace.gatech.edu'
    	simpath_default = '/nv/hp11/%s/scratch/simulations/'%userid
    else:
	raise NameError("Machine not recognized. Please update the machine name in download_simdata function") 

    if usersimpath==None:
    	simdirpath = os.path.join(simpath_default, simname)
    else:
    	simdirpath = os.path.join(usersimpath, simname)
  
	
    #Create rsync  input  argument
    user_host = '%s@%s'%(userid, hostname)
    remote_data_location = '%s:%s'%(user_host, simdirpath)

    rsync_cmd = ["rsync", "-avtr", remote_data_location, localpath]	


    #check for any additional arguments provided by user
    if additional_args==None:

        if include_checks==True:
            info('Include checkpoints ', 3, verbose)
            info("rsync command - "+' '.join( rsync_cmd), 3, verbose)
            sp.call(rsync_cmd)
        else:
            info('Exclude checkpoints', 3, verbose)
            rsync_cmd.append("--exclude=*check*")
            info("rsync command - "+' '.join( rsync_cmd),3, verbose)
            sp.call(rsync_cmd)
    else:

        if include_checks==True:
            rsync_cmd.append(additional_args)
            info('Include checkpoints', 3, verbose)
            info("rsync command - "+' '.join( rsync_cmd), 3, verbose)
            sp.call(rsync_cmd)
        else:
            info('Exclude checkpoints', 3, verbose)
            rsync_cmd = rsync_cmd + ["--exclude=*check*", additional_args]
            info("rsync command - "+' '.join( rsync_cmd),3, verbose)
            sp.call(rsync_cmd)
	

#download_simdata('Boosted_NS_IllGRMHD_Final', 'cygnus', 'bkhamesra3', '/localdata2/bkhamesra3/TrashDirectory', include_checks=False, sim_remotepath=None, verbose=3, additional_args='--exclude=*output-0000/CCTK* --progress')	

