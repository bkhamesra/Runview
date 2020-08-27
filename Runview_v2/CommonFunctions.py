import numpy as np
import os
import glob
from shutil import copy, copyfile
from colorama import Fore, Back, Style 

import matplotlib as mpl
mpl.rc('lines', linewidth=2, color='r')
mpl.rc('font', size=16)
mpl.rc('axes', labelsize=16, grid=True)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('legend', fontsize=14)
'''############################ To Do #####################################
 
-> Add function to convert from geometrical units to physical units

############################### To Do #####################################'''
 

def info_settings(msg, info_lvl):
	'''Prints the message based on the requested verbose output.
	   info_lvl:1 - only warnings and Errors 
	   	    2 - Success messages. 
	   	    3 - Summary messages. 
 	   	    4 - debugging mode. Print all the information.
		    0 - no output.''' 	
    
	if info_lvl==1:
		print(Fore.RED+msg, '\n')
	elif info_lvl==2:
		print(Fore.GREEN+msg, '\n')
	elif info_lvl>=3:
		print(Fore,BLUE+msg, '\n')
	elif info_lvl>=4:
		print(msg, '\n')
	

def Info(msg, info_lvl, info_lvl_requested):
	if info_lvl_requested==info_lvl:
		info_settings(msg, info_lvl_requested)

def DataDirectory(simpath):
	return os.path.join(simpath, 'Summary/Data')
	
def FiguresDirectory(simpath):
	return os.path.join(simpath, 'Summary/Figures')
	
def CopyFiles(simulation_path, filename, destination_path=None, numerize=False, verbose=0):
	"""Copy the relevant files from all the output
	
	 Parameters
	------------
	simulation_path - str
	    path of simulation directory with all the outputs
	filename - str
	    Filename which needs to be copied, can use wildcards (for. eg. Horizon data can be specified as h.t*.ah1.gp)
	destination_path - str/None
	    Where the files will be copied. If set to None, Files from all the outputs will be copied to <simpath>/Summary/Data
   	numerize - Bool
	    Add numbers of corresponding outputs from which file is copied
	verbose - int (0-4)
	    Print information about the process (0 - no information, 1 - Basic information, 2 - Used for debugging)
	
	
	To Do
	-----------------
	Add the verbose functionality to this function
	"""

	# Check for destination directory and create one if missing
	if destination_path==None:
		destination_path = os.path.join(simulation_path,"Summary/Data")
		
	info("Copyfiles >> %s Files will be copied to %s path"%(%filename, destination_path),3, verbose)	
	if not os.path.exists(destination_path):
		os.makedirs(destination_path)
		info("Copyfiles >> Destination directory not found. Will be created on the go",1, verbose)

	# Sort all outputs and collect the files
	output_path = sorted(glob.glob(os.path.join(simulation_path, 'output-????')))
	
	for output_dir in output_path:

		which_output = os.path.basename(output_dir)
		output_num = int(which_output.split('-')[:-4])
		
		# Standard output and error files are just inside output directories while other files are one or more level further deep in general (hence use recursive search). 
		if (filename[-3:]=="out") or (filename[-3:]=="err");
			filepath = os.path.join(output_dir, filename)
		else:
			filepath = os.path.join(output_dir, "*/%s"%filename, recursive=True)

		filelist = sorted(glob.glob(filepath))
		
		for files in filelist:
			destfile = os.path.join(dest,os.path.basename(files))
			if numerize==False:
				copyfile(files, destfile)
			else:
				if '.' in os.path.basename(files):
					file_extension = (files.split('.'))[-1]
					destfile = destfile.split('.%s'%file_extension)[0]+'-%d'%output_num +'.%s'%file_extension	
				else:
					destfile = destfile+'-%d'%output_num 

				copyfile(files,destfile)
		
			



def Combine_0D_Data(filename, simulation_path, destination_path, time_clm):
	
	"""Copy the data from the files from all the output for 0D Data

	Assumption - All files located inside simulation_path/output-????/*/	
	Parameters
	------------
	filename - str
		Filename which needs to be copied, can use wildcards (for. eg. Horizon data can be specified as h.t*.ah1.gp)
	simulation_path - str
	    path of simulation directory with all the outputs
	destination_path - str/None
		Where the files will be copied. If set to None, Data from all the outputs will be copied to <simpath>/Summary/Data/filename

	"""
	
	# Check for destination directory and create one if missing
	if destination_path==None:
		destination_path = os.path.join(simulation_path,"Summary/Data")
		
	info("Copyfiles >> %s Files will be copied to %s path"%(filename, destination_path),3, verbose)	
	if not os.path.exists(destination_path):
		os.makedirs(destination_path)
		info("Copyfiles >> Destination directory not found. Will be created on the go",1, verbose)

	hdr = '############################################################################### \n'
	filepath = sorted(glob.glob(simulation_path + '/output-????/*/' + filename, recursive=True))
	
	#Check if file exists in the outputs
	if len(filepath)<1:
		info("Combine_0D_Data >> %s file not found in the output"%filename, 1, verbose)
		return

	# Copy the relevant data from the output
	for files in filepath:
		info("Stitching file - {}".format(os.path.basename(files)), 3, verbose)
	    if files==filepath[0]:
			data = np.genfromtxt(files)
			time = data[:,time_clm]
			data_save = data
			for line in open(files,'r'):
				if line[0]=='#':
					hdr = hdr+ line
	    else:
			data = np.genfromtxt(files)
			#Find the index in the last output from where new output starts and crop
			#the last output data. This is preferred over cropping new output, since,
			#the last output may not incorporate any changes made in thorn parameters in the parfile
			idx = np.amax(np.where(time<data[0,time_clm]))
			data_save = data_save[:idx]
			time = np.append(time[:idx], data[:,time_clm])
			data_save = np.vstack((data_save, data))

	try:
	    file_output = open(os.path.join(destination_path, '%s'%filename),'w')
	    np.savetxt(file_output, data_save, header=hdr, delimiter='\t', newline='\n')
	    file_output.close()
	except NameError:
		return

def Combine_1D_Data(filename, simulation_path, destination_path, time_clm):
	
	"""Copy the data from the files from all the output for 1D Data

	Assumption - All files located inside simulation_path/output-????/*/	
	 Parameters
	------------
	filename - str
		Filename which needs to be copied, can use wildcards (for. eg. Horizon data can be specified as h.t*.ah1.gp)
	simulation_path - str
	    path of simulation directory with all the outputs
	destination_path - str/None
		Where the files will be copied. If set to None, Data from all the outputs will be copied to <simpath>/Summary/Data/filename

	"""
	
	#Define function to extract the data from one single output till the specified time
	def write_single_output(readfile, writefile, time):	
		with open(readfile,'r') as datafile:	
			for line in datafile.readlines():
				if line[0]=='#':
					outfile.write(line)
				elif (line.split('\t'))[time_clm]>=time:
					break
				else:
					outfile.write(line)

	
	# Check for destination directory and create one if missing
	if destination_path==None:
		destination_path = os.path.join(simulation_path,"Summary/Data")
		
	info("Copyfiles >> %s Files will be copied to %s path"%(%filename, destination_path),3, verbose)	
	if not os.path.exists(destination_path):
		os.makedirs(destination_path)
		info("Copyfiles >> Destination directory not found. Will be created on the go",1, verbose)

	hdr = '############################################################################### \n'
	filepath = sorted(glob.glob(simulation_path + '/output-????/*/' + filename, recursive=True))
	
	# Check if file exists in the outputs
	if len(filepath)<1:
		info("Combine_1D_Data >> %s file not found in the output"%filename, 1, verbose)
		return

	outfile = open(os.path.join(destination_path, filename), 'a')
	outfile.truncate(0)
		
	for filenum in range(len(filepath)):
		info("Stitching file - {}".format(os.path.basename(files), 3, verbose)
			
		if filenum<len(filepath)-1:
			time_next = np.genfromtxt(filepath[filenum+1], unpack=True, usecols=time_clm)[0]
			# Extract data from current output till the initial time from next output
			write_single_output(filepath[filenum], outfile, time_next)	
		else:
			with open(readfile,'r') as datafile:
				outfile.write(datafile.read)

	outfile.close()


def StylizePlot(ax, xlabel, ylabel, xlim=None, ylim=None)
	"""Add basic properties to plot

	Parameters
	------------------
	ax - plot to stylize (matplotlib.pyplot.plot object)
	xlabel - x axis label
	ylabel - y axis label
	xlim - array/list of min and max limits on x axis ([xmin, xmax])
	ylim - array/list of min and max limits on y axis ([ymin, ymax])"""

	ax.set_xlabel(r"$%s$"%xlabel)
	ax.set_ylabel(r"$%s$"%ylabel)
	if xlim!=None: ax.set_xlim(xlim[0], xlim[1])
	if ylim!=None: ax.set_ylim(ylim[0], ylim[1])
	ax.grid(True)
	ax.legend()
	return ax
