################# Trajectory Class and Functions ####################

# This File contains necessary class and functions to analyze and visualize the trajectory data of the compact object
#
# To Do - 
# 1. Add acceleration definition in spherical polar coordinates
# 2. Can add more plots to ShiftTracker
# 3. Can add eccentricity computation 



import os, glob
import numpy as np
from CommonClasses import Vector
import matplotlib as mpl
mpl.rc('lines', linewidth=2, color='r')
mpl.rc('font', size=16)
mpl.rc('axes', labelsize=18, grid=True)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import matplotlib.pyplot as plt



def func_RadialVelocity(pos, vel):
    """"Compute the radial velocity - d/dt (r)

      Params
    --------------
    pos: Vector type object
         position/displacement in cartesian coordinates
    vel: Vector type object
         velocity in cartesian coordinates"""

    return (vel.x*np.cos(pos.phi) + vel.y*np.sin(pos.phi))*np.sin(pos.theta) + vel.z*np.cos(theta)


def func_PhaseVelocity(pos, vel):
    """"Compute the phase velocity - d/dt (phi)

      Params
    --------------
    pos: Vector type object
         position/displacement in cartesian coordinates
    vel: Vector type object
         velocity in cartesian coordinates"""

    return np.divide((vel.y*np.cos(pos.phi) - vel.x*np.sin(pos.phi)), (pos.magnitude*np.sin(theta)))


def func_ThetaVelocity(pos, vel):
    """"Compute the theta velocity - d/dt (theta)

      Params
    --------------
    pos: Vector type object
         position/displacement in cartesian coordinates
    vel: Vector type object
         velocity in cartesian coordinates"""

    theta_dot =  np.cos(pos.theta)*np.cos(pos.phi)*vel.x + np.cos(pos.theta)*np.sin(pos.phi)*vel.y  - np.sin(pos.theta)*(vel.z)
    return theta_dot/pos.magnitude


class ShiftTracker():

    """ Shift Tracker class to extract the BH trajectory data from ShiftTracker Files;
	
	 Parameters
	------------
	INPUT - Filename (ShiftTracker%d.asc), Simulation Path, Object Number 
	Output - File location directory, Filename, position, velocity, acceleration Vectors"""


    def __init__(self, simpath, filename, object_number=1):

	#File name and path
        self.filename = filename
	self.simpath = simpath
	self.object_number = object_number
        self.filepath = os.path.join(DataDirectory(simpath), self.filename)
    	
	#Stitch Data if necessary
	if not os.path.exists(self.filepath):
	    self.StitchFile()
	
			
        assert os.path.exists(filepath), (Fore.RED+'File %s does not exists.'%(self.filename))
       
	#Load the data 
        self.it, self.t, self._x, self._y, self._z, self._vx, self._vy, self._vz, self._ax,\
        self._ay, self._az   = np.loadtxt(file, unpack=True, usecols=(0,1,2,3,4,5,6,7,8,9,10))
        
        self.pos = Vector(self.t, self._x, self._y, self._z)
        
        self.velocity = Vector(self.t, self._vx, self._vy, self._vz)
	self.acceleration = Vector(self.t, self._ax, self._ay, self._az)

	#Define Velocity in Spherical Polar Coordinates
	self.rdot = func_RadialVelocity(self.pos, self.velocity)
	self.phidot = func_PhaseVelocity(self.pos, self.velocity)
	self.thetadot = func_ThetaVelocity(self.pos, self.velocity)       

 
    def __StitchFile(self):
        """ Function to stitch ShiftTracker data """

	filename = self.filename
	simulation_path = self.simpath
	destination_path = os.path.basename(self.filepath)
	
	Combine_0D_Data(filename, simulation_path, destination_path, 1)
	return

       
    def __Info(self, verbose=3):
	""" Function to get basic information about initial data """

	hdr = 'BH%d Initial Trajctory Parameters: \n \n'%self.object_number
	msg1 = 'Initial Location = (%g, %g, %g), Distance from Origin = %g \n'%(self.pos.x[0], self.pos.y[0], self.pos.z[0], self.pos.radial_comp)
	msg2 = 'Initial Velocity = (%g, %g, %g)\n'%(self.velocity.x[0], self.velocity.y[0], self.velocity.z[0]
	msg3 = 'Initial radial velocity = %g, angular velocity (phi_dot) = %g \n '%(self.rdot, self.phidot)
	info(hdr+msg1, verbose, 3)
	info(msg2+msg3, verbose, 4)
	return

    def PlotTrajectory(self, plot_type, figsize=None, xlim=None, ylim=None):
	""" Create plots of x vs t, y vs t, y vs x
	
	 Parameters
	------------
	plot_type: String
		Specify the type of plot you want to generate (x-t, y-t, y-x, all)
	figsize: list/array
	c	Specify the figure dimension
	xlim: list/array
		Specify the limit of x axis, does not work for 'all'
	ylim: list/array
		Specify the limit of x axis, does not work for 'all'
	"""	

	figdir = FiguresDirectory(self.simpath)
	if figsize=None: figsize=[8,8]
	
	if plot_type=='all': 
	    numplots = 3
	    plot_arr = ['x-t','y-t','z-t']
	    assert xlim==None, (Fore.RED+'x limit has to be set to None for plot_type=all')   
	    assert ylim==None, (Fore.RED+'y limit has to be set to None for plot_type=all')   
	else:
	    numplots = 1
	    plot_att = [plot_type]

	for i in range(numplots):
	   
	    fig = plt.figure(figsize=figsize)
	    if plot_arr[i]=='x-t':
	        ax1 = plt.plot(self.t, self.pos.x)
		xlabel, ylabel = 'Time', 'x'
	    elif plot_arr[i]=='y-t':
	        ax1 = plt.plot(self.t, self.pos.y)
		xlabel, ylabel = 'Time', 'y'
	    elif plot_arr[i]=='y-x':
	        ax1 = plt.plot(self.x, self.pos.y)
		xlabel, ylabel = 'x', 'y'

	    ax1 = StylizePlot(ax1, xlabel, ylabel, xlim, ylim)
	    plt.savefig(figdir, '%s.png'%plot_arr[i], dpi=300)
 	    plt.close()
		
        
#To Do - Compute the velocity and Acceleration
class MinSearch():
    """MinSearch class to extract the Star trajectory data from MinSearch Files
      
       Parameters
      -------------
      INPUT - Simulation Path
      Output - File location directory, Filename, position, velocity, acceleration Vectors"""

    def __init__(self, simpath, filename, object_number=1):
        self.filename = filename
	self.simpath = simpath
        self.filepath = os.path.join(DataDirectory(simpath), self.filename)
	self.object_number = object_number
    	
	if not os.path.exists(self.filepath):
	    self.StitchFile()
			
        assert os.path.exists(filepath), (Fore.RED+'File %s does not exists.'%(self.filename))
        
        self.pos = Vector(self.t, self._x, self._y, self._z)
        self.it, self.t, self.x, self.y, self.z   = np.loadtxt(file, unpack=True, usecols=(0,1,2,3,4,))
        
        self.pos = Vector(self.t, self.x, self.y, self.z)
   
 
    def __StitchFile(self):
	filename = self.filename
	simulation_path = self.simpath
	destination_path = os.path.basename(self.filepath)
	
	Combine_0D_Data(filename, simulation_path, destination_path, 1)
	return
        
    def __Info(self, verbose=3):
	""" Function to get basic information about initial data """

	hdr = 'Star%d Initial Trajctory Parameters: \n \n'%self.object_number
	msg1 = 'Initial Location = (%g, %g, %g), Distance from Origin = %g \n'%(self.pos.x[0], self.pos.y[0], self.pos.z[0], self.pos.radial_comp)
	msg2 = 'Initial Velocity = (%g, %g, %g)\n'%(self.velocity.x[0], self.velocity.y[0], self.velocity.z[0]
	msg3 = 'Initial radial velocity = %g, angular velocity (phi_dot) = %g \n '%(self.rdot, self.phidot)
	info(hdr+msg1, verbose, 3)
	info(msg2+msg3, verbose, 4)
	return
	
    def PlotTrajectory(self, plot_type, figsize=None, xlim=None, ylim=None):
	""" Create plots of x vs t, y vs t, y vs x
	
	 Parameters
	------------
	plot_type: String
		Specify the type of plot you want to generate (x-t, y-t, y-x, all)
	figsize: list/array
		Specify the figure dimension
	xlim: list/array
		Specify the limit of x axis, does not work for 'all'
	ylim: list/array
		Specify the limit of x axis, does not work for 'all'
	"""	
	figdir = FiguresDirectory(self.simpath)

	if figsize=None: figsize=[8,8]
	
	if plot_type=='all': 
	    numplots = 3
	    plot_arr = ['x-t','y-t','z-t']
	    assert xlim==None, (Fore.RED+'x limit has to be set to None for plot_type=all')   
	    assert ylim==None, (Fore.RED+'y limit has to be set to None for plot_type=all')   
	else:
	    numplots = 1
	    plot_att = [plot_type]

	for i in range(numplots):
	   
	    fig = plt.figure(figsize=figsize)
	    if plot_arr[i]=='x-t':
	        ax1 = plt.plot(self.t, self.pos.x)
		xlabel, ylabel = 'Time', 'x'
	    elif plot_arr[i]=='y-t':
	        ax1 = plt.plot(self.t, self.pos.y)
		xlabel, ylabel = 'Time', 'y'
	    elif plot_arr[i]=='y-x':
	        ax1 = plt.plot(self.x, self.pos.y)
		xlabel, ylabel = 'x', 'y'

	    ax1 = StylizePlot(ax1, xlabel, ylabel, xlim, ylim)
	    plt.savefig(figdir, '%s.png'%plot_arr[i])
 	    plt.close()
        


class Trajectory_BH(ShiftTracker):

#Stitch required outputs, Search for common time, Find the separation, phase etc, Define Plots of trajectory, orbital separation and phase evolution, orbital frequency	
	def __init__(self, object_type, simpath):
	    
	    self.source_type = object_type
	    
	    assert (self.source_type=='BH' or self.source_type=='SBH'), (Fore.RED+'Source object is not single black hole (BH or SBH)! Trajectories cannot be computed.')
	   
	    filename = 'ShiftTracker0.asc'
	    super().__init__(self.filename, simpath)
	        
        
