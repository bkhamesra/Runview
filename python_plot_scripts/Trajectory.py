
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib
import glob
import os
from CommonFunctions import *

#Set MatPlotLib global parameters here
tick_label_size = 14
matplotlib.rcParams['xtick.labelsize'] = tick_label_size
matplotlib.rcParams['ytick.labelsize'] = tick_label_size


def func_phase(varphase):

	varphi = np.copy(varphase)
	for i in range(len(varphase)):
		if (varphase[i]<0 and varphase[i-1]>0):
			varphi[i:] = varphi[i:] + np.pi
	return varphi

def write_sep_data(outdir, data):
	output_traj = open(os.path.join(outdir, 'Separation.txt'),'w')
	hdr = '# Time \t Separation \t Orbital Phase \n'
	np.savetxt(output_traj, data, header=hdr, delimiter='\t', newline='\n')
	output_traj.close()
	

def Trajectory(wfdir, outdir):
	
  	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	trajectory_bh1 = open(os.path.join(datadir, "ShiftTracker0.asc"))
	trajectory_bh2 = open(os.path.join(datadir, "ShiftTracker1.asc"))
	time_bh1, x_bh1, y_bh1, z_bh1 = np.loadtxt(trajectory_bh1, unpack=True, usecols=(1,2,3,4))
	time_bh2, x_bh2, y_bh2, z_bh2 = np.loadtxt(trajectory_bh2, unpack=True, usecols=(1,2,3,4))


	r1 = np.array((x_bh1, y_bh1, z_bh1))
	r2 = np.array((x_bh2, y_bh2, z_bh2))
	separation = np.linalg.norm(r2-r1, axis=0)
	
	phase = np.arctan(np.divide(y_bh1, x_bh1))
	phi = func_phase(phase)

	#Output Data

	data = np.column_stack((time_bh1, separation, phi))	
	write_sep_data(datadir, data)

	#Plot 1: x vs t and y vs t
	
	f1, ax1 = plt.subplots()
	bh1, = ax1.plot(time_bh1, x_bh1, c='b',  linewidth=1, label="bh1")
	bh2, = ax1.plot(time_bh2, x_bh2, c='k',ls='--', linewidth=1, label = "bh2")
	ax1.set_xlabel('Time', fontsize = 18)
	ax1.set_ylabel('X', fontsize = 18)
	startx,endx = ax1.get_xlim()
	##plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10.)))
	ax1.grid(True)
	ax1.legend()#[bh1,bh2],['bh1','bh2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	plt.savefig(figdir + '/Trajectory_xvstime.png', dpi = 500)
	plt.close()
	
	f2, ax2 = plt.subplots()
	ax2.plot(time_bh1,y_bh1, 'b',linewidth=1, label = "bh1")
	ax2.plot(time_bh2, y_bh2, 'k--', linewidth=1, label = "bh2")
	ax2.set_xlabel('Time', fontsize = 18)
	ax2.set_ylabel('Y', fontsize=18)
	startx,endx = ax2.get_xlim()
	##plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10.)))
	ax2.grid(True)
	ax2.legend()#[bh1,bh2],['bh1','bh2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	plt.savefig(figdir + '/Trajectory_yvstime.png', dpi = 500)
	plt.close()
	
	#Plot 2: Trajectory - y vs x

	f3, ax3 = plt.subplots()
	bh1 = ax3.plot(x_bh1,y_bh1, color='b', linewidth=1, label="bh1")
	bh2 = ax3.plot(x_bh2,y_bh2, 'k--', linewidth=1, label="bh2")
	ax3.set_xlabel('X', fontsize = 18)
	ax3.set_ylabel('Y', fontsize = 18)
	ax3.legend()
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_xy.png', dpi = 500)
	plt.close()
	
	#Plot 3: Trajectory - separation vs time

	plt.plot(time_bh1, separation, color='b', linewidth=1)
	plt.xlabel('Time', fontsize = 18)
	plt.ylabel('Separation', fontsize = 18)
	startx,endx = plt.gca().get_xlim()
	#plt.xticks(np.arange(startx, endx, int(endx/10.- startx/10.)))
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_separation.png', dpi = 500)
	plt.close()
	
	
	#Plot 4: Combined

	plt.plot(time_bh1, phi, color='b', lw=1)
	plt.xlabel('Time', fontsize=18)
	plt.ylabel('Phase', fontsize=18)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_phase.png',dpi = 500)
	plt.close()

	
