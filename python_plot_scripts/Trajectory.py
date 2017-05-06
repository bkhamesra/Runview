
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib
import glob
import os
from CommonFunctions import *

#Set MatPlotLib global parameters here
tick_label_size = 8
matplotlib.rcParams['xtick.labelsize'] = tick_label_size
matplotlib.rcParams['ytick.labelsize'] = tick_label_size


def func_phase(varphase):

	varphi = np.copy(varphase)
	for i in range(len(varphase)):
		if (varphase[i]<0 and varphase[i-1]>0):
			varphi[i:] = varphi[i:] + np.pi
	return varphi

def writedata(outdir, data):
	output_traj = open(os.path.join(outdir, 'Separation.txt'),'w')
	hdr = '# Time \t Separation \t Orbital Phase \n'
	np.savetxt(output_traj, data, header=hdr, delimiter='\t', newline='\n')
	output_traj.close()
	

def Trajectory(wfdir, outdir):
	
  	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(dirpath, outdir)

	trajectory_BH1 = open(os.path.join(datadir, "ShiftTracker0.asc"))
	trajectory_BH2 = open(os.path.join(datadir, "ShiftTracker1.asc"))
	time_bh1, x_bh1, y_bh1, z_bh1 = np.loadtxt(trajectory_BH1, unpack=True, usecols=(1,2,3,4))
	time_bh2, x_bh2, y_bh2, z_bh2 = np.loadtxt(trajectory_BH2, unpack=True, usecols=(1,2,3,4))


	r1 = np.array((x_bh1, y_bh1, z_bh1))
	r2 = np.array((x_bh2, y_bh2, z_bh2))
	separation = np.linalg.norm(r2-r1, axis=0)
	
	phase = np.arctan(np.divide(y_BH1, x_BH1))
	phi = func_phase(phase)

	#Output Data

	data = np.column_stack((time_bh1, separation, phi))	
	writedata(datadir, data)

	#Plot 1: x vs t and y vs t
	
	BH1, = plt.plot(time_BH1, x_BH1, 'g', linewidth=1, label="BH1")
	BH2, = plt.plot(time_BH2, x_BH2, 'k--', linewidth=1, label = "BH2")
	plt.xlabel('Time', fontsize = 12)
	plt.ylabel('X', fontsize = 12)
	startx,endx = plt2.get_xlim()
	plt.xticks(np.arange(startx, endx, 50))
	plt.grid(True)
	plt.legend()#[BH1,BH2],['BH1','BH2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	plt.savefig(figdir + 'Trajectory_xvstime.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
	plt.close()
	
	plt.plot(time_BH1,y_BH1, 'g',linewidth=1, label = "BH1")
	plt.plot(time_BH2, y_BH2, 'k--', linewidth=1, label = "BH2")
	plt.xlabel('Time', fontsize = 12)
	plt.ylabel('Y', fontsize=12)
	startx,endx = plt3.get_xlim()
	plt.xticks(np.arange(startx, endx, 50))
	plt.grid(True)
	plt.legend()#[BH1,BH2],['BH1','BH2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	plt.savefig(figdir + 'Trajectory_yvstime.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
	plt.close()
	
	#Plot 2: Trajectory - y vs x

	fig, ax = plt.subplots()
	bh1 = ax.plot(x_BH1,y_BH1, color='g', linewidth=1, label="BH1")
	bh2 = ax.plot(x_BH2,y_BH2, 'k--', linewidth=1, label="BH2")
	ax.set_xlabel('X', fontsize = 12)
	ax.set_ylabel('Y', fontsize = 12)
	plt.legend()
	uplt.grid(True)
	plt.savefig(figdir+'Trajectory_xy.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
	plt.close()
	
	#Plot 3: Trajectory - separation vs time

	plt.plot(time_BH1, separation, color='b', linewidth=1)
	plt.xlabel('Time', fontsize = 12)
	plt.ylabel('Separation', fontsize = 12)
	startx,endx = plt.gca().get_xlim()
	plt.xticks(np.arange(startx, endx, 50))
	plt.grid(True)
	plt.savefig(figdir+'Trajectory_separation.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
	plt.close()
	
	
	#Plot 4: Combined

	plt.plot(time_BH1, phi, color='b')
	plt.xlabel('Time')
	plt.ylabel('Phase')
	plt.grid(True)
	plt.savefig(figdir+'Trajectory_separation.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
	plt.close()
)	
