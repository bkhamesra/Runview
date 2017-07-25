
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib
import glob, math
import os
from CommonFunctions import *
from Psi4 import maxamp_time
#Set MatPlotLib global parameters here
tick_label_size = 14
matplotlib.rcParams['xtick.labelsize'] = tick_label_size
matplotlib.rcParams['ytick.labelsize'] = tick_label_size

def func_phase(varphase):

	varphi = np.copy(varphase)
	for i in range(len(varphase)):
		if abs(varphase[i-1]-varphase[i]-np.pi)<0.1:
			varphi[i:] = varphi[i:] + np.pi
	return varphi

def write_sep_data(outdir, data):
	output_traj = open(os.path.join(outdir, 'Separation.txt'),'w')
	hdr = '# Time \t Separation \t Orbital Phase \n'
	np.savetxt(output_traj, data, header=hdr, delimiter='\t', newline='\n')
	output_traj.close()
	

def Trajectory(wfdir, outdir, locate_merger=False):
	
  	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	trajectory_bh1 = open(os.path.join(datadir, "ShiftTracker0.asc"))
	trajectory_bh2 = open(os.path.join(datadir, "ShiftTracker1.asc"))
	time_bh1, x_bh1, y_bh1, z_bh1 = np.loadtxt(trajectory_bh1, unpack=True, usecols=(1,2,3,4))
	time_bh2, x_bh2, y_bh2, z_bh2 = np.loadtxt(trajectory_bh2, unpack=True, usecols=(1,2,3,4))


	r1 = np.array((x_bh1, y_bh1, z_bh1))
	r2 = np.array((x_bh2, y_bh2, z_bh2))
	assert (len(x_bh1)==len(x_bh2)), "Length of position data are not equal. Please check length of Shifttracker files."
	separation = np.linalg.norm(r2-r1, axis=0)
	log_sep = np.log(separation)

	phase = np.arctan(np.divide(y_bh1, x_bh1))
	phi = func_phase(phase)
	logphi = np.log(phi)

	#Horizon Location
	if locate_merger==True:
		bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
		t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
		maxamp, t_maxamp = maxamp_time(wfdir, outdir)	
		t_maxamp = t_maxamp-75.		
		hrzn_idx = np.amin(np.where(time_bh1>=t_hrzn3))
		maxamp_idx = np.amin(np.where(time_bh1>=t_maxamp))

		x_bh1_hrzn, y_bh1_hrzn = x_bh1[hrzn_idx], y_bh1[hrzn_idx]
		x_bh1_amp, y_bh1_amp = x_bh1[maxamp_idx], y_bh1[maxamp_idx]
		x_bh2_hrzn, y_bh2_hrzn = x_bh2[hrzn_idx], y_bh2[hrzn_idx]
		x_bh2_amp, y_bh2_amp = x_bh2[maxamp_idx], y_bh2[maxamp_idx]

		x_hrzn = min(x_bh1_hrzn, x_bh2_hrzn)
		y_hrzn = min(y_bh1_hrzn, y_bh2_hrzn)
		sep_hrzn = separation[hrzn_idx]
		phi_hrzn = phi[hrzn_idx]
		logsep_hrzn = log_sep[hrzn_idx]
		logphi_hrzn = logphi[hrzn_idx]

		x_amp = min(x_bh1_amp, x_bh2_amp)
		y_amp = min(y_bh1_amp, y_bh2_amp)
		sep_amp = separation[maxamp_idx]
		phi_amp = phi[maxamp_idx]
		logphi_amp = logphi[maxamp_idx]
		logsep_amp = log_sep[maxamp_idx]

		time_arr = np.around(np.array((t_hrzn3, t_maxamp)),2)
		x_arr = np.around(np.array((x_hrzn, x_amp)),2)
		y_arr = np.around(np.array((y_hrzn, y_amp)),2)
		sep_arr = np.around(np.array((sep_hrzn, sep_amp)),2)
		logsep_arr = np.around(np.array((logsep_hrzn, logsep_amp)),2)
		phi_arr = np.around(np.array((phi_hrzn, phi_amp)),2)
		logphi_arr = np.around(np.array((logphi_hrzn, logphi_amp)),2)
		radius = (x_hrzn**2. + y_hrzn**2.)**0.5
		print("Final Horizon Detected at %f and Max Amplitude at %f"%(t_hrzn3, t_maxamp))


	#Output Data

	data = np.column_stack((time_bh1, separation, phi))	
	write_sep_data(datadir, data)

	#Plot 1: x vs t and y vs t
	
	f1, ax1 = plt.subplots()
	bh1, = ax1.plot(time_bh1, x_bh1, c='b',  linewidth=1, label="bh1")
	bh2, = ax1.plot(time_bh2, x_bh2, c='k',ls='--', linewidth=1, label = "bh2")
	startx,endx = ax1.get_xlim()
	starty,endy = ax1.get_ylim()
	plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10.)))
	
	if locate_merger==True:
	    ax1.plot([t_hrzn3,t_hrzn3], [starty,x_hrzn], 'g--', linewidth=1.5)
	    ax1.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
	    ax1.plot([t_maxamp,t_maxamp], [starty,x_amp], 'g--', linewidth=1.5)
	    ax1.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	    #for xy in zip(time_arr, x_arr):
	    #    ax1.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	ax1.set_xlabel('Time', fontsize = 18)
	ax1.set_ylabel('X', fontsize = 18)
	ax1.grid(True)
	ax1.legend()#[bh1,bh2],['bh1','bh2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	plt.savefig(figdir + '/Trajectory_xvstime.png', dpi = 500)
	plt.close()
	

	f2, ax2 = plt.subplots()
	ax2.plot(time_bh1,y_bh1, 'b',linewidth=1, label = "bh1")
	ax2.plot(time_bh2, y_bh2, 'k--', linewidth=1, label = "bh2")
	startx,endx = ax2.get_xlim()
	starty,endy = ax2.get_ylim()

	if locate_merger==True:
		ax2.plot([t_hrzn3,t_hrzn3], [starty,y_hrzn], 'g--', linewidth=1.5)
		ax2.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
		ax2.plot([t_maxamp,t_maxamp], [starty,y_amp], 'g--', linewidth=1.5)
		ax2.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	       # for xy in zip(time_arr, y_arr):
	       #     ax2.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10.)))
	ax2.set_xlabel('Time', fontsize = 18)
	ax2.set_ylabel('Y', fontsize=18)
	ax2.grid(True)
	ax2.legend()#[bh1,bh2],['bh1','bh2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	plt.savefig(figdir + '/Trajectory_yvstime.png', dpi = 500)
	plt.close()
	

	#Plot 2: Trajectory - y vs x

	f3, ax3 = plt.subplots()
	bh1 = ax3.plot(x_bh1,y_bh1, color='b', linewidth=1, label="bh1")
	bh2 = ax3.plot(x_bh2,y_bh2, 'k--', linewidth=1, label="bh2")
	
	if locate_merger:
		circle = plt.Circle((0,0), radius,color='orange', alpha =0.7, label="Final Apparent Horizon")
		ax3.add_artist(circle)

	ax3.set_xlabel('X', fontsize = 18)
	ax3.set_ylabel('Y', fontsize = 18)
	ax3.legend()
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_xy.png', dpi = 500)
	plt.close()
	
	#Plot 3: Trajectory - separation vs time

	plt.plot(time_bh1, separation, color='b', linewidth=1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()

	if locate_merger==True:
		plt.plot([t_hrzn3,t_hrzn3], [starty,sep_hrzn], 'g--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_maxamp,t_maxamp], [starty,sep_amp], 'g--', linewidth=1.5)
		plt.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	        #for xy in zip(time_arr, sep_arr):
	        #    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	plt.xlabel('Time', fontsize = 18)
	plt.ylabel('Separation', fontsize = 18)
	plt.xticks(np.arange(startx, endx, int(endx/10.- startx/10.)))
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_separation.png', dpi = 500)
	plt.close()
	
	plt.plot(time_bh1, separation, color='b', linewidth=1)

	if locate_merger==True:
		plt.xlim(t_hrzn3-30, t_maxamp+30)
		startx,endx = plt.gca().get_xlim()
		#plt.ylim(sep_amp-1, sep_hrzn+3)
		starty,endy = plt.gca().get_ylim()
		plt.plot([t_hrzn3,t_hrzn3], [starty,sep_hrzn], 'g--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_maxamp,t_maxamp], [starty,sep_amp], 'g--', linewidth=1.5)
		plt.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	        for xy in zip(time_arr, sep_arr):
	            plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	plt.xlabel('Time', fontsize = 18)
	plt.ylabel('Separation', fontsize = 18)
	#plt.xticks(np.arange(startx, endx, 10))
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_separation_zoom.png', dpi = 500)
	plt.close()
	
	plt.plot(time_bh1, log_sep, color='b', linewidth=1)

	if locate_merger==True:
		plt.xlim(t_hrzn3-20, t_maxamp+20)
		startx,endx = plt.gca().get_xlim()
		plt.ylim(logsep_amp-3, logsep_hrzn+3)
		starty,endy = plt.gca().get_ylim()
		plt.plot([t_hrzn3,t_hrzn3], [starty,logsep_hrzn], 'g--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_maxamp,t_maxamp], [starty,logsep_amp], 'g--', linewidth=1.5)
		plt.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	        for xy in zip(time_arr, logsep_arr):
	            plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	plt.xlabel('Time', fontsize = 18)
	plt.ylabel('log(Separation)', fontsize = 18)
	plt.xticks(np.arange(startx, endx, 10))
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_logseparation.png', dpi = 500)
	plt.close()


	#Plot 4: Combined

	plt.plot(time_bh1, phi, color='b', lw=1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()

	if locate_merger==True:
		plt.plot([t_hrzn3,t_hrzn3], [starty,phi_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_maxamp,t_maxamp], [starty,phi_amp], 'k--', linewidth=1.5)
		plt.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	        #for xy in zip(time_arr, phi_arr):
	         #   plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')


	plt.xlabel('Time', fontsize=18)
	plt.ylabel('Phase', fontsize=18)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_phase.png',dpi = 500)
	plt.close()

	
	plt.plot(time_bh1, phi, color='b', lw=1)
	

	if locate_merger==True:
		plt.xlim(t_hrzn3-20, t_maxamp+20)
		startx,endx = plt.gca().get_xlim()
		plt.ylim(phi_hrzn-7, phi_hrzn+4)
		starty,endy = plt.gca().get_ylim()
		plt.plot([t_hrzn3,t_hrzn3], [starty,phi_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_maxamp,t_maxamp], [starty,phi_amp], 'k--', linewidth=1.5)
		plt.text( t_maxamp,starty+0.2,'Max Amp', horizontalalignment='left', fontsize=12)
	        for x,y in zip(time_arr, phi_arr):
	            plt.annotate('(%s, %s)' % (x,y), xy=(x-5,y), textcoords='data')


	plt.xlabel('Time', fontsize=18)
	plt.ylabel('Phase', fontsize=18)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_phase_zoom.png',dpi = 500)
	plt.close()
	

	plt.plot(time_bh1, logphi, color='b', lw=1)
	

	if locate_merger==True:
		plt.xlim(t_hrzn3-20, t_maxamp+20)
		startx,endx = plt.gca().get_xlim()
		plt.ylim(logphi_hrzn-0.5, logphi_hrzn+0.5)
		starty,endy = plt.gca().get_ylim()
		plt.plot([t_hrzn3,t_hrzn3], [starty,logphi_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn3-1,starty,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_maxamp,t_maxamp], [starty,logphi_amp], 'k--', linewidth=1.5)
		plt.text( t_maxamp+1,starty,'Max Amp', horizontalalignment='left', fontsize=12)
	        for x,y in zip(time_arr, logphi_arr):
	            plt.annotate('(%s, %s)' % (x,y), xy=(x-5,y+0.03), textcoords='data')


	plt.xlabel('Time', fontsize=18)
	plt.ylabel('log(Phase)', fontsize=18)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_logphase.png',dpi = 500)
	plt.close()
