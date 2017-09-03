
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
		if abs(varphase[i-1]-varphase[i]-2.*np.pi)<0.1:
			varphi[i:] = varphi[i:] + 2.*np.pi
	return varphi

def write_sep_data(filename,hdr, outdir, data):
	output_traj = open(os.path.join(outdir, filename),'w')
	np.savetxt(output_traj, data, header=hdr, delimiter='\t', newline='\n')
	output_traj.close()
	

def merger_time(wfdir, outdir):

	datadir = DataDir(wfdir, outdir)
	bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
	t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
	return t_hrzn3

def multiplot(t1, y1, t2, y2, tmerger, xname, yname, figname, figdir, locate_merger=False):

	plt.plot(t1, y1, color='b', label="BH1")
	plt.plot(t2, y2, color='g', label="BH2")

	starty,endy = plt.gca().get_ylim()
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(t1>=tmerger))		
	    plt.plot([tmerger,tmerger], [starty,endy], 'k--', linewidth=1.5)
 	    plt.text( tmerger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	plt.ylim(starty, endy)
	plt.xlabel(xname)
	plt.ylabel(yname)
	plt.grid(True)	
	plt.legend(bbox_to_anchor=(0.15, 1.02, .7, .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
	plt.savefig(os.path.join(figdir, figname), dpi=500)
	plt.close()


def Trajectory(wfdir, outdir, locate_merger=False):
	
  	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	trajectory_bh1 = open(os.path.join(datadir, "ShiftTracker0.asc"))
	trajectory_bh2 = open(os.path.join(datadir, "ShiftTracker1.asc"))

	time_bh1, x_bh1, y_bh1, z_bh1, vx_bh1, vy_bh1, vz_bh1 = np.loadtxt(trajectory_bh1, unpack=True, usecols=(1,2,3,4,5,6,7))
	time_bh2, x_bh2, y_bh2, z_bh2, vx_bh2, vy_bh2, vz_bh2 = np.loadtxt(trajectory_bh2, unpack=True, usecols=(1,2,3,4,5,6,7))

	#Orbital Separation
	r1 = np.array((x_bh1, y_bh1, z_bh1))
	r2 = np.array((x_bh2, y_bh2, z_bh2))
	time = np.copy(time_bh1)
	assert (len(x_bh1)==len(x_bh2)), "Length of position data are not equal. Please check length of Shifttracker files."
		
	r_sep = (r1-r2).T
	x,y,z = r_sep.T

	rmag = norm(r_sep,1)
	separation = norm(r1-r2, 0)
	log_sep = np.log(separation)

	theta = np.arccos(np.divide(z,rmag))

	phase = np.arctan2(y, x)
	phi = func_phase(phase)
	logphi = np.log(phi)
 	
	#Orbital Velocity
	v1 = np.array((vx_bh1, vy_bh1, vz_bh1))
	v2 = np.array((vx_bh2, vy_bh2, vz_bh2))
	v_sep = (v1-v2).T
	vmag = norm(v_sep, 1)
	vx,vy,vz = v_sep.T

	# Derivatives
	rdot = (vx*np.cos(phi) + vy*np.sin(phi))*np.sin(theta) + vz*np.cos(theta)

	thdot = np.cos(theta)*np.cos(phi)*vx + np.cos(theta)*np.sin(phi)*vy -np.sin(theta)*vz		#z*(x*vx + y*vy) - vz*(x**2. + y**2.)
	thdot =	np.divide(thdot, rmag)									#np.divide(thdot, (rmag**2.)*np.sqrt(x**2. + y**2.))

	phdot = np.divide((vy*np.cos(phi) - vx*np.sin(phi)), (rmag*np.sin(theta)))			#np.divide((x*vy - y*vx), (x**2. + y**2.))
	nonan_idx = np.squeeze(np.where(np.isnan(phdot)==False))
	
	noinf_idx =  np.squeeze(np.where(abs(phdot[nonan_idx])< float('inf')))
	use_idx = np.sort(np.intersect1d(noinf_idx, nonan_idx))


	#Horizon Location
	if locate_merger==True:
		bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
		t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
		hrzn_idx = np.amin(np.where(time_bh1>=t_hrzn3))

		x_hrzn = min(x_bh1[hrzn_idx], x_bh2[hrzn_idx])
		y_hrzn = min(y_bh1[hrzn_idx], y_bh2[hrzn_idx])
		sep_hrzn = separation[hrzn_idx]
		phi_hrzn = phi[hrzn_idx]
		#logsep_hrzn = log_sep[hrzn_idx]
		#logphi_hrzn = logphi[hrzn_idx]

		radius = (x_hrzn**2. + y_hrzn**2.)**0.5
		#print("Final Horizon Detected at %f and Max Amplitude at %f"%(t_hrzn3, t_maxamp))


	#Output Data

	data_sep = np.column_stack((time_bh1[use_idx], separation[use_idx], phi[use_idx]))	
	hdr = '# Time \t Separation \t Orbital Phase \n'
	write_sep_data('ShiftTrackerRadiusPhase.asc',hdr, datadir, data_sep)
	
	data_der = np.column_stack((time_bh1[use_idx], rdot[use_idx], phdot[use_idx]))
	hdr = '# Time \t R_dot \t Theta_dot \n'
	write_sep_data('ShiftTrackerRdotThdot.asc', hdr, datadir, data_der)

	
	#Plots:

	#Plot 1: x vs t and y vs t
	
	f1, ax1 = plt.subplots()
	bh1, = ax1.plot(time_bh1, x_bh1, c='b',  linewidth=1, label="bh1")
	bh2, = ax1.plot(time_bh2, x_bh2, c='g', linewidth=1, label = "bh2")
	startx,endx = ax1.get_xlim()
	starty,endy = ax1.get_ylim()
	
	if locate_merger==True:
	    ax1.plot([t_hrzn3,t_hrzn3], [starty,x_hrzn], 'k--', linewidth=1.5)
	    ax1.text( t_hrzn3,starty+0.1,'AH3', horizontalalignment='right', fontsize=12)

	ax1.set_xlabel('Time', fontsize = 14)
	ax1.set_ylabel('X', fontsize = 14)
	ax1.grid(True)
	ax1.legend()
	plt.savefig(figdir + '/Trajectory_xvstime.png', dpi = 500)
	plt.close()
	

	f2, ax2 = plt.subplots()
	ax2.plot(time_bh1,y_bh1, 'b',linewidth=1, label = "bh1")
	ax2.plot(time_bh2, y_bh2, 'g', linewidth=1, label = "bh2")
	startx,endx = ax2.get_xlim()
	starty,endy = ax2.get_ylim()

	if locate_merger==True:
		ax2.plot([t_hrzn3,t_hrzn3], [starty,y_hrzn], 'k--', linewidth=1.5)
		ax2.text( t_hrzn3,starty+0.1,'AH3', horizontalalignment='right', fontsize=12)

	ax2.set_xlabel('Time', fontsize = 14)
	ax2.set_ylabel('Y', fontsize=14)
	ax2.grid(True)
	ax2.legend()
	plt.savefig(figdir + '/Trajectory_yvstime.png', dpi = 500)
	plt.close()
	

	#Plot 2: Trajectory - y vs x

	f3, ax3 = plt.subplots()
	bh1 = ax3.plot(x_bh1,y_bh1, color='b', linewidth=1, label="bh1")
	bh2 = ax3.plot(x_bh2,y_bh2, 'g', linewidth=1, label="bh2")
	
	if locate_merger:
		circle = plt.Circle((0,0), radius,color='orange', alpha =0.7, label="Final Apparent Horizon")
		ax3.add_artist(circle)

	ax3.set_xlabel('X', fontsize = 14)
	ax3.set_ylabel('Y', fontsize = 14)
	ax3.legend()
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_xy.png', dpi = 500)
	plt.close()

	
	#Plot 3: Trajectory - separation vs time

	plt.plot(time_bh1, separation, color='b', linewidth=1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()

	if locate_merger==True:
		plt.plot([t_hrzn3,t_hrzn3], [starty,sep_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.1,'AH3', horizontalalignment='right', fontsize=12)

	plt.xlabel('Time', fontsize = 14)
	plt.ylabel('Separation', fontsize = 14)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_separation.png', dpi = 500)
	plt.close()
	

	#Plot 4: Orbital Phase

	plt.plot(time_bh1, phi, color='b', lw=1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()

	if locate_merger==True:
		plt.plot([t_hrzn3,t_hrzn3], [starty,phi_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)


	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Phase', fontsize=14)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_phase.png',dpi = 500)
	plt.close()

	

	#Plot 5: Relative orbital Velocity/Rate of Orbital Separation 
	
	plt.plot(time_bh1, rdot, color='b', lw=1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()

	if locate_merger==True:
		plt.plot([t_hrzn3,t_hrzn3], [starty,rdot[hrzn_idx]], 'k--', linewidth=1.5)
		plt.text( t_hrzn3,starty+0.01,'AH3', horizontalalignment='right', fontsize=12)


	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Separation velocity', fontsize=14)
	plt.grid(True)
	plt.savefig(figdir+'/Trajectory_rdot.png',dpi = 500)
	plt.close()
	 

def RadiusPlots(wfdir, outdir, locate_merger=False):
	
	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	bh_diag0 = os.path.join(datadir,'BH_diagnostics.ah1.gp')
	bh_diag1 = os.path.join(datadir,'BH_diagnostics.ah2.gp')
	bh_diag2 = os.path.join(datadir,'BH_diagnostics.ah3.gp')

	if not os.path.exists(bh_diag1):
		debuginfo('%s file not found' %bh_diag1)
		return

	time_bh1, rmin1, rmax1, rmean1, r_areal1, grad_ar1 = np.genfromtxt(bh_diag0, usecols = (1,5,6,7,27,32), unpack=True, comments ='#')
	time_bh2, rmin2, rmax2, rmean2, r_areal2, grad_ar2= np.genfromtxt(bh_diag1, usecols = (1,5,6,7,27,32), unpack=True, comments ='#')
	time_bh3, rmin3, rmax3, rmean3, r_areal3, grad_ar3 = np.genfromtxt(bh_diag2, usecols = (1,5,6,7,27,32), unpack=True, comments ='#')
	
	t_merger = merger_time(wfdir, outdir)
	
	#Plot 1: Areal Radius Plots
	f1, (ax1,ax2) = plt.subplots(2,1)
	bh1, = ax1.plot(time_bh1, r_areal1, c='b',  linewidth=1, label="BH1")
	bh2, = ax1.plot(time_bh2, r_areal2, c='g',  linewidth=1, label="BH2")
	bh3, = ax1.plot(time_bh3, r_areal3, c='r',  linewidth=1, label="BH3")
	
	startx,endx = ax1.get_xlim()
	starty,endy = ax1.get_ylim()
	
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    ax1.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	  #  ax1.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	
	ax2.plot(time_bh1, grad_ar1, c='b',  linewidth=2, label="BH1")
	ax2.plot(time_bh2, grad_ar2, c='g',  linewidth=1, label="BH2")
	ax2.plot(time_bh3, grad_ar3, c='r',  linewidth=1, label="BH3")
	
	startx,endx = ax2.get_xlim()
	starty,endy = ax2.get_ylim()
	
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    ax2.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	    ax2.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)

	ax1.set_ylabel('Areal Radius', fontsize = 14)
	ax2.set_xlabel('Time', fontsize = 14)
	ax2.set_ylabel('Grad(Areal Radius)', fontsize = 14)
	ax1.grid(True)
	ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	ax2.grid(True)
	#ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.savefig(figdir + '/ArealRadius.png', dpi = 500)
	plt.close()
	

	#Plot 2: Coordinate Radius Plots
	f1, ax1 = plt.subplots()
	bh1, = ax1.plot(time_bh1, rmean1, c='b',  linewidth=2, label="BH1:mean radius")
	#ax1.plot(time_bh1, rmin1, ls='--', linewidth=1, c='b', label="BH1:min-max radius")
	#ax1.plot(time_bh1, rmax1, ls='--', linewidth=1, c='b')
	#ax1.fill_between(time_bh1, rmin1, rmax1, color='deepskyblue')

	bh2, = ax1.plot(time_bh2, rmean2, c='g',  linewidth=1, label="BH2:mean radius")
	#ax1.plot(time_bh2, rmin2, ls='--', linewidth=1, c='g', label="BH2:min-max radius")
	#ax1.plot(time_bh2, rmax2, ls='--', linewidth=1, c='g')
	#ax1.fill_between(time_bh2, rmin2, rmax2, color='mediumspringgreen')

	bh3, = ax1.plot(time_bh3, rmean3, c='r',  linewidth=1, label="BH3:mean radius")
	ax1.plot(time_bh3, rmin3, ls='--', linewidth=1, c='r', label="BH3:min-max radius")
	ax1.plot(time_bh3, rmax3, ls='--', linewidth=1, c='r')
	ax1.fill_between(time_bh3, rmin3, rmax3, color='coral', alpha=0.7)

	startx,endx = ax1.get_xlim()
	starty,endy = ax1.get_ylim()
	
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    ax1.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	    ax1.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	
	ax1.set_xlabel('Time', fontsize = 14)
	ax1.set_ylabel('Coordinate Radius', fontsize = 14)
	ax1.grid(True)
	ax1.legend( loc=2, prop={'size': 8})
	plt.savefig(figdir + '/CoordRadius.png', dpi = 500)
	plt.close()


def ProperDistance(wfdir, outdir, locate_merger=False):

  	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
	propdist_file = os.path.join(datadir, "ProperDistance.asc")
	if not os.path.exists(propdist_file):
		debuginfo('Proper Distance file not found' )
		return 
	time_pd, propdist = np.loadtxt(propdist_file, unpack=True, usecols=(1,2))
	
	#Plot 6: Proper Distance
	plt.plot(time_pd, propdist, color='b', lw=1)
	plt.ylim(0, propdist[0]+2)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()

	if locate_merger==True:
		t_hrzn3 = merger_time(wfdir, outdir)
		hrzn_idx = np.amin(np.where(time_pd>=t_hrzn3)) 
		plt.plot([t_hrzn3,t_hrzn3], [starty,propdist[hrzn_idx]], 'k--', linewidth=1)
		plt.text( t_hrzn3,starty+0.2,'AH3', horizontalalignment='right', fontsize=12)


	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Proper Distance', fontsize=14)
	plt.grid(True)
	plt.savefig(figdir+'/ProperDistance.png',dpi = 500)
	plt.close()



def TrumpetPlot(wfdir, outdir, locate_merger=False):
  	
	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
	
	bh_diag0 = os.path.join(datadir,'BH_diagnostics.ah1.gp')
	bh_diag1 = os.path.join(datadir,'BH_diagnostics.ah2.gp')
	bh_diag2 = os.path.join(datadir,'BH_diagnostics.ah3.gp')
	
	time_bh1_diag, x_bh1, y_bh1, z_bh1, rmean_bh1 =np.loadtxt(bh_diag0, unpack=True, usecols=(1,2,3,4,7))	
	time_bh2_diag, x_bh2, y_bh2, z_bh2, rmean_bh2 =np.loadtxt(bh_diag1, unpack=True, usecols=(1,2,3,4,7))	
	time_bh3_diag, x_bh3, y_bh3, z_bh3, rmean_bh3 =np.loadtxt(bh_diag2, unpack=True, usecols=(1,2,3,4,7))	
	

	#Orbital Separation
	r1 = np.array((x_bh1, y_bh1, z_bh1))
	r2 = np.array((x_bh2, y_bh2, z_bh2))
	r3 = np.array((x_bh3, y_bh3, z_bh3))
	
	r1_mag = norm(r1, 0)
	r2_mag = norm(r2, 0)
	r3_mag = norm(r3, 0)

	#Plot 1: Trumpet Plot
	f1, ax1 = plt.subplots()
	bh1, = ax1.plot(r1_mag, time_bh1_diag, c='b',  linewidth=2, label="bh1")
	ax1.plot(r1_mag-rmean_bh1, time_bh1_diag, ls='--', linewidth=1,c='b')
	ax1.plot(r1_mag+rmean_bh1, time_bh1_diag, ls='--', linewidth=1,c='b')
	ax1.fill_betweenx(time_bh1_diag, r1_mag-rmean_bh1, r1_mag + rmean_bh1, color='deepskyblue')

	bh2, = ax1.plot(-1.*r2_mag, time_bh2_diag, c='g', linewidth=2, label = "bh2")
	ax1.plot(-1.*r2_mag-rmean_bh2, time_bh2_diag, ls='--', linewidth=1,c='g')
	ax1.plot(-1.*r2_mag+rmean_bh2, time_bh2_diag, ls='--', linewidth=1,c='g')
	ax1.fill_betweenx(time_bh2_diag, -1.*r2_mag-rmean_bh2, -1.*r2_mag + rmean_bh2, color='mediumspringgreen')

	bh3, = ax1.plot(r3_mag, time_bh3_diag, c='r', linewidth=2, label = "bh3")
	ax1.plot(-1.*r3_mag-rmean_bh3, time_bh3_diag, ls='--', linewidth=1,c='r')
	ax1.plot(-1.*r3_mag+rmean_bh3, time_bh3_diag, ls='--', linewidth=1,c='r')
	ax1.fill_betweenx(time_bh3_diag, -1.*r3_mag-rmean_bh3, -1.*r3_mag + rmean_bh3, color='coral', alpha=0.2)

	startx,endx = ax1.get_xlim()
	starty,endy = ax1.get_ylim()
	
	ax1.set_ylabel('Time', fontsize = 14)
	ax1.set_xlabel('Radial Distance from Punctures', fontsize = 14)
	ax1.grid(True)
	ax1.legend()
	plt.savefig(figdir + '/Trumpets.png', dpi = 500)
	plt.close()



def Area_Mass_Plots(wfdir, outdir, locate_merger=False):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	bh_diag0 = os.path.join(datadir,'BH_diagnostics.ah1.gp')
	bh_diag1 = os.path.join(datadir,'BH_diagnostics.ah2.gp')
	bh_diag2 = os.path.join(datadir,'BH_diagnostics.ah3.gp')

	if not os.path.exists(bh_diag1):
		return

	time_bh1, area1, irr_m1 = np.genfromtxt(bh_diag0, usecols = (1,25,26,), unpack=True, comments ='#')
	time_bh2, area2, irr_m2 = np.genfromtxt(bh_diag1, usecols = (1,25,26,), unpack=True, comments ='#')
	time_bh3, area3, irr_m3 = np.genfromtxt(bh_diag2, usecols = (1,25,26,), unpack=True, comments ='#')

	#Time of merger
	t_merger = merger_time(wfdir, outdir)
	
	#Mass Plots
	plt.plot(time_bh1, irr_m1, color='b', label="BH1")
	plt.plot(time_bh2, irr_m2, color='g', label="BH2")
	plt.plot(time_bh3, irr_m3, color='r', label="BH3")
	
	starty,endy = plt.gca().get_ylim()
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    plt.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	    plt.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	plt.xlabel("Time (in M)")
	plt.ylabel("Irreducible Mass")
	plt.grid(True)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.savefig(os.path.join(figdir, "IrreducibleMasses.png"), dpi=500)
	plt.close()

	#Area Plots		
	plt.plot(time_bh1, area1, color='b', label="BH1")
	plt.plot(time_bh2, area2, color='g', label="BH2")
	plt.plot(time_bh3, area3, color='r', label="BH3")
	
	starty,endy = plt.gca().get_ylim()
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    plt.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	    plt.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	plt.xlabel("Time (in M)")
	plt.ylabel("Area")
	plt.grid(True)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.savefig(os.path.join(figdir, "Area.png"), dpi=500)
	plt.close()


def Spins(wfdir, outdir, locate_merger=False):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
	t_merger = merger_time(wfdir, outdir)
	ihspin0 = open(os.path.join(datadir,"ihspin_hn_0.asc"))
	ihspin1 = open(os.path.join(datadir,"ihspin_hn_1.asc"))
	time_bh1, sx1, sy1, sz1 = np.loadtxt(ihspin0, unpack=True, usecols=(0,1,2,3))
	time_bh2, sx2, sy2, sz2 = np.loadtxt(ihspin1, unpack=True, usecols=(0,1,2,3))
	
	s1 = np.sqrt(sx1**2. + sy1**2. + sz1**2.)
	s2 = np.sqrt(sx2**2. + sy2**2. + sz2**2.)
	
	sxplot = multiplot(time_bh1, sx1, time_bh2, sx2, t_merger, 'Time', r'$S_x$', 'Spinx', figdir, locate_merger=locate_merger)
	syplot = multiplot(time_bh1, sy1, time_bh2, sy2, t_merger,  'Time', r'$S_y$', 'Spiny', figdir, locate_merger=locate_merger)
	szplot = multiplot(time_bh1, sz1, time_bh2, sz2, t_merger,  'Time', r'$S_z$', 'Spinz', figdir, locate_merger=locate_merger)
	smag_plot = multiplot(time_bh1, s1, time_bh2, s2, t_merger, 'Time', 'mag(S)', 'Spinmag', figdir, locate_merger=locate_merger)



def ExpansionPlots(wfdir, outdir, locate_merger=False):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	bh_diag0 = os.path.join(datadir,'BH_diagnostics.ah1.gp')
	bh_diag1 = os.path.join(datadir,'BH_diagnostics.ah2.gp')
	bh_diag2 = os.path.join(datadir,'BH_diagnostics.ah3.gp')

	if not os.path.exists(bh_diag1):
		return
	#theta_l - expansion of outer normal, theta_n - expansion of ingoing normal

	time_bh1, theta_l1, theta_n1 = np.genfromtxt(bh_diag0, usecols = (1,28,29,), unpack=True, comments ='#')	
	time_bh2, theta_l2, theta_n2 = np.genfromtxt(bh_diag1, usecols = (1,28,29,), unpack=True, comments ='#')
	time_bh3, theta_l3, theta_n3 = np.genfromtxt(bh_diag2, usecols = (1,28,29,), unpack=True, comments ='#')
	
	#Time of merger
	t_merger = merger_time(wfdir, outdir)

	plt.scatter(time_bh2, theta_l2, marker='o',color='g', s=10,  facecolors='none', label="BH2")
	plt.scatter(time_bh1, theta_l1, marker='.', s=1, color='b',alpha=0.8, label="BH1")

#	plt.scatter(time_bh3, theta_l3, color='r', s=0.5, label="BH3")
#	plt.ylim(-1,1)
	starty,endy = plt.gca().get_ylim()
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    plt.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	    plt.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	plt.xlabel("Time (in M)")
	plt.ylabel("Expansion (outward normal)")
	plt.grid(True)
	plt.legend()#bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.tight_layout()
	plt.savefig(os.path.join(figdir, "OutwardExpansion.png"), dpi=500)
	plt.close()
	

	plt.plot(time_bh1, theta_n1 , color='b', label="BH1")
	plt.plot(time_bh2, theta_n2, color='g', label="BH2")
	plt.plot(time_bh3, theta_n3, color='r', label="BH3")
	
	starty,endy = plt.gca().get_ylim()
	if locate_merger==True:	
	    hrzn_idx = np.amin(np.where(time_bh1>=t_merger))		
	    plt.plot([t_merger,t_merger], [starty,endy], 'k--', linewidth=1.5)
 	    plt.text( t_merger,starty+0.001,'AH3', horizontalalignment='right', fontsize=12)
	plt.xlabel("Time (in M)")
	plt.ylabel("Expansion (inward normal)")
	plt.grid(True)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.savefig(os.path.join(figdir, "InwardExpansion.png"), dpi=500)
	plt.close()


def PunctureDynamics(wfdir, outdir, locate_merger=False):
	
	Trajectory(wfdir, outdir, locate_merger=locate_merger)
	ProperDistance(wfdir, outdir, locate_merger=locate_merger)
	TrumpetPlot(wfdir, outdir, locate_merger=locate_merger)
	Area_Mass_Plots(wfdir, outdir, locate_merger=locate_merger)
	Spins(wfdir, outdir, locate_merger=locate_merger)
	RadiusPlots(wfdir, outdir, locate_merger=locate_merger)
	ExpansionPlots(wfdir, outdir, locate_merger=locate_merger)
