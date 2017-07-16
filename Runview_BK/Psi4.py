import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import pylab
from shutil import copyfile
import os
from CommonFunctions import *

def maxamp_time(wfdir, outdir):

	datadir = DataDir(wfdir, outdir)
	figdir = FigDir(wfdir, outdir)	
		
	#Extract Psi4 info
	
	psi4 = "Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc"
	psi4r = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"
	if not(os.path.exists(os.path.join(datadir, psi4))):
		psi4 = psi4r
	
	if not(os.path.exists(os.path.join(datadir,psi4))):
		debuginfo('%s file not found' %psi4)
#		return

	psi4_file = open(os.path.join(datadir, psi4))
	time, real, imag = np.loadtxt(psi4_file, unpack=True)
	
	#Amplitude and Phase
	amp = abs(real+1.j *imag)		
	max_amp = np.amax(amp)
	return max_amp, time[np.where(amp==max_amp)]

	
def Psi4_Plots(wfdir, outdir, locate_merger=False):

	#Create waveform directory
	
	datadir = DataDir(wfdir, outdir)
	figdir = FigDir(wfdir, outdir)	
		
	#Extract Psi4 info
	
	psi4 = "Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc"
	psi4r = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"
	if not(os.path.exists(os.path.join(datadir, psi4))):
		psi4 = psi4r
	
	
	if not(os.path.exists(os.path.join(datadir,psi4))):
		debuginfo('%s file not found' %psi4)
		return

	psi4_file = open(os.path.join(datadir, psi4))
	time, real, imag = np.loadtxt(psi4_file, unpack=True)
	
	#Amplitude and Phase
	amp = abs(real+1.j *imag)			
	phi = -np.unwrap(np.angle(real+1j*imag))
	r =float( ((psi4.split('r'))[-1]).split('.asc')[0])

	psi4_output = open(os.path.join(datadir,'Psi4_l2m2_r75.txt'), 'w')
	hdr = '#Time \t Real \t Imaginary \t Amplitude \t Phase \n'
	data = np.column_stack((time, real, imag, amp, phi))
	np.savetxt(psi4_output, data, header=hdr, delimiter='\t', newline='\n')
	psi4_output.close()

	#Phase derivatives
	tanphi = -1.*np.array(np.divide(imag,real))	
	cosphi = np.array(np.divide(real,amp))
	n = len(tanphi)
	real_der_t = np.divide((real[1:n] - real[0:n-1]), (time[1:n] - time[0:n-1]))
	imag_der_t = np.divide((imag[1:n] - imag[0:n-1]),(time[1:n] - time[0:n-1]))
	
	phidot =-1.* (real[1:n]*imag_der_t - imag[1:n]*real_der_t) 
	phidot = np.divide(phidot, (amp[1:n]**2))
	time_red = time[1:n]
	

	#Max Amplitude	
	max_amp = np.amax(amp)
	max_amp_index = np.where(amp == max_amp)[0]
	t_max_amp = time[max_amp_index]
	phi_at_maxamp = phi[np.where(time==t_max_amp)]
	real_at_maxamp = real[np.where(time==t_max_amp)]
	imag_at_maxamp = imag[np.where(time==t_max_amp)]
	
	#Horizon Info:
	if locate_merger:
		t_hrzn = func_t_hrzn(datadir, locate_merger)
		t_hrzn = t_hrzn+75.
		idx_hrzn =np.amin( np.where(time>=t_hrzn))
		real_hrzn, imag_hrzn = real[idx_hrzn], imag[idx_hrzn]
		amp_hrzn, phi_hrzn = amp[idx_hrzn], phi[idx_hrzn] 

	if real_at_maxamp>=imag_at_maxamp: maxpsi4 = real_at_maxamp
	else: maxpsi4 = imag_at_maxamp
	
	
	#Plot1: Psi4 -  real and imaginary vs time
	plt.plot(time,real, 'b', label = "Real", linewidth =1)
	plt.plot(time, imag, 'k--', label = "Imaginary", linewidth=1)
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Psi4", fontsize=18)
	startx,endx = plt.gca().get_xlim()
	#plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10.)))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_plot.png", dpi = 500)
	plt.close()
	
	# Plot2: Psi4 - real and imaginary - near merger
	plt.plot(time,real, 'b', label = "Real", linewidth=1)
	#plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
	plt.plot(time,amp, 'k', linewidth=1, label="Amplitude")
	plt.xlim(t_max_amp-150,t_max_amp+100)
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()
	
	if locate_merger:
		plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k', linewidth =1.)
		plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=12)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k', linewidth =1.)
		plt.text(t_hrzn,amp_hrzn+0.00005,'AH3', horizontalalignment='right', fontsize=12)
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Psi4", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 40))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_plot_zoomed.png", dpi = 500)
	plt.close()
	
	
	#Plot 3: Psi4 - phase and Amplitude
	
	plt.plot(time,amp, 'b', linewidth=1, label="Amplitude")
	#plt.plot(time,real, 'b', label = "Real", linewidth =1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()
	if locate_merger:
		plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k', linewidth =1.)
		plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=12)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k', linewidth =1.)
		plt.text(t_hrzn,amp_hrzn+0.00003,'AH3', horizontalalignment='right', fontsize=12)

	plt.xlabel('Time', fontsize=18)
	plt.ylabel("Amplitude", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 150))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_amp.png", dpi = 1000)
	plt.close()
	
	
	plt.plot(time, phi, lw=1 )
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()
	if locate_merger:
		plt.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k', linewidth =1.)
		plt.text(t_max_amp,phi_at_maxamp+0.00003,'Max \n Amp', horizontalalignment='left', fontsize=12)	
		plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k', linewidth =1.)
		plt.text(t_hrzn,phi_hrzn+5,'AH3', horizontalalignment='right', fontsize=12)

	
	##plt.xticks(np.arange(startx, endx, 50))
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Phase",fontsize=18)
	
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_phase.png", dpi = 1000)
	plt.close()
	
