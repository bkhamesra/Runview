import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import pylab
from shutil import copyfile
import os

def Psi4_Plots(wfdir, outdir):

	#Create waveform directory
	
	datadir = DataDir(wfdir, outdir)
	figdir = FigDir(wfdir, outdir)	
		
	#Extract Psi4 info
	
	psi4 = ("Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc")
	assert(os.path.exists(os.path.join(data_dir, psi4)),'%s file not found' %psi4

	psi4_file = open(os.path.join(data_dir, psi4))
	time, real, imag = np.loadtxt(psi4_file, unpack=True)
	
	#Amplitude and Phase
	amp = abs(real+1.j *imag)			
	phi = -np.unwrap(np.angle(real+1j*imag))
	r =float( ((psi4.split('r'))[-1]).split('.asc')[0])

	psi4_output = open(os.path.join(outdir,('Psi4_l2m2_r%f.txt'%f)), 'w')
	hdr = '#Time \t Real \t Imaginary \t Amplitude \t Phase \n'
	data = np.column_stack((time, real, imag, amp, phi))
	np.savetxt(psi4_output, data, header=hdr, delimiter='\t', newline='\n'	)
	psi4_output.close()

	#Phase derivatives
	tanphi = -1.*np.array(np.divide(imag,real))	
	cosphi = np.array(np.divide(real,amplitude))
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
	
	if real_at_maxamp>=imag_at_maxamp: maxpsi4 = real_at_maxamp
	else: maxpsi4 = imag_at_maxamp
	
	
	#Plot1: Psi4 -  real and imaginary vs time
	plt.plot(time,real, 'b', label = "Real")
	plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
	plt.xlabel("Time")
	plt.ylabel("Psi4")
	#startx,endx = plt.gca().get_xlim()
	#plt.xticks(np.arange(startx, endx, 50))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend()
	plt.savefig(fig_dir+"Psi4_plot.png", dpi = 1000)
	plt.close()
	
	# Plot2: Psi4 - real and imaginary - near merger
	plt.plot(time,real, 'b', label = "Real")
	plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
	plt.xlim(t_max_amp-300,t_max_amp+200)
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()

	plt.plot([t_max_amp,t_max_amp], [starty,maxpsi4], 'k', linewidth =1.5)
	plt.text(t_max_amp,maxpsi4+0.00005,'Max Amplitude', horizontalalignment='center', fontsize=9)
	#plt.plot([t_horizon3,t_horizon3], [0,psi4_hrzn], 'k', linewidth=1.5)
	#plt.text( t_horizon3,psi4_hrzn + 0.00005,'AH3', horizontalalignment='center', fontsize=9)
	plt.xlabel("Time")
	plt.ylabel("Psi4")
	plt.xticks(np.arange(startx, endx, 10))
	plt.grid(True)
	plt.legend()
	plt.savefig(fig_dir+"Psi4_plot_zoomed.png", dpi = 1000)
	plt.close()
	
	
	#Plot 3: Psi4 - phase and Amplitude
	
	plt.plot(time,amp, 'b')
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()

	plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k--', linewidth =1.5)
	plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
	#plt.plot([t_horizon3,t_horizon3], [0,amp_hrzn], 'k--', linewidth=1.5)
	#plt.text( t_horizon3,amp_hrzn,'AH3', horizontalalignment='right', fontsize=9)

	plt.xlabel('Time')
	plt.ylabel("|Psi4|")
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	#plt.xticks(np.arange(startx, endx, 50))
	plt.grid(True)
	plt.legend()
	plt.savefig(fig_dir+"Psi4_amp.png", dpi = 1000)
	plt.close()
	
	
	plt.plot(time, phi )
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()

	#plt.plot([t_max_amp,t_max_amp], [starty, phi[np.where(time == t_max_amp)]], 'k--', linewidth=1.5)
	#plt.text(t_max_amp,phi_at_maxamp+10 ,'Max\nAmp', horizontalalignment='center', fontsize=9)
	#plt.plot([t_horizon3,t_horizon3], [-20,phi[np.amin(np.where(time>=t_horizon3))]], 'k--', linewidth=1.5)
	#plt.text( t_horizon3,phi_hrzn+5,'AH3', horizontalalignment='right', fontsize=9)
	plt.xlabel("Time")
	plt.ylabel("Phase")
	#plt.xticks(np.arange(startx, endx, 50))
	plt.grid(True)
	plt.legend()
	plt.savefig(fig_dir+"Psi4_phase.png", dpi = 1000)
	plt.close()
	
