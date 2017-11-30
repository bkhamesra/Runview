import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import pylab
from shutil import copyfile
import os
from CommonFunctions import *


def Compute_gwCycles(wfdir, outdir, locate_merger):
    
	#Create waveform directory
	
	datadir = DataDir(wfdir, outdir)
		
	#Extract Psi4 info
	
	psi4 = "Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc"
	psi4r = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"
	if not(os.path.exists(os.path.join(datadir, psi4))):
		psi4 = psi4r
	
	if not(os.path.exists(os.path.join(datadir,psi4))):
		debuginfo('%s file not found' %psi4)
		return

	psi4_file = open(os.path.join(datadir, psi4))
	time, real, imag = np.loadtxt(psi4_file, unpack=True, usecols=(0,1,2))
	
	#Amplitude and Phase
	amp = abs(real+1.j *imag)			
	phi = -np.unwrap(np.angle(real+1j*imag))
	
	time_qnm = max(qnm_time(datadir, outdir))
	time_maxamp = maxamp_time(datadir, outdir)[-1] 
	time_hrzn = func_t_hrzn(datadir, locate_merger)
	
	qnm_idx = np.amin(np.where(time>=time_qnm))
	maxamp_idx = np.amin(np.where(time>=time_maxamp))
	hrzn_idx = np.amin(np.where(time>=time_hrzn))
	
	numcycle_hrzn_maxamp = (phi[maxamp_idx] - phi[hrzn_idx])/(2*np.pi)
	numcycle_maxamp_qnm = (phi[qnm_idx] - phi[maxamp_idx])/(2*np.pi)
	numcycle_hrzn_qnm = (phi[qnm_idx] - phi[hrzn_idx])/(2*np.pi)

	return [numcycle_hrzn_maxamp, numcycle_maxamp_qnm, numcycle_hrzn_qnm]
	



def Psi4_Plots(wfdir, outdir, locate_merger=False, locate_qnm=False):

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
	time, real, imag = np.loadtxt(psi4_file, unpack=True, usecols=(0,1,2))
	
	
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
	#t_qnm_ampred = time[1:n]
	
	

	
	#Max Amplitude	
	max_amp = np.amax(amp)
	max_amp_idx = np.where(amp == max_amp)[0]
	
	t_max_amp = time[np.where(amp==np.amax(amp))][0]
	phi_at_maxamp = phi[np.where(time==t_max_amp)]
	real_at_maxamp = real[np.where(time==t_max_amp)]
	imag_at_maxamp = imag[np.where(time==t_max_amp)]
	
	if locate_qnm:	
		#Fitting:
		t_max_amp = time[np.where(amp==np.amax(amp))][0]
		t1_idx = np.amin(np.where(time>=t_max_amp+40))
		t2_idx = np.amin(np.where(time>=t_max_amp+80)) #After t>tmax_amp+90M,  oscillations have been observed in log(amp)
		log_amp = np.log(amp)
		logamp_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], log_amp[t1_idx:t2_idx], 1))
	
		phi_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], phi[t1_idx:t2_idx], 1))
		logamp_max = log_amp[max_amp_idx]
	
		#QNM:
		
		amp_reldiff = np.absolute(np.divide(log_amp - logamp_fit(time), log_amp)) #relative errors consider to apply same constraint over phase and amplitude fitting
		phi_reldiff = np.absolute(np.divide(phi - phi_fit(time), phi))
		qnm_amp_idx = np.amin(np.intersect1d(np.where(amp_reldiff<0.001), np.where(time>=t_max_amp))) #Results correct to 0.1%   
		qnm_phi_idx = np.amin(np.intersect1d(np.where(phi_reldiff<0.001), np.where(time>=t_max_amp))) 
		
		if qnm_phi_idx > qnm_amp_idx:
		    if amp_reldiff[qnm_phi_idx]> amp_reldiff[qnm_amp_idx]:
		        qnm_amp_idx = np.amin(np.intersect1d(np.where(amp_reldiff<0.001), np.where(time>=time[qnm_phi_idx])))  

	
		t_qnm_amp = round(time[qnm_amp_idx],2)
		amp_qnm = round(amp[qnm_amp_idx],6)
		logamp_qnm = round(log_amp[qnm_amp_idx],2)

		t_qnm_phi = time[qnm_phi_idx]
		phi_qnm = phi[qnm_phi_idx]
		print("QNM Amplitude time = %.2f and QNM phase time = %.2f"%(t_qnm_amp, t_qnm_phi))

		t_qnm = max(t_qnm_amp, t_qnm_phi)
		qnm_idx = max(qnm_amp_idx, qnm_phi_idx)
		
		
	#Horizon Info:
	if locate_merger:
	
		#Horizon
		t_hrzn = func_t_hrzn(datadir, locate_merger)
		hrzn_idx = np.amin( np.where(time>=t_hrzn))
		real_hrzn, imag_hrzn = real[hrzn_idx], imag[hrzn_idx]
		amp_hrzn, phi_hrzn = amp[hrzn_idx], phi[hrzn_idx] 
		logamp_hrzn = log_amp[hrzn_idx]

		t_qnm_amparr = np.around(np.array((t_hrzn, t_max_amp)),2)
		amp_arr = np.around(np.array((amp_hrzn, max_amp)),4)
		logamp_arr = np.around(np.array((logamp_hrzn, logamp_max)),4)
		phi_arr = np.around(np.array((phi_hrzn, phi_at_maxamp)),4)

	if real_at_maxamp>=imag_at_maxamp: maxpsi4 = real_at_maxamp
	else: maxpsi4 = imag_at_maxamp
	

	

	#Plot1: Psi4 -  real and imaginary vs time
	plt.plot(time,real, 'b', label = "Real", linewidth =1.5)
	plt.plot(time, imag, 'k--', label = "Imaginary", linewidth=1.5)
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
	plt.plot(time,real, 'b', label = "Real", linewidth=1.5)
	#plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
	plt.plot(time,amp, 'k', linewidth=1, label="Amplitude")
	plt.xlim(t_max_amp-150,t_max_amp+100)
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()
	
	if locate_merger:
		plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k--', linewidth =1.5)
		plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=12)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,amp_hrzn+0.00005,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_qnm,t_qnm], [starty,amp[qnm_idx]], 'k--', linewidth =1.5)
		plt.text(t_qnm,amp[qnm_idx]+0.00005,'QNM', horizontalalignment='left', fontsize=12)



	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Psi4", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 40))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_plot_zoomed.png", dpi = 500)
	plt.close()
	
	
	#Plot 3: Psi4 - Amplitude
	
	plt.plot(time,amp, 'b', linewidth=1, label="Amplitude")
	#plt.plot(time,real, 'b', label = "Real", linewidth =1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()
	if locate_merger:
		plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k--', linewidth =1.5)
		plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=10)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,amp_hrzn,'AH3', horizontalalignment='right', fontsize=10)
		plt.plot([t_qnm,t_qnm], [starty,amp[qnm_idx]], 'k--', linewidth =1.5)
		plt.text(t_qnm,amp[qnm_idx],'QNM', horizontalalignment='left', fontsize=10)
	    	#for xy in zip(t_qnm_amparr, amp_arr):
	            #plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	plt.xlabel('Time', fontsize=18)
	plt.ylabel("Amplitude", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 150))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_amp.png", dpi = 1000)
	plt.close()
	
	
	plt.plot(time,amp, 'b', linewidth=1, label="Amplitude")
	if locate_merger:
		plt.xlim(t_hrzn-20, time[qnm_amp_idx]+20)
		startx,endx = plt.gca().get_xlim()
		starty,endy = plt.gca().get_ylim()
	
		plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k--', linewidth =1.5)
		plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=12)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,amp_hrzn+0.00003,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_qnm,t_qnm], [starty,amp[qnm_idx]], 'k--', linewidth =1.5)
		plt.text(t_qnm,amp[qnm_idx]+0.00003,'QNM', horizontalalignment='left', fontsize=12)
	        
		plt.annotate('(%.2f, %.2g)' % (t_hrzn,amp_hrzn), xy=(t_hrzn,amp_hrzn), xytext=(t_hrzn-7,amp_hrzn), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_max_amp,max_amp), xy=(t_max_amp,max_amp), xytext=(t_max_amp-8,max_amp+0.000005), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_qnm,amp[qnm_idx]), xy=(t_qnm,amp[qnm_idx]), xytext=(t_qnm-5,amp[qnm_idx]+0.000005), textcoords='data')

	plt.xlabel('Time', fontsize=18)
	plt.ylabel("Amplitude", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 150))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_amp_zoom.png", dpi = 1000)
	plt.close()


	#Plot 4: Phase plots
	plt.plot(time, phi, lw=1 )

	if locate_merger:
		
		starty,endy = plt.gca().get_ylim()
		startx,endx = plt.gca().get_xlim()
		plt.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
		plt.text(t_max_amp,phi_at_maxamp+0.00003,'Max \n Amp', horizontalalignment='center', fontsize=10)	
		plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,starty+5,'AH3', horizontalalignment='right', fontsize=10)
		plt.plot([t_qnm,t_qnm], [starty,phi[qnm_idx]], 'k--', linewidth =1.5)
		plt.text(t_qnm,starty+5,'QNM', horizontalalignment='left', fontsize=10)

	
	##plt.xticks(np.arange(startx, endx, 50))
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Phase",fontsize=18)
	
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_phase.png", dpi = 1000)
	plt.close()
	

	plt.plot(time, phi, lw=1 )

	if locate_merger:
		plt.xlim(t_hrzn-20, t_qnm_phi+50)
		plt.ylim(phi_hrzn - 10, phi_qnm + 30)
		starty,endy = plt.gca().get_ylim()
		startx,endx = plt.gca().get_xlim()
		plt.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
		plt.text(t_max_amp,phi_at_maxamp+3,'Max Amp', horizontalalignment='center', fontsize=12)	
		plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,phi_hrzn+3,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_qnm_amp,t_qnm_amp], [starty,phi[qnm_amp_idx]], 'k--', linewidth =1.5)
		plt.text(t_qnm_amp,phi[qnm_amp_idx]+3,'QNM', horizontalalignment='right', fontsize=12)
		plt.annotate('(%.2f, %.2g)' % (t_hrzn,phi_hrzn), xy=(t_hrzn,phi_hrzn), xytext=(t_hrzn-7,phi_hrzn+1), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_max_amp,phi_at_maxamp), xy=(t_max_amp,phi_at_maxamp), xytext=(t_max_amp-7,phi_at_maxamp+1), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_qnm,phi[qnm_idx]), xy=(t_qnm,phi[qnm_idx]), xytext=(t_qnm-7,phi[qnm_idx]+1), textcoords='data')
	    	#for xy in zip(t_qnm_amparr, phi_arr):
	        #    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	
	##plt.xticks(np.arange(startx, endx, 50))
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Phase",fontsize=18)
	
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(figdir+"/Psi4_phase_zoom.png", dpi = 1000)
	plt.close()


	#Plot5: Fitting Linear part
	if locate_qnm:	
	    t_qnm_ampfit = time[t1_idx:t2_idx]
	    fig, (ax1,ax2) = plt.subplots(2,1)
	    ax1.plot(time, phi, 'b', label="Phase")
	    ax1.plot(time, phi_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    ax1.set_xlabel("time")
	    ax1.set_ylabel("Phase")
	    ax1.grid(True)
	    ax1.legend(loc = "lower right")
	    
	    ax2.plot(time, log_amp, 'b', label="log(Amp)")
	    ax2.plot(time, logamp_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    ax2.set_xlabel("Time")
	    ax2.set_ylabel("log(Amp)")
	    ax2.grid(True)
	    plt.legend()
	    #plt.show()
	    #plt.savefig(figdir + "QNM_fit.png", dpi=1000)
	    plt.close()


	    
	    fig, (ax1,ax2) = plt.subplots(2,1)
	    ax1.plot(time, phi, 'b', label="Phase")
	    ax1.plot(time, phi_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    
	    if locate_merger:
	        ax1.set_xlim(t_hrzn-100, t_max_amp+150)
		ax1.set_ylim(phi_hrzn-20, phi_at_maxamp+100)
		starty,endy = ax1.get_ylim()
		startx,endx = ax1.get_xlim()
		ax1.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
		ax1.text(t_max_amp,phi_at_maxamp+3,'Max Amp', horizontalalignment='center', fontsize=10)	
		ax1.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
		ax1.text(t_hrzn-1,starty+3,'AH3', horizontalalignment='right', fontsize=10)
		ax1.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
		ax1.text(t_qnm_phi+1,starty+3,'QNM', horizontalalignment='left', fontsize=10)	

	    ax1.set_xlabel("time")
	    ax1.set_ylabel("Phase")
	    ax1.grid(True)
	    ax1.legend(loc = "lower right")
	    
	    ax2.plot(time, log_amp, 'b', label="log(Amp)")
	    ax2.plot(time, logamp_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    if locate_merger:
		ax2.set_xlim(t_hrzn-100, t_max_amp+150)
		ax2.set_ylim(logamp_hrzn-20, logamp_max+5)
		starty,endy = ax2.get_ylim()
		startx,endx = ax2.get_xlim()
		ax2.plot([t_max_amp,t_max_amp], [starty,logamp_max], 'k--', linewidth =1.5)
		ax2.text(t_max_amp,logamp_max+1,'Max Amp', horizontalalignment='center', fontsize=10)
		ax2.plot([t_hrzn,t_hrzn], [starty,logamp_hrzn], 'k--', linewidth =1.5)
		ax2.text(t_hrzn-1,starty+1,'AH3', horizontalalignment='right', fontsize=10)
		ax2.plot([t_qnm_amp,t_qnm_amp], [starty,logamp_qnm], 'k--', linewidth =1.5)
		ax2.text(t_qnm_amp-1,starty+1,'QNM', horizontalalignment='left', fontsize=10)
	    ax2.set_xlabel("Time")
	    ax2.set_ylabel("log(Amp)")
	    ax2.grid(True)
	    plt.legend()
	    #plt.show()
	    plt.savefig(os.path.join(figdir, "Psi4_QNMfit_zoom.png"), dpi=1000)
	    plt.close()
	    
	    #Plot 6: Fitting Exponential Amplitude
	    plt.plot(time, amp, 'b', label = "Amplitude")
	    plt.plot(time, np.exp(logamp_fit(time)), 'r--', label="Exponential Fit")
	    if locate_merger:
		plt.ylim(0, max_amp+0.00001)
		plt.xlim(t_hrzn-50, t_max_amp+100)
		plt.plot([t_max_amp,t_max_amp], [0,max_amp], 'k--', linewidth =1.5)
		plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
		plt.plot([t_hrzn,t_hrzn], [0,amp_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn,amp_hrzn,'AH3', horizontalalignment='right', fontsize=9)
		plt.plot([t_qnm,t_qnm], [0,amp[qnm_idx]], 'k--', linewidth=1.5)
		plt.text( t_qnm,amp[qnm_idx],'QNM', horizontalalignment='left', fontsize=9)
	    plt.xlabel("Time")
	    plt.ylabel("Amplitude")
	    plt.grid(True)
	    #plt.show()
	    plt.savefig(os.path.join(figdir, "Psi4_ampfit.png"), dpi=1000)
	    plt.close()

	#Plot7: Errors in linear fits of phase and log(amp):
	    
	    err_phase = phi - phi_fit(time)
	    err_logamp = log_amp - logamp_fit(time)
	    err_amp = amp- np.exp(logamp_fit(time))	

	    fig, (ax1,ax2) = plt.subplots(2,1)
	    ax1.plot(time, phi - phi_fit(time), 'b')
	 #   ax1.plot(time, phi_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    
	    if locate_merger:
	        ax1.set_xlim(t_hrzn-20, t_max_amp+100)
		ax1.set_ylim(-1.*err_phase[max_amp_idx],err_phase[max_amp_idx])
		starty,endy = ax1.get_ylim()
		startx,endx = ax1.get_xlim()
		ax1.plot([t_max_amp,t_max_amp], [starty,err_phase[max_amp_idx]], 'k--', linewidth =1.5)
		ax1.text(t_max_amp,starty+0.05,'Max Amp', horizontalalignment='center', fontsize=10)	
		ax1.plot([t_hrzn,t_hrzn], [starty,err_phase[hrzn_idx]], 'k--', linewidth =1.5)
		ax1.text(t_hrzn-1,starty+0.05,'AH3', horizontalalignment='right', fontsize=10)
		ax1.plot([t_qnm_phi,t_qnm_phi], [starty,err_phase[qnm_idx]], 'k--', linewidth =1.5)
		ax1.text(t_qnm_phi+1,starty+0.05,'QNM', horizontalalignment='left', fontsize=10)	

	    ax1.set_xlabel("time")
	    ax1.set_ylabel("Error in Phase Fit")
	    ax1.grid(True)
	    ax1.legend(loc = "lower right")

	    ax2.plot(time, log_amp - logamp_fit(time), 'b')
	    #ax2.plot(time, logamp_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    if locate_merger:
		ax2.set_xlim(t_hrzn-20, t_max_amp+100)
		ax2.set_ylim(-1.*abs(err_logamp[max_amp_idx]), abs(err_logamp[max_amp_idx]), logamp_max+5)
		starty,endy = ax2.get_ylim()
		startx,endx = ax2.get_xlim()
		ax2.plot([t_max_amp,t_max_amp], [0,err_logamp[max_amp_idx]], 'k--', linewidth =1.5)
		ax2.text(t_max_amp,0.05,'Max Amp', horizontalalignment='center', fontsize=10)
		ax2.plot([t_hrzn,t_hrzn], [0,err_logamp[hrzn_idx]], 'k--', linewidth =1.5)
		ax2.text(t_hrzn-1,0.05,'AH3', horizontalalignment='right', fontsize=10)
		ax2.plot([t_qnm_amp,t_qnm_amp], [0,err_logamp[qnm_idx]], 'k--', linewidth =1.5)
		ax2.text(t_qnm_amp-1,err_logamp[qnm_idx],'QNM', horizontalalignment='left', fontsize=10)
	    ax2.set_xlabel("Time")
	    ax2.set_ylabel("Error in log(Amp) Fit")
	    ax2.grid(True)
	    plt.legend()
	    #plt.show()	
	    plt.savefig(os.path.join(figdir, "Psi4_errors.png"), dpi=1000)
	    plt.close()
	    
	#Error in exponential fits to Amplitude
	    plt.plot(time, amp - np.exp(logamp_fit(time)), 'b')
	   # plt.plot(time, np.exp(logamp_fit(time)), 'r--', label="Exponential Fit")
	    if locate_merger:
		plt.ylim(0, max_amp+0.00001)
		plt.xlim(t_hrzn-20, t_max_amp+100)
		plt.ylim(-0.0001,0.0001 )
		plt.plot([t_max_amp,t_max_amp], [0,err_amp[max_amp_idx]], 'k--', linewidth =1.5)
		plt.text(t_max_amp,0.000005,'Max Amp', horizontalalignment='center', fontsize=9)
		plt.plot([t_hrzn,t_hrzn], [0,err_amp[hrzn_idx]], 'k--', linewidth=1.5)
		plt.text( t_hrzn,0.000005,'AH3', horizontalalignment='right', fontsize=9)
		plt.plot([t_qnm,t_qnm], [0,err_amp[qnm_idx]], 'k--', linewidth=1.5)
		plt.text( t_qnm,err_amp[qnm_idx]+0.000005,'QNM', horizontalalignment='left', fontsize=9)
	    plt.xlabel("Time")
	    plt.ylabel("Error in Amplitude")
	    plt.grid(True)
	    plt.tight_layout()
	    #plt.show()
	    plt.savefig(os.path.join(figdir, "Ampfit_errors.png"), dpi=1000)
	    plt.close()

