###############################################################################
# Script - Energy_momentum.py
# Author - Bhavesh Khamesra
# Purpose -  Visualization of energy and Angular momentum carried by gravitational waves
###############################################################################


import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import pylab
from shutil import copyfile
import os
from CommonFunctions import *
import matplotlib as mpl
mpl.rc('lines', linewidth=2, color='r')
mpl.rc('font', size=16)
mpl.rc('axes', labelsize=16, grid=True)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

def maxamp_time(wfdir, outdir):
	
    '''Compute the time of maximum amplitude in Psi4
    
    ----------------- Parameters -------------------
    wfdir (type 'String') - path of Directory with relevant files
    outdir (type 'String') - path of Output Summary directory
    '''		
    datadir = DataDir(wfdir, outdir)
    	
    #Extract Psi4 info
    
    psi4 = "Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc"
    psi4r = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"
    psi4_mp = "mp_WeylScal4::Psi4i_l2_m2_r75.00.asc"
    if not(os.path.exists(os.path.join(datadir, psi4))):
        psi4 = psi4r
    if not(os.path.exists(os.path.join(datadir, psi4))):
    	psi4 = psi4_mp
    
    if not(os.path.exists(os.path.join(datadir,psi4))):
    	debuginfo('%s file not found' %psi4)
    	return
    
    psi4_file = open(os.path.join(datadir, psi4))
    time, real, imag = np.loadtxt(psi4_file, unpack=True)
    #Amplitude and Phase
    amp = abs(real+1.j *imag)		
    max_amp = np.amax(amp)
    t_maxamp = time[np.where(amp==max_amp)]
    return max_amp, t_maxamp

	
def Psi4_Plots(wfdir, outdir, locate_merger=False, locate_qnm=False):

    '''Analyse Psi4 data and generate plots
    
    ----------------- Parameters -------------------
    wfdir (type 'String') - path of Directory with relevant files
    outdir (type 'String') - path of Output Summary directory
    '''		
    
    #Create waveform directory
    
    datadir = DataDir(wfdir, outdir)
    figdir = FigDir(wfdir, outdir)	
    	
    #Extract Psi4 info
    
    psi4 = "Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc"
    psi4r = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"
    psi4_mp = "mp_WeylScal4::Psi4i_l2_m2_r75.00.asc"
    if not(os.path.exists(os.path.join(datadir, psi4))):
        psi4 = psi4r
    
    if not(os.path.exists(os.path.join(datadir, psi4))):
        psi4 = psi4_mp
    
    
    if not(os.path.exists(os.path.join(datadir,psi4))):
        debuginfo('%s file not found' %psi4)
        return
    
    psi4_file = open(os.path.join(datadir, psi4))
    time, real, imag = np.loadtxt(psi4_file, unpack=True, usecols=(0,1,2))
    
    
    #Amplitude and Phase
    amp = abs(real-1.j *imag)			
    phi = -np.unwrap(np.angle(real-1j*imag))
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
    t_qnm_ampred = time[1:n]
    
    
    
    
    #Max Amplitude	
    max_amp = np.amax(amp)
    max_amp_index = np.where(amp == max_amp)[0]
    
    t_max_amp = time[np.where(amp==np.amax(amp))][0]
    phi_at_maxamp = phi[np.where(time==t_max_amp)]
    real_at_maxamp = real[np.where(time==t_max_amp)]
    imag_at_maxamp = imag[np.where(time==t_max_amp)]
    
    if locate_qnm:	
    
        #Fitting:
        t_max_amp = time[np.where(amp==np.amax(amp))][0]
        t1_idx = np.amin(np.where(time>=t_max_amp+40))
        t2_idx = np.amin(np.where(time>=t_max_amp+80))
        log_amp = np.log(amp)
        logamp_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], log_amp[t1_idx:t2_idx], 1))
        
        phi_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], phi[t1_idx:t2_idx], 1))
        logamp_max = log_amp[max_amp_index]
        
        #Quasi Normal Modes
        
        amp_diff = np.divide(np.absolute(log_amp - logamp_fit(time)),np.absolute(log_amp))
        phi_diff = np.divide(np.absolute(phi - phi_fit(time)),np.absolute(phi))
        qnm_amp_idx = np.amin(np.intersect1d(np.where(amp_diff<0.01), np.where(time>t_max_amp)))
        qnm_phi_idx = np.amin(np.intersect1d(np.where(phi_diff<0.01), np.where(time>t_max_amp)))
        #print("QNM Amp Idx = %d, QNM Phi Idx = %d"%(qnm_amp_idx, qnm_phi_idx))
        #print("Max Amp Time = %g, QNM Amp Time = %g, QNM Phi time = %g"%(t_max_amp, time[qnm_amp_idx], time[qnm_phi_idx]))
    
        t_qnm_amp = round(time[qnm_amp_idx],2)
        amp_qnm = round(amp[qnm_amp_idx],6)
        logamp_qnm = round(log_amp[qnm_amp_idx],2)
    
        t_qnm_phi = time[qnm_phi_idx]
        phi_qnm = phi[qnm_phi_idx]
    
    
    #Horizon Info:
    if locate_merger:
    
        #Horizon
        t_hrzn = func_t_hrzn(datadir, locate_merger)
        t_hrzn = t_hrzn+75.
        idx_hrzn = np.amin( np.where(time>=t_hrzn))
        real_hrzn, imag_hrzn = real[idx_hrzn], imag[idx_hrzn]
        amp_hrzn, phi_hrzn = amp[idx_hrzn], phi[idx_hrzn] 
        logamp_hrzn = log_amp[idx_hrzn]
    
        t_qnm_amparr = np.around(np.array((t_hrzn, t_max_amp)),2)
        amp_arr = np.around(np.array((amp_hrzn, max_amp)),4)
        logamp_arr = np.around(np.array((logamp_hrzn, logamp_max)),4)
        phi_arr = np.around(np.array((phi_hrzn, phi_at_maxamp)),4)
    
    if real_at_maxamp>=imag_at_maxamp: maxpsi4 = real_at_maxamp
    else: maxpsi4 = imag_at_maxamp
    
    
    
    #Plot1: Psi4 -  real and imaginary vs time
    
    plt.plot(time,real, 'b', label = "Real", linewidth =1.5)
    plt.plot(time, imag, 'k--', label = "Imaginary", linewidth=1.5)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Psi4", fontsize=16)
    startx,endx = plt.gca().get_xlim()
    #plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10.)))
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
    plt.grid(True)
    plt.legend(loc=2)
    plt.tight_layout()
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
        plt.text(t_hrzn,amp_hrzn+0.00005,'CAH', horizontalalignment='right', fontsize=12)
        plt.plot([t_qnm_amp,t_qnm_amp], [starty,amp_qnm], 'k--', linewidth =1.5)
        plt.text(t_qnm_amp,amp_qnm+0.00005,'QNM', horizontalalignment='left', fontsize=12)
    
    
    plt.ylim(starty, endy)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Psi4", fontsize=16)
    #plt.xticks(np.arange(startx, endx, 40))
    plt.grid(True)
    #plt.legend(loc=2)
    plt.tight_layout()
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
        plt.text(t_hrzn,amp_hrzn,'CAH', horizontalalignment='right', fontsize=10)
        plt.plot([t_qnm_amp,t_qnm_amp], [starty,amp_qnm], 'k--', linewidth =1.5)
        plt.text(t_qnm_amp,amp_qnm,'QNM', horizontalalignment='left', fontsize=10)
        #for xy in zip(t_qnm_amparr, amp_arr):
            #plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    
    plt.ylim(starty, endy)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel("Amplitude", fontsize=16)
    #plt.xticks(np.arange(startx, endx, 150))
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
    plt.grid(True)
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(figdir+"/Psi4_amp.png", dpi = 500)
    plt.close()
    
    
    plt.plot(time,amp, 'b', linewidth=1, label="Amplitude")
    if locate_merger:
        plt.xlim(t_hrzn-20, time[qnm_amp_idx]+20)
        startx,endx = plt.gca().get_xlim()
        starty,endy = plt.gca().get_ylim()
        
        plt.plot([t_max_amp,t_max_amp], [starty,max_amp], 'k--', linewidth =1.5)
        plt.text(t_max_amp,max_amp+0.00004,'Max Amplitude', horizontalalignment='center', fontsize=12)
        plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
        plt.text(t_hrzn,amp_hrzn+0.00004,'CAH', horizontalalignment='right', fontsize=12)
        plt.plot([t_qnm_amp,t_qnm_amp], [starty,amp_qnm], 'k--', linewidth =1.5)
        plt.text(t_qnm_amp,amp_qnm+0.00004,'QNM', horizontalalignment='left', fontsize=12)
        
        #plt.annotate('(%.2f, %.2g)' % (t_hrzn,amp_hrzn), xy=(t_hrzn,amp_hrzn), xytext=(t_hrzn-7,amp_hrzn), textcoords='data')
        #plt.annotate('(%.2f, %.2g)' % (t_max_amp,max_amp), xy=(t_max_amp,max_amp), xytext=(t_max_amp-8,max_amp+0.000005), textcoords='data')
        #plt.annotate('(%.2f, %.2g)' % (t_qnm_amp,amp_qnm), xy=(t_qnm_amp,amp_qnm), xytext=(t_qnm_amp-5,amp_qnm+0.000005), textcoords='data')
        plt.ylim(starty, endy)
    
    plt.xlabel('Time', fontsize=16)
    plt.ylabel("Amplitude", fontsize=16)
    #plt.xticks(np.arange(startx, endx, 150))
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
    plt.grid(True)
    #plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(figdir+"/Psi4_amp_zoom.png", dpi = 500)
    plt.close()
    
    
    #Plot 4: Phase plots
    plt.plot(time, phi, 'b' )
    
    if locate_merger:
    	
        starty,endy = plt.gca().get_ylim()
        startx,endx = plt.gca().get_xlim()
        plt.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
        plt.text(t_max_amp,phi_at_maxamp+0.00003,'Max \n Amp', horizontalalignment='center', fontsize=10)	
        plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
        plt.text(t_hrzn,starty+5,'CAH', horizontalalignment='right', fontsize=10)
        plt.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
        plt.text(t_qnm_phi,starty+5,'QNM', horizontalalignment='left', fontsize=10)
        plt.ylim(starty, endy)
    
    ##plt.xticks(np.arange(startx, endx, 50))
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Phase",fontsize=16)
    
    plt.grid(True)
    plt.tight_layout()
    #plt.legend(loc=2)
    plt.savefig(figdir+"/Psi4_phase.png", dpi = 500)
    plt.close()
    
    
    plt.plot(time, phi, 'b' )
    
    if locate_merger:
        plt.xlim(t_hrzn-20, t_qnm_phi+50)
        plt.ylim(phi_hrzn - 20, phi_qnm + 30)
        starty,endy = plt.gca().get_ylim()
        startx,endx = plt.gca().get_xlim()
        plt.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
        plt.text(t_max_amp,phi_at_maxamp+1,'Max Amp', horizontalalignment='center', fontsize=12)	
        plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
        plt.text(t_hrzn,phi_hrzn+1,'CAH', horizontalalignment='right', fontsize=12)
        plt.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
        plt.text(t_qnm_phi,phi_qnm+1,'QNM', horizontalalignment='left', fontsize=12)
        #plt.annotate('(%.2f, %.2g)' % (t_hrzn,phi_hrzn), xy=(t_hrzn,phi_hrzn), xytext=(t_hrzn-7,phi_hrzn+1), textcoords='data')
        #plt.annotate('(%.2f, %.2g)' % (t_max_amp,phi_at_maxamp), xy=(t_max_amp,phi_at_maxamp), xytext=(t_max_amp-3,phi_at_maxamp+1), textcoords='data')
        #plt.annotate('(%.2f, %.2g)' % (t_qnm_phi,phi_qnm), xy=(t_qnm_phi,phi_qnm), xytext=(t_qnm_phi,phi_qnm+1), textcoords='data')
        #for xy in zip(t_qnm_amparr, phi_arr):
        #    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    
    
    ##plt.xticks(np.arange(startx, endx, 50))
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Phase",fontsize=16)
    
    plt.grid(True)
    plt.tight_layout()
    #plt.legend(loc=2)
    plt.savefig(figdir+"/Psi4_phase_zoom.png", dpi = 500)
    plt.close()
    
    
    #Plot5: Fitting Linear part
    if locate_qnm:	
        t_qnm_ampfit = time[t1_idx:t2_idx]
        fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
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
        plt.tight_layout()
        #plt.show()
        #plt.savefig(figdir + "QNM_fit.png", dpi=500)
        plt.close()
    
    
        
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(time, phi, 'b', label="Phase")
        ax1.plot(time, phi_fit(time), 'r--', linewidth=1, label="Linear Fit")
        if locate_merger:
            ax1.set_xlim(t_hrzn-100, t_max_amp+100)
    	ax1.set_ylim(phi_hrzn-100, phi_at_maxamp+100)
    	starty,endy = ax1.get_ylim()
    	startx,endx = ax1.get_xlim()
    	ax1.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
    	ax1.text(t_max_amp,phi_at_maxamp+3,'Max Amp', horizontalalignment='center', fontsize=10)	
    	ax1.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
    	ax1.text(t_hrzn-1,starty+3,'CAH', horizontalalignment='right', fontsize=10)
    	ax1.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
    	ax1.text(t_qnm_phi+1,starty+3,'QNM', horizontalalignment='left', fontsize=10)	
    
        ax1.yaxis.set_ticks(np.arange(50*(starty//50), 50*((endy//50)+1), 50))
        ax1.set_ylim(starty, endy)
        #ax1.set_xlabel("time")
        ax1.set_ylabel("Phase")
        ax1.grid(True)
        #ax1.legend(loc = "lower right")
        
        ax2.plot(time, log_amp, 'b', label="log(Amp)")
        ax2.plot(time, logamp_fit(time), 'r--', linewidth=1, label="Linear Fit")
        if locate_merger:
    	ax2.set_xlim(t_hrzn-100, t_max_amp+100)
    	ax2.set_ylim(logamp_hrzn-10, logamp_max+5)
    	starty,endy = ax2.get_ylim()
    	startx,endx = ax2.get_xlim()
    	ax2.plot([t_max_amp,t_max_amp], [starty,logamp_max], 'k--', linewidth =1.5)
    	ax2.text(t_max_amp,logamp_max+1,'Max Amp', horizontalalignment='center', fontsize=10)
    	ax2.plot([t_hrzn,t_hrzn], [starty,logamp_hrzn], 'k--', linewidth =1.5)
    	ax2.text(t_hrzn-1,starty+0.1,'CAH', horizontalalignment='right', fontsize=10)
    	ax2.plot([t_qnm_amp,t_qnm_amp], [starty,logamp_hrzn], 'k--', linewidth =1.5)
    	ax2.text(t_qnm_amp+1,starty+0.1,'QNM', horizontalalignment='left', fontsize=10)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("log(Amp)")
        ax2.grid(True)
        #plt.legend()
        #plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "Psi4_QNMfit_zoom.png"), dpi=500)
        plt.close()
        
    #Plot6: Errors in Fitting Linear part
    if locate_qnm:	
        t_qnm_ampfit = time[t1_idx:t2_idx]
        fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
        ax1.plot(time, phi-phi_fit(time), 'b')
        ax1.set_xlabel("time")
        ax1.set_ylabel("Phase Fit Errors")
        ax1.grid(True)
        ax1.legend(loc = "lower right")
        
        ax2.plot(time, log_amp - logamp_fit(time), 'b')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("log(Amp) Fit Errors")
        ax2.grid(True)
        plt.tight_layout()
        #plt.show()
        #plt.savefig(figdir + "QNM_fit.png", dpi=500)
        plt.close()
    
    
        
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(time, phi - phi_fit(time), 'b')
        if locate_merger:
            ax1.set_xlim(t_hrzn-100, t_max_amp+100)
    	ax1.set_ylim(-5, 5)
    	starty,endy = ax1.get_ylim()
    	startx,endx = ax1.get_xlim()
    	ax1.plot([t_max_amp,t_max_amp], [starty,phi_at_maxamp - phi_fit(t_max_amp)], 'k--', linewidth =1.5)
    	ax1.text(t_max_amp,phi_at_maxamp - phi_fit(t_max_amp)+0.5,'Max Amp', horizontalalignment='center', fontsize=10)	
    	ax1.plot([t_hrzn,t_hrzn], [starty,phi_hrzn - phi_fit(t_hrzn)], 'k--', linewidth =1.5)
    	ax1.text(t_hrzn-1,starty+0.5,'CAH', horizontalalignment='right', fontsize=10)
    	ax1.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm - phi_fit(t_qnm_phi)], 'k--', linewidth =1.5)
    	ax1.text(t_qnm_phi+1,starty+0.5,'QNM', horizontalalignment='left', fontsize=10)	
    
        #ax1.yaxis.set_ticks(np.arange(50*(starty//50), 50*((endy//50)+1), 50))
        ax1.set_ylim(starty, endy)
        #ax1.set_xlabel("time")
        ax1.set_ylabel("Phase Fit Errors", fontsize=14)
        ax1.grid(True)
        #ax1.legend(loc = "lower right")
        
        ax2.plot(time, log_amp - logamp_fit(time), 'b')
        if locate_merger:
    	ax2.set_xlim(t_hrzn-30, t_max_amp+100)
    	ax2.set_ylim(-5, 5)
    	starty,endy = ax2.get_ylim()
    	startx,endx = ax2.get_xlim()
    	ax2.plot([t_max_amp,t_max_amp], [starty,logamp_max - logamp_fit(t_max_amp)], 'k--', linewidth =1.5)
    	ax2.text(t_max_amp,logamp_max- logamp_fit(t_max_amp)+1,'Max Amp', horizontalalignment='center', fontsize=10)
    	ax2.plot([t_hrzn,t_hrzn], [starty,logamp_hrzn-logamp_fit(t_hrzn)], 'k--', linewidth =1.5)
    	ax2.text(t_hrzn-1,starty+0.1,'CAH', horizontalalignment='right', fontsize=10)
    	ax2.plot([t_qnm_amp,t_qnm_amp], [starty,logamp_qnm-logamp_fit(t_qnm_amp)], 'k--', linewidth =1.5)
    	ax2.text(t_qnm_amp+1,starty+0.1,'QNM', horizontalalignment='left', fontsize=10)
        ax2.set_xlabel("Time", fontsize=14)
        ax2.set_ylabel("log(Amp) Fit Errors", fontsize=14)
        ax2.grid(True)
        #plt.legend()
        #plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "Psi4_errors.png"), dpi=500)
        plt.close()
    
        #Plot 7: Fitting Exponential Amplitude
        plt.plot(time, amp, 'b', label = "Amplitude")
        plt.plot(time, np.exp(logamp_fit(time)), 'r--', label="Exponential Fit")
        if locate_merger:
    	plt.ylim(0, max_amp+0.00001)
    	plt.xlim(t_hrzn-50, t_max_amp+100)
    	plt.plot([t_max_amp,t_max_amp], [0,max_amp], 'k--', linewidth =1.5)
    	plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
    	plt.plot([t_hrzn,t_hrzn], [0,amp_hrzn], 'k--', linewidth=1.5)
    	plt.text( t_hrzn,amp_hrzn,'CAH', horizontalalignment='right', fontsize=9)
    	plt.plot([t_qnm_amp,t_qnm_amp], [0,amp_hrzn], 'k--', linewidth=1.5)
    	plt.text( t_qnm_amp,amp_qnm,'QNM', horizontalalignment='left', fontsize=9)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        #plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "Psi4_ampfit.png"), dpi=500)
        plt.close()
        
        #Plot 7: Fitting Exponential Amplitude
        plt.plot(time, np.absolute(amp - np.exp(logamp_fit(time))), 'b')
        if locate_merger:
    	plt.ylim(0, max_amp+0.00001)
    	plt.xlim(t_hrzn-50, t_max_amp+100)
    	plt.plot([t_max_amp,t_max_amp], [0,max_amp], 'k--', linewidth =1.5)
    	plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
    	plt.plot([t_hrzn,t_hrzn], [0,amp_hrzn], 'k--', linewidth=1.5)
    	plt.text( t_hrzn,amp_hrzn,'CAH', horizontalalignment='right', fontsize=9)
    	plt.plot([t_qnm_amp,t_qnm_amp], [0,amp_hrzn], 'k--', linewidth=1.5)
    	plt.text( t_qnm_amp,amp_qnm,'QNM', horizontalalignment='left', fontsize=9)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        #plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "Psi4_ampfit.png"), dpi=500)
        plt.close()
