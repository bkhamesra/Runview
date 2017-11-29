import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import pylab
import plotly.offline as py
import plotly.graph_objs as go
from shutil import copyfile
import os
from CommonFunctions import *

def maxamp_time(wfdir, outdir):

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
	time, real, imag = np.loadtxt(psi4_file, unpack=True)
	
	#Amplitude and Phase
	amp = abs(real+1.j *imag)		
	max_amp = np.amax(amp)
	return max_amp, time[np.where(amp==max_amp)]

	
def Psi4_Plots(wfdir, outdir, locate_merger=False, locate_qnm=False):

	#Create waveform directory
	
	datadir = DataDir(wfdir, outdir)
	statfigdir, dynfigdir = FigDir(wfdir, outdir)	
		
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
	t_qnm_ampred = time[1:n]
	
	

	
	#Max Amplitude	
	max_amp = np.amax(amp)
	max_amp_index = np.where(amp == max_amp)[0]
	
	t_maxamp = time[np.where(amp==np.amax(amp))][0]
	phi_at_maxamp = phi[np.where(time==t_maxamp)]
	real_at_maxamp = real[np.where(time==t_maxamp)]
	imag_at_maxamp = imag[np.where(time==t_maxamp)]
	
	if locate_qnm:	
		#Fitting:
		t_maxamp = time[np.where(amp==np.amax(amp))][0]
		t1_idx = np.amin(np.where(time>=t_maxamp+50))
		t2_idx = np.amin(np.where(time>=t_maxamp+90))
		log_amp = np.log(amp)
		logamp_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], log_amp[t1_idx:t2_idx], 1))
	
		phi_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], phi[t1_idx:t2_idx], 1))
		logamp_max = log_amp[max_amp_index]
	
		#QNM:
		
		amp_diff = np.absolute(log_amp - logamp_fit(time))
		phi_diff = np.absolute(phi - phi_fit(time))
		qnm_amp_idx = np.amin(np.intersect1d(np.where(amp_diff<0.1), np.where(time>=t_maxamp)))
		qnm_phi_idx = np.amin(np.intersect1d(np.where(phi_diff<0.1), np.where(time>=t_maxamp)))

		t_qnm_amp = round(time[qnm_amp_idx],2)
		amp_qnm = round(amp[qnm_amp_idx],6)
		logamp_qnm = round(log_amp[qnm_amp_idx],2)

		t_qnm_phi = time[qnm_phi_idx]
		phi_qnm = phi[qnm_phi_idx]


	#Horizon Info:
	if locate_merger:
		#Horizon
		t_hrzn = func_t_hrzn(datadir, locate_merger)
		t_hrzn3 = t_hrzn+75.
		idx_hrzn = np.amin( np.where(time>=t_hrzn))
		real_hrzn, imag_hrzn = real[idx_hrzn], imag[idx_hrzn]
		amp_hrzn, phi_hrzn = amp[idx_hrzn], phi[idx_hrzn] 
		logamp_hrzn = log_amp[idx_hrzn]

		t_qnm_amparr = np.around(np.array((t_hrzn, t_maxamp)),2)
		amp_arr = np.around(np.array((amp_hrzn, max_amp)),4)
		logamp_arr = np.around(np.array((logamp_hrzn, logamp_max)),4)
		phi_arr = np.around(np.array((phi_hrzn, phi_at_maxamp)),4)
		
		#maxamp again to prevent stupid stuff
		t_maxamp = time[np.where(amp==np.amax(amp))][0]

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
	plt.savefig(statfigdir+"/Psi4_plot.png", dpi = 500)
	plt.close()
	
	#Real 
	#See Trajectory_XYanimation for explanation of Plotly methods RU
	xRm = np.min(time)
	yRm = np.min(real)
	xRM = np.max(time)
	yRM = np.max(real)
	
	figurePsi4R = {
	    'data': [],
	    'layout': {},
	    'frames': []
	}
	
	dataPsi4R=[dict(x=time, y=real,
		     mode='lines',
		     name='Psi4R',
		     line=dict(width=2, color='blue')
		     )
		#dict(x=x_bh1[::100], y=y_bh1[::100],
		#     mode='lines',
		#     name='BH1',
		#     line=dict(width=2, color='orange')
		#     ),
		#dict(x=x_bh2[::100], y=y_bh2[::100],
		#     mode='lines',
		#     name='BH2',
		#     line=dict(width=2, color='blue')
		#    )
		]
	framesPsi4R=[dict(data=[
			#dict(x=[x_bh1[100*k]],
			#     y=[y_bh1[100*k]],
			#     mode='markers',
			#     name='BH1',
			#     marker=dict(color='red',size=10)
			#    ),
			#dict(x=[x_bh2[100*k]],
			#     y=[y_bh2[100*k]],
			#     mode='markers',
			#     name='BH2',
			#     marker=dict(color='green',size=10)
			#    ),
			dict(x=time[:k],
			       y=real[:k],
			       mode='lines',
			       line=dict(color='blue',width=2)
			      )
			 ], 
			  ) for k in range(len(time))]
			
	figurePsi4R['data'] = dataPsi4R
	figurePsi4R['layout']['xaxis'] = {'range':[xRm,xRM], 'autorange': False, 'zeroline': False, 'title': "Time"} # this is the better way to handle things when you have ridiculous numbers of attributes to fix
	figurePsi4R['layout']['yaxis'] = {'range':[yRm,yRM], 'autorange': False, 'zeroline': False, 'title': "Psi4"}
	figurePsi4R['layout']['title'] = "Real Component of Psi4 Vs Time"
	figurePsi4R['layout']['updatemenus'] = [
	  {
	    'buttons':[
		{'label': 'Play',
		 'method': 'animate',
		 'args': [None, {'frame':{'duration': 10, 'redraw':False}, 'mode':'immediate', 'transition':{'duration':0},'fromcurrent':True}]
		},
		{'label': 'Pause',
		 'method': 'animate',
		 'args': [[None], {'frame':{'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition':{'duration':0}, 'fromcurrent': True}]
		}
		]
	  }
	]
	figurePsi4R['frames'] = framesPsi4R
	py.plot(figurePsi4R, filename=dynfigdir+'Psi4R_animation.html')
		
	#Imaginary
	xIm = np.min(time)
	yIm = np.min(imag)
	xIM = np.max(time)
	yIM = np.max(imag)
	
	figurePsi4I = {
	    'data': [],
	    'layout': {},
	    'frames': []
	}
	
	dataPsi4I=[dict(x=time, y=imag,
		     mode='lines',
		     name='Psi4I',
		     line=dict(width=2, color='blue')
		     )
		#dict(x=x_bh1[::100], y=y_bh1[::100],
		#     mode='lines',
		#     name='BH1',
		#     line=dict(width=2, color='orange')
		#     ),
		#dict(x=x_bh2[::100], y=y_bh2[::100],
		#     mode='lines',
		#     name='BH2',
		#     line=dict(width=2, color='blue')
		#    )
		]
	framesPsi4I=[dict(data=[
			#dict(x=[x_bh1[100*k]],
			#     y=[y_bh1[100*k]],
			#     mode='markers',
			#     name='BH1',
			#     marker=dict(color='red',size=10)
			#    ),
			#dict(x=[x_bh2[100*k]],
			#     y=[y_bh2[100*k]],
			#     mode='markers',
			#     name='BH2',
			#     marker=dict(color='green',size=10)
			#    ),
			dict(x=time[:k],
			       y=imag[:k],
			       mode='lines',
			       line=dict(color='blue',width=2)
			      )
			 ], 
			  ) for k in range(len(time))]
	
	
	figurePsi4I['data'] = dataPsi4I
	figurePsi4I['layout']['xaxis'] = {'range':[xIm,xIM], 'autorange': False, 'zeroline': False, 'title': "Time"} # this is the better way to handle things when you have ridiculous numbers of attributes to fix
	figurePsi4I['layout']['yaxis'] = {'range':[yIm,yIM], 'autorange': False, 'zeroline': False, 'title': "Psi4"}
	figurePsi4I['layout']['title'] = "Imaginary Component of Psi4 Vs Time"
	figurePsi4I['layout']['updatemenus'] = [
	  {
	    'buttons':[
		{'label': 'Play',
		 'method': 'animate',
		 'args': [None, {'frame':{'duration': 10, 'redraw':False}, 'mode':'immediate', 'transition':{'duration':0},'fromcurrent':True}]
		},
		{'label': 'Pause',
		 'method': 'animate',
		 'args': [[None], {'frame':{'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition':{'duration':0}, 'fromcurrent': True}]
		}
		]
	  }
	]
	figurePsi4I['frames'] = framesPsi4I
	py.plot(figurePsi4I, filename=dynfigdir+'Psi4I_animation.html')
	
	# Plot2: Psi4 - real and imaginary - near merger
	plt.plot(time,real, 'b', label = "Real", linewidth=1.5)
	#plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
	plt.plot(time,amp, 'k', linewidth=1, label="Amplitude")
	plt.xlim(t_maxamp-150,t_maxamp+100)
	starty,endy = plt.gca().get_ylim()
	startx,endx = plt.gca().get_xlim()
	
	if locate_merger:
		plt.plot([t_maxamp,t_maxamp], [starty,max_amp], 'k--', linewidth =1.5)
		plt.text(t_maxamp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=12)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,amp_hrzn+0.00005,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_qnm_amp,t_qnm_amp], [starty,amp_qnm], 'k--', linewidth =1.5)
		plt.text(t_qnm_amp,amp_qnm+0.00005,'QNM', horizontalalignment='left', fontsize=12)



	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Psi4", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 40))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(statfigdir+"/Psi4_plot_zoomed.png", dpi = 500)
	plt.close()
	
	
	#Plot 3: Psi4 - Amplitude
	
	plt.plot(time,amp, 'b', linewidth=1, label="Amplitude")
	#plt.plot(time,real, 'b', label = "Real", linewidth =1)
	startx,endx = plt.gca().get_xlim()
	starty,endy = plt.gca().get_ylim()
	if locate_merger:
		plt.plot([t_maxamp,t_maxamp], [starty,max_amp], 'k--', linewidth =1.5)
		plt.text(t_maxamp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=10)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,amp_hrzn,'AH3', horizontalalignment='right', fontsize=10)
		plt.plot([t_qnm_amp,t_qnm_amp], [starty,amp_qnm], 'k--', linewidth =1.5)
		plt.text(t_qnm_amp,amp_qnm,'QNM', horizontalalignment='left', fontsize=10)
	    	#for xy in zip(t_qnm_amparr, amp_arr):
	            #plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	plt.xlabel('Time', fontsize=18)
	plt.ylabel("Amplitude", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 150))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend(loc=2)
	#plt.savefig(statfigdir+"/Psi4_amp.png", dpi = 1000)
	plt.close()
	
	#Amplitude
	xAm = np.min(time)
	#yAm = np.min(amp) not working for some reason, so stopgap implemented below
	xAM = np.max(time)
	yAM = np.max(amp)
	yAm = -1*yAM
	
	figurePsi4A = {
	    'data': [],
	    'layout': {},
	    'frames': []
	}
	
	dataPsi4A=[dict(x=time, y=real,
		     mode='lines',
		     name='Psi4A',
		     line=dict(width=2, color='blue')
		     )
		#dict(x=x_bh1[::100], y=y_bh1[::100],
		#     mode='lines',
		#     name='BH1',
		#     line=dict(width=2, color='orange')
		#     ),
		#dict(x=x_bh2[::100], y=y_bh2[::100],
		#     mode='lines',
		#     name='BH2',
		#     line=dict(width=2, color='blue')
		#    )
		]
	framesPsi4A=[dict(data=[
			#dict(x=[x_bh1[100*k]],
			#     y=[y_bh1[100*k]],
			#     mode='markers',
			#     name='BH1',
			#     marker=dict(color='red',size=10)
			#    ),
			#dict(x=[x_bh2[100*k]],
			#     y=[y_bh2[100*k]],
			#     mode='markers',
			#     name='BH2',
			#     marker=dict(color='green',size=10)
			#    ),
			dict(x=time[:k],
			       y=real[:k],
			       mode='lines',
			       line=dict(color='blue',width=2)
			      )
			 ], 
			  ) for k in range(len(time))]
			
	figurePsi4A['data'] = dataPsi4A
	figurePsi4A['layout']['xaxis'] = {'range':[xAm,xAM], 'autorange': False, 'zeroline': False, 'title': "Time"} # this is the better way to handle things when you have ridiculous numbers of attributes to fix
	figurePsi4A['layout']['yaxis'] = {'range':[yAm,yAM], 'autorange': False, 'zeroline': False, 'title': "Psi4"}
	figurePsi4A['layout']['title'] = "Amplitude of Psi4 Vs Time"
	figurePsi4A['layout']['updatemenus'] = [
	  {
	    'buttons':[
		{'label': 'Play',
		 'method': 'animate',
		 'args': [None, {'frame':{'duration': 10, 'redraw':False}, 'mode':'immediate', 'transition':{'duration':0},'fromcurrent':True}]
		},
		{'label': 'Pause',
		 'method': 'animate',
		 'args': [[None], {'frame':{'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition':{'duration':0}, 'fromcurrent': True}]
		}
		]
	  }
	]
	figurePsi4A['frames'] = framesPsi4A
	py.plot(figurePsi4A, filename=dynfigdir+'Psi4A_animation.html')
	
	
	plt.plot(time,amp, 'b', linewidth=1, label="Amplitude")
	if locate_merger:
		plt.xlim(t_hrzn-20, time[qnm_amp_idx]+20)
		startx,endx = plt.gca().get_xlim()
		starty,endy = plt.gca().get_ylim()
	
		plt.plot([t_maxamp,t_maxamp], [starty,max_amp], 'k--', linewidth =1.5)
		plt.text(t_maxamp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=12)
		plt.plot([t_hrzn,t_hrzn], [starty,amp_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,amp_hrzn+0.00003,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_qnm_amp,t_qnm_amp], [starty,amp_qnm], 'k--', linewidth =1.5)
		plt.text(t_qnm_amp,amp_qnm+0.00003,'QNM', horizontalalignment='left', fontsize=12)
	        
		plt.annotate('(%.2f, %.2g)' % (t_hrzn,amp_hrzn), xy=(t_hrzn,amp_hrzn), xytext=(t_hrzn-7,amp_hrzn), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_maxamp,max_amp), xy=(t_maxamp,max_amp), xytext=(t_maxamp-8,max_amp+0.000005), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_qnm_amp,amp_qnm), xy=(t_qnm_amp,amp_qnm), xytext=(t_qnm_amp-5,amp_qnm+0.000005), textcoords='data')

	plt.xlabel('Time', fontsize=18)
	plt.ylabel("Amplitude", fontsize=18)
	#plt.xticks(np.arange(startx, endx, 150))
	plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(statfigdir+"/Psi4_amp_zoom.png", dpi = 1000)
	plt.close()


	#Plot 4: Phase plots
	plt.plot(time, phi, lw=1 )

	if locate_merger:
		
		starty,endy = plt.gca().get_ylim()
		startx,endx = plt.gca().get_xlim()
		plt.plot([t_maxamp,t_maxamp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
		plt.text(t_maxamp,phi_at_maxamp+0.00003,'Max \n Amp', horizontalalignment='center', fontsize=10)	
		plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,starty+5,'AH3', horizontalalignment='right', fontsize=10)
		plt.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
		plt.text(t_qnm_phi,starty+5,'QNM', horizontalalignment='left', fontsize=10)

	
	##plt.xticks(np.arange(startx, endx, 50))
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Phase",fontsize=18)
	
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(statfigdir+"/Psi4_phase.png", dpi = 1000)
	plt.close()
	
	plt.plot(time, phi, lw=1 )
	
	plyopt = plyplot1(time, phi, "Time", "Phase", "Phase Plot") #see common functions for details; RU
	py.plot(plyopt, filename=dynfigdir + "Psi4_phase.html")#basic plot method, object and path/name
	
	if locate_merger:
		plt.xlim(t_hrzn-20, t_qnm_phi+50)
		plt.ylim(phi_hrzn - 10, phi_qnm + 30)
		starty,endy = plt.gca().get_ylim()
		startx,endx = plt.gca().get_xlim()
		plt.plot([t_maxamp,t_maxamp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
		plt.text(t_maxamp,phi_at_maxamp+3,'Max Amp', horizontalalignment='left', fontsize=12)	
		plt.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
		plt.text(t_hrzn,phi_hrzn+3,'AH3', horizontalalignment='right', fontsize=12)
		plt.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
		plt.text(t_qnm_phi,phi_qnm+3,'QNM', horizontalalignment='right', fontsize=12)
		plt.annotate('(%.2f, %.2g)' % (t_hrzn,phi_hrzn), xy=(t_hrzn,phi_hrzn), xytext=(t_hrzn-7,phi_hrzn+1), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_maxamp,phi_at_maxamp), xy=(t_maxamp,phi_at_maxamp), xytext=(t_maxamp-7,phi_at_maxamp+1), textcoords='data')
	        plt.annotate('(%.2f, %.2g)' % (t_qnm_phi,phi_qnm), xy=(t_qnm_phi,phi_qnm), xytext=(t_qnm_phi,phi_qnm+1), textcoords='data')
	    	#for xy in zip(t_qnm_amparr, phi_arr):
	        #    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

	
	##plt.xticks(np.arange(startx, endx, 50))
	plt.xlabel("Time", fontsize=18)
	plt.ylabel("Phase",fontsize=18)
	
	plt.grid(True)
	plt.legend(loc=2)
	plt.savefig(statfigdir+"/Psi4_phase_zoom.png", dpi = 1000)
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
	    #plt.savefig(statfigdir + "QNM_fit.png", dpi=1000)
	    plt.close()


	    
	    fig, (ax1,ax2) = plt.subplots(2,1)
	    ax1.plot(time, phi, 'b', label="Phase")
	    ax1.plot(time, phi_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    if locate_merger:
	        ax1.set_xlim(t_hrzn-100, t_maxamp+150)
		ax1.set_ylim(phi_hrzn-20, phi_at_maxamp+100)
		starty,endy = ax1.get_ylim()
		startx,endx = ax1.get_xlim()
		ax1.plot([t_maxamp,t_maxamp], [starty,phi_at_maxamp], 'k--', linewidth =1.5)
		ax1.text(t_maxamp,phi_at_maxamp+3,'Max Amp', horizontalalignment='center', fontsize=10)	
		ax1.plot([t_hrzn,t_hrzn], [starty,phi_hrzn], 'k--', linewidth =1.5)
		ax1.text(t_hrzn-1,starty+3,'AH3', horizontalalignment='right', fontsize=10)
		ax1.plot([t_qnm_phi,t_qnm_phi], [starty,phi_qnm], 'k--', linewidth =1.5)
		ax1.text(t_qnm_phi,phi_qnm+3,'QNM', horizontalalignment='center', fontsize=10)	

	    ax1.set_xlabel("time")
	    ax1.set_ylabel("Phase")
	    ax1.grid(True)
	    ax1.legend(loc = "lower right")
	    
	    ax2.plot(time, log_amp, 'b', label="log(Amp)")
	    ax2.plot(time, logamp_fit(time), 'r--', linewidth=1, label="Linear Fit")
	    if locate_merger:
		ax2.set_xlim(t_hrzn-100, t_maxamp+150)
		ax2.set_ylim(logamp_hrzn-20, logamp_max+5)
		starty,endy = ax2.get_ylim()
		startx,endx = ax2.get_xlim()
		ax2.plot([t_maxamp,t_maxamp], [starty,logamp_max], 'k--', linewidth =1.5)
		ax2.text(t_maxamp,logamp_max+1,'Max Amp', horizontalalignment='center', fontsize=10)
		ax2.plot([t_hrzn,t_hrzn], [starty,logamp_hrzn], 'k--', linewidth =1.5)
		ax2.text(t_hrzn-1,starty+0.00003,'AH3', horizontalalignment='right', fontsize=10)
		ax2.plot([t_qnm_amp,t_qnm_amp], [starty,logamp_hrzn], 'k--', linewidth =1.5)
		ax2.text(t_hrzn-1,starty+0.00003,'AH3', horizontalalignment='right', fontsize=10)
	    ax2.set_xlabel("Time")
	    ax2.set_ylabel("log(Amp)")
	    ax2.grid(True)
	    plt.legend()
	    #plt.show()
	    plt.savefig(os.path.join(statfigdir, "Psi4_QNMfit_zoom.png"), dpi=1000)
	    plt.close()
	    
	    #Plot 6: Fitting Exponential Amplitude
	    plt.plot(time, amp, 'b', label = "Amplitude")
	    plt.plot(time, np.exp(logamp_fit(time)), 'r--', label="Exponential Fit")
	    if locate_merger:
		plt.ylim(0, max_amp+0.00001)
		plt.xlim(t_hrzn-50, t_maxamp+100)
		plt.plot([t_maxamp,t_maxamp], [0,max_amp], 'k--', linewidth =1.5)
		plt.text(t_maxamp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
		plt.plot([t_hrzn,t_hrzn], [0,amp_hrzn], 'k--', linewidth=1.5)
		plt.text( t_hrzn,amp_hrzn,'AH3', horizontalalignment='right', fontsize=9)
	    plt.xlabel("Time")
	    plt.ylabel("Amplitude")
	    plt.grid(True)
	    #plt.show()
	    plt.savefig(os.path.join(statfigdir, "Psi4_ampfit.png"), dpi=1000)
	    plt.close()
	   
outDirSO = "/home/rudall/Runview/TestCase/OutputDirectory/SOetc_2/"
binSO = "/home/rudall/Runview/TestCase/BBH/SO_D9_q1.5_th2_135_ph1_90_m140/"
binQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_l11_M192-all/"
outDirQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_2/"

Psi4_Plots(binSO, outDirSO)
