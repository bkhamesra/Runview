import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import pylab
#import matplotlib
import glob, math
import os
from CommonFunctions import *
from Psi4 import maxamp_time
#Set MatPlotLib global parameters here
#tick_label_size = 14
#matplotlib.rcParams['xtick.labelsize'] = tick_label_size
#matplotlib.rcParams['ytick.labelsize'] = tick_label_size

def func_phase(varphase): #a definition

	varphi = np.copy(varphase)
	for i in range(len(varphase)):
		if abs(varphase[i-1]-varphase[i]-2.*np.pi)<0.1:
			varphi[i:] = varphi[i:] + 2.*np.pi
	return varphi

def write_sep_data(filename,hdr, outdir, data): #data handling
	output_traj = open(os.path.join(outdir, filename),'w')
	np.savetxt(output_traj, data, header=hdr, delimiter='\t', newline='\n')
	output_traj.close()
	
def vel_polar(x,y,z, vx, vy, vz):	#while one can simply treat velocity as vectors and perform transformation from cartesian to polar but that results in division by zero
	rvec = np.array((x,y,z))
	rho = norm(rvec, 0)
	th = np.arccos(np.divide(z,rho))
	ph = func_phase(np.arctan2(y,x))
	
	v = np.array((vx,vy,vz))
	vr = (vx*np.cos(ph) + vy*np.sin(ph))*np.sin(th) + vz*np.cos(th)
	thdot = np.cos(th)*np.cos(ph)*vx + np.cos(th)*np.sin(ph)*vy -np.sin(th)*vz		#z*(x*vx + y*vy) - vz*(x**2. + y**2.)
	vtheta = np.divide(thdot, rho)									#np.divide(thdot, (rmag**2.)*np.sqrt(x**2. + y**2.))

	vphi = np.divide((vy*np.cos(ph) - vx*np.sin(ph)), (rho*np.sin(th)))			#np.divide((x*vy - y*vx), (x**2. + y**2.))
	return [vr, vtheta, vphi]
	

def Trajectory(wfdir, outdir, locate_merger=False):
	
  	statfigdir, dynfigdir = FigDir(wfdir, outdir)  #path handling
	datadir = DataDir(wfdir, outdir)

	trajectory_bh1 = open(os.path.join(datadir, "ShiftTracker0.asc")) #data reading
	trajectory_bh2 = open(os.path.join(datadir, "ShiftTracker1.asc"))
	time_bh1, x_bh1, y_bh1, z_bh1, vx_bh1, vy_bh1, vz_bh1 = np.loadtxt(trajectory_bh1, unpack=True, usecols=(1,2,3,4,5,6,7)) #variable to column associations
	time_bh2, x_bh2, y_bh2, z_bh2, vx_bh2, vy_bh2, vz_bh2 = np.loadtxt(trajectory_bh2, unpack=True, usecols=(1,2,3,4,5,6,7))

	#Orbital Separation
	r1 = np.array((x_bh1, y_bh1, z_bh1)) #arrays of position data
	r2 = np.array((x_bh2, y_bh2, z_bh2))
	time = np.copy(time_bh1) #time data
	assert (len(x_bh1)==len(x_bh2)), "Length of position data are not equal. Please check length of Shifttracker files."
		
	vr_bh1, vth_bh1, vph_bh1  = vel_polar(x_bh1, y_bh1, z_bh1, vx_bh1, vy_bh1, vz_bh1)
	vr_bh2, vth_bh2, vph_bh2 = vel_polar(x_bh2, y_bh2, z_bh2, vx_bh2, vy_bh2, vz_bh2)
	
	r_sep = (r1-r2).T #displacement data, xyz are the rows
	x,y,z = r_sep.T #rows to columns, then xyz are those columns

	rmag = norm(r_sep,1) #from commonly used functions, finds norm

	separation = norm(r1-r2, 0)#more processing, fairly intuitive
	log_sep = np.log(separation)

	theta = np.arccos(np.divide(z,rmag))

	phase = np.arctan2(y, x)
	phi = func_phase(phase)
	logphi = np.log(phi)
 	
	#Orbital Velocity
	v1 = np.array((vx_bh1, vy_bh1, vz_bh1)) #orbital definitions of the above
	v2 = np.array((vx_bh2, vy_bh2, vz_bh2))
	v_sep = (v1-v2).T
	vmag = norm(v_sep, 1)
	vx,vy,vz = v_sep.T

	# Derivatives
	rdot = (vx*np.cos(phi) + vy*np.sin(phi))*np.sin(theta) + vz*np.cos(theta) #spherical derivative of separation

	thdot = np.cos(theta)*np.cos(phi)*vx + np.cos(theta)*np.sin(phi)*vy -np.sin(theta)*vz		#z*(x*vx + y*vy) - vz*(x**2. + y**2.)
	thdot =	np.divide(thdot, rmag)									#np.divide(thdot, (rmag**2.)*np.sqrt(x**2. + y**2.))

	phdot = np.divide((vy*np.cos(phi) - vx*np.sin(phi)), (rmag*np.sin(theta)))			#np.divide((x*vy - y*vx), (x**2. + y**2.))
	nonan_idx = np.squeeze(np.where(np.isnan(phdot)==False))
	
	noinf_idx =  np.squeeze(np.where(abs(phdot[nonan_idx])< float('inf')))
	use_idx = np.sort(np.intersect1d(noinf_idx, nonan_idx))

	#Horizon Location -- finds the location so that you can plot it if desired
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

	data_sep = np.column_stack((time_bh1[use_idx], separation[use_idx], phi[use_idx]))	
	hdr = '# Time \t Separation \t Orbital Phase \n'
	write_sep_data('ShiftTrackerRadiusPhase.asc',hdr, datadir, data_sep)
	
	data_der = np.column_stack((time_bh1[use_idx], rdot[use_idx], phdot[use_idx]))
	hdr = '# Time \t R_dot \t Theta_dot \n'
	write_sep_data('ShiftTrackerRdotThdot.asc', hdr, datadir, data_der)
	
	#data sets follow key:category name(e.g. trace, data) + object (e.g. BH1) + variables (X,Y,Sep,T)
	#Plot 1 X vs T  
	traceBH1XT= go.Scatter( #scatter is standard data type, accomodates discrete points and lines, the latter used here
	  x = time_bh1, 
	  y = x_bh1,
	  mode = "lines",
	  name = "BH1" #variables and labels should be fairly intuitive
	)
	
	traceBH2XT = go.Scatter( #I call them traces because that's what plotly calls them
	  x = time_bh2, 
	  y = x_bh2,
	  mode = "lines",
	  name = "BH2"
	)
	
	dataXT = [traceBH1XT, traceBH2XT] #data is a list containing all the graph objects. It could be initialized with the object initializations inside, but that quickly gets ugly
	layoutXT = go.Layout( #layout objects do exactly what you think they do
	  title = "X vs. Time for BBH System", #obvious
	  hovermode = "closest", #sets what data point the hover info will display for
	  xaxis = dict( #obvious, but note use of dict for these, although it doesn't follow dictionary notation. If in doubt, read the syntax errors
	    title = "Time" 
	  ),
	  yaxis = dict(
	    title = "X"
	  )
	)
	
	plotXT = go.Figure(data=dataXT, layout=layoutXT) #creates the figure object		
	py.plot(plotXT, filename=dynfigdir + "Trajectory_xvstime.html") #does the actual plotting, note delivery to dynamic figure directory, and naming convention similarity to static plots 
	plot_mpl(plotXT,image='png',filename="/home/rudall/Runview/TestCase")
	
	
	#Plot 2: Y vs T
	#as above
	traceBH1YT = go.Scatter(
	  x = time_bh1, 
	  y = y_bh1,
	  mode = "lines",
	  name = "BH1"
	)
	
	traceBH2YT = go.Scatter(
	  x = time_bh2, 
	  y = y_bh2,
	  mode = "lines",
	  name = "BH2"
	)
	
	dataYT = [traceBH1YT, traceBH2YT]
	layoutYT = go.Layout(
	  title = "Y vs. Time for BBH System",
	  hovermode = "closest", 
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Y"
	  )
	)
	
	plotYT = go.Figure(data=dataYT, layout=layoutYT)
	py.plot(plotYT, filename=dynfigdir + "Trajectory_yvstime.html")

	#Plot 3: Sep vs T
	#Largely as above
	traceSepT = go.Scatter(
	  x = time_bh1, 
	  y = separation,
	  mode = "lines",
	  name = "Distance"
	)
	
	traceLSepT = go.Scatter( #logarithmic data can be plotted on the same and toggled, as performed below
	  x = time_bh1, 
	  y = log_sep,
	  mode = "lines",
	  visible=False, #makes this data not load on startup
	  name = "Log Distance"
	)
	
	dataSepT = [traceSepT,traceLSepT]
	
	  
	updatemenusSepT = list([
	  dict(type="buttons",
	       active=-1,
	       buttons=list([
		 dict(label="Regular",
		      method='update',
		      args=[{'visible': [True,False]},
			    {'title':"Separation vs. Time for BBH System"}]),
		 dict(label="Log",
		      method='update',
		      args=[{'visible':[False,True]},
			    {'title':"Log Separation vs. Time for BBH System"}])]))])
	
	layoutSepT = go.Layout(
	  title = "Separation vs. Time for BBH System",
	  hovermode = "closest",
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Separation"
	  ),
	  updatemenus=updatemenusSepT
	)
	
	plotSepT = go.Figure(data=dataSepT, layout=layoutSepT)
	py.plot(plotSepT, filename=dynfigdir + "Trajectory_separation.html")

	#Plot 4: Log Sep vs T
	"""obsolete but kept for reference
	traceLSepT = go.Scatter(
	  x = time_bh1, 
	  y = log_sep,
	  mode = "lines",
	  name = "Log Distance"
	)
	
	dataLSepT = [traceLSepT]
	layoutLSepT = go.Layout(
	  title = "Log Separation vs. Time for BBH System",
	  hovermode = "closest",
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Log Separation"
	  )
	)
	
	plotLSepT = go.Figure(data=dataLSepT, layout=layoutLSepT)
	py.plot(plotLSepT, filename=dynfigdir + "Trajectory_logseparation.html")
	
	"""
	#Plot 5: Orbital Phase vs T
	#methods essentially identical to sep above, including log toggle
	traceOPT = go.Scatter(
	  x = time_bh1, 
	  y = phi,
	  mode = "lines",
	  name = "Phi"
	)
	
	traceLOPT = go.Scatter(
	  x = time_bh1, 
	  y = logphi,
	  visible=False,
	  mode = "lines",
	  name = "Log Phi"
	)
	
	dataOPT = [traceOPT,traceLOPT]
	
	updatemenusOPT = list([
	  dict(type="buttons",
	       active=-1,
	       buttons=list([
		 dict(label="Regular",
		      method='update',
		      args=[{'visible': [True,False]},
			    {'title':"Orbital Phase vs. Time for BBH System"}]),
		 dict(label="Log",
		      method='update',
		      args=[{'visible':[False,True]},
			    {'title':"Log Orbital Phase vs. Time for BBH System"}])]))])
	
	
	layoutOPT = go.Layout(
	  title = "Orbital Phase vs. Time for BBH System",
	  hovermode = "closest",
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Orbital Phase"
	  ),
	  updatemenus=updatemenusOPT
	)
	
	plotOPT = go.Figure(data=dataOPT, layout=layoutOPT)
	py.plot(plotOPT, filename=dynfigdir + "Trajectory_phase.html") 
	
	#Plot 6: Log Orbital Phase vs T
	"""obsolete but kept for reference
	traceLOPT = go.Scatter(
	  x = time_bh1, 
	  y = logphi,
	  mode = "lines",
	  name = "Log Phi"
	)
	
	dataLOPT = [traceLOPT]
	layoutLOPT = go.Layout(
	  title = "Log Orbital Phase vs. Time for BBH System",
	  hovermode = "closest",
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Log Orbital Phase"
	  )
	)
	
	plotLOPT = go.Figure(data=dataLOPT, layout=layoutLOPT)
	py.plot(plotLOPT, filename=dynfigdir + "Trajectory_logphase.html")
	
	"""
	#Plot 7: Velocity of Orbital Separation vs T
	#methods as above, although data reading is taken from Puncture Dynamics, unlike the rest taken from Trajectories
	traceVOST = go.Scatter(
	  x = time_bh1, 
	  y = rdot,
	  mode = "lines",
	  name = "V of Orb Sep"
	)
	
	dataVOST = [traceVOST]
	layoutVOST = go.Layout(
	  title = "Velocity of Orbital Separation vs. Time for BBH System",
	  hovermode = "closest",
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Velocity of Orbital Separation"
	  )
	)
	
	plotVOST = go.Figure(data=dataVOST, layout=layoutVOST)
	py.plot(plotVOST, filename=dynfigdir + "Trajectory_separation_velocity.html")
	
	#Plot 8: Velocity of Orbital Phase vs T
	#methods as above
	traceBH1VOPT = go.Scatter(
	  x = time_bh1, 
	  y = vph_bh1,
	  mode = "lines",
	  name = "BH1"
	)
	
	traceBH2VOPT = go.Scatter(
	  x = time_bh2, 
	  y = vph_bh2,
	  mode = "lines",
	  name = "BH2"
	)
	
	dataVOPT = [traceBH1VOPT, traceBH2VOPT]
	layoutVOPT = go.Layout(
	  title = "Velocity of Orbital Phase vs. Time for BBH System",
	  hovermode = "closest", 
	  xaxis = dict(
	    title = "Time"
	  ),
	  yaxis = dict(
	    title = "Velocity of Orbital Phase"
	  )
	)
	
	plotVOPT = go.Figure(data=dataVOPT, layout=layoutVOPT)
	py.plot(plotVOPT, filename=dynfigdir + "Trajectory_phase_velocity.html")
	
	#Animation 1: X vs Y animation:
	
	if np.min(x_bh1)< np.min(x_bh2): #finds the actual minima and maxima then sets them, for layout purposes
	  xm = np.min(x_bh1)-0.5
	else:
	  xm = np.min(x_bh2)-0.5
	if np.min(y_bh1)< np.min(y_bh2): 
	  ym = np.min(y_bh1)-0.5
	else:
	  ym = np.min(y_bh2)-0.5
	if np.max(x_bh1)> np.max(x_bh2): 
	  xM = np.max(x_bh1)+0.5
	else:
	  xM = np.max(x_bh2)+0.5
	if np.max(y_bh1)> np.max(y_bh2):
	  yM = np.max(y_bh1)+0.5
	else:
	  yM = np.max(y_bh2)+0.5
	
	
	figureXY = { #the figure is initialized ahead of time here to make things a bit cleaner
	    'data': [],
	    'layout': {},
	    'frames': []
	}
	
	dataXY=[dict(x=x_bh1, y=y_bh1, #data is needed even for animations, although it likely will not be used
		     mode='lines',
		     name='BH1',
		     line=dict(width=2, color='orange')
		     ),
		dict(x=x_bh2, y=y_bh2,
		     mode='lines',
		     name='BH2',
		     line=dict(width=2, color='blue')
		   )
		#dict(x=x_bh1[::100], y=y_bh1[::100], #if you wish to have static lines accompanying your frames, repeat the data a second time to make them appear
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
	framesXY=[dict(data=[ #frames are the core of the animation, fairly intuitive
			#dict(x=[x_bh1[100*k]], #sets x and y data for a single point (note marker mode)
			#     y=[y_bh1[100*k]], #100*k means only sampling 1/100th of the actual data, else the animation lags severely
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
			dict(x=x_bh1[:100*k:100], #arrays generate a line which can be viewed. Note how the data above was in fact an array as well, but here we do not include the extra brackets
			       y=y_bh1[:100*k:100], #per numpy slicing, we go to the same point as above, step size 100
			       mode='lines',
			       line=dict(color='orange',width=2)
			    ),
			dict(x=x_bh2[:100*k:100],
			       y=y_bh2[:100*k:100],
			       mode='lines',
			       line=dict(color='blue',width=2)
			      )
			 ], 
			  ) for k in range(len(time_bh1)//100)] #k iteration sets the frames, note how range follows from the above iteration over k
			
	
	figureXY['data'] = dataXY 
	figureXY['layout']['xaxis'] = {'range':[xm,xM], 'autorange': False, 'zeroline': False, 'title': "X Position"} # this is why we initialized the figure first - this would be very messy otherwise
	figureXY['layout']['yaxis'] = {'range':[ym,yM], 'autorange': False, 'zeroline': False, 'title': "Y Position"} #autorange: False prevents it from autoscaling, I'm not sure what zeroline is for
	figureXY['layout']['title'] = "X vs Y for Two Black Holes Approaching Merger"
	figureXY['layout']['updatemenus'] = [ #updatemenus controls buttons, sliders, etc
	  {
	    'buttons':[
		{'label': 'Play', #label is just the text
		 'method': 'animate', #method:animate for controlling animations
		 'args': [None, {'frame':{'duration': 10, 'redraw':False}, 'fromcurrent':True}] #passing None as the first argument makes it a play button. I have no idea what units duration is in. the rest should be decently intuitive
		},
		{'label': 'Pause',
		 'method': 'animate',
		 'args': [[None], {'frame':{'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition':{'duration':0}, 'fromcurrent': True}] #[None] corresponds to a pause button. I have no explanation why. 
		}
		]
	  }
	]
	figureXY['frames'] = framesXY
	py.plot(figureXY, filename=dynfigdir+'Trajectory_xy_animation.html')
	
	"""
	layoutXY=dict(xaxis=dict(range=[xm,xM], autorange=False, zeroline=False),
		      yaxis=dict(range=[ym,yM], autorange=False, zeroline=False),
		      title="X vs Y for Two Black Holes Approaching Merger",
		      updatemenus=[{'type': 'buttons', 
				    'buttons':[{'label':'Play',
						'method': 'animate',
						'args': [None,{'frame']},
					       {'label':'pause',
						'method': 'animate',
						'args':[None]}
						]}])
	"""
	
binQC0 = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_l11_M192-all/data/"
outDir = "/home/rudall/Runview/TestCase/OutputDirectory/SOetc_2/"
binSO = "/home/rudall/Runview/TestCase/BBH/SO_D9_q1.5_th2_135_ph1_90_m140/"
  
Trajectory(binSO, outDir)