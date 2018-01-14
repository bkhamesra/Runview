from inspect import getframeinfo, stack
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
#mpl.rcParams['lines.linewidth']=2

def output_data(parfile, data):
	
	datafile = file(parfile)	
	datafile.seek(0)
	for line in datafile:
		if data in line:
			break
	line = line.split()
	data_value = float(line[-1])
	datafile.close()
	return data_value

def debuginfo(message):

	caller = getframeinfo(stack()[1][0])
	filename = caller.filename.split("/")[-1]

	print "Warning: %s:%d - %s" % (filename, caller.lineno, message)

def DataDir(dirpath, outdir):

	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outdir, filename)
	datadir = os.path.join(outputdir,'data')

	if not os.path.exists(datadir):
		os.makedirs(datadir)
	return datadir


def FigDir(dirpath, outdir):

	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outdir, filename)
	figdir = os.path.join(outputdir,'figures')

	if not os.path.exists(figdir):
		os.makedirs(figdir)
	return figdir


def norm(vec, axis):

	return np.apply_along_axis(np.linalg.norm, axis, vec)	

def plot_postmergerinfo(ax, kwargs, starty, endy, height):

   if kwargs['locate_merger']:
        
	y_maxamp = kwargs['y1_maxamp']
	y_qnm = kwargs['y1_qnm']
	y_hrzn = kwargs['y1_hrzn']
	#t_maxamp = [kwargs['t%d_max_amp'%i] if kwargs['t%d_max_amp'%i]>kwargs['t%d_max_amp'%(i-1)dd]
	for i in range(1,kwargs['num_objects']+1):
	    
	    delta_y = (-starty + endy)/height 
            if kwargs['y%d_maxamp'%i]>y_maxamp: y_maxamp = kwargs['y%d_maxamp'%i]   
            if kwargs['y%d_hrzn'%i]>y_hrzn: y_hrzn = kwargs['y%d_hrzn'%i]   
            if kwargs['y%d_qnm'%i]>y_qnm: y_qnm = kwargs['y%d_qnm'%i]   
		
	    if i==1:

                ax.annotate('(%.2f, \n %.2g)' % (kwargs['t%d_hrzn'%i], kwargs['y%d_hrzn'%i]), xy=(kwargs['t%d_hrzn'%i], kwargs['y%d_hrzn'%i]), xytext=(kwargs['t%d_hrzn'%i]-65,kwargs['y%d_hrzn'%i]+delta_y*10), textcoords='data', fontsize=10)
                ax.annotate('(%.2f, %.2g)' % (kwargs['t%d_max_amp'%i], kwargs['y%d_maxamp'%i]), xy=(kwargs['t%d_max_amp'%i], kwargs['y%d_maxamp'%i]), xytext=(kwargs['t%d_max_amp'%i]-60,kwargs['y%d_maxamp'%i]+delta_y*45), textcoords='data', fontsize=10)
                ax.annotate('(%.2f, \n %.2g)' % (kwargs['t%d_qnm'%i], kwargs['y%d_qnm'%i]), xy=(kwargs['t%d_qnm'%i], kwargs['y%d_qnm'%i]), xytext=(kwargs['t%d_qnm'%i]+45,kwargs['y%d_qnm'%i]+delta_y*10), textcoords='data', fontsize=10)
		
	    elif i>1:
		if not np.isclose(kwargs['y%d_hrzn'%i], kwargs['y%d_hrzn'%i]):		
                    ax.annotate('(%.2f, %.2g)' % (kwargs['t%d_hrzn'%i], kwargs['y%d_hrzn'%i]), xy=(kwargs['t%d_hrzn'%i], kwargs['y%d_hrzn'%i]), xytext=(kwargs['t%d_hrzn'%i]-7,kwargs['y%d_hrzn'%i]+delta_y*10), textcoords='data', fontsize=10)
		if not np.isclose(kwargs['y%d_maxamp'%i], kwargs['y%d_maxamp'%i]):		
                    ax.annotate('(%.2f, %.2g)' % (kwargs['t%d_max_amp'%i], kwargs['y%d_maxamp'%i]), xy=(kwargs['t%d_max_amp'%i], kwargs['y%d_maxamp'%i]), xytext=(kwargs['t%d_max_amp'%i]-50,kwargs['y%d_maxamp'%i]+delta_y*10), textcoords='data', fontsize=10)
		if not np.isclose(kwargs['y%d_qnm'%i], kwargs['y%d_qnm'%i]):		
                    ax.annotate('(%.2f, %.2g)' % (kwargs['t%d_qnm'%i], kwargs['y%d_qnm'%i]), xy=(kwargs['t%d_qnm'%i], kwargs['y%d_qnm'%i]), xytext=(kwargs['t%d_qnm'%i]+70,kwargs['y%d_qnm'%i]+delta_y*10), textcoords='data', fontsize=10)


	ax.plot([kwargs['t1_max_amp'],kwargs['t1_max_amp']], [starty, kwargs['y1_maxamp']], 'k--', lw=1.5)
        ax.plot([kwargs['t1_hrzn'],kwargs['t1_hrzn']], [starty, kwargs['y1_hrzn']], 'k--', lw=1.5)
        ax.plot([kwargs['t1_qnm'],kwargs['t1_qnm']], [starty, kwargs['y1_qnm']], 'k--', lw=1.5)
	
        if kwargs['num_objects']==1:
        
        	ax.text(kwargs['t1_max_amp'],y_maxamp+delta_y*60,'Max Amplitude', horizontalalignment='center', fontsize=10)
        	ax.text(kwargs['t1_hrzn'],y_hrzn+delta_y*45,'AH3', horizontalalignment='right', fontsize=10)
        	ax.text(kwargs['t1_qnm'],y_qnm+delta_y*45,'QNM', horizontalalignment='left', fontsize=10)
	    
        else: 
        	ax.text(kwargs['t1_max_amp'],y_maxamp+delta_y*60,'Max Amplitude', horizontalalignment='center', fontsize=10)
        	ax.text(kwargs['t1_hrzn']-70,y_hrzn+delta_y*8,'AH3', horizontalalignment='right', fontsize=10)
        	ax.text(kwargs['t1_qnm'],y_qnm+delta_y*25,'QNM', horizontalalignment='left', fontsize=10)
	
	

def plot1(x,y,xlabel, ylabel, plotname, outdir, **kwargs): 
#kwargs would have details about common horizon, maxamp and qnm. 

        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	height = (fig.get_size_inches()*fig.dpi)[1] # size in pixels

	ax.plot(x,y, 'b', linewidth=1)
	ax.set_ylabel(ylabel, fontsize = 14)
	ax.set_xlabel(xlabel, fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4), fontsize=200)
	startx,endx = ax.get_xlim()
	starty,endy = ax.get_ylim()
	#plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10. )))
	
	if kwargs:
#	    plot_postmergerinfo(ax, kwargs, starty, endy, height)

	    if kwargs['locate_merger']:
	
	        delta_y = (-starty + endy)/height 
	        
	        plt.plot([kwargs['t1_max_amp'],kwargs['t1_max_amp']], [starty, kwargs['y1_maxamp']], 'k--', lw=1.5)
	        plt.plot([kwargs['t1_hrzn'],kwargs['t1_hrzn']], [starty, kwargs['y1_hrzn']], 'k--', lw=1.5)
	        plt.plot([kwargs['t1_qnm'],kwargs['t1_qnm']], [starty, kwargs['y1_qnm']], 'k--', lw=1.5)
   	        
	        plt.text(kwargs['t1_max_amp'],kwargs['y1_maxamp']+delta_y*30,'Max Amplitude', horizontalalignment='center', fontsize=10)
   	        plt.text(kwargs['t1_hrzn'],kwargs['y1_hrzn']+delta_y*30,'AH3', horizontalalignment='right', fontsize=10)
   	        plt.text(kwargs['t1_qnm'],kwargs['y1_qnm']+delta_y*30,'QNM', horizontalalignment='left', fontsize=10)

	        plt.annotate('(%.2f, %.2g)' % (kwargs['t1_hrzn'],kwargs['y1_hrzn']), xy=(kwargs['t1_hrzn'],kwargs['y1_hrzn']), xytext=(kwargs['t1_hrzn']-80,kwargs['y1_hrzn']+delta_y*10), textcoords='data', fontsize=10)
	        plt.annotate('(%.2f, %.2g)' % (kwargs['t1_max_amp'],kwargs['y1_maxamp']), xy=(kwargs['t1_max_amp'],kwargs['y1_maxamp']), xytext=(kwargs['t1_max_amp']-60,kwargs['y1_maxamp']+delta_y*10), textcoords='data', fontsize=10)
	        plt.annotate('(%.2f, %.2g)' % (kwargs['t1_qnm'],kwargs['y1_qnm']), xy=(kwargs['t1_qnm'],kwargs['y1_qnm']), xytext=(kwargs['t1_qnm']-57,kwargs['y1_qnm']+delta_y*10), textcoords='data', fontsize=10)
		
	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(outdir,(plotname+'.png')), dpi = 500)
	plt.close()


def plot2(x1,y1, x2, y2, xlabel, ylabel, plotname, outdir, **kwargs):

        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	height = (fig.get_size_inches()*fig.dpi)[1] # size in pixels
	bh1, = ax.plot(x1, y1, 'blue', linewidth=1, label="BH1")
	bh2, = ax.plot(x2, y2, 'red', ls='--', linewidth=1, label = "BH2")

	ax.set_ylabel(ylabel, fontsize = 14)
	ax.set_xlabel(xlabel, fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4), fontsize=20)
	startx,endx = ax.get_xlim()
	starty,endy = ax.get_ylim()
	#plt.xticks(np.arange(startx, endx, int(endx/10.-startx/10.)))
	
	if kwargs:
	   # plot_postmergerinfo(ax, kwargs, starty, endy, height)
	    if kwargs['locate_merger']:
	
	        delta_y = (-starty + endy)/height 
	        
	        y_maxamp = max(kwargs['y1_maxamp'], kwargs['y2_maxamp'])	    
	        y_hrzn = max(kwargs['y1_hrzn'], kwargs['y2_hrzn'])	    
	        y_qnm = max(kwargs['y1_qnm'], kwargs['y2_qnm'])	    
  
	        plt.plot([kwargs['t1_max_amp'],kwargs['t1_max_amp']], [starty, y_maxamp], 'k--', linewidth=1.5)
	        plt.plot([kwargs['t1_hrzn'],kwargs['t1_hrzn']], [starty, y_hrzn], 'k--', linewidth=1.5)
	        plt.plot([kwargs['t1_qnm'],kwargs['t1_qnm']], [starty, y_qnm], 'k--', linewidth=1.5)
   	       
	        if not np.isclose(kwargs['y1_maxamp'], kwargs['y2_maxamp'], rtol=1e-3):

	            plt.text(kwargs['t1_max_amp'],kwargs['y1_maxamp']+delta_y*30,'Max Amp', horizontalalignment='center', fontsize=10)
   	            plt.text(kwargs['t1_hrzn'],kwargs['y1_hrzn']+delta_y*30,'AH3', horizontalalignment='center', fontsize=10)
   	            plt.text(kwargs['t1_qnm'],kwargs['y1_qnm']+delta_y*30,'QNM', horizontalalignment='center', fontsize=10)

	            plt.text(kwargs['t2_max_amp'],kwargs['y2_maxamp']+delta_y*30,'Max Amplitude', horizontalalignment='center', fontsize=10)
   	            plt.text(kwargs['t2_hrzn'],kwargs['y2_hrzn']+delta_y*30,'AH3', horizontalalignment='center', fontsize=10)
	            plt.text(kwargs['t2_qnm'],kwargs['y2_qnm']+delta_y*30,'QNM', horizontalalignment='center', fontsize=10)
	            
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t1_hrzn'],kwargs['y1_hrzn']), xy=(kwargs['t1_hrzn'],kwargs['y1_hrzn']), xytext=(kwargs['t1_hrzn']-60,kwargs['y1_hrzn']+delta_y*10), textcoords='data', fontsize=10)
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t1_max_amp'],kwargs['y1_maxamp']), xy=(kwargs['t1_max_amp'],kwargs['y1_maxamp']), xytext=(kwargs['t1_max_amp']-60,kwargs['y1_maxamp']+delta_y*10), textcoords='data', fontsize=10)
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t1_qnm'],kwargs['y1_qnm']), xy=(kwargs['t1_qnm'],kwargs['y1_qnm']), xytext=(kwargs['t1_qnm']-57,kwargs['y1_qnm']+delta_y*10), textcoords='data', fontsize=10)
	
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t2_hrzn'],kwargs['y2_hrzn']), xy=(kwargs['t2_hrzn'],kwargs['y2_hrzn']), xytext=(kwargs['t2_hrzn']-60,kwargs['y2_hrzn']+delta_y*10), textcoords='data', fontsize=10)
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t2_max_amp'],kwargs['y2_maxamp']), xy=(kwargs['t2_max_amp'],kwargs['y2_maxamp']), xytext=(kwargs['t2_max_amp']-60,kwargs['y2_maxamp']+delta_y*10), textcoords='data', fontsize=10)
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t2_qnm'],kwargs['y2_qnm']), xy=(kwargs['t2_qnm'],kwargs['y2_qnm']), xytext=(kwargs['t2_qnm']-57,kwargs['y2_qnm']+delta_y*10), textcoords='data', fontsize=10)

	        else:
	
	            plt.text(kwargs['t1_max_amp'], y_maxamp+delta_y*30,'Max Amp', horizontalalignment='center', fontsize=10)
   	            plt.text(kwargs['t1_hrzn']-20, starty+delta_y*45,'AH3', horizontalalignment='right', fontsize=10)
   	            plt.text(kwargs['t1_qnm']+20, starty+delta_y*45,'QNM', horizontalalignment='left', fontsize=10)
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t1_hrzn'], y_hrzn), xy=(kwargs['t1_hrzn'], y_hrzn), xytext=(kwargs['t1_hrzn']-60, starty +delta_y*10), textcoords='data', fontsize=10)
	            plt.annotate('(%.2f, %.2g)' % (kwargs['t1_max_amp'], y_maxamp), xy=(kwargs['t1_max_amp'],y_maxamp), xytext=(kwargs['t1_max_amp']-40, y_maxamp+delta_y*10), textcoords='data', fontsize=10)
	            plt.annotate('(%.2f,\n %.2g)' % (kwargs['t1_qnm'], y_qnm), xy=(kwargs['t1_qnm'], y_qnm), xytext=(kwargs['t1_qnm']+20, starty+delta_y*10), textcoords='data', fontsize=10)

	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(outdir,(plotname+'.png')), dpi = 500)
	plt.close()

def plot3(x1,y1, x2, y2, x3, y3, xlabel, ylabel, plotname, outdir, **kwargs): 
#kwargs would have details about common horizon, maxamp and qnm. 

        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	height = (fig.get_size_inches()*fig.dpi)[1] # size in pixels

	ax.plot(x1,y1, 'b', linewidth=1.5, label='BH1')
	ax.plot(x2,y2, 'g--', linewidth=1.5, label='BH2')
	ax.plot(x3,y3, 'r', linewidth=1.5, label='BH3')
	ax.set_ylabel(ylabel, fontsize = 14)
	ax.set_xlabel(xlabel, fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4), fontsize=200)
	startx,endx = ax.get_xlim()
	starty,endy = ax.get_ylim()
	#plt.xticks(np.arange(startx, endx, int(endx/10. - startx/10. )))
	
	if kwargs:
	    plot_postmergerinfo(ax, kwargs, starty, endy, height)

	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(outdir,(plotname+'.png')), dpi = 500)
	plt.close()

#All the times are obtained in frame of reference of observer at r=75
def func_t_hrzn(datadir, locate_merger):

	if locate_merger==True:
		bhdiag3 = os.path.join(datadir, 'BH_diagnostics.ah3.gp')
		t_hrzn3 = np.loadtxt(bhdiag3, usecols = (1,), unpack=True)[0]
		return t_hrzn3+75  		
	else:
		return 


def maxamp_time(datadir, outdir):

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
	max_amp = np.amax(amp)
	return max_amp, time[np.where(amp==max_amp)]

def qnm_time(datadir, outdir):

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

	t_max_amp = time[np.where(amp==np.amax(amp))][0] # Time of max amp
	print t_max_amp
	t1_idx = np.amin(np.where(time>=t_max_amp+40))	 # 50M after max amplitude
	t2_idx = np.amin(np.where(time>=t_max_amp+80))	 # 90M after max amplitude

	#Fit log(amp) and phase to straight line 
	log_amp = np.log(amp)
	logamp_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], log_amp[t1_idx:t2_idx], 1))
	phi_fit = np.poly1d( np.polyfit(time[t1_idx:t2_idx], phi[t1_idx:t2_idx], 1))


	#QNM: Compute the point where the linear fits intersects the log(amplitude) and phase - Currently set to relative tolerance of 0.1%

	amp_reldiff = np.absolute(np.divide(log_amp - logamp_fit(time), log_amp)) #relative errors consider to apply same constraint over phase and amplitude fitting
	phi_reldiff = np.absolute(np.divide(phi - phi_fit(time), phi))
	qnm_amp_idx = np.amin(np.intersect1d(np.where(amp_reldiff<0.001), np.where(time>=t_max_amp)))  
	qnm_phi_idx = np.amin(np.intersect1d(np.where(phi_reldiff<0.001), np.where(time>=t_max_amp))) 


	if qnm_phi_idx > qnm_amp_idx:
	    if amp_reldiff[qnm_phi_idx]> amp_reldiff[qnm_amp_idx]:
	        qnm_amp_idx = np.amin(np.intersect1d(np.where(amp_reldiff<0.001), np.where(time>=time[qnm_phi_idx])))  
	t_qnm_amp = round(time[qnm_amp_idx],2)
	t_qnm_phi = time[qnm_phi_idx]
	return [t_qnm_amp, t_qnm_phi]
	

def merger_info_plot1(wfdir, outdir, time, y, locate_merger, r=75): #r - radius of extraction

	datadir = DataDir(wfdir, outdir)
	if locate_merger:
	    t_maxamp = maxamp_time(datadir, outdir)[1] + (r-75)
	    maxamp_idx = np.amin(np.where(time>=t_maxamp))
	    
	    t_qnm = np.amax(np.array(qnm_time(datadir, outdir)))+ (r-75)
	    qnm_idx = np.amin(np.where(time>=t_qnm))
	    
	    t_hrzn = func_t_hrzn(datadir, locate_merger)+ (r-75)
	    hrzn_idx = np.amin(np.where(time>=t_hrzn))
	    
	    dict_y = {'t1_max_amp':time[maxamp_idx],'t1_hrzn':time[hrzn_idx], 't1_qnm': time[qnm_idx], 'y1_maxamp': y[maxamp_idx], 'y1_hrzn': y[hrzn_idx], 'y1_qnm':y[qnm_idx], 'locate_merger':locate_merger, 'num_objects':1}
	else:
	    dict_y = {'locate_merger':locate_merger}
        return dict_y


def merger_info_plot2(wfdir, outdir, time1, y1, time2, y2, locate_merger, r=75):

	datadir = DataDir(wfdir, outdir)
	if locate_merger:
	    t_maxamp = maxamp_time(datadir, outdir)[1] +(r-75)
	    maxamp_idx1 = np.amin(np.where(time1>=t_maxamp))
	    maxamp_idx2 = np.amin(np.where(time2>=t_maxamp))
	    
	    t_qnm = np.amax(np.array(qnm_time(datadir, outdir)))+ (r-75)
	    qnm_idx1 = np.amin(np.where(time1>=t_qnm))
	    qnm_idx2 = np.amin(np.where(time2>=t_qnm))
	    
	    t_hrzn = func_t_hrzn(datadir, locate_merger)+ (r-75)
	    hrzn_idx1 = np.amin(np.where(time1>=t_hrzn))
	    hrzn_idx2 = np.amin(np.where(time2>=t_hrzn))
	    
	    dict_y = {'t1_max_amp':time1[maxamp_idx1],'t1_hrzn':time1[hrzn_idx1], 't1_qnm': time1[qnm_idx1], 'y1_maxamp': y1[maxamp_idx1], 'y1_hrzn': y1[hrzn_idx1], 'y1_qnm':y1[qnm_idx1], 't2_max_amp':time2[maxamp_idx2],'t2_hrzn':time2[hrzn_idx2], 't2_qnm': time2[qnm_idx2], 'y2_maxamp': y2[maxamp_idx2], 'y2_hrzn': y2[hrzn_idx2], 'y2_qnm':y2[qnm_idx2], 'locate_merger':locate_merger, 'num_objects':2}

	else:
	    dict_y = {'locate_merger':locate_merger}

        return dict_y


def merger_info_plot3(wfdir, outdir, time1, y1, time2, y2, time3, y3, locate_merger, r=75):

	datadir = DataDir(wfdir, outdir)
	if locate_merger:
	    t_maxamp = maxamp_time(datadir, outdir)[1] +(r-75)
	    maxamp_idx1 = np.amin(np.where(time1>=t_maxamp))
	    maxamp_idx2 = np.amin(np.where(time2>=t_maxamp))
	    maxamp_idx3 = np.amin(np.where(time3>=t_maxamp))
	    
	    t_qnm = np.amax(np.array(qnm_time(datadir, outdir)))+ (r-75)
	    qnm_idx1 = np.amin(np.where(time1>=t_qnm))
	    qnm_idx2 = np.amin(np.where(time2>=t_qnm))
	    qnm_idx3 = np.amin(np.where(time3>=t_qnm))
	    
	    t_hrzn = func_t_hrzn(datadir, locate_merger)+ (r-75)
	    hrzn_idx1 = np.amin(np.where(time1>=t_hrzn))
	    hrzn_idx2 = np.amin(np.where(time2>=t_hrzn))
	    hrzn_idx3 = np.amin(np.where(time3>=t_hrzn))
	    
	    dict_y = {'t1_max_amp':time1[maxamp_idx1],'t1_hrzn':time1[hrzn_idx1], 't1_qnm': time1[qnm_idx1], 'y1_maxamp': y1[maxamp_idx1], 'y1_hrzn': y1[hrzn_idx1], 'y1_qnm':y1[qnm_idx1], 't2_max_amp':time2[maxamp_idx2],'t2_hrzn':time2[hrzn_idx2], 't2_qnm': time2[qnm_idx2], 'y2_maxamp': y2[maxamp_idx2], 'y2_hrzn': y2[hrzn_idx2], 'y2_qnm':y2[qnm_idx2], 't3_max_amp':time3[maxamp_idx3],'t3_hrzn':time3[hrzn_idx3], 't3_qnm': time3[qnm_idx3], 'y3_maxamp': y3[maxamp_idx3], 'y2_hrzn': y3[hrzn_idx3], 'y3_qnm':y3[qnm_idx3], 'locate_merger':locate_merger, 'num_objects':2}

	else:
	    dict_y = {'locate_merger':locate_merger}

        return dict_y
