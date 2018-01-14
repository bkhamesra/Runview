import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
from Psi4 import maxamp_time, qnm_time 
#rc('font', **{'family':'serif','serif':['Computer Modern']})
#rc('text', usetex=True)
#mpl.rcParams['lines.linewidth']=2

def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled

def Energy_Momentum(wfdir, outdir, locate_merger=False):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
	psi4analysis = os.path.join(datadir,"psi4analysis_r75.00.asc")
	if not(os.path.exists(psi4analysis)):
	    debuginfo("%s file not found"%os.path.basename(psi4analysis))
	    return None

	time, Energy, px, py, pz, E_der, px_der, py_der, pz_der, jx_der, jy_der, jz_der, jx, jy, jz = np.loadtxt(psi4analysis, comments="#", unpack = True)
	
	Pmag = np.sqrt(px**2. + py**2. + pz**2)
	Jmag = np.sqrt(jx**2. + jy**2. + jz**2.)
	Jder = np.divide((jx_der*jx + jy_der*jy + jz_der*jz), Jmag)

	Jder = interpolate_gaps(Jder)  #To tackle any nans or infs


	dict_keys = ['Energy', 'Eder', 'Pmag', 'Pz', 'Jmag', 'Jz', 'Jder', 'Jderx', 'Jdery', 'Jderz']
	dict_vars = [Energy, E_der, Pmag, pz, Jmag, jz, Jder, jx_der, jy_der, jz_der]

	merger_dict = {}
	for k, v in zip(dict_keys, dict_vars):
 	    merger_dict[k] = merger_info_plot1(wfdir, outdir, time, v, locate_merger)	    


	# Energy and dE/dt plot
	energyplot = plot1(time, Energy, 'Time', 'Energy', 'Energy', figdir,  **merger_dict['Energy'])
	Ederplot = plot1(time, E_der, 'Time', 'dE/dt', 'Energy_derivative', figdir, **merger_dict['Eder'])


	# Angular Momentum and dJ/dt plots	
	
	Jplot = plot1(time, Jmag, 'Time', '|J|', 'AngMom',figdir, **merger_dict['Jmag'])

	Jderplot = plot1(time, Jder, 'Time', 'dJ/dt', 'AngMomDer', figdir, **merger_dict['Jder'])
	
        fig,(ax) = plt.subplots(1,1,sharex=True, squeeze=True)
	height = (fig.get_size_inches()*fig.dpi)[1] # size in pixels

	ax.plot(time, Jder, 'b', linewidth=1)
	ax.set_ylabel('dJ/dt', fontsize = 14)
	ax.set_xlabel('Time', fontsize = 14)
	ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4), fontsize=200)
	startx,endx = ax.get_xlim()
	starty,endy = ax.get_ylim()

	kwargs = merger_dict['Jder']	
	if locate_merger:
	
            delta_y = (-starty + endy)/height 
            
            plt.plot([kwargs['t1_max_amp'],kwargs['t1_max_amp']], [starty, kwargs['y1_maxamp']], 'k--', lw=1.5)
            plt.plot([kwargs['t1_hrzn'],kwargs['t1_hrzn']], [starty, kwargs['y1_hrzn']], 'k--', lw=1.5)
            plt.plot([kwargs['t1_qnm'],kwargs['t1_qnm']], [starty, kwargs['y1_qnm']], 'k--', lw=1.5)
            
            plt.text(kwargs['t1_max_amp'],kwargs['y1_maxamp']+delta_y*30,'Max Amp', horizontalalignment='left', fontsize=10)
            plt.text(kwargs['t1_hrzn'],kwargs['y1_hrzn']+delta_y*30,'AH3', horizontalalignment='right', fontsize=10)
            plt.text(kwargs['t1_qnm'],kwargs['y1_qnm']+delta_y*30,'QNM', horizontalalignment='left', fontsize=10)

            plt.annotate('(%.2f, %.2g)' % (kwargs['t1_hrzn'],kwargs['y1_hrzn']), xy=(kwargs['t1_hrzn'],kwargs['y1_hrzn']), xytext=(kwargs['t1_hrzn']-5,kwargs['y1_hrzn']+delta_y*10), textcoords='data', fontsize=10)
            plt.annotate('(%.2f, %.2g)' % (kwargs['t1_max_amp'],kwargs['y1_maxamp']), xy=(kwargs['t1_max_amp'],kwargs['y1_maxamp']), xytext=(kwargs['t1_max_amp'],kwargs['y1_maxamp']+delta_y*10), textcoords='data', fontsize=10)
            plt.annotate('(%.2f, %.2g)' % (kwargs['t1_qnm'],kwargs['y1_qnm']), xy=(kwargs['t1_qnm'],kwargs['y1_qnm']), xytext=(kwargs['t1_qnm'],kwargs['y1_qnm']+delta_y*10), textcoords='data', fontsize=10)
		
   	    ax.set_xlim(kwargs['t1_max_amp']-50, kwargs['t1_max_amp']+50)
	lgd = plt.legend()
	ax.grid(True)
	fig.savefig(os.path.join(figdir,('AngMomDer_zoom'+'.png')), dpi = 500)
	plt.close()

	
	Jzplot	= plot1(time, jz, 'Time', 'Jz','AngMom_z',figdir, **merger_dict['Jz'] )
	Jderzplot = plot1(time, jz_der, 'Time', 'dJz/dt','AngMomDer_z',figdir, **merger_dict['Jderz'] )
	
	
	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	jx_der, = ax1.plot(time, jx_der, 'b',label='dJx/dt', linewidth=1)
	ax1.plot(time, jy_der,'k', linewidth=1, label='dJy/dt')

	ax1.set_ylabel('Jder', fontsize = 18)
	ax1.set_xlabel('Time', fontsize = 18)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(figdir+'/AngMomentum_derivatives.png', dpi = 500)
	plt.close()

	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	jx, = ax1.plot(time, jx, 'b',label='Jx', linewidth=1)
	ax1.plot(time, jy,'k', linewidth=1, label='Jy')

	ax1.set_ylabel('J', fontsize = 18)
	ax1.set_xlabel('Time', fontsize = 18)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(figdir+'/AngMomentum_components.png', dpi = 500)
	plt.close()
#  Momentum plots

	Momplot = plot1(time, Pmag, 'Time','Mag(P)', 'Momentum_mag', figdir, **merger_dict['Pmag'])	

	Momzplot = plot1(time, pz, 'Time', 'Pz', 'Momentum_z', figdir, **merger_dict['Pz'])

	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	px, = ax1.plot(time, px, 'b',label='Px', linewidth=1)
	ax1.plot(time, py,'k', linewidth=1, label='Py')

	ax1.set_ylabel('P', fontsize = 18)
	ax1.set_xlabel('Time', fontsize = 18)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(figdir+'/Momentum_components.png', dpi = 500)
	plt.close()
