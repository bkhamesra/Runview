import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
rc('font', **{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['lines.linewidth']=2


def Energy_Momentum(wfdir, outdir):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
	psi4analysis = os.path.join(datadir,"psi4analysis_r75.00.asc")
	time, Energy, px, py, pz, E_der, px_der, py_der, pz_der, jx_der, jy_der, jz_der, jx, jy, jz = np.loadtxt(psi4analysis, comments="#", unpack = True)
	
	Pmag = np.sqrt(px**2. + py**2. + pz**2)
	Jmag = np.sqrt(jx**2. + jy**2. + jz**2.)
	Jder = np.sqrt(jx_der**2. + jy_der**2. + jz_der**2.)


#  Energy and dE/dt plot
	energyplot = plot1(time, Energy, 'Time', 'Energy', 'Energy', figdir)
	Ederplot = plot1(time, E_der, 'Time', r'$\frac{dE}{dt}$', 'Energy_derivative', figdir)


#  Angular Momentum and dJ/dt plots	
	
	Jplot = plot1(time, Jmag, 'Time', r'$|\textbk{J}|$', 'AngMom',figdir)

	Jderplot = plot1(time, Jder, 'Time', r'$|\frac{dJ}{dt}|$', 'AngMomDer', figdir)
	
	Jzplot	= plot1(time, jz, 'Time', r'$J_z$','AngMom_z',figdir )
	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	jx, = ax1.plot(time, jx, 'b',label='Jx', linewidth=2)
	ax1.plot(time, jy,'k', linewidth=2, label='Jy')

	ax1.set_ylabel('J', fontsize = 14)
	ax1.set_xlabel('Time', fontsize = 14)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(figdir+'/AngMomentum_components.png', dpi = 500)
	plt.close()

#  Momentum plots

	Momplot = plot1(time, Pmag, 'Time',r'$|\textbk{P}|$', 'Momentum_mag', figdir)	

	Momzplot = plot1(time, pz, 'Time', 'Pz', 'Momentum_z', figdir)

	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	px, = ax1.plot(time, px, 'b',label='Px', linewidth=2)
	ax1.plot(time, py,'k', linewidth=2, label='Py')

	ax1.set_ylabel('P', fontsize = 14)
	ax1.set_xlabel('Time', fontsize = 14)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(figdir+'/Momentum_components.png', dpi = 500)
	plt.close()
