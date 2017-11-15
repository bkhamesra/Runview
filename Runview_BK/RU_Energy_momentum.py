import plotly.offline as ply #different tag than everywhere else due to variable naming
import plotly.graph_objs as go
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
#rc('font', **{'family':'serif','serif':['Computer Modern']})
#rc('text', usetex=True)
#mpl.rcParams['lines.linewidth']=2


def Energy_Momentum(wfdir, outdir):

	statfigdir,dynfigdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)
   
	psi4analysis = os.path.join(datadir,"psi4analysis_r75.00.asc")
	if not(os.path.exists(psi4analysis)):
	    debuginfo("%s file not found"%os.path.basename(psi4analysis))
	    return None

	time, Energy, px, py, pz, E_der, px_der, py_der, pz_der, jx_der, jy_der, jz_der, jx, jy, jz = np.loadtxt(psi4analysis, comments="#", unpack = True)
	
	Pmag = np.sqrt(px**2. + py**2. + pz**2)
	Jmag = np.sqrt(jx**2. + jy**2. + jz**2.)
	Jder = np.sqrt(jx_der**2. + jy_der**2. + jz_der**2.)


#  Energy and dE/dt plot
	energyplot = plot1(time, Energy, 'Time', 'Energy', 'Energy', statfigdir) #MPL
	Ederplot = plot1(time, E_der, 'Time', 'dE/dt', 'Energy_derivative', statfigdir) #MPL
	plyenergyplot = plyplot1(time, Energy, 'Time', 'Energy', 'Energy') #PLY
	plyEderplot = plyplot1(time,E_der,'Time','dE/dt','Energy Derivative') #PLY
	ply.plot(plyenergyplot, filename=dynfigdir + "Energy.html")
	ply.plot(plyEderplot, filename=dynfigdir + "Energy_Derivative.html")
	
	

#  Angular Momentum and dJ/dt plots
	Jplot = plot1(time, Jmag, 'Time', '|J|', 'AngMom',statfigdir)
	plyJplot= plyplot1(time, Jmag,'Time', 'Jmag', 'Angular Momentum') #for details, see common functions; RU
	ply.plot(plyJplot,filename= dynfigdir + "AngMom.html") #basic plot method, object + path/filename
	
	Jderplot = plot1(time, Jder, 'Time', 'dJ/dt', 'AngMomDer', statfigdir)
	plyJderplot = plyplot1(time, Jder, 'Time', "dJ/dt","Derivative of Anular Momentum")
	ply.plot(plyJderplot,filename= dynfigdir + "AngMomDer.html")
	
	
	Jzplot	= plot1(time, jz, 'Time', 'Jz','AngMom_z',statfigdir )
	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	jx, = ax1.plot(time, jx, 'b',label='Jx', linewidth=1)
	ax1.plot(time, jy,'k', linewidth=1, label='Jy')

	ax1.set_ylabel('J', fontsize = 18)
	ax1.set_xlabel('Time', fontsize = 18)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(statfigdir+'/AngMomentum_components.png', dpi = 500)
	plt.close()

	plyJzplot = plyplot1(time, jz, 'Time', 'Jz', 'Z Component of Angular Momentum')
	ply.plot(plyJzplot,filename= dynfigdir + "AngMom_z.html")

#  Momentum plots

	Momplot = plot1(time, Pmag, 'Time','Mag(P)', 'Momentum_mag', statfigdir)
	plyMomplot = plyplot1(time, Pmag, "Time", "Mag(P)", "Momentum Magnitude")
	ply.plot(plyMomplot,filename= dynfigdir + "Momentum_mag.html")
	
	Momzplot = plot1(time, pz, 'Time', 'Pz', 'Momentum_z', statfigdir)
	plyMomzplot = plyplot1(time, pz, "Time", "Pz", 'Z Component of Momentum')
	ply.plot(plyMomzplot,filename= dynfigdir + "Momentum_z.html")
	
	fig,(ax1) = plt.subplots(1,1,sharex=True, squeeze=True)
	px, = ax1.plot(time, px, 'b',label='Px', linewidth=1)
	ax1.plot(time, py,'k', linewidth=1, label='Py')

	ax1.set_ylabel('P', fontsize = 18)
	ax1.set_xlabel('Time', fontsize = 18)
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()
	ax1.grid(True)
	fig.savefig(statfigdir+'/Momentum_components.png', dpi = 500)
	plt.close()

outDirSO = "/home/rudall/Runview/TestCase/OutputDirectory/SOetc_2/"
binSO = "/home/rudall/Runview/TestCase/BBH/SO_D9_q1.5_th2_135_ph1_90_m140/"
binQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_l11_M192-all/"
outDirQC = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_2/"

Energy_Momentum(binQC, outDirQC)