import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['lines.linewidth']=2

datadir = "/Users/Bhavesh/Documents/Research Work/Simulation/Event_Runs/Jan_4_17_Event/Event_Runs/BBH_Jan4Event_UID2_M160/"
outdir = "/Users/Bhavesh/Documents/Research Work/Simulation/Event_Runs/Jan_4_17_Event/Event_Runs/BBH_Jan4Event_UID2_M160/figures/"  
def energyplots(time_s,  Energy_s,  Eder_s):
	
	fig,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
	energy_s, = ax1.plot(time_s, Energy_s, 'b',label='BK', linewidth=2)
	ax2.plot(time_s, Eder_s,'b', linewidth=2)
#	ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
#	ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))

	r1 = ax1.set_ylabel('Energy', fontsize = 14)
	r2 = ax2.set_ylabel(r'$\frac{dE}{dt}$', fontsize = 16, labelpad = 12)
	ax2.set_xlabel('Time', fontsize = 14)
	r2.set_rotation(0)
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	
	lgd = plt.legend()#[energy_s, energy_b],['BK','(MC)'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)	
	ax1.grid(True)
	ax2.grid(True)
#	plt.show()
	#fig.savefig(outdir+'Energy_Comparison.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
	fig.savefig(outdir+'Energy_Comparison.png', dpi = 1000)
	plt.close()
	

def angmomplots(time_s, J_s, Jder_s):
	
	fig,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
	angmom_s, = ax1.plot(time_s, J_s, 'b',label='BK', linewidth=2)
	ax2.plot(time_s, Jder_s,'b', linewidth=2)
#	ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))


	r1 = ax1.set_ylabel(r'$|\textbk{J}|$', fontsize = 16)
	r2 = ax2.set_ylabel(r'$\frac{dJ}{dt}$', fontsize = 16, labelpad=10)
	ax2.set_xlabel('Time', fontsize = 14)
	r2.set_rotation(0)
#	ax1.yaxis.set_label_coords(-0.11, 0.5)
#	ax2.yaxis.set_label_coords(-0.13, 0.5)
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	
	lgd = plt.legend()#[angmom_s, angmom_b],['BK','(MC)'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	ax1.grid(True)
	ax2.grid(True)
#	plt.show()
	fig.savefig(outdir + 'AngMom_Comparison.png', dpi = 500)
	plt.close()


def momentumplots(time_s,  Px_s, Py_s):
	
	fig,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
	px_s, = ax1.plot(time_s, Px_s, 'b',label='BK', linewidth=2)
	ax2.plot(time_s, Py_s,'b', linewidth=2)
#	ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))

	r1 = ax1.set_ylabel(r'P_{x}', fontsize = 14)
	r2 = ax2.set_ylabel(r'P_{y}', fontsize = 16)
	
	ax2.set_xlabel('Time', fontsize = 14)
	r1.set_rotation(0)
	r2.set_rotation(0)
#	ax1.ticklabel_format(style='scientific')
#	ax2.ticklabel_format(style='scientific')
	#ax2.yaxis.tickFormat(d3.format('.02e'))	
	
	ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
	lgd = plt.legend()#[px_s, px_b],['BK','MC'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
	ax1.grid(True)
	ax2.grid(True)
#	plt.show()
	fig.savefig(outdir+'Momentum_Comparison.png', dpi = 500)
	plt.close()
	
	plt.plot(time_s, (Px_s**2 + Py_s**2)**0.5, 'b')
	plt.ylabel(r"P")	#, fontsize=14)
	plt.xlabel("Time")
	plt.legend()
	plt.grid(True)
#	plt.show()
	plt.savefig(outdir+'Pmag_Comparison.png', dpi=500)
	plt.close()	


#modification(check if required) - replace energy and energy flux plots using the commented parts
#energy_spmcs = (datadir + "ej_from_Psi4r_r50.00.asc")
#energy_bowen = (datadir + "ej_from_Psi4r_r50.00.asc")

psi4analysis_spmcs = (datadir+"psi4analysis_r75.00.asc")
psi4analysis_bowen = (datadir+"psi4analysis_r75.00.asc")

time1s, Es, pxs, pys, pzs, E_ders, px_ders, py_ders, pz_ders, jx_ders, jy_ders, jz_ders, jxs, jys, jzs = np.loadtxt(psi4analysis_spmcs, comments="#", unpack = True)

#time1b, Eb, pxb, pyb, pzb, E_derb, px_derb, py_derb, pz_derb, jx_derb, jy_derb, jz_derb, jxb, jyb, jzb = np.loadtxt(psi4analysis_bowen, comments="#", unpack = True)

#time2s, E_ders, Es, Jders, Js = np.loadtxt(energy_spmcs, comments="#", unpack = True)
#time2b, E_derb, Eb, Jderb, Jb = np.loadtxt(energy_bowen, comments="#", unpack = True)

energyplots(time1s,Es, E_ders)

momentumplots(time1s, pxs, pys)

Js = np.sqrt(jxs**2. + jys**2. + jzs**2.)
Jders = np.sqrt(jx_ders**2. + jy_ders**2. + jz_ders**2.)
angmomplots(time1s, Js, Jders )

