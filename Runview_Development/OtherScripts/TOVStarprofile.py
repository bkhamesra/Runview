
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import matplotlib.lines as lines

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#mpl.rcParams['lines.linewidth']=2

def stellarprofile(riso_s, riso_b, press_s, press_b, psi_s, psi_b, theta_s, theta_b):
	
	fig,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
	tov_s, = ax1.plot(riso_s, theta_s, 'b',label='BK', linewidth = 2)
	tov_b, = ax1.plot(riso_b, theta_b, 'g--',label='MC', linewidth =4)
	ax2.plot(riso_s, psi_s,'b', linewidth = 2)
	ax2.plot(riso_b, psi_b,'g--', linewidth = 4)
	
	r1 = ax1.set_ylabel(r'$\alpha*\psi$', fontsize = 16)
	r2 = ax2.set_ylabel(r'$\psi$', fontsize = 16)
	ax2.set_xlabel(r'$r_{iso}', fontsize = 16)
#	r2.set_rotation(0)
	
	lgd = plt.legend([tov_s, tov_b],['BK','MC'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)

	fig.savefig('TOV_Comparison.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 80)
	plt.close()
	
	plt.plot(riso_s, press_s, 'b', label = 'BK', linewidth=2)
	plt.plot(riso_b, press_b, 'g--', label = 'MC', linewidth = 4)
	plt.xlabel(r'$R_{iso}$')
	plt.ylabel('Pressure')
	plt.legend()
	plt.savefig('TOVPressure_Comparison')
	plt.close()

	
tov1_spmcs = ("BNS_SPMCS/tov_internaldata_0.asc")
tov2_spmcs = ("BNS_SPMCS/tov_internaldata_1.asc")
tov1_bowen = ("BNS_Bowen/tov_internaldata_0.asc")
tov2_bowen = ("BNS_Bowen/tov_internaldata_1.asc")

r1_s, press1_s, psi1_s, theta1_s = np.loadtxt(tov1_spmcs, comments="#", usecols = (0,1,2,3), unpack = True)
r2_s, press2_s, psi2_s, theta2_s = np.loadtxt(tov2_spmcs, comments="#", usecols = (0,1,2,3), unpack = True)

r1_b, press1_b, psi1_b, theta1_b = np.loadtxt(tov1_bowen, comments="#", usecols = (0,1,2,3), unpack = True)
r2_b, press2_b, psi2_b, theta2_b = np.loadtxt(tov2_bowen, comments="#", usecols = (0,1,2,3), unpack = True)


stellarprofile(r1_s, r1_b, press1_s, press1_b, psi1_s,psi1_b, theta1_s, theta1_b)

