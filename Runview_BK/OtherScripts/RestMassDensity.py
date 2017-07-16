import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
mpl.rcParams['lines.linewidth']=2

def restmassdensity(time_s, time_b, rho_s, rho_b):
	plt.plot(time_s, rho_s,'b', label="BK")
	plt.plot(time_b, rho_b,'g', label="MC")
	plt.xlabel("Time")
	plt.ylabel(r'$\rho$')
	plt.legend()
	plt.savefig("Restmassdensity.png")
	plt.close


rho_spmcs = ("BNS_SPMCS/rho.wmaximum.asc")
rho_bowen = ("BNS_Bowen/rho.wmaximum.asc")

time_s, rho_s = np.loadtxt(rho_spmcs, comments="#", usecols = (1,2), unpack = True)
time_b, rho_b = np.loadtxt(rho_bowen, comments="#", usecols = (1,2), unpack = True)

restmassdensity(time_s, time_b, rho_s, rho_b)
