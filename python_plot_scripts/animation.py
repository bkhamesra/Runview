import matplotlib.pyplot as plt
from matplotlib import animation
from CommonFunctions import DataDir
import numpy as np
import os

def BH_Motion(dirpath, outdir):

	datadir = DataDir(dirpath, outdir)

	time, x, y, z = np.loadtxt(os.path.join(datadir, 'ShiftTracker0.asc'), unpack=True, usecols=(1,2,3,4))
	xmin, xmax = np.amin(x), np.amax(x)
	ymin, ymax = np.amin(y), np.amax(y)
	plt.plot(x,y)
	plt.show()
	plt.close()

	fig = plt.figure()
	ax = plt.axes(xlim = (xmin-1, xmax+1), ylim = (ymin-1, ymax+1))
	line, = ax.plot([],[], lw=2, color='b', marker='o')

	
	def init():
		line.set_data([],[])
		return line,

	def animate(i):	
		line.set_data(x[:i],y[:i])
		return line, 
	
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=range(200), interval=30, blit=False, repeat=False)
	plt.show()


		
