import matplotlib.pyplot as plt
from matplotlib import animation
from CommonFunctions import DataDir
import numpy as np
import os

def BH_Motion(datadir):


	time_1, x1, y1, z1 = np.loadtxt(os.path.join(datadir, 'ShiftTracker0.asc'), unpack=True, usecols=(1,2,3,4))
	time_2, x2, y2, z2 = np.loadtxt(os.path.join(datadir, 'ShiftTracker1.asc'), unpack=True, usecols=(1,2,3,4))
	
	xmin, xmax = min(np.amin(x1), np.amin(x2)), max(np.amax(x1),np.amax(x2))
	ymin, ymax = min(np.amin(y1),np.amin(y2)), max(np.amax(y1),np.amax(y2))

	fig = plt.figure()
	ax1 = fig.add_subplot(111, autoscale_on=True, xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1))
#	ax2 = fig.add_subplot(211, autoscale_on=True, xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1))
	line, = ax1.plot([],[], lw=1, color='b', ls='--', marker='.')
	ax1.set_xlim(xmin-1,xmax+1)
	ax1.set_ylim(ymin-1,ymax+1)
#	scat = ax2.scatter(x1[0],y1[0],  color='b',  animated=True)
	colors = plt.cm.jet(np.linspace(0, 1, 2))
	i=0
	
	def init():
		line.set_data(x1[0],y1[0])
		#scat.set_array(x1[0],y1[0])
		return line,

	def animate(i):	
		thisx = [x1[:i], x2[:i]]
		thisy = [y1[:i], y2[:i]]
		line.set_data(thisx, thisy)
		#scat = plt.scatter(thisx, thisy)
		return line, 
	
	anim = animation.FuncAnimation(fig, animate,  frames=np.arange(0,len(time_1),100), interval=30, blit=False, repeat=False)
#	anim.save("BHmovie.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()


BH_Motion("/home/idealist/Downloads/Animation Python/SO_D9_q1.5_th2_135_ph1_135_m140/data")		
