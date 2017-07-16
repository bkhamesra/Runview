import matplotlib.pyplot as plt
from matplotlib import animation
from CommonFunctions import DataDir
import numpy as np
import os
N_BH = 2

def BH_Motion(datadir):


	time_1, x1, y1, z1 = np.loadtxt(os.path.join(datadir, 'ShiftTracker0.asc'), unpack=True, usecols=(1,2,3,4))
	time_2, x2, y2, z2 = np.loadtxt(os.path.join(datadir, 'ShiftTracker1.asc'), unpack=True, usecols=(1,2,3,4))
	
	xmin, xmax = min(np.amin(x1), np.amin(x2)), max(np.amax(x1),np.amax(x2))
	ymin, ymax = min(np.amin(y1),np.amin(y2)), max(np.amax(y1),np.amax(y2))
	
	data = np.empty((2, len(time_1), 2))
	data[0] = np.array((x1,y1)).T
	data[1] = np.array((x2,y2)).T

	fig, ax = plt.subplots()
	colors = plt.cm.jet(np.linspace(0, 1, N_BH))

	# set up lines and points
	lines = sum([ax.plot([], [], '-', c=c)
             for c in colors], [])
	pts = sum([ax.plot([], [], 'o', c=c)
           for c in colors], [])

	# prepare the axes limits
	ax.set_xlim((xmin-1,xmax+1))
	ax.set_ylim((ymin-1,ymax+1))

	
	def init():
    	    for line, pt in zip(lines, pts):
	        line.set_data([], [])

	        pt.set_data([], [])
    	    return lines + pts



	def animate(i):	
	
            for line, pt, d in zip(lines, pts, data):
		x,y = d[:i].T
		line.set_data(x,y)
		#thisx = [x1[:i], x2[:i]]
		#thisy = [y1[:i], y2[:i]]
		#lines.set_data(thisx, thisy)
		pt.set_data(x[-1:], y[-1:])
		#scat = plt.scatter(thisx, thisy)
		return lines + pts 
	
	anim = animation.FuncAnimation(fig, animate,  frames=np.arange(0,len(time_1),100), interval=30, blit=False, repeat=False)
#	anim.save("BHmovie.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
	plt.show()


BH_Motion("/home/idealist/Downloads/Animation Python/SO_D9_q1.5_th2_135_ph1_135_m140/data")		

