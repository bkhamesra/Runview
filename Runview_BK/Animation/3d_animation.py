import numpy as np
from scipy import integrate
import os
from CommonFunctions import DataDir
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from IPython.display import clear_output

N_BH = 2

datadir="/localdata2/bkhamesra3/simulations/Stampede/BBH/BBH_Ashtekar/BBH_Ashtekar_24Feb17_fullAHF_D6.2/BBH_Ashtekar_24Feb17_fullAHF_D6.2/BBH_Ashtekar_24Feb17_fullAHF_D6.2-all"

time_1, x1, y1, z1 = np.loadtxt(os.path.join(datadir, 'ShiftTracker0.asc'), unpack=True, usecols=(1,2,3,4))
time_2, x2, y2, z2 = np.loadtxt(os.path.join(datadir, 'ShiftTracker1.asc'), unpack=True, usecols=(1,2,3,4))

	
xmin, xmax = min(np.amin(x1),np.amin(x2)), max(np.amax(x1),np.amax(x2))
ymin, ymax = min(np.amin(y1),np.amin(y2)), max(np.amax(y1),np.amax(y2))
zmin, zmax = min(np.amin(z1),np.amin(z2)), max(np.amax(z1),np.amax(z2))


# Solve for the trajectories
x_t = np.empty((N_BH, min(len(time_1),len(time_2)), 3)) 		#np.asarray([integrate.odeint(lorentz_deriv, x0i, t)  for x0i in x0])
x_t[0] = (np.array((x1, y1, time_1))).T
x_t[1] = (np.array((x2, y2, time_2))).T

print np.shape(x_t)
# Set up figure & 3D axis for animation
fig = plt.figure()
ax = Axes3D(fig)		# fig.add_axes([0, 0, 1, 1], projection='3d')
#ax.axis('on')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_BH))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((xmin-1,xmax+1))
ax.set_ylim((ymin-1,ymax+1))
ax.set_zlim((time_1[0],time_1[-1]))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    i = (5 * i) % x_t.shape[1]
    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

#    ax.view_init(5, 0.001*i)
#    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=np.arange(0,len(time_1), 50), interval=1, blit=False)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()
