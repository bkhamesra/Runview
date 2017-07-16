import numpy as np
import os
from CommonFunctions import DataDir
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

f1,x1,y1,z1 = np.loadtxt('/localdata2/bkhamesra3/simulations/Stampede/BBH/BBH_Ashtekar/BBH_Ashtekar_24Feb17_fullAHF_D6.2/BBH_Ashtekar_24Feb17_fullAHF_D6.2/BBH_Ashtekar_24Feb17_fullAHF_D6.2-all/HorizonData/h.t0.ah1.gp', unpack=True, usecols=(2,3,4,5,))

#x1,y1 = np.meshgrid(x1, y1)
#r = np.sqrt(x1**2 + y1**2)
#
#surf = ax.plot_surface(x1,y1,z1, cmap=cm.binary, lw=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
#plt.close()

ax.plot_wireframe(y1,z1,f1)
plt.show()
plt.close()


