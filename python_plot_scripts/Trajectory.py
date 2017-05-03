
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib
import glob

#Set MatPlotLib global parameters here
tick_label_size = 8
matplotlib.rcParams['xtick.labelsize'] = tick_label_size
matplotlib.rcParams['ytick.labelsize'] = tick_label_size

# Impot data from files
#datadir = "/Users/Bhavesh/Documents/Research Work/Simulation/Event_Runs/Jan_4_17_Event/Event_Runs/BBH_Jan4Event_UID2_M160/"
#outdir = "/Users/Bhavesh/Documents/Research Work/Simulation/Event_Runs/Jan_4_17_Event/Event_Runs/BBH_Jan4Event_UID2_M160/figures/"  

trajectory_BH1 = open(datadir + "ShiftTracker0.asc")
trajectory_BH2 = open(datadir + "ShiftTracker1.asc")
time_arr1, x_arr1, y_arr1, z_arr1 = np.loadtxt(trajectory_BH1, unpack=True, usecols=(1,2,3,4))
time_arr2, x_arr2, y_arr2, z_arr2 = np.loadtxt(trajectory_BH2, unpack=True, usecols=(1,2,3,4))

time_BH1 = time_arr1
x_BH1 = x_arr1
y_BH1 = y_arr1
time_BH2 = time_arr2
x_BH2 = x_arr2
y_BH2 = y_arr2

r1 = np.array((x_arr1, y_arr1, z_arr1))
r2 = np.array((x_arr2, y_arr2, z_arr2))
separation = np.linalg.norm(r2-r1, axis=0)

#Plot 1: x vs t and y vs t

f1,(plt2,plt3) = plt.subplots(2,1, squeeze=True, sharex = True)
#f.set_size_inches(18.5, 10.5)

BH1, = plt2.plot(time_BH1, x_BH1, 'g', linewidth=1)
BH2, = plt2.plot(time_BH2, x_BH2, 'k--', linewidth=1)
plt2.set_xlabel('Time', fontsize = 12)
plt2.set_ylabel('X', fontsize = 12)
#plt2.set_xlim(0,300)
startx,endx = plt2.get_xlim()
plt.xticks(np.arange(startx, endx, 50))
plt2.grid(True)

yvst1 = plt3.plot(time_BH1,y_BH1, 'g',linewidth=1)
yvst2 = plt3.plot(time_BH2, y_BH2, 'k--', linewidth=1)
plt3.set_xlabel('Time', fontsize = 12)
plt3.set_ylabel('Y', fontsize=12)
#plt3.set_xlim(0,300)
startx,endx = plt3.get_xlim()
plt.xticks(np.arange(startx, endx, 50))
plt3.grid(True)

lgd = plt.legend([BH1,BH2],['BH1','BH2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
f1.savefig(outdir + 'Trajectory_xyvstime.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
#plt.show()
plt.close()


#Plot 1: x vs t and y vs t


BH1, = plt.plot(time_BH1, x_BH1, 'g', linewidth=1, label="BH1")
BH2, = plt.plot(time_BH2, x_BH2, 'k--', linewidth=1, label = "BH2")
plt.xlabel('Time', fontsize = 12)
plt.ylabel('X', fontsize = 12)
#plt.xlim(0,300)
startx,endx = plt2.get_xlim()
plt.xticks(np.arange(startx, endx, 50))
plt.grid(True)
plt.legend()#[BH1,BH2],['BH1','BH2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
plt.savefig(outdir + 'Trajectory_xvstime.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
#plt.show()
plt.close()

plt.plot(time_BH1,y_BH1, 'g',linewidth=1, label = "BH1")
plt.plot(time_BH2, y_BH2, 'k--', linewidth=1, label = "BH2")
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Y', fontsize=12)
#plt.xlim(0,300)
startx,endx = plt3.get_xlim()
plt.xticks(np.arange(startx, endx, 50))
plt.grid(True)

plt.legend()#[BH1,BH2],['BH1','BH2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
plt.savefig(outdir + 'Trajectory_yvstime.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
#plt.show()
plt.close()

#Plot 2: Trajectory - y vs x
fig, ax = plt.subplots()
bh1 = ax.plot(x_BH1,y_BH1, color='g', linewidth=1, label="BH1")
bh2 = ax.plot(x_BH2,y_BH2, 'k--', linewidth=1, label="BH2")
ax.set_xlabel('X', fontsize = 12)
ax.set_ylabel('Y', fontsize = 12)
startx,endx = plt.gca().get_xlim()
#plt.xticks(np.arange(startx, endx, 0.5))
starty,endy = plt.gca().get_ylim()
#plt.yticks(np.arange(starty, endy, 0.5))
#plt.xticks(np.arange(-1.*abs(round(x_BH1[0])), abs(round(x_BH1[0])), 0.5))

plt.legend()
#plt.grid(True)
plt.savefig(outdir+'Trajectory_xy.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
#plt.show()
plt.close()

#Plot 3: Trajectory - separation vs time
plt.plot(time_BH1, separation, color='b', linewidth=1)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Separation', fontsize = 12)
startx,endx = plt.gca().get_xlim()
plt.xticks(np.arange(startx, endx, 50))
#starty,endy = plt.gca().get_ylim()
#plt.yticks(np.arange(starty, endy, 0.5))
#plt.xlim(0,300)
plt.grid(True)
plt.savefig(outdir+'Trajectory_separation.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
#plt.show()
plt.close()


#Plot 4: Combined
f1,(plt1, plt2,plt3) = plt.subplots(3,1, squeeze=True)
#f.set_size_inches(18.5, 10.5)

BH1, = plt1.plot(x_BH1,y_BH1, 'g', label = 'BH 1', linewidth=1)
BH2, = plt1.plot(x_BH2,y_BH2, 'k--', label = 'BH 2', linewidth=1)
plt1.set_ylabel('Y', fontsize = 12)
plt1.set_xlabel('X', fontsize = 12)
#plt1.set_xlim(-4,4)
plt1.grid(True)

xvst1, = plt2.plot(time_BH1, x_BH1, 'g', linewidth=1)
xvst2, = plt2.plot(time_BH2, x_BH2, 'k--', linewidth=1)
plt2.set_xlabel('Time', fontsize = 12)
plt2.set_ylabel('X', fontsize = 12)
plt2.grid(True)

yvst1 = plt3.plot(time_BH1,y_BH1, 'g',linewidth=1)
yvst2 = plt3.plot(time_BH2, y_BH2, 'k--', linewidth=1)
plt3.set_xlabel('Time', fontsize = 12)
plt3.set_ylabel('Y', fontsize=12)
plt3.grid(True)

lgd = plt.legend([BH1,BH2],['BH1','BH2'],bbox_to_anchor=(1, 1), loc='upper left', ncol=1, borderpad=0.8)
f1.savefig(outdir+'Trajectories.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 1000)
#plt.show()
plt.close()


