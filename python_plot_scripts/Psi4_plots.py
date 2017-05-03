import numpy as np
import matplotlib.pyplot as  plt
from matplotlib import pylab
from shutil import copyfile
import os

#Create waveform directory
dirpath ="/Users/Bhavesh/Documents/Research Work/Simulation/Event_Runs/Jan_4_17_Event/Event_Runs/BBH_Jan4Event_UID4_M120"
filepath = dirpath.split('/')[-1]
output_dir ="/Users/Bhavesh/Desktop/images/"	 #"/Users/Bhavesh/Documents/Research Work/Simulation/Event_Runs/Jan_4_17_Event/Event_Runs/" + filepath

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

fig_dir = output_dir + "/figures/"
data_dir = output_dir + "/data/"

if not os.path.exists(data_dir):
	os.makedirs(data_dir)
if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)

#Extract Psi4 info

psi4 = ("Ylm_WEYLSCAL4::Psi4_l2_m2_r75.00.asc")
print data_dir+psi4
if not os.path.exists(data_dir + psi4):
	copyfile(dirpath+"/"+psi4, data_dir+psi4)
psi4_file = open(data_dir+psi4)
time, real, imag = np.loadtxt(psi4_file, unpack=True)

#Amplitude and Phase
amp = abs(real+1.j *imag)		# np.sqrt(real**2 + imag**2)
phi = -np.unwrap(np.angle(real+1j*imag))
r =float( ((psi4.split('r'))[-1]).split('.asc')[0])
print ("r = ",r)

#Phase derivatives
tanphi = -1.*np.array(np.divide(imag,real))	#Phase
secphi = np.sqrt(1. + tanphi**2)
n = len(tanphi)
real_der_t = np.divide((real[1:n] - real[0:n-1]), (time[1:n] - time[0:n-1]))
imag_der_t = np.divide((imag[1:n] - imag[0:n-1]),(time[1:n] - time[0:n-1]))

phidot =-1.* (real[1:n]*imag_der_t - imag[1:n]*real_der_t) 
phidot = np.divide(phidot, (amp[1:n]**2))
time_red = time[1:n]

phidot_der = np.divide((phidot[1:phidot.size] - phidot[0:phidot.size-1]),(time_red[1:phidot.size] - time_red[0:phidot.size-1]) )

#Polynomial Fitting
#phidot_fit = np.poly1d( np.polyfit(time[1:n], phidot, 6))
#print np.polyfit(time[1:n], phidot, 6)

max_amp = np.amax(amp)
max_amp_index = np.where(amp == max_amp)[0]
t_max_amp = time[max_amp_index]
phi_at_maxamp = phi[np.where(time==t_max_amp)]
real_at_maxamp = real[np.where(time==t_max_amp)]
imag_at_maxamp = imag[np.where(time==t_max_amp)]

if real_at_maxamp>=imag_at_maxamp: maxpsi4 = real_at_maxamp
else: maxpsi4 = imag_at_maxamp

t_horizon3 =  156.405333 + r	#Modify 
amp_hrzn = amp[np.amin(np.where(time>=t_horizon3))]
phi_hrzn =phi[np.amin(np.where(time>=t_horizon3))]
real_hrzn = real[np.amin(np.where(time>=t_horizon3))]
imag_hrzn = imag[np.amin(np.where(time>=t_horizon3))]
if real_hrzn>=imag_hrzn: psi4_hrzn = real_hrzn
else: psi4_hrzn = imag_hrzn
print("AH3 detected at {} and Amplitude maxima at {} \n".format(t_horizon3, t_max_amp))

#Plot1: Psi4 -  real and imaginary vs time
plt.plot(time,real, 'b', label = "Real")
plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Psi4")
startx,endx = plt.gca().get_xlim()
#plt.xticks(np.arange(startx, endx, 50))
plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig(fig_dir+"Psi4_plot.png", dpi = 1000)
plt.close()

# Plot2: Psi4 - real and imaginary - near merger
plt.plot(time,real, 'b', label = "Real")
plt.plot(time, imag, 'g--', label = "Imaginary", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Psi4")
#plt.plot([t_max_amp,t_max_amp], [0,maxpsi4], 'k', linewidth =1.5)
#plt.text(t_max_amp,maxpsi4+0.00005,'Max Amplitude', horizontalalignment='center', fontsize=9)
#plt.plot([t_horizon3,t_horizon3], [0,psi4_hrzn], 'k', linewidth=1.5)
#plt.text( t_horizon3,psi4_hrzn + 0.00005,'AH3', horizontalalignment='center', fontsize=9)
plt.xlim(80,250)
startx,endx = plt.gca().get_xlim()
plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
plt.xticks(np.arange(startx, endx, 10))
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig(fig_dir+"Psi4_plot_zoomed.png", dpi = 1000)
plt.close()

print time[np.amin(np.where(time>=t_horizon3))]

#Plot 3: Psi4 - phase and Amplitude
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)

Psi4_amp = ax1.plot(time,amp, 'b')
#ax1.plot([t_max_amp,t_max_amp], [0,max_amp], 'k--', linewidth =1.5)
#ax1.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
#ax1.plot([t_horizon3,t_horizon3], [0,amp_hrzn], 'k--', linewidth=1.5)
#ax1.text( t_horizon3,amp_hrzn,'AH3', horizontalalignment='right', fontsize=9)
ax1.set_ylabel("|Psi4|")
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
startx,endx = ax1.get_xlim()
#plt.xticks(np.arange(startx, endx, 50))

Psi4_phase = ax2.plot(time, phi )
#plt.plot(time, phidot_fit(time))
#ax2.plot([t_max_amp,t_max_amp], [-20, phi[np.where(time == t_max_amp)]], 'k--', linewidth=1.5)
#ax2.text(t_max_amp,phi_at_maxamp+10 ,'Max\nAmp', horizontalalignment='center', fontsize=9)
#ax2.plot([t_horizon3,t_horizon3], [-20,phi[np.amin(np.where(time>=t_horizon3))]], 'k--', linewidth=1.5)
#ax2.text( t_horizon3,phi_hrzn+5,'AH3', horizontalalignment='right', fontsize=9)
ax2.set_xlabel("Time")
ax2.set_ylabel("Phase")
startx,endx = ax2.get_xlim()
#plt.xticks(np.arange(startx, endx, 50))
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig(fig_dir+"Psi4_amp_phase.png", dpi = 1000)
plt.close()

#Plot 3: Psi4 - phase and Amplitude

plt.plot(time,amp, 'b')
#plt.plot([t_max_amp,t_max_amp], [0,max_amp], 'k--', linewidth =1.5)
#plt.text(t_max_amp,max_amp+0.00003,'Max Amplitude', horizontalalignment='center', fontsize=9)
#plt.plot([t_horizon3,t_horizon3], [0,amp_hrzn], 'k--', linewidth=1.5)
#plt.text( t_horizon3,amp_hrzn,'AH3', horizontalalignment='right', fontsize=9)
plt.ylabel("|Psi4|")
plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,4))
startx,endx = ax1.get_xlim()
#plt.xticks(np.arange(startx, endx, 50))
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig(fig_dir+"Psi4_amp.png", dpi = 1000)
plt.close()


plt.plot(time, phi )
#plt.plot(time, phidot_fit(time))
#plt.plot([t_max_amp,t_max_amp], [-20, phi[np.where(time == t_max_amp)]], 'k--', linewidth=1.5)
#plt.text(t_max_amp,phi_at_maxamp+10 ,'Max\nAmp', horizontalalignment='center', fontsize=9)
#plt.plot([t_horizon3,t_horizon3], [-20,phi[np.amin(np.where(time>=t_horizon3))]], 'k--', linewidth=1.5)
#plt.text( t_horizon3,phi_hrzn+5,'AH3', horizontalalignment='right', fontsize=9)
plt.xlabel("Time")
plt.ylabel("Phase")
#ax2.set_xlim(1300,1500)
#ax2.set_ylim(-1,1)
startx,endx = ax2.get_xlim()
#plt.xticks(np.arange(startx, endx, 50))
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig(fig_dir+"Psi4_phase.png", dpi = 1000)
plt.close()

#Plot 4: Psi4 - omega (phidot)
plt.scatter(time_red,phidot)
plt.xlabel("Time")
plt.ylabel(r"$\omega$")
#plt.xlim(0,500)
#plt.show()
#plt.savefig(fig_dir+"Psi4_Omega.png")
plt.close()
