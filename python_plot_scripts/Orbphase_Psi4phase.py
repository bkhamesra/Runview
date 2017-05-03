import numpy as np
import glob 
import matplotlib.pyplot as plt
from shutil import copyfile
import os


def func_phase(phase):
	phi = np.copy(phase)
	for i in range(len(phase)):
		if (phase[i]<0 and phase[i-1]>0):
			phi[i:] = phi[i:] + np.pi
	#		print("Phase increased by phi at i = {}, phi[i] = {}, phase[i] = {}".format(i, phi[i], phase[i]))
	return phi
def time_shift(time, phase):
	t_idx = np.amin(np.where(time>=100))
	time = time[t_idx:] 
	time = time-50
	phase = phase[t_idx:]
	return[time, phase]

def phase_shift(t1,t2, amp, psiphase, orbphase):
	ampmax = np.amax(amp)	
	t1_idx = np.amin(np.where(amp == ampmax))
	t2_idx = np.amin(np.where(t2 >= t1[t1_idx]))
	
 	diff = orbphase[0] -  psiphase[0]
#	print t1[t1_idx], t2[t2_idx], psiphase[t1_idx],orbphase[t2_idx],diff
	psiphase = psiphase+diff

	return psiphase

def phaseplots(dirpath, output_dir):

	filepath = dirpath.split('/')[-2]
  	output_dir = output_dir + filepath
	print output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	psi4 = ("Ylm_WEYLSCAL4::Psi4r_l2_m2_r50.00.asc")
	if not os.path.exists(output_dir +"/" + psi4):
		copyfile(dirpath+"/"+psi4, output_dir+"/"+psi4)

	psi4_file = open(output_dir+"/"+psi4)
	t1, real, imag = np.loadtxt(psi4_file, unpack=True)

	#Amplitude and Phase
	amp = abs(real+1.j *imag)		# np.sqrt(real**2 + imag**2)
	psiphase = -np.unwrap(np.angle(real+1j*imag))
	t1, psiphase = time_shift(t1, psiphase)
	

	trajectory_BH1 = open(dirpath +"/"+ "ShiftTracker0.asc")
	t2, x_BH1, y_BH1, z_BH1 = np.loadtxt(trajectory_BH1, unpack=True, usecols=(1,2,3,4))

	phase = np.arctan(np.divide(y_BH1, x_BH1))
	orbphase = func_phase(phase)

 	diff =	-15.-3
	psiphase = psiphase+diff
	
	plt.plot(t2,orbphase, 'g', label = "Orbital Phase")
	plt.plot(t1,psiphase, 'r', label = "Psi4 Phase")
	plt.title(filepath, fontsize=12)
	plt.xlabel("Time")
	plt.ylabel('Phase')
	plt.xlim(200,300)
	plt.ylim(0,40)
	plt.legend(loc=4)
	plt.show()
	#plt.savefig(output_dir+"/Orbital_Psi4_phase.png", spi=500)
#	plt.savefig(output_dir+"/Orbital_Psi4_phase_zoom.png", spi=500)
	plt.close()


# Insert the waveform path and output dir with / at the end of path. 
wfpath ="/nethome/numrel/datafiles/Waveforms/S-series-v3/D8_q1.0_a0.6_0.6_th90_225_r100_res140_CE/D8_q1.0_a0.6_0.6_th90_225_r100_res140_CE-all" 
output_dir = "/nethome/bkhamesra3/Desktop/Horizon_runs/Horizon_Runs/output/"
phaseplots(wfpath, output_dir)
