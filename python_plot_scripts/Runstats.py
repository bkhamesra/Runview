import numpy as np
import  matplotlib.pyplot as plt
import os,glob

def output_data(parfile, data):
	
	datafile = file(parfile)	
	datafile.seek(0)
	for line in datafile:
		if data in line:
			break
	line = line.split()
	data_value = float(line[-1])
	datafile.close()
	return data_value

def runstats(direc):

	outdir = direc
	filename = outdir.split('/')[-1]

	runstat_file = outdir + "/runstats.asc"
	runstat = open(runstat_file)
	
	parfile = glob.glob(outdir + "/*.par")[0]
#	parfile = open(parfile)
	
	series = filename.split("_")[0]
	q = float((filename.split("q")[-1]).split("_")[0])
	res = float((filename.split("m")[-1]).split("_")[0])

	sep = 2.*output_data(parfile, "par_b")

	spin1 = np.empty(3)
	spin2 = np.empty(3)
	spin1[0] = output_data(parfile, 'par_s_plus[0]')
	spin1[1] = output_data(parfile, 'par_s_plus[1]')
	spin1[2] = output_data(parfile, 'par_s_plus[2]')
	spin2[0] = output_data(parfile, 'par_s_minus[0]')
	spin2[1] = output_data(parfile, 'par_s_minus[1]')
	spin2[2] = output_data(parfile, 'par_s_minus[2]')

	if (np.count_nonzero(spin1)==0 and np.count_nonzero(spin2)==0):
		spintype = "Non-Spinning"
	elif (np.count_nonzero(spin1[0:2])==0 and np.count_nonzero(spin2[0:2])==0):
		spintype = "Aligned Spins"
	else:
		spintype = "Precessing"

		
	iteration,coord_time, walltime, speed, period, cputime = np.loadtxt(runstat, unpack=True, usecols=(0,1,2,3,4,5))
	
	walltime_hrs = walltime/3600.
	day = 0	
	for i in range(len(walltime_hrs)):
		if walltime_hrs[i]<walltime_hrs[i-1]:
			day=day+walltime_hrs[i-1]/24.

	#print coord_time[-1]/(24*day)

	avg_speed = np.mean(speed)
	cputime_total = cputime[-1]
	total_days = day

	print("{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n".format(series, q, sep, res, spintype, avg_speed, cputime_total, total_days))

	plt.plot(coord_time, speed)
	plt.xlabel('Time')
	plt.ylabel('Speed')
	plt.title("Run Stats")
	#plt.show()
	plt.close()
	
	
	plt.plot(walltime_hrs, coord_time)
	plt.ylabel("Period")
	plt.xlabel("Coord Time")
	plt.ylim(0,100)
	plt.title("Coord Time vs Walltime")
	#plt.show()
	plt.close()


wf_direc ="/Users/Bhavesh/Documents/ResearchWork/Simulation/benchmarking" 
print("Series \t Mass Ratio \t Separation \t Resolution \t Spin Type \t Average Speed (hour^-1) \t Total CPU Hours (SU) \t Total Wall-Time days")
#runstats(wf_direc)
for direc in os.listdir(wf_direc):
 	direc_path = os.path.join(wf_direc,direc)
	if os.path.isdir(direc_path):
		#print direc
		runstats(direc_path)
