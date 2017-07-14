import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from CommonFunctions import *
rc('font', **{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['lines.linewidth']=2


def write_massspin_data(outdir, filename, data):

		output = open(os.path.join(outdir, filename+'.asc'),'w')
		hdr = '# Time \t Horizon Mass \t ax \t ay \t az \n'
		np.savetxt(output, data, header=hdr, delimiter='\t', newline='\n')
		output.close()


def horizon_mass(irr_m1, irr_m2, spin1, spin2):
	
	spin1_mag = np.linalg.norm(spin1, axis=1)
	spin2_mag = np.linalg.norm(spin2, axis=1)

	m1 = np.sqrt(irr_m1**2. +spin1_mag**2./(4.*irr_m1**2.))
	m2 = np.sqrt(irr_m2**2. +spin2_mag**2./(4.*irr_m2**2.))
	if np.any(m1)<0: 
		raise ValueError('Negative Masses Encountered for m1 = {}. Please check the data'.format(m1[np.where(m1<0)]))
	elif np.any(m2)<0:
		raise ValueError('Negative Masses Encountered for m1 = {} and m2={}. Please check the data'.format(m2[np.where(m2<0)]))
	return m1, m2

def Mass_Plots(wfdir, outdir):

	figdir = FigDir(wfdir, outdir)
	datadir = DataDir(wfdir, outdir)

	bh_diag0 = os.path.join(datadir,'BH_diagnostics.ah1.gp')
	bh_diag1 = os.path.join(datadir,'BH_diagnostics.ah2.gp')
	ihspin0 = os.path.join(datadir,"ihspin_hn_0.asc")
	ihspin1 = os.path.join(datadir,"ihspin_hn_1.asc")

	if os.path.exists(bh_diag1):
		time_bh1, irr_m1 = np.loadtxt(bh_diag0, usecols = (1,26), unpack=True, comments = '#')
		time_bh2, irr_m2 = np.loadtxt(bh_diag1, usecols = (1,26), unpack =True, comments = '#')
		minlen = min(len(time_bh1), len(time_bh2))-1
		
		plot2(time_bh1, irr_m1, time_bh2, irr_m2, 'Time', 'Irreducible Mass', "Irreducible_Mass", figdir)

	if os.path.exists(ihspin0):	
		t_bh1, sx1, sy1, sz1 = np.loadtxt(ihspin0, unpack=True, usecols=(0,1,2,3))
		t_bh2, sx2, sy2, sz2 = np.loadtxt(ihspin1, unpack=True, usecols=(0,1,2,3))
	
		spin1 = np.array((sx1, sy1, sz1)).T
		spin2 = np.array((sx2, sy2, sz2)).T

		try:
			max_t1_idx = np.amin(np.where(t_bh1>=time_bh1[-1]))
		except ValueError:
			max_t1_idx = len(t_bh1)
		try:
			max_t2_idx = np.amin(np.where(t_bh2>=time_bh2[-1]))
		except ValueError:
			max_t2_idx = len(t_bh2)

		t1_cutoff = t_bh1[:max_t1_idx]
		t2_cutoff = t_bh2[:max_t2_idx]

		match_time1_idx = np.empty(len(t1_cutoff))
		match_time2_idx = np.empty(len(t2_cutoff))
		for i,j in zip(range(len(t1_cutoff)), range(len(t2_cutoff))):
			match_time1_idx[i] = np.amin(np.where(time_bh1>=t1_cutoff[i]))		
			match_time2_idx[j] = np.amin(np.where(time_bh2>=t2_cutoff[j]))		
		
		match_time1 = time_bh1[match_time1_idx.astype(int)]
		match_time2 = time_bh2[match_time2_idx.astype(int)]
		dtime_bh1 = match_time1 - time_bh1[match_time1_idx.astype(int)-1]
		dtime_bh2 = match_time2 - time_bh2[match_time2_idx.astype(int)-1]

		time1_diff = time_bh1[match_time1_idx.astype(int)] - t1_cutoff
		dt_bh1 = t_bh1[1:] - t_bh1[:-1]
		dt_bh2 = t_bh2[1:] - t_bh2[:-1]
		if np.size(np.where((match_time1[1:] - match_time1[:-1])==0))>0:
			raise ValueError ("Repeated Value of time encountered in horizon mass computation \n")
		if np.any((time1_diff[1:]>dtime_bh1[1:])):
			print time1_diff - dtime_bh1
			raise ValueError("Time of spin computation not same as mass computation \n")
		
		spin1 = spin1[:max_t1_idx]
		spin2 = spin2[:max_t2_idx]	
		irr_m1 = irr_m1[match_time1_idx.astype(int)]
		irr_m2 = irr_m2[match_time2_idx.astype(int)]
		mass1, mass2 = horizon_mass(irr_m1, irr_m2, spin1, spin2)
		
		a1 = np.divide(spin1.T, mass1**2.)
		a2 = np.divide(spin2.T, mass2**2.)
		
		data1 = np.column_stack((t1_cutoff, mass1, a1[0], a1[1], a1[2]))
		data2 = np.column_stack((t2_cutoff, mass2, a2[0], a2[1], a2[2]))
		write_massspin_data(datadir, 'hn_mass_spin_0', data1)
		write_massspin_data(datadir, 'hn_mass_spin_1', data2)
