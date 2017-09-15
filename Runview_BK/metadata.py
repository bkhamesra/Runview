from init_data import initial_data
import numpy as np
import os, glob 
from eccentricity import *
import time
from CommonFunctions import *
from Psi4 import maxamp_time

def mag(vector):

	magnitude = np.sqrt(vector[0]**2. + vector[1]**2. + vector[2]**2.)
	return magnitude

def simulation_name(dirpath):
	
	file_name = dirpath.split('/')[-1]
	wf_junkrad = 'wf_junkrad.txt'
    
	wfdata = np.genfromtxt(wf_junkrad, dtype=None, comments='#', usecols=(0,1), skip_header=1, delimiter = '\t', names = ('GTID', 'simname'))
    	GTname, wfname = wfdata['GTID'], wfdata['simname']

	if np.array(np.where(wfname==file_name)).size==0:
		raise ValueError('*(metadata) >> GT simulation name incorrectly listed in wf_junkrad.txt. Please check the file.')
 	elif np.array(np.where(wfname==file_name)).size>1:
		raise ValueError('*(metadata) >> Multiple occurences of GT simulation - {} in wf_junkrad.txt - Please check the file.'.format(file_name))
	else: 
		idx = (np.where(wfname == file_name))[0][0]
		return GTname[idx]



def simulation_type(spin1, spin2):

	if (np.count_nonzero(spin1) ==0 and np.count_nonzero(spin2)==0): 
		simtype = 'non-spinning'
	elif (np.count_nonzero(spin1[0:2])>0 or np.count_nonzero(spin2[0:2])>0) :
		simtype = 'precessing'
	else:							
		simtype = 'aligned-spins'
	
	return simtype

def metadata(wfdir, outdir, locate_merger):
	
	datadir = DataDir(wfdir, outdir)
	
# Required Files	
	filename = wfdir.split("/")[-1]
	print datadir
	parfile = glob.glob(os.path.join(datadir, '*.par'))[0]
		
# Required Information from Parameter file
	spin1 = np.empty(3)
	spin2 = np.empty(3)
	p1 = np.empty(3)
	p2 = np.empty(3)
	r1 = np.empty(3)
	r2 = np.empty(3)
	warning1 = ''
	warning2 = ''

	initdata = initial_data(parfile)
	spin1 = initdata['spin_BH1']
	spin2 = initdata['spin_BH2']
	p1 = initdata['momentum_BH1']
	p2 = initdata['momentum_BH2']
	r1 = initdata['pos_BH1']
	r1 = initdata['pos_BH2']
	try:
		q = float((filename.split("q")[-1]).split("_")[0])
	except ValueError:
		q=1.

	m_plus = q/(1.+q) 
	m_minus = 1./(1.+q)
	eta = m_plus*m_minus/(m_plus+m_minus)**2.

	simtype = simulation_type(spin1,spin2) 
	
	delta_r = r2-r1
	init_sep = np.linalg.norm(delta_r)
	q = m_plus/m_minus
	eta = m_plus*m_minus/(m_plus + m_minus)**2.
	
	if locate_merger:
		maxamp, t_maxamp = maxamp_time(wfdir, outdir)
		t_hrzn = func_t_hrzn(datadir, locate_merger)

 	
		#Computing eccentricity and mean anomaly
	eccentricity = ecc_and_anomaly(datadir, 75.)[0]

	#print("*(metadata) >> Final Information \n")
	#print("*(metadata) >> Mass Ratio of system = {} \n".format(q))
	#print("*(metadata) >> Initial separation  = {} and nhat = {} \n".format(init_sep, nhat))
	#print("*(metadata) >> Momentum of BH1 = {}, and Momentum of BH2 = {} \n".format( p1, p2))
	
	#print("*(metadata) >> Spin of BH1 = {} and Spin of BH2={} \n".format( spin1, spin2))
	#print("*(metadata) >> Masses of BH: mplus = {}, m_minus = {} and q = {} \n". format(m_plus,m_minus, q ))
	#print("*(metadata) >> Orbital Angular Momentum vector is = {} \n".format( Lhat))
	#print("*(metadata) >> Mean Anomaly =  {} and Eccentricity = {} \n".format( mean_anomaly, eccentricity))
	
	nr_metadata = {}
	nr_metadata['NR-group'] = 'Georgia Tech'
	nr_metadata['NR-code'] = 'MAYA'
	nr_metadata['simulation-type'] = simtype
	#nr_metadata['name'] = simulation_name(dirpath)
	nr_metadata['alternative-names'] = filename
	nr_metadata['modification-date'] = time.strftime("%Y-%m-%d")
	nr_metadata['Contact Person'] = 'Deirdre Shoemaker'
	nr_metadata['point-of-contact-email'] = 'deirdre.shoemaker@physics.gatech.edu'
	nr_metadata['license'] = 'public'
	nr_metadata['INSPIRE-bibtex-keys'] = 'Jani:2016wkt'	
	nr_metadata['NR-techniques'] = 'Puncture-ID, BSSN, Psi4-integrated, Finite-Radius-Waveform, ApproxKillingVector-Spin, Christodoulou-Mass'	
	
	nr_metadata['object1'] = 'BH'
	nr_metadata['object2'] = 'BH'
	nr_metadata['init_sep'] = init_sep
	nr_metadata['mass1'] = round(m_plus, 8)
	nr_metadata['mass2'] = round(m_minus, 8)
	nr_metadata['mass-ratio'] = q
	nr_metadata['eta'] = round(eta, 8)
	nr_metadata['spin1'] = spin1
	nr_metadata['spin2'] = spin2
	nr_metadata['PN_approximant'] = 'None'
	if locate_merger:
		nr_metadata['final_horizon'] = t_hrzn
		nr_metadata['max_amp'] = t_maxamp
	#nr_metadata['eccentricity'] = eccentricity
	#nr_metadata['mean_anomaly'] = mean_anomaly

	return nr_metadata, parfile


