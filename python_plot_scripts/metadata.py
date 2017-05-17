from init_data import initial_data
import numpy as np
import os
from metadata_functions import *
from eccentricity import *
import time

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

def metadata(wfdir, outdir):
	
	datadir = DataDir(wfdir, outdir)
	
# Required Files	
	filename = wfdir.split("/")[-1]
	parfile = os.path.join(datadir, (filename+'.par'))
		
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
	

 	#Computing eccentricity and mean anomaly
	[mean_anomaly, eccentricity] = ecc_and_anomaly(dirpath, 75.)

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
	nr_metadata['type'] = 'NRinjection'		
	nr_metadata['simulation-type'] = simtype
	nr_metadata['name'] = simulation_name(dirpath)
	nr_metadata['alternative-names'] = filename
	nr_metadata['modification-date'] = time.strftime("%Y-%m-%d")
	nr_metadata['Contact Person'] = 'Deirdre Shoemaker'
	nr_metadata['point-of-contact-email'] = 'deirdre.shoemaker@physics.gatech.edu'
	nr_metadata['license'] = 'public'
	nr_metadata['INSPIRE-bibtex-keys'] = 'Jani:2016wkt'	
	nr_metadata['NR-techniques'] = 'Puncture-ID, BSSN, Psi4-integrated, Finite-Radius-Waveform, ApproxKillingVector-Spin, Christodoulou-Mass'	#check with Pablo/Deirdre
	
	if Error_Series==True:
		nr_metadata['files-in-error-series'] = '\'\''	#need to check for each simulation
		nr_metadata['comaparable-simulations'] = '\'\''
		print("*(Metadata)>> Please enter the name of simulation and error series manually to metadata.py")
	else:
		nr_metadata['files-in-error-series'] = -1
		nr_metadata['comparable-simulation'] = comp_sim	#need to check for each simulation

	nr_metadata['production-run'] = 1		#set to 0 for lower resolution runs
	nr_metadata['object1'] = 'BH'
	nr_metadata['object2'] = 'BH'
	nr_metadata['init_sep'] = init_sep
	nr_metadata['mass1'] = round(m_plus, 8)
	nr_metadata['mass2'] = round(m_minus, 8)
	nr_metadata['eta'] = round(eta, 8)
	nr_metadata['spin1x'] = round(spin1[0]/m_plus**2, 8)
	nr_metadata['spin1y'] = round(spin1[1]/m_plus**2, 8)
	nr_metadata['spin1z'] = round(spin1[2]/m_plus**2, 8)
	nr_metadata['spin2x'] = round(spin2[0]/m_minus**2, 8)
	nr_metadata['spin2y'] = round(spin2[1]/m_minus**2, 8)
	nr_metadata['spin2z'] = round(spin2[2]/m_minus**2, 8)
	nr_metadata['nhatx'] = round(nhat[0], 8)
	nr_metadata['nhaty'] = round(nhat[1], 8)
	nr_metadata['nhatz'] = round(nhat[2], 8)
	nr_metadata['LNhatx'] = round(Lhat[0], 8)
	nr_metadata['LNhaty'] = round(Lhat[1], 8)
	nr_metadata['LNhatz'] = round(Lhat[2], 8)
	nr_metadata['PN_approximant'] = 'None'
        nr_metadata['eccentricity'] = eccentricity
	nr_metadata['mean_anomaly'] = mean_anomaly

	nr_metadata['Warning1'] = warning1
	nr_metadata['Warning2'] = warning2
	return nr_metadata


