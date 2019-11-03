
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib as mpl
import glob, math, os
from CommonFunctions import *

#Set MatPlotLib global parameters here
tick_label_size = 14
mpl.rcParams['xtick.labelsize'] = tick_label_size
mpl.rcParams['ytick.labelsize'] = tick_label_size


#Determinant of surface metric

def QLM_DeterminantPlots(wfdir, outdir, tlast):  
    '''This function checks and produces movies of variation of metric determinant of MOTS and horizons. 
     wfdir - waveform directory, outdir - output directory, tlast - last time iteration to be plotted '''
    figdir = FigDir(wfdir, outdir)
    datadir = DataDir(wfdir, outdir)

    files = sorted(glob.glob(os.path.join(datadir, 'qlm_3det*.x.asc')))
    if not(len(files)>0):
	raise NameError("Files not found. Surface Determinant plots cannot be produced")
    
    for f in files: 
	qlm_det_plot(datadir, figdir,  f)

def qlm_det_plot(datadir, figdir, f):
   
    print f 
    bh_det = os.path.join(datadir, f)
    bh_idx = int((f.split('[')[1]).split(']')[0])

    if os.path.exists(f):
        bh_det_dir = os.path.join(figdir, 'QLM_Det_BH%d'%bh_idx)
	if not(os.path.exists(bh_det_dir)):
	    os.makedirs(bh_det_dir)
        t, x, det = np.loadtxt( bh_det, unpack=True, usecols=(8,9,12))
        t_uniq = np.unique(t)
        t_uniq = t_uniq[t_uniq<tlast]
        idx = [np.amin(np.where(t==ti)) for ti in t_uniq]
	
        for n, j in enumerate(idx):
            plt.figure(figsize=(15,6))
            xvar, detvar = x[idx[j]:idx[j+1]-1], det[idx[j]:idx[j+1]-1]
            
            idx_order = np.argsort(xvar)
            xarr = xvar[idx_order]
            detarr = detvar[idx_order]
            plt.plot(xarr, detarr,  c='#1f77b4')                                                                
            plt.xlabel('x')                                                                              
            plt.ylabel(r'$det(q_{ij})$')

            #This part of code is not functioning properly, needs to be fixed, This should add a
            #rectangular boundary at the top aroung the timer
            
            starty, endy = plt.gca().get_ylim()
            startx, endx = plt.gca().get_xlim()                                                          
            xmid, deltax = (startx/2. + endx/2.), (-startx + endx)/10.                                     
            deltay = (endy-starty)/10.               
            #rect = patches.Rectangle((xmid-deltax,endy-10*deltay),deltax, 100*deltay, linewidth=1,edgecolor='r',facecolor='none') 
            
            plt.text( xmid,endy+deltay,'t=%.2g'%t[j], horizontalalignment='center', fontsize=16)       
            plt.savefig(os.path.join(bh1_det_dir, 'BH%d_detq_%d.png'%(bh_idx, n)), dpi=500)
            plt.close()
            
        

   
