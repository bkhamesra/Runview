# Scipt Name - eccentricity.py
# Author - Bhavesh Khamesra
# Use: This script computes the initial eccentricity for BBH. This can be extended to compute the eccentricity evolution for any compact binary system. 

#References - "Post-Newtonian Quasicircular Initial Orbits for Numerical Relativity", HEALY et. al (arXiv:1702.00872),
#         "Reducing eccentricity in black-hole binary evolutions with initial parameters", Husa et al, (PRD 77, 044037 (2008)

#Notes: The method follows Healy et. al. approach. The function used for fitting decaying part of orbital separation differs from Sascha et al., our function fits better. Healy et al do not completely specify the polynomial used for fitting. The function used for fitting sinusoidal part differs from Healy et. al. approach, here we use polynomial fit. 
#.....................................................................................................................................................................



import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
from numpy.polynomial.polynomial import polyval
from scipy.optimize import curve_fit


def sinefit(time, data ):
    ''' Sine fitting function of form amp*sin(omega*t + phi) + x3 '''   
    
    mean_guess = np.mean(data)
    std_guess = 3.*np.std(data)/(2**0.5)
    phase_guess = 0.
    omega_guess = 2*np.pi/100.
    
    #Determine the coefficients     
    datafit_guess1 = std_guess*np.sin(omega_guess*time+phase_guess)+mean_guess
    optimize_func = lambda x:x[0]*np.sin(x[1]*time+x[2])+x[3]-data
    std_est, omega_est, phase_est, mean_est = leastsq(optimize_func, [std_guess, omega_guess, phase_guess, mean_guess])[0]
    
    datafit = std_est*np.sin(omega_est*time+phase_est)+mean_est 
    return [datafit, datafit_guess1]

def polyfit(time, data, degree ):
    ''' Polynomial fitting function of degree n ( x0 + x1*t +x2*t^2 ... + xn*t^n)  '''  
    
    root_time = time    #time**0.5  
    polyfit = np.poly1d(np.polyfit(root_time, data,degree)) 
    return polyfit

def polyfunc(time,a0,a1,a2,a3,b1,b2,b3):
    ''' Polynomial fitting function 2 of form ( x0 + x1*t + x2*t^2 + x3*t^3 + x4/t + x5/t^2 + x6/t^3  '''   
    poly = a0 + a1*time**1 + a2*time**2. + a3*time**3.  + np.divide(b1,time) + np.divide(b2, time**2) + np.divide(b3,time**3)   #works better
    return poly

def polyfunc_sascha(time, a0,a1,a2,a3,a4):
    ''' Polynomial fitting function from Sascha's paper'''
    poly = a0 + a1*time + a2*time**2. + a3*time**3. + a4*time**4. #As used in Sascha'a paper (PHYSICAL REVIEW D 77, 044037 (2008))
    return poly

def sinefunc(time, amp1, omega1, phi01, amp2, omega2, phi02):
    ''' Sine fitting function with two frequency of form amp1*sin(omega1*t + phi1) +  amp2*sin(omega2*t + phi2) ''' 
    return amp1*np.sin(omega1*time + phi01) + amp2*np.sin(omega2*time + phi02)


def fitfunc_coeff(fitting_function,time, data):
    ''' Determine coefficients of fitting function'''
    fittingfunc_coeff, cov = curve_fit(fitting_function, time, data)
    return fittingfunc_coeff
    
def func_max(time, data):
    ''' Find maxima of function'''
    der = np.divide((data[1:] - data[:-1]), (time[1:] - time[:-1]))
    i=1
    while i<len(der):
    
        if der[i]==0 and der[i-1]>0:
            maxima = data[i]
            tmax = time[i]
            break
        elif der[i]<0 and der[i-1]>0:
            maxima = data[i]/2.+data[i-1]/2.
            tmax = time[i]/2. + time[i-1]/2.
            break
        i=i+1
    
    try:
        return [maxima, tmax]
    except NameError:
        print("*(metadata) >>  Maxima: no maxima found \n")
            
def func_min(time, data):
    ''' Find minima of function'''
    der = np.divide((data[1:] - data[:-1]), (time[1:] - time[:-1]))
    i=0
    for i in range(len(der)):
        if der[i]==0 and der[i-1]<0:
            minima = data[i]
            tmin = time[i]
            break
        elif der[i]>0 and der[i-1]<0:
            minima = data[i]/2.+data[i-1]/2.
            tmin = time[i]/2 + time[i-1]/2
            break
     
    try:
        return [minima, tmin]
    except NameError:
        print("*(metadata) >> Minima: no minima found \n")

def mean_anomaly(t, time_arr, r1, r2, deltar_arr):
    ''' Calculate mean anomaly'''
    
    index_t = np.amin(np.where(time_arr>=t))    
    
    k=0
    T_prev = -1.
    T_next = -1.

    for k in range(len(time_arr)):  
        if k==0 and r1[1][0]==0 : 
            T_prev = 0.
            l = k
            break
        elif r1[1][k-1]<0. and r1[1][k]>=0:
             T_prev = time_arr[k]
             l = k
             break
    if T_prev==-1:  
        print("*(metadata) >> Mean Anomaly: Time of previous periapsis not found. Time period will be used \n")
    else:
        print("*(metadata) >> Mean Anomaly: Time of previous periapsis = {} \n".format(T_prev))
    
    
    while l <len(time_arr):
        if l==0: l = l+20   
        elif np.sign(r1[1][l-5]) == -1.*np.sign(r1[1][l]) and np.sign(r1[1][l])==np.sign(r1[1][k+5]):
            T_next = time_arr[l]
            break
        l = l+1
             
    if T_next==-1:  
        print("*(metadata) >> Mean Anomaly: Time of next periapsis not found \n")
    else:   
        print("*(metadata) >> Mean Anomaly: Time of next periapsis = {} \n".format(T_next))

    time_period = T_next-T_prev
    if T_prev >-1 and T_prev<t:
        mean_anomaly = 2.*np.pi*(t-T_prev)/(T_next-T_prev)
    elif T_prev > t:
        mean_anomaly = 2.*np.pi*t/time_period
    return [mean_anomaly, T_prev, T_next]


def import_data(dirpath):
    '''Import data from ShiftTracker Files'''   
#Check if the necessary files exists: parfile, shifttracker, ihspin, hnmass
    filename = dirpath.split("/")[-1]
    parfile = (dirpath.split('/')[-1]) + ('.par')
    shifttracker0 = dirpath+'/ShiftTracker0.asc'
    shifttracker1 = dirpath+'/ShiftTracker1.asc'
    
#Import the data
    data_ST1 = np.loadtxt(shifttracker0, usecols=(1, 2, 3, 4), comments = '#')
    data_ST2 = np.loadtxt(shifttracker1, usecols=(1, 2, 3, 4), comments = '#')
    
    n = min(len(data_ST1), len(data_ST2))
    
    data_ST1 = data_ST1[0:n]
    data_ST2 = data_ST2[0:n]
    r1 = np.empty((3, n))
    r2 = np.empty((3, n))
    (time2, r1[0][:],r1[1][:],r1[2][:]) = data_ST1[:,0], data_ST1[:,1], data_ST1[:,2], data_ST1[:,3]
    (time1, r2[0],r2[1],r2[2]) = data_ST2[:,0], data_ST2[:,1], data_ST2[:,2], data_ST2[:,3]
    return [time1, time2, r1,r2]



def ecc_and_anomaly(dirpath,  jkrad_time):  
    '''Compute eccentricity and mean anomaly'''
    [time1, time2, r1,r2] = import_data(dirpath)

    orbsep_vec = r2-r1
    orbsep_mag = np.sqrt(orbsep_vec[0]**2 + orbsep_vec[1]**2 + orbsep_vec[2]**2)

    if time1.any()!=time2.any():
        time = (time1+time2)/2.
        #print("*(metadata) >> Warning: Time values are different in 2 shift tracker files, Average time will be used for eccentricities \n")
    else:
        time = time1    

    
    plt.plot(time, orbsep_mag, color = 'r',label= 'Orbital Separation')
    plt.xlabel("Time")
    plt.ylabel("Orbital Separation")
    plt.legend()
    plt.savefig(dirpath + "/Orbital_Separation.png")
    plt.close()

    x1 = r1[0]
    y1 = r1[1]  
    x2 = r2[0]
    y2 = r2[1]

    #Compute Mean Anomaly:
    [mean_anom, tprev, tnext] = mean_anomaly(jkrad_time, time,r1,r2, orbsep_mag)
    if tprev ==0:
        tprev_idx = 0
    else:
        tprev_idx = np.where(time==tprev)

    tnext_idx = np.argwhere(time==tnext)
    
    if (type(tprev_idx) is  np.ndarray) and np.size(tprev_idx)==1:
        tprev_idx = (tprev_idx[0])
    
    if (type(tnext_idx) is  np.ndarray) and np.size(tnext_idx)==1:
        tnext_idx = np.asscalar(tnext_idx)
    #print("*(metadata) >> Mean Anomaly = {} \n ".format( mean_anom))

    #print time[tprev_idx], time[tnext_idx]
    plt.plot(x1[:tnext_idx], y1[:tnext_idx], color='r',label="BH1")
    plt.plot(x2[:tnext_idx], y2[:tnext_idx], color='k',label="BH2")
    plt.legend()
    plt.savefig(dirpath + '/Orbit1.png')
    plt.close()
    
    # Define the cutoff index and fitting time interval (400M should be good - BBH should have 1-2 orbits)
    # Ideal way would be to compute time taken when two orbits are completed 

    cutoff_idx = np.amin(np.amin(np.where(time>=jkrad_time)))

    timeidx_400 = np.amin(np.where(time>=400))


    time_fit = time[0:timeidx_400]
    datafit = orbsep_mag[0:timeidx_400] 
    D = orbsep_mag[0:timeidx_400]


    #Fitting the decaying part of orbital separation with polynomial in t**0.5

    fittingfunc_coeff = curve_fit(polyfunc, time_fit**0.5, datafit)[0]
    Dfit_coeff = fittingfunc_coeff  #fitfunc_coeff(time_fit, datafit, polyfunc)
    [a0,a1,a2,a3,b1,b2,b3] = Dfit_coeff
    Dfit_data = polyfunc(time_fit**0.5, a0,a1,a2,a3,b1,b2,b3) 
    

    #Check the order of coefficients - should be smaller for higher powers of time
    #print("\n*(metadata) >> Eccentricity -  Polynomial used for fitting orbital sepation = {} + {}*t + {}*t^2 + {}*t^3 + {}/t + {}/t**2 + {}/t**3 \n".format(a0,a1,a2,a3,b1,b2,b3))
    
    plt.plot(time_fit, D, color = 'r', label='Orbital Separation')
    plt.plot(time_fit, Dfit_data, 'k--',  label='Polynomial Fitting')
    plt.xlim(0,500)
    plt.xlabel("Time")
    plt.ylabel("Orbital Separation")
    plt.savefig(dirpath + '/Orbitalseparation_fitting.png')
    plt.legend()
    plt.close() 


    #Computing the eccentricity - Following Jim's paper (arXiV 1702.00872): Fitting is done using polynomial and not sinusoidal. 
    
    eccentricity = np.divide((D - Dfit_data),Dfit_data)

    eccfit_poly = polyfit(time_fit, eccentricity, 14)

    #Analyzing the fit:
    data = eccentricity - eccfit_poly(time_fit)
    mean = np.mean(data)
    var  = np.var(data)
    #print("*(metadata) >> Eccentricity:Comparing the Fit -  Mean = {} (Ideal = 0) and variance = {} (Ideal = 0)\n".format(mean, var))  
    
    plt.plot(time_fit, eccentricity, 'r', label='Eccentricity')
    plt.plot(time_fit, eccfit_poly(time_fit), 'k--', label='Polynomial Fit') 
    plt.xlabel("Time")
    plt.ylabel("Eccentricity")
    plt.legend()
    plt.savefig(dirpath + '/Eccentricity_fitting.png')
    plt.close()
    
    eccentricity_cutoff = eccentricity[cutoff_idx:]
    time_cutoff = time_fit[cutoff_idx:]
        
    plt.plot(time_cutoff, eccentricity_cutoff, 'r', label='Eccentricity')
    plt.plot(time_cutoff, eccfit_poly(time_cutoff), 'k--', label='Polynomial Fit')
    plt.xlabel("Time")
    plt.ylabel("Eccentricity")
    plt.legend()
    plt.savefig(dirpath + "/Eccfit_after_junk.png")
    plt.close()
    
    maxima, tmax  = func_max(time_cutoff, eccfit_poly(time_cutoff))
    minima, tmin = func_min(time_cutoff, eccfit_poly(time_cutoff))
    
    print("*(metadata) >> Eccentricity: max = {} found at time = {} and min = {} found at time = {} \n".format(maxima, tmax, minima, tmin))
    
    ecc_init = (abs(maxima) + abs(minima))/2.
    print(" *(metadata) >> Initial Eccentricity = {} \n".format(ecc_init)) 

    return [mean_anom, ecc_init]


