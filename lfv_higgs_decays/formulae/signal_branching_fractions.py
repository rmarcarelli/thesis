#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import measure


from phys.formulae.ALP_EFT import ALP_decay_rate, ALP_fermion_decay_rate, Higgs_ALP_decay_rate
from phys.constants import hc_mGeV, mH, H_width_SM, crossx_H_prod, ml
from phys.utils import poisson_confidence_interval
from lfv_higgs_decays.fits.displaced_fits import ATLAS_fit, MATH_fit

#-------------------------------------------------------------------------#
# General
#-------------------------------------------------------------------------#

#ma is in GeV, ta is in meters (so it is really decay length, not lifetime)
def ta_to_Cij(ta, ma, idx = (0, 0), Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    #if hasattr(ma, '__iter__'):
    #    return np.array([ta_to_Cij(ta,m,idx,Cff,Cgg,Lam) for m in ma])
    
    ma = np.array(ma).reshape(-1, 1)
    ta = np.array(ta).reshape(1, -1)
    
    rate = ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)/Cll[idx[0]][idx[1]]**2 #normalize by the coefficient we are constraining
    return np.sqrt(hc_mGeV/(ta * rate)).squeeze()

def Cij_to_ta(Cij, ma, idx = (0, 0), Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    ma = np.array(ma).reshape(-1, 1)
    Cij = np.array(Cij).reshape(1, -1)
    
    rate = ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)/Cll[idx[0]][idx[1]]**2     
    
    return (hc_mGeV/(Cij**2 * rate)).squeeze()

#specifically, converts to C_{ah} (or equivalently \bar{C}_ah) by default
def BR_H_X_to_Cah(BR, BR_aa_X, ma, idx = 0, Cah = (1, 0), Lam = 1000):
    #if hasattr(ma, '__iter__'):
    #    return np.array([BR_H_X_to_Cah(br, br_X, m, idx, Cah, Lam) for br, br_X, m in zip(BR, BR_aa_X, ma)])
    
    if BR.shape == ma.shape:
        BR = BR.reshape(-1, 1)
    ma = ma.reshape(-1, 1)
    BR_aa_X = np.array(BR_aa_X).reshape(-1, 1)

    rate = Higgs_ALP_decay_rate(ma, Cah[0], Cah[1], Lam = Lam)/Cah[idx]**2 #normalize by the coefficient we are constraining

    return np.where(BR_aa_X > BR, np.sqrt(BR/(BR_aa_X - BR) * H_width_SM/rate), 1e16).squeeze()

#move elsewhere?
def find_contours(x, y, Z, Z_val = 1):
    contours = measure.find_contours(Z, Z_val)

    boundary_pts = []
    for cont in contours:
        i, j = cont.T[0], cont.T[1]  # Extract row/col indices
        x_coords = np.interp(i, np.arange(len(x)), x)  # Map cols to x
        y_coords = np.interp(j, np.arange(len(y)), y)  # Map rows to y
        boundary_pts.append(np.column_stack((x_coords, y_coords)).T)

    return boundary_pts

#-------------------------------------------------------------------------#
# For prompt analysis
#-------------------------------------------------------------------------#

def BR_aa_OSSF0(ma, Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):

    ALP_fermion_rates = ALP_fermion_decay_rate(ma.reshape(-1, 1, 1),
                                               ml.reshape(1, 1, 3),
                                               ml.reshape(1, 3, 1),
                                               np.array(Cll).reshape(-1, 3, 3),
                                               Lam = Lam)
    
    r_e = [1, 0, 0.1782] #rate of each lepton to be identified as an electron
    r_m = [0, 1, 0.1739] #rate of each lepton to be identified as a muon (assuming muon is detected before decaying to e)
    
    #e^+ mu^- mask
    ep_mm_mask = np.array([[r_e[i] * r_m[j] for i in range(3)] for j in range(3)]).reshape(1, 3, 3)

    #e^- mu^+ mask
    em_mp_mask = np.array([[r_e[j] * r_m[i] for i in range(3)] for j in range(3)]).reshape(1, 3, 3)
     
    ALP_width = ALP_decay_rate(ma, mf = ml, Cff = Cll, Cgg = Cgg, Lam = Lam)
    
    BR_ep_mm = np.sum(ep_mm_mask*ALP_fermion_rates, axis = (1, 2))/ALP_width # either both ALPs decay to e^+ mu^-
    BR_em_mp = np.sum(em_mp_mask*ALP_fermion_rates, axis = (1, 2))/ALP_width # ... or both ALPs decay to e^- mu^+

    BR_OSSF0 = BR_ep_mm**2 + BR_em_mp**2 

    return BR_OSSF0

# Simple linear interpolation for signal efficiency
def H_to_OSSF0_signal_efficiency(ma):
    mass_MG, eff_MG = np.loadtxt('lfv_higgs_decays/data/H_to_OSSF0_MG.txt',
                                 skiprows = 1,
                                 usecols = [0, 4]).T
    
    return np.interp(ma, masses_MG, eff_MG)
    
def f_detect(ma, ta, L_det):
    ma = np.array(ma).reshape(-1, 1, 1)
    ta = np.array(ta)
    if len(ta.shape) == 1:
        ta = np.array(ta).reshape(1, -1, 1)
    elif len(ta.shape) > 1:
        ta = np.array(ta).reshape(ta.shape + (1,))
    gamma = mH/(2*ma)
    bg = np.sqrt(gamma**2 - 1)
    La = bg*ta #decay length

    theta = np.linspace(0, np.pi/2, 1000).reshape(1, 1, -1)
    integrand = np.sin(theta) * (1 - np.exp(-L_det/(La*np.sin(theta))))**2
    f = np.trapz(integrand, x = theta)

    return f


#store these in a file ... 
masses = np.array([3.56, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62.5])
unnorm_crossx = np.array([6.207e-8,3.398e-8,3.171e-8,3.02e-8,2.805e-8,
                        2.552e-8,2.259e-8,1.934e-8,1.593e-8,1.248e-8,9.104e-9,5.83e-9,2.662e-9,2.814e-11])
crossx = unnorm_crossx*57/17.63 #MadGraph had an incorrect total crossx of 17.63 for ggH fusion, so normalize properly
N_events = np.array([941, 556, 383, 351, 344, 352, 304, 364, 313, 420, 482, 548, 643, 698])
eff = N_events/50000

masses_small = np.array([1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4])
eff_small = np.array([0.0823,0.0076,0.01868,0.0225,0.02466,0.02388,0.02299,0.0213,0.02016])

masses_MG = np.append(masses_small, masses)
eff_MG = np.append(eff_small, eff)

# Computes the 95% CL upper bound on the coupling Cah (or Cah' depending on the 
# value of 'which', assuming a hierarchy set by Cah. By default, 'Cah' is set to
# None, and 'which' deterimines whether it is (1, 0) (no Cah' coupling) or (0, 1)
# (no Cah coupling). However, more general hierarchies are allowed. For example, 
# setting Cah = (1, 1) implies that Cah and Cah' are equal in magnitude. Then,
# whether you are finding the bound on Cah or Cah' depends on whether you set 
# which to 'Cah' or 'Cahp'.

def CMS_Cah_limits(ma, ta, Cah = None, which = 'Cah',  Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 137):  
    
    # By default, computes limits on C_{ah} (or equivalent \bar{C}_{ah} defined in the text)
    if not Cah:
        Cah = (which == 'Cah', which == 'Cahp')
    
    # Scale number of (expected) observed events by luminosity
    n_obs = 7*L/137 
    
    # Compute 95% CL upper bound on mean number of signal events, assuming
    # the number of observed events matches the expected number from SM.
    n_max = poisson_confidence_interval(0.95, n_obs, n_obs)[1] 
    
    # Compute efficiency from MadGraph data, combined with the fraction of
    # ALPs which decay inside the detector
    L_det = 1.1 # in meters, rough size of CMS detector
    eff = H_to_OSSF0_signal_efficiency(ma).reshape(-1, 1) * f_detect(ma, ta, L_det)
    
    # Compute 95% CL upper limit on branching fraction
    BR_max = n_max/(eff*crossx_H_prod*L)
    
    # Convert from branching fraction to Higgs coupling
    if which == 'Cah':
        Cah_CMS = BR_H_X_to_Cah(BR_max, BR_aa_OSSF0(ma, Cll = Cll), ma, idx = 0, Cah = Cah, Lam = Lam)
    if which == 'Cahp':
        Cah_CMS = BR_H_X_to_Cah(BR_max, BR_aa_OSSF0(ma, Cll = Cll), ma, idx = 1, Cah = Cah, Lam = Lam)
        
    return Cah_CMS.squeeze()
    

#-------------------------------------------------------------------------#
# For ATLAS analysis
#-------------------------------------------------------------------------#

def rate_a_jets(ma, Cll = [[1]*3]*3, Lam = 1000):
    r = [0, 0, 0.6479] #rate into jets is zero except for tau
    rate = 0
    for i, Ci in enumerate(Cll):
        for j, Cij in enumerate(Ci):
            jet_rate = r[i] + r[j] - r[i]*r[j] #probability either lepton decays to jets
            rate += jet_rate * ALP_fermion_decay_rate(ma, ml[i], ml[j], Cij = Cij, Lam = 1000)
    return rate

def BR_aa_jets(ma, Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    BR_a_jets = rate_a_jets(ma, Cll, Lam)/ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)
    return BR_a_jets**2

def ATLAS_Cah_limits(ma, ta, Cah = None, which = 'Cah',  Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 36):  
    
    if not Cah:
        Cah = (which == 'Cah', which == 'Cahp')    
    
    BR_max = np.sqrt(36/L)*ATLAS_fit(ma, ta) #fit at L = 36, so scale accordingly
    BR_jets = BR_aa_jets(ma, Cll = Cll, Cgg = Cgg, Lam = Lam)
    
    if which == 'Cah':
        Cah_ATLAS = BR_H_X_to_Cah(BR_max, BR_jets, ma, idx = 0, Cah = Cah, Lam = Lam)
    if which == 'Cahp':
        Cah_ATLAS = BR_H_X_to_Cah(BR_max, BR_jets, ma, idx = 1, Cah = Cah, Lam = Lam)
        
    return Cah_ATLAS.squeeze()

#-------------------------------------------------------------------------#
# For MATHUSLA analysis
#-------------------------------------------------------------------------#


def MATH_Cah_limits(ma, ta, Cah= None, which = 'Cah', Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 3000):  
    
    if not Cah:
        Cah = (which == 'Cah', which == 'Cahp')        

    BR_max = MATH_fit(ma, ta)
    
    if which == 'Cah':
        Cah_MATH = BR_H_X_to_Cah(BR_max, 1, ma, idx = 0, Cah = Cah, Lam = Lam)
    if which == 'Cahp':
        Cah_MATH = BR_H_X_to_Cah(BR_max, 1, ma, idx = 1, Cah = Cah, Lam = Lam)
        
    return Cah_MATH.squeeze()
