#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from skimage import measure

from phys.formulae.ALP_EFT import ALP_decay_rate, ALP_fermion_decay_rate, Higgs_ALP_decay_rate
from phys.constants import hc_mGeV, mH, H_width_SM, ml

#-------------------------------------------------------------------------#
# General
#-------------------------------------------------------------------------#

#ma is in GeV, ta is in meters (so it is really decay length, not lifetime)
def ta_to_Cij(ta, ma, idx = (0, 0), Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):

    rate = ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)/Cll[idx[0]][idx[1]]**2
   
    return np.sqrt(hc_mGeV/(ta * rate))

def Cij_to_ta(Cij, ma, idx = (0, 0), Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    
    rate = ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)/Cll[idx[0]][idx[1]]**2
    
    return (hc_mGeV/(Cij**2 * rate))

#specifically, converts to C_{ah} (or equivalently \bar{C}_ah) by default
def BR_H_X_to_Cah(BR, BR_aa_X, ma, idx = 0, Cah = (1, 0), Lam = 1000):

    rate = Higgs_ALP_decay_rate(ma, Cah[0], Cah[1], Lam = Lam)/Cah[idx]**2 

    return np.where(BR_aa_X > BR, np.sqrt(BR/(BR_aa_X - BR) * H_width_SM/rate), 1e16)

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

    ma = np.array(ma)
    dims = [1 for s in ma.shape]
    
    ALP_fermion_rates = ALP_fermion_decay_rate(ma.reshape(*ma.shape, 1, 1),
                                               ml.reshape(*dims, 1, 3),
                                               ml.reshape(*dims, 3, 1),
                                               np.array(Cll).reshape(*dims, 3, 3),
                                               Lam = Lam)
    
    r_e = [1, 0, 0.1782] #rate of each lepton to be identified as an electron
    r_m = [0, 1, 0.1739] #rate of each lepton to be identified as a muon (assuming muon is detected before decaying to e)
    
    #e^+ mu^- mask
    ep_mm_mask = np.array([[r_e[i] * r_m[j] for i in range(3)] for j in range(3)]).reshape(*dims, 3, 3)

    #e^- mu^+ mask
    em_mp_mask = np.array([[r_e[j] * r_m[i] for i in range(3)] for j in range(3)]).reshape(*dims, 3, 3)
     
    ALP_width = ALP_decay_rate(ma, mf = ml, Cff = Cll, Cgg = Cgg, Lam = Lam)

    BR_ep_mm = np.sum(ep_mm_mask*ALP_fermion_rates, axis = (-2, -1))/ALP_width # either both ALPs decay to e^+ mu^-
    BR_em_mp = np.sum(em_mp_mask*ALP_fermion_rates, axis = (-2, -1))/ALP_width # ... or both ALPs decay to e^- mu^+

    BR_OSSF0 = BR_ep_mm**2 + BR_em_mp**2 

    return BR_OSSF0

# Simple linear interpolation for signal efficiency
def H_to_OSSF0_signal_efficiency(ma):
    
    mass_MG, eff_MG = np.loadtxt('lfv_higgs_decays/data/H_to_OSSF0_MG.txt',
                                 skiprows = 1,
                                 usecols = [0, 4]).T
    
    return np.interp(ma, mass_MG, eff_MG)
    
def f_detect(ma, ta, L_det):
    
    ma = np.array(ma)
    ta = np.array(ta)
    
    ma = np.array(ma).reshape(*ma.shape, 1)
    ta = np.array(ta).reshape(*ta.shape, 1)
    
    gamma = mH/(2*ma)
    bg = np.sqrt(gamma**2 - 1)
    La = bg*ta #decay length

    dims = [1 for s in ma.shape]
    theta = np.linspace(0, np.pi/2, 1000).reshape(*dims, -1)
    
    integrand = np.sin(theta) * (1 - np.exp(-L_det/(La*np.sin(theta))))**2
    f = np.trapz(integrand, x = theta)
    
    return f.squeeze()
    

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

