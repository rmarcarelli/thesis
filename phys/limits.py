#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
ALL limits can be computed in this file, assuming the data was properly generated...

These limits are valid for any leptonically-coupled particle with 
hierarchy described in matrix form, i.e.

            | gee   gem   get |
      gll = | gme   gmm   gmt |
            | gte   gtm   gtt |

The only thing that matters here is the ratio of the couplings. Then, one .

If index is unspecified... just picks the largest coupling.

Alternatively, one could cast limits on a coupling gij with all other couplings
fixed or zero (i.e., one can consider constraint on "get" when gll = 1e-3 for all
diagonal couplings, and all other off-diagonal couplings are zero). The hierarchy
approach is slightly easier to implement, because the branching-fractions depends
only on the hierarchy so is independent of the over-all size of the couplings.This
still allows one to set certain couplings to zero, and impose hierarchies between
e.g. diagonal and off-diagonal components. 
"""

import numpy as np
from skimage import measure



#General function useful for a few purposes
def find_contours(x, y, Z, Z_val = 1):
    contours = measure.find_contours(Z, Z_val)

    boundary_pts = []
    for cont in contours:
        i, j = cont.T[0], cont.T[1]  # Extract row/col indices
        x_coords = np.interp(i, np.arange(len(x)), x)  # Map cols to x
        y_coords = np.interp(j, np.arange(len(y)), y)  # Map rows to y
        boundary_pts.append(np.column_stack((x_coords, y_coords)).T)

    return boundary_pts

"""
----- Limits from LFV Lepton Decays ---

"""
from dipole_form_factors.formulae.decay_rates_new import radiative_decay_rate, trilepton_decay_rate


lepton_widths = np.array([np.inf, 2.99e-19, 2.27e-12])
radiative_processes = [(1, 0), (2, 0), (2, 1)]
#90% limits
radiative_decay_branching_limits = radiative_decay_BR_limits = {(1, 0): 4.2e-13, #from MEG ()
                                                                (2, 0): 3.3e-8,  #from BaBar ()
                                                                (2, 1): 4.4e-8,  #from BaBar ()
                                                                }

def radiative_decay_limit(m, process, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    i, j, k, l = idx
    if g is None:
        g = np.zeros((3, 3))
        g[i][j] = 1
        g[j][i] = 1
        g[k][l] = 1
        g[l][k] = 1
    normalized_rate = radiative_decay_rate(m, *process, g, th, d, ph, mode, ALP, Lam)/(g[i][j]*g[k][l])**2 + 1e-64
    rate_limit = lepton_widths[process[0]] * radiative_decay_branching_limits[process]

    return (rate_limit/normalized_rate)**(1/4)


trilepton_processes = [(1, 0, 0, 0),
                       (2, 0, 0, 0),
                       (2, 0, 0, 1),
                       (2, 0, 1, 0),
                       (2, 1, 1, 0),
                       (2, 1, 1, 0),
                       (2, 0, 1, 1),
                       (2, 1, 1, 1)]

#90% limits                -  -  -  +
trilepton_decay_limits = {(1, 0, 0, 0): 1.0e-12, #\mu \rightarrow 3e
                          (2, 0, 0, 0): 2.7e-8, #\tau \rightarrow 3e
                          (2, 0, 0, 1): 1.8e-8, #\tau \rightarrow e e \bar{\mu}
                          (2, 0, 1, 0): 1.8e-8, #\tau \rightarrow e \mu \bar{e}
                          (2, 1, 1, 0): 1.7e-8, #\tau \rightarrow \mu \mu \bar{e}
                          (2, 0, 1, 1): 1.7e-8, #\tau \rightarrow e \mu \bar{\mu}
                          (2, 1, 1, 1): 2.1e-8 #\tau \rightarrow 3\mu
                         }

def trilepton_decay_limit(m, process, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    i, j, k, l = idx
    if g is None:
        g = np.zeros((3, 3))
        g[i][j] = 1
        g[j][i] = 1
        g[k][l] = 1
        g[l][k] = 1
        
    normalized_rate = trilepton_decay_rate(m, *process, g, th, d, ph, mode, ALP, Lam)/(g[i][j]*g[k][l])**2 + 1e-64
    rate_limit = lepton_widths[process[0]] * trilepton_decay_limits[process]
    
    return (rate_limit/normalized_rate)**(1/4)


"""
----- Limits from lepton dipole moments ---

"""
from dipole_form_factors.formulae.dipole_moments_new import magnetic_dipole_moment_contribution, electric_dipole_moment_contribution

#EDM AND MDM LIMITS
anomalies = {'e Rb': (34e-14, 16e-14),
             'e Cs': (-101e-14, 27e-14),
             'mu': (249e-11, 48e-11)}


MDM_exp_error = [13e-14, #e (average of Rb and Cs)
                 40e-11, #mu
                 3.2e-3  #tau
                 ]

#90% limits
EDM_limits = [4.1e-30,
              1.8e-19,
              1.85e-17]


def magnetic_dipole_moment_limit(m, i, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):

    _i, _j = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        
    da = magnetic_dipole_moment_contribution(m, i, g, th, d, ph, mode, ALP, Lam)
    
    return np.sqrt(2*MDM_exp_error[i]/da)

def electric_dipole_moment_limit(m, i, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = 'max CPV', ALP = False, Lam = 1000):
    
    _i, _j = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        
    EDM = electric_dipole_moment_contribution(m, i, g, th, d, ph, mode, ALP, Lam)
    
    return np.sqrt(EDM_limits/EDM)


#EXPLANATIONS TO G-2
def g_2_explanation(m, which_anomaly, idx, nsig = 2, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    assert which_anomaly in ['e Rb', 'e Cs', 'mu']
    anomaly, sig = anomalies[which_anomaly]
    
    if which_anomaly[0] == 'e':
        i = 0
    else:
        i = 1
        
    da = magnetic_dipole_moment_contribution(m, i, g, th, d, ph, mode, ALP, Lam)
    
    return sorted([np.sqrt((anomaly - nsig * sig)/ da), np.sqrt((anomaly + nsig * sig)/ da)])
        
    
        
#LFV AT LEPTON NUCLEUS COLLIDERS
def EIC_limit(mass, g = None):
    
    #crossx = 
    
    pass

def MuSIC_limit(mass, g = None):
    pass

def MuBeD_limit(mass, g = None):
    pass

"""
----- Limits from Higgs decays at CERN (ALP only) ---


"""
from phys.formulae.ALP_EFT import ALP_decay_rate, ALP_fermion_decay_rate, Higgs_ALP_decay_rate
from phys.constants import hc_mGeV, mH, H_width_SM, crossx_H_prod, ml
from phys.utils import poisson_confidence_interval
from lfv_higgs_decays.fits.signal_branching_fractions import (BR_H_X_to_Cah,
                                                              H_to_OSSF0_signal_efficiency,
                                                              f_detect, 
                                                              BR_aa_OSSF0,
                                                              BR_aa_jets)
from lfv_higgs_decays.fits.displaced_fits import ATLAS_fit, MATH_fit


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

def MATH_Cah_limits(ma, ta, Cah= None, which = 'Cah', Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 3000):  
    
    if not Cah:
        Cah = (which == 'Cah', which == 'Cahp')        

    BR_max = MATH_fit(ma, ta)
    
    if which == 'Cah':
        Cah_MATH = BR_H_X_to_Cah(BR_max, 1, ma, idx = 0, Cah = Cah, Lam = Lam)
    if which == 'Cahp':
        Cah_MATH = BR_H_X_to_Cah(BR_max, 1, ma, idx = 1, Cah = Cah, Lam = Lam)
        
    return Cah_MATH.squeeze()


#DARK GAUGE BOSONS
def existing_limits():
    pass


