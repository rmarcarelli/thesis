#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

These limits are valid for any leptonically-coupled particle with 
hierarchy described in matrix form, i.e.

            | gee   gem   get |
      gll = | gme   gmm   gmt |
            | gte   gtm   gtt |

The only thing that matters here is the ratio of the couplings.

Alternatively, one could cast limits on a coupling gij with all other couplings
fixed or zero (i.e., one can consider constraint on "get" when gll = 1e-3 for all
diagonal couplings, and all other off-diagonal couplings are zero). The hierarchy
approach is slightly easier to implement, because the branching-fractions depends
only on the hierarchy so is independent of the over-all size of the couplings.This
still allows one to set certain couplings to zero, and impose hierarchies between
e.g. diagonal and off-diagonal components. 

"""

import numpy as np
from scipy.special import erfinv

from .compute import (radiative_decay_rate,
                      trilepton_decay_rate,
                      magnetic_dipole_moment_contribution,
                      electric_dipole_moment_contribution)
from phys.constants import ml

### LIMITS FROM LFV LEPTON DECAYS ###

lepton_widths = np.array([np.inf,
                          2.99e-19,
                          2.27e-12
                          ])

radiative_processes = [(1, 0), #\mu \rightarrow e\gamma
                       (2, 0), #\tau \rightarrow e\gamma
                       (2, 1)  #\tau \rightarrow \mu\gamma
                       ]
#90% limits
radiative_decay_branching_limits = {(1, 0): 4.2e-13, #from MEG 
                                    (2, 0): 3.3e-8,  #from BaBar
                                    (2, 1): 4.4e-8,  #from BaBar
                                    }    

def radiative_decay_limit(m, process, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000, confidence = 0.9):    
    
    _i, _j, _k, _l = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        g[_k][_l] = 1
        g[_l][_k] = 1
        
    normalized_rate = radiative_decay_rate(m, *process, g, th, d, ph, mode, ALP, Lam)/(g[_i][_j]*g[_k][_l])**2 + 1e-64
    rate_limit = lepton_widths[process[0]] * radiative_decay_branching_limits[process]
    
    if ALP:
        normalized_rate /= (1000/Lam)**4 # units of TeV^-1

    # By default, limit is 90% confidence. For a Poisson counting
    # experiment with zero observed events, the upper bound on the
    # mean is proportional to -log(1 - confidence). 
    # In practice, this will barely change anything. For example.
    # (log(1 - 0.95)/log(1 - 0.9))^(1/4) = 1.068
    
    z_score_conf = -np.log(1-confidence)
    z_score_90 = -np.log(0.1)
    factor = z_score_conf/z_score_90

    return (factor * rate_limit/normalized_rate)**(1/4)


trilepton_processes = [(1, 0, 0, 0), #\mu \rightarrow 3e
                       (2, 0, 0, 0), #\tau \rightarrow 3e
                       (2, 0, 0, 1), #\tau \rightarrow e e \bar{\mu}
                       (2, 0, 1, 0), #\tau \rightarrow e \mu \bar{e}
                       (2, 1, 1, 0), #\tau \rightarrow \mu \mu \bar{e} 
                       (2, 0, 1, 1), #\tau \rightarrow e \mu \bar{\mu}
                       (2, 1, 1, 1) #\tau \rightarrow 3\mu
                      ]

#90% limits
#                          -  -  -  +
trilepton_decay_limits = {(1, 0, 0, 0): 1.0e-12, # (SINDRUM, http://doi.org/10.1016/0550-3213(88)90462-2)
                          (2, 0, 0, 0): 2.7e-8, # (Belle, https://doi.org/10.1016/j.physletb.2010.03.037)
                          (2, 0, 0, 1): 1.5e-8, # (Belle, https://doi.org/10.1016/j.physletb.2010.03.037)
                          (2, 0, 1, 0): 1.8e-8, # (Belle, https://doi.org/10.1016/j.physletb.2010.03.037)
                          (2, 1, 1, 0): 1.7e-8, # (Belle, https://doi.org/10.1016/j.physletb.2010.03.037)
                          (2, 0, 1, 1): 2.7e-8, # (Belle, https://doi.org/10.1016/j.physletb.2010.03.037)
                          (2, 1, 1, 1): 2.1e-8  # (Belle, https://doi.org/10.1016/j.physletb.2010.03.037)
                         }

def trilepton_decay_limit(m, process, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000, confidence = 0.9):
    _i, _j, _k, _l = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        g[_k][_l] = 1
        g[_l][_k] = 1
        
    normalized_rate = trilepton_decay_rate(m, *process, g, th, d, ph, mode, ALP, Lam)/(g[_i][_j]*g[_k][_l])**2 + 1e-64
    rate_limit = lepton_widths[process[0]] * trilepton_decay_limits[process]
    
    if ALP:
        normalized_rate = np.where(m < ml[_i],
                                   normalized_rate/(1000/Lam)**2,
                                   normalized_rate/(1000/Lam)**4)


    # By default, limit is 90% confidence. For a Poisson counting
    # experiment with zero observed events, the upper bound on the
    # mean is proportional to -log(1 - confidence). 
    
    z_score_conf = -np.log(1-confidence)
    z_score_90 = -np.log(0.1)
    factor = z_score_conf/z_score_90

            
    return np.where(m < ml[_i],
                    (factor * rate_limit/normalized_rate)**(1/2),
                    (factor * rate_limit/normalized_rate)**(1/4)
                    )


###  LIMITS FROM LEPTON DIPOLE MOMENTS ###


anomalies = {'e Rb': (34e-14, 16e-14),
             'e Cs': (-101e-14, 27e-14),
             'mu': (249e-11, 48e-11)}

MDM_exp_error = [13e-14, #e (average of Rb and Cs)
                 40e-11, #mu
                 3.2e-3  #tau
                ]

#90% limits
EDM_limits = [4.1e-30, #e
              1.8e-19, #mu
              1.85e-17 #tau
             ]

def magnetic_dipole_moment_limit(m, i, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000, confidence = 0.9):
    _i, _j = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        
    da = magnetic_dipole_moment_contribution(m, i, g, th, d, ph, mode, ALP, Lam)/g[_i][_j]**2 + 1e-64
    
    if ALP:
        da /= (1000/Lam)**2 # units of TeV^-1


    # By default, limit is 68% confidence (1 sigma).
    # Can compute the z-score from the confidence
    # level via z = sqrt(2) * erfinv(confidence).
    
    z_score_conf = np.sqrt(2)*erfinv(confidence)
    z_score_default = 1 # 1 sigma by default
    factor = (z_score_conf/z_score_default)**(1/4)
    
    return np.sqrt(np.abs(factor * MDM_exp_error[i]/da))

def electric_dipole_moment_limit(m, i, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = 'max CPV', ALP = False, Lam = 1000, confidence = 0.9):
    _i, _j = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        
    EDM = electric_dipole_moment_contribution(m, i, g, th, d, ph, mode, ALP, Lam)
    
    if ALP:
        EDM /= (1000/Lam)**2 # units of TeV^-1

    # By default, limit is 90% confidence. For a Poisson counting
    # experiment with zero observed events, the upper bound on the
    # mean is proportional to -log(1 - confidence). 
    
    z_score_conf = -np.log(1-confidence)
    z_score_90 = -np.log(0.1)
    factor = z_score_conf/z_score_90
    
    return np.sqrt(np.abs(factor * EDM_limits[i]/EDM))


###  EXPLANATIONS TO LEPTON G-2 ANOMALIES ###
def g_2_explanation(m, which_anomaly, idx, g = None, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000, nsig = 2):
    assert which_anomaly in ['e Rb', 'e Cs', 'mu']
    anomaly, sig = anomalies[which_anomaly]

    if which_anomaly[0] == 'e':
        i = 0
    else:
        i = 1

    _i, _j = idx
    if g is None:
        g = np.zeros((3, 3))
        g[_i][_j] = 1
        g[_j][_i] = 1
        
    da = magnetic_dipole_moment_contribution(m, i, g, th, d, ph, mode, ALP, Lam)
    
    if ALP:
        da /= (1000/Lam)**2 # units of TeV^-1

    if anomaly > 0:
        return np.sqrt((anomaly - nsig * sig)/ da), np.sqrt((anomaly + nsig * sig)/ da)
    else:
        return np.sqrt((anomaly + nsig * sig)/ da), np.sqrt((anomaly - nsig * sig)/ da)
        
        

