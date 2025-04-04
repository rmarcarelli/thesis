#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import measure

#from phys.formulae.ALP_EFT import ALP_decay_rate, ALP_fermion_decay_rate, Higgs_ALP_decay_rate
from phys.constants import crossx_H_prod
from phys.utils import poisson_confidence_interval
from lfv_higgs_decays.fits.displaced_fits import ATLAS_fit, MATH_fit
from lfv_higgs_decays.formulae.higgs_decay_signals import (Cij_to_ta,
                                                           BR_H_X_to_Cah,
                                                           H_to_OSSF0_signal_efficiency,
                                                           BR_aa_OSSF0,
                                                           f_detect,
                                                           BR_aa_jets
                                                           )
#-------------------------------------------------------------------------#
# For CMS analysis
#-------------------------------------------------------------------------#

def Cah_limit_CMS(ma, ta, Cah = None, which = 'Cah',  Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 137):  
    
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
    eff = H_to_OSSF0_signal_efficiency(ma) * f_detect(ma, ta, L_det)
    
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


def Cah_limit_ATLAS(ma, ta, Cah = None, which = 'Cah',  Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 36):  
    
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

def Cah_limit_MATH(ma, ta, Cah= None, which = 'Cah', Cll = [[1]*3]*3, Cgg = 0, Lam = 1000, L = 3000):  
    
    assert which in ['Cah', 'Cahp']
    
    if not Cah:
        Cah = (which == 'Cah', which == 'Cahp')    

    BR_max = MATH_fit(ma, ta)
    
    if which == 'Cah':
        Cah_MATH = BR_H_X_to_Cah(BR_max, 1, ma, idx = 0, Cah = Cah, Lam = Lam)
    if which == 'Cahp':
        Cah_MATH = BR_H_X_to_Cah(BR_max, 1, ma, idx = 1, Cah = Cah, Lam = Lam)
        
    return Cah_MATH.squeeze()

#-------------------------------------------------------------------------#
# Limits on Cll
#-------------------------------------------------------------------------#

#Move elsewhere?
def find_contours(x, y, Z, Z_val = 1):
    contours = measure.find_contours(Z, Z_val)

    boundary_pts = []
    for cont in contours:
        i, j = cont.T[0], cont.T[1]  # Extract row/col indices
        x_coords = np.interp(i, np.arange(len(x)), x)  # Map cols to x
        y_coords = np.interp(j, np.arange(len(y)), y)  # Map rows to y
        boundary_pts.append(np.column_stack((x_coords, y_coords)).T)

    return boundary_pts

detectors = {'CMS': Cah_limit_CMS,
             'ATLAS': Cah_limit_ATLAS,
             'MATHUSLA': Cah_limit_MATH
             }

def Cll_limit(ma, Cah, ij = (0, 0), which = 'Cah',
              Cll_range = (1e-10, 1e2), npts = 400,
              detector = None, projection = False,
              Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):

    assert which in ['Cah', 'Cahp']

    if not detector:
        detector = detectors.keys()
        
    if type(detector) == str:
        detector = [detector]
        
    assert all(det in detectors.keys() for det in detector)
    
    Cij = np.geomspace(*Cll_range, npts)
    ta = Cij_to_ta(Cij.reshape(1, -1), ma.reshape(-1, 1), idx = ij, Cll = Cll, Cgg = Cgg, Lam = Lam)

    #absolute limits on Cah    
    det_kwargs = (None, which, Cll, Cgg, Lam)

    if projection:
        det_kwargs += (3000,) #projection luminosity
    
    Cah_limits = np.min([detectors[det](ma.reshape(-1, 1), ta, *det_kwargs) for det in detector], axis = 0)

    contours = find_contours(ma, Cij, Cah_limits, Cah)
    
    return contours

