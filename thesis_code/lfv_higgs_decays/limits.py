#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental limits for Higgs decay analyses.

This module provides functions for calculating experimental limits on
Higgs-ALP couplings from various experiments including CMS, ATLAS, and MATHUSLA.
The functions handle displaced vertex searches and branching fraction limits.
"""

import numpy as np

from phys.constants import crossx_H_prod
from phys.utils import poisson_confidence_interval, find_contours
from lfv_higgs_decays.fits.displaced_fits import ATLAS_fit, MATH_fit
from lfv_higgs_decays.compute.higgs_decay_signals import (Cij_to_ta,
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
    """
    Calculate CMS limits on Higgs-ALP couplings.
    
    This function computes 95% CL upper limits on Higgs-ALP couplings
    from CMS displaced vertex searches. It uses Poisson statistics and
    detector efficiency factors to convert observed event counts to
    coupling limits.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    ta : float or array-like
        ALP decay length in meters
    Cah : tuple, optional
        Higgs-ALP coupling parameters (default: inferred from 'which')
    which : str, optional
        Which coupling to limit: 'Cah' or 'Cahp' (default: 'Cah')
    Cll : array-like, optional
        ALP-lepton coupling matrix (default: unit couplings)
    Cgg : float, optional
        ALP-photon coupling strength (default: 0)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    L : float, optional
        Integrated luminosity in fb^-1 (default: 137)
    
    Returns
    -------
    float or array-like
        95% CL upper limit on Higgs-ALP coupling
    """
    
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
    """
    Calculate ATLAS limits on Higgs-ALP couplings.
    
    This function computes limits on Higgs-ALP couplings from ATLAS
    displaced vertex searches using fitted branching fraction limits.
    The limits are scaled according to integrated luminosity.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    ta : float or array-like
        ALP decay length in meters
    Cah : tuple, optional
        Higgs-ALP coupling parameters (default: inferred from 'which')
    which : str, optional
        Which coupling to limit: 'Cah' or 'Cahp' (default: 'Cah')
    Cll : array-like, optional
        ALP-lepton coupling matrix (default: unit couplings)
    Cgg : float, optional
        ALP-photon coupling strength (default: 0)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    L : float, optional
        Integrated luminosity in fb^-1 (default: 36)
    
    Returns
    -------
    float or array-like
        Upper limit on Higgs-ALP coupling
    """
    
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
    """
    Calculate MATHUSLA limits on Higgs-ALP couplings.
    
    This function computes limits on Higgs-ALP couplings from MATHUSLA
    displaced vertex searches using fitted branching fraction limits.
    MATHUSLA is a proposed surface detector for long-lived particles.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    ta : float or array-like
        ALP decay length in meters
    Cah : tuple, optional
        Higgs-ALP coupling parameters (default: inferred from 'which')
    which : str, optional
        Which coupling to limit: 'Cah' or 'Cahp' (default: 'Cah')
    Cll : array-like, optional
        ALP-lepton coupling matrix (default: unit couplings)
    Cgg : float, optional
        ALP-photon coupling strength (default: 0)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    L : float, optional
        Integrated luminosity in fb^-1 (default: 3000)
    
    Returns
    -------
    float or array-like
        Upper limit on Higgs-ALP coupling
    """
    
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

detectors = {'CMS': Cah_limit_CMS,
             'ATLAS': Cah_limit_ATLAS,
             'MATHUSLA': Cah_limit_MATH
             }

def Cll_limit(ma, Cah, ij = (0, 0), which = 'Cah',
              Cll_range = (1e-10, 1e2), npts = 400,
              detector = None, projection = False,
              Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    """
    Calculate limits on ALP-lepton couplings from Higgs decay searches.
    
    This function computes limits on ALP-lepton couplings Cll by finding
    contours where the Higgs-ALP coupling Cah matches experimental limits.
    It can use multiple detectors and handle projections to future luminosities.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    Cah : float or array-like
        Higgs-ALP coupling value to find contours for
    ij : tuple, optional
        Indices (i, j) of the Cll coupling to limit (default: (0, 0))
    which : str, optional
        Which Higgs coupling: 'Cah' or 'Cahp' (default: 'Cah')
    Cll_range : tuple, optional
        Range of Cll values to scan (default: (1e-10, 1e2))
    npts : int, optional
        Number of points for Cll scan (default: 400)
    detector : str or list, optional
        Detector(s) to use (default: all available)
    projection : bool, optional
        Whether to use projection luminosity (default: False)
    Cll : array-like, optional
        Background ALP-lepton coupling matrix (default: unit couplings)
    Cgg : float, optional
        ALP-photon coupling strength (default: 0)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    
    Returns
    -------
    array-like
        Cll values along the contour where Cah matches experimental limits
    """

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

