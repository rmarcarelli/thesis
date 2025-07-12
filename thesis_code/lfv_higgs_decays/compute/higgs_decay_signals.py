#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from pathlib import Path


from phys.scalar_ALP_EFT import ALP_decay_rate, ALP_fermion_decay_rate, Higgs_ALP_decay_rate
from phys.constants import hc_mGeV, mH, H_width_SM, ml


PATH = str(Path(__file__).resolve().parents[1])

#-------------------------------------------------------------------------#
# General
#-------------------------------------------------------------------------#

def ta_to_Cij(ta, ma, idx = (0, 0), Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    """
    Convert ALP decay length to coupling strength.
    
    This function converts the ALP decay length (in meters) to the
    corresponding coupling strength Cij, given the ALP mass and other couplings.
    
    Parameters
    ----------
    ta : float or array-like
        ALP decay length in meters
    ma : float or array-like
        ALP mass in GeV
    idx : tuple, optional
        Indices (i, j) of the coupling to extract (default (0, 0))
    Cll : array-like, optional
        ALP-lepton coupling matrix (normalized to unit coupling at idx)
    Cgg : float, optional
        ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float or array-like
        Coupling strength Cij in GeV⁻¹
    """

    rate = ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)/Cll[idx[0]][idx[1]]**2
   
    return np.sqrt(hc_mGeV/(ta * rate))

def Cij_to_ta(Cij, ma, idx = (0, 0), Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    """
    Convert coupling strength to ALP decay length.
    
    This function converts the ALP coupling strength Cij to the
    corresponding decay length (in meters), given the ALP mass and other couplings.
    
    Parameters
    ----------
    Cij : float or array-like
        Coupling strength in GeV⁻¹
    ma : float or array-like
        ALP mass in GeV
    idx : tuple, optional
        Indices (i, j) of the coupling to extract (default (0, 0))
    Cll : array-like, optional
        ALP-lepton coupling matrix (normalized to unit coupling at idx)
    Cgg : float, optional
        ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float or array-like
        ALP decay length in meters
    """
    
    rate = ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)/Cll[idx[0]][idx[1]]**2
    
    return (hc_mGeV/(Cij**2 * rate))

#specifically, converts to C_{ah} (or equivalently \bar{C}_ah) by default
def BR_H_X_to_Cah(BR, BR_aa_X, ma, idx = 0, Cah = (1, 0), Lam = 1000):
    """
    Convert branching ratio to Higgs-ALP coupling.
    
    This function converts a branching ratio measurement to the corresponding
    Higgs-ALP coupling strength Cah (or Cah'), given the ALP mass and other parameters.
    
    Parameters
    ----------
    BR : float or array-like
        Observed branching ratio
    BR_aa_X : float or array-like
        Background branching ratio
    ma : float or array-like
        ALP mass in GeV
    idx : int, optional
        Index of the coupling to extract (default 0 for Cah)
    Cah : tuple, optional
        Higgs-ALP coupling parameters (Cah, Cahp)
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float or array-like
        Higgs-ALP coupling strength Cah in GeV⁻¹
    """

    rate = Higgs_ALP_decay_rate(ma, Cah[0], Cah[1], Lam = Lam)/Cah[idx]**2 

    return np.where(BR_aa_X > BR, np.sqrt(BR/(BR_aa_X - BR) * H_width_SM/rate), 1e16)

#-------------------------------------------------------------------------#
# For prompt analysis
#-------------------------------------------------------------------------#

def BR_aa_OSSF0(ma, Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    """
    Calculate branching ratio for H → aa → OSSF0 (opposite-sign same-flavor).
    
    This function computes the branching ratio for Higgs decay to ALPs
    followed by ALP decay to opposite-sign same-flavor lepton pairs,
    accounting for lepton identification efficiencies.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    Cll : array-like, optional
        ALP-lepton coupling matrix
    Cgg : float, optional
        ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float or array-like
        Branching ratio for H → aa → OSSF0
    """

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

def H_to_OSSF0_signal_efficiency(ma):
    """
    Calculate signal efficiency for H → aa → OSSF0 using MadGraph data.
    
    This function interpolates the signal efficiency from MadGraph simulation
    data for Higgs decay to ALPs followed by ALP decay to opposite-sign
    same-flavor lepton pairs.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    
    Returns
    -------
    float or array-like
        Signal efficiency
    """
    
    mass_MG, eff_MG = np.loadtxt(PATH +'/data/H_to_OSSF0_MG.txt',
                                 skiprows = 1,
                                 usecols = [0, 4]).T
    
    return np.interp(ma, mass_MG, eff_MG)
    
def f_detect(ma, ta, L_det):
    """
    Calculate the probability that an ALP decays within the detector volume.
    
    This function computes the probability that an ALP decays within the detector volume,
    accounting for the boost factor and decay length distribution.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    ta : float or array-like
        ALP decay length in meters
    L_det : float
        Detector length in meters
    
    Returns
    -------
    float or array-like
        Probability that an ALP decays within the detector volume
    """
    
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
    """
    Calculate ALP decay rate to jets.
    
    This function computes the rate for ALPs to decay to jets through hadronic tau decays.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    Cll : array-like, optional
        ALP-lepton coupling matrix
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float or array-like
        ALP decay rate to jets
    """
    r = [0, 0, 0.6479] #rate into jets is zero except for tau
    rate = 0
    for i, Ci in enumerate(Cll):
        for j, Cij in enumerate(Ci):
            jet_rate = r[i] + r[j] - r[i]*r[j] #probability either lepton decays to jets
            rate += jet_rate * ALP_fermion_decay_rate(ma, ml[i], ml[j], Cij = Cij, Lam = 1000)
    return rate

def BR_aa_jets(ma, Cll = [[1]*3]*3, Cgg = 0, Lam = 1000):
    """
    Calculate branching ratio for H → aa → jets.
    
    This function computes the branching ratio for Higgs decay to ALPs followed by ALP decay 
    to jets through hadronic tau decays.
    
    Parameters
    ----------
    ma : float or array-like
        ALP mass in GeV
    Cll : array-like, optional
        ALP-lepton coupling matrix
    Cgg : float, optional
        ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float or array-like
        Branching ratio for H → aa → jets
    """
    BR_a_jets = rate_a_jets(ma, Cll, Lam)/ALP_decay_rate(ma, Cff = Cll, Cgg = Cgg, Lam = Lam)
    return BR_a_jets**2

