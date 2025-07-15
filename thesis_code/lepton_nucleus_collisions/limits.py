#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental limits for lepton-nucleus collision experiments.

This module provides functions for calculating experimental limits on LFV couplings
fro the EIC, MuSIC, and MuBeD experiments.
"""

import numpy as np

from phys.utils import poisson_confidence_interval
from phys.scalar_ALP_EFT import ALP_to_scalar, scalar_fermion_branching_fraction
from phys.constants import cm2_fb, mm, mt
from lepton_nucleus_collisions.experiments import EIC, MuSIC, MuBeD, FinalState
from lepton_nucleus_collisions.compute.transformations import TRANSFORM

def default_kernel(experiment, final_state, eta_min = -np.inf, eta_max = np.inf, gamma_min = 1.0, gamma_max = np.inf, particle = 'boson'):
    """
    Create default acceptance kernel for experimental cuts.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration
    final_state : FinalState
        Final state configuration
    eta_min : float, optional
        Minimum pseudorapidity (default: -inf)
    eta_max : float, optional
        Maximum pseudorapidity (default: inf)
    gamma_min : float, optional
        Minimum boost factor (default: 1.0)
    gamma_max : float, optional
        Maximum boost factor (default: inf)
    particle : str, optional
        Particle type for transformation (default: 'boson')
        
    Returns
    -------
    function
        Kernel function that applies experimental cuts
    """
    
    def kernel(experiment, final_state, x, y, x_var = 'log_gamma', y_var = 'eta'):
        
        in_context = {'experiment': experiment,
                      'final_state': final_state,
                      'frame': 'ion',
                      'particle': 'boson'}
        out_context = {'experiment': experiment,
                      'final_state': final_state,
                      'frame': 'lab',
                      'particle': particle}

        gamma, eta = TRANSFORM(x, y, x_in = x_var, y_in = y_var, x_out = 'gamma', in_context = in_context, out_context = out_context, include_jacobian = False)
        
        return (eta > eta_min)*(eta < eta_max)*(gamma > gamma_min)*(gamma < gamma_max)
    
    return kernel
    

def EIC_limit(masses,
              electron_loss_rate = 1e-2,
              electron_positron_misID_rate = 1e-3,
              tau_efficiency = 1e-2,
              idx = (0, 2), #constraining g_{e \tau}
              g = [[1]*3]*3, #hierarchy
              th = [[0]*3]*3,
              ALP = True,
              Lam = 1000,
              L = 100,
              eta_min = -3.5,
              eta_max = 3.5,
              CL = 0.9,
              method = 'exact',
              interpolate_mass = True):
    """
    Calculate EIC limits on LFV couplings.
    
    Parameters
    ----------
    masses : array-like
        Scalar masses in GeV
    electron_loss_rate : float, optional
        Electron loss rate (default: 1e-2)
    electron_positron_misID_rate : float, optional
        Electron-positron misidentification rate (default: 1e-3)
    tau_efficiency : float, optional
        Tau detection efficiency (default: 1e-2)
    idx : tuple, optional
        Coupling indices to constrain (default: (0, 2) for g_{e tau})
    g : array-like, optional
        Coupling hierarchy matrix (default: [[1]*3]*3)
    th : array-like, optional
        PV angle matrix (default: [[0]*3]*3)
    ALP : bool, optional
        Whether constraining ALP or scalar(default: True)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    L : float, optional
        Integrated luminosity in fb^-1 (default: 100)
    eta_min : float, optional
        Minimum pseudorapidity (default: -3.5)
    eta_max : float, optional
        Maximum pseudorapidity (default: 3.5)
    CL : float, optional
        Confidence level (default: 0.9)
    method : str, optional
        Calculation method (default: 'exact')
    interpolate_mass : bool, optional
        Whether to interpolate mass (default: True)
        
    Returns
    -------
    array-like
        Upper limits on the specified coupling
    """
    
    if ALP:
        #don't concern ourselves with the angles...
        d = np.zeros((3, 3))
        ph = np.zeros((3, 3))
        G, th, _, _ = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    else:
        G = g

    A = EIC.A
    
    L_nuc = L/A
    final_states = [FinalState(method, 0.01, 'tau', 'scalar', mass, PV_angle = th[0][2]) for mass in masses]
    crossx_signal = np.array([EIC.cross_section(final_state,
                                                kernel = default_kernel(EIC, final_state, eta_min = eta_min, eta_max = eta_max),
                                                units = 'fb',
                                                g = G[0][2],
                                                interpolate_mass = interpolate_mass) for final_state in final_states])

    
    r_te = 0.1782
    background_efficiency = electron_positron_misID_rate * (2 - r_te)
    background_efficiency+= electron_loss_rate * r_te
    background_efficiency*= tau_efficiency
    

    crossx_BG = 2.6e4*1000 # from ... 
    N_BG = background_efficiency * crossx_BG * L_nuc #
    
    B_te = scalar_fermion_branching_fraction(masses, 0, 2, g = G, th = th)
    B_tt = scalar_fermion_branching_fraction(masses, 2, 2, g = G, th = th)

    signal_efficiency = 2*tau_efficiency*(1-r_te)*(B_te + r_te * B_tt)

    N_events_normalized = signal_efficiency * crossx_signal * L_nuc / g[idx[0]][idx[1]]**2
    
    n_max = poisson_confidence_interval(CL, N_BG, N_BG)[1]

    limit = np.sqrt(n_max/N_events_normalized)
    return np.where(np.isfinite(limit), limit, 1e16)

def MuSIC_limit(masses,
                muon_loss_rate = 1e-3,
                muon_antimuon_misID_rate = 1e-3,
                tau_efficiency = 1e-1,
                idx = (1, 2), #constraining g_{\mu \tau}
                g = [[1]*3]*3, #hierarchy
                th = [[0]*3]*3,
                ALP = True,
                Lam = 1000,
                L = 100,
                eta_min = -6.0,
                eta_max = 6.0,
                CL = 0.9, 
                method = 'exact',
                interpolate_mass = True):
    """
    Calculate MuSIC limits on LFV couplings.
    
    Parameters
    ----------
    masses : array-like
        Scalar masses in GeV
    muon_loss_rate : float, optional
        Muon loss rate (default: 1e-3)
    muon_antimuon_misID_rate : float, optional
        Muon-antimuon misidentification rate (default: 1e-3)
    tau_efficiency : float, optional
        Tau detection efficiency (default: 1e-1)
    idx : tuple, optional
        Coupling indices to constrain (default: (1, 2) for g_{mu tau})
    g : array-like, optional
        Coupling hierarchy matrix (default: [[1]*3]*3)
    th : array-like, optional
        PV angle matrix (default: [[0]*3]*3)
    ALP : bool, optional
        Whether constraining ALP or scalar (default: True)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    L : float, optional
        Integrated luminosity in fb^-1 (default: 100)
    eta_min : float, optional
        Minimum pseudorapidity (default: -6.0)
    eta_max : float, optional
        Maximum pseudorapidity (default: 6.0)
    CL : float, optional
        Confidence level (default: 0.9)
    method : str, optional
        Calculation method (default: 'exact')
    interpolate_mass : bool, optional
        Whether to interpolate mass (default: True)
        
    Returns
    -------
    array-like
        Upper limits on the specified coupling
    """
    
    if ALP:
        #don't concern ourselves with the angles...
        d = np.zeros((3, 3))
        ph = np.zeros((3, 3))
        G, th, _, _ = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    else:
        G = g

    A = MuSIC.A
    L_nuc = L/A
    final_states = [FinalState(method, 0.01, 'tau', 'scalar', mass, PV_angle = th[1][2]) for mass in masses]
    crossx_signal = np.array([MuSIC.cross_section(final_state,
                                                  kernel = default_kernel(MuSIC, final_state, eta_min = eta_min, eta_max = eta_max),
                                                  units = 'fb',
                                                  g = G[1][2],
                                                  interpolate_mass =interpolate_mass) for final_state in final_states])
        
    r_tm = 0.1739
    background_efficiency = muon_antimuon_misID_rate * (2 - r_tm)
    background_efficiency+= muon_loss_rate * r_tm
    background_efficiency *= tau_efficiency
    
    crossx_BG = 1.1e5*1000 #from ... 
    N_BG = background_efficiency * crossx_BG * L_nuc #
    
    B_tm = scalar_fermion_branching_fraction(masses, 1, 2, g = G, th = th)
    B_tt = scalar_fermion_branching_fraction(masses, 2, 2, g = G, th = th)
    
    signal_efficiency = 2*tau_efficiency*(1-r_tm)*(B_tm + r_tm * B_tt)
    
    N_events_normalized = L_nuc * signal_efficiency * crossx_signal / g[idx[0]][idx[1]]**2
    
    n_max = poisson_confidence_interval(CL, N_BG, N_BG)[1]
    
    limit = np.sqrt(n_max/N_events_normalized)
    return np.where(np.isfinite(limit), limit, 1e16)

def MuBeD_limit(masses,
                target_length = 2, #cm
                tau_efficiency = 0.15, #3-prong decays only
                track_resolution = 2, #cm
                N_MOT = 1e16,
                idx = (1, 2), #which coupling is being constrained
                g = [[1]*3]*3,
                th = [[0]*3]*3,
                ALP = True,
                Lam = 1000,
                CL = 0.9,
                method = 'exact',
                interpolate_mass = True):
    """
    Calculate MuBeD limits on LFV couplings.
    
    Parameters
    ----------
    masses : array-like
        Scalar masses in GeV
    target_length : float, optional
        Target length in cm (default: 2)
    tau_efficiency : float, optional
        Tau detection efficiency for 3-prong decays (default: 0.15)
    track_resolution : float, optional
        Track resolution in cm (default: 2)
    N_MOT : float, optional
        Number of muons on target (default: 1e16)
    idx : tuple, optional
        Coupling indices to constrain (default: (1, 2))
    g : array-like, optional
        Coupling hierarchy matrix (default: [[1]*3]*3)
    th : array-like, optional
        PV angle matrix (default: [[0]*3]*3)
    ALP : bool, optional
        Whether constraining ALP or scalar (default: True)
    Lam : float, optional
        ALP EFT scale in GeV (default: 1000)
    CL : float, optional
        Confidence level (default: 0.9)
    method : str, optional
        Calculation method (default: 'exact')
    interpolate_mass : bool, optional
        Whether to interpolate mass (default: True)
        
    Returns
    -------
    array-like
        Upper limits on the specified coupling
        
    Notes
    -----
    MuBeD uses a lead target with emulsion detectors for tau detection.
    Requires both taus to be detected with specific kinematic cuts.
    """
    
    if ALP:
        #don't concern ourselves with the angles...
        d = np.zeros((3, 3))
        ph = np.zeros((3, 3))
        G, th, _, _ = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    else:
        G = g
        
    M = MuBeD.M
    final_states = [FinalState(method, 1.0, 'tau', 'scalar', mass, PV_angle = th[1][2]) for mass in masses]
    crossx_tot = np.array([MuBeD.cross_section(final_state,
                                               units = 'fb',
                                               g = G[1][2],
                                               interpolate_mass = interpolate_mass) for final_state in final_states])
        
    
    # For MuBeD, we assume there is a "target_length" cm thick chunk of led,
    # interlaced with emulsion detectors with the ability to detect taus with 
    # 2mm resolution.
    #
    # So the tau must propagate at least 2mm, and its decay products must be
    # captured in the spectrometer. 
    # 
    # There are two tau's, but we only require that we capture one, along with 
    # a *positive* muon. We additionally veto on identification of a mu^-. Otherwise,
    # too large of a background from Bethe-Heitler tau^+ tau^- pair production.
    

    lead_density = 6.4e24 #GeV/cm^3
    
    #Luminosity (assuming the particle decays promptly)
    L = N_MOT * lead_density*  target_length/M / cm2_fb #fb^-1
    # [GeV/cm^3] [cm]/[GeV]

    c = 3e10 #cm/s
    tau_lifetime = 2.9e-13 #s
    g_min = track_resolution/(c * tau_lifetime)

    #We require identification of both taus. First, the "converted" tau:    
    crossx_convert = np.array([MuBeD.cross_section(final_state,
                                                   kernel = default_kernel(MuBeD, final_state, gamma_min = g_min, particle = 'lepton'),
                                                   units = 'fb',
                                                   g = G[1][2],
                                                   interpolate_mass = interpolate_mass) for final_state in final_states])

     
    p_convert = crossx_convert / crossx_tot * tau_efficiency
    
    
    # Next, the tau which is a decay product from the ALP. For this, we assume 
    # that \gamma_\tau = \gamma_\phi * E_\tau, where E_\tau is the energy imparted
    # to the \tau. For this, only need to worry about ALPs with mass greater than
    # the tau mass.
    Et = (masses**2 + mt**2)/(2*masses)
    
    p_decay_tm = []
    p_decay_tt = []
    #for m,final_state, crossx in zip(masses, final_states, crossx_tot):
        #\phi -> \mu \tau
    Et = (masses**2 + mt**2 - mm**2)/(2*masses)
    g_min_phi = g_min * mt/Et

    
    crossx_decay_tm = np.array([MuBeD.cross_section(final_state,
                                                   kernel = default_kernel(MuBeD, final_state, gamma_min = g_min),
                                                   units = 'fb',
                                                   g = G[1][2],
                                                   interpolate_mass = interpolate_mass) for g_min, final_state in zip(g_min_phi, final_states)])

    p_decay_tm = crossx_decay_tm/crossx_tot
    
    #\phi -> \tau \tau
    Et = masses**2/(2*masses)
    g_min_phi = g_min * mt/Et 

    crossx_decay_tt = np.array([MuBeD.cross_section(final_state,
                                                   kernel = default_kernel(MuBeD, final_state, gamma_min = g_min),
                                                   units = 'fb',
                                                   g = G[1][2],
                                                   interpolate_mass = interpolate_mass) for g_min, final_state in zip(g_min_phi, final_states)])

    p_decay_tt = crossx_decay_tt/crossx_tot
    
    p_decay_tm = np.array(p_decay_tm)
    p_decay_tt = np.array(p_decay_tt)

    B_tm = scalar_fermion_branching_fraction(masses, 1, 2, g = G, th = th)
    B_tt = scalar_fermion_branching_fraction(masses, 2, 2, g = G, th = th)
    r_tm = 0.1739
    
    p_decay = (p_decay_tm * B_tm  + p_decay_tt * B_tt * r_tm) * tau_efficiency
            
    N_events_normalized = L * crossx_tot * p_decay * p_convert / g[idx[0]][idx[1]]**2

    n_max = poisson_confidence_interval(CL, 0, 0)[1]
    
    limit = np.sqrt(n_max/N_events_normalized)
    return np.where(np.isfinite(limit), limit, 1e16)
    