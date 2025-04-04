#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from phys.utils import poisson_confidence_interval
from phys.formulae.scalar_ALP_EFT import ALP_to_scalar, scalar_fermion_branching_fraction
from phys.constants import cm2_fb, ml, me, mm, mt
from lepton_nucleus_collisions.utils.process import process_run_card
from lepton_nucleus_collisions.process_crossx_data import crossx, dcrossx, distribution


def EIC_limit(masses,
              electron_loss_rate = 1e-3,
              electron_positron_misID_rate = 1e-3,
              tau_efficiency = 1e-2,
              idx = (0, 2), #constraining g_{e \tau}
              g = [[1]*3]*3, #hierarchy
              th = [[0]*3]*3,
              ALP = True,
              Lam = 1000,
              L = 100,
              CL = 0.9):
    
    
    #How about a --> e^+ \mu^- or a --> mu^+ e^-
    
    if ALP:
        #don't concern ourselves with the angles...
        d = np.zeros((3, 3))
        ph = np.zeros((3, 3))
        G, th, _, _ = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    else:
        G = g

    A, masses_from_file = process_run_card('EIC_Gold.txt', ['A', 'masses'])
    L_nuc = L/A
    crossx_from_file = crossx('EIC_Gold',
                              ('tau', 'scalar', 1.0, 'exact', False),
                              units = 'fb',
                              Y_min = -3.5,
                              Y_max = 3.5) * G[0][2]**2   
    
    crossx_signal = np.interp(masses, masses_from_file, crossx_from_file)
    
    r_te = 0.1782
    background_efficiency = electron_positron_misID_rate * (2 - r_te)
    background_efficiency+= electron_loss_rate * r_te
    background_efficiency*= tau_efficiency
    

    crossx_BG = 2.6e4*1000 #from ... 
    N_BG = background_efficiency * crossx_BG * L_nuc #
    
    B_te = scalar_fermion_branching_fraction(masses, 0, 2, g = G, th = th)
    B_tt = scalar_fermion_branching_fraction(masses, 2, 2, g = G, th = th)

    signal_efficiency = 2*tau_efficiency*(1-r_te)*(B_te + r_te * B_tt)
    
    N_events_normalized = signal_efficiency * crossx_signal * L_nuc / g[idx[0]][idx[1]]**2
    
    n_max = poisson_confidence_interval(CL, N_BG, N_BG)[1]
    
    return np.sqrt(n_max/N_events_normalized)

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
                CL = 0.9):
    
    if ALP:
        #don't concern ourselves with the angles...
        d = np.zeros((3, 3))
        ph = np.zeros((3, 3))
        G, th, _, _ = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    else:
        G = g

    A, masses_from_file = process_run_card('MuSIC.txt', ['A', 'masses'])
    L_nuc = L/A
    crossx_from_file = crossx('MuSIC',
                              ('tau', 'scalar', 1.0, 'exact', False),
                              units = 'fb',
                              Y_min = -6,
                              Y_max = 6) * G[1][2]**2   
    
    crossx_signal = np.interp(masses, masses_from_file, crossx_from_file)
    
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
    
    return np.sqrt(n_max/N_events_normalized)

def MuBeD_limit(masses,
                target_length = 2, #cm
                tau_efficiency = 0.15, #3-prong decays only
                track_resolution = 0.2, #cm
                N_MOT = 1e16,
                idx = (1, 2), #which coupling is being constrained
                g = [[1]*3]*3,
                th = [[0]*3]*3,
                ALP = True,
                Lam = 1000,
                CL = 0.9):
    
    if ALP:
        #don't concern ourselves with the angles...
        d = np.zeros((3, 3))
        ph = np.zeros((3, 3))
        G, th, _, _ = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    else:
        G = g
        
    M, masses_from_file = process_run_card('MuCol.txt', ['M', 'masses'])
    crossx_from_file = crossx('MuCol',
                              ('tau', 'scalar', 1.0, 'exact', False),
                              units = 'fb') * G[1][2]**2   
    
    crossx_tot = np.interp(masses, masses_from_file, crossx_from_file)
    
    
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
    L = N_MOT * lead_density/M / cm2_fb #fb^-1

    c = 3e10 #cm/s
    tau_lifetime = 2.9e-13 #s
    g_min = track_resolution/(c * tau_lifetime)
    
    #We require identification of both taus. First, the "converted" tau:
    crossx_from_file = crossx('MuCol',
                              ('tau', 'scalar', 1.0, 'exact', False),
                              units = 'fb',
                              X_min = g_min,
                              final_state_particle = 'lepton') * G[1][2]**2
    crossx_convert = np.interp(masses, masses_from_file, crossx_from_file)   
       
    p_convert = crossx_convert/crossx_tot * tau_efficiency
    
    # Next, the tau which is a decay product from the ALP. For this, we assume 
    # that \gamma_\tau = \gamma_\phi * E_\tau, where E_\tau is the energy imparted
    # to the \tau. For this, only need to worry about ALPs with mass greater than
    # the tau mass.
    Et = (masses**2 + mt**2)/(2*masses)
    
    p_decay_tm = []
    p_decay_tt = []
    for m in masses_from_file:
        #\phi -> \mu \tau
        Et = (m**2 + mt**2 - mm**2)/(2*m)
        gamma, dist = distribution('MuCol', ('tau', 'scalar', 1.0, 'exact', False, m), which = 'X')
        dist = np.nan_to_num(dist)
        g_min_phi = g_min * mt/Et
        p = np.trapz(dist * (gamma > g_min_phi), x = gamma) #* B_tm * tau_efficiency
        p_decay_tm.append(p)
        
        #\phi -> \tau \tau
        Et = m**2/(2*m)
        gamma, dist = distribution('MuCol', ('tau', 'scalar', 1.0, 'exact', False, m), which = 'X')
        g_min_phi = 2 * g_min * mt/Et #since each tau takes half the energy, multiply by 2
        B_tt = scalar_fermion_branching_fraction(m, 2, 2, g = G, th = th)
        #p+= np.trapz(dist*(gamma > g_min_phi), x = gamma) * B_tt * tau_efficiency * 0.1739 # tau^+ -> mu^+
        p = np.trapz(dist * (gamma > g_min_phi), x = gamma)
        p_decay_tt.append(p)
    
    p_decay_tm = np.interp(masses, masses_from_file, p_decay_tm)
    p_decay_tt = np.interp(masses, masses_from_file, p_decay_tt)
    
    B_tm = scalar_fermion_branching_fraction(masses, 1, 2, g = G, th = th)
    B_tt = scalar_fermion_branching_fraction(masses, 2, 2, g = G, th = th)
    r_tm = 0.1739
    
    p_decay = (p_decay_tm * B_tm  + p_decay_tt * B_tt * r_tm)* tau_efficiency
            
    N_events_normalized = L * crossx_tot * p_decay * p_convert / g[idx[0]][idx[1]]**2

    n_max = poisson_confidence_interval(CL, 0, 0)[1]
    

    return np.sqrt(n_max/N_events_normalized.squeeze()) #zero background, 90% confidence
    