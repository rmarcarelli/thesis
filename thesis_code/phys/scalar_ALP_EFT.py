#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import alpha, ml, mH, vH

#-------------------------------------------------------------------------#
# Scalar Decays
#-------------------------------------------------------------------------#

def scalar_decay_rate(m, mf = ml, g = [[1]*3]*3, th = [[0]*3]*3):
    """
    Calculate the total scalar decay rate to fermions.
    
    Parameters
    ----------
    m : float
        Scalar mass
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    g : array-like, optional
        Scalar-fermion coupling strengths
    th : array-like, optional
       PV angles for scalar-fermion couplings
    
    Returns
    -------
    float
        Total scalar decay rate
    """
    
    rate = 0
    for i, gi in enumerate(g):
        for j, gij in enumerate(gi):
            rate += scalar_fermion_decay_rate(m, mf[i], mf[j], gij, th[i][j])

    return rate

def scalar_fermion_decay_rate(m, mi, mj, gij = 1, th_ij = 0):
    """
    Calculate the scalar decay rate to a specific fermion pair.
    
    Parameters
    ----------
    m : float
        Scalar mass
    mi : float
        First fermion mass
    mj : float
        Second fermion mass
    gij : float, optional
        Scalar-fermion coupling strength
    th_ij : float, optional
        Phase parameter for scalar-fermion coupling
    
    Returns
    -------
    float
        Decay rate to the specified fermion pair
    """

    mij2 = mi**2 + mj**2 + 2*mi*mj*np.cos(th_ij)
    M2 = 4 * gij**2 * (m**2  - mij2)
    

    phase_space = np.sqrt((m > (mi + mj))*(m**2 - (mi - mj)**2)*(m**2 - (mi + mj)**2))/(32*np.pi*m**3)
    
    rate = phase_space * M2
    
    return rate

def scalar_fermion_branching_fraction(m, i, j, mf = ml, g = [[1]*3]*3, th = [[0]*3]*3):
    """
    Calculate the branching fraction for scalar decay to a specific fermion pair.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first fermion
    j : int
        Index of second fermion
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    g : array-like, optional
        Scalar-fermion coupling strengths
    th : array-like, optional
        PV angles for scalar-fermion couplings
    
    Returns
    -------
    float
        Branching fraction to the specified fermion pair
    """

    return scalar_fermion_decay_rate(m, mf[i], mf[j], gij = g[i][j], th_ij = th[i][j])/scalar_decay_rate(m, mf = mf, g = g, th = th)


#-------------------------------------------------------------------------#
# ALP Decays
#-------------------------------------------------------------------------#

def ALP_decay_rate(ma, mf = ml, Cff = [[1]*3]*3, TH = [[0]*3]*3, Cgg = 0, Lam = 1000):
    """
    Calculate the total ALP decay rate including fermion and photon channels.
    
    Parameters
    ----------
    ma : float
        ALP mass
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    Cff : array-like, optional
        ALP-fermion coupling strengths
    TH : array-like, optional
        PV angles for ALP-fermion couplings
    Cgg : float, optional
        ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float
        Total ALP decay rate
    """
    
    rate = 0
    for i, Ci in enumerate(Cff):
        for j, Cij in enumerate(Ci):
            rate += ALP_fermion_decay_rate(ma, mf[i], mf[j], Cij, TH[i][j], Lam)
            
    rate += ALP_photon_decay_rate(ma, mf = ml, Cff = Cff, TH = TH, Cgg = Cgg, Lam = Lam)
    
    return rate

def B1(t):
    """
    Calculate the B1 function used in ALP-photon decay rate.
    
    Parameters
    ----------
    t : float or array
        Parameter t = 4*mf^2/ma^2
    
    Returns
    -------
    float or array
        Value of B1 function
    """
    
    #x>=1
    t_gt_1 = np.arcsin(1/np.sqrt(t))
    #x<1
    t_lt_1 = np.pi/2 + 1j/2*np.log((1+np.sqrt(1-t))/(1-np.sqrt(1-t)))
    
    ft = np.where(t >= 1, t_gt_1, t_lt_1)

    return 1 - t*ft**2

def ALP_photon_decay_rate(ma, mf = ml, Cff =  [[1]*3]*3, TH = [[0]*3]*3, Cgg = 0, Lam = 1000):
    """
    Calculate the ALP decay rate to photons including fermion loop contributions.
    
    Parameters
    ----------
    ma : float
        ALP mass
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    Cff : array-like, optional
        ALP-fermion coupling strengths
    TH : array-like, optional
        PV angles for ALP-fermion couplings
    Cgg : float, optional
        Direct ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float
        ALP decay rate to photons
    """
    
    Cgg_eff = Cgg
    for i in range(len(mf)):
        Cgg_eff += Cff[i][i]*np.cos(TH[i][i])/(8*np.pi**2) * B1(4*mf[i]**2/ma**2)

    return 4*np.pi * alpha**2 * ma**3 /Lam**2 * np.abs(Cgg_eff)**2

def ALP_fermion_decay_rate(ma, mi, mj, Cij = 1, TH_ij = 0, Lam = 1000):
    """
    Calculate the ALP decay rate to a specific fermion pair.
    
    Parameters
    ----------
    ma : float
        ALP mass
    mi : float
        First fermion mass
    mj : float
        Second fermion mass
    Cij : float, optional
        ALP-fermion coupling strength
    TH_ij : float, optional
        Phase parameter for ALP-fermion coupling
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float
        ALP decay rate to the specified fermion pair
    """
        
    mij2 = mi**2 + mj**2 + 2*mi*mj*np.cos(TH_ij)
    M2 = 4 * Cij**2 * (mij2* ma**2  - (mi**2-mj**2)**2)/Lam**2
    phase_space = np.sqrt((ma > (mi + mj))*(ma**2 - (mi - mj)**2)*(ma**2 - (mi + mj)**2))/(32*np.pi*ma**3)
    rate = phase_space * M2
    
    return rate

def ALP_fermion_branching_fraction(ma, i, j, mf = ml, Cff = [[1]*3]*3, TH = [[0]*3]*3, Cgg = 0, Lam = 1000):
    """
    Calculate the branching fraction for ALP decay to a specific fermion pair.
    
    Parameters
    ----------
    ma : float
        ALP mass
    i : int
        Index of first fermion
    j : int
        Index of second fermion
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    Cff : array-like, optional
        ALP-fermion coupling strengths
    TH : array-like, optional
        PV angles for ALP-fermion couplings
    Cgg : float, optional
        ALP-photon coupling strength
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float
        Branching fraction to the specified fermion pair
    """
    
    return ALP_fermion_decay_rate(ma, mf[i], mf[j], Cff[i][j], TH[i][j], Lam)/ALP_decay_rate(ma, mf, Cff, TH, Lam)

#-------------------------------------------------------------------------#
# Higgs Decays to ALPs
#-------------------------------------------------------------------------#

def Higgs_ALP_decay_rate(ma, Cah, Cahp = 0, Lam = 1000):
    """
    Calculate the Higgs decay rate to ALPs.
    
    Parameters
    ----------
    ma : float
        ALP mass
    Cah : float
        Higgs-ALP coupling parameter
    Cahp : float, optional
        Additional Higgs-ALP coupling parameter
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    float
        Higgs decay rate to ALPs
    """
    
    Cah_bar = Cah - (2*ma**2)/(mH**2 - 2*ma**2) * Cahp
    
    return vH**2*mH**3*np.sqrt(1 - 4*ma**2/mH**2)*(1-2*ma**2/mH**2)**2*Cah_bar**2/(32*np.pi*Lam**4)
        

#-------------------------------------------------------------------------#
# Conversion from ALP couplings to scalar couplings
#-------------------------------------------------------------------------#

def ALP_to_scalar(C, TH, D, PH, Lam = 1000, mf = ml):
    """
    Convert ALP couplings to equivalent scalar couplings.
    
    Parameters
    ----------
    C : array-like
        ALP-fermion coupling strengths
    TH : array-like
        PV angles for ALP-fermion couplings
    D : array-like
        C or CP violating phases for scalar couplings
    PH : array-like
        C or CP violating phases for scalar couplings
    Lam : float, optional
        ALP EFT scale
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    
    Returns
    -------
    tuple
        (g, th, d, ph) where g, th, d, and ph are the corresponding scalar parameters.
    """
    g = np.zeros_like(C, dtype = np.float64)
    th = np.zeros_like(TH, dtype = np.float64)
    d = np.zeros_like(D, dtype = np.float64)
    ph = np.zeros_like(PH, dtype = np.float64)
    
    I = len(mf)
    J = len(mf)

    for i in range(I):
        for j in range(J):
            g[i][j] = C[i][j] * np.sqrt(mf[i]**2 + mf[j]**2 + 2*mf[i]*mf[j]*np.cos(2*TH[i][j]))/Lam
            th[i][j] = np.arctan((mf[i]+mf[j])/(mf[i]-mf[j]) / np.tan(TH[i][j]))
            d[i][j] = D[i][j] - np.pi/2
            ph[i][j] = PH[i][j] - np.pi/2
    return g, th, d, ph

#-------------------------------------------------------------------------#
# Conversion from scalar couplings to ALP couplings
#-------------------------------------------------------------------------#

def scalar_to_ALP(g, th, d, ph, Lam = 1000, mf = ml):
    """
    Convert scalar couplings to equivalent ALP couplings.
    
    Parameters
    ----------
    g : array-like
        Scalar-fermion coupling strengths
    th : array-like
        PV angles for scalar-fermion couplings
    d : array-like
        C or CP violating parameters for scalar couplings
    ph : array-like
        C or CP violating parameters for scalar couplings
    Lam : float, optional
        ALP EFT scale
    mf : array-like, optional
        Fermion masses, defaults to lepton masses
    
    Returns
    -------
    tuple
        (C, TH, D, PH) where C, TH, D, and PH are the corresponding ALP parameters.
    """
    C = np.zeros_like(g, dtype = np.float64)
    TH = np.zeros_like(th, dtype = np.float64)
    D = np.zeros_like(d, dtype = np.float64)
    PH = np.zeros_like(ph, dtype = np.float64)
    
    I = len(mf)
    J = len(mf)
    
    for i in range(I):
        for j in range(J):
            if th[i][j] == np.pi/2: #ensures C[i][j] not infinite for pure pseudo-scalar with i = j
                C[i][j] = g[i][j] * Lam/(mf[i] + mf[j])
            else: #otherwise, it *is* infinite if th[i][j] != pi/2 and i = j because no finite value of Cij works
                C[i][j] = g[i][j] * np.sqrt(mf[i]**2 + mf[j]**2 + 2*mf[i]*mf[j]*np.cos(2*th[i][j])) * Lam/np.abs(mf[i]**2 - mf[j]**2)                
            TH[i][j] = np.arctan((mf[i]+mf[j])/(mf[i]-mf[j]) / np.tan(th[i][j]))                
            D[i][j] = d[i][j] - np.pi/2
            PH[i][j] = ph[i][j] - np.pi/2
    return C, TH, D, PH