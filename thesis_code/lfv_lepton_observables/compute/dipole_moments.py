#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


from .form_factor_functions_approx import gp, gm
from phys.constants import alpha, ml, hc_cmGeV
from phys.scalar_ALP_EFT import ALP_to_scalar

#-------------------------------------------------------------------------#
# Magnetic dipole moment for li given a set of LFV couplings in terms of
# their magnitudes g, angles th, and phases ph and d. Can explicitly turn 
# on PC or chiral coupling (without setting phases) by choosing mode = 'PC'
# or mode = 'chiral'. Can convert to LFV ALP by setting ALP = True, (in this
# case, the couplings refer to the analogous LFV ALP couplings). Rather than
# computing f_plus and f_minus each time, their values are passed through as
# arguments fp_ij and fm_ij. The mass m is also passed through, but this is
# only used for the ALP photon contribution. 
#-------------------------------------------------------------------------#

def magnetic_dipole_moment_contribution(m, i, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    """
    Calculate the magnetic dipole moment contribution for lepton i.
    
    This function computes the magnetic dipole moment (g-2) contribution
    for lepton i from new physics couplings. It can handle different
    coupling modes (PC, chiral) and ALP scenarios.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of lepton (0=e, 1=μ, 2=τ)
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    ph : array-like, optional
        C or CP violating phases for couplings
    mode : str, optional
        Coupling mode: 'PC' for parity-conserving, 'chiral' for chiral
    ALP : bool, optional
        Whether to include ALP photon contributions
    Lam : float, optional
        ALP EFT scale (used only if ALP=True)
    
    Returns
    -------
    float
        Magnetic dipole moment contribution Δa_i
    """

    if mode == 'PC':
        th = [[0]*3]*3
        d = [[0]*3]*3
        ph = [[0]*3]*3
        
    if mode == 'chiral':
        th = [[np.pi/4]*3]*3
        d = [[0]*3]*3 if ALP else [[np.pi/2]*3]*3
        ph = [[0]*3]*3
    
    return np.real(F2(m, i, g, th, d, ph, ALP, Lam))


#-------------------------------------------------------------------------#
# Electric dipole moment for li given a set of LFV couplings in terms of
# their magnitudes g, angles th, and phases ph and d. Can explicitly turn 
# on PC or chiral coupling (without setting phases) by choosing mode = 'PC'
# or mode = 'chiral'. Can convert to LFV ALP by setting ALP = True, (in this
# case, the couplings refer to the analogous LFV ALP couplings). Rather than
# computing f_plus and f_minus each time, their values are passed through as
# arguments fp_ij and fm_ij. The mass m is also passed through, but this is
# only used for the ALP photon contribution. 
#-------------------------------------------------------------------------#

def electric_dipole_moment_contribution(m, i, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    """
    Calculate the electric dipole moment contribution for lepton i.
    
    This function computes the electric dipole moment contribution
    for lepton i from new physics couplings. It can handle different
    coupling modes (PC, chiral, max CPV) and ALP scenarios.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of lepton (0=e, 1=μ, 2=τ)
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    ph : array-like, optional
        C or CP violating phases for couplings
    mode : str, optional
        Coupling mode: 'PC' for parity-conserving, 'chiral' for chiral, 'max CPV' for maximal CP violation
    ALP : bool, optional
        Whether to include ALP photon contributions
    Lam : float, optional
        ALP EFT scale (used only if ALP=True)
    
    Returns
    -------
    float
        Electric dipole moment contribution in e·cm
    """

    if mode == 'PC':
        th = [[0]*3]*3
        d = [[0]*3]*3
        ph = [[0]*3]*3
        
    if mode == 'chiral':
        th = [[np.pi/4]*3]*3
        d = [[0]*3]*3 if ALP else [[np.pi/2]*3]*3
        ph = [[0]*3]*3
        
    if mode == 'max CPV':
        th = [[np.pi/4]*3]*3
        d = [[np.pi/2]*3]*3 if ALP else [[0]*3]*3
        
    return np.real(F3(m, i, g, th, d, ph, ALP, Lam))* 1/(2*ml[i]) * hc_cmGeV
 
#-------------------------------------------------------------------------#
# li on-diagonal dipole form factors with internal lj
#-------------------------------------------------------------------------#

def F2(m, i, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, ALP = False, Lam = 1000):
    """
    Calculate the F2 dipole form factor for lepton i.
    
    This function computes the F2 form factor for diagonal transitions
    li → li γ, including both lepton loop contributions and ALP photon contributions.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of lepton
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    ph : array-like, optional
        C or CP violating phases for couplings
    ALP : bool, optional
        Whether to include ALP photon contributions
    Lam : float, optional
        ALP EFT scale (used only if ALP=True)
    
    Returns
    -------
    complex
        F2 dipole form factor
    """

    m = np.complex128(m)
    
    f2 = 0
    if ALP:
        f2 += F2_ALP_photon(m, i, g, th, Lam  = Lam)
        g, th, d, ph = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    
    for j in range(3):
        f2 += F2_lepton(m, i, j, g, th, d)
        
    return f2

def F3(m, i, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, ALP = False, Lam = 1000):
    """
    Calculate the F3 dipole form factor for lepton i.
    
    This function computes the F3 form factor for diagonal transitions
    li → li γ, including both lepton loop contributions and ALP photon contributions.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of lepton
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    ph : array-like, optional
        C or CP violating phases for couplings
    ALP : bool, optional
        Whether to include ALP photon contributions
    Lam : float, optional
        ALP EFT scale (used only if ALP=True)
    
    Returns
    -------
    complex
        F3 dipole form factor
    """
    
    m = np.complex128(m)
    
    if ALP:
        g, th, d, ph = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    
    f3 = 0
    for j in range(3):
        f3 += F3_lepton(m, i, j, g, th, d)
        
    return f3

def F2_lepton(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3):
    """
    Calculate the lepton loop contribution to F2 form factor.
    
    This function computes the contribution to F2 from internal
    lepton j in the loop for diagonal transitions li → li γ.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of external lepton
    j : int
        Index of internal lepton in loop
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    
    Returns
    -------
    complex
        Lepton loop contribution to F2
    """

    ff = gp(m, i, j) + gm(m, i, j)*np.cos(2*th[i][j])
    ff*= g[i][j]**2/(32*np.pi**2) 
    return ff

def F3_lepton(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3):
    """
    Calculate the lepton loop contribution to F3 form factor.
    
    This function computes the contribution to F3 from internal
    lepton j in the loop for diagonal transitions li → li γ.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of external lepton
    j : int
        Index of internal lepton in loop
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    
    Returns
    -------
    complex
        Lepton loop contribution to F3
    """

    ff = gm(m, i, j)*np.sin(2*th[i][j])*np.cos(d[i][j])
    ff*= -g[i][j]**2/(32*np.pi**2)
    return ff

#-------------------------------------------------------------------------#
# li on-diagonal dipole form factors with internal gamma and li (for ALP only)
#-------------------------------------------------------------------------#

def h_gamma(x):
    """
    Calculate the h_gamma function used in ALP photon contributions.
    
    This function appears in the calculation of ALP photon loop
    contributions to diagonal dipole form factors.
    
    Parameters
    ----------
    x : float or array-like
        Kinematic variable x = (m/ma)²
    
    Returns
    -------
    float or array-like
        Value of h_gamma function
    """
    exact = 1 + (x**2/6)*np.log(x) - x/3
    exact-= (x+2)/3 * np.sqrt((x-4)*x)*np.log((np.sqrt(x)+np.sqrt(x-4))/2)
    large = (3 + 2*np.log(x))/2 + 4*(-2 + 3*np.log(x))/(9*x) #treat large x separately to avoid floating point errors
    return np.where(x < 1e6, exact, large)

def F2_ALP_photon(m, i, C = [[1]*3]*3, TH = [[0]*3]*3, Lam = 1000):
    """
    Calculate the ALP photon contribution to F2 form factor.
    
    This function computes the contribution to F2 from ALP-photon
    loops for diagonal transitions li → li γ.
    
    Parameters
    ----------
    m : float
        ALP mass
    i : int
        Index of lepton
    C : array-like
        ALP-fermion coupling strengths
    TH : array-like
        PV angles for ALP-fermion couplings
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    complex
        ALP photon contribution to F2
    """
    Aii = C[i][i]*np.cos(TH[i][i])
    Cgg = (C[i][i])/(8*np.pi**2) # only contribution from Cii
    
    xi = (m/ml[i])**2

    I = 2*np.log(Lam/ml[i])- h_gamma(xi)
    
    return -64*np.pi*alpha/(16*np.pi**2) * Cgg * Aii * (ml[i]/Lam)**2  * I