#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .form_factor_functions_approx import f2p, f2m, f3p, f3m
from .special_functions import Li2
from phys.constants import alpha, ml
from phys.scalar_ALP_EFT import ALP_to_scalar

#-------------------------------------------------------------------------#
# Total li -> lj gamma decay rate given a set of LFV couplings in terms of
# their magnitudes g, angles th, and phases ph and d. Can explicitly turn 
# on PC or chiral coupling (without setting phases) by choosing mode = 'PC'
# or mode = 'chiral'. Can convert to LFV ALP by setting ALP = True, (in this
# case, the couplings refer to the analogous LFV ALP couplings). Rather than
# computing f_plus and f_minus each time, their values are passed through as
# arguments fp_ij and fm_ij. The mass m is also passed through, but this is
# only used for the ALP photon contribution. 
#-------------------------------------------------------------------------#

def radiative_decay_rate(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    """
    Calculate the total radiative decay rate li → lj γ.
    
    This function computes the complete radiative decay rate including
    both F2 and F3 form factor contributions. It can handle different
    coupling modes (PC, chiral) and ALP scenarios.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
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
        Radiative decay rate li → lj γ
    """

    if mode == 'PC':
        th = [[0]*3]*3
        d = [[0]*3]*3
        ph = [[0]*3]*3
        
    if mode == 'chiral':
        th = [[np.pi/4]*3]*3
        d = [[0]*3]*3 if ALP else [[np.pi/2]*3]*3
        ph = [[0]*3]*3
    
    f2 = F2(m, i, j, g, th, d, ph, ALP, Lam)
    f3 = F3(m, i, j, g, th, d, ph, ALP, Lam)
        
    return (alpha/2) * ((ml[i]-ml[j])**2 * np.abs(f2)**2 + (ml[i]+ml[j])**2*np.abs(f3)**2)/ml[i]
 

#-------------------------------------------------------------------------#
# Transition dipole form factors
#-------------------------------------------------------------------------#

def F2(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, ALP = False, Lam = 1000):
    """
    Calculate the F2 transition dipole form factor.
    
    This function computes the F2 form factor for li → lj γ transitions,
    including both lepton loop contributions and ALP photon contributions.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
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
        F2 transition dipole form factor
    """
    
    m = np.complex128(m)
    
    f2 = 0
    
    if ALP:
        f2 += F2_ALP_photon(m, i, j, g, th, d, ph, Lam = Lam)
        g, th, d, ph = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    
    for k in range(3):
        f2 += F2_lepton(m, i, j, k, g, th, d, ph)

    return f2

def F3(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, ALP = False, Lam = 1000):
    """
    Calculate the F3 transition dipole form factor.
    
    This function computes the F3 form factor for li → lj γ transitions,
    including both lepton loop contributions and ALP photon contributions.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
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
        F3 transition dipole form factor
    """

    m = np.complex128(m)
    
    f3 = 0
    
    if ALP:
        f3 += F3_ALP_photon(m, i, j, g, th, d, ph, Lam  = Lam)
        g, th, d, ph = ALP_to_scalar(g, th, d, ph, Lam = Lam)
    
    for k in range(3):
        f3 += F3_lepton(m, i, j, k, g, th, d, ph)

    return f3

def F2_lepton(m, i, j, k, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3):
    """
    Calculate the lepton loop contribution to F2 form factor.
    
    This function computes the contribution to F2 from internal
    lepton k in the loop for li → lj γ transitions.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
    k : int
        Index of internal lepton in loop
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    ph : array-like, optional
        C or CP violating phases for couplings
    
    Returns
    -------
    complex
        Lepton loop contribution to F2
    """
    
    f2p_ijk = f2p(m, i, j, k)
    f2m_ijk = f2m(m, i, j, k)

    d_diff = d[j][k]-d[i][k]
    ph_diff = ph[j][k] - ph[i][k]
    
    ff = np.cos(th[i][k])*np.cos(th[j][k])*(f2p_ijk+f2m_ijk)
    ff+= np.exp(1j*d_diff)*np.sin(th[i][k])*np.sin(th[j][k])*(f2p_ijk-f2m_ijk)
    ff = np.exp(1j*ph_diff)*g[i][k]*g[j][k]*ff/(32*np.pi**2)
    return ff

def F3_lepton(m, i, j, k, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3):
    """
    Calculate the lepton loop contribution to F3 form factor.
    
    This function computes the contribution to F3 from internal
    lepton k in the loop for li → lj γ transitions.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
    k : int
        Index of internal lepton in loop
    g : array-like, optional
        Coupling strengths
    th : array-like, optional
        PV angles for couplings
    d : array-like, optional
        C or CP violating phases for couplings
    ph : array-like, optional
        C or CP violating phases for couplings
    
    Returns
    -------
    complex
        Lepton loop contribution to F3
    """

    f3p_ijk = f3p(m, i, j, k)
    f3m_ijk = f3m(m, i, j, k)
    
    ph_diff = ph[j][k] - ph[i][k]

    ff = np.exp(-1j*d[j][k])*np.cos(th[i][k])*np.sin(th[j][k])*(f3p_ijk+f3m_ijk)
    ff+= -np.exp(1j*d[i][k])*np.sin(th[i][k])*np.cos(th[j][k])*(f3p_ijk-f3m_ijk)
    ff = np.exp(1j*ph_diff)*g[i][k]*g[j][k]*ff/(32*np.pi**2)
    return ff

#-------------------------------------------------------------------------#
# li -> lj gamma form factors through internal gamma and lj (for ALP only)
#-------------------------------------------------------------------------#

def g_gamma(x):
    """
    Calculate the g_gamma function used in ALP photon contributions.
    
    This function appears in the calculation of ALP photon loop
    contributions to dipole form factors.
    
    Parameters
    ----------
    x : float or array-like
        Kinematic variable x = (m/ma)²
    
    Returns
    -------
    complex or array-like
        Value of g_gamma function
    """
    x = np.complex128(x)
    return np.log(x)/(x-1)+(x-1)*np.log(x/(x-1))+2
 
def F2_ALP_photon(m, i, j, C, TH, D, PH, Lam = 1000):
    """
    Calculate the ALP photon contribution to F2 form factor.
    
    This function computes the contribution to F2 from ALP-photon
    loops for li → lj γ transitions.
    
    Parameters
    ----------
    m : float
        ALP mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
    C : array-like
        ALP-fermion coupling strengths
    TH : array-like
        PV angles for ALP-fermion couplings
    D : array-like
        C or CP violating phases for ALP couplings
    PH : array-like
        C or CP violating phases for ALP couplings
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    complex
        ALP photon contribution to F2
    """

    Aij = C[i][j]*np.cos(TH[i][j])*np.exp(1j*(PH[i][j] + D[i][j]))
    Cgg = (C[0][0]+C[1][1]+C[2][2])/(8*np.pi**2)
    
    xi = (m/ml[i])**2

    I = 2*np.log(Lam**2/m**2) - g_gamma(xi)

    return -alpha/(2*np.pi) * Cgg * Aij * (ml[i]/Lam)**2  * I

 
def F3_ALP_photon(m, i, j, C, TH, D, PH, Lam = 1000):
    """
    Calculate the ALP photon contribution to F3 form factor.
    
    This function computes the contribution to F3 from ALP-photon
    loops for li → lj γ transitions.
    
    Parameters
    ----------
    m : float
        ALP mass
    i : int
        Index of initial lepton
    j : int
        Index of final lepton
    C : array-like
        ALP-fermion coupling strengths
    TH : array-like
        PV angles for ALP-fermion couplings
    D : array-like
        C or CP violating phases for ALP couplings
    PH : array-like
        C or CP violating phases for ALP couplings
    Lam : float, optional
        ALP EFT scale
    
    Returns
    -------
    complex
        ALP photon contribution to F3
    """
    
    Vij = C[i][j]*np.sin(TH[i][j])*np.exp(1j*(PH[i][j]))
    Cgg = (C[0][0]+C[1][1]+C[2][2])/(8*np.pi**2)

    xi = (m/ml[i])**2

    I = 2*np.log(Lam**2/m**2) - g_gamma(xi)
        
    return -alpha/(2*np.pi) * Cgg * Vij * (ml[i]/Lam)**2  * I

#-------------------------------------------------------------------------#
# Total trilepton decay rate ... 
#-------------------------------------------------------------------------#

def trilepton_decay_rate(m, i, j, k, l, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    """
    Calculate the total trilepton decay rate li → lj lk l̅l.
    
    This function computes the complete trilepton decay rate including
    both radiative and tree-level contributions, with proper handling
    of on-shell and off-shell scenarios.
    
    Parameters
    ----------
    m : float
        Scalar/ALP mass
    i : int
        Index of initial lepton
    j : int
        Index of first final lepton
    k : int
        Index of second final lepton
    l : int
        Index of third final lepton (must equal j, k, or j=k)
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
        Total trilepton decay rate
    """

    g = np.array(g)
    
    assert l == j or l == k or j == k
    
    if mode == 'PC':
        th = [[0]*3]*3
        d = [[0]*3]*3
        ph = [[0]*3]*3
        
    if mode == 'chiral':
        th = [[np.pi/4]*3]*3
        d = [[0]*3]*3 if ALP else [[np.pi/2]*3]*3
        ph = [[0]*3]*3
        
    # Radiative contribution
    radiative_rate = (alpha/(6*np.pi))*(2*np.log(ml[i]/ml[j]) - 3)*radiative_decay_rate(m, i, j, g, th, d, ph, mode, ALP, Lam)
    radiative_rate += (alpha/(6*np.pi))*(2*np.log(ml[i]/ml[k]) - 3)*radiative_decay_rate(m, i, k, g, th, d, ph, mode, ALP, Lam) 
        
    if ALP:
        g, th, d, ph = ALP_to_scalar(g, th, d, ph, Lam = Lam)
        
    # Tree-level contribution
    tree_level_rate = ((g[i][j]*g[k][l])**2 + (g[i][k]*g[j][l])**2)*h1(np.complex128(m/ ml[i])**2)
    tree_level_rate += 2*g[i][j]*g[k][l]*g[i][k]*g[j][l]*S(i, j, k, l, th, d, ph)*h2(np.complex128(m/ml[i])**2)
    tree_level_rate = ml[i]/(512*np.pi**3) * tree_level_rate
    
    # If m_\phi < m_\tau... use narrow width approximation
    on_shell_rate = ml[i]/(16*np.pi) * (1 - m**2 / ml[i]**2)**2 
    
    # Up to mass-dependence of the decay width, the branching-fraction is just
    # the ratio of the squared couplings to the sum of all squared couplings. 
    on_shell_rate*= ((g[i][j] * g[k][l])**2 + ((g[i][k] * g[j][l])**2))/np.sum(g**2)
    
    tree_level_rate = np.where(m > ml[i], tree_level_rate, np.maximum(tree_level_rate, on_shell_rate))
    
    total_rate = tree_level_rate + radiative_rate
        
    return total_rate

def h1(x):
    """
    Calculate the h1 function used in trilepton decay rate.
    
    This function appears in the tree-level contribution to
    trilepton decay rates.
    
    Parameters
    ----------
    x : float or array-like
        Kinematic variable x = (m/ml)²
    
    Returns
    -------
    float or array-like
        Value of h1 function
    """
    return np.where(x > 100, 1/(6*x**2), -5 + 6*x + 2*(1 - 4*x  + 3*x**2)*np.log(np.abs((x-1)/x)))

def h2(x):
    """
    Calculate the h2 function used in trilepton decay rate.
    
    This function appears in the tree-level contribution to
    trilepton decay rates.
    
    Parameters
    ----------
    x : float or array-like
        Kinematic variable x = (m/ml)²
    
    Returns
    -------
    float or array-like
        Value of h2 function
    """
    return np.where(x > 100, 1/(12*x**2), 1 - 4*x - np.pi**2/3 * x**2 + 4*(x-1)*x*np.log(x/(x-1)) + 2*x**2 * (np.log(x/(2*x-1))**2 + 2*Li2(x/(2*x-1)))).real

def S(i, j, k, l, th, d, ph):
    """
    Calculate the S function for trilepton decay interference terms.
    
    This function computes the interference term between different
    coupling combinations in trilepton decays.
    
    Parameters
    ----------
    i : int
        Index of initial lepton
    j : int
        Index of first final lepton
    k : int
        Index of second final lepton
    l : int
        Index of third final lepton
    th : array-like
        PV angles for couplings
    d : array-like
        C or CP violating phases for couplings
    ph : array-like
        C or CP violating phases for couplings
    
    Returns
    -------
    float
        Value of S function
    """
    
    s = U(th[i][j], d[i][j])*Ub(th[j][l], d[j][l])*(U(th[i][j], d[i][j])*Ub(th[j][l], d[j][l])).conj()
    s+= Ub(th[i][j], d[i][j])*U(th[j][l], d[j][l])*(Ub(th[i][j], d[i][j])*U(th[j][l], d[j][l])).conj()
    s*=np.exp(1j*(ph[i][j]+ph[j][l] - ph[i][k]-ph[k][l]))
    
    return s.real/2
                                                                    
def U(th, d):
    """
    Calculate the U function for coupling combinations.
    
    Parameters
    ----------
    th : float
        PV angle
    d : float
        C or CP violating phase
    
    Returns
    -------
    complex
        Value of U function
    """
    return np.cos(th) + 1j*np.exp(1j*d)*np.sin(th)

def Ub(th, d):
    """
    Calculate the Ub (U bar) function for coupling combinations.
    
    Parameters
    ----------
    th : float
        PV angle
    d : float
        C or CP violating phase
    
    Returns
    -------
    complex
        Value of Ub function
    """
    return np.cos(th) - 1j*np.exp(1j*d)*np.sin(th)