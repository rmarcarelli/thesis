#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Approximate form factor functions for LFV calculations.

This module provides approximate form factor functions for lepton flavor
violating processes. These approximations are valid in different mass
hierarchies and are used to speed up calculations while maintaining
good accuracy.
"""

import numpy as np

from .special_functions import Li2
from phys.constants import ml

#-------------------------------------------------------#
#Approximate master functions for dipole form factors
#-------------------------------------------------------#

def f_plus_approx(i, j, k):
    """
    Create an approximate function for f_plus based on lepton mass hierarchy.
    
    This function returns an approximation function for f_plus that
    automatically selects the appropriate approximation based on the
    relative sizes of lepton masses i, j, k.
    
    Parameters
    ----------
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    k : int
        Index of third lepton
    
    Returns
    -------
    function
        Approximation function f_plus_approx(ui, uj, uk)
    """
    if i < j:
        return f_plus_approx(j, i, k)
    
    def approx(ui, uj, uk):
        xi, xj, xk = 1/ui**2, 1/uj**2, 1/uk**2
    
        if i == j:
            return g_plus_approx(i, k)(ui, uk)
        if i == k:
            return plus_large_m(f_plus_iji)(xi, xj, xk)
        if k < i:
            return plus_large_m(f_plus_small_k)(xi, xj, xk)
        if k > i:
            return plus_large_m(f_plus_large_k)(xi, xj, xk)
        
    return approx

def f_minus_approx(i, j, k):
    """
    Create an approximate function for f_minus based on lepton mass hierarchy.
    
    This function returns an approximation function for f_minus that
    automatically selects the appropriate approximation based on the
    relative sizes of lepton masses i, j, k.
    
    Parameters
    ----------
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    k : int
        Index of third lepton
    
    Returns
    -------
    function
        Approximation function f_minus_approx(ui, uj, uk)
    """
    if i < j:
        return f_plus_approx(j, i, k)
    
    def approx(ui, uj, uk):
        xi, xj, xk = 1/ui**2, 1/uj**2, 1/uk**2
    
        if i == j:
            return g_minus_approx(i, k)(ui, uk)
        if i == k:
            return minus_large_m(f_minus_iji)(xi, xj, xk)
        if k < i:
            return minus_large_m(f_minus_small_k)(xi, xj, xk)
        if k > i:
            return minus_large_m(f_minus_large_k)(xi, xj, xk)
        
    return approx

### helper functions (expressed in terms of xl = 1/ul^2)

#----------------------------------#
#Approximations for i = k >> j
#----------------------------------#

def f_plus_iji(xi, xj, xk):
    """
    Approximate f_plus for the case i = k >> j.
    
    This function provides an approximation for f_plus when
    lepton i equals lepton k and both are much heavier than j.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    xk : float or array-like
        Variable xk = 1/uk² where uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_plus for i = k >> j
    """
    
    exact = 2*xi - 3 - (xi - 3)*xi*np.log(xi)
    exact+= 2*(xi-1)*np.sqrt(xi*(xi-4))*np.log((xi + np.sqrt(xi*(xi-4)))/(2*np.sqrt(xi)))
    exact-= np.log((xi + np.sqrt(xi*(xi-4)))/(2*xi))**2
    exact-= 2*Li2(1-xi)
    exact-= 2*Li2((xi-np.sqrt(xi*(xi-4)))/(2*xi))
    exact+= 2*Li2((2-xi-np.sqrt(xi*(xi-4)))/2)

    approx = 1/(3*xi) + (2-np.log(xi))/xi**2
    
    return np.where(xi > 100, approx, exact)

def f_minus_iji(xi, xj, xk):
    """
    Approximate f_minus for the case i = k >> j.
    
    This function provides an approximation for f_minus when
    lepton i equals lepton k and both are much heavier than j.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    xk : float or array-like
        Variable xk = 1/uk² where uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_minus for i = k >> j
    """
    
    exact = -2 + (xi - 3)*xi/(xi-1)*np.log(xi)
    exact-= 2*np.sqrt(xi*(xi-4))*np.log((xi + np.sqrt(xi*(xi-4)))/(2*np.sqrt(xi)))
    exact-= np.log((xi + np.sqrt(xi*(xi-4)))/(2*xi))**2
    exact-= 2*Li2(1-xi)
    exact-= 2*Li2((xi-np.sqrt(xi*(xi-4)))/(2*xi))
    exact+= 2*Li2((2-xi-np.sqrt(xi*(xi-4)))/2)

    approx = (-3+2*np.log(xi))/xi + (-47 + 42*np.log(xi))/(6*xi**2)
    
    return np.where(xi > 100, approx, exact)

#----------------------------------#
#Approximations for i >> j, k
#----------------------------------#

def f_plus_small_k(xi, xj, xk):
    """
    Approximate f_plus for the case i >> j, k.
    
    This function provides an approximation for f_plus when
    lepton i is much heavier than both j and k.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    xk : float or array-like
        Variable xk = 1/uk² where uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_plus for i >> j, k
    """
    
    exact = np.conj((-1 + 2*xi + 2*(xi-1)*xi*np.log((xi-1)/xi)))
    
    approx = 1/(3*xi) + 1/(6*xi**2)
    
    return np.where(xi > 100, approx, exact)

def f_minus_small_k(xi, xj, xk):
    """
    Approximate f_minus for the case i >> j, k.
    
    This function provides an approximation for f_minus when
    lepton i is much heavier than both j and k.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    xk : float or array-like
        Variable xk = 1/uk² where uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_minus for i >> j, k
    """
    
    exact = np.conj(1 + (np.log((xi-1)*xk/xi) + xi - 1)*np.log((xi-1)/xi) + Li2(1/xi))
    
    approx = (3-2*np.log(xk))/(2*xi) + (17-6*np.log(xk))/(12*xi**2)

    return -2*np.sqrt(xi)/np.sqrt(xk) * np.where(xi > 100, approx, exact)

#----------------------------------#
#Approximations for k >> i >> j
#----------------------------------#

def f_plus_large_k(xi, xj, xk):
    """
    Approximate f_plus for the case k >> i >> j.
    
    This function provides an approximation for f_plus when
    lepton k is much heavier than i, which is much heavier than j.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    xk : float or array-like
        Variable xk = 1/uk² where uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_plus for k >> i >> j
    """
    f_val = (2*xk**2 + 5*xk - 1)/(6*(xk-1)**3) - xk**2/(xk-1)**4 * np.log(xk)
    return xk*(1/np.sqrt(xi)+1/np.sqrt(xj))**2 * f_val

def f_minus_large_k(xi, xj, xk):
    """
    Approximate f_minus for the case k >> i >> j.
    
    This function provides an approximation for f_minus when
    lepton k is much heavier than i, which is much heavier than j.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    xk : float or array-like
        Variable xk = 1/uk² where uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_minus for k >> i >> j
    """
    return np.sqrt(xk)*(1/np.sqrt(xi) + 1/np.sqrt(xj))*(-(3*xk-1)/(xk-1)**2 + 2*xk**2/(xk-1)**3 * np.log(xk))
       
#----------------------------------#
#Approximations for m >> mi, mj, mk       
#----------------------------------#

def f_large_m(ui, uj, uk):
    """
    Approximate f for the case m >> mi, mj, mk.
    
    This function provides an approximation for f when the scalar mass
    is much larger than all lepton masses.
    
    Parameters
    ----------
    ui : float or array-like
        Variable ui = mi/m
    uj : float or array-like
        Variable uj = mj/m
    uk : float or array-like
        Variable uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f for m >> mi, mj, mk
    """
    return -2*uk*(ui+uj+uk)*(1+2*np.log(uk))+((ui+uj)**2 - 3*(ui+uj)*uk+6*uk**2+12*uk**2*np.log(uk))

def f_minus_large_m(ui, uj, uk):
    """
    Approximate f_minus for the case m >> mi, mj, mk.
    
    This function provides an approximation for f_minus when the scalar mass
    is much larger than all lepton masses.
    
    Parameters
    ----------
    ui : float or array-like
        Variable ui = mi/m
    uj : float or array-like
        Variable uj = mj/m
    uk : float or array-like
        Variable uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_minus for m >> mi, mj, mk
    """
    return -(ui+uj)*uk*(3 + 4*np.log(uk))

def f_plus_large_m(ui, uj, uk):
    """
    Approximate f_plus for the case m >> mi, mj, mk.
    
    This function provides an approximation for f_plus when the scalar mass
    is much larger than all lepton masses.
    
    Parameters
    ----------
    ui : float or array-like
        Variable ui = mi/m
    uj : float or array-like
        Variable uj = mj/m
    uk : float or array-like
        Variable uk = mk/m
    
    Returns
    -------
    complex or array-like
        Approximate value of f_plus for m >> mi, mj, mk
    """
    return (ui+uj)**2/3

#----------------------------------#
#Wrappers for applying a different approximation at large m 
#----------------------------------#
def plus_large_m(f_plus_approx):
    """
    Wrapper function to apply large m approximation for f_plus.
    
    This function wraps an f_plus approximation to automatically
    switch to the large m approximation when xi < 1e6.
    
    Parameters
    ----------
    f_plus_approx : function
        The f_plus approximation function to wrap
    
    Returns
    -------
    function
        Wrapped approximation function
    """
    def approx(xi, xj, xk):
        #x_max = np.maximum(xi, xk)
        ui, uj, uk = 1/np.sqrt(xi), 1/np.sqrt(xj), 1/np.sqrt(xk)
        return np.where(xi < 1e6,
                        f_plus_approx(xi, xj, xk),
                        f_plus_large_m(ui, uj, uk))
    
    return approx
    
def minus_large_m(f_minus_approx):
    """
    Wrapper function to apply large m approximation for f_minus.
    
    This function wraps an f_minus approximation to automatically
    switch to the large m approximation when xi < 1e6.
    
    Parameters
    ----------
    f_minus_approx : function
        The f_minus approximation function to wrap
    
    Returns
    -------
    function
        Wrapped approximation function
    """
    def approx(xi, xj, xk):
        #x_max = np.maximum(xi, xk)
        ui, uj, uk = 1/np.sqrt(xi), 1/np.sqrt(xj), 1/np.sqrt(xk)
        return np.where(xi < 1e6,
                        f_minus_approx(xi, xj, xk),
                        f_minus_large_m(ui, uj, uk))
    return approx


#----------------------------------------------------------------#
#Approximate master functions for diagonal dipole form factors
#----------------------------------------------------------------#

def g_plus_approx(i, j):
    """
    Create an approximate function for g_plus based on lepton mass hierarchy.
    
    This function returns an approximation function for g_plus that
    automatically selects the appropriate approximation based on the
    relative sizes of lepton masses i and j.
    
    Parameters
    ----------
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    
    Returns
    -------
    function
        Approximation function g_plus_approx(ui, uj)
    """
    
    def approx(ui, uj):
        xi, xj = 1/ui**2, 1/uj**2
        
        if i == j:
            return g_plus_ii(xi, xj)
        if j < i:
            return g_plus_small_j(xi, xj)
        if j > i:
            return g_plus_large_j(xi, xj)
    
    return approx

def g_minus_approx(i, j):
    """
    Create an approximate function for g_minus based on lepton mass hierarchy.
    
    This function returns an approximation function for g_minus that
    automatically selects the appropriate approximation based on the
    relative sizes of lepton masses i and j.
    
    Parameters
    ----------
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    
    Returns
    -------
    function
        Approximation function g_minus_approx(ui, uj)
    """
    
    def approx(ui, uj):
        xi, xj = 1/ui**2, 1/uj**2
        
        if i == j:
            return g_minus_ii(xi, xj)
        if j < i:
            return g_minus_small_j(xi, xj)
        if j > i:
            return g_minus_large_j(xi, xj)
    
    return approx
    
#----------------------------------#
#Approximation for i = j
#----------------------------------#  
def g_plus_ii(xi, xj):
    """
    Approximate g_plus for the case i = j.
    
    This function provides an approximation for g_plus when
    lepton i equals lepton j.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    
    Returns
    -------
    complex or array-like
        Approximate value of g_plus for i = j
    """
    
    exact = 2 - 4*xi + 2*xi*(xi - 2)*np.log(xi)
    exact-= 4*(xi**2 - 4*xi + 2)*np.sqrt(xi/(xi-4))*np.log((np.sqrt(xi) + np.sqrt(xi-4))/2)

    approx = (25 + 4*xi - 12*np.log(xi))/(3*xi**2)
    
    return np.where(xi > 1e3, approx, exact)

def g_minus_ii(xi, xj):
    """
    Approximate g_minus for the case i = j.
    
    This function provides an approximation for g_minus when
    lepton i equals lepton j.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    
    Returns
    -------
    complex or array-like
        Approximate value of g_minus for i = j
    """
    
    exact = 4 - 2*xi*np.log(xi)
    exact+= 4*(xi - 2)*np.sqrt(xi/(xi-4))*np.log((np.sqrt(xi) + np.sqrt(xi-4))/2)

    approx = -2*(32 + 9*xi - 6*(4 + xi)*np.log(xi))/(3*xi**2)
    
    return np.where(xi > 1e3, approx, exact)

#----------------------------------#
#Approximation for j << i
#----------------------------------#
def g_plus_small_j(xi, xj):
    """
    Approximate g_plus for the case j << i.
    
    This function provides an approximation for g_plus when
    lepton j is much lighter than lepton i.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    
    Returns
    -------
    complex or array-like
        Approximate value of g_plus for j << i
    """
    
    exact = -2-4*xi + 4*xi**2 * np.conj(np.log(xi/(xi - 1)))
    
    approx = 4/(3*xi) + 1/xi**2
    
    return np.where(xi > 1e3, approx, exact)
    
def g_minus_small_j(xi, xj):
    """
    Approximate g_minus for the case j << i.
    
    This function provides an approximation for g_minus when
    lepton j is much lighter than lepton i.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    
    Returns
    -------
    complex or array-like
        Approximate value of g_minus for j << i
    """
    
    exact = 1 -  (xi**2+1)/(xi-1)*np.conj(np.log(xi/(xi-1))) + 1/(xi-1)*np.log(xj)
    
    approx = (-3 + 2*np.log(xj))/(2*xi) + (-17 + 6*np.log(xj))/(6*xi**2)
    
    return 4*np.sqrt(xi/xj) *np.where(xi > 1e3, approx, exact)        


#----------------------------------#
#Approximation for j >> i
#----------------------------------#
def g_plus_large_j(xi, xj):
    """
    Approximate g_plus for the case j >> i.
    
    This function provides an approximation for g_plus when
    lepton j is much heavier than lepton i.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    
    Returns
    -------
    complex or array-like
        Approximate value of g_plus for j >> i
    """
    return 2*xj/xi * ((2*xj**2 + 5*xj - 1)/(3*(xj-1)**3) - (2*xj**2/(xj-1)**4)*np.log(xj))

def g_minus_large_j(xi, xj):
    """
    Approximate g_minus for the case j >> i.
    
    This function provides an approximation for g_minus when
    lepton j is much heavier than lepton i.
    
    Parameters
    ----------
    xi : float or array-like
        Variable xi = 1/ui² where ui = mi/m
    xj : float or array-like
        Variable xj = 1/uj² where uj = mj/m
    
    Returns
    -------
    complex or array-like
        Approximate value of g_minus for j >> i
    """
    return 2*np.sqrt(xj/xi) * ((1 - 3*xj)/(xj - 1)**2 + (2*xj**2)/(xj-1)**3 * np.log(xj))

    
#----------------------------------------------------------------------#
# Helper functions for substituting into form factors
#----------------------------------------------------------------------#

def f2p(m, i, j, k):
    """
    Calculate f2p form factor using approximations.
    
    This function computes the f2p form factor using the
    appropriate approximation based on lepton mass hierarchy.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    k : int
        Index of third lepton
    
    Returns
    -------
    complex
        Value of f2p form factor
    """
    u = np.complex128([_ml/m for _ml in ml])
    return f_plus_approx(i, j, k)(u[i], u[j], u[k])

def f2m(m, i, j, k):
    """
    Calculate f2m form factor using approximations.
    
    This function computes the f2m form factor using the
    appropriate approximation based on lepton mass hierarchy.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    k : int
        Index of third lepton
    
    Returns
    -------
    complex
        Value of f2m form factor
    """
    u = np.complex128([_ml/m for _ml in ml])
    return f_minus_approx(i, j, k)(u[i], u[j], u[k])

def f3p(m, i, j, k):
    """
    Calculate f3p form factor using approximations.
    
    This function computes the f3p form factor using the
    appropriate approximation based on lepton mass hierarchy.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    k : int
        Index of third lepton
    
    Returns
    -------
    complex
        Value of f3p form factor
    """
    u = np.complex128([_ml/m for _ml in ml])
    return f_plus_approx(i, j, k)(u[i], -u[j], u[k])

def f3m(m, i, j, k):
    """
    Calculate f3m form factor using approximations.
    
    This function computes the f3m form factor using the
    appropriate approximation based on lepton mass hierarchy.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    k : int
        Index of third lepton
    
    Returns
    -------
    complex
        Value of f3m form factor
    """
    u = np.complex128([_ml/m for _ml in ml])
    return f_minus_approx(i, j, k)(u[i], -u[j], u[k])

def gp(m, i, j):
    """
    Calculate gp form factor using approximations.
    
    This function computes the gp form factor using the
    appropriate approximation based on lepton mass hierarchy.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    
    Returns
    -------
    complex
        Value of gp form factor
    """
    u = np.complex128([_ml/m for _ml in ml])
    return g_plus_approx(i, j)(u[i], u[j])

def gm(m, i, j):
    """
    Calculate gm form factor using approximations.
    
    This function computes the gm form factor using the
    appropriate approximation based on lepton mass hierarchy.
    
    Parameters
    ----------
    m : float
        Scalar mass
    i : int
        Index of first lepton
    j : int
        Index of second lepton
    
    Returns
    -------
    complex
        Value of gm form factor
    """
    u = np.complex128([_ml/m for _ml in ml])
    return g_minus_approx(i, j)(u[i], u[j])