#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mpmath as mp
mp.mp.dps = 30
from scipy.special import spence

#-----------------------------------------------------#
#Special functions for evaluating dipole form factors
#-----------------------------------------------------#

def Li2(x):
    """
    Calculate the dilogarithm function Li2(x).
    
    Uses scipy's spence function which computes Li2(1-x).
    
    Parameters
    ----------
    x : complex or array-like
        Argument of the dilogarithm function
    
    Returns
    -------
    complex or array-like
        Value of Li2(x)
    """
    # spence doesn't work with mp, so have to 
    # convert back to numpy
    x = np.complex128(x)
    return spence(1-x)

def DiLog(u, e):
    """
    Calculate the dilogarithm with branch condition based on parameter e.
    
    This function handles the branch structure of the dilogarithm
    function based on the sign of the parameter e.
    
    Parameters
    ----------
    u : complex or array-like
        Argument of the dilogarithm
    e : complex or array-like
        Parameter that determines the branch condition
    
    Returns
    -------
    complex or array-like
        Value of DiLog(u, e) with appropriate branch structure
    """
    # np.where doesn't work with mp, so have to
    # convert back to numpy
    e = np.complex128(e)
    
    return np.where(e < 0, Li2(u), -0.5*mp.log(1/(1-u))**2-Li2(u/(u-1)))

def KallenL(x, y, z):
    """
    Calculate the Källén function λ(x², y², z²).
    
    The Källén function is defined as:
    λ(x², y², z²) = x² + y² + z² - 2xy - 2xz - 2yz
    
    This function appears in phase space integrals and
    determines the kinematic boundaries of processes.
    
    Parameters
    ----------
    x : float or array-like
        First variable
    y : float or array-like
        Second variable
    z : float or array-like
        Third variable
    
    Returns
    -------
    float or array-like
        Value of the Källén function λ(x², y², z²)
    """
    
    return x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z

def DiLog_fn(x, z, e):
    """
    Calculate an intermediate function for C0 integrals.
    
    This function computes a combination of DiLog functions
    used in the evaluation of three-point scalar integrals C0.
    
    Parameters
    ----------
    x : float or array-like
        First kinematic variable
    z : float or array-like
        Second kinematic variable
    e : float or array-like
        Parameter determining branch structure
    
    Returns
    -------
    complex or array-like
        Value of the intermediate DiLog function
    """
    
    argm = 2*e/(-1+x**2+z**2 - mp.sqrt(KallenL(1, x**2, z**2)))
    argp = 2*e/(-1+x**2+z**2 + mp.sqrt(KallenL(1, x**2, z**2)))
    
    return DiLog(argm, e) + DiLog(argp, -e)

def C0(x, y, z):
    """
    Calculate the three-point scalar integral C0.
    
    This is a special case of the ScalarC0 function as defined in PackageX.
    C0 integrals appear in loop calculations and form factor evaluations.
    
    Parameters
    ----------
    x : float or array-like
        First kinematic variable
    y : float or array-like
        Second kinematic variable
    z : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Value of the C0 integral
    """
    
    Dxy = DiLog_fn(x, z, -1+z**2) - DiLog_fn(x, z, -1+x**2+z**2) 
    Dyx = DiLog_fn(y, z, -1+z**2) - DiLog_fn(y, z, -1+y**2+z**2)
    
    return (Dxy - Dyx)/(x**2 - y**2)

def B(x, y):
    """
    Calculate the DiscB function as defined in PackageX.
    
    This function appears in two-point scalar integral calculations
    and is used in form factor evaluations.
    
    Parameters
    ----------
    x : float or array-like
        First kinematic variable
    y : float or array-like
        Second kinematic variable
    
    Returns
    -------
    complex or array-like
        Value of the B function
    """
    
    arg = (1-x**2+y**2 + mp.sqrt(KallenL(1, x**2, y**2)))/(2*y)
    
    return mp.sqrt(KallenL(1, x**2, y**2))*mp.log(arg)/x**2