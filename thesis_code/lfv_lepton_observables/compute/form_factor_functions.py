#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mpmath as mp
mp.mp.dps = 30

from .special_functions import B, C0

#-----------------------------------------------------#
#Master functions for dipole form factor with i =/= j
#-----------------------------------------------------#

@np.vectorize #Unfortunately, mpmath does not work with arrays
def f(ui, uj, uk):
    """
    Calculate the master function f for dipole form factors with i ≠ j.
    
    This function evaluates the complete form factor expression
    for off-diagonal lepton flavor transitions. It combines B functions,
    logarithmic terms, constants, and C0 integrals.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable (normalized mass ratio)
    uj : float or array-like
        Second kinematic variable (normalized mass ratio)
    uk : float or array-like
        Third kinematic variable (normalized mass ratio)
    
    Returns
    -------
    complex or array-like
        Value of the master function f(ui, uj, uk)
    """
    if ui == uj:
        return g_plus(ui, uk) + g_minus(ui, uk)
    elif ui == -uj:
        return g_minus(ui, uk)
    
    #increased precision
    ui = mp.mpc(ui)
    uj = mp.mpc(uj)
    uk = mp.mpc(uk)
    
    f_val = fBik(ui, uj, uk)*B(ui, uk)
    f_val+= fBjk(ui, uj, uk)*B(uj, uk)
    f_val+= fLog(ui, uj, uk)*mp.log(uk)
    f_val+= fConst(ui, uj, uk)
    f_val+= fC0(ui, uj, uk)*C0(ui, uj, uk)
    
    return np.complex128(f_val)

#split function into ``odd'' and ``even'' functions of ui, uj
def f_plus(ui, uj, uk):
    """
    Calculate the even part of the master function f.
    
    This function extracts the part of f that is even under
    the transformation (ui, uj) → (-ui, -uj).
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Even part of f(ui, uj, uk)
    """
    return (f(ui, uj, uk) + f(-ui, -uj, uk))/2

def f_minus(ui, uj, uk):
    """
    Calculate the odd part of the master function f.
    
    This function extracts the part of f that is odd under
    the transformation (ui, uj) → (-ui, -uj).
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Odd part of f(ui, uj, uk)
    """
    return (f(ui, uj, uk) - f(-ui, -uj, uk))/2

#----------------------------------#
#Helper functions for evaluating f
#----------------------------------#

def fBik(ui, uj, uk):
    """
    Calculate the coefficient of B(ui, uk) in the master function f.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Coefficient of B(ui, uk) term
    """
    return ((2*uk+uj)*ui**2 - (2*ui + uj)*(uk**2 + 2*ui*uk - 1))/(ui*(ui**2 - uj**2))

def fBjk(ui, uj, uk):  
    """
    Calculate the coefficient of B(uj, uk) in the master function f.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Coefficient of B(uj, uk) term
    """
    return fBik(uj, ui, uk)

def fLog(ui, uj, uk): 
    """
    Calculate the coefficient of log(uk) in the master function f.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Coefficient of log(uk) term
    """
    return -(((ui+uj)*(1-uk**2)-ui*uj*uk)**2 - 3*(ui*uj*uk)**2)/(ui**3*uj**3)

def fConst(ui, uj, uk):  
    """
    Calculate the constant term in the master function f.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Constant term in f
    """
    return -(1+ui*uj-uk**2)/(ui*uj)

def fC0(ui, uj, uk): 
    """
    Calculate the coefficient of C0(ui, uj, uk) in the master function f.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable
    uj : float or array-like
        Second kinematic variable
    uk : float or array-like
        Third kinematic variable
    
    Returns
    -------
    complex or array-like
        Coefficient of C0(ui, uj, uk) term
    """
    return -2*(ui+uj+uk)*uk

#----------------------------------------------------------------------#
# Master functions relevant for magnetic dipole form factor with i = j
#----------------------------------------------------------------------#

@np.vectorize #Unfortunately, mpmath does not work with arrays
def g_plus(ui, uj):
    """
    Calculate the g_plus function for magnetic dipole form factors with i = j.
    
    This function evaluates the even part of the form factor
    for diagonal lepton flavor transitions.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable (normalized mass ratio)
    uj : float or array-like
        Second kinematic variable (normalized mass ratio)
    
    Returns
    -------
    complex or array-like
        Value of g_plus(ui, uj)
    """
    ui = mp.mpc(ui)
    uj = mp.mpc(uj)
    
    g_val = -2*(ui**4 - (1-uj**2)*((1 + ui**2 - uj**2)**2 - 3*ui**2))
    g_val/= (1+ui**2 - uj**2)**2 - 4*ui**2
    g_val*= B(ui, uj)
    g_val+= (2*(1-uj**2)**2 - 2*ui**2 * uj**2)/ui**2 * mp.log(uj)
    g_val+= 2 + ui**2 - 2*uj**2 

    return np.complex128(-2/ui**2 * g_val)
 
@np.vectorize #Unfortunately, mpmath does not work with arrays
def g_minus(ui, uj):
    """
    Calculate the g_minus function for magnetic dipole form factors with i = j.
    
    This function evaluates the odd part of the form factor
    for diagonal lepton flavor transitions.
    
    Parameters
    ----------
    ui : float or array-like
        First kinematic variable (normalized mass ratio)
    uj : float or array-like
        Second kinematic variable (normalized mass ratio)
    
    Returns
    -------
    complex or array-like
        Value of g_minus(ui, uj)
    """
    ui = mp.mpc(ui)
    uj = mp.mpc(uj)
    
    g_val = (1 + ui**2 - uj**2)**2 - 2*ui**2
    g_val/= (1+ui**2 - uj**2)**2 - 4*ui**2
    g_val*= B(ui, uj)
    g_val+= (1 + ui**2 - uj**2)/ui**2 * mp.log(uj)
    g_val+= 1

    return np.complex128(4*uj/ui * g_val)