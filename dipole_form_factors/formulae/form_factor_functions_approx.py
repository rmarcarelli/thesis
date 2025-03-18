#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mpmath as mp
mp.mp.dps = 30

from dipole_form_factors.formulae.special_functions import Li2

#-------------------------------------------------------#
#Approximate master functions for dipole form factors
#-------------------------------------------------------#

def f_plus_approx(i, j, k):
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
    f_val = 2*xi - 3 - (xi - 3)*xi*np.log(xi)
    f_val+= 2*(xi-1)*np.sqrt(xi*(xi-4))*np.log((xi + np.sqrt(xi*(xi-4)))/(2*np.sqrt(xi)))
    f_val-= np.log((xi + np.sqrt(xi*(xi-4)))/(2*xi))**2
    f_val-= 2*Li2(1-xi)
    f_val-= 2*Li2((xi-np.sqrt(xi*(xi-4)))/(2*xi))
    f_val+= 2*Li2((2-xi-np.sqrt(xi*(xi-4)))/2)
    return f_val

def f_minus_iji(xi, xj, xk):
    f_val = -2 + (xi - 3)*xi/(xi-1)*np.log(xi)
    f_val-= 2*np.sqrt(xi*(xi-4))*np.log((xi + np.sqrt(xi*(xi-4)))/(2*np.sqrt(xi)))
    f_val-= np.log((xi + np.sqrt(xi*(xi-4)))/(2*xi))**2
    f_val-= 2*Li2(1-xi)
    f_val-= 2*Li2((xi-np.sqrt(xi*(xi-4)))/(2*xi))
    f_val+= 2*Li2((2-xi-np.sqrt(xi*(xi-4)))/2)
    return f_val

#----------------------------------#
#Approximations for i >> j, k
#----------------------------------#

def f_plus_small_k(xi, xj, xk):
    return np.conj((-1 + 2*xi + 2*(xi-1)*xi*np.log((xi-1)/xi)))

def f_minus_small_k(xi, xj, xk):
    f_val = 1 + (np.log((xi-1)*xk/xi) + xi - 1)*np.log((xi-1)/xi) + Li2(1/xi)
    f_val*= -2*np.sqrt(xi)/np.sqrt(xk)
    #f_val += np.sqrt(xi)/np.sqrt(xj * xk) * (1 + (xi - 1)*np.log((xi-1)/xi))
    
    return np.conj(f_val)


def f_plus_large_k(xi, xj, xk):
    f_val = (2*xk**2 + 5*xk - 1)/(6*(xk-1)**3) - xk**2/(xk-1)**4 * np.log(xk)
    return xk*(1/np.sqrt(xi)+1/np.sqrt(xj))**2 * f_val

def f_minus_large_k(xi, xj, xk):
    return np.sqrt(xk)*(1/np.sqrt(xi) + 1/np.sqrt(xj))*(-(3*xk-1)/(xk-1)**2 + 2*xk**2/(xk-1)**3 * np.log(xk))
       
#----------------------------------#
#Approximations for m >> mi, mj, mk       
#----------------------------------#

def f_large_m(ui, uj, uk):
    return -2*uk*(ui+uj+uk)*(1+2*np.log(uk))+((ui+uj)**2 - 3*(ui+uj)*uk+6*uk**2+12*uk**2*np.log(uk))

def f_minus_large_m(ui, uj, uk):
    return -(ui+uj)*uk*(3 + 4*np.log(uk))

def f_plus_large_m(ui, uj, uk):
    return (ui+uj)**2/3

#----------------------------------#
#Wrappers for applying a different approximation at large m 
#----------------------------------#
def plus_large_m(f_plus_approx):
    def approx(xi, xj, xk):
        #x_max = np.maximum(xi, xk)
        ui, uj, uk = 1/np.sqrt(xi), 1/np.sqrt(xj), 1/np.sqrt(xk)
        return np.where(xi < 1e6,
                        f_plus_approx(xi, xj, xk),
                        f_plus_large_m(ui, uj, uk))
    
    return approx
    
def minus_large_m(f_minus_approx):
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
    
    exact = 2 - 4*xi + 2*xi*(xi - 2)*np.log(xi)
    exact-= 4*(xi**2 - 4*xi + 2)*np.sqrt(xi/(xi-4))*np.log((np.sqrt(xi) + np.sqrt(xi-4))/2)

    return np.where(xi > 1e6,
                    (25 + 4*xi - 12*np.log(xi))/(3*xi**2), #for large xi, to avoid floating point errors
                    exact)
def g_minus_ii(xi, xj):
    
    exact = 4 - 2*xi*np.log(xi)
    exact+= 4*(xi - 2)*np.sqrt(xi/(xi-4))*np.log((np.sqrt(xi) + np.sqrt(xi-4))/2)
    
    return np.where(xi > 1e6,
                    -2*(32 + 9*xi - 6*(4 + xi)*np.log(xi))/(3*xi**2), #for large xi, to avoid floating point errors
                    exact)

#----------------------------------#
#Approximation for j >> i
#----------------------------------#
def g_plus_large_j(xi, xj):
    return 2*xj/xi * ((2*xj**2 + 5*xj - 1)/(3*(xj-1)**3) - (2*xj**2/(xj-1)**4)*np.log(xj))

def g_minus_large_j(xi, xj):
    return 2*np.sqrt(xj/xi) * ((1 - 3*xj)/(xj - 1)**2 + (2*xj**2)/(xj-1)**3 * np.log(xj))

#----------------------------------#
#Approximation for j << i
#----------------------------------#
def g_plus_small_j(xi, xj):
    return -2-4*xi + 4*xi**2 * np.conj(np.log(xi/(xi - 1)))

def g_minus_small_j(xi, xj):
    return 4*np.sqrt(xi/xj) * (1 -  (xi**2+1)/(xi-1)*np.conj(np.log(xi/(xi-1))) + 1/(xi-1)*np.log(xj))            
