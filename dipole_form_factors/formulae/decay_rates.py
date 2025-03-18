#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from phys.constants import alpha, ml
from phys.formulae.ALP_EFT import ALP_to_scalar

decay_rates = np.array([np.inf, 2.99e-19, 2.27e-12]) #from PDG

branching_limits = {(1, 0): 4.2e-13, #from MEG ()
                    (2, 0): 3.3e-8,  #from BaBar ()
                    (2, 1): 4.4e-8,  #from BaBar ()
                    }


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

def rate_ij(f2p, f2m, f3p, f3m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):

    if mode == 'PC':
        th = [[0]*3]*3
        d = [[0]*3]*3
        ph = [[0]*3]*3
        
    if mode == 'chiral':
        th = [[np.pi/4]*3]*3
        d = [[0]*3]*3 if ALP else [[np.pi/2]*3]*3
        ph = [[0]*3]*3

    if ALP:
        C = g
        TH = th
        D = d
        PH = ph
        
        g, th, d, ph = ALP_to_scalar(C, TH, D, PH, Lam)
    
    
    f2 = 0
    f3 = 0
    for k in range(3):
        f2p_ij = f2p[(i, j, k)]
        f2m_ij = f2m[(i, j, k)]
        f3p_ij = f3p[(i, j, k)]
        f3m_ij = f3m[(i, j, k)]

        f2 += F2(f2p_ij, f2m_ij, i, j, k, g, th, d, ph)
        f3 += F3(f3p_ij, f3m_ij, i, j, k, g, th, d, ph)
        
    if ALP:
        m = f2p[(i, j, 0)] #m is the same for each
        f2 += F2_ALP_photon(m, i, j, C, TH, D, PH, Lam)
        f3 += F3_ALP_photon(m, i, j, C, TH, D, PH, Lam)
        
    return (alpha/2) * ((ml[i]-ml[j])**2 * np.abs(f2)**2 + (ml[i]+ml[j])**2*np.abs(f3)**2)/ml[i]
 


#-------------------------------------------------------------------------#
# Contribution to li -lj gamma decay rate from a single internal lepton k
# (ignoring interference)
#-------------------------------------------------------------------------#

def rate_ijk(f2p, f2m, f3p, f3m, i, j, k, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3,  mode = None, ALP = False, Lam = 1000):
    if mode == 'PC':
        th = [[0]*3]*3
        ph = [[0]*3]*3
        d = [[0]*3]*3
        
    if mode == 'chiral':
        th = [[np.pi/4]*3]*3
        d = [[0]*3]*3 if ALP else [[np.pi/2]*3]*3
        ph = [[0]*3]*3

    if ALP:
        C = g
        TH = th
        PH = ph
        D = d
        
        g, th, d, ph = ALP_to_scalar(C, TH, D, PH)
    
    f2 = F2(f2p[(i, j, k)], f2m[(i, j, k)], i, j, k, g, th, d, ph)
    f3 = F3(f3p[(i, j, k)], f3m[(i, j, k)], i, j, k, g, th, d, ph)
        
    return (alpha/2) * ((ml[i]-ml[j])**2 * np.abs(f2)**2 + (ml[i]+ml[j])**2*np.abs(f3)**2)/ml[i]



#---------------------------------------------------------------------------#
# Photon contribution to li -> lj gamma decay rate in the case of an LFV ALP
# (ignoring interference)
#---------------------------------------------------------------------------#

def rate_ALP_photon(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, Lam = 1000):
    C = g
    TH = th
    D = d
    PH = ph
    
    f2 = F2_ALP_photon(m, i, j, C, TH, D, PH, Lam)
    f3 = F3_ALP_photon(m, i, j, C, TH, D, PH, Lam)
    
    return (alpha/2) * ((ml[i]-ml[j])**2 * np.abs(f2)**2 + (ml[i]+ml[j])**2*np.abs(f3)**2)/ml[i]
       
 

#-------------------------------------------------------------------------#
# li -> lj form factors with internal lk
#-------------------------------------------------------------------------#

def F2(f2p_ijk, f2m_ijk, i, j, k, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3):
    args = f2p_ijk, f2m_ijk, g[i][k], g[j][k], th[i][k], th[j][k], d[i][k], d[j][k], ph[i][k], ph[j][k]
    return F2_ijk(*args)

def F3(f3p_ijk, f3m_ijk, i, j, k, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3):
    args = f3p_ijk, f3m_ijk, g[i][k], g[j][k], th[i][k], th[j][k], d[i][k], d[j][k], ph[i][k], ph[j][k]
    return F3_ijk(*args)


def F2_ijk(f2p_ijk, f2m_ijk, g_ik = 1, g_jk = 1, th_ik = 0, th_jk = 0, d_ik = 0, d_jk = 0, ph_ik = 0, ph_jk = 0):
    
    d_diff = d_jk-d_ik
    ph_diff = ph_jk - ph_ik
    
    ff = np.cos(th_ik)*np.cos(th_jk)*(f2p_ijk+f2m_ijk)
    ff+= np.exp(1j*d_diff)*np.sin(th_ik)*np.sin(th_jk)*(f2p_ijk-f2m_ijk)
    ff = np.exp(1j*ph_diff)*g_ik*g_jk*ff/(32*np.pi**2)
    return ff

def F3_ijk(f3p_ijk, f3m_ijk,  g_ik = 1, g_jk = 1, th_ik = 0, th_jk = 0, d_ik = 0, d_jk = 0, ph_ik = 0, ph_jk = 0):
    
    ph_diff = ph_jk - ph_ik

    ff = np.exp(-1j*d_jk)*np.cos(th_ik)*np.sin(th_jk)*(f3p_ijk+f3m_ijk)
    ff+= -np.exp(1j*d_ik)*np.sin(th_ik)*np.cos(th_jk)*(f3p_ijk-f3m_ijk)
    ff = np.exp(1j*ph_diff)*g_ik*g_jk*ff/(32*np.pi**2)
    return ff


#-------------------------------------------------------------------------#
# li -> lj gamma form factors through internal gamma and lj (for ALP only)
#-------------------------------------------------------------------------#

def g2(x):
    x = np.complex128(x)
    return -np.log(x)/(x-1)-(x-1)*np.log(x/(x-1))-2
 
def F2_ALP_photon(m, i, j, C, TH, D, PH, Lam = 1000):
    Cij = C[i][j]
    THij = TH[i][j]
    Dij = D[i][j]
    PHij = PH[i][j]
    Aij = Cij*np.cos(THij)*np.exp(1j*(PHij + Dij))
    Cgg = (C[0][0]+C[1][1]+C[2][2])/(8*np.pi**2)
    
    xi = (m/ml[i])**2

    I = 2*np.log(Lam**2/m**2) + g2(xi)

    return -alpha/(2*np.pi) * Cgg * Aij * (ml[i]/Lam)**2  * I

 
def F3_ALP_photon(m, i, j, C, TH, D, PH, Lam = 1000):
    Cij = C[i][j]
    THij = TH[i][j]
    Dij = D[i][j]    
    PHij = PH[i][j]
    Vij = Cij*np.sin(THij)*np.exp(1j*(PHij))
    Cgg = (C[0][0]+C[1][1]+C[2][2])/(8*np.pi**2)

    xi = (m/ml[i])**2

    I = 2*np.log(Lam**2/m**2) + g2(xi)
        
    return -alpha/(2*np.pi) * Cgg * Vij * (ml[i]/Lam)**2  * I
