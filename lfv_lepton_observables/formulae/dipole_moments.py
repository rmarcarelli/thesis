#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from lfv_lepton_observables.formulae.form_factor_functions_approx import gp, gm

from phys.constants import alpha, ml, hc_cmGeV
from phys.formulae.ALP_EFT import ALP_to_scalar

#anomalies (+/- 2 sigma)
da_2sig_e_Rb = ((34-2*16)*1e-14, (34+2*16)*1e-14)
da_2sig_e_Cs = ((-101-2*27)*1e-14, (-101+2*27)*1e-14)
da_2sig_mu = ((249-2*48)*1e-11, (249+2*48)*1e-11) 

#g-2 standard error (if there isn't an anomaly)
da_sig_exp = [13e-14,#take the average of the errors from Rb and Cs
              40e-11,
              3.2e-3]

#dipole moment limits
d_l = [4.1e-30,
       1.8e-19,
       1.85e-17]

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

    f2 = 0
    if ALP:
        f2 += F2_ALP_photon(m, i, g, Lam  = Lam)
        g, th, d, ph = ALP_to_scalar(g, th, d, ph)
    
    for j in range(3):
        f2 += F2_lepton(m, i, j, g, th, d)
        
    return f2

def F3(m, i, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, ALP = False, Lam = 1000):
    
    if ALP:
        g, th, d, ph = ALP_to_scalar(g, th, d, ph)
    
    f3 = 0
    for j in range(3):
        f3 += F3_lepton(m, i, j, g, th, d)
        
    return f3

def F2_lepton(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3):

    ff = gp(m, i, j) + gm(m, i, j)*np.cos(2*th[i][j])
    ff*= g[i][j]**2/(32*np.pi**2) 
    return ff

def F3_lepton(m, i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3):

    ff = gm(m, i, j)*np.sin(2*th[i][j])*np.cos(d[i][j])
    ff*= -g[i][j]**2/(32*np.pi**2)
    return ff

#-------------------------------------------------------------------------#
# li on-diagonal dipole form factors with internal gamma and li (for ALP only)
#-------------------------------------------------------------------------#

def h_gamma(x):
    exact = 1 + (x**2/6)*np.log(x) - x/3
    exact-= (x+2)/3 * np.sqrt((x-4)*x)*np.log((np.sqrt(x)+np.sqrt(x-4))/2)
    large = (3 + 2*np.log(x))/2 + 4*(-2 + 3*np.log(x))/(9*x) #treat large x separately to avoid floating point errors
    return np.where(x < 1e6, exact, large)

def F2_ALP_photon(m, i, C = [[1]*3]*3, Lam = 1000):
    Cii = C[i][i]
    Cgg = (C[i][i])/(8*np.pi**2) # only contribution from Cii
    
    xi = (m/ml[i])**2

    I = 2*np.log(Lam/ml[i])- h_gamma(xi)
    
    return -64*np.pi*alpha/(16*np.pi**2) * Cgg * Cii * (ml[i]/Lam)**2  * I