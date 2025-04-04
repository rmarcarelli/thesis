#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from phys.constants import alpha, ml, mH, vH


#-------------------------------------------------------------------------#
# Scalar Decays
#-------------------------------------------------------------------------#

#specialized to leptons by default
def scalar_decay_rate(m_phi, mf = ml, g = [[1]*3]*3):
    
    rate = 0
    for i, gi in enumerate(g):
        for j, gij in enumerate(gi):
            rate += scalar_fermion_decay_rate(m_phi, mf[i], mf[j], gij)

    return rate


def scalar_fermion_decay_rate(m_phi, mi, mj, gij = 1):
    
    rate = gij**2/(8*np.pi*m_phi**3)
    rate*= (m_phi**2 - (mi-mj)**2)**(3/2)
    rate*= (m_phi**2 - (mi+mj)**2)**(1/2)
    
    return np.where(m_phi < mi+mj, 0, rate)

def scalar_fermion_branching_fraction(m_phi, i, j, mf = ml, g = [[1]*3]*3):
    
    return scalar_fermion_decay_rate(m_phi, i, j, mf = mf, g = g)/scalar_decay_rate(m_phi, mf = mf, g = g)

#-------------------------------------------------------------------------#
# ALP Decays
#-------------------------------------------------------------------------#

#specialized to leptons by default
def ALP_decay_rate(ma, mf = ml, Cff = [[1]*3]*3, Cgg = 0, Lam = 1000):
    
    rate = 0
    for i, Ci in enumerate(Cff):
        for j, Cij in enumerate(Ci):
            rate += ALP_fermion_decay_rate(ma, mf[i], mf[j], Cij, Lam)
            
    rate += ALP_photon_decay_rate(ma, mf = ml, Cff = Cff, Cgg = Cgg, Lam = Lam)
    
    return rate

def B1(t):
    
    #x>=1
    t_gt_1 = np.arcsin(1/np.sqrt(t))
    #x<1
    t_lt_1 = np.pi/2 + 1j/2*np.log((1+np.sqrt(1-t))/(1-np.sqrt(1-t)))
    
    ft = np.where(t >= 1, t_gt_1, t_lt_1)

    return 1 - t*ft**2

def ALP_photon_decay_rate(ma, mf = ml, Cff =  [[1]*3]*3, Cgg = 0, Lam = 1000):
    
    Cgg_eff = Cgg
    for i in range(len(mf)):
        Cgg_eff += Cff[i][i]/(8*np.pi**2) * B1(4*mf[i]**2/ma**2)

    return 4*np.pi * alpha**2 * ma**3 /Lam**2 * np.abs(Cgg_eff)**2

def ALP_fermion_decay_rate(ma, mi, mj, Cij = 1, Lam = 1000):
    
    rate = Cij**2/(8*np.pi*Lam**2) * (mi + mj)**2/ma**3
    rate*= (ma**2 - (mi-mj)**2)**(3/2)
    rate*= (ma**2 - (mi+mj)**2)**(1/2)
    
    return np.where(ma < mi+mj, 0, rate)


#-------------------------------------------------------------------------#
# Higgs Decays to ALPs
#-------------------------------------------------------------------------#

def Higgs_ALP_decay_rate(ma, Cah, Cahp = 0, Lam = 1000):
    
    Cah_bar = Cah - (2*ma**2)/(mH**2 - 2*ma**2) * Cahp
    
    return vH**2*mH**3*np.sqrt(1 - 4*ma**2/mH**2)*(1-2*ma**2/mH**2)**2*Cah_bar**2/(32*np.pi*Lam**4)
        

#-------------------------------------------------------------------------#
# Conversion from ALP couplings to scalar couplings
#-------------------------------------------------------------------------#

def ALP_to_scalar(C, TH, D, PH, Lam = 1000, mf = ml):

    g = np.empty(np.shape(C))
    th = np.empty(np.shape(TH))
    d = np.empty(np.shape(D))
    ph = np.empty(np.shape(PH))
    
    I = len(mf)
    J = len(mf)
    
    for i in range(I):
        for j in range(J):
            g[i,j] = C[i][j] * np.sqrt(mf[i]**2 + mf[j]**2 + 2*mf[i]*mf[j]*np.cos(2*TH[i][j]))/Lam
            th[i,j] = np.arctan((mf[i]+mf[j])/(mf[i]-mf[j]) / np.tan(TH[i][j]))
            d[i,j] = D[i][j] - np.pi/2
            ph[i,j] = PH[i][j] - np.pi/2
    return g, th, d, ph

#-------------------------------------------------------------------------#
# Conversion from scalar couplings to ALP couplings
#-------------------------------------------------------------------------#

def scalar_to_ALP(g, th, d, ph, Lam = 1000, mf = ml):

    C = np.empty(np.shape(g))
    TH = np.empty(np.shape(th))
    D = np.empty(np.shape(d))
    PH = np.empty(np.shape(ph))
    
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