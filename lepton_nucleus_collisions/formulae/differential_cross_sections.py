#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from phys.constants import ml, alpha
from lepton_nucleus_collisions.utils.process import lepton_idx
#from lepton_nucleus_collisions.utils import chunk

def calc_dcrossx(initial_state_data, final_state_data, Ek, th_k):
    #Process input parameters
    initial_lepton, E_lepton, Z, A, M, form_factor_squared, E_nuc, v_nuc, E = initial_state_data    
    mi =  ml[lepton_idx(initial_lepton)]
    
    final_lepton, particle_type, t_cut_off, method, m = final_state_data    
    mf = ml[lepton_idx(final_lepton)]

    #t in ion rest frame
    t_min = ((m + mf)**2 - mi**2)**2/(4*E**2)
    t_max = min(t_cut_off, (2*E*M/(E+M))**2) #Either (2*E*M/(E+M))**2 or cut-off from form factor
    t = np.logspace(np.log10(t_min), np.log10(t_max), 400)
    
    #Reshape for efficient broadcasting
    EK = Ek.reshape(-1, 1, 1)
    TH = th_k.reshape(1, -1, 1)
    T = t.reshape(1, 1, -1)
    
    #Calculate once
    t_indep_params = t_independent_params(M, mf, mi, E, m, EK, TH)
    
    #Compute the exact cross-section
    if method == 'exact':
        #t_chunks = chunk(t, 50)
        
        dcx = np.zeros((len(Ek), len(th_k)))
        dcx_PV = np.zeros((len(Ek), len(th_k)))
        
        #Compute d\sigma/(dE d\theta dt)
        params = all_params(t_indep_params, T)
        dcx, dcx_PV = dcrossx[particle_type](params, method)
        ft2 = form_factor_squared(T)/T**2
        dcx = ft2*dcx
        dcx_PV = ft2*dcx_PV
        
        #Integrate over t
        dcx = np.trapz(dcx, x = t)
        dcx_PV = np.trapz(dcx_PV, x = t)

    #Compute cross-section with (Improved) Weizsacker-Williams approximation
    else:
        #Extract relevant parameters
        V, u = t_indep_params[-2], t_indep_params[-1]

        #Compute exact solution to cos^2(th_q^0) = 1
        rad = (E - EK + M)*u + 2*M*V**2
        rad+= V*np.sqrt(4*(M*V)**2 + u*(4*M*(E-EK+M) + u))
        
        T_1 = M*rad/((E-EK+M)**2 - V**2)
        T_2 = M*u**2/rad
        
        T_min = np.nan_to_num(np.minimum(T_1, T_2)) #exact T_min, not approximate
        T_max = np.nan_to_num(np.maximum(T_1, T_2)) #exact T_max, not approximate
        
        T_min = T_min * (T_min > 0)
        T_max = T_max * (T_max > 0)
        
        if method == 'WW':
            #Compute CHI for WW
            region = (T >= T_min)*(T <= T_max)
            CHI = np.trapz((T - T_min)*form_factor_squared(T)/T**2 * region, x = t) 
        else:
            #Compute CHI for IWW
            region = (t > t_min)*(t <= t_max)*(t_max > t_min)
            CHI = np.trapz((t - t_min)*form_factor_squared(t)/t**2 * region, x = t)
            
        params = all_params(t_indep_params, T_min)
        dcx, dcx_PV = dcrossx[particle_type](params, method)
        dcx = CHI * np.where(T_min == 0, 0, dcx/(2 * T_min)).squeeze(-1)
        dcx_PV = CHI * np.where(T_min == 0, 0, dcx_PV/(2 * T_min)).squeeze(-1)
            
    return dcx, dcx_PV

def t_independent_params(M, mf, mi, E, m, Ek, th_k):
    
    p = np.sqrt(E**2 - mi**2)
    k = np.sqrt(Ek**2 - m**2)
    cos_th_k = np.cos(th_k)
    sin_th_k = np.sin(th_k)

    V = np.sqrt((p - k)**2 + 4*p*k*np.sin(th_k/2)**2)
    
    #u = (E - Ek)**2 - V**2 - mf**2
    u = ((E-p)  - (Ek-k))*((E+p) - (Ek+k)) - 4*p*k*np.sin(th_k/2)**2 - mf**2
    
    return M, mf, mi, E, p, m, Ek, k, th_k, cos_th_k, sin_th_k, V, u

def all_params(t_indep_params, t):
    
    M, mf, mi, E, p, m, Ek, k, th_k, cos_th_k, sin_th_k, V, u = t_indep_params
    
    q0 = -t/(2*M)
    Q = np.sqrt(t + (t/(2*M))**2)
        
    cos_th_q = (u - (1 + (E-Ek)/M)*t)/(2*Q*V)
    sin_th_q = np.sqrt(1 - cos_th_q**2)
    
    return M, mf, mi, E, p, m, Ek, k, th_k, cos_th_k, sin_th_k, V, u, q0, Q, cos_th_q, sin_th_q, t

def scalar(params, method):

    M, mf, mi, E, p, m, Ek, k, th_k, cos_th_k, sin_th_k, V, u, q0, Q, cos_th_q, sin_th_q, t = params
    
    if method == 'WW' or method == 'IWW':
        cos_th_q = np.sign(cos_th_q)
        sin_th_q = 0
       
    kinematically_allowed = (Ek > m) * (E - Ek + q0 > mf) * (cos_th_q**2 <= 1)

    #dot products
    qdp0 = cos_th_q*(Q/V)*p*(p-k*cos_th_k)
    qdp1 = sin_th_q*(Q/V)*p*k*sin_th_k
    
    s0 = -(1 + E/M)*t - 2*qdp0
    s1 = -2*qdp1
    
    if method == 'WW' or method == 'IWW':
        PC = -(s0 + u)**2/(s0*u)
        
        term = ((s0+u)*(1 + mi**2/s0 + mf**2/u)-m**2)/(s0*u)
        PC+= 2*(m**2 - (mi + mf)**2)*term
        PV = 8*mf*mi*term
        
        coeff = alpha**2/(4*np.pi)*k*np.sin(th_k)/(p*V)
        PC = np.where(kinematically_allowed, coeff*PC, 0)
        PV = np.where(kinematically_allowed, coeff*PV, 0)
        
        return PC, PV

    I1 = np.where(s0**2 <= s1**2, 0, np.sign(s0)/np.sqrt(s0**2 - s1**2))
    I2 = s0*I1**3

    
    T1 = s0/u + 2 + u*I1
    T2 = - (2*M)*Ek/u + (2*M)*Ek*((2*M)*Ek-u)/u * I1  + T1/4
    T3 = 1/u**2 + 2/u * I1 + I2

    P1_s1 = (2*M*(E-Ek) - 0.5*t)
    P0_P1_s01 = (2*M*E - t/2)*u
    
    T4 = (P1_s1**2 + (P0_P1_s01)*(2*P1_s1*I1 + (P0_P1_s01)*I2))/u**2
    
    P2 = 4*M**2 + t

    PV = P2 * t * T3 - 4*T4
    
    PC = P2*T1 - 4*t*T2 + (m**2 - (mf + mi)**2)*PV
    PV = (4*mi*mf)*PV

    coeff = (alpha/M)**2/(32*np.pi)*k*np.sin(th_k)/(p*V)
    PC = np.where(kinematically_allowed, coeff*PC, 0)
    PV = np.where(kinematically_allowed, coeff*PV, 0)
    
    return PC, PV


def vector(params, method):

    M, mf, mi, E, p, m, Ek, k, th_k, cos_th_k, sin_th_k, V, u, q0, Q, cos_th_q, sin_th_q, t = params
    
    if method == 'WW' or method == 'IWW':
        cos_th_q = np.sign(cos_th_q)
        sin_th_q = 0
   
    kinematically_allowed = (Ek > m) * (E - Ek + q0 > mf) * (cos_th_q**2 <= 1)
    
    #dot products
    qdp0 = cos_th_q*(Q/V)*p*(p-k*cos_th_k)
    qdp1 = sin_th_q*(Q/V)*p*k*sin_th_k

    s0 = -(1 + E/M)*t - 2*qdp0
    s1 = -2*qdp1
    
    dm2 = (mi - mf)**2
    
    if method == 'WW' or method == 'IWW':
        T1 = (s0 + u)**2/(s0*u)
        
        PC = 4 - T1*(2 + dm2/m**2)    
        PC+= 2*(1-dm2/m**2)*(2*m**2 + (mi + mf)**2)/(s0*u) * ((s0+u)*(1 + mi**2/s0 + mf**2/u)-m**2)
        
        PV = -4*mf*mi/m**2 * T1
        PV+= -24*mf*mi/(s0*u) * ((s0+u)*(1 + mi**2/s0 + mf**2/u)-m**2)
        
        coeff = alpha**2/(4*np.pi)*k*np.sin(th_k)/(p*V)
        PC = np.where(kinematically_allowed, coeff*PC, 0)
        PV = np.where(kinematically_allowed, coeff*PV, 0)
        
        return PC, PV

    I1 = np.where(s0**2 <= s1**2, 0, np.sign(s0)/np.sqrt(s0**2 - s1**2))
    I2 = s0*I1**3
    
    T1 = s0/u + 2 + u*I1
    T2 = - (2*M)*Ek/u + (2*M)*Ek*((2*M)*Ek-u)/u * I1  + T1/4
    T3 = 1/u**2 + 2/u * I1 + I2
    

    P1_s1 = (2*M*(E-Ek) - 0.5*t) #P1/s1
    P0_P1_s01 = (2*M*E - t/2)*u #
    T4 = (P1_s1**2 + (P0_P1_s01)*(2*P1_s1*I1 + (P0_P1_s01)*I2))/u**2

    #T5 + T6 together
    T56 = -(((4*M)*(E-Ek) + u) - t)*(1 + (t - (4*M)*E)*I1)/(2*u) + T2
    T7 = (-1 + (2*(m**2-dm2) - t - u)*I1)/(2*u)
     
    P2 = 4*M**2 + t

    PC = ((2 + dm2/m**2)*T1 - 4)*P2 - (4*dm2/m**2)*t*T2
    PC+= -8*t*(T56+ T7*P2)
    PC+= (1-dm2/m**2)*(2*m**2 + (mi + mf)**2)*(P2*t*T3 - 4*T4)
    
    PV = T1*P2 - 4*t*T2 + (8*m**2) * P2 * (t/u) * I1
    PV+= -3*m**2 * (P2*t*T3 - 4*T4)
    PV*= 4*mi*mf/m**2
    
    coeff = (alpha/M)**2/(32*np.pi)*k*np.sin(th_k)/(p*V)
    
    PC = np.where(kinematically_allowed, coeff*PC, 0)
    PV = np.where(kinematically_allowed, coeff*PV, 0)

    return PC, PV

dcrossx = {"scalar": scalar, "vector": vector}
