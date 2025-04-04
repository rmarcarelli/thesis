#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache

from lepton_nucleus_collisions.utils.process import (process_run_card,
                                                     read_dcrossx,
                                                     open_dcrossx,
                                                     lepton_idx)

from phys.constants import hc2_fbGeV2, ml

@lru_cache(maxsize = 10000)
def crossx(experiment, params, units = 'pb', frame = 'lab', X_var = 'GAMMA', Y_var = 'ETA', X_min = None, X_max = None, Y_min = None, Y_max = None, final_state_particle = 'phi'):

    masses = process_run_card(f'{experiment}.txt', 'masses')
    
    #lepton, particle_type, t_min, method, PV, (mass, optional)
    
    #supports params including and not including mass...
    if len(params) == 5:
        cx = np.array([crossx(experiment, params + (m,), units, frame, X_var, Y_var, X_min, X_max, Y_min, Y_max, final_state_particle) for m in masses])
        return np.array(cx)
    
    #supports mass *not* in masses through linear interpolation:
    mass = params[-1]
    if mass not in masses:
        return np.interp(mass, masses, crossx(experiment, params[:-1], units))
    
    if not (X_min or X_max or Y_min or Y_max):
        file_name = f'lepton_nucleus_collisions/data/{experiment}.h5'
        cx = open_dcrossx(file_name, params, 'SIG')
        
    else:

        if not (Y_min or Y_max):
            x, ds_dx = dcrossx(experiment, params, frame = frame, which = 'X', X_var = X_var, Y_var = Y_var, final_state_particle = final_state_particle)
    
            constraint = 1
            if X_min:
                constraint = constraint*(x > X_min)
            if X_max:
                constraint = constraint*(x < X_max)
    
            cx = np.trapz(ds_dx * constraint, x = x)
        
        elif not (X_min or X_max):
            y, ds_dy = dcrossx(experiment, params, frame = frame, which = 'Y', X_var = X_var, Y_var = Y_var, final_state_particle = final_state_particle)
    
            constraint = 1
            if Y_min:
                constraint = constraint*(y > Y_min)
            if Y_max:
                constraint = constraint*(y < Y_max)
    
            cx = np.trapz(ds_dy * constraint, x = y)
            
        else:
            X_min = X_min if X_min else -np.inf
            X_max = X_max if X_max else np.inf
            Y_min = Y_min if Y_min else -np.inf
            Y_max = Y_max if Y_max else np.inf
            
            x, y, ds_dx_dy = dcrossx(experiment, params, frame = frame, which = 'XY', X_var = X_var, Y_var = Y_var, final_state_particle = final_state_particle)
            
            X = x.reshape(-1, 1)
            Y = y.reshape(1, -1)
            
            cx = np.trapz(np.trapz(ds_dx_dy*constraint*(X > X_min)*(X < X_max)*(Y > Y_min)*(Y < Y_max), x = x), y = y)
                
    if units == 'fb':
        return hc2_fbGeV2*cx
    if units == 'pb':
        return (hc2_fbGeV2/1000)*cx
    
    return cx

@lru_cache(maxsize = 10000)
def distribution(experiment, params, frame = 'lab', which = 'X', X_var = 'GAMMA', Y_var = 'ETA', X_pts = 1000, Y_pts = 1000, final_state_particle = 'phi'):

    dcx = dcrossx(experiment, params, frame = frame, which = which, X_var = X_var, Y_var = Y_var, X_pts = X_pts, Y_pts = Y_pts, final_state_particle = final_state_particle)
    cx = crossx(experiment, params, units = 'GeV')
    
    return dcx[0], dcx[1]/cx
    

@lru_cache(maxsize = 10000)
def dcrossx(experiment, params, frame = 'lab', which = 'X', X_var = 'GAMMA', Y_var = 'ETA', X_pts = 1000, Y_pts = 1000, final_state_particle = 'phi'):

    assert frame in ['lab', 'ion']
    assert which in ['X', 'Y', 'XY']
    assert X_var in ['X', 'E', 'GAMMA'] and Y_var in ['COS_THETA', 'THETA', 'ETA']

    v_nuc, E = process_run_card(f'{experiment}.txt', ['v_nuc', 'E'])
    lf, m = params[0], params[-1]
    mlf = ml[lepton_idx(lf)]

    f = np.sqrt((1 + v_nuc)/(1 - v_nuc)) if frame == 'lab' else 1 # Doppler factor
    E_frame = E/f

    m_final = m if final_state_particle == 'phi' else mlf

    #GAMMA by default
    x_min = 1.0+1e-3
    x_max = (max(1, E_frame/m_final) + m_final/(4*E_frame))*1.2 #for good measure
    
    x = np.geomspace(x_min, x_max, X_pts)
    
    if X_var == 'X':
        x *= m_final/E_frame
    
    if X_var == 'E':
        x *= m_final
     
    #ETA by default
    y_min = -30
    y_max = 30
    y = np.linspace(y_min, y_max, Y_pts)
    
    if Y_var == 'THETA':
        y = 2*np.arctan(np.exp(-y))
    
    if Y_var == 'COS_THETA':
        y = 2*np.arctan(np.exp(-y))
        y = np.cos(y)
        #for cos_theta, want to sample more 

    
    dcx = differential_cross_sections(experiment, params, x, y,
                                      frame = frame,
                                      which = which,
                                      XY_vars = (X_var, Y_var),
                                      final_state_particle = final_state_particle)
    
    
    
    if which == 'X':
        return x, dcx
    elif which == 'Y':
        return y, dcx
    else:
        return x, y, dcx

def differential_cross_sections(experiment, params, x, y, frame = 'lab', which = 'XY', XY_vars = ('E', 'THETA'), final_state_particle = 'phi'):
    
    assert frame in ['lab', 'ion']
    assert which in ['X', 'Y', 'XY']
    assert final_state_particle in ['phi', 'lepton']
    
    file_name = f'lepton_nucleus_collisions/data/{experiment}.h5'
    

    X,Y = np.meshgrid(x, y, indexing = 'ij')
    
    Ek, THk, J = transform(X, Y, experiment, params, XY_vars, frame = frame, final_state_particle = final_state_particle)

    dcrossx_data = open_dcrossx(file_name, params)
    dcrossx_dX_dY = J * dcrossx_interp(Ek, THk, dcrossx_data)


    if which == 'XY':
        return dcrossx_dX_dY
    if which == 'X':
        dcrossx_dX = np.trapz(dcrossx_dX_dY, x = y, axis = 1)
        return dcrossx_dX
    if which == 'Y':
        dcrossx_dY = np.trapz(dcrossx_dX_dY, x = x, axis = 0)
        return dcrossx_dY
    

def transform(x, y, experiment, params, XY_vars, frame = 'ion', final_state_particle = 'phi'):
    
    X_var, Y_var = XY_vars
    
    li, Ei, M, v_ion = process_run_card(f'{experiment}.txt', ['li', 'E', 'M', 'v_nuc'])
    lf, m_phi= params[0], params[-1]
    mi = ml[lepton_idx(li)]
    mf = ml[lepton_idx(lf)]
    
    m_final = mf if final_state_particle == 'lepton' else m_phi
    
    if X_var == 'GAMMA':
        E, J_E = m_final * x, m_final
    
    if X_var == 'X':
        E, J_E = x*Ei, Ei
    
    if X_var == 'E':
        E, J_E = x, 1
    

    if Y_var == 'THETA':
        TH, J_TH = y, 1
    
    if Y_var == 'ETA':
        TH, J_TH = 2*np.arctan(np.exp(-y)), -1/np.cosh(y)
    
    if Y_var == 'COS_THETA':
        TH, J_TH = np.arccos(y), -1/np.sqrt(1-y**2)

    J = J_E * J_TH
    if frame == 'lab':
    
        E, TH = E_THETA_lab_transform(m_final, v_ion, E, TH)
        
    if final_state_particle == 'lepton':

        t_min = ((m_phi+mf)**2 - mi**2)**2 / (4*Ei**2)
        q0 = -t_min/(4*M)
        Q = np.sqrt(t_min+q0**2)
        E, TH, J_phi = PHI_LEPTON_transform(m_phi, mf, q0, Q, Ei, E, TH)
        
        J = J * J_phi
    
    return E, TH, np.abs(J)


def E_THETA_lab_transform(m, v_ion, E_lab, THETA_lab):
    g_ion = 1/np.sqrt(1 - v_ion**2)
    gk_lab = E_lab/m
    vk_lab = np.sqrt(1 - 1/gk_lab**2)
    
    sin = np.sin(THETA_lab)
    cos = np.cos(THETA_lab)
    sin_half = np.sin(THETA_lab/2)
    
    THETA = np.arctan(sin/(g_ion *((1+v_ion/vk_lab) - 2*sin_half**2)))
    THETA = np.where(THETA < 0, np.pi + THETA, THETA)
    
    E = g_ion*E_lab*(1 + v_ion * vk_lab * cos)

    return E, THETA

def PHI_LEPTON_transform(m_phi, m_f, q0, Q, Ei, E_f, TH_f):
    E_phi = Ei  - E_f + q0
    p_f = np.sqrt(E_f**2 - m_f**2)
    k_phi = np.sqrt(E_phi**2 - m_phi**2)

    kinematic_constraint = np.abs(p_f * np.sin(TH_f)) < k_phi
    

    #Missing important quadrant information
    TH_phi = np.where(kinematic_constraint, np.arcsin(p_f * np.sin(TH_f)/k_phi), 0)
    
    #crude way of determining quadrant of TH_phi
    sign1 = np.abs(k_phi*np.cos(TH_phi) + p_f * np.cos(TH_f) - (Ei + Q))
    sign2 = np.abs(k_phi*np.cos(TH_phi) - p_f * np.cos(TH_f) - (Ei + Q))
    
    TH_phi = np.where(sign1 < sign2, TH_phi, np.pi - TH_phi)

    J = (E_f**2 - m_f**2  - (E_phi**2 - m_phi**2)*np.sin(TH_phi)**2)
    J = np.sqrt(J/(E_phi**2 - m_phi**2))/np.cos(TH_phi) * kinematic_constraint
    
    return np.nan_to_num(E_phi), np.nan_to_num(TH_phi), np.nan_to_num(J)
    

def dcrossx_interp(X, Y, crossx_data):
    
    Xp, Yp, DSIG_DX_DY, DSIG_DX, DSIG_DY, SIG = read_dcrossx(crossx_data)
    
    interp = RegularGridInterpolator((Xp,Yp),
                                     DSIG_DX_DY,
                                     bounds_error = False,
                                     fill_value = 0)
        
    data = np.array([X.ravel(), Y.ravel()]).T
    return interp(data).reshape(X.shape)