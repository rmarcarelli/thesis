#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from lepton_nucleus_collisions.utils.process import (process_run_card,
                                                     read_dcrossx,
                                                     open_dcrossx)

def lab_frame_distributions(run, params, gamma, eta, save_axes = (0, 1), which = 'phi'):
    
    v_nuc = process_run_card(f'{run}.txt', var = 'v_nuc')
    final_lepton, particle_type, t_cut_off, method, PV, m = params
    
    #get Ek and th_k
    file_name = f'lepton_nucleus_collisions/data/{run}.h5'
    dcrossx_data = open_dcrossx(file_name, params)
    interpolation = interpolate_ion_frame_energy_angle_differential_crossx(dcrossx_data)

    th_lab = 2*np.arctan(np.exp(-eta))
    J_th = 1/np.cosh(eta) #Jacobian of transformation
    E_lab = gamma * m #energy
    J_E = m
    
    #reshape for broadcasting
    E_lab = E_lab.reshape(-1, 1) 
    th_lab = th_lab.reshape(1, -1) 

    Ek, THk = E_THETA_lab_transform(m, v_nuc, E_lab, th_lab) #convert to frame of nucleus
    
    #if which == 'lepton':
        
    #   then should really have
    #   El, THl = E_THETA_lab_transform(ml, v_nuc, E_lab, th_lab)
    #   THEN need conversion from El, THl to Ek, THk.
    #   Not too hard... maybe later
        
    #    E, M = process_run_card(f'{run}.txt', var = 'v_nuc')
    #    t_min = (((m+mf)**2 - mi**2)/(2*E))**2
    #    J_lepton = np.sqrt((E - Ek + t_min/(2*M))**2 - mf**2 - (Ek**2 - m**2)*np.sin(thk)**2)
    #    J_lepton = J_lepton/(np.sqrt(Ek**2 - m**2) * np.cos(THk))
    
    #    dcrossx_dgamma_deta = J_E * J_th * interpolation(Ek, THk) 
    
    dcrossx_dgamma_deta = J_E * J_th * interpolation(Ek, THk) 

    axes = ()
    
    if (0, 1) in save_axes:
        axes += (dcrossx_dgamma_deta,)
    if 0 in save_axes:
        dcrossx_dgamma = np.trapz(dcrossx_dgamma_deta, x = eta, axis = 1)
        axes += (dcrossx_dgamma,)
    if 1 in save_axes:
        dcrossx_deta = np.trapz(dcrossx_dgamma_deta, x = gamma, axis = 0)
        axes += (dcrossx_deta,)
        
    return axes

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

def interpolate_ion_frame_energy_angle_differential_crossx(crossx_data):
    
    E, TH, DSIG_DE_DTH, _, _, _ = read_dcrossx(crossx_data)
    
    interpolation = RegularGridInterpolator((E, TH),
                                            DSIG_DE_DTH,
                                            bounds_error = False,
                                            fill_value = 0)
    
    def dcrossx_interpolation(X, Y, func = interpolation):
        data = np.array([X.ravel(), Y.ravel()]).T
        return func(data).reshape(X.shape)

    return dcrossx_interpolation