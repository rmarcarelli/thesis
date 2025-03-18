#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py

from phys.constants import ml, hc_fmGeV

#run and collider card processing

def read_card(card):
    parameters = {} 
    with open(card, 'r') as file:
        
        for line in file:
            
            line = line.strip()            
            if ':' in line:
                
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                parameters[key] = value
            
    return parameters

def process_collider_card(collider_card, var = None):
    parameters = read_card(collider_card)
    
    collider_name = parameters['Collider name']
    lepton = parameters['Initial lepton']
    E_lepton = float(parameters['Lepton energy'])
    
    #nucleus_name = parameters['Nucleus name']
    E_nuc = float(parameters['Nucleus energy'])
    A = int(parameters['A'])
    Z = int(parameters['Z'])
    M = float(parameters['M'])
        
    p_lepton = np.sqrt(E_lepton**2 - ml[lepton_idx(lepton)]**2)

    g_nuc = E_nuc/M
    v_nuc = np.sqrt(1-1/g_nuc**2)
    
    E = g_nuc * (E_lepton + v_nuc * p_lepton)
    
    if E_nuc == M:
        
        def form_factor_squared(t):            
            a = 0.79/hc_fmGeV #GeV^-1
            RA = 1.2*A**(1/3)/hc_fmGeV #GeV^-1
            q=np.sqrt(t)
            F=(3*Z/(RA*q)**3)*(np.sin(q*RA)-q*RA*np.cos(q*RA))/(1+(a*q)**2)
            
            a = 111/(Z**(1/3)*ml[0])
            F1 = F**2 * (1/(1 + 1/(a**2*t)))**2 
            
            ap = 571.4/(Z**(2/3)*ml[0])
            dp = 0.71
            
            x = t/3.52
        
            F2 = (1/(1+1/(ap**2 * t)))**2 * 1/(1 + t/dp)**4 * (Z*(1 + 7.78*x) + (A-Z)*3.65*x)/(1+x)
            
            return F1 + F2
            
    else:
        def form_factor_squared(t):
            a = 0.79/hc_fmGeV #GeV^-1
            RA = 1.2*A**(1/3)/hc_fmGeV #GeV^-1
            q=np.sqrt(t)
            F=(3*Z/(RA*q)**3)*(np.sin(q*RA)-q*RA*np.cos(q*RA))/(1+(a*q)**2)
            
            return F**2
        
    initial_state_data = (lepton, E_lepton, Z, A, M, form_factor_squared, E_nuc, v_nuc, E)
    
    var_dict = {'collider': collider_name,
                'lepton': lepton,
                'E_lepton': E_lepton,
                'Z': Z,
                'A': A,
                'M': M,
                'FF2': form_factor_squared,
                'E_nuc': E_nuc,
                'v_nuc': v_nuc,
                'E': E}
    
    if var:    
        if type(var) == str:
            var = [var]
            
        requested_var = ()
        for v in var:
            requested_var += (var_dict[v],)
            
        return requested_var if len(requested_var) > 1 else requested_var[0]
                    
    return collider_name, initial_state_data

def process_run_card(run_card, var = None):
    parameters = read_card('lepton_nucleus_collisions/run_cards/' + run_card)
    
    run_name = parameters["Run name"]
    collider_card = parameters["Collider card"]
    final_leptons = parameters["Final lepton"].replace(' ', '').split(',')
    particle_types = parameters["Particle type"].replace(' ', '').split(',')
    t_cut_off = parameters["t_cut_off"].replace(' ', '').split(',')
    method = parameters["method"].replace(' ', '').split(',')
    particle_masses = parameters["Particle mass"].replace(' ', '').split(',')
    
    t_cut_off = [float(t) for t in t_cut_off]
    particle_masses = [float(m) for m in particle_masses]
    
    collider_name, initial_state_data = process_collider_card('lepton_nucleus_collisions/collider_cards/' + collider_card)
    
    final_state_data = (final_leptons, particle_types, t_cut_off, method, particle_masses)
    
    initial_lepton, E_lepton, Z, A, M, form_factor_squared, E_nuc, v_nuc, E = initial_state_data
    
    var_dict = {'collider': collider_name,
                'li': initial_lepton,
                'E_lepton': E_lepton,
                'Z': Z,
                'A': A,
                'M': M,
                'FF2': form_factor_squared,
                'E_nuc': E_nuc,
                'v_nuc': v_nuc,
                'E': E,
                'lf': final_leptons,
                'particle': particle_types,
                't_cut_off': t_cut_off,
                'method': method,
                'masses': particle_masses}
    
    if var:    
        if type(var) == str:
            var = [var]
            
        requested_var = ()
        for v in var:
            requested_var += (var_dict[v],)
                
        return requested_var if len(requested_var) > 1 else requested_var[0]

    return collider_name, run_name, initial_state_data, final_state_data


def lepton_idx(lepton):
    return 0*(lepton == 'e') + 1*(lepton == 'mu') + 2*(lepton == 'tau')

#cross-section processing

def process_dcrossx(A, B, DSIG_DA_DB):
    DSIG_DA = np.trapz(DSIG_DA_DB, x = B, axis = 1)
    DSIG_DB = np.trapz(DSIG_DA_DB, x = A, axis = 0)
    SIG = np.trapz(DSIG_DA, x = A)

    crossx_data = np.empty((len(A) + 2, len(B) + 2))
    crossx_data[1:-1,0] = A
    crossx_data[0,1:-1] = B
    crossx_data[1:-1,1:-1] = DSIG_DA_DB
    crossx_data[1:-1, -1] = DSIG_DA
    crossx_data[-1, 1:-1] = DSIG_DB
    crossx_data[-1, -1] = SIG
    
    return crossx_data
    
def read_dcrossx(dcrossx_data):
    A = dcrossx_data[1:-1, 0]
    B = dcrossx_data[0,1:-1]
    DSIG_DA_DB = dcrossx_data[1:-1,1:-1]
    DSIG_DA = dcrossx_data[1:-1, -1]
    DSIG_DB = dcrossx_data[-1, 1:-1]
    SIG = dcrossx_data[-1, -1]
    
    return A, B, DSIG_DA_DB, DSIG_DA, DSIG_DB, SIG   
    
def save_dcrossx(file_name, crossx_data, params):
    
        #file_name = 'data/' + collider_name + '_E_TH.h5'
        
        with h5py.File(file_name, 'a') as f:
            group = f
            for p in params:
                group = group.require_group(str(p))
            if 'dcrossx' in group:
                del group['dcrossx']
            group.create_dataset('dcrossx', data = np.float32(crossx_data), compression = 'gzip')
            
def open_dcrossx(file_name, params, var = None):
    with h5py.File(file_name) as f:
        group = f
        for p in params:
            group = group[str(p)]
        dcrossx_data = np.array(group['dcrossx'])
        
        if var:
        
            A, B, DSIG_DA_DB, DSIG_DA, DSIG_DB, SIG = read_dcrossx(dcrossx_data)
            
            var_dict = {'E': A,
                        'THETA': B,
                        'DSIG_DE_DTHETA': DSIG_DA_DB,
                        'DSIG_DE': DSIG_DA,
                        'DSIG_DTHETA': DSIG_DB,
                        'SIG': SIG}
            
            if type(var) == str:
                var = [var]
            
            requested_var = ()
            for v in var:
                requested_var += (var_dict[v],)
                    
            return requested_var if len(requested_var) > 1 else requested_var[0]
    
    return dcrossx_data