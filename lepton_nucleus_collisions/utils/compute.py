#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import itertools
import numpy as np

from scipy.interpolate import RegularGridInterpolator

from lepton_nucleus_collisions.utils.process import (process_run_card,
                                                     process_dcrossx,
                                                     save_dcrossx
                                                     )

from lepton_nucleus_collisions.formulae.differential_cross_sections import calc_dcrossx

# Generates differential cross-section in E and Theta, given collider and run 
# specifications
def compute_energy_angle_differential_crossx(run_card):
    
    collider_name, run_name, initial_state_data, final_state_data = process_run_card(run_card)
    initial_lepton, E_lepton, Z, A, M, form_factor_squared, E_nuc, v_nuc, E = initial_state_data
    
    final_lepton, particle_type, t_cut_off, method, m = final_state_data
    combos = itertools.product(*(m, final_lepton, particle_type, t_cut_off, method))
    
    #Time estimation
    total_time = 0
    current_iteration = 1
    total_iterations = np.prod([len(data) for data in final_state_data])
    dt = 0    
    
    for m, final_lepton, particle_type, t_cut_off, method in combos:        
        start = time.time()
        
        final_state = final_lepton, particle_type, t_cut_off, method, m
        
        folder = 'lepton_nucleus_collisions/data/'
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        #Eta in ion rest frame
        eta_min = -5
        eta_max = 30
        eta = np.linspace(eta_min, eta_max, 200)
        
        #Theta in ion rest frame
        th_k = 2*np.arctan(np.exp(-eta))[::-1]
        
        #Energy in ion rest frame (lowest it can be is m)
        Ek = np.linspace(m, E, 100)
        
        DSIG_DE_DTH, DSIG_DE_DTH_PV = calc_dcrossx(initial_state_data, final_state, Ek, th_k)

        crossx_data = process_dcrossx(Ek, th_k, DSIG_DE_DTH)
        crossx_data_PV = process_dcrossx(Ek, th_k, DSIG_DE_DTH_PV)

        #save differential cross-sections in h5 file
        file_name = folder + f'/{collider_name}.h5'
        PV = False
        params = (final_lepton, particle_type, t_cut_off, method, PV, m)
        save_dcrossx(file_name, crossx_data, params)
        
        PV = True
        params = (final_lepton, particle_type, t_cut_off, method, PV, m)
        save_dcrossx(file_name, crossx_data_PV, params)
        

        stop = time.time()
        
        #Calculate rough time remaining
        total_time += stop-start
        estimated_time_in_seconds = (total_time/current_iteration)*(total_iterations - current_iteration)
        current_iteration += 1
        estimated_time = seconds_to_string(estimated_time_in_seconds)
        
        
        #Update every second
        dt += stop - start
        if dt > 1:
            print(f"Calculating differential cross-section for {run_name} in ion frame: {estimated_time} remaining", end = '\r')
            dt = 0

def seconds_to_string(duration):
    days = int(duration//(24*60*60))
    hours = int(duration % (24*60*60) //(60 * 60))
    mins = int(duration % (60*60)//60)
    secs = int(duration % 60)
    
    if days > 0:
        duration_string = f'{days}d, {hours}h'
    elif hours > 0:
        duration_string = f'{hours}h, {mins}m'
    elif mins > 0:
        duration_string = f'{mins}m, {secs}s'
    else:
        duration_string = f'{secs}s'
        
    return duration_string