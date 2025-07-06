#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from functools import lru_cache
from scipy.interpolate import RegularGridInterpolator

def cache_exists(experiment, final_state, key):
    
    internal_path = final_state.PATH
    if final_state.PV_angle is not None:
        internal_path, _, _ = internal_path.rpartition('/')
    full_path = internal_path + '/' + key
    
    with h5py.File(experiment.FILE_PATH, 'r') as f:
        return full_path in f
        
def cache_data(experiment, final_state, key, data):

    if isinstance(key, list):
        assert isinstance(data, list)
        for k, d in zip(key, data):
            cache_data(experiment, final_state, k, d)
    else:
        # If PV_angle is None, then data is essentially a tuple
        # with the PC and PV component as the first and second 
        # components, so saving an individual final state is 
        # discouraged.
        
        if final_state.PV_angle is not None:
            print('Note: saving multiple PV angles for the final-state is redundant.'
                  'It is better to set PV_angle = None, which will automatically store'
                  'the PC and PV component of the (differential) cross-section.')
    
        with h5py.File(experiment.FILE_PATH, 'a') as f:
            group = f.require_group(final_state.PATH)
            if key in group:
                del group[key]
                
            group.create_dataset(key, data = np.float32(data), compression = 'gzip')
        

def load_data(experiment, final_state, key):
    
    assert cache_exists(experiment, final_state, key), f"{final_state} data {key} is not saved for {experiment}."

    internal_path = final_state.PATH
    
    with h5py.File(experiment.FILE_PATH) as f:

        # If this succeeds, data is saved for the final_state with this PV_angle (which can be None)
        try:
            return np.array(f[internal_path][key])

        # If it doesn't, given that cache_exists returns True, no data is saved for this specific PV_angle,
        # but the individual PC and PV components are saved.
        except KeyError:
            internal_path, _, _ = internal_path.rpartition('/') # removes the PV_angle

            data = np.array(f[internal_path][key])
            if len(data) == 2: # then PC and PV component are first and second component
                PC, PV = data
                return PC + np.sin(final_state.PV_angle)**2 * PV

            return data

def grid_interpolation(grid, data):
    # Handles the PV case automatically.
    # Note that this will not work on the off-chance that 
    # you have sampled exactly two x or y values... I will
    # assume this is not the case.
    if len(grid) == 0:
        return lambda: data

    if len(data) == 2:
        return lambda *args: np.stack((grid_interpolation(grid, data[0])(*args),
                                       grid_interpolation(grid, data[1])(*args)))
    
    if len(grid) == 1:
        return lambda x: np.interp(x, grid[0], data, left = 0, right = 0)
    if len(grid) == 2:
        func = RegularGridInterpolator(grid, data, fill_value = 0, bounds_error = False)
        
        def interp(x, y):
            x, y = np.broadcast_arrays(x, y)
            return func(np.stack([x.ravel(),y.ravel()], axis = -1)).reshape(x.shape)
        
        return interp
    

@lru_cache(maxsize = 10000)
def cached_interpolation(experiment, final_state, key):
    
    assert key in ['crossx', 'dcrossx_dx', 'dcrossx_dy', 'dcrossx_dx_dy']
    
    x_grid = load_data(experiment, final_state, 'x')
    y_grid = load_data(experiment, final_state, 'y')
    data = load_data(experiment, final_state, key)
    
    if key == 'crossx':
        grid = ()
    if key == 'dcrossx_dx':
        grid = (x_grid,)
    if key == 'dcrossx_dy':
        grid = (y_grid,)
    if key == 'dcrossx_dx_dy':
        grid = (x_grid, y_grid)
    
    return grid_interpolation(grid, data)
