#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data caching utilities for lepton-nucleus collision calculations.

This module provides functions for caching and loading differential cross section
data using HDF5 files, with support for PC/PV component handling and interpolation.
"""

import numpy as np
import h5py
from functools import lru_cache
from scipy.interpolate import RegularGridInterpolator
import os

def cache_exists(experiment, final_state, key):
    """
    Check if cached data exists for given experiment, final state, and key.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration
    final_state : FinalState
        Final state configuration
    key : str
        Data key to check
        
    Returns
    -------
    bool
        True if cached data exists, False otherwise
    """
    
    internal_path = final_state.PATH
    if final_state.PV_angle is not None:
        internal_path, _, _ = internal_path.rpartition('/')
    full_path = internal_path + '/' + key
    
    with h5py.File(experiment.FILE_PATH, 'r') as f:
        return full_path in f
        
def cache_data(experiment, final_state, key, data):
    """
    Cache data for given experiment and final state.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration
    final_state : FinalState
        Final state configuration
    key : str or list of str
        Data key(s) to cache
    data : array-like or list of array-like
        Data to cache (must match key type)
        
    Notes
    -----
    If key is a list, data must also be a list of the same length.
    For PV_angle=None, data contains PC and PV components as tuple.
    """
    if isinstance(key, list):
        assert isinstance(data, list)
        for k, d in zip(key, data):
            cache_data(experiment, final_state, k, d)
    else:
        # Ensure the cache directory exists before writing
        os.makedirs(os.path.dirname(experiment.FILE_PATH), exist_ok=True)
        
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
    """
    Load cached data for given experiment and final state.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration
    final_state : FinalState
        Final state configuration
    key : str
        Data key to load
        
    Returns
    -------
    array-like
        Cached data
        
    Raises
    ------
    AssertionError
        If cached data does not exist
        
    Notes
    -----
    Handles PC/PV component reconstruction for specific PV angles.
    """
    
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
    """
    Create interpolation function for gridded data.
    
    Parameters
    ----------
    grid : tuple of array-like
        Grid coordinates (can be empty for scalar data)
    data : array-like
        Data to interpolate
        
    Returns
    -------
    function
        Interpolation function
        
    Notes
    -----
    Handles PC/PV components automatically.
    Supports 0D (scalar), 1D, and 2D interpolation.
    """
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
    """
    Create cached interpolation function for given experiment and final state.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration
    final_state : FinalState
        Final state configuration
    key : str
        Data key ('crossx', 'dcrossx_dx', 'dcrossx_dy', 'dcrossx_dx_dy')
        
    Returns
    -------
    function
        Cached interpolation function
    """
    
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
