"""
Integration and transformation wrapper utilities for lepton-nucleus collision calculations.

This module provides decorators and utility functions for automatic integration
and coordinate transformations in differential cross section calculations.
"""

import numpy as np
import functools

from .variables import LOG_GAMMA, ETA
from .transformations import TRANSFORM


def default_x(experiment, final_state, frame = 'ion', particle = 'boson', n_pts = 100, x_var = LOG_GAMMA):
    """
    Generate default x-coordinate grid for integration.
    
    This function creates a default grid for the x-coordinate (typically log_gamma)
    based on the experiment kinematics and final state particle properties.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration object
    final_state : FinalState
        Final state configuration object
    frame : str, optional
        Reference frame ('ion' or 'lab', default: 'ion')
    particle : str, optional
        Final state particle ('boson' or 'lepton', default: 'boson')
    n_pts : int, optional
        Number of grid points (default: 100)
    x_var : Variable class, optional
        Variable type for x-coordinate (default: LOG_GAMMA)
        
    Returns
    -------
    array-like
        Grid of x-coordinate values for integration
    """
    
    mass = final_state.boson_mass if particle == 'boson' else final_state.lepton_mass
    
    # this can definitely be optimized...
    E = experiment.Ei if (frame == 'lab') else experiment.E
    x_min = np.log(1.0+1e-3)
    x_max = np.log(max(1, E/mass) +  1.2 * mass/E)

    log_gamma = np.linspace(x_min, x_max, n_pts)

    context = {'experiment': experiment,
               'final_state': final_state,
               'frame': frame,
               'particle': particle}

    return LOG_GAMMA.inverse(log_gamma, var = x_var, context = context)

def default_y(n_pts = 200, y_var = ETA):
    """
    Generate default y-coordinate grid for integration.
    
    This function creates a default grid for the y-coordinate (typically eta)
    covering the full pseudorapidity range.
    
    Parameters
    ----------
    n_pts : int, optional
        Number of grid points (default: 200)
    y_var : Variable class, optional
        Variable type for y-coordinate (default: ETA)
        
    Returns
    -------
    array-like
        Grid of y-coordinate values for integration
    """
    eta = np.linspace(-30, 30, n_pts)
    context = {'sign': np.sign(eta)}
    return ETA.inverse(eta, var = y_var, context = context)
    

def auto_integrate(force_canonical = False):
    """
    Decorator for automatic integration of differential cross sections.
    
    This decorator automatically handles integration over x and y coordinates
    when they are not provided, using default grids and trapezoidal integration.
    
    Parameters
    ----------
    force_canonical : bool, optional
        Whether to force use of canonical variables (log_gamma, eta) (default: False)
        
    Returns
    -------
    function
        Decorated function with automatic integration capability
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(experiment, final_state, x = None, y = None, x_var = LOG_GAMMA, y_var = ETA, frame = 'ion', particle = 'boson', n_pts_x = 100, n_pts_y = 200, **kwargs):
            integrate_x = x is None
            integrate_y = y is None
            
            if force_canonical:
                x_var = LOG_GAMMA
                y_var = ETA
            
            if integrate_x:

                x_grid = default_x(experiment, final_state, frame = frame, particle = particle, x_var = x_var, n_pts = n_pts_x) # define log_gamma based on frame, particle
                x = x_grid.reshape(-1, 1)
                                
                if y is not None:
                    y = y.reshape(1, -1)
            if integrate_y:
                y_grid = default_y(n_pts = n_pts_y, y_var = y_var) # define eta based on frame, particle
                y = y_grid.reshape(1, -1)
                
                if x is not None:
                    x = x.reshape(-1, 1)

            result = func(experiment, final_state, x, y, x_var = x_var, y_var = y_var, frame = frame, particle = particle, **kwargs)

            if integrate_x:
                result = np.trapezoid(result, x = x_grid, axis = -2)
            if integrate_y:
                result = np.trapezoid(result, x = y_grid, axis = -1)
            return result
        return wrapper
    return decorator

def apply_canonical_transformation():
    """
    Decorator for automatic transformation to canonical coordinates.
    
    This decorator automatically transforms input coordinates to canonical
    variables (log_gamma, eta) in the ion frame for the boson particle,
    applying the appropriate Jacobian factor.
    
    Returns
    -------
    function
        Decorated function with automatic coordinate transformation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(experiment, final_state, x = None, y = None, x_var = LOG_GAMMA, y_var = ETA, frame = 'ion', particle = 'boson', **kwargs):
                        
            # context for transformation
            in_context = {'experiment': experiment,
                          'final_state': final_state,
                          'frame': frame,
                          'particle': particle}
            
            out_context = {'experiment': experiment,
                          'final_state': final_state,
                          'frame': 'ion',
                          'particle': 'boson'}

            if experiment.v_nuc == 0:
                frame = 'ion'
            
            log_gamma, eta, jacobian = TRANSFORM(x, y, x_in = x_var, y_in = y_var, in_context = in_context, out_context = out_context)

            return jacobian * func(experiment, final_state, log_gamma, eta, **kwargs)
        return wrapper
    return decorator