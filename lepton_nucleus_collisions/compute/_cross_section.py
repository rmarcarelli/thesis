#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from phys.constants import hc2_fbGeV2

def production_cross_section(experiment, final_state, n_pts_t = 400, **kwargs):
    return experiment.differential_cross_section(final_state, x = None, y = None, **kwargs)

def cross_section(experiment, final_state, kernel = None, n_pts_x = 100, n_pts_y = 200, units = 'pb', g = 1):
    """
    Parameters
    ----------
    experiment: Experiment
        An Experiment object which contains the initial-state
        kinematic information of the experiment in question.
        
    m: float
        The mass of the produced boson, in GeV.
    
    kernel: function, optional
        An integration kernel which incorporates important signal information, 
        such as detector geometry, detection efficiencies, and displaced decay
        probabilities. If None, the kernel is the identity.

        The default is None.
        
    n_pts_E: int, optional
        How many points are sampled for the energy of the produced boson.

        The default is 100.

    n_pts_th: int, optional
        How many points are sampled for the angle of the produced boson.

        The default is 200.

    units: str, optional
        Which units the cross-section is to be returned it. Either 'fb', 'pb',
        or 'GeV' (this really means 1/GeV^2).

        The default is 'pb'.

    g: float
        The coupling between the dark boson and the lepton vector current.
        
        The default is 1.

    Returns
    -------
    float
        The total cross-section obtained by integrating the differential cross-section
        for producing the boson against the provided kernel. 

    """

    assert units in ['GeV', 'pb', 'fb']

    if units == 'fb':
        return hc2_fbGeV2*cross_section(experiment, final_state, kernel, n_pts_x, n_pts_y, 'GeV', g)
    if units == 'pb':
        return cross_section(experiment, final_state, kernel, n_pts_x, n_pts_y, 'fb', g)/1000
    if g != 1:
        return g**2 * cross_section(experiment, final_state, kernel, n_pts_x, n_pts_y, units, 1)

    return _normalized_cross_section_in_GeV(experiment, final_state, kernel, n_pts_x, n_pts_y)
    
@lru_cache(maxsize = 1000)
def _normalized_cross_section_in_GeV(experiment, final_state, kernel = None, n_pts_x = 100, n_pts_y = 200):
    
    log_gamma = np.linspace(0, np.log(experiment.E/final_state.boson_mass), n_pts_x)
    eta = np.linspace(-30, 30, n_pts_y)
    
    dcrossx = experiment.differential_cross_section(final_state, log_gamma, eta)

    if not kernel:
        kernel = lambda w, x, y, z: 1
    
    return np.trapezoid(np.trapezoid(kernel(experiment, final_state, Ek.reshape(-1, 1), th_k.reshape(1, -1)) * dcrossx, x = th_k), x = Ek)