"""
Energetic kinematic variables for lepton-nucleus collision calculations.

This module defines various energetic variables (log_gamma, gamma, velocity,
energy, momentum, energy fraction) and their transformation functions to/from
the canonical log_gamma variable.
"""

import numpy as np

from .base import Variable, IDENTITY

######### ENERGETIC VARIABLES ##########

# LOG(GAMMA)

LOG_GAMMA = Variable(['logg','loggamma', 'logboost'],
                     IDENTITY,
                     IDENTITY,
                     name = 'LOG_GAMMA')

#GAMMA

def GAMMA_to_LOG_GAMMA(gamma, *, context = None, include_jacobian = True):
    """
    Transform gamma (boost factor) to log_gamma.
    
    Parameters
    ----------
    gamma : float or array-like
        Boost factor (Lorentz factor)
    context : dict, optional
        Context (unused)
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (log_gamma, jacobian) if include_jacobian=True, otherwise log_gamma
    """
    if include_jacobian:
        return np.log(gamma), 1/(gamma)
    else:
        return np.log(gamma)
        
def LOG_GAMMA_to_GAMMA(log_gamma, *, context = None, include_jacobian = True):
    """
    Transform log_gamma to gamma (boost factor).
    
    Parameters
    ----------
    log_gamma : float or array-like
        Log of the boost factor
    context : dict, optional
        Context (unused)
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (gamma, jacobian) if include_jacobian=True, otherwise gamma
    """
    if include_jacobian:
        return np.exp(log_gamma), np.exp(log_gamma)
    else:
        return np.exp(log_gamma)

GAMMA = Variable(['g', 'gamma', 'boost'],
                 GAMMA_to_LOG_GAMMA,
                 LOG_GAMMA_to_GAMMA,
                 LOG_GAMMA,
                 name = 'GAMMA')

# VELOCITY
def VELOCITY_to_LOG_GAMMA(v, *, context = None, include_jacobian = True):
    """
    Transform velocity to log_gamma.
    
    Parameters
    ----------
    v : float or array-like
        Particle velocity (in units of c)
    context : dict, optional
        Context (unused)
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (log_gamma, jacobian) if include_jacobian=True, otherwise log_gamma
    """
    if include_jacobian:
        return -np.log(1 - v**2)/2, v/(1-v**2)
    return -np.log(1 - v**2)/2

def LOG_GAMMA_to_VELOCITY(log_gamma, *, context = None, include_jacobian = True):
    """
    Transform log_gamma to velocity.
    
    Parameters
    ----------
    log_gamma : float or array-like
        Log of the boost factor
    context : dict, optional
        Context (unused)
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (velocity, jacobian) if include_jacobian=True, otherwise velocity
    """
    if include_jacobian:
        return np.sqrt(1 - np.exp(-2*log_gamma)), np.where(log_gamma == 0, 0, np.exp(-log_gamma)/np.sqrt(np.exp(2*log_gamma) - 1))
    return np.sqrt(1 - np.exp(-2*log_gamma))

VELOCITY = Variable(['v', 'velocity'],
                    VELOCITY_to_LOG_GAMMA,
                    LOG_GAMMA_to_VELOCITY,
                    LOG_GAMMA,
                    name = 'VELOCITY')

# ENERGY

def ENERGY_to_LOG_GAMMA(energy, *, context = None, include_jacobian = True):
    """
    Transform energy to log_gamma.
    
    Parameters
    ----------
    energy : float or array-like
        Particle energy
    context : dict
        Context containing final_state and particle info
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (log_gamma, jacobian) if include_jacobian=True, otherwise log_gamma
        
    Notes
    -----
    Requires context with 'final_state' and optionally 'particle' ('boson' or 'lepton').
    """
    # Process context
    assert context is not None and 'final_state' in context
    final_state = context['final_state']
    
    particle = context.get('particle', 'boson')
    assert particle.strip().lower() in ['boson', 'lepton']

    # Compute transformation
    mass = final_state.boson_mass if particle == 'boson' else final_state.lepton_mass
    gamma = energy/mass
    
    if include_jacobian:
        #d(log_gamma)/d(energy) = d(log_gamma)/d(gamma) d(gamma)/d(energy)
        log_gamma, jacob = GAMMA_to_LOG_GAMMA(gamma, context = context)
        return log_gamma, jacob / mass
    log_gamma = GAMMA_to_LOG_GAMMA(gamma, context = context, include_jacobian = False)
    return log_gamma
        

def LOG_GAMMA_to_ENERGY(log_gamma, *, context = None, include_jacobian = True):
    """
    Transform log_gamma to energy.
    
    Parameters
    ----------
    log_gamma : float or array-like
        Log of the boost factor
    context : dict
        Context containing final_state and particle info
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (energy, jacobian) if include_jacobian=True, otherwise energy
        
    Notes
    -----
    Requires context with 'final_state' and optionally 'particle' ('boson' or 'lepton').
    """
    # Process context
    assert context is not None and 'final_state' in context
    final_state = context['final_state']
    
    particle = context.get('particle', 'boson')
    assert particle.strip().lower() in ['boson', 'lepton']

    # Compute transformation
    mass = final_state.boson_mass if particle == 'boson' else final_state.lepton_mass

    if include_jacobian:    
        #d(energy)/d(log_gamma) = d(energy)/d(gamma) d(gamma)/d(log_gamma)
        gamma, jacob = LOG_GAMMA_to_GAMMA(log_gamma, context = context)
        return mass * gamma, mass * jacob
    gamma = LOG_GAMMA_to_GAMMA(log_gamma, include_jacobian = False)
    return mass * gamma

ENERGY = Variable(['e', 'en', 'energy'],
                  ENERGY_to_LOG_GAMMA,
                  LOG_GAMMA_to_ENERGY,
                  LOG_GAMMA,
                  name = 'ENERGY')

# MOMENTUM

def MOMENTUM_to_LOG_GAMMA(momentum, *, context = None, include_jacobian = True):
    """
    Transform momentum to log_gamma.
    
    Parameters
    ----------
    momentum : float or array-like
        Particle momentum
    context : dict
        Context containing final_state and particle info
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (log_gamma, jacobian) if include_jacobian=True, otherwise log_gamma
        
    Notes
    -----
    Requires context with 'final_state' and optionally 'particle' ('boson' or 'lepton').
    """
    # Process context
    assert context is not None and 'final_state' in context
    final_state = context['final_state']
    
    particle = context.get('particle', 'boson')
    assert particle.strip().lower() in ['boson', 'lepton']

    # Compute transformation
    mass = final_state.boson_mass if particle == 'boson' else final_state.lepton_mass
    gamma = np.sqrt(1 + (momentum/mass)**2)
    if include_jacobian:
        #d(log_gamma)/d(energy) = d(log_gamma)/d(gamma) d(gamma)/d(energy)
        log_gamma, jacob = GAMMA_to_LOG_GAMMA(gamma, context = context)
        return log_gamma, jacob / mass * momentum / np.sqrt(mass**2 + momentum**2)
    log_gamma = GAMMA_to_LOG_GAMMA(gamma, context = context, include_jacobian = False)
    return log_gamma
        

def LOG_GAMMA_to_MOMENTUM(log_gamma, *, context = None, include_jacobian = True):
    """
    Transform log_gamma to momentum.
    
    Parameters
    ----------
    log_gamma : float or array-like
        Log of the boost factor
    context : dict
        Context containing final_state and particle info
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (momentum, jacobian) if include_jacobian=True, otherwise momentum
        
    Notes
    -----
    Requires context with 'final_state' and optionally 'particle' ('boson' or 'lepton').
    """
    # Process context
    assert context is not None and 'final_state' in context
    final_state = context['final_state']
    
    particle = context.get('particle', 'boson')
    assert particle.strip().lower() in ['boson', 'lepton']

    # Compute transformation
    mass = final_state.boson_mass if particle == 'boson' else final_state.lepton_mass

    if include_jacobian:    
        #d(energy)/d(log_gamma) = d(energy)/d(gamma) d(gamma)/d(log_gamma)
        gamma, jacob = LOG_GAMMA_to_GAMMA(log_gamma, context = context)
        return mass*np.sqrt(gamma**2 - 1), np.where(gamma == 1, 0, mass * jacob * gamma/np.sqrt(gamma**2-1))
    gamma = LOG_GAMMA_to_GAMMA(log_gamma, include_jacobian = False)
    return mass*np.sqrt(gamma**2 - 1)

MOMENTUM = Variable(['p', 'k', 'momentum'],
                    MOMENTUM_to_LOG_GAMMA,
                    LOG_GAMMA_to_MOMENTUM,
                    LOG_GAMMA,
                    name = 'MOMENTUM')

# ENERGY_FRACTION

def ENERGY_FRACTION_to_LOG_GAMMA(x, *, context = None, include_jacobian = True):
    """
    Transform energy fraction to log_gamma.
    
    Parameters
    ----------
    x : float or array-like
        Energy fraction (energy / initial_energy)
    context : dict
        Context containing experiment, final_state, particle, and frame info
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (log_gamma, jacobian) if include_jacobian=True, otherwise log_gamma
        
    Notes
    -----
    Requires context with 'experiment', 'final_state', and optionally 'particle' 
    ('boson' or 'lepton') and 'frame' ('lab' or 'ion').
    """
    # Process context
    assert context is not None and 'experiment' in context and 'final_state' in context
    experiment = context['experiment']
    
    particle = context.get('particle', 'boson')
    assert particle.strip().lower() in ['boson', 'lepton']
    
    frame = context.get('frame', 'lab')
    assert frame.strip().lower() in ['lab', 'ion']
    
    initial_energy = experiment.Ei if frame == 'lab' else experiment.E #Ei is lab frame initial lepton energy, E is ion-frame initial lepton energy
    energy = x * initial_energy
    
    if include_jacobian:
        # d(log_gamma)/d(x) = d(log_gamma)/d(energy) d(energy)/d(x)
        log_gamma, jacob = ENERGY_to_LOG_GAMMA(energy, context = context)
        return log_gamma, jacob * initial_energy
    log_gamma = ENERGY_to_LOG_GAMMA(energy, context = context, include_jacobian = False)
    return log_gamma

def LOG_GAMMA_to_ENERGY_FRACTION(log_gamma, *, context = None, include_jacobian = True):
    """
    Transform log_gamma to energy fraction.
    
    Parameters
    ----------
    log_gamma : float or array-like
        Log of the boost factor
    context : dict
        Context containing experiment, final_state, particle, and frame info
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (energy_fraction, jacobian) if include_jacobian=True, otherwise energy_fraction
        
    Notes
    -----
    Requires context with 'experiment', 'final_state', and optionally 'particle' 
    ('boson' or 'lepton') and 'frame' ('lab' or 'ion').
    """
    # Process context
    assert context is not None and 'experiment' in context and 'final_state' in context
    experiment = context['experiment']
    
    particle = context.get('particle', 'boson')
    assert particle.strip().lower() in ['boson', 'lepton']
    
    frame = context.get('frame', 'lab')
    assert frame.strip().lower() in ['lab', 'ion']
    
    initial_energy = experiment.Ei if frame == 'lab' else experiment.E #Ei is lab frame initial lepton energy, E is ion-frame initial lepton energy

    if include_jacobian:
        # d(x)/d(log_gamma) = d(x)/d(energy) d(energy)/d(log_gamma)
        energy, jacob = LOG_GAMMA_to_ENERGY(log_gamma, context = context)
        return energy/initial_energy, jacob/initial_energy
    energy = LOG_GAMMA_to_ENERGY(log_gamma, context = context, include_jacobian = False)
    return energy/initial_energy

ENERGY_FRACTION = Variable(['x', 'energyfraction'],
                           ENERGY_FRACTION_to_LOG_GAMMA,
                           LOG_GAMMA_to_ENERGY_FRACTION,
                           LOG_GAMMA,
                           name = 'ENERGY_FRACTION')
