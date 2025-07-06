import numpy as np
import functools
from dataclasses import replace

from phys.constants import hc2_fbGeV2
from ..io.cache import cache_exists, cached_interpolation

def convert():
    """    
    This only applies to (differential) cross-sections.

    Converts cross-sections from natural units (GeV^-2) to
    fb or pb, and optionally multiplies by a coupling g^2.
    
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, units = 'GeV', g = 1, **kwargs):

            # Can make this more general, but works for our purposes
            assert units in ['GeV', 'pb', 'fb'], "units must be GeV, pb, or fb"

            # Don't bother computing anything if g = 0
            if g == 0:
                return np.array(0)
                
            # Convert to pb
            if units == 'pb':
                return g**2 * hc2_fbGeV2 * func(*args, **kwargs)/1000
                
            # Convert to fb
            if units == 'fb':
                return g**2 * hc2_fbGeV2 * func(*args, **kwargs)
            
            return func(*args, **kwargs)

        return wrapper
        
    return decorator


def cached_evaluation(key):
    """
    A wrapper which checks whether data is cached,
    and if so, pulls it from the experiment's H5 file.
    """
    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(experiment, final_state, *args, interpolate_mass = False,  from_file = True, **kwargs):           
            
            if from_file:
            
                # if from_file is True and a cache exists, interpolate existing data
                if cache_exists(experiment, final_state, key):
                    return cached_interpolation(experiment, final_state, key)(*args)

                # if from_file is True but cache doesn't exist, tries to interpolate the mass value
                if interpolate_mass:

                    # compute which boson masses are saved for this final state
                    cached_masses = experiment.cached_masses(*final_state.params[:-2])

                    # can only interpolate if mass lies within the saved mass range
                    can_interpolate = np.any(cached_masses > final_state.boson_mass)*np.any(cached_masses < final_state.boson_mass)
                    if can_interpolate:
                        # compute nearest upper and lower masses to final_state.boson_mass
                        lower_mass = cached_masses[cached_masses < final_state.boson_mass].max()
                        upper_mass = cached_masses[cached_masses > final_state.boson_mass].min()

                        # initialize the final_states whose cached data needs to be interpolated
                        lower_final_state = replace(final_state, boson_mass = lower_mass)
                        upper_final_state = replace(final_state, boson_mass = upper_mass)

                        # make sure data is the same shape so that interpolation is possible
                        lower_data = cached_evaluation(key)(func)(experiment, lower_final_state, *args, **kwargs)
                        upper_data = cached_evaluation(key)(func)(experiment, upper_final_state, *args, **kwargs)
                        
                        if lower_data.shape == upper_data.shape:
                            # simple linear interpolation
                            t = (final_state.boson_mass - lower_mass)/(upper_mass - lower_mass)
                            return lower_data * (1-t) + upper_data * t
            
            return func(experiment, final_state, *args, **kwargs)
            
        return wrapper 
        
    return decorator