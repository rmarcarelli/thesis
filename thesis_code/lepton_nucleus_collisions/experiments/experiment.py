#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment and final state classes for lepton-nucleus collision calculations.

This module defines the Experiment and FinalState classes that handle experiment
configurations, kinematics, form factors, and cross section calculations for
lepton-nucleus collisions.
"""

import numpy as np
import yaml
import os
from pathlib import Path
from functools import lru_cache, cached_property
import h5py
from dataclasses import dataclass
from typing import Optional

from .wrappers import convert, cached_evaluation
from ..compute import canonical_differential_cross_section, default_x, default_y
from ..compute.variables import LOG_GAMMA, ETA
from ..compute.wrappers import apply_canonical_transformation
from ..io import cache
from phys.constants import leptons, lepton_masses, me, mp, mn, hc_fmGeV


PATH = str(Path(__file__).resolve().parents[1])

@dataclass(frozen = True)
class Experiment():
    """
    Experiment configuration for lepton-nucleus collision calculations.
    
    This class encapsulates all experiment-specific parameters including
    lepton and nucleus properties, kinematics, form factors, and provides
    methods for cross section calculations and data caching.
    
    Parameters
    ----------
    name : str
        Experiment name
    lepton : str
        Lepton type ('e', 'mu', 'tau')
    Ei : float
        Initial lepton energy in lab frame (GeV)
    Z : int
        Nuclear charge number
    A : int
        Nuclear mass number
    E_nuc : float, optional
        Nucleus energy in lab frame (GeV, default: nucleus mass)
    coherent : bool, optional
        Whether to use coherent form factor (default: False)
    """
    
    ROOT_FILE_PATH = PATH + '/cached_differential_cross_sections'
    ROOT_CARD_PATH = PATH + '/experiments/cards'

    name: str
    lepton: str
    Ei: float
    Z: int
    A: int
    E_nuc: Optional[float] = None
    coherent: bool = False
    
    def __post_init__(self):
        """Validate experiment parameters after initialization."""
        assert self.lepton in leptons
        
        if self.E_nuc is None:
            object.__setattr__(self, 'E_nuc', self.M)
    
    @property
    def FILE_PATH(self):
        """Path to cached data file for this experiment."""
        return self.ROOT_FILE_PATH +'/'+ self.name + '.h5'
    
    @property
    def CARD_PATH(self):
        """Path to experiment configuration card."""
        return self.ROOT_CARD_PATH + '/' + self.name + '.yaml'
    
    # lepton properties
    @property
    def lepton_mass(self):
        """Lepton mass in GeV."""
        return  lepton_masses[self.lepton]
    
    @property
    def pi(self):
        """Initial lepton momentum in lab frame (GeV)."""
        return np.sqrt(self.Ei**2 - self.lepton_mass**2)
    
    @property
    def E(self):
        """Lepton energy in ion frame (GeV)."""
        return self.gamma_nuc * (self.Ei + self.v_nuc * self.pi)
    
    @property
    def p(self):
        """Lepton momentum in ion frame (GeV)."""
        return np.sqrt(self.E**2 - self.lepton_mass**2)
    
    # ion kinematics
    @property
    def M(self):
        """Nucleus mass in GeV."""
        return self.Z * mp + (self.A-self.Z)*mn
        
    @property
    def p_nuc(self):
        """Nucleus momentum in lab frame (GeV)."""
        return np.sqrt(self.E_nuc**2 - self.M**2)
    @property
    def gamma_nuc(self):
        """Nucleus boost factor."""
        return self.E_nuc/self.M
    
    @property
    def v_nuc(self):
        """Nucleus velocity in units of c."""
        return np.sqrt(1 - 1/self.gamma_nuc**2)
        
    @cached_property
    def form_factor_squared(self):
        """
        Atomic/nuclear electromagnetic form factor squared.
        
        Returns
        -------
        function
            Form factor squared as function of momentum transfer t
            
        Notes
        -----
        Includes both elastic and inelastic contributions for incoherent scattering,
        with appropriate modifications for different nucleus energies.
        """
        # Coherent and incoherent atomic/nuclear electromagnetic form factor
        def ff2(t):
            a = 0.79/hc_fmGeV #GeV^-1
            RA = 1.2*self.A**(1/3)/hc_fmGeV #GeV^-1
            q=np.sqrt(t)
            F=(3*(self.Z-1)/(RA*q)**3)*(np.sin(q*RA)-q*RA*np.cos(q*RA))/(1+(a*q)**2) #evaluate at Z - 1 instead of Z so Z = 1 goes to 0...
            
            G_elastic = F**2
            if self.E_nuc == self.M:
                a = 111/(self.Z**(1/3)*me)
                G_elastic *= (1/(1 + 1/(a**2*t)))**2 

            if not self.coherent:          
                dp = 0.71
                x = t/3.52
                G_inelastic = 1/(1 + t/dp)**4 * (self.Z*(1 + 7.78*x) + (self.A-self.Z)*3.65*x)/(1+x)

                if self.E_nuc == self.M:
                    ap = 571.4/(self.Z**(2/3)*me)
                    G_inelastic *= (1/(1+1/(ap**2 * t)))**2

                return G_elastic + G_inelastic

            return G_elastic
        
        return ff2

    @lru_cache(maxsize = 10000)
    def cross_section(self, final_state, kernel = None, x_var = LOG_GAMMA, y_var = ETA, n_pts_x = 100, n_pts_y = 200, **kwargs):
        """
        Calculate cross section with optional kernel integration.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        kernel : function, optional
            Kernel function for integration
        x_var : Variable, optional
            Energetic variable (default: LOG_GAMMA)
        y_var : Variable, optional
            Angular variable (default: ETA)
        n_pts_x : int, optional
            Number of x integration points (default: 100)
        n_pts_y : int, optional
            Number of y integration points (default: 200)
        **kwargs
            Additional arguments passed to differential cross section
            
        Returns
        -------
        float or array-like
            Cross section value(s)
        """

        if not kernel:
            return self.production_cross_section(final_state, **kwargs)
        x_grid = default_x(self, final_state, n_pts = n_pts_x, x_var = x_var)
        y_grid = default_y(n_pts = n_pts_y, y_var = y_var)
        x = x_grid.reshape(-1, 1)
        y = y_grid.reshape(1, -1)

        dcrossx = self.differential_cross_section(final_state, x, y, x_var = x_var, y_var = y_var, **kwargs)

        return np.trapezoid(np.trapezoid(kernel(self, final_state, x, y, x_var = x_var, y_var = y_var) * dcrossx, x = y_grid), x = x_grid)

    @lru_cache(maxsize = 10000)
    def production_cross_section(self, final_state, **kwargs):
        """
        Calculate production cross section.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        **kwargs
            Additional arguments
            
        Returns
        -------
        float or array-like
            Production cross section
        """
        return self.differential_cross_section(final_state, x = None, y = None, **kwargs)
    
    @convert()
    def differential_cross_section(self, final_state, x = None, y = None, **kwargs):
        """
        Calculate differential cross section.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        x : array-like, optional
            x-coordinate grid
        y : array-like, optional
            y-coordinate grid
        **kwargs
            Additional arguments
            
        Returns
        -------
        array-like
            Differential cross section
        """
        if x is not None:
            if y is not None:
                return self._xy_differential_cross_section(final_state, x, y, **kwargs)
            return self._x_differential_cross_section(final_state, x, **kwargs)
        if y is not None:
            return self._y_differential_cross_section(final_state, y, **kwargs)
                
        return self._cross_section(final_state, **kwargs)

    def distribution(self, final_state, x = None, y = None, **kwargs):
        """
        Calculate normalized distribution.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        x : array-like, optional
            x-coordinate grid
        y : array-like, optional
            y-coordinate grid
        **kwargs
            Additional arguments
            
        Returns
        -------
        array-like
            Normalized distribution
        """
        
        # Want to replicate the precision input to distribution, although this
        # still uses a different (default) set of x and y to compute the total cross section
        
        if x is not None:
            kwargs['n_pts_x'] = len(x)
            
        if y is not None:
            if x is None:
                kwargs['n_pts_y'] = len(y)
            else:
                kwargs['n_pts_y'] = len(y[0])
        
        dcrossx = self.differential_cross_section(final_state, x = x, y = y, **kwargs)
        crossx = self.differential_cross_section(final_state, x = None, y = None, **kwargs)

        # If the PV_angle is None, then crossx has a length of 2 (one for the PC component and
        # one for the PV component). This essentially only plots the PC distribution if the 
        # angle in the final_state is unspecified.
        if final_state.PV_angle is None:
            return dcrossx[0]/crossx[0]
        else:
            return dcrossx/crossx
    
    def cache_exists(self, final_state, key):
        """Check if cached data exists for given final state and key."""
        return cache.cache_exists(self, final_state, key)

    def load_data(self, final_state, key):
        """Load cached data for given final state and key."""
        return cache.load_data(self, final_state, key)

    def cache_data(self, final_state, key, data):
        """Cache data for given final state and key."""
        return cache.cache_data(self, final_state, key, data)
    
    def cached_log_gamma(self, final_state):
        """Load cached log_gamma grid for given final state."""
        return np.array(cache.load_data(self, final_state, 'x'))

    def cached_log_eta(self, final_state):
        """Load cached eta grid for given final state."""
        return np.array(cache.load_data(self, final_state, 'y'))

    def cached_final_states(self):
        """
        Get all cached final states for this experiment.
        
        Returns
        -------
        dict
            Dictionary mapping final state keys to FinalState objects
        """
        
        cached_final_state_dict = {}
        with h5py.File(self.FILE_PATH, 'r') as f:
            for method in f.keys():
                for t_cut_off in f[method].keys():
                    for lepton in f[method][t_cut_off].keys():
                        for boson_type in f[method][t_cut_off][lepton].keys():
                            for boson_mass in f[method][t_cut_off][lepton][boson_type].keys():
                                
                                key = (method, t_cut_off, lepton, boson_type, boson_mass)
                                
                                fs = FinalState(method = method,
                                                t_cut_off = t_cut_off,
                                                lepton = lepton,
                                                boson_type = boson_type,
                                                boson_mass = boson_mass,
                                                )
                                cached_final_state_dict[key] = fs
        return cached_final_state_dict

    def cached_masses(self, method, t_cut_off, lepton, boson_type, PV_angle = None):
        """
        Get cached boson masses for given parameters.
        
        Parameters
        ----------
        method : str
            Calculation method
        t_cut_off : float or str
            t-cutoff value
        lepton : str
            Lepton type
        boson_type : str
            Boson type
        PV_angle : float, optional
            PV angle (default: None)
            
        Returns
        -------
        array-like
            Sorted array of cached boson masses
        """
        if type(t_cut_off) != str:
            t_cut_off = f'{t_cut_off:.3e}'
        with h5py.File(self.FILE_PATH, 'r') as f:
            try:
                masses = np.sort(np.array(list(f[method][t_cut_off][lepton][boson_type][PV_angle].keys()), dtype = np.float32))
            except:
                masses = np.sort(np.array(list(f[method][t_cut_off][lepton][boson_type].keys()), dtype = np.float32))
        return masses
        
    def cache_canonical_differential_cross_section(self, final_state, log_gamma = None, eta = None, n_pts_x = 100, n_pts_y = 200, n_pts_t = 400):
        """
        Cache canonical differential cross section data.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        log_gamma : array-like, optional
            Log_gamma grid (default: auto-generated)
        eta : array-like, optional
            Eta grid (default: auto-generated)
        n_pts_x : int, optional
            Number of x points (default: 100)
        n_pts_y : int, optional
            Number of y points (default: 200)
        n_pts_t : int, optional
            Number of t integration points (default: 400)
        """

        save_final_state = FinalState(*final_state.params[:-1])
        
        if log_gamma is None:
            log_gamma = default_x(self, final_state, n_pts = n_pts_x)
        if eta is None:
            eta = default_y(n_pts = n_pts_y)
        
        dcrossx_dx_dy = self._canonical_xy_differential_cross_section(final_state, x = log_gamma.reshape(-1, 1), y = eta.reshape(1, -1), n_pts_t = n_pts_t, from_file = False)
        dcrossx_dx = np.trapezoid(dcrossx_dx_dy, x = eta, axis = -1)
        dcrossx_dy = np.trapezoid(dcrossx_dx_dy, x = log_gamma, axis = -2)
        crossx = np.trapezoid(dcrossx_dx, x = log_gamma, axis = -1)
                
        key = ['x', 'y', 'dcrossx_dx_dy', 'dcrossx_dx', 'dcrossx_dy', 'crossx']
        data = [log_gamma, eta, dcrossx_dx_dy, dcrossx_dx, dcrossx_dy, crossx] 
        cache.cache_data(self, save_final_state, key, data)

    def _cross_section(self, final_state, **kwargs):
        """
        Internal method to calculate cross section.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        **kwargs
            Additional arguments including frame, particle, variables, etc.
            
        Returns
        -------
        float or array-like
            Cross section value(s)
        """
        
        frame = kwargs.get('frame', 'ion')
        kwargs['frame'] = frame
        
        particle = kwargs.get('particle', 'boson')
        kwargs['particle'] = particle
        
        x_var = kwargs.get('x_var', LOG_GAMMA)
        kwargs['x_var'] = x_var
        
        n_pts_x = kwargs.get('n_pts_x', 100)
        kwargs['n_pts_x'] = n_pts_x
        
        y_var = kwargs.get('y_var', ETA)
        kwargs['y_var'] = y_var
        
        n_pts_y = kwargs.get('n_pts_y', 200)
        kwargs['n_pts_y'] = n_pts_y
        
        canonical = (frame == 'ion') and (particle == 'boson')
        canonical_x = x_var == LOG_GAMMA
        canonical_y = y_var == ETA
        
        x_grid = default_x(self, final_state, frame = frame, particle = particle, n_pts = n_pts_x, x_var = x_var)
        y_grid = default_y(n_pts = n_pts_y, y_var = y_var)
        
        if canonical:
            if canonical_x and canonical_y:
                return self._canonical_cross_section(final_state, **kwargs)
            elif canonical_x:
                # if x is canonical
                dcrossx = self._y_differential_cross_section(final_state, y_grid, **kwargs)
                return np.trapezoid(dcrossx, y_grid)
            elif canonical_y:
                # if x is canonical
                dcrossx = self._x_differential_cross_section(final_state, x_grid, **kwargs)
                return np.trapezoid(dcrossx, x_grid)
            
        dcrossx = self._xy_differential_cross_section(final_state, x_grid.reshape(-1, 1), y_grid.reshape(1, -1),  **kwargs)
        return np.trapezoid(np.trapezoid(np.nan_to_num(dcrossx), y_grid), x_grid)
        
    @lru_cache(maxsize = 10000)
    @cached_evaluation('crossx')
    def _canonical_cross_section(self, final_state, **kwargs):
        """Calculate canonical cross section with caching."""
        return canonical_differential_cross_section(self, final_state, x = None, y = None, **kwargs)
        
    def _x_differential_cross_section(self, final_state, x, x_var = LOG_GAMMA, **kwargs):
        """
        Internal method to calculate x-differential cross section.
        
        Parameters
        ----------
        final_state : FinalState
            Final state configuration
        x : array-like
            x-coordinate grid
        x_var : Variable, optional
            x-coordinate variable type (default: LOG_GAMMA)
        **kwargs
            Additional arguments
            
        Returns
        -------
        array-like
            x-differential cross section
        """
                
        frame = kwargs.get('frame', 'ion')
        kwargs['frame'] = frame
        
        particle = kwargs.get('particle', 'boson')
        kwargs['particle'] = particle
        
        kwargs['x_var'] = x_var
        
        kwargs['n_pts_x'] = len(x)
        
        y_var = kwargs.get('y_var', ETA)
        kwargs['y_var'] = y_var
        
        n_pts_y = kwargs.get('n_pts_y', 200)
        kwargs['n_pts_y'] = n_pts_y
        
        
        # if we have the canonical frame and particle, then we can transform
        # *within* the frame without mixing the two variables.
        canonical = ((frame == 'ion') or (self.v_nuc == 0)) and (particle == 'boson')
        if canonical:
            context = kwargs.copy()
            context['experiment'] = self
            context['final_state'] = final_state
            log_gamma, jacobian = LOG_GAMMA.transform(x, var = x_var, context = context)
            return jacobian * self._canonical_x_differential_cross_section(final_state, log_gamma.squeeze(), **kwargs)
                
        # otherwise, we must define a dummy y-coordinate to integrate over, according
        # to the provided specifications
        y_grid = default_y(n_pts = n_pts_y, y_var = y_var)
        dcrossx = self._xy_differential_cross_section(final_state, x.reshape(-1, 1), y_grid.reshape(1, -1), **kwargs)
        
        return np.trapezoid(np.nan_to_num(dcrossx), x = y_grid, axis = -1)

    @cached_evaluation('dcrossx_dx')
    def _canonical_x_differential_cross_section(self, final_state, x, **kwargs):
        return canonical_differential_cross_section(self, final_state, x = x, y = None, **kwargs)

    def _y_differential_cross_section(self, final_state, y, y_var = ETA, **kwargs):
        
        frame = kwargs.get('frame', 'ion')
        kwargs['frame'] = frame
        
        particle = kwargs.get('particle', 'boson')
        kwargs['particle'] = particle
        
        x_var = kwargs.get('x_var', LOG_GAMMA)
        kwargs['x_var'] = x_var
        
        n_pts_x = kwargs.get('n_pts_x', 100)
        kwargs['n_pts_x'] = n_pts_x
        
        kwargs['y_var'] = y_var
        
        kwargs['n_pts_y'] = len(y)
        
        
        # if we have the canonical frame and particle, then we can transform
        # *within* the frame without mixing the two variables.
        canonical = ((frame == 'ion') or (self.v_nuc == 0)) and (particle == 'boson')
        if canonical:
            eta, jacobian = ETA.transform(y, var = y_var)
            return jacobian * self._canonical_y_differential_cross_section(final_state, eta.squeeze(), **kwargs)
        
        # otherwise, we must define a dummy x-coordinate to integrate over, according
        # to the provided specifications
        x_grid = default_x(self, final_state, frame = frame, particle = particle, n_pts = n_pts_x, x_var = x_var)
        dcrossx = self._xy_differential_cross_section(final_state, x_grid.reshape(-1, 1), y.reshape(1, -1), **kwargs)
        
        return np.trapezoid(np.nan_to_num(dcrossx), x = x_grid, axis = -2)

    @cached_evaluation('dcrossx_dy') #pulls from file if cached
    def _canonical_y_differential_cross_section(self, final_state, y, **kwargs):
        return canonical_differential_cross_section(self, final_state, x = None, y = y, **kwargs)

    @apply_canonical_transformation() #transforms input vars to canonical vars
    def _xy_differential_cross_section(self, final_state, x, y, **kwargs):

        return self._canonical_xy_differential_cross_section(final_state, x, y, **kwargs)

    @cached_evaluation('dcrossx_dx_dy') # pulls from file if cached
    def _canonical_xy_differential_cross_section(self, final_state, x, y, **kwargs):
        return canonical_differential_cross_section(self, final_state, x = x, y = y, **kwargs)

    def __repr__(self):
        return self.name

    @classmethod
    def from_card(cls, path):
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)
    
    def to_card(self):
        
        os.makedirs(os.path.dirname(self.CARD_PATH), exist_ok = True)
        
        with open(self.CARD_PATH, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)


    def to_dict(self):
        return {'name': self.name,
                'lepton': self.lepton,
                'Ei': self.Ei,
                'Z': self.Z, 
                'A': self.A,
                'E_nuc': self.E_nuc,
                'coherent': self.coherent
               }
        
    
E137 = Experiment('E137', 'e', 20, 13, 26, coherent = False)
E137.to_card()

EIC = Experiment('EIC', 'e', 18, 79, 197, E_nuc = 110 * 197, coherent = True)
EIC.to_card()

MuBeD = Experiment('MuBeD', 'mu', 1000, 82, 208, coherent = False)
MuBeD.to_card()

MuSIC = Experiment('MuSIC', 'mu', 1000, 79, 197, E_nuc = 110 * 197, coherent = True)
MuSIC.to_card()


@dataclass(frozen = True)
class FinalState():
    """
    Final state configuration for lepton-nucleus collision calculations.
    
    This class encapsulates the final state parameters including the calculation
    method, kinematic cuts, particle types, and masses.
    
    Parameters
    ----------
    method : str
        Calculation method
    t_cut_off : float
        t-cutoff value
    lepton : str
        Final state lepton type
    boson_type : str
        Boson type
    boson_mass : float
        Boson mass in GeV
    PV_angle : float, optional
        PV angle (default: None)
    """
    method: str
    t_cut_off: float
    lepton: str
    boson_type: str
    boson_mass: float
    PV_angle: Optional[float] = None
    
    def __post_init__(self):
        """Validate final state parameters after initialization."""
        assert self.lepton in leptons
        assert self.boson_type in ['scalar', 'vector', 'pseudoscalar', 'axialvector']
        assert self.lepton in leptons
        assert self.method in ['exact', 'WW', 'IWW']
        
        
    @property
    def lepton_mass(self):
        """Final state lepton mass in GeV."""
        return lepton_masses[self.lepton]

    @property
    def PATH(self):
        """Path identifier for this final state."""
        path = f'/{self.method}/{self.t_cut_off:.3e}/{self.lepton}/{self.boson_type}/{self.boson_mass:.3e}'
        if self.PV_angle is not None:
            path = path + f'/{self.PV_angle:.3e}'
        return path
        
    @property
    def params(self):
        """Tuple of final state parameters."""
        return (self.method, self.t_cut_off, self.lepton, self.boson_type, self.boson_mass, self.PV_angle)
        
    def cross_section(self, experiment, **kwargs):
        """Calculate cross section for this final state."""
        return experiment.cross_section(self, **kwargs)

    def production_cross_section(self, experiment, **kwargs):
        """Calculate production cross section for this final state."""
        return experiment.production_cross_section(self, **kwargs)
    
    def differential_cross_section(self, experiment, **kwargs):
        """Calculate differential cross section for this final state."""
        return experiment.differential_cross_section(self, **kwargs)

    def distribution(self, experiment, **kwargs):
        """Calculate distribution for this final state."""
        return experiment.distribution(self, **kwargs)
    
    def cache_exists(self, experiment, key):
        """Check if cached data exists for this final state."""
        return experiment.cache_exists(self, key)

    def load_data(self, experiment, key):
        """Load cached data for this final state."""
        return experiment.load_data(self, key)

    def cache_data(self, experiment, key, data):
        """Cache data for this final state."""
        return experiment.cache_data(self, key, data)
    

    