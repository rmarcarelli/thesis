#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from phys.constants import alpha

from .variables import LOG_GAMMA, ENERGY, MOMENTUM, ETA, SIN_THETA, SIN_HALF_THETA
from .wrappers import auto_integrate, apply_canonical_transformation

@auto_integrate()
@apply_canonical_transformation()
def differential_cross_section(experiment, final_state, x, y, n_pts_t = 400, **kwargs):
    """
    Calculate differential cross section for lepton-nucleus collisions.
    
    This is the main function for computing differential cross sections
    in lepton-nucleus collisions, with automatic integration and jacobian
     transformations applied via decorators.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration object
    final_state : FinalState
        Final state configuration object
    x : float or array-like
        Energetic variable (log_gamma by default)
    y : float or array-like
        Angular variable (eta by default)
    n_pts_t : int, optional
        Number of points for t integration (default 400)
    **kwargs : dict
        Additional keyword arguments
    
    Returns
    -------
    float or array-like
        Differential cross section
    """
    return canonical_differential_cross_section(experiment, final_state, x, y, n_pts_t = 400)
    
@auto_integrate(force_canonical = True)
def canonical_differential_cross_section(experiment, final_state, log_gamma, eta, n_pts_t = 400, **kwargs):
    """
    Calculate canonical differential cross section for lepton-nucleus collisions.
    
    This function computes the differential cross section in canonical coordinates
    (log_gamma, eta) with automatic integration over the transfer momentum t.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration object
    final_state : FinalState
        Final state configuration object
    log_gamma : float or array-like
        Log of the boost factor gamma
    eta : float or array-like
        Pseudorapidity
    n_pts_t : int, optional
        Number of points for t integration (default 400)
    **kwargs : dict
        Additional keyword arguments
    
    Returns
    -------
    float or array-like
        Canonical differential cross section
    """
    #Experiment parameters 
    mi = experiment.lepton_mass
    E = experiment.E
    M = experiment.M
    form_factor_squared = experiment.form_factor_squared
    
    # Final-state parameters
    mf = final_state.lepton_mass
    m = final_state.boson_mass
    t_cut_off = final_state.t_cut_off
    PV_angle = final_state.PV_angle
    method = final_state.method
    
    #cross section
    dcrossx = scalar if final_state.boson_type == 'scalar' else vector

    # transfer momentum
    t_min = ((m + mf)**2 - mi**2)**2/(4*E**2)
    t_max = min(t_cut_off, (2*E*M/(E+M))**2)
    t = np.geomspace(t_min, t_max, n_pts_t) # want to sample small t more strongly
    
    if PV_angle is None:
        T = t.reshape(1, -1, 1, 1)
    else:
        T = t.reshape(-1, 1, 1)
    
    if method == 'exact':
        params = cross_section_parameters(experiment, final_state, log_gamma, eta, T)
        
    else:
        T_min, T_max = T_bounds(experiment, final_state, log_gamma, eta)
        params = cross_section_parameters(experiment, final_state, log_gamma, eta, T_min)
            
    #if PV_angle is none, this is 2xlen(EN)xlen(TH)xlen(t)
    dcx = dcrossx(params, method, PV_angle = PV_angle)
        
    if method == 'exact':
        dcx = np.trapezoid(form_factor_squared(T)/T**2 * dcx, x = t, axis = -3)
    else:
        if method == 'WW':
            region = (T > T_min)*(T < T_max) * (T_max > T_min)
            chi = np.trapezoid((T - T_min) * form_factor_squared(T)/T**2 * region, x = t, axis = -3)

        if method == 'IWW':
            region = (t > t_min)*(t < t_max)*(t_max > t_min)
            chi = np.trapezoid((t - t_min) * form_factor_squared(t)/t**2 * region, x = t)
        dcx = np.where(T_min == 0, 0, chi*dcx)
        
    return dcx.squeeze()

def cross_section_parameters(experiment, final_state, x, y, t, x_var = LOG_GAMMA, y_var = ETA):
    """
    Calculate kinematic parameters for cross section computation.
    
    This function computes all the kinematic variables needed for
    the cross section calculation, including energies, momenta, and angles.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration object
    final_state : FinalState
        Final state configuration object
    x : float or array-like
        Energetic variable
    y : float or array-like
        Angular variable
    t : float or array-like
        Transfer momentum squared
    x_var : Variable, optional
        Variable type for x (default LOG_GAMMA)
    y_var : Variable, optional
        Variable type for y (default ETA)
    
    Returns
    -------
    tuple
        All kinematic parameters needed for cross section calculation
    """
    mi = experiment.lepton_mass
    E = experiment.E
    p = experiment.p
    M = experiment.M
    
    mf = final_state.lepton_mass
    m = final_state.boson_mass

    Ek = ENERGY(x, var = x_var, context = {'final_state': final_state, 'particle': 'boson'})
    k = MOMENTUM(x, var = x_var, context = {'final_state': final_state, 'particle': 'boson'})
    
    sin_theta = SIN_THETA(y, var = y_var)
    sin_half_theta = SIN_HALF_THETA(y, var = y_var)
    
    V = np.sqrt((p - k)**2 + 4*p*k*sin_half_theta**2)
    u = ((E-p)  - (Ek-k))*((E+p) - (Ek+k)) - 4*p*k*sin_half_theta**2 - mf**2
 
    Q = np.sqrt(t + (t/(2*M))**2)
    cos_th_q = (u - (1 + (E-Ek)/M)*t)/(2*Q*V)
    cos_th_q = cos_th_q if final_state.method == 'exact' else np.sign(cos_th_q)
    sin_th_q = np.sqrt(1 - cos_th_q**2)
            
    params =  M, mf, mi, E, p, m, Ek, k, sin_theta, sin_half_theta, V, u, Q, cos_th_q, sin_th_q, t
    
    return params

def T_bounds(experiment, final_state, x, y, x_var = LOG_GAMMA, y_var = ETA):
    """
    Calculate bounds on transfer momentum T for approximation methods.
    
    This function computes the minimum and maximum values of the transfer
    momentum T for use in Weizsäcker-Williams (WW) and improved WW (IWW)
    approximation methods.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment configuration object
    final_state : FinalState
        Final state configuration object
    x : float or array-like
        First kinematic variable
    y : float or array-like
        Second kinematic variable
    x_var : Variable, optional
        Variable type for x (default LOG_GAMMA)
    y_var : Variable, optional
        Variable type for y (default ETA)
    
    Returns
    -------
    tuple
        (T_min, T_max) bounds on transfer momentum
    """
    
    #Experiment parameters 
    Ei = experiment.E
    pi = experiment.p
    M = experiment.M

    # Final-state parameters
    mf = final_state.lepton_mass


    # Relevant kinematic variables
    E = ENERGY(x, var = x_var, context = {'final_state': final_state, 'particle': 'boson'})
    k = MOMENTUM(x, var = x_var, context = {'final_state': final_state, 'particle': 'boson'})
    sin_half_theta = SIN_HALF_THETA(y, var = y_var)
    
    ### Compute T_min, T_max ###
    V = np.sqrt((pi - k)**2 + 4*pi*k*sin_half_theta**2)
    u = ((Ei-pi)  - (E-k))*((Ei+pi) - (E+k)) - 4*pi*k*sin_half_theta**2 - mf**2
    # First, we compute the (positive-root) radical that appears in the solution to 
    # the quadratic formula for t.
    radical = M * ((Ei-E+M)*u + 2*M*V**2 + V*np.sqrt(4*(M*V)**2 + u*(4*M*(Ei-E+M) + u)))
    
    # This is usually, but not always, T_max.
    T_1 = radical/((Ei-E+M)**2 - V**2)
    
    # This is usually, but not always, T_min. By computing it this way, we can avoid 
    # floating-point errors that would otherwise arise from computing T_2 in rational
    # form (i.e. where the radical is written in the numerator and is *conjugate* to 
    # "radical" defined above.)
    T_2 = (M*u)**2/radical

    # Given that we haven't imposed any kinematic constraints yet.
    T_min = np.nan_to_num(np.minimum(T_1, T_2)) #exact T_min
    T_max = np.nan_to_num(np.maximum(T_1, T_2)) #exact T_max
    
    T_min = T_min * (T_min > 0)
    T_max = T_max * (T_max > 0)
    
    return T_min, T_max

def scalar(params, method, PV_angle = None):
    """
    Calculate scalar boson differential cross section.
    
    This function computes the differential cross section for scalar boson
    production in lepton-nucleus collisions, including both parity-conserving
    (PC) and parity-violating (PV) contributions.
    
    Parameters
    ----------
    params : tuple
        Kinematic parameters (M, mf, mi, E, p, m, Ek, k, sin_theta, sin_half_theta, V, u, Q, cos_th_q, sin_th_q, t)
    method : str
        Calculation method: 'exact', 'WW', or 'IWW'
    PV_angle : float or None, optional
        Parity-violating angle (None for both PC and PV, 0 for PC only)
    
    Returns
    -------
    float or array-like
        Scalar boson differential cross section
    """

    M, mf, mi, E, p, m, Ek, k, sin_theta, sin_half_theta, V, u, Q, cos_th_q, sin_th_q, t = params
    
    if method == 'WW' or method == 'IWW':
        cos_th_q = np.sign(cos_th_q)
        sin_th_q = 0
       
    kinematically_allowed = (Ek > m) * (E - Ek - t/(2*M) > mf) * (cos_th_q**2 <= 1)

    #dot products
    qdp0 = cos_th_q*(Q/V)*p*((p-k) + 2*k*sin_half_theta**2)
    qdp1 = sin_th_q*(Q/V)*p*k*sin_theta
    
    s0 = -(1 + E/M)*t - 2*qdp0
    s1 = -2*qdp1
    
    if method == 'WW' or method == 'IWW':
        PC = -(s0 + u)**2/(s0*u)
        
        term = ((s0+u)*(1 + mi**2/s0 + mf**2/u)-m**2)/(s0*u)
        PC+= 2*(m**2 - (mi + mf)**2)*term
        
        #coeff = alpha**2/(4*np.pi)*k*sin_theta/(p*V)
        coeff = alpha**2/(4*np.pi)*k*Ek*sin_theta**2/(p*V)
        PC = np.where(kinematically_allowed, coeff*PC/(2*t), 0)

        # Only compute PV if necessary
        if PV_angle != 0:
            PV =  8*mf*mi*term
            PV = np.where(kinematically_allowed, coeff*PV/(2*t), 0)
        
            if not PV_angle:
                return np.stack((PC, PV))
            return PC + np.sin(PV_angle)**2*PV
        return PC

    I1 = np.where(s0**2 <= s1**2, 0, np.sign(s0)/np.sqrt(s0**2 - s1**2))
    I2 = s0*I1**3

    # Terms 1-3 are straightforward
    T1 = s0/u + 2 + u*I1
    T2 = - (2*M)*Ek/u + (2*M)*Ek*((2*M)*Ek-u)/u * I1  + T1/4
    T3 = 1/u**2 + 2/u * I1 + I2

    # Treat term 4 carefully to deal with floating point error
    P1_s1 = (2*M*(E-Ek) - 0.5*t)
    P0_P1_s01 = (2*M*E - t/2)*u
    T4 = (P1_s1**2 + (P0_P1_s01)*(2*P1_s1*I1 + (P0_P1_s01)*I2))/u**2
    
    P2 = 4*M**2 + t
    
    PC = P2*T1 - 4*t*T2 + (m**2 - (mf + mi)**2)*(P2 * t * T3 - 4*T4)
    
    #coeff = (alpha/M)**2/(32*np.pi)*k*sin_theta/(p*V)
    coeff = (alpha/M)**2/(32*np.pi)*Ek*k*sin_theta**2/(p*V)
    PC = np.where(kinematically_allowed, coeff*PC, 0)
    
    # Only compute PV if necessary
    if PV_angle != 0:
        PV = (4*mi*mf)*(P2 * t * T3 - 4*T4)
        PV = np.where(kinematically_allowed, coeff*PV, 0)
        if PV_angle is None:
            return np.stack((PC, PV))
        return PC + np.sin(PV_angle)**2 * PV
    return PC


def vector(params, method, PV_angle = None):
    """
    Calculate vector boson differential cross section.
    
    This function computes the differential cross section for vector boson
    production in lepton-nucleus collisions, including both parity-conserving
    (PC) and parity-violating (PV) contributions.
    
    Parameters
    ----------
    params : tuple
        Kinematic parameters (M, mf, mi, E, p, m, Ek, k, sin_theta, sin_half_theta, V, u, Q, cos_th_q, sin_th_q, t)
    method : str
        Calculation method: 'exact', 'WW', or 'IWW'
    PV_angle : float or None, optional
        Parity-violating angle (None for both PC and PV, 0 for PC only)
    
    Returns
    -------
    float or array-like
        Vector boson differential cross section
    """

    M, mf, mi, E, p, m, Ek, k, sin_theta, sin_half_theta, V, u, Q, cos_th_q, sin_th_q, t = params
    
    if method == 'WW' or method == 'IWW':
        cos_th_q = np.sign(cos_th_q)
        sin_th_q = 0
   
    kinematically_allowed = (Ek > m) * (E - Ek - t/(2*M) > mf) * (cos_th_q**2 <= 1)
    
    #dot products
    qdp0 = cos_th_q*(Q/V)*p*((p-k) + 2*k*sin_half_theta**2)
    qdp1 = sin_th_q*(Q/V)*p*k*sin_theta

    s0 = -(1 + E/M)*t - 2*qdp0
    s1 = -2*qdp1
    
    dm2 = (mi - mf)**2
    
    if method == 'WW' or method == 'IWW':
        T1 = (s0 + u)**2/(s0*u)
        
        PC = 4 - T1*(2 + dm2/m**2)    
        PC+= 2*(1-dm2/m**2)*(2*m**2 + (mi + mf)**2)/(s0*u) * ((s0+u)*(1 + mi**2/s0 + mf**2/u)-m**2)

        #coeff = alpha**2/(4*np.pi)*k*sin_theta/(p*V)
        coeff = alpha**2/(4*np.pi)*Ek*k*sin_theta**2/(p*V)
        PC = np.where(kinematically_allowed, coeff*PC/(2*t), 0)

        # Only compute PV if necessary
        if PV_angle != 0:
            PV = -4*mf*mi/m**2 * T1
            PV+= -24*mf*mi/(s0*u) * ((s0+u)*(1 + mi**2/s0 + mf**2/u)-m**2)
            PV = np.where(kinematically_allowed, coeff*PV/(2*t), 0)
            if PV_angle is None:
                return np.stack((PC, PV))
            return PC + np.sin(PV_angle)**2 * PV
        return PC

    I1 = np.where(s0**2 <= s1**2, 0, np.sign(s0)/np.sqrt(s0**2 - s1**2))
    I2 = s0*I1**3

    # Terms 1-3 are straightforward
    T1 = s0/u + 2 + u*I1
    T2 = - (2*M)*Ek/u + (2*M)*Ek*((2*M)*Ek-u)/u * I1  + T1/4
    T3 = 1/u**2 + 2/u * I1 + I2
    
    # Treat Term 4 carefully to deal with floating point error
    P1_s1 = (2*M*(E-Ek) - 0.5*t) #P1/s1
    P0_P1_s01 = (2*M*E - t/2)*u #
    T4 = (P1_s1**2 + (P0_P1_s01)*(2*P1_s1*I1 + (P0_P1_s01)*I2))/u**2

    # Term 5 + Term 6 together
    T56 = -(((4*M)*(E-Ek) + u) - t)*(1 + (t - (4*M)*E)*I1)/(2*u) + T2
    
    # Term 7
    T7 = (-1 + (2*(m**2-dm2) - t - u)*I1)/(2*u)
     
    P2 = 4*M**2 + t

    PC = ((2 + dm2/m**2)*T1 - 4)*P2 - (4*dm2/m**2)*t*T2
    PC+= -8*t*(T56+ T7*P2)
    PC+= (1-dm2/m**2)*(2*m**2 + (mi + mf)**2)*(P2*t*T3 - 4*T4)
        
    #coeff = (alpha/M)**2/(32*np.pi)*k*sin_theta/(p*V)
    coeff = (alpha/M)**2/(32*np.pi)*Ek*k*sin_theta**2/(p*V)
    
    PC = np.where(kinematically_allowed, coeff*PC, 0)
    
    # Only complete PV if necessary
    if PV_angle != 0:
        PV = T1*P2 - 4*t*T2 + (8*m**2) * P2 * (t/u) * I1
        PV+= -3*m**2 * (P2*t*T3 - 4*T4)
        PV*= 4*mi*mf/m**2
        PV = np.where(kinematically_allowed, coeff*PV, 0)
        if PV_angle is None:
            return np.stack((PC, PV))
        return PC + np.sin(PV_angle)**2*PV
    return PC
    
