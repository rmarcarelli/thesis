"""


"""

import numpy as np

from .variables import LOG_GAMMA, MOMENTUM, ETA, SIN_THETA, SIN_HALF_THETA

def TRANSFORM(x, y, x_in = LOG_GAMMA, y_in = ETA, x_out = LOG_GAMMA, y_out = ETA, include_jacobian = True, in_context = None, out_context = None):
    """
    
    """
    
    assert 'experiment' in in_context and 'experiment' in out_context
    assert in_context['experiment'] == out_context['experiment']
    experiment = in_context['experiment']

    # transform specified initial variables (x_in, y_in) to canonical variables (log_gamma, eta)
    log_gamma, x_jacob = LOG_GAMMA.transform(x, var = x_in, context = in_context)
    eta, y_jacob = ETA.transform(y, var = y_in, context = in_context)
    jacob = x_jacob * y_jacob #update jacobian

    # initial and final-state frame (default always in ion frame)
    frame_in = in_context.get('frame', 'ion')
    frame_out = out_context.get('frame', 'ion')
    
    # final-state particle before and after transformation (default is 'boson')
    particle_in = in_context.get('particle', 'boson')
    particle_out = out_context.get('particle', 'boson')
    
    frame_transform = frame_in != frame_out
    particle_transform = particle_in != particle_out

    
    # assign the velocity between frames
    # (by code convention, ion is moving to the left in 'lab' frame,
    # so one must boost to right (positive u) to go from 'lab' to 'ion')
    if frame_in == frame_out:
        u = 0
    elif frame_in == 'lab':
        u = experiment.v_nuc
    else:
        u = -experiment.v_nuc
    
    if particle_transform:
        if frame_in == 'lab':
            log_gamma, eta, frame_jacob = CANONICAL_FRAME_TRANSFORM(log_gamma, eta, experiment.v_nuc)
            jacob = jacob * frame_jacob
            # now in ion frame
        log_gamma, eta, particle_jacob = CANONICAL_PARTICLE_TRANSFORM(log_gamma, eta, in_context = in_context, out_context = out_context)
        jacob = jacob * particle_jacob # update jacobian
        if frame_out == 'lab':
            log_gamma, eta, frame_jacob = CANONICAL_FRAME_TRANSFORM(log_gamma, eta, -experiment.v_nuc)
            jacob = jacob * frame_jacob            
    
    elif frame_transform:
            log_gamma, eta, frame_jacob = CANONICAL_FRAME_TRANSFORM(log_gamma, eta, u)
            jacob = jacob * frame_jacob
            
    #  transform final canonical variables (log_gamma, eta) to specified final variables (x_out, y_out)
    x, x_jacob = LOG_GAMMA.inverse_transform(log_gamma, var = x_out, context = out_context)
    y, y_jacob = ETA.inverse_transform(eta, var = y_out, context = out_context)
    jacob = jacob * x_jacob * y_jacob #update jacobian
            
    if include_jacobian:
        return x, y, jacob
    return x,y
            



    # if frames are different, convert canonical variables (log_gamma, eta) from frame_in to frame_out
    if frame_in != frame_out:
        log_gamma, eta, frame_jacob = CANONICAL_FRAME_TRANSFORM(log_gamma, eta, u)
        jacob = jacob * frame_jacob # update jacobian

    # if particles are different, convert canonical variables (log_gamma, eta) from particle_in to particle_out
    if particle_in != particle_out:
        
        # we have now transformed frames
        in_context['frame'] = out_context['frame']
        
        log_gamma, eta, particle_jacob = CANONICAL_PARTICLE_TRANSFORM(log_gamma, eta, in_context = in_context, out_context = out_context)
        jacob = jacob * particle_jacob # update jacobian


    #  transform final canonical variables (log_gamma, eta) to specified final variables (x_out, y_out)
    x, x_jacob = LOG_GAMMA.inverse_transform(log_gamma, var = x_out, context = out_context)
    y, y_jacob = ETA.inverse_transform(eta, var = y_out, context = out_context)
    jacob = jacob * x_jacob * y_jacob #update jacobian

    if include_jacobian:
        return x, y, jacob
    return x,y

def CANONICAL_FRAME_TRANSFORM(log_gamma, eta, u, include_jacobian = True):
    """
    
    
    To compute the jacobian, we use the fact that 
    d(E',theta')/d(E, theta) = 1
    so
    d(log_gamma', eta')/d(log_gamma, eta)
    = [d(log_gamma')/d(E') d(E)/d(log_gamma)]*1*[d(eta')/d(theta') d(theta)/d(eta)]
    = [gamma/gamma'] * [cosh(eta')/cosh(eta)]
    = exp(log(gamma) - log(gamma')) * cosh(eta')/cosh(eta)
    
    """

    # if speed between frames is zero, just return log_gamma and eta
    if u == 0:
        if include_jacobian:
            return log_gamma, eta, 1
        return log_gamma, eta

    # compute frame boost / lorentz factor
    gamma_u = 1/np.sqrt(1 - u**2)

    #velocity of final-state particle
    v = np.sqrt(1-np.exp(-2*log_gamma)) 
    
    
    # compute useful angular variables for transforming eta between frames
    sin_theta = SIN_THETA(eta)
    sin_half_theta = SIN_HALF_THETA(eta)
    
    jacob = 1/np.cosh(eta) #d(theta)/d(eta)
    
    # eta of final-state particle in new frame
    cot = (gamma_u *((1 + u/v) - 2*sin_half_theta**2))/sin_theta
    eta = np.arcsinh(cot)
    
    jacob = jacob * np.cosh(eta) #d(eta)/d(theta)
    
    factor = gamma_u * ((1 + u * v) - 2 * u * v * sin_half_theta**2)
    jacob = jacob/factor #d(log_gamma)/d(E) * d(E)/d(log_gamma)
    
    # log_gamma of final-state particle in new frame
    log_gamma = log_gamma + np.log(factor)

    if include_jacobian:
        return log_gamma, eta, jacob
    return log_gamma, eta

# ONLY TRANSFORMS IN ION FRAME
def CANONICAL_PARTICLE_TRANSFORM(log_gamma, eta, include_jacobian = True, in_context = None, out_context = None):
    """
    
    """
    
    assert 'experiment' in in_context and 'experiment' in out_context
    assert in_context['experiment'] == out_context['experiment']
    experiment = in_context['experiment']
    
    assert 'final_state' in in_context and 'final_state' in out_context
    assert in_context['final_state'] == out_context['final_state']
    final_state = in_context['final_state']
    
    assert 'particle' in in_context and 'particle' in out_context
    if in_context['particle'] == out_context['particle']:
        return log_gamma, eta, 1
    
    mi = experiment.lepton_mass #initial-state lepton mass
    m = final_state.boson_mass #final-state boson mass
    mf = final_state.lepton_mass #final-state lepton mass
    E = experiment.E
    p = experiment.p
    M = experiment.M
    
    # input

    # energy and momentum
    p_in = MOMENTUM(log_gamma, **in_context)
    
    # photon transfer variables
    t_min = ((m+mf)**2 - mi**2)**2 / (4*E**2)
    
    q0 = -t_min/(4*M)
    Q = np.sqrt(t_min+q0**2)
    
    p0 = p + Q
    p_out = np.sqrt(p0**2 + p_in**2 - 2*p0 * p_in * np.tanh(eta))
    
    eta = np.log(np.abs(-np.sinh(eta) + np.cosh(eta)*(p0 + p_out)/p_in))     

    log_gamma = LOG_GAMMA(p_out, var = MOMENTUM, **out_context)
    
    # dp/dlogg dlogg'/dp'
    #jacobian = np.exp(2 * (log_gamma - log_gamma_out)) * np.sqrt((1+np.exp(2*log_gamma_out))/(1+np.exp(2*log_gamma)))
    m_in = m if in_context['particle'] == 'boson' else mf
    m_out = m if out_context['particle'] == 'boson' else mf
    jacobian = (p_in**2 - m_in**2)/(p_out**2 - m_out**2) * p_out/p_in
    
    return log_gamma, eta, np.nan_to_num(jacobian)

    