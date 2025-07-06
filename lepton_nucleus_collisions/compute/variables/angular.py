import numpy as np

from .base import Variable, IDENTITY

########## ANGULAR VARIABLES ###########

# ETA

ETA = Variable(['eta', 'pseudorapidity'],
               IDENTITY, 
               IDENTITY,
               name = 'ETA')

# THETA

def THETA_to_ETA(theta, *, context = None, include_jacobian = True):
    if include_jacobian:
        return -np.log(np.tan(theta/2)), -1/np.sin(theta)
    return -np.log(np.tan(theta/2))
    
def ETA_to_THETA(eta, *, context = None, include_jacobian = True):
    if include_jacobian:
        return 2*np.arctan(np.exp(-eta)), -1/np.cosh(eta)
    return 2*np.arctan(np.exp(-eta))

THETA = Variable(['th', 'theta', 'angle'],
                 THETA_to_ETA,
                 ETA_to_THETA,
                 ETA,
                 name = 'THETA')

# SIN(THETA)

def SIN_THETA_to_ETA(sin_theta, *, context = None, include_jacobian = True):
    
    sign = context.get('sign', 1)
    
    if include_jacobian:
        return sign * np.arccosh(1/sin_theta), -sign/(sin_theta * np.sqrt(1-sin_theta**2))
    return sign * np.arccosh(1/sin_theta)

def ETA_to_SIN_THETA(eta, *, context = None, include_jacobian = True):

    if include_jacobian:
        return  1/np.cosh(eta), -np.tanh(eta)/np.cosh(eta)
    return 1/np.cosh(eta)

SIN_THETA = Variable(['sin', 'sinth', 'sintheta','sinangle','sine', 'sineth', 'sinetheta','sineangle'],
                     SIN_THETA_to_ETA,
                     ETA_to_SIN_THETA,
                     ETA,
                     name = 'SIN_THETA')

# SIN(THETA/2)

def SIN_HALF_THETA_to_ETA(sin_half_theta, *, context = None, include_jacobian = True):
    
    if include_jacobian:
        return np.log(np.sqrt(1/sin_half_theta**2 - 1)), -1/(sin_half_theta * (1 - sin_half_theta**2))
    return np.log(np.sqrt(1/sin_half_theta**2 - 1))
        
def ETA_to_SIN_HALF_THETA(eta, *, context = None, include_jacobian = True):
    if include_jacobian:
        return 1/np.sqrt(1+np.exp(2*eta)), -np.exp(2*eta)/(1 + np.exp(2*eta))**(3/2)
    return 1/np.sqrt(1+np.exp(2*eta))
        
SIN_HALF_THETA = Variable(['sinhalf', 'sinhalfth', 'sinhalftheta','sinhalfangle','sinehalf', 'sinehalfth', 'sinehalftheta','sinehalfangle'],
                          SIN_HALF_THETA_to_ETA,
                          ETA_to_SIN_HALF_THETA,
                          ETA,
                          name = 'SIN_HALF_THETA')

# COS(THETA)

def COS_THETA_to_ETA(cos_theta, *, context = None, include_jacobian = True):
    if include_jacobian:
        return np.arctanh(cos_theta), 1/(1-cos_theta**2)
    return np.arctanh(cos_theta)

def ETA_to_COS_THETA(eta, *, context = None, include_jacobian = True):
    if include_jacobian:
        return np.tanh(eta), 1/np.cosh(eta)**2
    return np.tanh(eta)
    
COS_THETA = Variable(['cos', 'costh', 'costheta','cosangle','cosine', 'cosineth', 'cosinetheta','cosineangle'],
                     COS_THETA_to_ETA,
                     ETA_to_COS_THETA,
                     ETA,
                     name = 'COS_THETA')