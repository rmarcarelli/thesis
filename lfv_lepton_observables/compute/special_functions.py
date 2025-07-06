#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mpmath as mp
mp.mp.dps = 30
from scipy.special import spence

#-----------------------------------------------------#
#Special functions for evaluating dipole form factors
#-----------------------------------------------------#

#Li2(x) is just spence(1-x)
def Li2(x):
    # spence doesn't work with mp, so have to 
    # convert back to numpy
    x = np.complex128(x)
    return spence(1-x)

#Li2 but with branch conditioned on e
def DiLog(u, e):
    # np.where doesn't work with mp, so have to
    # convert back to numpy
    e = np.complex128(e)
    
    return np.where(e < 0, Li2(u), -0.5*mp.log(1/(1-u))**2-Li2(u/(u-1)))

#KallenLambda function
def KallenL(x, y, z):
    
    return x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z

#intermediary function for calculating C0
def DiLog_fn(x, z, e):
    
    argm = 2*e/(-1+x**2+z**2 - mp.sqrt(KallenL(1, x**2, z**2)))
    argp = 2*e/(-1+x**2+z**2 + mp.sqrt(KallenL(1, x**2, z**2)))
    
    return DiLog(argm, e) + DiLog(argp, -e)

#Special case of the ScalarC0 function as defined in PackageX
def C0(x,y,z):  
    
    Dxy = DiLog_fn(x, z, -1+z**2) - DiLog_fn(x, z, -1+x**2+z**2) 
    Dyx = DiLog_fn(y, z, -1+z**2) - DiLog_fn(y, z, -1+y**2+z**2)
    
    return (Dxy - Dyx)/(x**2 - y**2)

#Special case for DiscB function as defined in PackageX
def B(x, y):
    
    arg = (1-x**2+y**2 + mp.sqrt(KallenL(1, x**2, y**2)))/(2*y)
    
    return mp.sqrt(KallenL(1, x**2, y**2))*mp.log(arg)/x**2