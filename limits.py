#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#ALL limits can be computed in this file, assuming the data was properly generated

# These limits are valid for any leptonically-coupled particle with 
# hierarchy described in matrix form, i.e.
#
#            | gee   gem   get |
#      gll = | gme   gmm   gmt |
#            | gte   gtm   gtt |
#
# The only thing that matters here is the ratio of the couplings. Then, one .
#
# If index is unspecified... just picks the largest coupling.
#
# Alternatively, one could cast limits on a coupling gij with all other couplings
# fixed or zero (i.e., one can consider constraint on "get" when gll = 1e-3 for all
# diagonal couplings, and all other off-diagonal couplings are zero). The hierarchy
# approach is slightly easier to implement, because the branching-fractions depends
# only on the hierarchy so is independent of the over-all size of the couplings.This
# still allows one to set certain couplings to zero, and impose hierarchies between
# e.g. diagonal and off-diagonal components. 

def gij_limit():
    pass


#LFV DECAYS
def LFV_limit(i, j, g = [[1]*3]*3, th = [[0]*3]*3, d = [[0]*3]*3, ph = [[0]*3]*3, ALP = False):
    
    
    pass

#EDM AND MDM LIMITS


#EXPLANATIONS TO G-2
def g_2_explanation():
    pass

#LFV AT LEPTON NUCLEUS COLLIDERS
def EIC_limit(mass, g = None):
    
    #crossx = 
    
    pass

def MuSIC_limit(mass, g = None):
    pass

def MuBeD_limit(mass, g = None):
    pass

#HIGGS DECAY LIMITS (apply to ALPs only)
def Cah_CMS_limit():
    pass

def Cah_ATLAS_limit():
    pass

def Cah_MATHUSLA_limit():
    pass

def CMS_Cll_limit():
    pass

def Cll_limit():
    
    #find regions
    
    pass

#DARK GAUGE BOSONS
def existing_limits():
    pass


