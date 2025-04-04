#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Just some random useful functions.
"""

import numpy as np
from scipy.special import gammaln

#-------------------------------------------------------------------------#
# A reproduction of the analysis from Feldman and Cousins' "A Unified Approach
# to the Classical Statistical Analysis of Small Signals" (arxiv/physics/9711021).
#-------------------------------------------------------------------------#

def logP(n_observed, mean, background):
    return n_observed*np.log(mean+background + 1e-16) - (mean+background) - gammaln(n_observed+1)

#Would be nice to update to work for arbitrary background. Right now, works up to ~ 5000.

def find_n_interval(CL, mean, background):
    if CL > 1:
        CL = CL/100

    n_observed = np.arange(max(background-5000, 0), background + 5000) #5000 is arbitrary
    best_mean = np.maximum(n_observed - background, 0)

    logp_best = logP(n_observed, best_mean, background)
    logp = logP(n_observed, mean, background)
    logR = logp - logp_best
        
    logR_sorted, logp_sorted, logp_best_sorted, n_sorted = np.array(list(zip(*reversed(sorted(zip(logR, logp, logp_best, n_observed))))))
    
    idx = np.argmax(np.cumsum(np.exp(logp_sorted)) >= CL)
    n_range = n_sorted[:idx+1]
    return n_range.min(), n_range.max()

def poisson_confidence_interval(CL, n_observed, background, tol = 0.01):
    best_mean = np.maximum(n_observed - background, 0)
    lower_mean = best_mean
    upper_mean = best_mean

    lower_mean_min = 0
    lower_mean_max = lower_mean
    while np.abs(lower_mean_max - lower_mean_min) > tol/2:
        lower, upper = find_n_interval(CL, lower_mean, background)
        
        if lower <= n_observed and upper >= n_observed:
            lower_mean_max = lower_mean
            lower_mean = (lower_mean_min + lower_mean)/2
        else:
            lower_mean_min = lower_mean
            lower_mean = (lower_mean_max + lower_mean)/2

    upper_mean_min = upper_mean
    upper_mean_max = 2000 #just a big number
    
    while np.abs(upper_mean_max - upper_mean_min) > tol/2:
        lower, upper = find_n_interval(CL, upper_mean, background)
        
        if lower <= n_observed and upper >= n_observed:
            upper_mean_min = upper_mean
            upper_mean = (upper_mean_max + upper_mean)/2
        else:
            upper_mean_max = upper_mean
            upper_mean = (upper_mean_min + upper_mean)/2

    return lower_mean, upper_mean