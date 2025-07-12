#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some useful functions for computing limits and projections in the
other notebooks in this codebase.
"""

import numpy as np

#-------------------------------------------------------------------------#
# A reproduction of the analysis from Feldman and Cousins' "A Unified Approach
# to the Classical Statistical Analysis of Small Signals" (arxiv/physics/9711021).
#-------------------------------------------------------------------------#
from scipy.special import gammaln

def logP(n_observed, mean, background):
    """
    Calculate the log-likelihood for Poisson statistics.
    
    Implements the Feldman-Cousins unified approach for statistical analysis
    of small signals. This is the log-likelihood function used in the analysis.
    
    Args:
        n_observed (float): Number of observed events
        mean (float): Expected signal mean
        background (float): Expected background mean
        
    Returns:
        float: Log-likelihood value
    """
    return n_observed*np.log(mean+background + 1e-16) - (mean+background) - gammaln(n_observed+1)

#Would be nice to update to work for arbitrary background. Right now, works up to ~ 5000.
def find_n_interval(CL, mean, background):
    """
    Find the confidence interval for observed events given a signal mean.
    
    Uses the Feldman-Cousins method to determine the range of observed events
    that would be consistent with a given signal mean at the specified
    confidence level.
    
    Args:
        CL (float): Confidence level (0-1 or 0-100)
        mean (float): Expected signal mean
        background (float): Expected background mean
        
    Returns:
        tuple: (n_min, n_max) - Range of observed events
        
    Note:
        Currently works for backgrounds up to ~5000. For larger backgrounds,
        the search range may need to be adjusted.
    """
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
    """
    Calculate confidence interval for signal mean given observed events.
    
    Uses the Feldman-Cousins method to find the confidence interval for the
    signal mean, given the observed number of events and expected background.
    
    Args:
        CL (float): Confidence level (0-1)
        n_observed (float): Number of observed events
        background (float): Expected background mean
        tol (float, optional): Tolerance for convergence. Defaults to 0.01.
        
    Returns:
        tuple: (lower_mean, upper_mean) - Confidence interval for signal mean
    """
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

#-------------------------------------------------------------------------#
# Useful function for finding contours that are more complex than
# a simple maximum or minimum region.
#-------------------------------------------------------------------------#

from skimage import measure

def find_contours(x, y, Z, Z_val = 1):
    """
    Find contours in a 2D array and map them to coordinate space.
    
    Uses scikit-image's measure.find_contours to detect contours at a given
    threshold value, then maps the pixel coordinates back to the original
    coordinate system.
    
    Args:
        x (array): x-coordinates of the grid
        y (array): y-coordinates of the grid  
        Z (2D array): 2D array to find contours in
        Z_val (float, optional): Threshold value for contour detection. 
                                Defaults to 1.
        
    Returns:
        list: List of contour points in coordinate space. Each contour is
              returned as a 2D array with shape (2, n_points) where the
              first row contains x-coordinates and the second row contains
              y-coordinates.
    """
    contours = measure.find_contours(Z, Z_val)

    boundary_pts = []
    for cont in contours:
        i, j = cont.T[0], cont.T[1]  # Extract row/col indices
        x_coords = np.interp(i, np.arange(len(x)), x)  # Map cols to x
        y_coords = np.interp(j, np.arange(len(y)), y)  # Map rows to y
        boundary_pts.append(np.column_stack((x_coords, y_coords)).T)

    return boundary_pts
