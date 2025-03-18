#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from phys.constants import mH

######## Fit Functions ########

#-------------------------------------------------------------------------#
# ATLAS Fit
#-------------------------------------------------------------------------#

def line(p1, p2, x):
    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    y = m*(x - p1[0]) + p1[1]
    return y

def poly(x, coeff):
    n = len(coeff)
    result = np.array([coeff[i]*x**i for i in range(n)])
    return np.sum(result, axis=0)

def piecewise_fit(X,l00,l01,l02,l03,l10,l11,l12,l13,l20,l21,l22,l23,
                m40,m41,m42,m43,m50,m51,m52,m53,m60,m61,m62,m63,
                m70,m71,m72,m73,m80,m81,m82,m83,r00,r01,r02,r03,r10,
                r11,r12,r13,r20,r21,r22,r23):
    ma, ta = X
    #ma = 1/np.sqrt(mh**2/(4*mx**2) - 1)
    L = np.log10(0.2)
    R = np.log10(2)

    l0list = np.array([l00, l01, l02, l03])
    l1list = np.array([l10, l11, l12, l13])
    l2list = np.array([l20, l21, l22, l23])

    r0list = np.array([r00, r01, r02, r03])
    r1list = np.array([r10, r11, r12, r13])
    r2list = np.array([r20, r21, r22, r23])
    
    m4list = np.array([m40, m41, m42, m43])
    m5list = np.array([m50, m51, m52, m53])
    m6list = np.array([m60, m61, m62, m63])
    m7list = np.array([m70, m71, m72, m73])
    m8list = np.array([m80, m81, m82, m83])
    
    #matrix * (m0,m1,m2,m3) = f(L,R,li,ri,m4,m5,m6,m7,m8)
    #need to solve for (m0,m1,m2,m3) (Ax=b)
    A = np.array([[1,L,L**2,L**3],[1,R,R**2,R**3],[0,1,2*L,3*L**2],[0,1,2*R,3*R**2]])
    b = np.array([poly(L, np.array([l0list,l1list,l2list])) - L**4*poly(L, np.array([m4list,m5list,m6list,m7list,m8list])),
                  poly(R, np.array([r0list,r1list,r2list])) - R**4*poly(R, np.array([m4list,m5list,m6list,m7list,m8list])),
                  poly(L, np.array([l1list,2*l2list])) - L**3*poly(L, np.array([4*m4list,5*m5list,6*m6list,7*m7list,8*m8list])),
                  poly(R, np.array([r1list,2*r2list])) - R**3*poly(R, np.array([4*m4list,5*m5list,6*m6list,7*m7list,8*m8list]))])
    
    invA = np.linalg.inv(A)
    mlist = np.array([np.dot(invA, b[:,i]) for i in range(4)]).transpose()
    mlist = [mlist[0], mlist[1], mlist[2], mlist[3], m4list, m5list, m6list, m7list, m8list]
    
    lcoeff = np.array([poly(ma, l0list), poly(ma, l1list), poly(ma, l2list)])
    rcoeff = np.array([poly(ma, r0list), poly(ma, r1list), poly(ma, r2list)])
    mcoeff = np.array([poly(ma, m) for m in mlist])
    
    left = poly(ta, lcoeff)*np.heaviside(L-ta, 0.5)
    right = poly(ta, rcoeff)*np.heaviside(ta-R, 0.5)
    mid = poly(ta, mcoeff)*np.heaviside(ta-L,0.5)*np.heaviside(R-ta, 0.5)
    
    return left + mid + right

#fit params
pvar_lifetime = np.array([-3.42653307e+00,  1.86900312e-02, 3.72829127e-03,
                          -4.44380701e-05, -1.29553799e+00, 1.05753863e-01,
                          2.43183623e-03, -2.94984427e-05, 2.39397999e+00,
                          -1.20103456e-01,  7.83913519e-03, -7.30430401e-05,
                          4.10736193e+01, -4.37780937e+00,  1.22164268e-01,
                          -1.03248430e-03, 8.89076329e+01, -7.40608562e+00,
                          1.92638929e-01, -1.56223267e-03, -1.30892493e+02,
                          1.53428311e+01, -4.26772157e-01,  3.59743305e-03,
                          -4.37882985e+02,  4.36831518e+01, -1.19769084e+00,
                          1.00497374e-02, -2.73603892e+02,  2.62708891e+01,
                          -7.23858734e-01,  6.10939909e-03, -1.27768508e+00,
                          -2.13877857e-01,  7.87162155e-03, -7.00183829e-05,
                          2.80652716e+00, -1.81490537e-01,  1.89790933e-03,
                          7.56136804e-06, -1.17939649e+00,  1.58649289e-01,
                          -3.72801959e-03,  2.67759132e-05])


def ATLAS_fit(ma, ta):
    #if hasattr(ma, '__iter__'):
    #    return np.array([ATLAS_fit(m, ta) for m in ma])

    ma = np.array(ma).reshape(-1, 1)
    ta = np.array(ta)
    if len(ta.shape) == 1:
        ta = ta.reshape(1, -1)


    br = 10.0**piecewise_fit((ma, np.log10(ta)), *pvar_lifetime)
    

    slope = np.diff(np.log10(br), axis = 1)/np.diff(np.log10(ta), axis = 1)
    #slope = np.array([np.log10(br[:, i+1]/br[:,i])/np.log10(ta[:,i+1]/ta[:,i]) for i in range(len(ta) - 1)])
    #print(slope.shape)
    slope = np.where(np.isnan(slope), np.inf, slope)
    idx = np.nanargmin(abs(slope - 2), axis = 1)
    
    
    if ta.shape[0] == 1:
        p1 = (np.log10(ta[:,idx]).reshape(-1, 1),
              np.log10(br[np.arange(len(idx)), idx]).reshape(-1, 1))
        p2 = (np.log10(ta[:,idx+1]).reshape(-1, 1),
              np.log10(br[np.arange(len(idx)), idx+1]).reshape(-1, 1))
        
        logline = 10.0**line(p1, p2, np.log10(ta))
        
        ma_gtr = br
        ma_lsr = br*np.heaviside(ta[:,idx].reshape(-1, 1)-ta, 0.5)
        ma_lsr+= logline*np.heaviside(ta-ta[:,idx].reshape(-1, 1), 0.5)
        
        return (ma > 10)*ma_gtr + (ma <=10)*ma_lsr
    else:
        p1 = (np.log10(ta[np.arange(len(idx)),idx]).reshape(-1, 1),
              np.log10(br[np.arange(len(idx)), idx]).reshape(-1, 1))
        p2 = (np.log10(ta[np.arange(len(idx)),idx+1]).reshape(-1, 1),
              np.log10(br[np.arange(len(idx)), idx+1]).reshape(-1, 1))

    
        logline = 10.0**line(p1, p2, np.log10(ta))

        ma_gtr = br
        ma_lsr = br*np.heaviside(ta[np.arange(len(idx)),idx].reshape(-1, 1)-ta, 0.5)
        ma_lsr+= logline*np.heaviside(ta-ta[np.arange(len(idx)),idx].reshape(-1, 1), 0.5)
        
        return (ma > 10)*ma_gtr + (ma <=10)*ma_lsr
        
#-------------------------------------------------------------------------#
# MATHUSLA Fit
#-------------------------------------------------------------------------#

########MATHUSLA projections########
def Pdecay(x, L1, L2):
    return np.exp(-L1/x) - np.exp(-L2/x)

def Pdecay_MATH(ta, ma, L1, L2):
    b = mH/(2*ma)
    x = b*ta
    return 0.05*Pdecay(x, L1, L2)


def Nobs(ta, ma, L1, L2):
    Lum = 3000 #fb^-1
    crossx_Hsm = 57000 #fb
    return (Lum*crossx_Hsm)*2*Pdecay_MATH(ta,ma,L1,L2)

# This underestimates 
def MATH_approx(ta, ma, L1, L2):
    return 4/Nobs(ta, ma, L1, L2)

#only fits left half of MATHUSLA, right half given approximately by MATH_approx
def piecewise_fit_MATH(X, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33,
                       m40, m41, m42, m43, m50, m51, m52, m53, m60, m61, m62, m63):
    ma, ta = X
    b = mH/(2*ma)
    M = (200 - 180)/(b*np.log(200/180))
    
    A0 = MATH_approx(M, ma, 180, 200)

    m1list = np.array([m10, m11, m12, m13])
    m2list = np.array([m20, m21, m22, m23])
    m3list = np.array([m30, m31, m32, m33])
    m4list = np.array([m40, m41, m42, m43])
    m5list = np.array([m50, m51, m52, m53])
    m6list = np.array([m60, m61, m62, m63])
    
    mlist = np.array([m1list, m2list, m3list, m4list, m5list, m6list])
    mcoeff = np.array([poly(ma, m) for m in mlist])
    left = np.log10(A0) + (ta - np.log10(M))**2*poly((ta - np.log10(M))**2, mcoeff)

    return left

#fit params
pvar_MATH = np.array([ 3.51684719e-01,  7.29260819e-02, -8.32342227e-04,
                      -2.25553611e-06, 1.65052299e+00, -2.00571319e-01,
                      5.63402078e-03, -5.11227679e-05, -8.46464353e-01,
                      1.26063019e-01, -4.37469282e-03,  4.71727879e-05,
                      1.48586745e-01, -2.37232743e-02,  8.90459819e-04,
                      -1.01744668e-05, -1.07659506e-02,  1.79802106e-03,
                      -7.07570830e-05,  8.35961519e-07, 2.64249753e-04,
                      -4.64888163e-05,  1.89715115e-06, -2.29518272e-08])

def MATH_fit(ma, ta):
    #if hasattr(ma, '__iter__'):
    #    return np.array([MATH_fit(m, ta) for m in ma])
    
    ma = np.array(ma).reshape(-1, 1)
    ta = np.array(ta)
    if len(ta.shape) == 1:
        ta = ta.reshape(1, -1)
    
    b = mH/(2*ma)
    M = (200 - 180)/(b*np.log(200/180))
    left = 10.0**piecewise_fit_MATH((ma,np.log10(ta)), *pvar_MATH)
    right = MATH_approx(ta, ma, 180, 200)
    return left*np.heaviside(M - ta,0.5) + right*np.heaviside(ta - M,0.5)