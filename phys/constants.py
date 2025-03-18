#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#conversions
hc_mGeV = 1.973e-15
hc_cmGeV = 1.973e-14 #hbar c = 1.973e-14 cm GeV
hc_fmGeV = 0.1973 #hbar c = 0.1973 fm GeV

hc2_fbGeV2 = 3.894e11 #(hbar c)^2 = 3.894e11 fb GeV^2

#physics constants
e = 0.3028
alpha = e**2/(4*np.pi)

#leptons
me = 5.11e-4
mm = 0.106
mt = 1.77
ml = np.array([me, mm, mt])

#Higgs
vH = 242 #Higgs vev
mH = 125 #Higgs mass
H_width_SM = 4e-3 #GeV
crossx_H_prod = 57*1000 #fb, Higgs production cross-section at LHC
