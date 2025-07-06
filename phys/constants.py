#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains some useful physical constants 
that are used throughout this code base. 

"""

import numpy as np

#unit conversions
hc_mGeV = 1.973e-16 #hbar*c = 1.973e-16 m GeV
hc_cmGeV = 1.973e-14 # hbar*c = 1.973e-14 cm GeV
hc_fmGeV = 0.1973 # hbar*c = 0.1973 fm GeV

hc2_fbGeV2 = 3.894e11 #(hbar c)^2 = 3.894e11 fb GeV^2
cm2_fb = 1e39 # 1 cm^2 = 10^39 fb

#physics constants
e = 0.3028 # electric charge in natural units
alpha = e**2/(4*np.pi) # fine structure constant in natural units

#leptons
me = 5.11e-4 # electron mass (GeV)
mm = 0.106 # muon mass (GeV)
mt = 1.77 # tau mass (GeV)
leptons = ['e', 'mu', 'tau'] # list of leptons
ml = np.array([me, mm, mt]) # list of lepton masses
lepton_masses = dict(zip(leptons, ml)) # dictonary of lepton masses

#nucleons
mp = 0.938 # proton mass (GeV)
mn = 0.939 # neutron mass (GeV)

#Higgs
vH = 242 # Higgs vev (GeV)
mH = 125 # Higgs mass (GeV)
H_width_SM = 4e-3 # SM Higgs width (GeV)
crossx_H_prod = 57*1000 # Higgs production cross-section at LHC (fb)
