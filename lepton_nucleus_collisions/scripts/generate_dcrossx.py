#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Run this script in thesis folder with the following command:
    
python -m lepton_nucleus_collisions.scripts.generate_dcrossx

Python script which computes and saves the energy-angle differential 
cross-section for l A -> l' A phi as an hdf5 file. For full analysis, 
the run files chosen perform a complete scan over all parameters for 
each collider. 
    
Final-state lepton: e, mu, or tau

This choice determines whether the interaction is lepton-flavor-conserving or
lepton-flavor-violating, and if it is lepton-flavor-violating, which lepton
conversion occurs.

Final-state particle type:  scalar or vector

This choice determines whether the final-state particle produced is a scalar
or a vector.

t_cut_off: 0.01 or 1.0 (in GeV^2)
    
This choice determines when to cut-off the form-factor. Essentially, 0.01 GeV^2
cuts off photons with wavelengths smaller than the size of the nucleus, so only
accounts for coherent interactions with protons, whereas 1.0 GeV^2 cuts off
photons with wavelengths smaller than the size of the nucleons. In principle,
one could continue beyond this, but would need to use PDFs to account for the 
interaction of the photon with the quarks. 

method: 'exact', 'WW', or 'IWW'

This choice determines which method is used to integrate the cross-section.

The "exact" method numerically integrates the 2-> 3 cross-section, with 
the only approximation being the courseness of the lattice which is integrated
over.

The "WW" method uses the Weizsacker-Williams approximation, which 
essentially treats the nucleus as a source of off-shell photons. Then, the
cross-section is approximated as the effective photon flux times the 2->2
2->2 photon-lepton cross-section with photon momentum q^2 = -t_min, where 
t_min is the minimum possible kinematically allowed photon transfer for the 
process. 

The "IWW" method is the "Improved" Weizsacker-Williams approximation, where 
"improved" refers to a time improvement rather than an improvement in the 
accuracy. Here, we just take it to represent the approximation that t_min is 
independent of the kinematics of the final-state particles, which in turn 
makes the effective photon flux a singular number rather than a quantity
which depends on the kinematics of the final-state particles. Usually, this
approximation is coupled with an approximation over the angular integral, so
that only the differential cross-section w.r.t. the final state energy/energy
fraction of the scalar remains. Given that we are interested in producing
full energy-angle distributions, we do not .

Final-state particle mass: Sweep from 10^-3 GeV to 500 GeV



"""
from lepton_nucleus_collisions.utils.compute import compute_energy_angle_differential_crossx

run_names = ["E137.txt", "EIC_Gold.txt", "MuCol.txt","MuSIC.txt"]

for run_name in run_names:
    print('\n------------' + run_name + '-------------\n')
    compute_energy_angle_differential_crossx(run_name, save = True)
    print('\n')
