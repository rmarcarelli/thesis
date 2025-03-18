#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import numpy as np

if __name__ == "__main__":
    
    masses = np.logspace(-3, np.log10(500), 200)
    masses = [f'{mass:.2e}' for mass in masses]
    
    #Computes the differential cross-section for e A -> tau x A at EIC, with x (pseudo)scalar
    params = {"Run name": "E137_full",
              "Collider card": "E137.txt",
              "Final lepton": ["e", "mu", "tau"],
              "Particle type": ["scalar", "vector"],
              "t_cut_off": [1e-2, 1.0],
              "method": ["exact", "WW", "IWW"],
              "Particle mass": masses}

    with open("run_cards/" + params["Run name"] + ".txt", "w") as file:
        for key in params.keys():
            if isinstance(params[key], list):
                file.write(str(key) + ': ' + ', '.join(map(str, params[key])) + '\n')
            else:
                file.write(str(key) + ': ' + str(params[key]) + '\n')
    
    
    
    
    params = {"Run name": "EIC_full",
              "Collider card": "EIC_Gold.txt",
              "Final lepton": ["e", "mu", "tau"],
              "Particle type": ["scalar", "vector"],
              "t_cut_off": [1e-2, 1.0],
              "method": ["exact", "WW", "IWW"],
              "Particle mass": masses}
    
    with open("run_cards/" + params["Run name"] + ".txt", "w") as file:
        for key in params.keys():
            if isinstance(params[key], list):
                file.write(str(key) + ': ' + ', '.join(map(str, params[key])) + '\n')
            else:
                file.write(str(key) + ': ' + str(params[key]) + '\n')
                
    params = {"Run name": "MuCol_full",
              "Collider card": "MuCol.txt",
              "Final lepton": ["e", "mu", "tau"],
              "Particle type": ["scalar", "vector"],
              "t_cut_off": [1e-2, 1.0],
              "method": ["exact", "WW", "IWW"],
              "Particle mass": masses}
    
    with open("run_cards/" + params["Run name"] + ".txt", "w") as file:
        for key in params.keys():
            if isinstance(params[key], list):
                file.write(str(key) + ': ' + ', '.join(map(str, params[key])) + '\n')
            else:
                file.write(str(key) + ': ' + str(params[key]) + '\n')
                
    params = {"Run name": "MuSIC_full",
              "Collider card": "MuSIC.txt",
              "Final lepton": ["e", "mu", "tau"],
              "Particle type": ["scalar", "vector"],
              "t_cut_off": [1e-2, 1.0],
              "method": ["exact", "WW", "IWW"],
              "Particle mass": masses}

    with open("run_cards/" + params["Run name"] + ".txt", "w") as file:
        for key in params.keys():
            if isinstance(params[key], list):
                file.write(str(key) + ': ' + ', '.join(map(str, params[key])) + '\n')
            else:
                file.write(str(key) + ': ' + str(params[key]) + '\n')