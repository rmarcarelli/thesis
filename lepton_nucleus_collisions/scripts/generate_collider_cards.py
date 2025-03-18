#!/usr/bin/env python3
# -*- coding: utf-8 -*-


if __name__ == '__main__':

    #E137
    params = {"Collider name": "E137", 
              "Initial lepton": "e",
              "Lepton energy": 20, #GeV
              "Nucleus name": "Aluminum",
              "A" : 26,
              "Z": 13,
              "M": 24.2,
              "Nucleus energy": 24.2}
    
    with open("collider_cards/" + params["Collider name"] + ".txt", "w") as file:
        for key in params.keys():
            file.write(str(key) + ": " + str(params[key]) + '\n')
        
    
    #EIC
    params = {"Collider name": "EIC_Gold", 
              "Initial lepton": "e",
              "Lepton energy": 18, #GeV
              "Nucleus name": "Gold",
              "A" : 197,
              "Z": 79,
              "M": 183,
              "Nucleus energy": 110*197}
    
    with open("collider_cards/" + params["Collider name"] + ".txt", "w") as file:
        for key in params.keys():
            file.write(str(key) + ": " + str(params[key]) + '\n')    
    
    #MuCol
    params = {"Collider name": "MuCol", 
              "Initial lepton": "mu",
              "Lepton energy": 1000, #GeV
              "Nucleus name": "Lead",
              "A" : 208,
              "Z": 82,
              "M": 193,
              "Nucleus energy": 193} #same as M, because it is at rest
    
    with open("collider_cards/" + params["Collider name"] + ".txt", "w") as file:
        for key in params.keys():
            file.write(str(key) + ": " + str(params[key]) + '\n')
    
    #MuSIC
    params = {"Collider name": "MuSIC", 
              "Initial lepton": "mu",
              "Lepton energy": 1000, #GeV
              "Nucleus name": "Gold",
              "A" : 197,
              "Z": 79,
              "M": 183,
              "Nucleus energy": 110*197}
    
    with open("collider_cards/" + params["Collider name"] + ".txt", "w") as file:
        for key in params.keys():
            file.write(str(key) + ": " + str(params[key]) + '\n')