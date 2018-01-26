#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:31:36 2018

@author: sotzee
"""

import scipy.optimize as opt

def Radius_ofbindingmass(ofbindingmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos):
    pressure_center=pressure_center_ofbindingmass(ofbindingmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos)
    radiusmass=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MR',eos)
    return [pressure_center,radiusmass[0],radiusmass[1]]

def Properity_ofbindingmass(ofbindingmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos):
    pressure_center=pressure_center_ofbindingmass(ofbindingmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos)
    [M,R,beta,M_binding,momentofinertia,yR,tidal]=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MRBIT',eos)
    return [pressure_center,M,R,beta,M_binding,momentofinertia,yR,tidal]

def pressure_center_ofbindingmass(ofbindingmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    result=opt.brenth(Ofbindingmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofbindingmass,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos))
    return result

def Ofbindingmass(pressure_center,ofbindingmass,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    return -ofbindingmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'B',eos)

# =============================================================================
# from eos_class import EOS_BPSwithPolyCSS
# from fractions import Fraction
# from FindMaxmass import Maxmass_transition
# a=EOS_BPSwithPolyCSS([0.059259259259259255, 16.0, 0.29600000000000004, 269.13890217871915, 0.5984, 5000.0, 1.1840000000000002, 196.49744783369798, 278.56127496744625, Fraction(1, 1)])
# print Maxmass_transition(1e-8,1e-4,a)
# =============================================================================
