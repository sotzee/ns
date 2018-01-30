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
# from tov_f import  MassRadius_transition
# a=EOS_BPSwithPolyCSS([0.059259259259259255, 10.0, 0.29600000000000004, 4899.580408690366, 0.5984, 5000.0, 1.1840000000000002, 13.769209715806836, 223.70874222311249, Fraction(1, 3)])
# output_maxmass=Maxmass_transition(1e-8,1e-4,a)
# # =============================================================================
# # print MassRadius_transition(1114.7628784179688,1e-8,1e-4,'MRBIT',a)
# # print MassRadius_transition(49.944557523691664,1e-8,1e-4,'MRBIT',a)
# # print MassRadius_transition(59.944557523691664,1e-8,1e-4,'MRBIT',a)
# # =============================================================================
# print MassRadius_transition(595.33359527587891,1e-8,1e-4,'B',a)
# print MassRadius_transition(13.769209715806836,1e-8,1e-4,'B',a)
# print MassRadius_transition(14.769209715806836,1e-8,1e-4,'B',a)
# print Properity_ofbindingmass(0.518664614281,14.769209715806836,595.33359527587891,MassRadius_transition,1e-8,1e-4,1,a)
# 
# =============================================================================

