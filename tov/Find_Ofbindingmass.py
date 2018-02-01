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
# a=EOS_BPSwithPolyCSS([0.059259259259259255, 10.0, 0.29600000000000004, 175.05700209813543, 0.5984, 5000.0, 1.1840000000000002, 81.61498460032989, 360.08183812497651, Fraction(1, 1)]
# )
# output_maxmass=Maxmass_transition(1e-8,1e-4,a)
# print output_maxmass
# # =============================================================================
# # print MassRadius_transition(1114.7628784179688,1e-8,1e-4,'MRBIT',a)
# # print MassRadius_transition(49.944557523691664,1e-8,1e-4,'MRBIT',a)
# # print MassRadius_transition(59.944557523691664,1e-8,1e-4,'MRBIT',a)
# # =============================================================================
# Baryon_right=MassRadius_transition(output_maxmass[5],1e-8,1e-4,'B',a)
# output_right=[output_maxmass[5]]+MassRadius_transition(output_maxmass[5],1e-8,1e-4,'MRBIT',a)
# output_after_peak=Properity_ofbindingmass(Baryon_right,output_maxmass[5]+1,output_maxmass[3],MassRadius_transition,1e-8,1e-4,1,a)
# print output_after_peak[0],output_after_peak[1],output_after_peak[4]
# print output_right[0],output_right[1],output_right[4]
# 
# =============================================================================

