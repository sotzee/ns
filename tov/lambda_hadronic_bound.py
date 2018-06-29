#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:14:27 2018

@author: sotzee
"""
from eos_class import EOS_BPSwithPoly
import numpy as np
from tov_f import MassRadius,Radius_correction_ratio
#from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass
Preset_Pressure_final=1e-8
Preset_rtol=1e-4

args_upper_bound_no_maxmass=[0.059259259259259255,
 30.0,
 0.29600000000000004,
 305.8554783223734,
 0.5984,
 1930.4,
 1.1840000000000002]
eos_upper_bound_no_maxmass=EOS_BPSwithPoly(args_upper_bound_no_maxmass)
pc_upper_bound_no_maxmass=Maxmass(Preset_Pressure_final,Preset_rtol,eos_upper_bound_no_maxmass)[0]

args_lower_bound=[0.059259259259259255,
 8.4,
 0.29600000000000004,
 106.77589359156613,
 0.5984,
 737.29507167870486,
 1.1840000000000002]
eos_lower_bound=EOS_BPSwithPoly(args_lower_bound)
pc_lower_bound=Maxmass(Preset_Pressure_final,Preset_rtol,eos_lower_bound)[0]

pc_list=10**np.linspace(0,-1.5,50)

beta=[[],[]]
Lambda=[[],[]]
mass=[[],[]]
# =============================================================================
# for i in range(len(pc_list)):
#     print i
#     result_upper_bound_no_maxmass=MassRadius(pc_upper_bound_no_maxmass*pc_list[i],Preset_Pressure_final,Preset_rtol,'MRBIT',eos_upper_bound_no_maxmass)
#     mass[1].append(result_upper_bound_no_maxmass[0])
#     Lambda[1].append(result_upper_bound_no_maxmass[6])
#     beta[1].append(result_upper_bound_no_maxmass[2]*Radius_correction_ratio(pc_upper_bound_no_maxmass*pc_list[i],Preset_Pressure_final,result_upper_bound_no_maxmass[2],eos_upper_bound_no_maxmass))
#     result_lower_bound=MassRadius(pc_lower_bound*pc_list[i],Preset_Pressure_final,Preset_rtol,'MRBIT',eos_lower_bound)
#     mass[0].append(result_lower_bound[0])
#     Lambda[0].append(result_lower_bound[6])
#     beta[0].append(result_lower_bound[2]*Radius_correction_ratio(pc_lower_bound*pc_list[i],Preset_Pressure_final,result_lower_bound[2],eos_lower_bound))
# 
# =============================================================================
