# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:32:52 2017

@author: sotzee
"""

#tov integral over log-p is used to calculate mass, momentum of initial and tidal
Preset_Pressure_final = 1e-8
Preset_Pressure_final_MR = 1e-8
start_from = 0 #start_from=0 for calcuate all the parameter space

#tov integral over r is used to calculate radius only, since it uses build in RK45 method which cost time
TurnOn_radius_onepointfour = True
Preset_pressure_center_low = 1.0         # this parameter is only for setting boundary for searching 1.4 solar mass stars
Preset_Pressure_final_index = 1          # equal 2 means Preset_Pressure_final for solving radius is Preset_Pressure_final**2, which include more low density area
number_per_parameter = 5
concentration = 2

baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.74*0.16
baryon_density3 = 7.4*0.16

#EoS parameter for Hybrid star
from fractions import Fraction
pressure1=10.0 #Mevfm-3
Preset_gamma2 = [121,2.0,10.0] #[num,low,high]polytropic index between pressure1 and pressure_trans.
pressure3=5000.
Preset_num_pressure_trans=100 #Mevfm-4
Preset_det_density = [61,0.0,1.0]  #{det_density/density_trans}[num,low,high]reduced density change at density_trans.
cs2=Fraction('3/3') #sound speed square


#EoS parameter for Hadronic star
Preset_pressure1=[13*5+1,7.,20.] ##[num,lower_bound,up_bound] 
Preset_pressure2=[50,100.] #[num,lower_bound] 
Preset_pressure3=[70,150.] #[num,lower_bound] 
