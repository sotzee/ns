# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:58:14 2016

@author: Sotzee
"""


import scipy.optimize as opt

def Radius_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos):
    pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos)
    radiusmass=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MR',eos)
    return [pressure_center,radiusmass[0],radiusmass[1]]

def Properity_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos):
    pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos)
    [M,R,beta,M_binding,momentofinertia,yR,tidal]=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MRBIT',eos)
    return [pressure_center,M,R,beta,M_binding,momentofinertia,yR,tidal]

def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos))
    return result

def Ofmass(pressure_center,ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    return -ofmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'M',eos)


def Properity_ofmass_two_peak(ofmass,Preset_pressure_center_low,pressure_center_left,pressure_center_after_peak,pressure_center_right,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos,f_log_name):
    mass_left = MassRadius_function(pressure_center_left,Preset_Pressure_final,Preset_rtol,'M',eos)
    mass_after_peak = MassRadius_function(pressure_center_after_peak,Preset_Pressure_final,Preset_rtol,'M',eos)
    mass_right = MassRadius_function(pressure_center_right,Preset_Pressure_final,Preset_rtol,'M',eos)
    
    if(mass_right>ofmass):
        processOutput_onepointfour = Properity_ofmass(ofmass,Preset_pressure_center_low,pressure_center_right,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos)
    else:
        processOutput_onepointfour = [0,0,0,0,0,0,0,0]
    
    if(mass_left>ofmass>mass_after_peak):
        processOutput_onepointfour_quark = Properity_ofmass(ofmass,pressure_center_after_peak,pressure_center_left,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos)
    else:
        processOutput_onepointfour_quark = [0,0,0,0,0,0,0,0]
    
    if(mass_after_peak>ofmass>mass_right): #not physical, happend due to numerical error.
        processOutput_onepointfour = [0,0,0,0,0,0,0,0]
        processOutput_onepointfour_quark = [pressure_center_after_peak]+MassRadius_function(pressure_center_after_peak,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MRBIT',eos)

    check_error_1=processOutput_onepointfour[0]+processOutput_onepointfour_quark[0]==0 and (mass_left>ofmass or mass_right>ofmass)
    check_error_2=0.99*ofmass>processOutput_onepointfour_quark[1]>0 or processOutput_onepointfour_quark[1]>1.01*ofmass
    check_error_3=0.99*ofmass>processOutput_onepointfour[1]>0 or processOutput_onepointfour[1]>1.01*ofmass
    if(check_error_1 or check_error_2 or check_error_3):
        f_log=open(f_log_name,'wb')
        f_log.write('Serious Error happends when running function Properity_ofmass_two_peak() at Find_OfMass \n')
        f_log.write(str([check_error_1,check_error_2,check_error_3])+'\n')
        f_log.write(str(eos.args)+'\n')
        f_log.write(str([ofmass,Preset_pressure_center_low,pressure_center_left,pressure_center_after_peak,pressure_center_right,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index])+'\n')
        f_log.write(str([mass_left,mass_after_peak,mass_right])+'\n')
        f_log.write(str(processOutput_onepointfour)+'\n')
        f_log.write(str(processOutput_onepointfour_quark)+'\n')
        f_log.close()
    else:
        return processOutput_onepointfour,processOutput_onepointfour_quark


# =============================================================================
# import warnings
# warnings.filterwarnings('error')
# from eos_class import EOS_BPSwithPolyCSS
# from fractions import Fraction
# #args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# a=EOS_BPSwithPolyCSS([0.059259259259259255, 16.0, 0.29600000000000004, 267.2510854860387, 0.5984, 5000.0, 1.1840000000000002, 192.09507105726632, 286.83819801870789, Fraction(1, 1)])
# from FindMaxmass import Maxmass_transition
# from tov_f import MassRadius_transition
# print 'x'
# [transition_type,MaximumMass_pressure_center,MaximumMass,Left_pressure_center,Left_Mass,Right_pressure_center,Right_Mass]=Maxmass_transition(1e-8,1e-4,a)
# print [transition_type,MaximumMass_pressure_center,MaximumMass,Left_pressure_center,Left_Mass,Right_pressure_center,Right_Mass]
# 
# try:
#     print Properity_ofmass(1.4,1.,MaximumMass_pressure_center,MassRadius_transition,1e-8,1e-4,1,a)
# except RuntimeWarning:
#     print 'error?!'
# =============================================================================

#test:
    
# =============================================================================
# from FindMaxmass import Maxmass_transiton
# from tov_f import MassRadius_transition
# Preset_Pressure_final=1e-8
# Preset_rtol=1e-5
# from eos_class import EOS_BPSwithPolyCSS
# baryon_density0=0.16/2.7
# baryon_density1=1.85*0.16
# baryon_density2=3.74*0.16
# baryon_density3=7.4*0.16
# pressure1=10.0
# pressure2=948.110996928029
# pressure3=500.
# pressure_trans=57.11512144758545
# det_density=44.809925136050168
# cs2=1.0/3.0
# args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# a=EOS_BPSwithPolyCSS(args)
# [MaximumMass_pressure_center,MaximumMass,twopeak_forsure] = Maxmass_transiton(Preset_Pressure_final,Preset_rtol,a)
# print [MaximumMass_pressure_center,MaximumMass,twopeak_forsure]
# print(Properity_ofmass(1.4,1,MaximumMass_pressure_center,MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a))
# print(Radius_ofmass(1.4,1,MaximumMass_pressure_center,MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a))
# =============================================================================

# =============================================================================
# def tidal_bar(m1,m2,tidal1,tidal2):
#     return 16./13*((m1+12*m2)*m1**4*tidal1+(m2+12*m1)*m2**4*tidal2)/(m1+m2)**5
# m1=1.5
# m2=1.307893373
# tidal1=Properity_ofmass(m1,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal2=Properity_ofmass(m2,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal1=tidal1[-1]
# tidal2=tidal2[-1]
# print tidal_bar(m1,m2,tidal1,tidal2)
# 
# m1=1.6
# m2=1.229251003
# tidal1=Properity_ofmass(m1,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal2=Properity_ofmass(m2,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal1=tidal1[-1]
# tidal2=tidal2[-1]
# print tidal_bar(m1,m2,tidal1,tidal2)
# 
# m1=1.7
# m2=1.161293337
# tidal1=Properity_ofmass(m1,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal2=Properity_ofmass(m2,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal1=tidal1[-1]
# tidal2=tidal2[-1]
# print tidal_bar(m1,m2,tidal1,tidal2)
# 
# m1=1.8
# m2=1.101947023
# tidal1=Properity_ofmass(m1,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal2=Properity_ofmass(m2,1,MaximumMass_pressure_center,Mass_transition,MassRadius_transition,Preset_Pressure_final,value,2,a)
# tidal1=tidal1[-1]
# tidal2=tidal2[-1]
# print tidal_bar(m1,m2,tidal1,tidal2)
# =============================================================================
