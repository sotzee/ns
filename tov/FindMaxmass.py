# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:13:47 2016

@author: Sotzee (use smarter method to find maximum mass)
"""

import scipy.optimize as opt
import numpy as np
from tov_f import Mass_transition_formax,Mass_formax

# transition tpye:
# 0 ~ no transition
# 1 ~ continuous one peak
# 2 ~ absent one peak at transition
# 3 ~ absent two peaks
# 4 ~ continuous two peaks

def Maxmass_transition(Preset_Pressure_final,Preset_rtol,eos):
    Preset_rtol=Preset_rtol*0.01
    rtol_opt=Preset_rtol*10
    if(2*eos.det_density>eos.density_trans+3*eos.pressure_trans):
        Mass_trans=Mass_transition_formax([eos.pressure_trans],Preset_Pressure_final,Preset_rtol,eos)
        result1=opt.minimize(Mass_transition_formax,800.0,tol=rtol_opt,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
        if(np.abs(result1.x[0]-eos.pressure_trans)<0.01*(eos.pressure_trans+result1.x[0])):
            return [2,eos.pressure_trans,-Mass_trans,eos.pressure_trans,-Mass_trans,eos.pressure_trans,-Mass_trans]
        else:
            if(result1.fun<Mass_trans):
                return [3,result1.x[0],-result1.fun,result1.x[0],-result1.fun,eos.pressure_trans,-Mass_trans]
            else:
                return [3,eos.pressure_trans,-Mass_trans,result1.x[0],-result1.fun,eos.pressure_trans,-Mass_trans]
    else:
        result1=opt.minimize(Mass_transition_formax,800.0,tol=rtol_opt,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
        result2=opt.minimize(Mass_transition_formax,eos.pressure_trans+1.,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
        #print result1.x[0],-result1.fun
        #print result2.x[0],-result2.fun
        if(np.abs(result1.x[0]-result2.x[0])<0.05*(result2.x[0]+result1.x[0])):
            if(result1.fun<result2.fun):
                return [1,result1.x[0],-result1.fun,result1.x[0],-result1.fun,result1.x[0],-result1.fun]
            else:
                return [1,result2.x[0],-result2.fun,result2.x[0],-result2.fun,result2.x[0],-result2.fun]
        else:
            if(result1.fun<result2.fun):
                return [4,result1.x[0],-result1.fun,result1.x[0],-result1.fun,result2.x[0],-result2.fun]
            else:
                return [4,result2.x[0],-result2.fun,result1.x[0],-result1.fun,result2.x[0],-result2.fun]

def Maxmass(Preset_Pressure_final,Preset_rtol,eos):
    result=opt.minimize(Mass_formax,100.0,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
    return [result.x[0],-result.fun,0,result.x[0],0]

# =============================================================================
# from eos_class import EOS_BPSwithPolyCSS
# from fractions import Fraction
# a=EOS_BPSwithPolyCSS([0.059259259259259255, 10.0, 0.29600000000000004, 175.05700209813543, 0.5984, 5000.0, 1.1840000000000002, 81.61498460032989, 360.08183812497651, Fraction(1, 1)])
# print Maxmass_transition(1e-8,1e-4,a)
# N=100
# from tov_f import MassRadius_transition
# print MassRadius_transition(1354.1826057434082,1e-8,1e-4,'M',a)
# 
# pressure_center=np.linspace(5.,150.,N)
# mass=np.linspace(20.,500.,N)
# radius=np.linspace(20.,500.,N)
# M_binding=np.linspace(20.,500.,N)
# for i in range(N):
#     mass[i],radius[i]=MassRadius_transition(pressure_center[i],1e-7,1e-5,'MR',a)
#     M_binding[i]=MassRadius_transition(pressure_center[i],1e-7,1e-5,'B',a)
# import matplotlib.pyplot as plt
# #plt.plot(radius,mass)
# plt.plot(pressure_center,M_binding)
# plt.plot(pressure_center,mass)
# =============================================================================

# =============================================================================
# baryon_density0=0.16/2.7
# baryon_density1=1.85*0.16
# baryon_density2=3.74*0.16
# baryon_density3=7.4*0.16
# pressure1=10.0
# pressure2=150.
# pressure3=1000.
# pressure_trans=120
# det_density=100
# cs2=1.0
# import BPSwithPolyExtention
# args=[BPSwithPolyExtention,[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]]
# #args=[BPSwithPoly,[0.059259259259259255, 7.0, 0.29600000000000004, 170.03642273919303, 0.5984, 171.30170501586613, 1.1840000000000002, 0.0, 0.0, 0.0]]
# value=200
# Preset_Pressure_final=1e-5
# print Maxmass(Preset_Pressure_final,value,args)
# =============================================================================
