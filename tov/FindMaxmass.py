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
    result1=opt.minimize(Mass_transition_formax,800.0,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
    result2=opt.minimize(Mass_transition_formax,eos.pressure_trans+1.,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
    #print result1.x[0],result1.fun
    #print result2.x[0],result2.fun
    Mass_trans=Mass_transition_formax([eos.pressure_trans],Preset_Pressure_final,Preset_rtol,eos)
    if(np.abs(result2.x[0]-eos.pressure_trans)<0.001*(eos.pressure_trans+result2.x[0])):
        result2.x[0]=eos.pressure_trans
        result2.fun=Mass_trans
        if(np.abs(result2.x[0]-result1.x[0])<0.01*(result1.x[0]+result2.x[0]) and np.abs(result2.fun-result1.fun)<-0.0001*(result1.fun+result2.fun)):
            return [eos.pressure_trans,-Mass_trans,2]
        else:
            if(result1.fun<result2.fun):
                return [result1.x[0],-result1.fun,3]
            else:
                return [result2.x[0],-result2.fun,3]
    else:
        if(np.abs(result2.x[0]-result1.x[0])<0.01*(result1.x[0]+result2.x[0]) and np.abs(result2.fun-result1.fun)<-0.0001*(result1.fun+result2.fun)):
            return [result2.x[0],-result2.fun,1]
        else:
            if(result1.fun<result2.fun):
                return [result1.x[0],-result1.fun,4]
            else:
                return [result2.x[0],-result2.fun,4]

def Maxmass(Preset_Pressure_final,Preset_rtol,eos):
    result=opt.minimize(Mass_formax,100.0,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
    return [result.x[0],-result.fun,0]


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

# =============================================================================
# N=100
# from tov_f import MassRadius_transition
# pressure_center=np.linspace(20.,1000.,N)
# mass=np.linspace(20.,500.,N)
# radius=np.linspace(20.,500.,N)
# for i in range(N):
#     mass[i],radius[i]=MassRadius_transition(pressure_center[i],1e-7,1e-5,'MR',a)
# import matplotlib.pyplot as plt
# plt.plot(radius,mass)
# 
# print MassRadius_transition(732.94907226562543,1e-7,1e-5,'MR',a)
# print MassRadius_transition(737,1e-7,1e-5,'MR',a)
# =============================================================================
