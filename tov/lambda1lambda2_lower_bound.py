#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:43:45 2018

@author: sotzee
"""

from eos_class import EOS_CSS
from tov_CSS import MassRadius_CSS,Mass_CSS_formax
import numpy as np
import scipy.optimize as opt

def Maxmass(eos):
    result=opt.minimize(Mass_CSS_formax,100.0,tol=0.001,args=(eos),method='Nelder-Mead')
    return [0,result.x[0],-result.fun,result.x[0],-result.fun,result.x[0],-result.fun]

def Ofmaxmass(density0,ofmaxmass,Maxmass_function,args):
    eos=EOS_CSS([density0]+args)
    return -ofmaxmass+Maxmass_function(eos)[2]

def density0_ofmaxmass(ofmaxmass,Preset_density0_low,Preset_density0_high,Maxmass_function,eos_args):
    result=opt.brenth(Ofmaxmass,Preset_density0_low,Preset_density0_high,args=(ofmaxmass,Maxmass_function,eos_args))
    return result

Maxmass_transition(Preset_Pressure_final,Preset_rtol,eos)

# =============================================================================
# a=EOS_CSS([500, 0.0, 0.16, 1./3])
# print MassRadius_CSS(50,'MR',a)
# print Mass_CSS_formax(50,a)
# print Maxmass(a)
# =============================================================================
# =============================================================================
# args=[0.0, 0.16, 1./3]
# density0 = density0_ofmaxmass(2.,100,800,Maxmass,args)
# a=EOS_CSS([density0]+args)
# print Maxmass(a)
# =============================================================================

def Properity_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,eos):
    pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,eos)
    [M,R,beta,M_binding,momentofinertia,yR,tidal]=MassRadius_function(pressure_center,'MRBIT',eos)
    return [pressure_center,M,R,beta,M_binding,momentofinertia,yR,tidal]

def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,eos):
    result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,eos))
    return result

def Ofmass(pressure_center,ofmass,MassRadius_function,eos):
    #print MassRadius_function(pressure_center,'M',eos)
    return -ofmass+MassRadius_function(pressure_center,'M',eos)

def get_bound_lower(q,m1_grid,m2_grid,ofmaxmass,args,n):
    tidal1=m1_grid.copy()
    tidal2=m1_grid.copy()
    love2q6love1=m1_grid.copy()
    for i in range(len(m1_grid)):
        for j in range(len(m1_grid[0])):
            density0 = density0_ofmaxmass(ofmaxmass,100,800,Maxmass,args)
            print density0
            a=EOS_CSS([density0]+args)
            tidal1[i,j] = Properity_ofmass(m1_grid[i,j],10,Maxmass(a)[1],MassRadius_CSS,a)[7]
            tidal2[i,j] = Properity_ofmass(m2_grid[i,j],10,Maxmass(a)[1],MassRadius_CSS,a)[7]
            love2q6love1[i,j] = tidal2[i,j] *q[i]**n/tidal1[i,j]
    return tidal1,tidal2

def plot_bound_lower(q,m1_grid,m2_grid,ofmaxmass,args,n,subplot):
    for i in range(len(chip_mass)):
        subplot.plot(q,get_bound_lower(q,m1_grid,m2_grid,ofmaxmass,args,n),label='$M_{ch}$=%.2f'%chip_mass[i])
    subplot.legend(loc=1,prop={'size':10},frameon=False)
    subplot.set_title('$M_{max}=%.1f M_\odot, c^2_s=%.2f$ upper bound'%(ofmaxmass,args[2]))
    subplot.set_xlabel('q',fontsize=18)
    subplot.set_ylabel('$\\bar \lambda_2 q^%d /\\bar \lambda_1$'%(n),fontsize=18)

def mass_binary(mc,q):
    return [mc*(1+q)**0.2*q**0.4,mc*(1+q)**0.2/q**0.6]

chip_mass= np.linspace(1.1769, 1.1968 ,2)
q=np.linspace(0.7,1.,7)
chip_mass_grid,q_grid = np.meshgrid(chip_mass,q)

m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1

import matplotlib.pyplot as plt
n=6
ofmaxmass=2.4
cs2=1./4.
args=[0,0,cs2]
tidal1,tidal2=get_bound_lower(q,m1_grid,m2_grid,ofmaxmass,args,n)
for i in range(len(chip_mass)):
    plt.plot(q,tidal2[:,i]*q**6/tidal1[:,i],label='$M_{ch}$=%.2f'%chip_mass[i])
plt.xlim(0.7,1)
plt.legend(loc=1,prop={'size':10},frameon=False)
plt.title('$M_{max}<%.1f M_\odot, c^2_s>%.2f$ lower bound'%(ofmaxmass,args[2]))
plt.xlabel('q',fontsize=18)
plt.ylabel('$\\bar \lambda_2 q^%d /\\bar \lambda_1$'%(n),fontsize=18)
plt.show()

# =============================================================================
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(3,3,sharex=True, sharey=True,figsize=(12, 12))
# import BPS
# baryon_density0 = 0.16/2.7
# baryon_density1 = 1.85*0.16
# baryon_density2 = 3.74*0.16
# baryon_density3 = 7.4*0.16
# 
# pressure1=20
# pressure3=1000000
# cs2=1.
# pressure0=BPS.eosPressure_frombaryon(baryon_density0)
# density0=BPS.eosDensity(pressure0)
# 
# gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
# density1=(density0-pressure0/(gamma1-1.))*(pressure1/pressure0)**(1./gamma1)+pressure1/(gamma1-1.)
# 
# def causality_trans(gamma2,pressure_trans):
#     density_trans=(density1-pressure1/(gamma2-1.))*(pressure_trans/pressure1)**(1./gamma2)+pressure_trans/(gamma2-1.)
#     return gamma2*pressure_trans/(density_trans+pressure_trans)-1.
# 
# def Properity_ofmass_at_transition(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index):
#     pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol)
#     gamma2=opt.newton(causality_trans,10,args=(pressure_center,))
#     pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
#     eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
#     [M,R,beta,M_binding,momentofinertia,yR,tidal]=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MRBIT',eos)
#     return [pressure_center,M,R,beta,M_binding,momentofinertia,yR,tidal,pressure2]
# 
# def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol):
#     result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol))
#     return result
# 
# def Ofmass(pressure_center,ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol):
#     gamma2=opt.newton(causality_trans,10,args=(pressure_center,))
#     pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
#     eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
#     return -ofmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'M',eos)
# 
# tidal1=m1.copy()
# tidal2=m2.copy()
# for i in range(len(m1)):
#     print i
#     for j in range(len(m1[0])):
#         pressure_trans,tidal2[i,j],pressure2 = np.array(Properity_ofmass_at_transition(m2[i,j],1,1000,MassRadius,Preset_Pressure_final,Preset_rtol,1))[[0,7,8]]
#         args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]
#         a=EOS_BPSwithPoly(args)
#         tidal1[i,j]=Properity_ofmass(m1[i,j],10,pressure_trans+1,MassRadius,Preset_Pressure_final,Preset_rtol,1,a)[7]
# =============================================================================
