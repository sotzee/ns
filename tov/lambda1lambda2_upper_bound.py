#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:17:52 2018

@author: sotzee
"""

from eos_class import EOS_BPSwithPolyCSS,EOS_BPSwithPoly,EOS_BPS
from tov_f import MassRadius_transition,MassRadius, Mass_transition_formax
from Find_OfMass import Properity_ofmass
import numpy as np
import scipy.optimize as opt
baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.74*0.16
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
pressure3=1000000

def causality_trans(gamma2,pressure_trans):
    pressure1=pressure1_max
    gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
    density1=(density0-pressure0/(gamma1-1.))*(pressure1/pressure0)**(1./gamma1)+pressure1/(gamma1-1.)
    density_trans=(density1-pressure1/(gamma2-1.))*(pressure_trans/pressure1)**(1./gamma2)+pressure_trans/(gamma2-1.)
    return gamma2*pressure_trans/(density_trans+pressure_trans)-1.

def Properity_ofmass_at_transition(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index):
    pressure1=pressure1_max
    pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol)
    gamma2=opt.newton(causality_trans,10,args=(pressure_center,))
    pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
    eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
    [M,R,beta,M_binding,momentofinertia,yR,tidal]=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MRBIT',eos)
    return [pressure_center,M,R,beta,M_binding,momentofinertia,yR,tidal,pressure2]

def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol):
    result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol))
    return result

def Ofmass(pressure_center,ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol):
    pressure1=pressure1_max
    gamma2=opt.newton(causality_trans,10,args=(pressure_center,))
    pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
    eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
    return -ofmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'M',eos)

def Maxmass(Preset_Pressure_final,Preset_rtol,eos):
    result=opt.minimize(Mass_transition_formax,1000.0,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
    return [0,result.x[0],-result.fun,result.x[0],-result.fun,result.x[0],-result.fun]

def det_density_ofmaxmass(ofmaxmass,Preset_det_density_low,Preset_det_density_high,Maxmass_function,Preset_Pressure_final,Preset_rtol,eos_args):
    cs2=1.
    def Ofmaxmass(det_density,ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,args):
        eos=EOS_BPSwithPolyCSS(args+[det_density,cs2])
        return -ofmaxmass+Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)[2]
    result=opt.brenth(Ofmaxmass,Preset_det_density_low,Preset_det_density_high,args=(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,eos_args))
    return result

def cs2_ofmaxmass(ofmaxmass,Preset_cs2_low,Preset_cs2_high,Maxmass_function,Preset_Pressure_final,Preset_rtol,eos_args):
    det_density=0
    def Ofmaxmass(cs2,ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,args):
        eos=EOS_BPSwithPolyCSS(args+[det_density,cs2])
        return -ofmaxmass+Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)[2]
    result=opt.brenth(Ofmaxmass,Preset_cs2_low,Preset_cs2_high,args=(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,eos_args))
    return result

def bound_upper(m1,m2,pressure1,ofmaxmass):
    cs2=1.
    pressure_trans,tidal2,pressure2 = np.array(Properity_ofmass_at_transition(m2,1,1000,MassRadius,Preset_Pressure_final,Preset_rtol,1))[[0,7,8]]
    eos_args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans]
    det_density = det_density_ofmaxmass(ofmaxmass,0,1000,Maxmass,Preset_Pressure_final,Preset_rtol,eos_args)
    args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
    a=EOS_BPSwithPolyCSS(args)
    tidal1=Properity_ofmass(m1,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    return tidal1,tidal2,args

def bound_lower(m1,m2,pressure1,ofmaxmass):
    pressure2=100 #trivial parameter
    pressure_trans=pressure1
    gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
    pressure_s=pressure1*(baryon_density_s/baryon_density1)**gamma1
    pressure_trans=pressure_s
    det_density=0
    eos_args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans]
    cs2 = cs2_ofmaxmass(ofmaxmass,0.25,1.,Maxmass,Preset_Pressure_final,Preset_rtol,eos_args)
    args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
    a=EOS_BPSwithPolyCSS(args)
    tidal1=Properity_ofmass(m1,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    tidal2=Properity_ofmass(m2,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    return tidal1,tidal2,args

def bound_lower_no_maxmass_max(m1,m2,pressure1,ofmaxmass):
    pressure2=100 #trivial parameter
    gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
    pressure_s=pressure1*(baryon_density_s/baryon_density1)**gamma1
    pressure_trans=pressure_s
    det_density=0
    args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,1]
    a=EOS_BPSwithPolyCSS(args)
    tidal1=Properity_ofmass(m1,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    tidal2=Properity_ofmass(m2,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    return tidal1,tidal2,args

def bound_lower_with_args(m1,m2,pressure1,args):
    gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
    pressure_s=pressure1*(baryon_density_s/baryon_density1)**gamma1
    pressure_trans=pressure_s
    a=EOS_BPSwithPolyCSS(args)
    tidal1=Properity_ofmass(m1,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    tidal2=Properity_ofmass(m2,pressure_trans+1,Maxmass(Preset_Pressure_final,Preset_rtol,a)[1],MassRadius_transition,Preset_Pressure_final,Preset_rtol,1,a)[7]
    return tidal1,tidal2,args

def get_bound(q,m1_grid,m2_grid,maxmass_range,p1_range,upper_or_lower):
    tidal1=m1_grid.copy()
    tidal2=m1_grid.copy()
    eos_args=[]
    maxmass_min,maxmass_max = maxmass_range
    pressure1_min,pressure1_max = p1_range
    if(upper_or_lower=='upper'):
        pressure1=pressure1_max
        ofmaxmass = maxmass_min
        bound_function=bound_upper
    elif(upper_or_lower=='lower'):
        pressure1=pressure1_min
        ofmaxmass = maxmass_max
        args = bound_lower(1.4,1.4,pressure1,maxmass_max)[2]
        ofmaxmass = args
        bound_function=bound_lower_with_args
    elif(upper_or_lower=='lower_no_maxmass_max'):
        pressure1=pressure1_min
        args = bound_lower_no_maxmass_max(1.4,1.4,pressure1,maxmass_max)[2]
        ofmaxmass = args
        bound_function=bound_lower_with_args
    for i in range(len(m1_grid)):
        print i
        eos_args.append([])
        for j in range(len(m1_grid[0])):
            eos_args[i].append([])
            tidal1[i,j],tidal2[i,j],eos_args[i][j] = bound_function(m1_grid[i,j],m2_grid[i,j],pressure1,ofmaxmass)
    return tidal1,tidal2,eos_args



# def get_tidal1tidal2(m1_grid,m2_grid,MassRadius_function,eos):
#     tidal1=m1_grid.copy()
#     tidal2=m1_grid.copy()
#     for i in range(len(m1_grid)):
#         for j in range(len(m1_grid[0])):
#             tidal1[i,j] = Properity_ofmass(m1_grid[i,j],10,Maxmass(Preset_Pressure_final,Preset_rtol,eos)[1],MassRadius_function,Preset_Pressure_final,Preset_rtol,1,eos)[7]
#             tidal2[i,j] = Properity_ofmass(m2_grid[i,j],10,Maxmass(Preset_Pressure_final,Preset_rtol,eos)[1],MassRadius_function,Preset_Pressure_final,Preset_rtol,1,eos)[7]
#     return tidal1,tidal2


def mass_binary(mc,q):
    return [mc*(1+q)**0.2*q**0.4,mc*(1+q)**0.2/q**0.6]

Preset_rtol=1e-4
Preset_Pressure_final=1e-8

chip_mass= np.linspace(1.05, 1.4,8)
q=np.linspace(0.7,1.,13)
chip_mass_grid,q_grid = np.meshgrid(chip_mass,q)

m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1

maxmass_min=2.0
maxmass_max=2.4
pressure1_max = 20.
pressure1_min = 8.4

tidal1_upper,tidal2_upper,eos_args_upper = get_bound(q,m1_grid,m2_grid,[maxmass_min,maxmass_max],[pressure1_min,pressure1_max],'upper')
tidal1_lower,tidal2_lower,eos_args_lower = get_bound(q,m1_grid,m2_grid,[maxmass_min,maxmass_max],[pressure1_min,pressure1_max],'lower')
tidal1_lower_no_maxmass_max,tidal2_lower_no_maxmass_max,eos_args_no_maxmass_max = get_bound(q,m1_grid,m2_grid,[maxmass_min,maxmass_max],[pressure1_min,pressure1_max],'lower_no_maxmass_max')
import pickle
f=open('hybrid_upper_bound.dat','wb')
pickle.dump([tidal1_upper,tidal2_upper,eos_args_upper],f)
f.close()
f=open('hybrid_lower_bound.dat','wb')
pickle.dump([tidal1_lower,tidal2_lower,eos_args_lower],f)
f.close()
f=open('hybrid_lower_bound_no_maxmass_max.dat','wb')
pickle.dump([tidal1_lower_no_maxmass_max,tidal2_lower_no_maxmass_max,eos_args_no_maxmass_max],f)
f.close()

f=open('hybrid_upper_bound.dat','rb')
tidal1_upper,tidal2_upper,eos_args_upper=pickle.load(f)
f.close()
f=open('hybrid_lower_bound.dat','rb')
tidal1_lower,tidal2_lower,eos_args_lower=pickle.load(f)
f.close()
f=open('hybrid_lower_bound_no_maxmass_max.dat','rb')
tidal1_lower_no_maxmass_max,tidal2_lower_no_maxmass_max,eos_args_no_maxmass_max=pickle.load(f)
f.close()

import matplotlib.pyplot as plt
n=6
cmap = plt.cm.get_cmap('jet')
(np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int)
colors = cmap((np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int))
f, axs= plt.subplots(4,1, sharex=True,figsize=(10, 80./3))
for i in range(len(chip_mass)):
    axs[0].plot(q,q**n*tidal2_upper[:,i]/tidal1_upper[:,i],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[0].legend(loc=1,prop={'size':10},frameon=False)
    axs[0].set_title('$M_{max}>%.1f M_\odot, p_1<%.1f$ MeV fm$^{-3}$ upper bound'%(maxmass_min,pressure1_max))
    axs[0].set_xlabel('q',fontsize=18)
    axs[0].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[0].set_ylim(0,180)
    #axs[0].set_yscale('log')
    
    axs[1].plot(q,q**n*tidal2_lower[:,i]/tidal1_lower[:,i],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[1].legend(loc=4,prop={'size':10},frameon=False)
    axs[1].set_title('$M_{max}<%.1f M_\odot, p_1>%.2f$ MeV fm$^{-3}$ lower bound'%(maxmass_max,pressure1_min))
    axs[1].set_xlabel('q',fontsize=18)
    axs[1].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[1].set_ylim(1,7)
    #axs[1].set_yscale('log')
    
    axs[2].plot(q,q**n*tidal2_lower_no_maxmass_max[:,i]/tidal1_lower_no_maxmass_max[:,i],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[2].legend(loc=4,prop={'size':10},frameon=False)
    axs[2].set_title('$p_1>%.2f$ MeV fm$^{-3}$ lower bound'%(pressure1_min))
    axs[2].set_xlabel('q',fontsize=18)
    axs[2].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[2].set_ylim(1,5)
    #axs[2].set_yscale('log')
    
    bound_log=axs[3].scatter(q,q*0.0,c=[chip_mass[i]]*len(q),cmap=plt.cm.jet)
    axs[3].plot(q,q**n*tidal2_upper[:,i]/tidal1_upper[:,i],color=colors[i])
    axs[3].plot(q,q**n*tidal2_lower[:,i]/tidal1_lower[:,i],'--',color=colors[i])
    axs[3].plot(q,q**n*tidal2_lower_no_maxmass_max[:,i]/tidal1_lower_no_maxmass_max[:,i],':',color=colors[i])
    if(i==0):
        axs[3].plot([0,0,0],[0,1,2],color='k',label='$%.1f M_\odot<M_{max}<%.1f M_\odot$ upper bound'%(maxmass_min,maxmass_max))
        axs[3].plot([0,0,0],[0,1,2],'--',color='k',label='$%.1f M_\odot<M_{max}<%.1f M_\odot$ lower bound'%(maxmass_min,maxmass_max))
        axs[3].plot([0,0,0],[0,1,2],':',color='k',label='$%.1f M_\odot<M_{max}< \infty$         lower bound'%(maxmass_min))
    axs[3].legend(loc=7,prop={'size':15},frameon=False)
    #axs[3].set_title('$%.2f$ MeV fm$^{-3}<p_1<%.0f$ MeV fm$^{-3}$ upper and lower bounds'%(pressure1_min,pressure1_max))
    axs[3].set_xlabel('q',fontsize=25)
    axs[3].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=25)
    axs[3].set_ylim(0.5,22)
    axs[3].set_yscale('log')

f.colorbar(bound_log,ax=axs[3])
bound_log.set_clim(chip_mass[0], chip_mass[-1])
plt.xlim(0.7,1)
plt.show()

# =============================================================================
# ofmaxmass=2.0
# Preset_det_density_low=0
# Preset_det_density_high=1000
# Maxmass_function=Maxmass
# Preset_rtol=1e-4
# Preset_Pressure_final=1e-8
# pressure1=3.
# pressure_trans=pressure1*(baryon_density_s/baryon_density1)**(np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0))
# pressure2=100
# eos_args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans]
# cs2=2./3
# n=6
# #print det_density_ofmaxmass(ofmaxmass,Preset_det_density_low,Preset_det_density_high,Maxmass_function,Preset_Pressure_final,Preset_rtol,eos_args)
# 
# def mass_binary(mc,q):
#     return [mc*(1+q)**0.2*q**0.4,mc*(1+q)**0.2/q**0.6]
# 
# def get_tidal1tidal2(m1_grid,m2_grid,MassRadius_function,eos):
#     tidal1=m1_grid.copy()
#     tidal2=m1_grid.copy()
#     for i in range(len(m1_grid)):
#         for j in range(len(m1_grid[0])):
#             tidal1[i,j] = Properity_ofmass(m1_grid[i,j],10,Maxmass(Preset_Pressure_final,Preset_rtol,eos)[1],MassRadius_function,Preset_Pressure_final,Preset_rtol,1,eos)[7]
#             tidal2[i,j] = Properity_ofmass(m2_grid[i,j],10,Maxmass(Preset_Pressure_final,Preset_rtol,eos)[1],MassRadius_function,Preset_Pressure_final,Preset_rtol,1,eos)[7]
#     return tidal1,tidal2
# 
# chip_mass= np.linspace(1.1, 1.4 ,4)
# q=np.linspace(0.7,1.,4)
# chip_mass_grid,q_grid = np.meshgrid(chip_mass,q)
# 
# m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1
# 
# import matplotlib.pyplot as plt
# f, axs= plt.subplots(3,3, sharex=True,figsize=(10, 10))
# for a in range(3):
#     for b in range(3):
#         ofmaxmass=2+0.2*b
#         cs2=(3+a)/12.
#         det_density=det_density_ofmaxmass(ofmaxmass,Preset_det_density_low,Preset_det_density_high,Maxmass_function,Preset_Pressure_final,Preset_rtol,eos_args)
#         args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
#         eos_final=EOS_BPSwithPolyCSS(args)
#         tidal1,tidal2 = get_tidal1tidal2(m1_grid,m2_grid,MassRadius_transition,eos_final)
# 
#         for i in range(len(chip_mass)):
#             axs[a,b].set_title('$M_{max}=%.2f M_\odot,  c_s^2=%.2f$'%(ofmaxmass,cs2))
#             axs[a,b].plot(q,tidal2[:,i]*q**n/tidal1[:,i],label='$M_{ch}$=%.2f'%chip_mass[i])
#             axs[a,b].legend(loc=1,prop={'size':10},frameon=False)
# plt.xlabel('q',fontsize=18)
# plt.xlim(0.7,1)
# plt.ylabel('$\\bar \lambda_2 q^%d /\\bar \lambda_1$'%(n),fontsize=18)
# plt.show()
# =============================================================================
