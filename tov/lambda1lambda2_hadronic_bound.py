#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:27:13 2018

@author: sotzee
"""
import numpy as np
from eos_class import EOS_BPS,EOS_BPSwithPoly,EOS_PiecewisePoly
from tov_f import MassRadius,Mass_formax
from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass

import scipy.optimize as opt
from scipy.misc import derivative
baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.74*0.16
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
Preset_rtol=1e-6
Preset_Pressure_final=1e-8
tol_p1=0.3

# =============================================================================
# def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol):
#     result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol))
#     return result
# 
# def Ofmass(pressure_center,ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol):
#     pressure1=pressure1_max
#     gamma2=opt.newton(causality_trans,10,args=(pressure_center,))
#     pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
#     eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
#     return -ofmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'M',eos)
# 
# =============================================================================
def causality_central_pressure(pressure_center,density2,pressure2,gamma3):
    #print pressure_center,np.where(pressure_center>0,(gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)-1,-1+pressure_center/1000.)
    return np.where(pressure_center>0,(gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)-1,-1+pressure_center/1000.)

def caulality_central_pressure_at_peak(pressure3,pressure1,pressure2,Preset_Pressure_final,Preset_rtol):
    eos = EOS_PiecewisePoly([density0,pressure0,baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
    #print '==================Finding pressure_center at p3=%f'%pressure3
    pressure_center=opt.newton(causality_central_pressure,pressure3,tol=0.1,args=(eos.density2,pressure2,eos.gamma3))
    #print pressure3,derivative(MassRadius,pressure_center,dx=1e-6,args=(Preset_Pressure_final,Preset_rtol,'M',eos))
    
    derivative_center_pressure=derivative(MassRadius,pressure_center,dx=1e-2,args=(Preset_Pressure_final,Preset_rtol,'M',eos))
    
    #print 'derivative = %f at center pressure = %f'%(derivative_center_pressure,pressure_center)
    return derivative_center_pressure

def p2p3_ofmaxmass(ofmaxmass,Preset_p2,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1):
    print '==================Finding p2 of maxmass%.2f at p1=%.2f'%(ofmaxmass,p1)
    pressure3_result=[0]
    pressure_center_maxmass=[0]
    def Ofmaxmass(p2,ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,args_p1):
        print '==================Finding p3 at p2=%f'%p2
        pressure3=opt.newton(caulality_central_pressure_at_peak,750.0,tol=0.1,args=(args_p1,p2,Preset_Pressure_final,Preset_rtol))
        pressure3_result[0]=pressure3
        args=[baryon_density0,args_p1,baryon_density1,p2,baryon_density2,pressure3,baryon_density3]
        eos=EOS_BPSwithPoly(args)
        maxmass_result = Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)
        pressure_center_maxmass[0]=maxmass_result[1]
        print 'maxmass=%f'%maxmass_result[2]
        return -ofmaxmass+maxmass_result[2]
    result=opt.newton(Ofmaxmass,Preset_p2,tol=0.1,args=(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1))
    return result,pressure3_result[0],pressure_center_maxmass[0]

import pickle
def main_upper_bound_configuration():
    p1=np.linspace(3.75,30,106)
    p2=np.linspace(3.75,30,106)
    p3=np.linspace(3.75,30,106)
    pc_maxmass=np.linspace(3.75,30,106)
    
    for i in range(len(p1)):
        p2[i],p3[i],pc_maxmass[i]=p2p3_ofmaxmass(2.0,105.,Maxmass,Preset_Pressure_final,Preset_rtol,p1[i])
    
    f=open('hadronic_upper_bound_p1_p2p3pc.dat','wb')
    pickle.dump([p1,p2,p3,pc_maxmass],f)
    f.close()

f=open('hadronic_upper_bound_p1_p2p3pc.dat','wb')
p1,p2,p3,pc_maxmass=pickle.load(f)
f.close()
def func(x,a,b,c,d,e,f):
    return a+b*x+c*x**2+d*x**3+e*x**4+f*x**5

x0=[100,0,0,0,0,0]
fit_result_p2=opt.curve_fit(func, p1,p2, x0)
x0=[600,0,0,0,0,0]
fit_result_p3=opt.curve_fit(func, p1,p3, x0)
x0=[1000,0,0,0,0,0]
fit_result_pc_maxmass=opt.curve_fit(func, p1,pc_maxmass, x0)

def bound_upper(pressure1,m1,m2):
    #pressure2,pressure3,pressure_center_maxmass = p2p3_ofmaxmass(ofmaxmass,105,Maxmass,Preset_Pressure_final,Preset_rtol,pressure1)
    pressure2=func(pressure1,*fit_result_p2[0])
    pressure3=func(pressure1,*fit_result_p3[0])
    #pressure_center_maxmass=func(pressure1,*fit_result_pc_maxmass[0])
    args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]
    a=EOS_BPSwithPoly(args)
    tidal1=Properity_ofmass(m1,pressure1,pressure3,MassRadius,Preset_Pressure_final,Preset_rtol,1,a)[7]
    tidal2=Properity_ofmass(m2,pressure1,pressure3,MassRadius,Preset_Pressure_final,Preset_rtol,1,a)[7]
    print pressure1, tidal1/tidal2
    return tidal1/tidal2,tidal1,tidal2,args
#bound_upper(10.,1.7,1.5)

#plt.plot(p1,func(p1,*fit_result_p2[0]))
#plt.plot(p1,func(p1,*fit_result_p3[0]))

# =============================================================================
# def get_bound_upper(pressure1,m1,m2):
#     result=opt.minimize(bound_upper,pressure1,args=(m1,m2),tol=0.2,options={'eps':0.1})
#     return 1./result.fun
# get_bound_upper(20,1.6,1.5)    
# =============================================================================

# =============================================================================
# def get_bound_upper(pressure1_array,m1,m2):
#     t1t2_prev=1
#     t2t1='pressure1 upper range too low'
#     for i in range(len(pressure1_array)):
#         t1t2=bound_upper(pressure1_array[i],m1,m2)
#         if(t1t2>t1t2_prev):
#             t2t1=1.0/t1t2_prev
#             break
#         else:
#             t1t2_prev=t1t2
#     if(i==1):
#         t2t1='pressure1 lower range too high'
#     return t2t1
# =============================================================================

def preset_p1(m1,m2):
    return 17.+ m1/m2*2.**(5*((m1+m2)/2.-1.05))

def get_bound_upper(m1,m2):
    pressure1=preset_p1(m1,m2)
    t1t2_lef = bound_upper(pressure1-tol_p1,m1,m2)
    t1t2_mid = bound_upper(pressure1,m1,m2)
    t1t2_rig = bound_upper(pressure1+tol_p1,m1,m2)
    
    for i in range(11):
        pressure1_movement=tol_p1*(t1t2_lef[0]-t1t2_rig[0])/(2*(t1t2_lef[0]+t1t2_rig[0]-2*t1t2_mid[0]))
        pressure1+=pressure1_movement
        if(np.abs(pressure1_movement)<tol_p1):
            t1t2_mid = bound_upper(pressure1,m1,m2)
            break
        t1t2_lef = bound_upper(pressure1-tol_p1,m1,m2)
        t1t2_mid = bound_upper(pressure1,m1,m2)
        t1t2_rig = bound_upper(pressure1+tol_p1,m1,m2)
    if(i==10):
        return 'fail to converge after 11 trial'
    else:
        return t1t2_mid[1:]


def causality_p2(pressure2,pressure1):
    pressure3=pressure2
    args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]
    a=EOS_BPSwithPoly(args)
    return a.eosPiecewisePoly.gamma2*pressure2/(a.eosPiecewisePoly.density2+pressure2)-1.

def causality_p3(pressure3,pressure1,pressure2):
    args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]
    a=EOS_BPSwithPoly(args)
    density3=(a.eosPiecewisePoly.density2-pressure2/(a.eosPiecewisePoly.gamma3-1))\
    *(pressure3/pressure2)**(1./a.eosPiecewisePoly.gamma3)\
    +pressure3/(a.eosPiecewisePoly.gamma3-1)
    return a.eosPiecewisePoly.gamma3*pressure3/(density3+pressure3)-1.

p1_lower_bound=3.75
p2_causal=opt.newton(causality_p2,100.,args=(p1_lower_bound,))
p3_preset=opt.newton(causality_p3,800.,args=(p1_lower_bound,p2_causal))
print p1_lower_bound,p2_causal
print p3_preset
print opt.newton(caulality_central_pressure_at_peak,p3_preset,tol=0.1,args=(p1_lower_bound,p2_causal,Preset_Pressure_final,Preset_rtol))
args_lower_bound = [0.059259259259259255, 3.75, 0.29600000000000004,144.85571948688346, 0.5984, 951.205339935, 1.1840000000000002]
eos_lower_bound = EOS_BPSwithPoly(args_lower_bound)

def get_bound_lower(m1,m2):
    tidal1=Properity_ofmass(m1,10.,1000.,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_lower_bound)[7]
    tidal2=Properity_ofmass(m2,10.,1000.,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_lower_bound)[7]
    return tidal1,tidal2,args_lower_bound

def get_bound(m1_grid,m2_grid,upper_or_lower):
    tidal1=m1_grid.copy()
    tidal2=m1_grid.copy()
    eos_args=[]
    if(upper_or_lower=='upper'):
        bound_function=get_bound_upper
    elif(upper_or_lower=='lower'):
        bound_function=get_bound_lower

    for i in range(len(m1_grid)):
        print i
        eos_args.append([])
        for j in range(len(m1_grid[0])):
            eos_args[i].append([])
            tidal1[i,j],tidal2[i,j],eos_args[i][j] = bound_function(m1_grid[i,j],m2_grid[i,j])
    return tidal1,tidal2,eos_args

def mass_binary(mc,q):
    return [mc*(1+q)**0.2*q**0.4,mc*(1+q)**0.2/q**0.6]

chip_mass= np.linspace(1.05, 1.4,8)
q=np.linspace(0.7,1.,12,endpoint=False)
chip_mass_grid,q_grid = np.meshgrid(chip_mass,q)

m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1

tidal1_upper,tidal2_upper,eos_args_upper = get_bound(m1_grid,m2_grid,'upper')
tidal1_lower,tidal2_lower,eos_args_lower = get_bound(m1_grid,m2_grid,'lower')
f=open('hadronic_upper_bound.dat','wb')
pickle.dump([tidal1_upper,tidal2_upper,eos_args_upper],f)
f.close()
f=open('hadronic_lower_bound.dat','wb')
pickle.dump([tidal1_lower,tidal2_lower,eos_args_lower],f)
f.close()

import matplotlib.pyplot as plt
n=6
maxmass_min=2.0
pressure1_min=3.75
pressure1_max=30.
cmap = plt.cm.get_cmap('jet')
(np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int)
colors = cmap((np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int))
f, axs= plt.subplots(3,1, sharex=True,figsize=(6, 15))
for i in range(len(chip_mass)):
    axs[0].plot(list(q)+[1],list(q**n*tidal2_upper[:,i]/tidal1_upper[:,i])+[1],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[0].legend(loc=1,prop={'size':10},frameon=False)
    axs[0].set_title('$M_{max}>%.1f M_\odot$ upper bound'%(maxmass_min))
    axs[0].set_xlabel('q',fontsize=18)
    axs[0].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[0].set_ylim(0,180)
    #axs[0].set_yscale('log')
    
    axs[1].plot(list(q)+[1],list(q**n*tidal2_lower[:,i]/tidal1_lower[:,i])+[1],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[1].legend(loc=4,prop={'size':10},frameon=False)
    axs[1].set_title('$p_1>%.2f$ MeV fm$^{-3}$ lower bound'%(pressure1_min))
    axs[1].set_xlabel('q',fontsize=18)
    axs[1].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[1].set_ylim(1,7)
    #axs[1].set_yscale('log')
    
    bound_log=axs[2].scatter(q,q*0.0,c=[chip_mass[i]]*len(q),cmap=plt.cm.jet)
    axs[2].plot(list(q)+[1],list(q**n*tidal2_upper[:,i]/tidal1_upper[:,i])+[1],color=colors[i])
    axs[2].plot(list(q)+[1],list(q**n*tidal2_lower[:,i]/tidal1_lower[:,i])+[1],'--',color=colors[i])
    if(i==0):
        axs[2].plot([0,0,0],[0,1,2],color='k',label='$M_{max}>%.1f M_\odot$ upper bound'%(maxmass_min))
        axs[2].plot([0,0,0],[0,1,2],'--',color='k',label='$p_1>%.2f$ MeV fm$^{-3}$ lower bound'%(pressure1_min))
    axs[2].legend(loc=1,prop={'size':8},frameon=False)
    axs[2].set_title('upper and lower bounds')
    axs[2].set_xlabel('q',fontsize=18)
    axs[2].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    axs[2].set_ylim(0.5,22)
    axs[2].set_yscale('log')

f.colorbar(bound_log,ax=axs[2])
bound_log.set_clim(chip_mass[0], chip_mass[-1])
plt.xlim(0.7,1)
plt.show()