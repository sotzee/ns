#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:12:30 2018

@author: sotzee
"""

from eos_class import EOS_CSS
from tov_CSS import Mass_CSS_formax,MassRadius_CSS
import numpy as np
import scipy.optimize as opt

cs2_lower_bound=0.2
cs2_upper_bound=0.4

def Maxmass(eos):
    result=opt.minimize(Mass_CSS_formax,100.0,tol=0.001,args=(eos),method='Nelder-Mead')
    return [0,result.x[0],-result.fun,result.x[0],-result.fun,result.x[0],-result.fun]

def density_surface_ofmaxmass(ofmaxmass,density_surface_low,density_surface_high,Maxmass_function,cs2):
    def Ofmaxmass(density_surface,ofmaxmass,Maxmass_function):
        eos=EOS_CSS([density_surface,0.,1.,cs2])
        return -ofmaxmass+Maxmass_function(eos)[2]
    result=opt.brenth(Ofmaxmass,density_surface_low,density_surface_high,args=(ofmaxmass,Maxmass_function))
    return result

def Properity_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,eos):
    pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,eos)
    [M,R,beta,M_binding,momentofinertia,yR,tidal]=MassRadius_function(pressure_center,'MRBIT',eos)
    return [pressure_center,M,R,beta,M_binding,momentofinertia,yR,tidal]

def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,eos):
    result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,eos))
    return result

def Ofmass(pressure_center,ofmass,MassRadius_function,eos):
    return -ofmass+MassRadius_function(pressure_center,'M',eos)



maxmass_min=2.0
maxmass_max=2.4
cs2_min=0.2
cs2_max=1.

eos_upper_bound=EOS_CSS([density_surface_ofmaxmass(maxmass_min,50.,2000.,Maxmass,cs2_max),0.,1.,cs2_max])
eos_lower_bound=EOS_CSS([density_surface_ofmaxmass(maxmass_max,50.,2000.,Maxmass,cs2_min),0.,1.,cs2_min])

def get_tidal1tidal2(m1,m2,eos):
    maxmass_pc=Maxmass(eos)[1]
    tidal1=Properity_ofmass(m1,5.,maxmass_pc,MassRadius_CSS,eos)[7]
    tidal2=Properity_ofmass(m2,5.,maxmass_pc,MassRadius_CSS,eos)[7]
    return tidal1,tidal2
def get_bound(m1_grid,m2_grid,upper_or_lower):
    tidal1=m1_grid.copy()
    tidal2=m1_grid.copy()
    
    if(upper_or_lower=='upper'):
        eos=eos_upper_bound
    elif(upper_or_lower=='lower'):
        eos=eos_lower_boundMassRadius_CSS
    for i in range(len(m1_grid)):
        print i
        for j in range(len(m1_grid[0])):
            tidal1[i,j],tidal2[i,j] = get_tidal1tidal2(m1_grid[i,j],m2_grid[i,j],eos)
    return tidal1,tidal2

def mass_binary(mc,q):
    return [mc*(1+q)**0.2*q**0.4,mc*(1+q)**0.2/q**0.6]

#chip_mass= np.linspace(1.05, 1.4,8)
chip_mass= np.linspace(1.188, 1.188,1)
q=np.linspace(0.7,1.,13)
chip_mass_grid,q_grid = np.meshgriimage.pngd(chip_mass,q)

m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1

tidal1_upper,tidal2_upper = get_bound(m1_grid,m2_grid,'upper')
tidal1_lower,tidal2_lower = get_bound(m1_grid,m2_grid,'lower')

import matplotlib.pyplot as plt
n=6
maxmass_min=2.0
pressure1_min=3.75
pressure1_max=30.
cmap = plt.cm.get_cmap('jet')
(np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int)
colors = cmap((np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int))
f, axs= plt.subplots(3,1, sharex=True,figsize=(10, 20))
for i in range(len(chip_mass)):
    axs[0].plot(list(q)+[1],list(q**n*tidal2_upper[:,i]/tidal1_upper[:,i])+[1],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[0].legend(loc=1,prop={'size':10},frameon=False)
    axs[0].set_title('$M_{max}>%.1f M_\odot$, $c_s^2<%.1f$ upper bound'%(maxmass_min,cs2_max))
    axs[0].set_xlabel('q',fontsize=18)
    axs[0].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[0].set_ylim(0,180)
    #axs[0].set_yscale('log')

    axs[1].plot(list(q)+[1],list(q**n*tidal2_lower[:,i]/tidal1_lower[:,i])+[1],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[1].legend(loc=4,prop={'size':10},frameon=False)
    axs[1].set_title('$M_{max}<%.1f M_\odot$, $c_s^2>%.1f$ lower bound'%(maxmass_max,cs2_min))
    axs[1].set_xlabel('q',fontsize=18)
    axs[1].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[1].set_ylim(1,7)
    #axs[1].set_yscale('log')

    bound_log=axs[2].scatter(q,q*0.0,c=[chip_mass[i]]*len(q),cmap=plt.cm.jet)
    axs[2].plot(list(q)+[1],list(q**n*tidal2_upper[:,i]/tidal1_upper[:,i])+[1],color=colors[i])
    axs[2].plot(list(q)+[1],list(q**n*tidal2_lower[:,i]/tidal1_lower[:,i])+[1],'--',color=colors[i])
    if(i==0):
        axs[2].plot([0,0,0],[0,1,2],color='k',label='$M_{max}>%.1f M_\odot$, $c_s^2<%.1f$ upper bound'%(maxmass_min,cs2_max))
        axs[2].plot([0,0,0],[0,1,2],'--',color='k',label='$M_{max}<%.1f M_\odot$, $c_s^2>%.1f$ lower bound'%(maxmass_max,cs2_min))
    axs[2].legend(loc=1,prop={'size':15},frameon=False)
    #axs[2].set_title('upper and lower bounds')
    axs[2].set_xlabel('q',fontsize=25)
    axs[2].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=25)
    axs[2].set_ylim(0.5,22)
    axs[2].set_yscale('log')

f.colorbar(bound_log,ax=axs[2])
bound_log.set_clim(chip_mass[0], chip_mass[-1])
plt.xlim(0.7,1)
plt.show()


from fractions import Fraction
cs2=[Fraction(1,3),Fraction(1,2),Fraction(2,3),Fraction(5,6),Fraction(1,1)]
#cs2=[0.2,0.4]
eos_bound=[]
tidal_at_maxmass=[]
compactness_at_maxmass=[]
maxmass=2.0
for i in range(len(cs2)):
    eos_bound.append(EOS_CSS([density_surface_ofmaxmass(maxmass,50.,2000.,Maxmass,cs2[i]),0.,1.,cs2[i]]))
    mrbit_result=MassRadius_CSS(Maxmass(eos_bound[-1])[1],'MRBIT',eos_bound[-1])
    tidal_at_maxmass.append(mrbit_result[-1])
    compactness_at_maxmass.append(mrbit_result[2])
def get_for_chirp_mass_maxmass(m1_grid,m2_grid,eos):
    tidal1=m1_grid.copy()
    tidal2=m1_grid.copy()
    for i in range(len(m1_grid)):
        for j in range(len(m1_grid[0])):
            tidal1[i,j],tidal2[i,j] = get_tidal1tidal2(m1_grid[i,j],m2_grid[i,j],eos)
    return tidal1,tidal2

tidal1=[]
tidal2=[]
for i in range(len(cs2)):
    print('cs2=%.4f'%cs2[i])
    chip_mass= maxmass*np.linspace(0.4375, 0.7,8)
    q=np.linspace(0.7,1.,13)
    chip_mass_grid,q_grid = np.meshgrid(chip_mass,q)

    m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1
    tidal1_upper,tidal2_upper = get_for_chirp_mass_maxmass(m1_grid,m2_grid,eos_bound[i])
    tidal1.append(tidal1_upper)
    tidal2.append(tidal2_upper)
for i in range(len(chip_mass)):
    plt.plot(q,(tidal2[-1]/tidal1[-1])[:,i]*q**6)
    plt.plot(q,(tidal2[0]/tidal1[0])[:,i]*q**6)

import matplotlib.pyplot as plt
n=6
cmap = plt.cm.get_cmap('jet')
(np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int)
colors = cmap((np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int))
f, axs= plt.subplots(3,1, sharex=True,figsize=(10, 20))
for i in range(len(chip_mass)):
    axs[0].plot(q,(tidal2[0]/tidal1[0])[:,i]*q**n,color=colors[i],label='$M_{ch}/M_{max}$=%.4f'%(chip_mass[i]/maxmass))
    axs[0].legend(loc=1,prop={'size':14},frameon=False)
    axs[0].set_title('$c_s^2=%s$'%(cs2[0]))
    axs[0].set_xlabel('q',fontsize=18)
    axs[0].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=22)
    axs[0].set_ylim(0.5,2)
    #axs[0].set_yscale('log')

    axs[1].plot(q,(tidal2[2]/tidal1[2])[:,i]*q**n,color=colors[i],label='$M_{ch}/M_{max}$=%.4f'%(chip_mass[i]/maxmass))
    axs[1].legend(loc=1,prop={'size':14},frameon=False)
    axs[1].set_title('$c_s^2=%s$'%(cs2[2]))
    axs[1].set_xlabel('q',fontsize=18)
    axs[1].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=22)
    axs[1].set_ylim(0.5,2)
    #axs[1].set_yscale('log')

    bound_log=axs[2].scatter(q,q*0.0,c=[chip_mass[i]]*len(q),cmap=plt.cm.jet)
    axs[2].plot(q,(tidal2[4]/tidal1[4])[:,i]*q**n,color=colors[i],label='$M_{ch}/M_{max}$=%.4f'%(chip_mass[i]/maxmass))
    axs[2].legend(loc=1,prop={'size':14},frameon=False)
    axs[2].set_title('$c_s^2=%s$'%(cs2[4]))
    axs[2].set_xlabel('q',fontsize=18)
    axs[2].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=22)
    axs[2].set_ylim(0.5,2)
    #axs[2].set_yscale('log')

f.colorbar(bound_log,ax=axs[0])
bound_log.set_clim(chip_mass[0]/maxmass, chip_mass[-1]/maxmass)
f.colorbar(bound_log,ax=axs[1])
bound_log.set_clim(chip_mass[0]/maxmass, chip_mass[-1]/maxmass)
f.colorbar(bound_log,ax=axs[2])
bound_log.set_clim(chip_mass[0]/maxmass, chip_mass[-1]/maxmass)
plt.xlim(0.7,1)
plt.show()



# # For single quark star:
from fractions import Fraction
cs2=[Fraction(1,3),Fraction(1,2),Fraction(2,3),Fraction(5,6),Fraction(1,1)]
maxmass=2.0
beta=[]
Lambda=[]
mass=[]
N=50
for j in range(len(cs2)):
    eos_20=EOS_CSS([density_surface_ofmaxmass(maxmass,100.,1000.,Maxmass,cs2[j]),0.,1.,cs2[j]])
    beta_20=[]
    Lambda_20=[]
    mass_20=[]
    pc_20=Maxmass(eos_20)[1]*0.1**(np.linspace(0,1.2,N))
    for i in range(N):
        result=MassRadius_CSS(pc_20[i],'MRBIT',eos_20)
        mass_20.append(result[0])
        beta_20.append(result[2])
        Lambda_20.append(result[-1])
    beta.append(np.array(beta_20))
    Lambda.append(np.array(Lambda_20))
    mass.append(np.array(mass_20)/maxmass)
n=7
for j in range(len(cs2)):
    plt.plot(beta[j],cs2[j]**0.*np.array(Lambda[j])*np.array(beta[j])**n,label='$s=%s$'%cs2[j])
plt.legend()
plt.xlabel('$M/M_{max}$')
plt.ylabel('$\Lambda \\beta^%d$'%(n))

