#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:37:32 2019

@author: sotzee
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.constants import c,G,e
dlnx_cs2=1e-6

class EOS_intepolation(object):
    def __init__(self,chempo_min,ns,baryondensity,density,pressure):
        self.eos_array=np.array([baryondensity,density,pressure])
        self.eosPressure_frombaryon = interp1d(baryondensity,pressure, kind='linear')
        self.eosDensity = interp1d(pressure,density, kind='linear')
        self.eosBaryonDensity = interp1d(pressure,baryondensity, kind='linear')
        self.chempo_surface=chempo_min
        self.baryon_density_s=ns
        self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5/1000
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)

def eos_array_thicker(eos_array,det_baryon_density_max):
    eos_array_det=eos_array[:,1:]-eos_array[:,:-1]
    eos_array_new=eos_array_det[:,:,np.newaxis]*np.linspace(0,1,1+int(eos_array_det[0].max()/det_baryon_density_max))[np.newaxis,np.newaxis,1:]
    eos_array_new=(eos_array_new+eos_array[:,:-1,np.newaxis]).transpose((0,1,2)).reshape((3,-1))
    return np.concatenate((eos_array[:,[0]],eos_array_new),axis=1)

from scipy.stats import norm
class EOS_variation_intepolation(object):
    def __init__(self,around_baryon_density,sigma_baryon_density,magnitude_pressure_ratio,args):
        self.EOS_intepolation_=EOS_intepolation(*args)
        self.around_baryon_density=around_baryon_density
        self.sigma_baryon_density=sigma_baryon_density
        self.magnitude_pressure_ratio=magnitude_pressure_ratio
        self.eos_array=eos_array_thicker(self.EOS_intepolation_.eos_array,sigma_baryon_density/10.)
        self.chempo_surface=self.EOS_intepolation_.chempo_surface
        self.baryon_density_s=self.EOS_intepolation_.baryon_density_s
        self.pressure_s=self.EOS_intepolation_.pressure_s
        self.density_s=self.EOS_intepolation_.density_s
        self.unit_mass=self.EOS_intepolation_.unit_mass
        self.unit_radius=self.EOS_intepolation_.unit_radius
        self.unit_N=self.EOS_intepolation_.unit_N
        
        def variation_factor(x,center,sigma):
            return (self.magnitude_pressure_ratio-1)*norm(center,sigma).pdf(x)*(2*np.pi)**0.5*sigma+1
        self.eosPressure_frombaryon = interp1d(self.eos_array[0],variation_factor(self.eos_array[0],self.around_baryon_density,self.sigma_baryon_density)*self.eos_array[2], kind='linear')
        self.eosDensity = interp1d(variation_factor(self.eos_array[0],self.around_baryon_density,self.sigma_baryon_density)*self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity = interp1d(variation_factor(self.eos_array[0],self.around_baryon_density,self.sigma_baryon_density)*self.eos_array[2],self.eos_array[0], kind='linear')
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    
path='../EOS_CompOSE/'
eos=[]
EOS_LIST=['APR','CMF','SKA','SLY4']
EOS_ns_LIST=[0.16,0.15,0.155,0.159]
for EOS_i,EOS_ns_i in zip(EOS_LIST,EOS_ns_LIST):
    nb=np.loadtxt(path+EOS_i+'/eos.nb',skiprows=2)
    mn,mp=np.loadtxt(path+EOS_i+'/eos.thermo',max_rows=1)[:2]
    thermo=np.loadtxt(path+EOS_i+'/eos.thermo',skiprows=1)
    chempo_min=(thermo[0,5]+1)*mn
    eos.append(EOS_intepolation(chempo_min,EOS_ns_i,nb,(thermo[:,9]+1)*(nb*mn),thermo[:,3]*nb))

import matplotlib.pyplot as plt
import show_properity as sp
fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
sp.show_eos(axes,eos,0,1,500,pressure_range=[0.01,300,'log'],legend=EOS_LIST)


from MassRadius_hadronic import MassRadius
from FindMaxmass import Maxmass
from Find_OfMass import Properity_ofmass

MRBIT_LIST=[]
eos_properity=[]
for eos_i in eos:
    #print len(MRBIT_LIST)
    MRBIT_LIST.append([])
    pc_max,maxmass=Maxmass(1e-8,1e-5,eos_i)[1:3]
    print pc_max,maxmass
    onepointfour_result=Properity_ofmass(1.4,10,pc_max,MassRadius,1e-8,1e-5,1,eos_i)
    pc_min=Properity_ofmass(1.0,10,pc_max,MassRadius,1e-8,1e-5,1,eos_i)[0]
    pc=pc_min*np.exp(np.linspace(0,np.log(pc_max/pc_min),50))
    eos_properity.append([pc_max,maxmass,onepointfour_result[0],onepointfour_result[2],onepointfour_result[7]])
    for pc_i in pc:
        MRBIT_LIST[-1].append(MassRadius(pc_i,1e-8,1e-5,'MRBIT',eos_i))
MRBIT_LIST=np.array(MRBIT_LIST)
eos_properity=np.array(eos_properity)

for i in range(len(eos)):
    plt.plot(MRBIT_LIST[i,:,1],MRBIT_LIST[i,:,0],'.')
    np.savetxt(EOS_LIST[i]+'_TOV_result',MRBIT_LIST[i])
    np.savetxt(EOS_LIST[i]+'_EOS_data',eos[i].eos_array.transpose())
plt.xlabel('radius(km)')
plt.ylabel('mass($M_\odot$)')







path='../EOS_CompOSE/'
EOS_i='SKA'
EOS_ns_i=0.155
n_center_list=np.linspace(0.05,0.8,31)
eos_variation=[]
for n_center in n_center_list:
    nb=np.loadtxt(path+EOS_i+'/eos.nb',skiprows=2)
    mn,mp=np.loadtxt(path+EOS_i+'/eos.thermo',max_rows=1)[:2]
    thermo=np.loadtxt(path+EOS_i+'/eos.thermo',skiprows=1)
    chempo_min=(thermo[0,5]+1)*mn
    eos_variation.append(EOS_variation_intepolation(n_center,0.02,1.04,[chempo_min,EOS_ns_i,nb,(thermo[:,9]+1)*(nb*mn),thermo[:,3]*nb]))
import matplotlib.pyplot as plt
import show_properity as sp
fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
sp.show_eos(axes,eos_variation,0,5,500,pressure_range=[0.01,500,'log'],legend=EOS_LIST)

eos_properity=[]
properity_ofmass=[]
ofmass=[1.2,1.4,1.6,1.8,2.0]
color_list=['b','y','g','r','c']
for eos_i in eos_variation:
    #print len(MRBIT_LIST)
    properity_ofmass.append([])
    pc_max,maxmass=Maxmass(1e-8,1e-4,eos_i)[1:3]
    eos_properity.append([pc_max,maxmass])
    print pc_max,maxmass
    for ofmass_i in ofmass:
        print ofmass_i
        properity_ofmass[-1].append(Properity_ofmass(ofmass_i,10,pc_max,MassRadius,1e-8,1e-6,1,eos_i))
eos_properity=np.array(eos_properity)
properity_ofmass=np.array(properity_ofmass)
for i in range(len(ofmass)):
    np.savetxt(path+'Variation/Variation_properity_ofmass_%.1f.txt'%ofmass[i],properity_ofmass[:,i,:])


fig=plt.figure(figsize=(8,6))
plt.plot([],[],' ',label='in MeV fm$^{-3}$')
for i in range(5):
    plt.plot(n_center_list,properity_ofmass[:,i,0]/properity_ofmass[-1,i,0],label='$p_{c%.1f}$=%.2f'%(ofmass[i],properity_ofmass[-1,i,0]),color=color_list[i])
    for i in range(5):
        plt.plot([eos_variation[i].eosBaryonDensity(properity_ofmass[-1,i,0])],[1],'o',color=color_list[i])
plt.xlabel('n(fm$^{-3}$)')
plt.ylabel('$(\delta p_c/p_c)_M$')
plt.legend()
fig.savefig(path+'Variation/Variation_properity_0')

y_label_MRBIT=['$(\delta M/M)_M$','$(\delta R/R)_M$','$(\delta \\beta/\\beta)_M$','$(\delta M_{unbind}/M_{unbind})_M$','$(\delta \\bar I/\\bar I)_M$','$(\delta k_2/k_2)_M$','$(\delta \Lambda/\Lambda)_M$']
for k in range(len(y_label_MRBIT)):
    fig=plt.figure(figsize=(8,6))
    for i in range(5):
        plt.plot(n_center_list,properity_ofmass[:,i,k+1]/properity_ofmass[-1,i,k+1],label='M=%.1f M$\odot$'%ofmass[i],color=color_list[i])
    plt.xlabel('n(fm$^{-3}$)')
    plt.ylabel(y_label_MRBIT[k])
    plt.legend()
    for i in range(5):
        plt.plot([eos_variation[i].eosBaryonDensity(properity_ofmass[-1,i,0])],[1],'o',color=color_list[i])
    fig.savefig(path+'Variation/Variation_properity_%d'%(k+1))

# =============================================================================
# for i in range(5):
#     plt.plot(n_center_list,properity_ofmass[:,i,2]/properity_ofmass[-1,i,2],label='M=%.1f M$\odot$'%ofmass[i])
# plt.xlabel('n(fm$^{-3}$)')
# plt.ylabel('$(\delta R/R)_M$')
# plt.legend()
# 
# ofmass=[1.2,1.4,1.6,1.8,2.0]
# for i in range(5):
#     plt.plot(np.linspace(0.05,0.8,31),properity_ofmass[:,i,5]/properity_ofmass[-1,i,5],label='M=%.1f M$\odot$'%ofmass[i])
# plt.xlabel('n(fm$^{-3}$)')
# plt.ylabel('$(\delta \\bar I/\\bar I)_M$')
# plt.legend()
# 
# ofmass=[1.2,1.4,1.6,1.8,2.0]
# for i in range(5):
#     plt.plot(np.linspace(0.05,1,20),properity_ofmass[:,i,7]/properity_ofmass[-1,i,7],label='M=%.1f M$\odot$'%ofmass[i])
# plt.xlabel('n(fm$^{-3}$)')
# plt.ylabel('$(\delta \Lambda/\Lambda)_M$')
# plt.legend()
# =============================================================================


# =============================================================================
# MRBIT_ofmass=[]
# for i in range(len(eos_variation)):
#     print i
#     #print len(MRBIT_LIST)
#     MRBIT_ofmass.append([])
#     for j in range(len([1.2,1.4,1.6,1.8,2.0])):
#         MRBIT_ofmass[-1].append(MassRadius(properity_ofmass[i][j][0],1e-8,1e-6,'MRBIT',eos_i))
# MRBIT_ofmass=np.array(MRBIT_ofmass)
# ofmass=[1.2,1.4,1.6,1.8,2.0]
# for i in range(len(ofmass)):
#     np.savetxt(path+'MRBIT_variation_%.1f.txt'%ofmass[i],MRBIT_ofmass[:,i,:])
# 
# 
# y_label_MRBIT=['$(\delta R/R)_M$','$(\delta \\beta/\\beta)_M$','$(\delta M_{unbind}/M_{unbind})_M$','$(\delta \\bar I/\\bar I)_M$','$(\delta k_2/k_2)_M$','$(\delta \Lambda/\Lambda)_M$']
# for k in range(len(y_label_MRBIT)):
#     plt.figure(figsize=(8,6))
#     for i in range(5):
#         plt.plot(n_center_list,MRBIT_ofmass[:,i,k+1]/MRBIT_ofmass[-1,i,k+1],label='M=%.1f M$\odot$'%ofmass[i],color=color_list[i])
#     plt.xlabel('n(fm$^{-3}$)')
#     plt.ylabel(y_label_MRBIT[k])
#     plt.legend()
#     for i in range(5):
#         plt.plot([eos_variation[i].eosBaryonDensity(properity_ofmass[-1,i,0])],[1],'o',color=color_list[i])
# =============================================================================
