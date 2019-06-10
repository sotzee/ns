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
from unitconvert import toMevfm
dlnx_cs2=1e-6

class EOS_intepolation(object):
    def __init__(self,chempo_min,ns,baryondensity,density,pressure):
        self.eos_array=np.array([baryondensity,density,pressure])
        self.eosPressure_frombaryon = interp1d(baryondensity,pressure, kind='quadratic')
        self.eosDensity = interp1d(pressure,density, kind='quadratic')
        self.eosBaryonDensity = interp1d(pressure,baryondensity, kind='quadratic')
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
    def __init__(self,around_baryon_density,sigma_baryon_density,magnitude_pressure_ratio,EOS_intepolation_):
        self.EOS_intepolation_=EOS_intepolation_
        self.around_baryon_density=around_baryon_density
        self.sigma_baryon_density=sigma_baryon_density
        self.magnitude_pressure_ratio=magnitude_pressure_ratio
        self.eos_array=eos_array_thicker(self.EOS_intepolation_.eos_array,sigma_baryon_density/10.)
        self.eos_array[0]=self.EOS_intepolation_.eosBaryonDensity(self.eos_array[2])
        self.eos_array[1]=self.EOS_intepolation_.eosDensity(self.eos_array[2])
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

# =============================================================================
# def fitting_f(x, A, B, C, D):
#     return A*x**B + C*x**D
# class EOS_simple_fitting(object):
#     def __init__(self,chempo_min,ns,args):
#         self.args=args
#         self.chempo_surface=chempo_min
#         self.baryon_density_s=ns
#         self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
#         self.density_s=self.eosDensity(self.pressure_s)
#         self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#         self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5/1000
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#     def eosPressure_frombaryon(self,pressure):
#         return 0
#     def eosDensity(self,pressure):
#         return fitting_f(*([pressure]+list(self.args)))
#     def eosChempo(self,pressure):
#         return self.chempo_surface
#     def eosBaryonDensity(self,pressure):
#         return 0
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
# =============================================================================

path='../EOS_CompOSE/'
eos=[]
EOS_LIST=['APR','CMF','SKA']
EOS_ns_LIST=[0.16,0.15,0.155]
# =============================================================================
# EOS_LIST=['APR','SLY4']
# EOS_ns_LIST=[0.16,0.159]
# =============================================================================
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

path='../EOS_Tables_Ozel/'
eos=[]
EOS_LIST=['ap4','mpa1','wff1','sly']
EOS_ns_LIST=[0.16,0.16,0.16,0.159]
for EOS_i,EOS_ns_i in zip(EOS_LIST,EOS_ns_LIST):
    eos_array_i=np.loadtxt(path+EOS_i+'.dat',skiprows=0)
    nb=toMevfm(eos_array_i[:,0]/1.66*1e24,'baryondensity')
    pr=toMevfm(eos_array_i[:,1],'density')
    ep=toMevfm(eos_array_i[:,2],'density')
    chempo_min=(ep[0]+pr[0])
    eos.append(EOS_intepolation(chempo_min,EOS_ns_i,nb,ep,pr))
import matplotlib.pyplot as plt
import show_properity as sp
fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
sp.show_eos(axes,eos,0,1,500,pressure_range=[0.01,300,'log'],legend=EOS_LIST)

# =============================================================================
# from scipy.optimize import curve_fit
# from FindMaxmass import Maxmass
# index_bondary=2.
# for i in range(4):
#     pc_max=Maxmass(1e-8,1e-5,eos[i])[1]
#     logic=np.logical_and(eos[i].eos_array[1]>100,eos[i].eos_array[2]<pc_max)
#     match_pr=eos[i].eos_array[2,logic][0]
#     match_ep=eos[i].eos_array[1,logic][0]
#     def get_A(B,C,D):
#         return (match_pr-C*match_ep**D)/match_ep**B
#     def fitting_f(x, B, C, D):
#         A=(match_pr-C*match_ep**D)/match_ep**B
#         return A*x**B + C*x**D
#     popt, pcov = curve_fit(fitting_f, eos[i].eos_array[1,logic], eos[i].eos_array[2,logic], bounds=([index_bondary, 0, 0], [10, 10, index_bondary]))
#     abcd=[get_A(*popt)]+list(popt)
#     print(abcd)
#     plt.plot(eos[i].eos_array[1,logic],eos[i].eos_array[2,logic])
#     plt.plot(eos[i].eos_array[1,logic],fitting_f(*([eos[i].eos_array[1,logic]]+list(popt))))
# =============================================================================

from FindMaxmass import Maxmass
from MassRadius_hadronic import MassRadius
from Find_OfMass import Properity_ofmass
MRBIT_LIST=[]
eos_properity=[]
for eos_i in eos:
    #print len(MRBIT_LIST)
    MRBIT_LIST.append([])
    pc_max,maxmass=Maxmass(1e-8,1e-5,eos_i)[1:3]
    print(pc_max,maxmass)
    onepointfour_result=Properity_ofmass(1.4,10,pc_max,MassRadius,1e-8,1e-5,1,eos_i)
    pc_min=Properity_ofmass(1.0,10,pc_max,MassRadius,1e-8,1e-5,1,eos_i)[0]
    pc_min=10
    pc=pc_min*np.exp(np.linspace(0,np.log(pc_max/pc_min),50))
    eos_properity.append([pc_max,maxmass,onepointfour_result[0],onepointfour_result[2],onepointfour_result[7]])
    for pc_i in pc:
        MRBIT_LIST[-1].append(MassRadius(pc_i,1e-8,1e-5,'MRBIT',eos_i))
MRBIT_LIST=np.array(MRBIT_LIST)
eos_properity=np.array(eos_properity)
for i in range(len(eos)):
    plt.plot(MRBIT_LIST[i,:,1],MRBIT_LIST[i,:,0],'.')
    np.savetxt(path+EOS_LIST[i]+'_TOV_result',MRBIT_LIST[i])
    np.savetxt(path+EOS_LIST[i]+'_EOS_data',eos[i].eos_array.transpose())
plt.xlabel('radius(km)')
plt.ylabel('mass($M_\odot$)')




import matplotlib.pyplot as plt
import show_properity as sp
n_center_list=np.linspace(0.05,1.3,51)
eos_variation=[]
for i in range(len(eos)):
    eos_variation.append([])
    for n_center in n_center_list:
        eos_variation[i].append(EOS_variation_intepolation(n_center,0.01,1.02,eos[i]))
    fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
    sp.show_eos(axes,eos_variation[i],0,5,500,pressure_range=[0.01,1100,'log'],legend=EOS_LIST)


eos_properity=[]
properity_ofmass=[]
properity_ofmaxmass=[]
ofmass=[1.2,1.4,1.6,1.8,2.0]
color_list=['b','y','g','r','c']
for i in range(len(eos_variation)):
    eos_properity.append([])
    properity_ofmass.append([])
    properity_ofmaxmass.append([])
    for eos_variation_i in eos_variation[i]:
        #print len(MRBIT_LIST)
        properity_ofmass[i].append([])
        pc_max,maxmass=Maxmass(1e-8,1e-5,eos_variation_i)[1:3]
        eos_properity[i].append([pc_max,maxmass])
        properity_ofmaxmass[i].append([pc_max]+list(MassRadius(pc_max,1e-8,1e-6,'MRBIT',eos_variation_i)))
        print(pc_max,maxmass)
        for ofmass_i in ofmass:
            print(ofmass_i)
            properity_ofmass[i][-1].append(Properity_ofmass(ofmass_i,10,pc_max,MassRadius,1e-8,1e-6,1,eos_variation_i))
eos_properity=np.array(eos_properity)
properity_ofmass=np.array(properity_ofmass)
properity_ofmaxmass=np.array(properity_ofmaxmass)
import pickle
f=open(path+'Variation/result.dat','wb')
pickle.dump([eos_properity,properity_ofmass,properity_ofmaxmass],f)
f.close()
# =============================================================================
# np.savetxt(path+'Variation/Variation_properity_ofmaxmass.txt',properity_ofmaxmass)
# for i in range(len(ofmass)):
#     np.savetxt(path+'Variation/Variation_properity_ofmass_%.1f.txt'%ofmass[i],properity_ofmass[:,i,:])
# =============================================================================

fig=plt.figure(figsize=(8,6))
for i in range(len(eos)):
    plt.plot(n_center_list,eos_properity[i,:,1]/eos_properity[i,-1,1],label=EOS_LIST[i],color=color_list[i])
    plt.plot([eos_variation[i][-1].eosBaryonDensity(eos_properity[i,-1,0])],[eos_properity[i,-1,1]/eos_properity[i,-1,1]],'o',color=color_list[i])
plt.legend()
plt.xlabel('n(fm$^{-3}$)')
plt.ylabel('$(\delta M/M)_{max}$')
fig.savefig(path+'Variation/Variation_properity_maxmass')

# =============================================================================
# y_label_Maxmass=['$(\delta p_{c}/p_{c})_{max}$','$(\delta M/M)_{max}$','$(\delta R/R)_{max}$','$(\delta \\beta/\\beta)_{max}$','$(\delta M_{unbind}/M_{unbind})_{max}$','$(\delta \\bar I/\\bar I)_{max}$','$(\delta k_2/k_2)_{max}$','$(\delta \Lambda/\Lambda)_{max}$']
# for k in range(len(y_label_Maxmass)):
#     fig=plt.figure(figsize=(8,6))
#     plt.plot(n_center_list,properity_ofmaxmass[:,k]/properity_ofmaxmass[-1,k])
#     plt.xlabel('n(fm$^{-3}$)')
#     plt.ylabel(y_label_Maxmass[k])
#     plt.legend()
#     for i in range(5):
#         plt.plot([eos_variation[i].eosBaryonDensity(properity_ofmaxmass[-1,0])],[1],'o')
#     fig.savefig(path+'Variation/Variation_properity_ofmaxmass')
# =============================================================================

linestyles = ['-', '--', ':', '-.']
fig=plt.figure(figsize=(8,6))
plt.plot([],[],' ',label='in MeV fm$^{-3}$')
for j in range(len(eos)):
    for i in range(5):
        plt.plot(n_center_list,properity_ofmass[j,:,i,0]/properity_ofmass[j,-1,i,0],label=EOS_LIST[j]+' $p_{c%.1f}$=%.2f'%(ofmass[i],properity_ofmass[j,-1,i,0]),linestyle=linestyles[j],color=color_list[i])
    for i in range(5):
        plt.plot([eos_variation[j][i].eosBaryonDensity(properity_ofmass[j,-1,i,0])],[1],'o',color=color_list[i])
    plt.xlabel('n(fm$^{-3}$)')
    plt.ylabel('$(\delta p_c/p_c)_M$')
    plt.legend()
fig.savefig(path+'Variation/Variation_properity_pc')


fig_name=['M','R','beta','M_bind','I','k2','tidal']
y_label_MRBIT=['$(\delta M/M)_M$','$(\delta R/R)_M$','$(\delta \\beta/\\beta)_M$','$(\delta M_{unbind}/M_{unbind})_M$','$(\delta \\bar I/\\bar I)_M$','$(\delta k_2/k_2)_M$','$(\delta \Lambda/\Lambda)_M$']
for k in range(len(y_label_MRBIT)):
    fig=plt.figure(figsize=(8,6))
    for j in range(len(eos)):
        for i in range(5):
            plt.plot(n_center_list,properity_ofmass[j,:,i,k+1]/properity_ofmass[j,-1,i,k+1],label=EOS_LIST[j]+' M=%.1f M$\odot$'%ofmass[i],linestyle=linestyles[j],color=color_list[i])
        plt.xlabel('n(fm$^{-3}$)')
        plt.ylabel(y_label_MRBIT[k])
        plt.legend()
        for i in range(5):
            plt.plot([eos_variation[j][i].eosBaryonDensity(properity_ofmass[j,-1,i,0])],[1],'o',color=color_list[i])
    fig.savefig(path+'Variation/Variation_properity_'+fig_name[k])

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
