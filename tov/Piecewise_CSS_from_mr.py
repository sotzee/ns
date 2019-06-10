#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:33:00 2019

@author: sotzee
"""

import numpy as np
from scipy import interpolate
from eos_class import EOS_BPS
import matplotlib.pyplot as plt

Preset_Pressure_final=1e-10
Preset_rtol=1e-6
cs2_rtol=1e-4
cs2_max=1.00
cs2_min=0.01
index_roundoff_compensate=2e-14

def log_array(array_lim,delta_factor,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
    k=np.log(delta_factor)/(N-1)
    a=(array_lim.max()-array_lim.min())/(np.exp(k*N)-1)
    return a,k,array_lim.min()+a*(np.exp(np.linspace(0,N*k,N+1))-1)
def log_array_extend(array_lim_min,a,k,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
    return array_lim_min+a*(np.exp(np.linspace(0,N*k,N+1))-1)
# =============================================================================
# def log_array_N(array,array_lim,delta_factor,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
#     k=np.log(delta_factor)/(N-1)
#     a=(array_lim.max()-array_lim.min())/(np.exp(k*N)-1)
#     return np.log((array-array_lim.min())/a+1)/k
# =============================================================================
def log_index(array_i,array_lim_min,a,k,N):
    array_i=array_i-index_roundoff_compensate
    #print(array_i,array_lim,a,k,-(-np.log(np.where(array_i<array_lim.min(),0,(array_i-array_lim.min())/a)+1)/k).astype('int'))
    return N-1-(N-np.log(np.where(array_i<array_lim_min,0,(array_i-array_lim_min)/a)+1)/k).astype('int')
#test:
# =============================================================================
# log_array_result=log_array(np.array([3,10]),2.,10)
# log_index_result=log_index(np.linspace(0,10,100),np.array([3,10]),log_array_result[0],log_array_result[1],10)
# print(log_array_result)
# print(log_index_result)
# =============================================================================

import scipy.optimize as opt
from MassRadius_hadronic import MassRadius
def MassRadius_compare(log_cs_2,pressure,eos,i):
    cs2=(cs2_max-cs2_min)/(1+np.exp(log_cs_2))+cs2_min
    print('###try %.10f, cs2=%f'%(log_cs_2,cs2))
    #print(eos.density_array,pressure,eos.eosDensity(pressure))
    eos.cs2[i]=cs2
    eos.density_array[i]=eos.density_array[i-1]+(eos.pressure_array[i]-eos.pressure_array[i-1])/cs2
    #print(eos.density_array,pressure,eos.eosDensity(pressure))
    #print(pressure,eos.eosDensity(pressure),eos.eosCs2(pressure))
    #print('begin MR calculation.')
    #print(pressure,density_init,cs2)
    #print(pressure,eos.eosDensity(pressure),eos.eosCs2(pressure))
    mass,radius=MassRadius(pressure,Preset_Pressure_final,Preset_rtol,'MR',eos)
    #print(mass,radius,interpolate.splev(mass, eos.R_from_M_tck, der=0))
    #print(radius-interpolate.splev(mass, eos.R_from_M_tck, der=0))
    return radius-interpolate.splev(mass, eos.R_from_M_tck, der=0)

class EOS_PiecewiseCSS(object):
    def __init__(self,args,init_type=0):
        if(init_type==0):
            pressure_lim,dp_factor,Np=args
            pressure_lim=np.array(pressure_lim)
            a,k,self.pressure_array=log_array(pressure_lim,dp_factor,Np)
            self.log_index_args=[pressure_lim.min(),a,k,Np]
        elif(init_type==1):
            pressure_min,a,k,Np=args
            self.pressure_array=log_array_extend(pressure_min,a,k,Np)
            self.log_index_args=[pressure_min,a,k,Np]
        self.density_array=np.array([eos_bps.eosDensity(self.pressure_array[0])])
        self.baryon_density=np.array([eos_bps.eosBaryonDensity(self.pressure_array[0])])
        self.cs2=np.array([eos_bps.eosCs2(self.pressure_array[0])])
        self.baryon_density_s=0.16
        self.chempo_surface=eos_bps.chempo_surface
        self.pressure_s=eos_bps.pressure_s
        self.density_s=eos_bps.density_s
        self.unit_mass=eos_bps.unit_mass
        self.unit_radius=eos_bps.unit_radius
        self.unit_N=eos_bps.unit_N
    def eosDensity(self,pressure):
        #print('eosDensity with pressure=%f'%pressure)
        #print(self.density_array)
        #print(pressure==self.pressure_array[1])
        #print(pressure,self.pressure_array[1])
        array_index=log_index(*([pressure]+self.log_index_args))
        #print(array_index)
        #array_index=np.max([array_index,np.full(np.shape(array_index),-1)],axis=0)
        array_index=np.min([array_index,np.full(np.shape(array_index),len(self.density_array)-1)],axis=0)
        #print(self.density_array)
# =============================================================================
#         if(array_index+1==len(self.cs2)):
#             print(array_index,pressure,self.pressure_array[array_index],pressure-self.pressure_array[array_index])
#             print(log_index(*([pressure]+self.log_index_args)))
#             print(log_index(*([self.pressure_array[array_index]]+self.log_index_args)))
#             print(self.pressure_array)
#             print(self.density_array)
#             print(self.cs2)
# =============================================================================
        density = (pressure-self.pressure_array[array_index])/self.cs2[array_index+1]+self.density_array[array_index]
        #print(density)
        return np.where(array_index<0,eos_bps.eosDensity(pressure),density)
    def eosBaryonDensity(self,pressure):
        array_index=log_index(*([pressure]+self.log_index_args))
        #array_index=np.max([array_index,np.full(np.shape(array_index),-1)],axis=0)
        array_index=np.min([array_index,np.full(np.shape(array_index),len(self.density_array)-1)],axis=0)
        B=(self.density_array[array_index]*self.cs2[array_index]-self.pressure_array[array_index])/(1.0+self.cs2[array_index])
        B=np.where(B>0,B,0)
        baryondensity = self.baryon_density[array_index]*((pressure+B)/(self.pressure_array[array_index]+B))**(1.0/(1.0+self.cs2[array_index]))
        #print(pressure,B,baryondensity)
        return np.where(array_index<0,eos_bps.eosBaryonDensity(pressure),baryondensity)
    def eosCs2(self,pressure):
        array_index=log_index(*([pressure]+self.log_index_args))
        #array_index=np.max([array_index,np.full(np.shape(array_index),-1)],axis=0)
        array_index=np.min([array_index,np.full(np.shape(array_index),len(self.density_array)-1)],axis=0)
        return np.where(array_index<0,eos_bps.eosCs2(pressure),self.cs2[array_index])
    def eosChempo(self,pressure):
        array_index=log_index(*([pressure]+self.log_index_args))
        #array_index=np.max([array_index,np.full(np.shape(array_index),-1)],axis=0)
        array_index=np.min([array_index,np.full(np.shape(array_index),len(self.density_array)-1)],axis=0)
        chempo=(pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
        return np.where(array_index<0,eos_bps.eosChempo(pressure),chempo)

# =============================================================================
#     def set_eos(self,mr_array):
#         plt.plot(mr_array[:,1],mr_array[:,0])
#         self.R_from_M_tck = interpolate.splrep(mr_array[:,0],mr_array[:,1], s=0)
#         i=1
#         density_init=self.density_array[i-1]+(self.pressure_array[i]-self.pressure_array[i-1])/self.cs2[i-1]
#         self.cs2=np.concatenate((self.cs2,np.array([self.cs2[-1]])),axis=0)
#         self.density_array=np.concatenate((self.density_array,np.array([density_init])))
#         sol=opt.newton(MassRadius_compare,np.log(1./self.cs2[-1]-1),rtol=tol,args=(self.pressure_array[i],self,i))
#         print(1./(1+np.exp(sol)))
# =============================================================================
    def set_eos(self,mr_array):
        fig,axis=plt.subplots(1,2)
        axis[0].plot(mr_array[:,1],mr_array[:,0])
        self.R_from_M_tck = interpolate.splrep(mr_array[:969,0],mr_array[:969,1], s=0)
        for i in range(1,len(self.pressure_array)):
            print('###################i=%d'%(i))
            print(self.cs2)
            print(self.density_array)
            print(self.baryon_density)
            density_init=self.density_array[i-1]+(self.pressure_array[i]-self.pressure_array[i-1])/self.cs2[i-1]
            self.cs2=np.concatenate((self.cs2,np.array([self.cs2[-1]])),axis=0)
            self.density_array=np.concatenate((self.density_array,np.array([density_init])))
            #print(self.cs2,self.density_array,self.eosDensity(self.pressure_array[i]))
            print(self.cs2)
            print(self.density_array)
            print(self.baryon_density)
            sol=opt.newton(MassRadius_compare,np.log((cs2_max-cs2_min)/(self.cs2[-1]-cs2_min)-1),rtol=cs2_rtol,args=(self.pressure_array[i],self,i))
            print('root finding finish...')
            mr=MassRadius(self.pressure_array[i],Preset_Pressure_final,Preset_rtol,'MR',self)
            print(mr[1],interpolate.splev(mr[0], self.R_from_M_tck, der=0))
            self.cs2[-1]=np.min([(cs2_max-cs2_min)/(1+np.exp(sol))+cs2_min,1.])
            self.density_array[i]=self.density_array[i-1]+(self.pressure_array[i]-self.pressure_array[i-1])/self.cs2[-1]
            self.baryon_density=np.concatenate((self.baryon_density,np.array([self.eosBaryonDensity(self.pressure_array[i])])))
            axis[1].plot([self.density_array[i]],[self.pressure_array[i]],'*')
            axis[1].plot([eos_bps.eosDensity(self.pressure_array[i])],[self.pressure_array[i]],'o')
        #some_radius=interpolate.splev(some_mass, R_from_M_tck, der=0)
        
eos_bps=EOS_BPS()
# =============================================================================
# Preset_Pressure_final=1e-10
# Preset_rtol=1e-6
# mr_bps=[] 
# for pc_i in log_array(np.array([1,1100]),2000,1000)[2]:
#     mr_bps.append(MassRadius(pc_i,Preset_Pressure_final,Preset_rtol,'MR',eos_bps))
# mr_bps=np.array(mr_bps)
# plt.plot(mr_bps[:,1],mr_bps[:,0],'.')
# np.savetxt('sly4_mr.txt',mr_bps)
# =============================================================================
mr_bps=np.loadtxt('sly4_mr.txt')

#a=EOS_PiecewiseCSS([[3.,10.],10.,5]))       #[3.0, 0.4170938377798279, 0.5756462732485115, 5]
#a=EOS_PiecewiseCSS([[3.,30.],40.,7])        #[3.0, 0.37000266298316076, 0.6148132423523227, 7]
#a=EOS_PiecewiseCSS([3.0, 0.4170938377798279, 0.5756462732485115, 12],init_type=1)

a=EOS_PiecewiseCSS([[3.,10.],10.,5])
a.set_eos(mr_bps)


# =============================================================================
# a=EOS_PiecewiseCSS([3,80],5,20)
# print(a.pressure_array)
# print(a.density_array,a.eosDensity(np.array([0.1,1,2,3,4,5,6,7,8,9,10,20,50,80,100])))
# a.density_array=np.concatenate((a.density_array,np.array([167.287152])))
# print(a.density_array)
# a.cs2=np.concatenate((a.cs2,np.array([((4.53234784-3.)/(169.5287152-167.287152))])))
# 
# print(a.density_array,a.eosDensity(np.array([0.1,1,2,3,4,5,6,7,8,9,10,20,50,80,100])))
# 
# print(a.baryon_density)
# print(a.cs2)
# =============================================================================
# =============================================================================
#     def eosBaryonDensity(self,pressure):
#         #print('hello')
#         #print(pressure)
#         array_index=-int_helper+(int_helper+np.log((pressure-self.pressure_lim.min())/self.pressure_array_a+1)/self.pressure_array_k).astype('int')
#         #print(array_index)
#         array_index=np.max([array_index,np.array([-1])])
#         #print(array_index)
#         B=(self.density_array[array_index]-self.pressure_array[array_index]/self.cs2[array_index])/(1.0+1.0/self.cs2[array_index])
#         baryondensity = self.baryon_density[array_index]*((pressure+B)/(self.pressure_array[array_index]+B))**(1.0/(1.0+self.cs2[array_index]))
#         return np.where(array_index<=0,eos_bps.eosBaryonDensity(pressure),baryondensity)
#     def eosCs2(self,pressure):
#         array_index=-int_helper+(int_helper+np.log((pressure-self.pressure_lim.min())/self.pressure_array_a+1)/self.pressure_array_k).astype('int')
#         array_index=np.max([array_index,np.array([-1])])
#         return np.where(array_index<=0,eos_bps.eosCs2(pressure),self.cs2[array_index])
#     def eosChempo(self,pressure):
#         array_index=-int_helper+(int_helper+np.log((pressure-self.pressure_lim.min())/self.pressure_array_a+1)/self.pressure_array_k).astype('int')
#         array_index=np.max([array_index,np.array([-1])])
#         chempo=(pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
#         return np.where(array_index<=0,eos_bps.eosChempo(pressure),chempo)
# =============================================================================
