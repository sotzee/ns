#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:54:09 2018

@author: sotzee
"""
# =============================================================================
# import config_hadronic as config
# from setParameter import setParameter_hadronic
# import numpy as np
# from eos_class import EOS_item
# import matplotlib.pyplot as plt
# 
# unit_MeV4_to_MeVfm3=1.302e-7
# m0=939.5654
# n_s=0.16
# A0=m0**4/np.pi**2*unit_MeV4_to_MeVfm3
# def beyond_stability_flag(p,eos):
#     cs2=eos.eosCs2(p)
#     chempo=eos.eosChempo(p)
#     #baryondensity=eosBaryonDensity(p)
#     energydensity=eos.eosDensity(p)
#     if(1-cs2-4.*A0*((chempo/m0)**2-1)**2.5/(45*chempo*(energydensity+p)/m0)>0):
#         flag_original = True
#     else:
#         flag_original = False
#     if(cs2<(energydensity-p/3)/(energydensity+p)):
#         flag_modified=True
#     else:
#         flag_modified=False
#     return [flag_original,flag_modified]
# 
# 
# fig, axs = plt.subplots(1,2,sharex=True, sharey=True,figsize=(12, 4))
# fig.subplots_adjust(hspace=0.0)
# fig.subplots_adjust(wspace=0.0)
# parameter=setParameter_hadronic(config.baryon_density0,config.Preset_pressure1,config.baryon_density1,config.Preset_pressure2,config.baryon_density2,config.Preset_pressure3,config.baryon_density3)
# for i in range(np.size(parameter)/np.size(parameter[0])):
#     parameter[i]=EOS_item([config.baryon_density0,parameter[i][0],config.baryon_density1,parameter[i][1],config.baryon_density2,parameter[i][3],config.baryon_density3])
# for i in range(len(parameter)):
#     eos=config.eos_config(parameter[i].args)
#     p=np.linspace(1,300,100)
#     rho=np.linspace(1,300,100)
#     flag=[True,True]
#     for i in range(len(p)):
#         rho[i]=eos.eosDensity(p[i])
#         stability_flag = beyond_stability_flag(p[i],eos)
#         flag[0]=flag[0] and stability_flag[0]
#         flag[1]=flag[1] and stability_flag[1]
#     if(flag[0]):
#         axs[0].plot(rho,p)
#         axs[1].plot(rho,p)
#     else:
#         axs[0].plot(rho,p)
#     plt.xlim(0,800)
# =============================================================================
import config_hadronic as config
from setParameter import setParameter_hadronic
import numpy as np
from eos_class import EOS_item
import matplotlib.pyplot as plt

unit_MeV4_to_MeVfm3=1.302e-7
m0=939.5654
n_s=0.16
A0=m0**4/np.pi**2*unit_MeV4_to_MeVfm3
def beyond_stability_flag(p,eos):
    cs2=eos.eosCs2(p)
    chempo=eos.eosChempo(p)
    #baryondensity=eosBaryonDensity(p)
    energydensity=eos.eosDensity(p)
    if(1-cs2-4.*A0*((chempo/m0)**2-1)**2.5/(45*chempo*(energydensity+p)/m0)>0):
        flag_original=True
    else:
        flag_original=False
    if(cs2<(energydensity-p/3)/(energydensity+p)):
        flag_modified=True
    else:
        flag_modified=False
    return [flag_original,flag_modified]

fig, axs = plt.subplots(1,3,sharex=True, sharey=True,figsize=(12, 4))
fig.subplots_adjust(hspace=0.0)
fig.subplots_adjust(wspace=0.0)
parameter=setParameter_hadronic(config.baryon_density0,config.Preset_pressure1,config.baryon_density1,config.Preset_pressure2,config.baryon_density2,config.Preset_pressure3,config.baryon_density3)
for i in range(np.size(parameter)/np.size(parameter[0])):
    parameter[i]=EOS_item([config.baryon_density0,parameter[i][0],config.baryon_density1,parameter[i][1],config.baryon_density2,parameter[i][3],config.baryon_density3])
for i in range(len(parameter)):
    eos=config.eos_config(parameter[i].args)
    p=np.linspace(1,300,100)
    rho=np.linspace(1,300,100)
    flag=[True,True]
    for i in range(len(p)):
        rho[i]=eos.eosDensity(p[i])
        #flag=flag and beyond_stability_flag(p[i],eos)
        stability_flag = beyond_stability_flag(p[i],eos)
        flag[0]=flag[0] and stability_flag[0]
        flag[1]=flag[1] and stability_flag[1]
    axs[0].plot(rho,p)
    if(flag[0]):
        axs[1].plot(rho,p)
    if(flag[1]):
        axs[2].plot(rho,p)
# =============================================================================
#     if((not flag[0]) and flag[1]):
#         print eos.args
# =============================================================================
axs[0].set_title('causal constrain')
axs[1].set_title('stability constrain')
axs[2].set_title('modified stability constrain')
axs[0].set_xlabel('energy density ($MeV fm^-3$)')
axs[1].set_xlabel('energy density ($MeV fm^-3$)')
axs[2].set_xlabel('energy density ($MeV fm^-3$)')
axs[0].set_ylabel('pressure ($MeV fm^-3$)')
plt.xlim(0,800)
plt.savefig('figure/eos_constrain_to_poly')