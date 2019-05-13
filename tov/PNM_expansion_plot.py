#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:41:34 2018

@author: sotzee
"""

from PNM_expansion import EOS_CSS,EOS_SLY4_match_EXPANSION_PNM
import matplotlib.pyplot as plt
import numpy as np
import cPickle

path = "./"
dir_name='Lambda_PNM_margueron_calculation_parallel'
#dir_name='Lambda_PNM_around_vacuum_calculation_parallel'
def read_file(file_name):
    f_file=open(file_name,'rb')
    content=np.array(cPickle.load(f_file))
    f_file.close()
    return content

args=read_file(path+dir_name+'/Lambda_hadronic_calculation_args.dat')  #!!!args will be redefined by remove the first trial parameters below
args_shape=args.shape[:-1]
eos_logic=np.reshape(read_file(path+dir_name+'/Lambda_hadronic_calculation_eos_success.dat'),args_shape)    #successful calculated (stable and causal)
eos_flat=read_file(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat')                         #all successful eos store with maximum mass constrain

maxmass_result=np.reshape(read_file(path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'),(-1,3))
maxmass_result=np.full(args_shape+(3,),np.array([0,0,1]),dtype='float')
maxmass_result[eos_logic]=read_file(path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat')
maxmass_result=maxmass_result.transpose((len(args_shape),)+tuple(range(len(args_shape))))
logic=read_file(path+dir_name+'/Lambda_hadronic_calculation_eos_success_maxmass.dat').reshape(args_shape)   #all successful eos store with maximum mass constrain
logic_maxmass=maxmass_result[1]>=2
logic_causality=maxmass_result[2]<1

Properity_onepointfour=np.full(args_shape+(6,),np.array([0,1.4,0,0,0,10000]),dtype='float')
Properity_onepointfour[logic]=read_file(path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat')
Properity_onepointfour=Properity_onepointfour.transpose((len(args_shape),)+tuple(range(len(args_shape))))
logic_tidal800=Properity_onepointfour[-1]<=800

mass_beta_Lambda_result=read_file(path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat')
chirp_q_Lambdabeta6_Lambda1Lambda2=read_file(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat')

N_m,N_E,N_L,N_K,N_Q,N_Z=np.shape(args)[:-1]
n_s   = args[0,0,0,0,0,0,0]
m     = args[0,0,0,0,0,0,1]
m_eff = args[:,0,0,0,0,0,2]
E_pnm = args[0,:,0,0,0,0,3]
L_pnm = args[0,0,:,0,0,0,4]
K_pnm = args[0,0,0,:,0,0,5]
Q_pnm = args[0,0,0,0,:,0,6]
Z_pnm = args[0,0,0,0,0,:,7]
# =============================================================================
# N_E,N_L,N_K,N_Q=np.shape(args)[0:-1]
# n_s   = args[0,0,0,0,0]
# m     = args[0,0,0,0,1]
# E_pnm = args[:,0,0,0,2]
# L_pnm = args[0,:,0,0,3]
# K_pnm = args[0,0,:,0,4]
# Q_pnm = args[0,0,0,:,5]
# =============================================================================


args=args.transpose([len(args_shape)]+range(len(args_shape)))[2:]

logic_p1_30=[]
logic_p1_374=[]
logic_p1_85=[]
#logic_odd_eos=[]
for i in range(len(eos_flat)):
    logic_p1_30.append(eos_flat[i].eosBaryonDensity(30.)>1.85*n_s)
    logic_p1_374.append(eos_flat[i].eosBaryonDensity(3.74)<1.85*n_s)
    logic_p1_85.append(eos_flat[i].eosBaryonDensity(8.5)<1.85*n_s)
    #logic_odd_eos.append(eos_flat[i].eosBaryonDensity(0.27)<0.15 and eos_flat[i].args[4]==5)
tmp=np.copy(logic)
tmp[logic]=np.array(logic_p1_30)
logic_p1_30=tmp
tmp=np.copy(logic)
tmp[logic]=np.array(logic_p1_374)
logic_p1_374=tmp
tmp=np.copy(logic)
tmp[logic]=np.array(logic_p1_85)
logic_p1_85=tmp

logic_raidus_onepointfour=Properity_onepointfour[:,2]<13500.

chirp_mass=chirp_q_Lambdabeta6_Lambda1Lambda2[:,0]
q=chirp_q_Lambdabeta6_Lambda1Lambda2[:,1]
Lambda_binary_beta6=chirp_q_Lambdabeta6_Lambda1Lambda2[:,2]
Lambda2Lambda1=chirp_q_Lambdabeta6_Lambda1Lambda2[:,3]
Lambda2Lambda1q6=Lambda2Lambda1*q**6

# =============================================================================
# plt.plot(mass_beta_Lambda_result[logic_p1_30,0]/mass_beta_Lambda_result[logic_p1_30,1]*(2*10**30*6.674*10**(-11)/9/10**16/1000),mass_beta_Lambda_result[logic_p1_30,0],'.')
# plt.xlabel('R(km)')
# plt.ylabel('M/M$_\odot$')
# =============================================================================

args_txt=['L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$','Z$_{pnm}$']
args_txt_unit=['MeV','MeV','MeV','MeV']
args_order=(0,1,2,3)
args_show=args[2:,1,1,::5,::4,:,:]
eos_logic_show=eos_logic[1,1,::5,::4,:,:]
logic_show=logic[1,1,::5,::4,:,:]
# =============================================================================
# args_txt=['E$_{pnm}$','L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$']
# args_txt_unit=['MeV','MeV','MeV','MeV']
# # =============================================================================
# # args_order=(0,1,2,3)
# # args_show=args[:,:,::5,:,:]
# # eos_logic_show=eos_logic[:,::5,:,:]
# # logic_show=logic[:,::5,:,:]
# # =============================================================================
# args_order=(0,2,1,3)
# args_show=args[:,:,:,::6,:]
# eos_logic_show=eos_logic[:,:,::6,:]
# logic_show=logic[:,:,::6,:]
# =============================================================================
from plot_logic import plot_5D_logic
plot_5D_logic(eos_logic_show,args_show,args_txt,args_order,figsize=(16,15))
plot_5D_logic(logic_show,args_show,args_txt,args_order,figsize=(16,15))


# =============================================================================
# args_txt=['E$_{pnm}$','L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$']
# args_txt_unit=['MeV','MeV','MeV','MeV']
# args_order=(0,2,1,3)
# args_show=args[:,:,:,::10,:]
# Properity_onepointfour_show=Properity_onepointfour[5,:,:,::10,:]
# maxmass_mass_show=maxmass_result[1,:,:,::10,:]
# maxmass_cs2_show=maxmass_result[2,:,:,::10,:]
# maxmass_cs2_show[maxmass_cs2_show==1]=0
# from plot_contour import plot_5D_contour,reverse_colourmap
# from matplotlib import cm
# fig1,axis1=plot_5D_contour(Properity_onepointfour_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=20,manual_level=[[800,'k','dashed',4]],colorbar_label='$\Lambda_{1.4}$',cmap = cm.bwr)
# fig2,axis2=plot_5D_contour(maxmass_mass_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=12,Z_extra_level=[[800,'k','dashed',4]],array_Z_extra=Properity_onepointfour_show,colorbar_label='M$_{max}$/M$_\odot$',cmap = reverse_colourmap(cm.bwr))
# fig3,axis3=plot_5D_contour(maxmass_cs2_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=12,manual_level=[[0,'k','solid',4],[1,'k','solid',4]],Z_extra_level=[[800,'k','dashed',4]],array_Z_extra=Properity_onepointfour_show,colorbar_label='$(c_s^2)_{max}$',cmap = reverse_colourmap(cm.bwr))
# E_bind=16
# J_L_matching_bound=np.loadtxt('E_L_matching_bound.txt')
# from scipy.interpolate import interp1d
# J_L_matching_bound_upper=interp1d(J_L_matching_bound[:,0]+E_bind,J_L_matching_bound[:,1])
# J_L_matching_bound_lower_gamma_over_2=interp1d(J_L_matching_bound[:,0]+E_bind,J_L_matching_bound[:,2])
# J_L_matching_bound_lower_gamma_over_1=interp1d(J_L_matching_bound[:,0]+E_bind,J_L_matching_bound[:,3])
# for i in range(len(axis1[0])):
#     for j in range(len(axis1[0][i])):
#         for J_L_matching_bound in [J_L_matching_bound_upper,J_L_matching_bound_lower_gamma_over_2,J_L_matching_bound_lower_gamma_over_1]:
#             for axis_i in axis1+axis2+axis3:
#                 axis_i[i,j].plot(J_L_matching_bound(args_show[0,:,0,:,0][j,i]+E_bind)+0*Q_pnm,Q_pnm,lw=2)
#                 axis_i[i,j].set_xlim(20,120)
# fig1[0].savefig(path+dir_name+'/bulk_para_space_with_Lambda')
# fig2[0].savefig(path+dir_name+'/bulk_para_space_with_maxmass_mass')
# fig3[0].savefig(path+dir_name+'/bulk_para_space_with_maxmass_cs2')
# =============================================================================

args_txt=['m$_{eff}$','E$_{pnm}$','L$_{pnm}$','K$_{pnm}$']
args_txt_unit=['MeV','MeV','MeV','MeV']
args_order=(1,0,2,3)
args_show=args[:4,:,:,:,:,0,0]
Properity_onepointfour[5][np.logical_not(logic)]=10000
Properity_onepointfour_show_low=np.min(Properity_onepointfour[5],axis=(4,5))
Properity_onepointfour_show_low[Properity_onepointfour_show_low>9999]=np.min(Properity_onepointfour[5])
Properity_onepointfour_show_low[Properity_onepointfour_show_low>1300]=1300
Properity_onepointfour[5][np.logical_not(logic)]=np.min(Properity_onepointfour[5])
Properity_onepointfour_show_high=np.max(Properity_onepointfour[5],axis=(4,5))
Properity_onepointfour_show_high[Properity_onepointfour_show_high>1300]=1300
from plot_contour import plot_5D_contour
from matplotlib import cm
fig1,axis1=plot_5D_contour(Properity_onepointfour_show_low[:,1:],args_show[:,:,1:],args_txt,args_order,figsize=(16,15),N_desire=10,manual_level=[[800,'k','dashed',4]],colorbar_label='$\Lambda_{1.4}$',cmap = cm.bwr)
fig2,axis2=plot_5D_contour(Properity_onepointfour_show_high[:,1:],args_show[:,:,1:],args_txt,args_order,figsize=(16,15),N_desire=10,manual_level=[[800,'k','dashed',4]],colorbar_label='$\Lambda_{1.4}$',cmap = cm.bwr)
fig1[0].savefig(path+dir_name+'/bulk_para_space_with_Lambda_low')
fig2[0].savefig(path+dir_name+'/bulk_para_space_with_Lambda_high')
args_txt=['L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$','Z$_{pnm}$']
args_txt_unit=['MeV','MeV','MeV','']
args_order=(0,1,2,3)
args_show=args[2:,1,1,::5,::4,:,:]
maxmass_cs2_show=maxmass_result[2,1,1,::5,::4,:,:]
maxmass_cs2_show[maxmass_cs2_show==1]=0
maxmass_cs2_show[maxmass_cs2_show<0]=0
maxmass_mass_show=maxmass_result[1,1,1,::5,::4,:,:]
maxmass_mass_show[maxmass_mass_show<2]=2
maxmass_mass_show[maxmass_cs2_show==0]=2
logic_bound=logic[1,1,::5,::4,:,:].astype('float')
from plot_contour import plot_5D_contour,reverse_colourmap
from matplotlib import cm
fig1,axis1=plot_5D_contour(maxmass_cs2_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=14,manual_level=[[0,'k','solid',4],[1,'k','solid',4]],colorbar_label='$(c_s^2)_{max}$',cmap = reverse_colourmap(cm.bwr))
fig2,axis2=plot_5D_contour(maxmass_mass_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=8,manual_level=[[2,'k','solid',4]],colorbar_label='M$_{max}$/M$_\odot$',cmap = reverse_colourmap(cm.bwr))
fig1[0].savefig(path+dir_name+'/bulk_para_space_with_maxmass_cs2')
fig2[0].savefig(path+dir_name+'/bulk_para_space_with_maxmass_mass')



args_txt=['m$_{eff}$','E$_{pnm}$','L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$','Z$_{pnm}$']
args_txt_unit=['MeV','MeV','MeV','MeV','MeV','MeV']
from eos_class import EOS_BPS
SLY4_EoS=EOS_BPS()
import show_properity
args_index_center=[1,1,10,12,8,13] #set center EoS to have parameter args[:,1,4,8,6]
args_index_around=[7,2,1,1] #firt parameter has step 1, second parameter has step 2 ...
                            #which means second parameter should pick [0,2,4,6, ...]
# =============================================================================
# args_txt=['E$_{pnm}$','L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$',]
# args_txt_unit=['MeV','MeV','MeV','MeV']
# from eos_class import EOS_BPS
# SLY4_EoS=EOS_BPS()
# import show_properity
# args_index_center=[2,3,25,25] #set center EoS to have parameter args[:,1,4,8,6]
# args_index_around=[1,1,1,1] #firt parameter has step 1, second parameter has step 2 ...
#                             #which means second parameter should pick [0,2,4,6, ...]
# =============================================================================
for show_eos_index,show_eos_index_name,xlim,ylim in zip([1,5],['pressure','cs2'],[[0.16,1],[0.16,1]],[[0,850],[0,1]]):
    fig, axes = plt.subplots(2, 2,figsize=(12,10),sharex=True,sharey=True)
    title=np.core.defchararray.add(args_txt,np.array(['=']*len(args_txt)))
    title=np.core.defchararray.add(title,np.round(args.transpose(range(1,len(args_shape)+1)+[0])[tuple(args_index_center)],decimals=1).astype(str))
    title=np.core.defchararray.add(title,np.array([' ']*len(args_txt)))
    title=np.core.defchararray.add(title,args_txt_unit)
    title=', '.join(title)
    fig.suptitle(title,fontsize=15)
    fig.tight_layout(pad=3)
    i=2
    for axes_i,step_i in zip(axes.flatten(),args_index_around):
        logic_vector=show_properity.vector_logic(logic,args_index_center,i,step_n=step_i)
        legend_list=np.around(args[i][logic][logic_vector[logic]],decimals=2).astype(str)
        legend_list=np.core.defchararray.add(np.array([args_txt[i]+'=']*len(legend_list)),legend_list)
        legend_list=np.core.defchararray.add(legend_list,np.array([' '+args_txt_unit[i]]*len(legend_list)))
        show_properity.show_eos(axes_i,[SLY4_EoS],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=['SLY4'],lw=3,legend_fontsize=15)
        show_properity.show_eos(axes_i,eos_flat[logic_vector[logic]],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=legend_list,legend_fontsize=15)
        axes_i.set_xlim(*xlim)
        axes_i.set_ylim(*ylim)
        if(i%2==1):
            axes_i.set_ylabel('')
        if(i/2==0):
            axes_i.set_xlabel('')
        i+=1
    fig.savefig(dir_name+'/parameter_impact_on_eos_'+show_eos_index_name)
    fig.clear()

# =============================================================================
# args_txt=['E$_{pnm}$','L$_{pnm}$','K$_{pnm}$','Q$_{pnm}$',]
# args_txt_unit=['MeV','MeV','MeV','MeV']
# from eos_class import EOS_BPS
# SLY4_EoS=EOS_BPS()
# import show_properity
# args_index_center=[2,3,25,26] #set center EoS to have parameter args[:,1,4,8,6]
# args_index_around=[1,2,4,2] #firt parameter has step 1, second parameter has step 2 ...
#                             #which means second parameter should pick [0,2,4,6, ...]
# for show_eos_index,show_eos_index_name,xlim,ylim in zip([1,5],['pressure','cs2'],[[0.16,1],[0.16,1]],[[0,850],[0,1]]):
#     fig, axes = plt.subplots(2, 2,figsize=(12,10),sharex=True,sharey=True)
#     title=np.core.defchararray.add(args_txt,np.array(['=']*len(args_txt)))
#     title=np.core.defchararray.add(title,np.round(args.transpose(range(1,len(args_shape)+1)+[0])[tuple(args_index_center)],decimals=1).astype(str))
#     title=np.core.defchararray.add(title,np.array([' ']*len(args_txt)))
#     title=np.core.defchararray.add(title,args_txt_unit)
#     title=', '.join(title)
#     fig.suptitle(title,fontsize=15)
#     fig.tight_layout(pad=3)
#     i=0
#     for axes_i,step_i in zip(axes.flatten(),args_index_around):
#         logic_vector=show_properity.vector_logic(logic,args_index_center,i,step_n=step_i)
#         legend_list=np.around(args[i][logic][logic_vector[logic]],decimals=2).astype(str)
#         legend_list=np.core.defchararray.add(np.array([args_txt[i]+'=']*len(legend_list)),legend_list)
#         legend_list=np.core.defchararray.add(legend_list,np.array([' '+args_txt_unit[i]]*len(legend_list)))
#         show_properity.show_eos(axes_i,[SLY4_EoS],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=['SLY4'],lw=3,legend_fontsize=15)
#         show_properity.show_eos(axes_i,eos_flat[logic_vector[logic]],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=legend_list,legend_fontsize=15)
#         axes_i.set_xlim(*xlim)
#         axes_i.set_ylim(*ylim)
#         if(i%2==1):
#             axes_i.set_ylabel('')
#         if(i/2==0):
#             axes_i.set_xlabel('')
#         i+=1
#     fig.savefig(dir_name+'/parameter_impact_on_eos_'+show_eos_index_name)
#     fig.clear()
# from physicalconst import c,G,mass_sun
# mass_sun_Gc2=mass_sun*G/c**2/100000
# for show_eos_index,show_eos_index_name,xlim,ylim in zip([None],['Mass-Radius'],[[10,15]],[[0.5,3]]):
#     fig, axes = plt.subplots(2, 2,figsize=(12,10),sharex=True,sharey=True)
#     title=np.core.defchararray.add(args_txt,np.array(['=']*len(args_txt)))
#     title=np.core.defchararray.add(title,np.round(args.transpose(range(1,len(args_shape)+1)+[0])[tuple(args_index_center)],decimals=1).astype(str))
#     title=np.core.defchararray.add(title,np.array([' ']*len(args_txt)))
#     title=np.core.defchararray.add(title,args_txt_unit)
#     title=', '.join(title)
#     fig.suptitle(title,fontsize=15)
#     fig.tight_layout(pad=3)
#     i=0
#     for axes_i,step_i in zip(axes.flatten(),args_index_around):
#         logic_vector=show_properity.vector_logic(logic,args_index_center,i,step_n=step_i)
#         legend_list=np.around(args[i][logic][logic_vector[logic]],decimals=2).astype(str)
#         legend_list=np.core.defchararray.add(np.array([args_txt[i]+'=']*len(legend_list)),legend_list)
#         legend_list=np.core.defchararray.add(legend_list,np.array([' '+args_txt_unit[i]]*len(legend_list)))
#         logic_to_plot=logic_vector[logic]
#         show_properity.show_properity_eos(axes_i,mass_sun_Gc2*mass_beta_Lambda_result[logic_to_plot,0,:]/mass_beta_Lambda_result[logic_to_plot,1,:],mass_beta_Lambda_result[logic_to_plot,0,:],'R(km)','M/M$_\odot$',legend=legend_list,legend_fontsize=15)
#         axes_i.set_xlim(*xlim)
#         axes_i.set_ylim(*ylim)
#         if(i%2==1):
#             axes_i.set_ylabel('')
#         if(i/2==0):
#             axes_i.set_xlabel('')
#         i+=1
#     fig.savefig(dir_name+'/parameter_impact_on_eos_'+show_eos_index_name)
#     fig.clear()
# =============================================================================



from physicalconst import c,G,mass_sun
mass_sun_Gc2=mass_sun*G/c**2/100000
for show_eos_index,show_eos_index_name,xlim,ylim in zip([None],['Mass-Radius'],[[10,15]],[[0.5,3]]):
    fig, axes = plt.subplots(2, 2,figsize=(12,10),sharex=True,sharey=True)
    title=np.core.defchararray.add(args_txt,np.array(['=']*len(args_txt)))
    title=np.core.defchararray.add(title,np.round(args.transpose(range(1,len(args_shape)+1)+[0])[tuple(args_index_center)],decimals=1).astype(str))
    title=np.core.defchararray.add(title,np.array([' ']*len(args_txt)))
    title=np.core.defchararray.add(title,args_txt_unit)
    title=', '.join(title)
    fig.suptitle(title,fontsize=15)
    fig.tight_layout(pad=3)
    i=2
    for axes_i,step_i in zip(axes.flatten(),args_index_around):
        logic_vector=show_properity.vector_logic(logic,args_index_center,i,step_n=step_i)
        legend_list=np.around(args[i][logic][logic_vector[logic]],decimals=2).astype(str)
        legend_list=np.core.defchararray.add(np.array([args_txt[i]+'=']*len(legend_list)),legend_list)
        legend_list=np.core.defchararray.add(legend_list,np.array([' '+args_txt_unit[i]]*len(legend_list)))
        logic_to_plot=logic_vector[logic]
        show_properity.show_properity_eos(axes_i,mass_sun_Gc2*mass_beta_Lambda_result[logic_to_plot,0,:]/mass_beta_Lambda_result[logic_to_plot,1,:],mass_beta_Lambda_result[logic_to_plot,0,:],'R(km)','M/M$_\odot$',legend=legend_list,legend_fontsize=15)
        axes_i.set_xlim(*xlim)
        axes_i.set_ylim(*ylim)
        if(i%2==1):
            axes_i.set_ylabel('')
        if(i/2==0):
            axes_i.set_xlabel('')
        i+=1
    fig.savefig(dir_name+'/parameter_impact_on_eos_'+show_eos_index_name)
    fig.clear()

import show_properity
from eos_class import EOS_BPS
for show_eos_index,show_eos_index_name,xlim,ylim in zip([1,5],['pressure','cs2'],[[0.16,1],[0.16,1]],[[0,2000],[0,1]]):
    fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
    show_properity.show_eos(axes,eos_flat[::10],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'])
    show_properity.show_eos(axes,[EOS_BPS()],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=['SLY4'],lw=3,legend_fontsize=15)
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    fig.savefig(dir_name+'/eos_n_'+show_eos_index_name)
    fig.clear()


p1_limit_array_list=[]
p1_limit_txt=['','_p1_less_30','_p1_over_374','_p1_less_30_over_374','_p1_over_85','_p1_less_30_over_85']
p1_limit_array_list.append(logic[logic])
p1_limit_array_list.append(logic_p1_30[logic])
p1_limit_array_list.append(logic_p1_374[logic])
p1_limit_array_list.append(np.logical_and(logic_p1_374,logic_p1_30)[logic])
p1_limit_array_list.append(logic_p1_85[logic])
p1_limit_array_list.append(np.logical_and(logic_p1_85,logic_p1_30)[logic])
line_color=['b','c','r','g','y','k','orange','purple','gold','olive','m','navy','lime','grey','peru','violet']
for additional_constrain,file_txt in zip(p1_limit_array_list,p1_limit_txt):
    fig = plt.figure(figsize=(8,6))  
    for j in range(int((maxmass_result[:,logic][1,additional_constrain].max()-2.0)*10)+1):
        logic_in_maxmass_range=np.logical_and(additional_constrain,np.logical_and(maxmass_result[1,logic]>2.0+0.1*j,maxmass_result[1,logic]<2.0+0.1*(j+1)))
        plt.plot(np.concatenate(mass_beta_Lambda_result[logic_in_maxmass_range,0]),np.concatenate(mass_beta_Lambda_result[logic_in_maxmass_range,2])*np.concatenate(mass_beta_Lambda_result[logic_in_maxmass_range,1])**6,'.',color=line_color[j],markersize=1)
        plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='%.1f<M$_{max}$/M$_\odot$<%.1f'%(2+0.1*j,2+0.1*(j+1)))
    plt.xlabel('$M/M_\odot$',fontsize=20)
    plt.ylabel('$\Lambda (GM/Rc^2)^6$',fontsize=20)
    plt.xlim(1.0,1.8)
    plt.ylim(0.006,0.012)
    plt.legend(loc=3,frameon=False)
    fig.savefig(dir_name+'/Lambdabeta6_mass'+file_txt)
    fig.clear()


line_color=['b','c','r','g','y','k','orange','purple','gold','olive','m','navy','lime','grey','peru','violet']
for additional_constrain,file_txt in zip(p1_limit_array_list,p1_limit_txt):
    fig = plt.figure(figsize=(8,6))
    for j in range(int((maxmass_result[:,logic][1,additional_constrain].max()-2.0)*10)+1):
        logic_in_maxmass_range=np.logical_and(additional_constrain,np.logical_and(maxmass_result[1,logic]>2.0+0.1*j,maxmass_result[1,logic]<2.0+0.1*(j+1)))
        plt.plot(np.concatenate(chirp_mass[logic_in_maxmass_range]),np.concatenate(Lambda_binary_beta6[logic_in_maxmass_range]),'.',color=line_color[j],markersize=1)
        plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='%.1f<M$_{max}$/M$_\odot$<%.1f'%(2+0.1*j,2+0.1*(j+1)))
    plt.xlabel('$M_{ch}/M_\odot$',fontsize=20)
    plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$',fontsize=20)
    plt.xlim(np.concatenate(chirp_mass).min(),np.concatenate(chirp_mass).max())
    plt.ylim(np.concatenate(Lambda_binary_beta6).min()*0.8,np.concatenate(Lambda_binary_beta6).max())
    plt.xlim(1.0,1.8)
    plt.ylim(0.00,0.006)
    plt.legend(loc=3,frameon=False)
    fig.savefig(dir_name+'/Binary_Lambdabeta6_chirpmass'+file_txt)
    fig.clear()



import matplotlib as mpl   
from show_properity import show_properity
from physicalconst import c,G,mass_sun
for logic_p1_low_limit,file_txt,p1_over in zip([logic_p1_374,logic_p1_85],['_p1_over_374','_p1_over_85'],[3.74,8.5]):
    fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
    mass_sun_Gc2=mass_sun*G/c**2/100000
    #p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)[logic]
    p1_limit_array=logic[logic]
    maxmass_array=(maxmass_result[1,logic][p1_limit_array])
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
    maxmass_array_sort.sort()
    axes[0].set_title('p$_1$ no constrain')
    axes[0].set_ylim(1.0,3.0)
    show_properity(axes[0],mass_sun_Gc2*mass_beta_Lambda_result[p1_limit_array,0,:]/mass_beta_Lambda_result[p1_limit_array,1,:],mass_beta_Lambda_result[p1_limit_array,0,:],'R(km)','M/M$_\odot$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
    #p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)[logic]
    p1_limit_array=np.logical_and(logic_p1_low_limit,logic_p1_30)[logic]
    maxmass_array=(maxmass_result[1,logic][p1_limit_array])
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
    maxmass_array_sort.sort()
    axes[1].set_title('%.2f MeV fm$^{-3}$ <p$_1$ < 30 MeV fm$^{-3}$'%(p1_over))
    show_properity(axes[1],mass_sun_Gc2*mass_beta_Lambda_result[p1_limit_array,0,:]/mass_beta_Lambda_result[p1_limit_array,1,:],mass_beta_Lambda_result[p1_limit_array,0,:],'R(km)','M/M$_\odot$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
    ax_colorbar=fig.add_axes([0.9, 0.2, 0.03, 0.6])
    cmap = mpl.colors.ListedColormap(line_color[:(len(maxmass_array_sort)-1)])
    norm = mpl.colors.BoundaryNorm(maxmass_array_sort, cmap.N)
    cb3 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap,
                                norm=norm,
    # =============================================================================
    #                                 boundaries=maxmass_array_sort,
    #                                 extend='both',
    #                                 # Make the length of each extension
    #                                 # the same as the length of the
    #                                 # interior colors:
    #                                 extendfrac='auto',
    #                                 ticks=maxmass_array_sort,
    #                                 spacing='uniform',
    # =============================================================================
                                orientation='vertical')
    cb3.set_label('M/M$_\odot$')
    fig.savefig(path+dir_name+'/Radius_Mass'+file_txt)
    fig.clear()

def Low_tidal_cutoff_MC(mass,maxmass):
    mass_over_maxmass=mass/maxmass
    return np.exp(13.42-23.04*mass_over_maxmass+20.56*mass_over_maxmass**2-9.615*mass_over_maxmass**3)
def Low_tidal_cutoff_UG(mass):
    mass_over_maxmass=mass
    return np.exp(18.819-19.862*mass_over_maxmass+10.881*mass_over_maxmass**2-2.5713*mass_over_maxmass**3)
m_plot=np.linspace(1,2,101)
tidal_lower_bound_MC=Low_tidal_cutoff_MC(m_plot,2.0)
tidal_lower_bound_UG=Low_tidal_cutoff_UG(m_plot)

for logic_p1_low_limit,file_txt,p1_over in zip([logic_p1_374,logic_p1_85],['_p1_over_374','_p1_over_85'],[3.74,8.5]):
    fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
    #p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)[logic]
    p1_limit_array=logic[logic]
    maxmass_array=(maxmass_result[1,logic])[p1_limit_array]
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))#[::-1]
    axes[0].set_title('p$_1$ no constrain')
    show_properity(axes[0],mass_beta_Lambda_result[p1_limit_array,2,:],mass_beta_Lambda_result[p1_limit_array,0,:],'$\Lambda$','M/M$_\odot$',x_scale='log',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
    axes[0].plot(tidal_lower_bound_MC,m_plot,lw=5,label='MC Bound')
    axes[0].plot(tidal_lower_bound_UG,m_plot,lw=5,label='UG Bound')
    axes[0].legend(frameon=False)
    plt.xlim(3,10000)
    plt.ylim(1,3)
    #p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)[logic]
    p1_limit_array=np.logical_and(logic_p1_low_limit,logic_p1_30)[logic]
    maxmass_array=(maxmass_result[1,logic])[p1_limit_array]
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))#[::-1]
    axes[1].set_title('%.2f MeV fm$^{-3}$ <p$_1$ < 30 MeV fm$^{-3}$'%(p1_over))
    show_properity(axes[1],mass_beta_Lambda_result[p1_limit_array,2,:],mass_beta_Lambda_result[p1_limit_array,0,:],'$\Lambda$','M/M$_\odot$',x_scale='log',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
    axes[1].plot(tidal_lower_bound_MC,m_plot,lw=5,label='MC Bound')
    axes[1].plot(tidal_lower_bound_UG,m_plot,lw=5,label='UG Bound')
    axes[1].legend(frameon=False)
    plt.xlim(3,10000)
    plt.ylim(1,3)
    fig.savefig(path+dir_name+'/Lambda_Mass'+file_txt)
    fig.clear()




def Lambda1Lambda2_fitting_2018(q,chirp_mass):
    from scipy.interpolate import interp1d
    chirp_mass_grid=np.linspace(1,1.4,9)
    n_=interp1d(chirp_mass_grid,[5.1717,5.2720,5.3786,5.4924,5.6138,5.7449,5.8960,6.0785,6.3047],kind='quadratic')
    n_0=interp1d(chirp_mass_grid,[6.4658,6.7470,7.0984,7.5546,8.1702,8.9715,9.9713,11.234,12.833],kind='quadratic')
    n_1=interp1d(chirp_mass_grid,[-0.2489,-0.32672,-0.44315,-0.62431,-0.91294,-1.3177,-1.8091,-2.3970,-3.0232],kind='quadratic')
    return q**(n_(chirp_mass)),q**(n_0(chirp_mass)+q*n_1(chirp_mass))

p1_limit_array_list=[]
p1_limit_txt=['','_p1_less_30','_p1_over_374','_p1_less_30_over_374','_p1_over_85','_p1_less_30_over_85']
p1_limit_array_list.append(logic[logic])
p1_limit_array_list.append(logic_p1_30[logic])
p1_limit_array_list.append(logic_p1_374[logic])
p1_limit_array_list.append(np.logical_and(logic_p1_374,logic_p1_30)[logic])
p1_limit_array_list.append(logic_p1_85[logic])
p1_limit_array_list.append(np.logical_and(logic_p1_85,logic_p1_30)[logic])
color_list=line_color=['b','c','r','g','y','k','orange','purple','gold','olive','m','navy','lime','grey','peru','violet']*2
def show_properity(ax,properity_array_x,properity_array_y,x_label,y_label,chirpmass_range,x_scale='linear',y_scale='linear',properity_array_z=1,properity_array_z_sort=[0,2],z_label=1,color_list=color_list):
    for i in range(len(properity_array_z_sort)-1):
        logic_sort=np.logical_and(properity_array_z>=np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),properity_array_z<np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]]))
        logic_in_chirpmass_range=np.logical_and(np.concatenate(chirpmass_range[0][logic_sort])>chirpmass_range[1],np.concatenate(chirpmass_range[0][logic_sort])<chirpmass_range[2])
        ax.plot(np.concatenate(properity_array_x[logic_sort])[logic_in_chirpmass_range],np.concatenate(properity_array_y[logic_sort])[logic_in_chirpmass_range],'.',color=color_list[i])#,label='%.1f<%s<%.1f'%(np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),z_label,np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]])))
import matplotlib as mpl
for p1_limit_array,file_txt in zip(p1_limit_array_list,p1_limit_txt):
    fig, axes = plt.subplots(4, 2,figsize=(10,16),sharex=True,sharey=True)
    chirpmass_min=1.0
    q_fit=np.linspace(0.7,1.0,31)
    maxmass_array=(maxmass_result[1,logic])[p1_limit_array]
    #maxmass_array_sort=np.linspace(2.0,int(maxmass_array.max()*10+1)/10.,4)
    maxmass_array_sort=np.linspace(2.0,int(maxmass_array.max()*10+1)/10.,int((maxmass_array.max()-2)*10+2))
    for ix in range(4):
        #ax_colorbar=fig.add_axes([0.485, 0.70, 0.01, 0.2])
        ax_colorbar=fig.add_axes([0.9, 0.2, 0.03, 0.6])
        cmap = mpl.colors.ListedColormap(line_color[:(len(maxmass_array_sort)-1)])
        norm = mpl.colors.BoundaryNorm(maxmass_array_sort, cmap.N)
        cb3 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap,
                                    norm=norm,
    # =============================================================================
    #                                 boundaries=maxmass_array_sort,
    #                                 extend='both',
    #                                 # Make the length of each extension
    #                                 # the same as the length of the
    #                                 # interior colors:
    #                                 extendfrac='auto',
    #                                 ticks=maxmass_array_sort,
    #                                 spacing='uniform',
    # =============================================================================
                                    orientation='vertical')
        cb3.set_label('M/M$_\odot$')
        for iy in range(2):
            j=2*ix+iy
            show_properity(axes[ix,iy],q[p1_limit_array],Lambda2Lambda1q6[p1_limit_array],'q','$\Lambda_2/\Lambda1 q^6$',[chirp_mass[p1_limit_array],int(chirpmass_min*20+j)/20.,int(chirpmass_min*20+j+1)/20.],properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
            #logic_in_chirpmass_range=np.logical_and(np.concatenate(chirp_mass)>int(chirpmass_min*20+j)/20.,np.concatenate(chirp_mass)<int(chirpmass_min*20+j+1)/20.)
            #axes[ix,iy].plot(np.concatenate(q)[logic_in_chirpmass_range],np.concatenate(Lambda2Lambda1q6)[logic_in_chirpmass_range],'.',color=line_color[j],markersize=1)
            axes[ix,iy].plot([0],[0],'.',color='k',markersize=5,label='Margueron PNM colored by maximum mass')
            fit_lower,tmp=Lambda1Lambda2_fitting_2018(q_fit,int(chirpmass_min*20+j)/20)
            tmp,fit_upper=Lambda1Lambda2_fitting_2018(q_fit,int(chirpmass_min*20+j+1)/20.)
            axes[ix,iy].plot(q_fit,q_fit**6/fit_lower,'-',lw=2,color='k',label='Piecewise Polytropic bounds')
            axes[ix,iy].plot(q_fit,q_fit**6/fit_upper,'-',lw=2,color='k')
            axes[ix,iy].set_xlim(0.7,1)
            #axes[ix,iy].set_ylim(0.5,7)
            axes[ix,iy].set_title('%.2f<M$_{ch}$/M$_\odot$<%.2f'%(int(chirpmass_min*20+j)/20.,int(chirpmass_min*20+j+1)/20.))
            axes[ix,iy].legend(loc=2,frameon=False,fontsize=9)
            axes[ix,iy].set_ylim(0.7,1.6)
            if(iy==0):
                axes[ix,iy].set_ylabel('$\Lambda_2/\Lambda_1 q^6$',fontsize=20)
            if(ix==3):
                axes[ix,iy].set_xlabel('q',fontsize=20)
    fig.savefig(path+dir_name+'/Lambda1Lambda2beta6_Mass'+file_txt)
    fig.clear()
    
    
# =============================================================================
# n_L,n_K=[5,6]
# for i in range(N_m):
#     m_i=i
#     fig, axes = plt.subplots(n_L, n_K,figsize=(20,12),sharex=True,sharey=True)
#     fig.suptitle('m$_{eff}$/m=%.2f'%(m_eff[m_i]/m), fontsize=16)
#     for j in range(n_L):
#         L_i=2*j
#         for k in range(n_K):
#             K_i=2*k
#             axes[j,k].imshow(np.reshape(logic,(N_m,N_L,N_K,N_Q,N_Z))[i,L_i,K_i,:,:].transpose(),aspect='auto',origin='lower',extent=(Q_pnm.min(),Q_pnm.max(),Z_pnm.min(),Z_pnm.max()))
#             axes[j,k].set_title('L=%d MeV, K=%d MeV'%(L_pnm[L_i],K_pnm[K_i]))
#             if(k==0):
#                 axes[j,k].set_ylabel('$Z_n$ MeV')
#             if(j==n_L-1):
#                 axes[j,k].set_xlabel('$Q_n$ MeV')
#     fig.savefig(path+dir_name+'/parameter_space_LKQZ_meff=%.2f.png'%(m_eff[m_i]/m))
# 
# 
# fig = plt.figure(figsize=(8,6))  
# line_color=['b','c','r','g','y','k','orange','purple','gold','pink','lime','violet']
# for j in range(int(((maxmass_result[logic])[logic_p1_30,1].max()-2.0)*10)+1):
#     logic_in_maxmass_range=np.logical_and(maxmass_result[:,1]>2.0+0.1*j,maxmass_result[:,1]<2.0+0.1*(j+1))
#     logic_in_maxmass_range=np.logical_and(logic_in_maxmass_range[logic],logic_p1_30)
#     #logic_in_maxmass_range=np.logical_and(logic_in_maxmass_range[logic],logic_raidus_onepointfour)
#     if(line_color[j]=='purple'):
#         check_this_eos=eos_flat[logic_in_maxmass_range]
#     plt.plot(np.concatenate((mass_beta_Lambda_result)[logic_in_maxmass_range,0]),np.concatenate((mass_beta_Lambda_result)[logic_in_maxmass_range,2])*np.concatenate((mass_beta_Lambda_result)[logic_in_maxmass_range,1])**6,'.',color=line_color[j],markersize=1)
#     plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='2.%d<M$_{max}$/M$_\odot$<2.%d'%(j,j+1))
# plt.xlabel('$M/M_\odot$',fontsize=20)
# plt.ylabel('$\Lambda (GM/Rc^2)^6$',fontsize=20)
# plt.xlim(1.0,1.8)
# plt.ylim(0.006,0.012)
# plt.legend(loc=3,frameon=False)
# 
# fig = plt.figure(figsize=(8,6))  
# line_color=['b','c','r','g','y','k','orange','purple','gold','pink','lime','violet']
# for j in range(int(((maxmass_result[logic])[logic_p1_30,1].max()-2.0)*10)+1):
#     logic_in_maxmass_range=np.logical_and(maxmass_result[:,1]>2.0+0.1*j,maxmass_result[:,1]<2.0+0.1*(j+1))
#     logic_in_maxmass_range=np.logical_and(logic_in_maxmass_range[logic],logic_p1_30)
#     plt.plot(np.concatenate(chirp_mass[logic_in_maxmass_range]),np.concatenate(Lambda_binary_beta6[logic_in_maxmass_range]),'.',color=line_color[j],markersize=1)
#     plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='2.%d<M$_{max}$/M$_\odot$<2.%d'%(j,j+1))
# plt.xlabel('$M_{ch}/M_\odot$',fontsize=20)
# plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$',fontsize=20)
# plt.xlim(np.concatenate(chirp_mass).min(),np.concatenate(chirp_mass).max())
# plt.ylim(np.concatenate(Lambda_binary_beta6).min()*0.8,np.concatenate(Lambda_binary_beta6).max())
# plt.xlim(1.0,1.4)
# plt.ylim(0.00,0.006)
# plt.legend(loc=3,frameon=False)
# 
# 
# 
# from show_properity import show_properity
# maxmass_array=(maxmass_result[logic])[:,1]
# maxmass_array_sort=np.linspace(2.0,int(maxmass_array.max()*10+1)/10.,int((maxmass_array.max()-2)*10+2))
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# show_properity(axes,chirp_mass,Lambda_binary_beta6,'M$_{ch}$/M$_\odot$','$\\bar \Lambda (M_{ch}/R_{1.4})^6$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
# 
# from physicalconst import c,G,mass_sun
# fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
# mass_sun_Gc2=mass_sun*G/c**2/100000
# p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)
# maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
# maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
# axes[0].set_title('3.74 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
# show_properity(axes[0],mass_sun_Gc2*mass_beta_Lambda_result[p1_limit_array,0,:]/mass_beta_Lambda_result[p1_limit_array,1,:],mass_beta_Lambda_result[p1_limit_array,0,:],'R(km)','M/M$_\odot$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
# axes[0].set_ylim(1.0,3.0)
# p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)
# maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
# maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
# axes[1].set_title('8.5 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
# show_properity(axes[1],mass_sun_Gc2*mass_beta_Lambda_result[p1_limit_array,0,:]/mass_beta_Lambda_result[p1_limit_array,1,:],mass_beta_Lambda_result[p1_limit_array,0,:],'R(km)','M/M$_\odot$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
# fig.savefig(path+dir_name+'/Radius_Mass')
# 
# def Low_tidal_cutoff_MC(mass,maxmass):
#     mass_over_maxmass=mass/maxmass
#     return np.exp(13.42-23.04*mass_over_maxmass+20.56*mass_over_maxmass**2-9.615*mass_over_maxmass**3)
# def Low_tidal_cutoff_UG(mass):
#     mass_over_maxmass=mass
#     return np.exp(18.819-19.862*mass_over_maxmass+10.881*mass_over_maxmass**2-2.5713*mass_over_maxmass**3)
# m_plot=np.linspace(1,2,101)
# tidal_lower_bound_MC=Low_tidal_cutoff_MC(m_plot,2.0)
# tidal_lower_bound_UG=Low_tidal_cutoff_UG(m_plot)
# 
# fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
# p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)
# maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
# maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
# axes[0].set_title('3.74 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
# show_properity(axes[0],mass_beta_Lambda_result[p1_limit_array,2,:],mass_beta_Lambda_result[p1_limit_array,0,:],'$\Lambda$','M/M$_\odot$',x_scale='log',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
# axes[0].plot(tidal_lower_bound_MC,m_plot,lw=5,label='Maxmimum Compact Bound')
# axes[0].plot(tidal_lower_bound_UG,m_plot,lw=5,label='Unitary Gas Bound')
# plt.xlim(1,30000)
# plt.ylim(1,3)
# p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)
# maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
# maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
# axes[1].set_title('8.5 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
# show_properity(axes[1],mass_beta_Lambda_result[p1_limit_array,2,:],mass_beta_Lambda_result[p1_limit_array,0,:],'$\Lambda$','M/M$_\odot$',x_scale='log',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
# axes[1].plot(tidal_lower_bound_MC,m_plot,lw=5,label='Maxmimum Compact Bound')
# axes[1].plot(tidal_lower_bound_UG,m_plot,lw=5,label='Unitary Gas Bound')
# plt.xlim(1,30000)
# plt.ylim(1,3)
# fig.savefig(path+dir_name+'/Lambda_Mass')
# 
# from show_properity import show_eos
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# show_eos(axes,eos_flat,0,1,500,pressure_range=[0.01,30,'log'])
# axes.set_xlim(0,1.85*0.16)
# axes.set_yscale('log')
# plt.plot(0.16*np.linspace(0.5,2,100),939+12.6*(np.linspace(0.5,2,100))**(2./3))
# fig.savefig(path+dir_name+'/eos_show')
# 
# 
# def Lambda1Lambda2_fitting_2018(q,chirp_mass):
#     from scipy.interpolate import interp1d
#     chirp_mass_grid=np.linspace(1,1.4,9)
#     n_=interp1d(chirp_mass_grid,[5.1717,5.2720,5.3786,5.4924,5.6138,5.7449,5.8960,6.0785,6.3047],kind='quadratic')
#     n_0=interp1d(chirp_mass_grid,[6.4658,6.7470,7.0984,7.5546,8.1702,8.9715,9.9713,11.234,12.833],kind='quadratic')
#     n_1=interp1d(chirp_mass_grid,[-0.2489,-0.32672,-0.44315,-0.62431,-0.91294,-1.3177,-1.8091,-2.3970,-3.0232],kind='quadratic')
#     return q**(n_(chirp_mass)),q**(n_0(chirp_mass)+q*n_1(chirp_mass))
#     
# chirpmass_min=1.0
# q_fit=np.linspace(0.7,1.0,31)
# plt.plot([0],[0],'--k',lw=2,label='bounds for piecewise polytropic EoS')
# for j in (8-np.array(range(9))):
#     logic_in_chirpmass_range=np.logical_and(np.concatenate(chirp_mass)>int(chirpmass_min*20+j)/20.,np.concatenate(chirp_mass)<int(chirpmass_min*20+j+1)/20.)
#     plt.plot(np.concatenate(q)[logic_in_chirpmass_range],np.concatenate(Lambda2Lambda1q6)[logic_in_chirpmass_range],'.',color=line_color[j],markersize=1)
#     plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='%.2f<M$_{ch}$/M$_\odot$<%.2f'%(int(chirpmass_min*20+j)/20.,int(chirpmass_min*20+j+1)/20.))
#     fit_lower,fit_upper=Lambda1Lambda2_fitting_2018(q_fit,int(chirpmass_min*20+j)/20.)
#     plt.plot(q_fit,q_fit**6/fit_lower,'--',lw=2,color=line_color[j])
#     plt.plot(q_fit,q_fit**6/fit_upper,'--',lw=2,color=line_color[j])
# plt.xlabel('q',fontsize=20)
# plt.ylabel('$\Lambda_2/\Lambda1 q^6$',fontsize=20)
# plt.xlim(0.7,1)
# plt.ylim(0.5,1.5)
# plt.legend(loc=1,frameon=False,fontsize=8)
# 
# 
# color_list=['b','c','r','g','y','k','orange','purple','gold','pink','lime','violet']*2
# def show_properity(ax,properity_array_x,properity_array_y,x_label,y_label,chirpmass_range,x_scale='linear',y_scale='linear',properity_array_z=1,properity_array_z_sort=[0,2],z_label=1,color_list=color_list):
#     for i in range(len(properity_array_z_sort)-1):
#         logic_sort=np.logical_and(properity_array_z>=np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),properity_array_z<np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]]))
#         logic_in_chirpmass_range=np.logical_and(np.concatenate(chirpmass_range[0][logic_sort])>chirpmass_range[1],np.concatenate(chirpmass_range[0][logic_sort])<chirpmass_range[2])
#         ax.plot(np.concatenate(properity_array_x[logic_sort])[logic_in_chirpmass_range],np.concatenate(properity_array_y[logic_sort])[logic_in_chirpmass_range],'.',color=color_list[i])#,label='%.1f<%s<%.1f'%(np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),z_label,np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]])))
# 
# import matplotlib as mpl   
# fig, axes = plt.subplots(4, 2,figsize=(10,16),sharex=True,sharey=True)
# chirpmass_min=1.0
# q_fit=np.linspace(0.7,1.0,31)
# p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)
# maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
# #maxmass_array_sort=np.linspace(2.0,int(maxmass_array.max()*10+1)/10.,4)
# maxmass_array_sort=np.linspace(2.0,int(maxmass_array.max()*10+1)/10.,int((maxmass_array.max()-2)*10+2))
# for ix in range(4):
#     #ax_colorbar=fig.add_axes([0.485, 0.70, 0.01, 0.2])
#     ax_colorbar=fig.add_axes([0.9, 0.2, 0.03, 0.6])
#     cmap = mpl.colors.ListedColormap(line_color[:(len(maxmass_array_sort)-1)])
#     norm = mpl.colors.BoundaryNorm(maxmass_array_sort, cmap.N)
#     cb3 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap,
#                                 norm=norm,
# # =============================================================================
# #                                 boundaries=maxmass_array_sort,
# #                                 extend='both',
# #                                 # Make the length of each extension
# #                                 # the same as the length of the
# #                                 # interior colors:
# #                                 extendfrac='auto',
# #                                 ticks=maxmass_array_sort,
# #                                 spacing='uniform',
# # =============================================================================
#                                 orientation='vertical')
#     cb3.set_label('M/M$_\odot$')
#     for iy in range(2):
#         j=2*ix+iy
#         show_properity(axes[ix,iy],q[p1_limit_array],Lambda2Lambda1q6[p1_limit_array],'q','$\Lambda_2/\Lambda1 q^6$',[chirp_mass[p1_limit_array],int(chirpmass_min*20+j)/20.,int(chirpmass_min*20+j+1)/20.],properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
#         #logic_in_chirpmass_range=np.logical_and(np.concatenate(chirp_mass)>int(chirpmass_min*20+j)/20.,np.concatenate(chirp_mass)<int(chirpmass_min*20+j+1)/20.)
#         #axes[ix,iy].plot(np.concatenate(q)[logic_in_chirpmass_range],np.concatenate(Lambda2Lambda1q6)[logic_in_chirpmass_range],'.',color=line_color[j],markersize=1)
#         axes[ix,iy].plot([0],[0],'.',color='k',markersize=5,label='Margueron PNM colored by maximum mass')
#         fit_lower,tmp=Lambda1Lambda2_fitting_2018(q_fit,int(chirpmass_min*20+j)/20)
#         tmp,fit_upper=Lambda1Lambda2_fitting_2018(q_fit,int(chirpmass_min*20+j+1)/20.)
#         axes[ix,iy].plot(q_fit,q_fit**6/fit_lower,'-',lw=2,color='k',label='Piecewise Polytropic bounds')
#         axes[ix,iy].plot(q_fit,q_fit**6/fit_upper,'-',lw=2,color='k')
#         axes[ix,iy].set_xlim(0.7,1)
#         #axes[ix,iy].set_ylim(0.5,7)
#         axes[ix,iy].set_title('%.2f<M$_{ch}$/M$_\odot$<%.2f'%(int(chirpmass_min*20+j)/20.,int(chirpmass_min*20+j+1)/20.))
#         axes[ix,iy].legend(loc=2,frameon=False,fontsize=9)
#         axes[ix,iy].set_ylim(0.7,1.6)
#         if(iy==0):
#             axes[ix,iy].set_ylabel('$\Lambda_2/\Lambda1 q^6$',fontsize=20)
#         if(ix==3):
#             axes[ix,iy].set_xlabel('q',fontsize=20)
# fig.savefig(path+dir_name+'/Lambda1Lambda1beta6_Mass')
# =============================================================================
