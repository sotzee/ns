#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:09:03 2019

@author: sotzee
"""

import matplotlib.pyplot as plt
import numpy as np
import cPickle
from Quarkyonic import EOS_Quarkyonic#,path,dir_name,mass_beta_Lambda_result,chirp_q_Lambdabeta6_Lambda1Lambda2

path = "./"
dir_name='Lambda_Quarkyonic_calculation_parallel'

def read_file(file_name):
    f_file=open(file_name,'rb')
    content=np.array(cPickle.load(f_file))
    f_file.close()
    return content

args=read_file(path+dir_name+'/Quarkyonic_args.dat')
args_shape=np.shape(args)[1:]
eos_logic=read_file(path+dir_name+'/Quarkyonic_eos_logic.dat')     #all eos store and calculated (stable and causal)
eos_success=read_file(path+dir_name+'/Quarkyonic_eos_success.dat') #matched successfully
eos_flat=read_file(path+dir_name+'/Quarkyonic_eos.dat')

maxmass_result=np.full(args_shape+(3,),np.array([0,0,1]),dtype='float')
maxmass_result[eos_logic]=read_file(path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat')
maxmass_result=maxmass_result.transpose((len(args_shape),)+tuple(range(len(args_shape))))
logic_maxmass=maxmass_result[1]>=2
logic_causality=maxmass_result[2]<0.99999
logic=np.logical_and(logic_maxmass,logic_causality)

Properity_onepointfour=np.full(args_shape+(6,),np.array([0,1.4,0,0,0,10000]),dtype='float')
Properity_onepointfour[eos_logic]=np.array([0,1.4,0,0,0,300])
Properity_onepointfour[logic]=read_file(path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat')
Properity_onepointfour=Properity_onepointfour.transpose((len(args_shape),)+tuple(range(len(args_shape))))
logic_tidal800=Properity_onepointfour[-1]<=800

mass_beta_Lambda_result=read_file(path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat')
chirp_q_Lambdabeta6_Lambda1Lambda2=read_file(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat')

N1,N2,N3,N4=np.shape(args)[:4]
n_s=0.16
m=939
E_pnm = args[0,:,0,0,0]
L_pnm = args[1,0,:,0,0]
Lambda_quarkyonic = args[2,0,0,:,0]
kappa_quarkyonic  = args[3,0,0,0,:]

eos_success_lambda_over_2=[]
matching_para=[]
pressure_s=[]
density_s=[]
logic_matching=[]
logic_p1_30=[]
#logic_p1_374=[]
#logic_p1_85=[]
#logic_odd_eos=[]
for i in range(len(eos_flat)):
    eos_success_lambda_over_2.append(eos_flat[i].matching_para[3]>2)
    matching_para.append(eos_flat[i].matching_para)
    density_s.append(eos_flat[i].density_s)
    pressure_s.append(eos_flat[i].pressure_s)
    logic_matching.append(eos_flat[i].matching_success)
    logic_p1_30.append(eos_flat[i].eosBaryonDensity(30.)>1.85*n_s)
    #logic_p1_374.append(eos_flat[i].eosBaryonDensity(3.74)<1.85*n_s)
    #logic_p1_85.append(eos_flat[i].eosBaryonDensity(8.5)<1.85*n_s)
    #logic_odd_eos.append(eos_flat[i].eosBaryonDensity(0.27)<0.15 and eos_flat[i].args[4]==5)
tmp=np.copy(eos_success)
tmp[eos_logic]=eos_success_lambda_over_2
eos_success_lambda_over_2=np.logical_and(tmp,eos_success)
tmp=np.full(args_shape,0.5)
tmp[eos_logic]=np.array(pressure_s)
pressure_s=tmp
tmp=np.full(args_shape,0.5)
tmp[eos_logic]=np.array(density_s)
density_s=tmp
tmp=np.full(args_shape,0.5)
tmp[eos_logic]=np.array(matching_para)[:,3]
gamma_matching=tmp
tmp=np.copy(eos_logic)
tmp[eos_logic]=logic_matching
logic_matching=tmp
tmp=np.copy(eos_logic)
tmp[eos_logic]=logic_p1_30
logic_p1_30=tmp
# =============================================================================
# tmp=np.copy(eos_logic)
# tmp[eos_logic]=logic_p1_374
# logic_p1_374=tmp
# tmp=np.copy(eos_logic)
# tmp[eos_logic]=logic_p1_85
# logic_p1_85=tmp
# =============================================================================

chirp_mass=chirp_q_Lambdabeta6_Lambda1Lambda2[:,0]
q=chirp_q_Lambdabeta6_Lambda1Lambda2[:,1]
Lambda_binary_beta6=chirp_q_Lambdabeta6_Lambda1Lambda2[:,2]
Lambda2Lambda1=chirp_q_Lambdabeta6_Lambda1Lambda2[:,3]
Lambda2Lambda1q6=Lambda2Lambda1*q**6

args_txt=['E','L','$\Lambda$','$\kappa$']
args_txt_unit=['MeV','MeV','MeV','']
args_order=(0,2,1,3)
args_show=args[:,:,:,::5,:]
eos_logic_show=eos_logic[:,:,::5,:]
eos_success_show=eos_success[:,:,::5,:]
logic_show=logic[:,:,::5,:]
from plot_logic import plot_5D_logic
plot_5D_logic(eos_logic_show,args_show,args_txt,args_order,figsize=(16,15))
plot_5D_logic(eos_success_show,args_show,args_txt,args_order,figsize=(16,15))
plot_5D_logic(logic_show,args_show,args_txt,args_order,figsize=(16,15))

Properity_onepointfour_show=Properity_onepointfour[5,:,:,::5,:]
maxmass_result_show=maxmass_result[1,:,:,::5,:]
from plot_contour import plot_5D_contour,reverse_colourmap
from matplotlib import cm
fig1,axis1=plot_5D_contour(Properity_onepointfour_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=20,manual_level=[[800,'k','dashed',4]],Z_extra_level=[[2,'k','solid',4]],array_Z_extra=maxmass_result_show,colorbar_label='$\Lambda_{1.4}$',cmap = cm.bwr)
fig2,axis2=plot_5D_contour(maxmass_result_show,args_show,args_txt,args_order,figsize=(16,15),N_desire=12,manual_level=[[2,'k','solid',4]],Z_extra_level=[[800,'k','dashed',4]],array_Z_extra=Properity_onepointfour_show,colorbar_label='M$_{max}$/M$_\odot$',cmap = reverse_colourmap(cm.bwr))
E_bind=16
J_L_matching_bound=np.loadtxt('E_L_matching_bound.txt')
from scipy.interpolate import interp1d
J_L_matching_bound_upper=interp1d(J_L_matching_bound[:,0]+E_bind,J_L_matching_bound[:,1])
J_L_matching_bound_lower_gamma_over_2=interp1d(J_L_matching_bound[:,0]+E_bind,J_L_matching_bound[:,2])
J_L_matching_bound_lower_gamma_over_1=interp1d(J_L_matching_bound[:,0]+E_bind,J_L_matching_bound[:,3])
for i in range(len(axis1[0])):
    for j in range(len(axis1[0][i])):
        for J_L_matching_bound in [J_L_matching_bound_upper,J_L_matching_bound_lower_gamma_over_2,J_L_matching_bound_lower_gamma_over_1]:
            for axis_i in axis1+axis2:
                axis_i[i,j].plot(J_L_matching_bound(args_show[0,:,0,:,0][j,i]-m+E_bind)+0*kappa_quarkyonic,kappa_quarkyonic,lw=2)
                axis_i[i,j].set_xlim(0,200)
fig1[0].savefig(path+dir_name+'/rmf_bulk_para_space_with_Lambda')
fig2[0].savefig(path+dir_name+'/rmf_bulk_para_space_with_maxmass')

hadronic_EoS=EOS_Quarkyonic([0.16,954.5,40,3000,0.3,(939,313,313)])
quarkonic_no_potential_EoS=EOS_Quarkyonic([0.16,973.3455, 67.268,380,0.3,(939,313,313)])
import show_properity
args_index_center=[1,4,8,6] #set center EoS to have parameter args[:,1,4,8,6]
args_index_around=[1,2,3,4] #firt parameter has step 1, second parameter has step 2 ...
                            #which means second parameter should pick [0,2,4,6, ...]
for show_eos_index,show_eos_index_name,xlim,ylim in zip([1,5],['pressure','cs2'],[[0.16,1],[0.16,1]],[[0,850],[0,1]]):
    fig, axes = plt.subplots(2, 2,figsize=(12,10),sharex=True,sharey=True)
    title=np.core.defchararray.add(args_txt,np.array(['=']*len(args_txt)))
    title=np.core.defchararray.add(title,np.round(args.transpose(range(1,len(args_shape)+1)+[0])[tuple(args_index_center)],decimals=1).astype(str))
    title=np.core.defchararray.add(title,np.array([' ']*len(args_txt)))
    title=np.core.defchararray.add(title,args_txt_unit)
    title=', '.join(title)
    fig.suptitle(title,fontsize=20)
    fig.tight_layout(pad=3)
    i=0
    for axes_i,step_i in zip(axes.flatten(),args_index_around):
        logic_vector=show_properity.vector_logic(logic,args_index_center,i,step_n=step_i)
        legend_list=np.around(args[i][eos_logic][logic_vector[eos_logic]],decimals=2).astype(str)
        legend_list=np.core.defchararray.add(np.array([args_txt[i]+'=']*len(legend_list)),legend_list)
        legend_list=np.core.defchararray.add(legend_list,np.array([' '+args_txt_unit[i]]*len(legend_list)))
        show_properity.show_eos(axes_i,[hadronic_EoS],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=['SLY4'],lw=3,legend_fontsize=15)
        show_properity.show_eos(axes_i,[quarkonic_no_potential_EoS],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=['Quarkyonic a=b=0'],lw=3,legend_fontsize=15)
        show_properity.show_eos(axes_i,eos_flat[logic_vector[eos_logic]],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=legend_list,legend_fontsize=15)
        axes_i.set_xlim(*xlim)
        axes_i.set_ylim(*ylim)
        if(i%2==1):
            axes_i.set_ylabel('')
        if(i/2==0):
            axes_i.set_xlabel('')
        i+=1
    fig.savefig(dir_name+'/parameter_impact_on_eos_'+show_eos_index_name)
    fig.clear()

from physicalconst import c,G,mass_sun
mass_sun_Gc2=mass_sun*G/c**2/100000
import show_properity
args_index_center=[1,4,8,6] #set center EoS to have parameter args[:,1,4,8,6]
args_index_around=[1,2,3,4] #firt parameter has step 1, second parameter has step 2 ...
                            #which means second parameter should pick [0,2,4,6, ...]
for show_eos_index,show_eos_index_name,xlim,ylim in zip([None],['Mass-Radius'],[[10,15]],[[0.5,3]]):
    fig, axes = plt.subplots(2, 2,figsize=(12,10),sharex=True,sharey=True)
    title=np.core.defchararray.add(args_txt,np.array(['=']*len(args_txt)))
    title=np.core.defchararray.add(title,np.round(args.transpose(range(1,len(args_shape)+1)+[0])[tuple(args_index_center)],decimals=1).astype(str))
    title=np.core.defchararray.add(title,np.array([' ']*len(args_txt)))
    title=np.core.defchararray.add(title,args_txt_unit)
    title=', '.join(title)
    fig.suptitle(title,fontsize=20)
    fig.tight_layout(pad=3)
    i=0
    for axes_i,step_i in zip(axes.flatten(),args_index_around):
        logic_vector=show_properity.vector_logic(logic,args_index_center,i,step_n=step_i)
        legend_list=np.around(args[i][eos_logic][logic_vector[eos_logic]],decimals=2).astype(str)
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

from eos_class import EOS_BPS
import show_properity
for show_eos_index,show_eos_index_name,xlim,ylim in zip([1,5],['pressure','cs2'],[[0.16,1],[0.16,1]],[[0,2000],[0,1]]):
    fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
    show_properity.show_eos(axes,eos_flat[np.logical_and(eos_success,logic)[eos_logic]][::10],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'])
    show_properity.show_eos(axes,[EOS_BPS()],0,show_eos_index,500,baryon_density_range=[n_s,1,'log'],legend=['SLY4'],lw=3,legend_fontsize=15)
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    fig.savefig(dir_name+'/eos_n_'+show_eos_index_name)
    fig.clear()

p1_limit_array_list=[]
p1_limit_txt=['','_p1_less_30','_lambda_over_2','_lambda_over_2_p1_less_30']
p1_limit_array_list.append(eos_success[logic])
p1_limit_array_list.append(np.logical_and(eos_success,logic_p1_30)[logic])
p1_limit_array_list.append(eos_success_lambda_over_2[logic])
p1_limit_array_list.append(np.logical_and(eos_success_lambda_over_2,logic_p1_30)[logic])
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
for eos_success_i,file_txt in zip([eos_success,eos_success_lambda_over_2],[p1_limit_txt[0],p1_limit_txt[2]]):
    fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
    mass_sun_Gc2=mass_sun*G/c**2/100000
    #p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)[logic]
    p1_limit_array=eos_success_i[logic]
    maxmass_array=(maxmass_result[1,logic][p1_limit_array])
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
    maxmass_array_sort.sort()
    axes[0].set_title('p$_1$ no constrain')
    axes[0].set_ylim(1.0,4.0)
    show_properity(axes[0],mass_sun_Gc2*mass_beta_Lambda_result[p1_limit_array,0,:]/mass_beta_Lambda_result[p1_limit_array,1,:],mass_beta_Lambda_result[p1_limit_array,0,:],'R(km)','M/M$_\odot$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
    #p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)[logic]
    p1_limit_array=np.logical_and(eos_success_i,logic_p1_30)[logic]
    maxmass_array=(maxmass_result[1,logic][p1_limit_array])
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
    maxmass_array_sort.sort()
    axes[1].set_title('p$_1$ < 30 MeV fm$^{-3}$')
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

for eos_success_i,file_txt in zip([eos_success,eos_success_lambda_over_2],[p1_limit_txt[0],p1_limit_txt[2]]):
    fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
    #p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)[logic]
    p1_limit_array=eos_success_i[logic]
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
    p1_limit_array=np.logical_and(eos_success_i,logic_p1_30)[logic]
    maxmass_array=(maxmass_result[1,logic])[p1_limit_array]
    maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))#[::-1]
    axes[1].set_title('p$_1$ < 30 MeV fm$^{-3}$')
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