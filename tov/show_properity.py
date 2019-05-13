#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:26:07 2019

@author: sotzee
"""

import numpy as np
#show M-R or M_Lambda
color_list=['b','c','r','g','y','k','orange','purple','gold','pink','lime','violet']*2
def show_properity(ax,properity_array_x,properity_array_y,x_label,y_label,logic=1,x_scale='linear',y_scale='linear',properity_array_z=1,properity_array_z_sort=[0,2],z_label=1,color_list=color_list):
    if(type(logic)==int):
        logic=np.full(len(properity_array_x),True,dtype=bool)
    for i in range(len(properity_array_z_sort)-1):
        logic_sort=np.logical_and(properity_array_z>=np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),properity_array_z<np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]]))
        logic_and=np.logical_and(logic,logic_sort)
        ax.plot(np.concatenate(properity_array_x[logic_and]),np.concatenate(properity_array_y[logic_and]),'.',color=color_list[i],label='%.1f<%s<%.1f'%(np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),z_label,np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]])))
    ax.legend(frameon=False,loc=0)
    ax.set_xlabel(x_label)
    ax.set_xscale(x_scale)
    ax.set_ylabel(y_label)
    ax.set_yscale(y_scale)

def show_properity_eos(ax,properity_array_x,properity_array_y,x_label,y_label,x_scale='linear',y_scale='linear',xy_label_size=20,legend=[],legend_fontsize=15,legend_loc=0,marker=None,lw=1,color_list=color_list):
    if(len(legend)==len(properity_array_x)):
        label=legend
    else:
        label=np.full(len(properity_array_x),None)
    for i in range(len(properity_array_x)):
        ax.plot(properity_array_x[i],properity_array_y[i],marker=marker,lw=lw,label=label[i],color=color_list[i])
    if(len(legend)==len(properity_array_x)):
        ax.legend(frameon=False,fontsize=legend_fontsize,loc=legend_loc)
    ax.set_xlabel(x_label,fontsize=xy_label_size)
    ax.set_ylabel(y_label,fontsize=xy_label_size)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

def vector_logic(eos_logic,eos_parameter_index_all,true_vector_index,step_n=1):
    true_vector_logic=np.full(eos_logic.shape,False)
    eos_parameter_index_rest=eos_parameter_index_all[:true_vector_index]+eos_parameter_index_all[(true_vector_index+1):]
    true_vector_logic=true_vector_logic.transpose(range(true_vector_index)+range(true_vector_index+1,len(eos_parameter_index_all))+[true_vector_index])
    true_vector_logic[tuple(eos_parameter_index_rest)][range(eos_parameter_index_all[true_vector_index]%step_n,eos_logic.shape[true_vector_index],step_n)]=True
    true_vector_logic=true_vector_logic.transpose(range(true_vector_index)+[len(eos_parameter_index_all)-1]+range(true_vector_index,len(eos_parameter_index_all)-1))
    return np.logical_and(eos_logic,true_vector_logic)

def sample(low_high_log,N):
    if(low_high_log[2]=='log'):
        sample_result=np.logspace(np.log10(low_high_log[0]),np.log10(low_high_log[1]),N)
    else:
        sample_result=np.linspace(low_high_log[0],low_high_log[1],N)
    return sample_result

def show_eos(ax,eos,x_index,y_index,N,baryon_density_range=False,pressure_range=False,xy_label_size=20,legend=[],legend_fontsize=15,legend_loc=0,marker=None,lw=1):#index baryon_density(0), pressure(1), energy density(2), energy per baryon(3), chempo(4)
    pressure_density_energyPerBaryon_chempo=[]
    if(len(legend)==len(eos)):
        label=legend
    else:
        label=np.full(len(eos),None)
        #return 'legend number %d, eos number %d'%(len(legend),len(eos))
    if(baryon_density_range):
        baryon_density_i=sample(baryon_density_range,N)
        for eos_i,label_i in zip(eos,label):
            pressure_density_energyPerBaryon_chempo_i=[]
            pressure_density_energyPerBaryon_chempo_i.append(baryon_density_i)
            pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosPressure_frombaryon(baryon_density_i))
            pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_density_energyPerBaryon_chempo_i[1]))
            pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/baryon_density_i)
            pressure_density_energyPerBaryon_chempo_i.append((pressure_density_energyPerBaryon_chempo_i[1]+pressure_density_energyPerBaryon_chempo_i[2])/baryon_density_i)
            pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosCs2(pressure_density_energyPerBaryon_chempo_i[1]))
            pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
            ax.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index],  marker=marker,lw=lw,label=label_i)
    elif(pressure_range):
        pressure_i=sample(pressure_range,N)
        for eos_i,label_i in zip(eos,label):
            pressure_i=pressure_range[0]*np.linspace(1,pressure_range[1]/pressure_range[0],N)
            pressure_density_energyPerBaryon_chempo_i=[]
            pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosBaryonDensity(pressure_i))
            pressure_density_energyPerBaryon_chempo_i.append(pressure_i)
            pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_i))
            pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/pressure_density_energyPerBaryon_chempo_i[0])
            pressure_density_energyPerBaryon_chempo_i.append((pressure_i+pressure_density_energyPerBaryon_chempo_i[2])//pressure_density_energyPerBaryon_chempo_i[0])
            pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosCs2(pressure_density_energyPerBaryon_chempo_i[1]))
            pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
            ax.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index], marker=marker,lw=lw,label=label_i)
    if(len(legend)==len(eos)):
        ax.legend(frameon=False,fontsize=legend_fontsize,loc=legend_loc)
    pressure_density_energyPerBaryon_chempo=np.array(pressure_density_energyPerBaryon_chempo)
    label_text=['Baryon density(fm$^{-3}$)','Pressure(MeV fm$^{-3}$)','Energy density(MeV fm$^{-3}$)','Energy per baryon(MeV)','Chemical potential(MeV)','Sound speed square']
    ax.set_xlabel(label_text[x_index],fontsize=xy_label_size)
    ax.set_ylabel(label_text[y_index],fontsize=xy_label_size)
    #plt.xlim(pressure_density_energyPerBaryon_chempo[:,x_index,:].min(),pressure_density_energyPerBaryon_chempo[:,x_index,:].max())
    #plt.ylim(pressure_density_energyPerBaryon_chempo[:,y_index,:].min(),pressure_density_energyPerBaryon_chempo[:,y_index,:].max())

def check_eos_type(eos_array,eos_class):
    one_element=eos_array
    for i in range(len(np.shape(eos_array))):
        one_element=one_element[0]
    return type(one_element)==eos_class