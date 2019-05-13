#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:01:47 2019

@author: sotzee
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab,cm,colorbar,colors

def factors(n):    # (cf. https://stackoverflow.com/a/15703327/849891)
    j = 2
    while n > 1:
        for i in range(j, int(np.sqrt(n+0.05)) + 1):
            if n % i == 0:
                n /= i ; j = i
                yield i
                break
        else:
            if n > 1:
                yield n; break
from itertools import combinations

def fix_contour_level(range_Z,N_desire):
    times10=0.001
    while(int(times10*(range_Z[1]-range_Z[0]))<N_desire-2):
        times10*=10
    N=int(times10*(range_Z[1]-range_Z[0]))+1
    factors_N=list(factors(N))
    factors_N_all=[]+factors_N
    for i in range(2,len(factors_N)+1):
        factors_N_all+= list(np.product(np.array(list(combinations(factors_N, i))),axis=1))
    choose_N=factors_N_all[np.argmin(np.abs(np.array(factors_N_all)-N_desire))]
    print_digits=np.max([int(np.log10(times10)),0]).astype(int)
    return np.linspace(int(times10*(range_Z[0]))/times10,(int(times10*(range_Z[1]))+1)/times10,choose_N+1),'%.'+str(print_digits)+'f'

def reverse_colourmap(cmap, name = 'my_cmap_r'):      
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

def plot_5D_contour(array_Z,array,tex_args,plot_order,figsize=(10,6),N_desire=5,manual_level=[],Z_extra_level=[],array_Z_extra=[],colorbar_label=None,cmap = cm.jet):
    shape=array_Z.shape
    if(sorted(plot_order) != list(range(len(shape)))):
        print('plot_order=%s not suitable for logic array of size %s'%(plot_order,shape))
        return None
    array_Z_transposed=(np.transpose(array_Z,plot_order))
    if(len(array_Z_extra)==0):
        array_Z_extra_transposed=array_Z_transposed
    else:
        array_Z_extra_transposed=(np.transpose(array_Z_extra,plot_order))
    array_transposed=np.transpose(array,(0,)+tuple(np.array(plot_order)+1))
    shape=array_Z_transposed.shape
    fig_number=(len(shape)%2)*(shape[0]-1)+1
    if len(shape)%2==0:
        array_Z_transposed=array_Z_transposed[np.newaxis,:]
        array_Z_extra_transposed=array_Z_extra_transposed[np.newaxis,:]
    range_Z=[array_Z[array_Z>0].min(),array_Z[array_Z<10000].max()]
    levels,print_digits=fix_contour_level(range_Z,N_desire)
    norm = cm.colors.Normalize(vmin=levels[0], vmax=levels[-1])
    
    fig_all=[]
    axes_all=[]
    for fig_i in range(fig_number):
        shape_i=shape[(len(shape)%2):-2]
        if(np.size(shape_i)==2):
            fig, axes = plt.subplots(shape_i[1],shape_i[0],sharex=True,sharey=True,figsize=figsize)
            for i,j in [[i,j] for i in range(shape_i[0]) for j in range(shape_i[1])]:
                #args_range=(array[plot_order[-2],0,0,0,0],array[plot_order[-2],-1,-1,-1,-1],array[plot_order[-1],0,0,0,0],array[plot_order[-1],-1,-1,-1,-1])
                #axes[shape_i[1]-1-j,i].imshow(array_Z_transposed[fig_i,i,j].transpose(),aspect='auto',origin='lower',extent=args_range,cmap=cm.jet)
                axes[shape_i[1]-1-j,i].contourf(array_transposed[plot_order[-2],i,j], array_transposed[plot_order[-1],i,j], array_Z_transposed[fig_i,i,j],levels,cmap=cm.get_cmap(cmap,len(levels)-1),norm=norm)
                for Z_extra_level_i in Z_extra_level:
                    axes[shape_i[1]-1-j,i].contour(array_transposed[plot_order[-2],i,j], array_transposed[plot_order[-1],i,j], array_Z_extra_transposed[fig_i,i,j],[Z_extra_level_i[0]],colors=Z_extra_level_i[1],linestyles=Z_extra_level_i[2],linewidths=Z_extra_level_i[3])
                for manual_level_i in manual_level:
                    axes[shape_i[1]-1-j,i].contour(array_transposed[plot_order[-2],i,j], array_transposed[plot_order[-1],i,j], array_Z_transposed[fig_i,i,j],[manual_level_i[0]],colors=manual_level_i[1],linestyles=manual_level_i[2],linewidths=manual_level_i[3])
                if i==0:
                    label_index_tuple=[plot_order[-3],0,0,0,0]
                    label_index_tuple[plot_order[-3]+1]=j
                    label_index_tuple=tuple(label_index_tuple)
                    axes[shape_i[1]-1-j,i].set_ylabel(tex_args[plot_order[-3]]+'=%.2f'%array[label_index_tuple]+'\n'+tex_args[plot_order[-1]],fontsize=15)
                if j==0:
                    label_index_tuple=[plot_order[-4],0,0,0,0]
                    label_index_tuple[plot_order[-4]+1]=i
                    label_index_tuple=tuple(label_index_tuple)
                    axes[shape_i[1]-1-j,i].set_xlabel(tex_args[plot_order[-2]]+'\n'+tex_args[plot_order[-4]]+'=%.2f'%array[label_index_tuple],fontsize=15)
        ax_colorbar=fig.add_axes([0.9, 0.2, 0.03, 0.6])
        cb3 = colorbar.ColorbarBase(ax_colorbar, cmap=cm.get_cmap(cmap,len(levels)-1),norm=norm,orientation='vertical')
        cb3.set_label(colorbar_label,fontsize=20)
        fig_all.append(fig)
        axes_all.append(axes)
    return fig_all,axes_all

# =============================================================================
# fig,axes=plot_5D_contour((Properity_onepointfour[5,:,::3,:,::2]),args[:,:,::3,:,::2],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15),N_desire=12,manual_level=[[800,'k','--',3]],Z_extra_level=[[2.0,'k','-',3]],array_Z_extra=(maxmass_result[1,:,::3,:,::2]),colorbar_label='$\Lambda_{1.4}$',cmap=cm.bwr)
# fig,axes=plot_5D_contour((maxmass_result[1,:,::3,:,::2]),args[:,:,::3,:,::2],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15),N_desire=12,Z_extra_level=[[800,'k','--',3]],manual_level=[[2.0,'k','-',3]],array_Z_extra=(Properity_onepointfour[5,:,::3,:,::2]),colorbar_label='$M_{max}/M_\odot$',cmap=reverse_colourmap(cm.bwr))
# =============================================================================
