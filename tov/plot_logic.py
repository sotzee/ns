#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:00:21 2019

@author: sotzee
"""
import numpy as np
import matplotlib.pyplot as plt
def plot_5D_logic(logic,array,tex_args,plot_order,figsize=(10,6)):
    shape=logic.shape
    if(sorted(plot_order) != list(range(len(shape)))):
        print('plot_order=%s not suitable for logic array of size %s'%(plot_order,shape))
        return None
    logic_transposed=np.transpose(logic,plot_order)
    shape=logic_transposed.shape
    fig_number=(len(shape)%2)*(shape[0]-1)+1
    if len(shape)%2==0:
        logic_transposed=logic_transposed[np.newaxis,:]
    for fig_i in range(fig_number):
        shape_i=shape[(len(shape)%2):-2]
        if(np.size(shape_i)==2):
            fig, axes = plt.subplots(shape_i[1],shape_i[0],sharex=True,sharey=True,figsize=figsize)
            for i,j in [[i,j] for i in range(shape_i[0]) for j in range(shape_i[1])]:
                args_range=(array[plot_order[-2],0,0,0,0],array[plot_order[-2],-1,-1,-1,-1],array[plot_order[-1],0,0,0,0],array[plot_order[-1],-1,-1,-1,-1])
                axes[shape_i[1]-1-j,i].imshow(logic_transposed[fig_i,i,j].transpose(),aspect='auto',origin='lower',extent=args_range,cmap='Blues')
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
        elif(np.size(shape_i)==0):
            fig = plt.figure(figsize=figsize)
            args_range=(array[plot_order[-2],0,0],array[plot_order[-2],-1,-1],array[plot_order[-1],0,0],array[plot_order[-1],-1,-1])
            plt.imshow(logic_transposed[0].transpose(),aspect='auto',origin='lower',extent=args_range,cmap='Blues')
    return fig

# =============================================================================
# args=np.mgrid[0.5*939:0.8*939:5j,30:34:6j,20:120:7j,0:0.03:8j]
# logic=np.full(args.shape[1:],False,bool)
# logic[0,-1,0,0]=True
# logic[0,-2,0,0]=True
# plot_5D_logic(logic,args,['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))
# =============================================================================
