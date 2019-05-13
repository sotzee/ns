#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:49:46 2018

@author: sotzee
"""

import matplotlib.pyplot as plt
import numpy as np
import cPickle
from PNM_expansion import EOS_SLY4_match_EXPANSION_PNM,EOS_CSS

path = "./"
dir_name='Lambda_PNM_around_vacuum_calculation_parallel'

def read_file(file_name):
    f_file=open(file_name,'rb')
    content=np.array(cPickle.load(f_file))
    f_file.close()
    return content

args=read_file(path+dir_name+'/Lambda_hadronic_calculation_args.dat')
eos_flat=read_file(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat')
eos_success=read_file(path+dir_name+'/Lambda_hadronic_calculation_eos_success.dat')
maxmass_result=np.full((len(eos_success),3),np.array([0,0,1]),dtype='float')
maxmass_result[eos_success]=read_file(path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat')
Properity_onepointfour=read_file(path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat')
mass_beta_Lambda_result=read_file(path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat')
chirp_q_Lambdabeta6_Lambda1Lambda2=read_file(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat')

N1,N2,N3,N4=np.shape(args)[:4]
n_s   = args[0,0,0,0,0]
m     = args[0,0,0,0,1]
E_pnm = args[:,0,0,0,2]
L_pnm = args[0,:,0,0,3]
K_pnm = args[0,0,:,0,4]
Q_pnm = args[0,0,0,:,5]

logic_maxmass=maxmass_result[:,1]>=2
logic_causality=maxmass_result[:,2]<0.99999
logic=np.logical_and(logic_maxmass,logic_causality)

logic_p1_30=[]
logic_p1_374=[]
logic_p1_85=[]
#logic_odd_eos=[]
for i in range(len(eos_flat)):
    logic_p1_30.append(eos_flat[i].eosBaryonDensity(30.)>1.85*n_s)
    logic_p1_374.append(eos_flat[i].eosBaryonDensity(3.74)<1.85*n_s)
    logic_p1_85.append(eos_flat[i].eosBaryonDensity(8.5)<1.85*n_s)
    #logic_odd_eos.append(eos_flat[i].eosBaryonDensity(0.27)<0.15 and eos_flat[i].args[4]==5)
logic_p1_30=np.array(logic_p1_30)
logic_p1_374=np.array(logic_p1_374)
logic_p1_85=np.array(logic_p1_85)

chirp_mass=chirp_q_Lambdabeta6_Lambda1Lambda2[:,0]
q=chirp_q_Lambdabeta6_Lambda1Lambda2[:,1]
Lambda_binary_beta6=chirp_q_Lambdabeta6_Lambda1Lambda2[:,2]
Lambda2Lambda1=chirp_q_Lambdabeta6_Lambda1Lambda2[:,3]
Lambda2Lambda1q6=Lambda2Lambda1*q**6

fig, axes = plt.subplots(2, 3,figsize=(10,6))
for i in range(2):
    for j in range(3):
        L_i=(3*i+j)*5
        axes[i,j].imshow(np.reshape(logic_causality,(N1,N2,N3,N4))[2,L_i].transpose(),aspect='auto',origin='lower',extent=(K_pnm.min(),K_pnm.max(),Q_pnm.min(),Q_pnm.max()))
        axes[i,j].set_title('L=%d MeV'%(L_pnm[L_i]))
        if(j==0):
            axes[i,j].set_ylabel('$Q_n$ MeV')
        if(i==1):
            axes[i,j].set_xlabel('$K_n$ MeV')

fig = plt.figure(figsize=(8,6))  
line_color=['b','c','r','g','y','k','orange','purple','gold']
for j in range(int((maxmass_result[logic][:,1].max()-2.0)*10)+1):
    logic_in_maxmass_range=np.logical_and(maxmass_result[logic][:,1]>2.0+0.1*j,maxmass_result[logic][:,1]<2.0+0.1*(j+1))
    plt.plot(np.concatenate(mass_beta_Lambda_result[logic_in_maxmass_range,0]),np.concatenate(mass_beta_Lambda_result[logic_in_maxmass_range,2])*np.concatenate(mass_beta_Lambda_result[logic_in_maxmass_range,1])**6,'.',color=line_color[j],markersize=1)
    plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='2.%d<M$_{max}$/M$_\odot$<2.%d'%(j,j+1))
plt.xlabel('$M/M_\odot$',fontsize=20)
plt.ylabel('$\Lambda (GM/Rc^2)^6$',fontsize=20)
plt.xlim(1.0,1.8)
plt.ylim(0.006,0.011)
plt.legend(loc=3,frameon=False)

fig = plt.figure(figsize=(8,6))  
line_color=['b','c','r','g','y','k','orange','purple','gold']
for j in range(int((maxmass_result[logic][:,1].max()-2.0)*10)+1):
    logic_in_maxmass_range=np.logical_and(maxmass_result[logic][:,1]>2.0+0.1*j,maxmass_result[logic][:,1]<2.0+0.1*(j+1))
    plt.plot(np.concatenate(chirp_mass[logic_in_maxmass_range]),np.concatenate(Lambda_binary_beta6[logic_in_maxmass_range]),'.',color=line_color[j],markersize=1)
    plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='2.%d<M$_{max}$/M$_\odot$<2.%d'%(j,j+1))
plt.xlabel('$M_{ch}/M_\odot$',fontsize=20)
plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$',fontsize=20)
plt.xlim(np.concatenate(chirp_mass).min(),np.concatenate(chirp_mass).max())
plt.ylim(np.concatenate(Lambda_binary_beta6).min()*0.8,np.concatenate(Lambda_binary_beta6).max())
plt.xlim(1.0,1.8)
plt.ylim(0.00,0.005)
plt.legend(loc=3,frameon=False)


import matplotlib as mpl   
from show_properity import show_properity
from physicalconst import c,G,mass_sun
fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
mass_sun_Gc2=mass_sun*G/c**2/100000
p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)
maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
maxmass_array_sort.sort()
axes[0].set_title('3.74 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
axes[0].set_ylim(1.0,3.0)
show_properity(axes[0],mass_sun_Gc2*mass_beta_Lambda_result[p1_limit_array,0,:]/mass_beta_Lambda_result[p1_limit_array,1,:],mass_beta_Lambda_result[p1_limit_array,0,:],'R(km)','M/M$_\odot$',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)
maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
maxmass_array_sort.sort()
axes[1].set_title('8.5 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
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
fig.savefig(path+dir_name+'/Radius_Mass')

def Low_tidal_cutoff_MC(mass,maxmass):
    mass_over_maxmass=mass/maxmass
    return np.exp(13.42-23.04*mass_over_maxmass+20.56*mass_over_maxmass**2-9.615*mass_over_maxmass**3)
def Low_tidal_cutoff_UG(mass):
    mass_over_maxmass=mass
    return np.exp(18.819-19.862*mass_over_maxmass+10.881*mass_over_maxmass**2-2.5713*mass_over_maxmass**3)
m_plot=np.linspace(1,2,101)
tidal_lower_bound_MC=Low_tidal_cutoff_MC(m_plot,2.0)
tidal_lower_bound_UG=Low_tidal_cutoff_UG(m_plot)

fig, axes = plt.subplots(1, 2,figsize=(12,6),sharex=True,sharey=True)
p1_limit_array=np.logical_and(logic_p1_374,logic_p1_30)
maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
axes[0].set_title('3.74 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
show_properity(axes[0],mass_beta_Lambda_result[p1_limit_array,2,:],mass_beta_Lambda_result[p1_limit_array,0,:],'$\Lambda$','M/M$_\odot$',x_scale='log',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
axes[0].plot(tidal_lower_bound_MC,m_plot,lw=5,label='Maxmimum Compact Bound')
axes[0].plot(tidal_lower_bound_UG,m_plot,lw=5,label='Unitary Gas Bound')
plt.xlim(1,30000)
plt.ylim(1,3)
p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)
maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
maxmass_array_sort=np.linspace(int(maxmass_array.max()*10+1)/10.,2.0,int((maxmass_array.max()-2)*10+2))
axes[1].set_title('8.5 MeV fm$^{-3}$ < p$_1$ < 30 MeV fm$^{-3}$')
show_properity(axes[1],mass_beta_Lambda_result[p1_limit_array,2,:],mass_beta_Lambda_result[p1_limit_array,0,:],'$\Lambda$','M/M$_\odot$',x_scale='log',properity_array_z=maxmass_array,properity_array_z_sort=maxmass_array_sort,z_label='M$_{max}$/M$_\odot$',color_list=line_color)
axes[1].plot(tidal_lower_bound_MC,m_plot,lw=5,label='Maxmimum Compact Bound')
axes[1].plot(tidal_lower_bound_UG,m_plot,lw=5,label='Unitary Gas Bound')
plt.xlim(1,30000)
plt.ylim(1,3)
fig.savefig(path+dir_name+'/Lambda_Mass')


def Lambda1Lambda2_fitting_2018(q,chirp_mass):
    from scipy.interpolate import interp1d
    chirp_mass_grid=np.linspace(1,1.4,9)
    n_=interp1d(chirp_mass_grid,[5.1717,5.2720,5.3786,5.4924,5.6138,5.7449,5.8960,6.0785,6.3047],kind='quadratic')
    n_0=interp1d(chirp_mass_grid,[6.4658,6.7470,7.0984,7.5546,8.1702,8.9715,9.9713,11.234,12.833],kind='quadratic')
    n_1=interp1d(chirp_mass_grid,[-0.2489,-0.32672,-0.44315,-0.62431,-0.91294,-1.3177,-1.8091,-2.3970,-3.0232],kind='quadratic')
    return q**(n_(chirp_mass)),q**(n_0(chirp_mass)+q*n_1(chirp_mass))
    
chirpmass_min=1.0
q_fit=np.linspace(0.7,1.0,31)
plt.plot([0],[0],'--k',lw=2,label='bounds for piecewise polytropic EoS')
for j in (8-np.array(range(9))):
    logic_in_chirpmass_range=np.logical_and(np.concatenate(chirp_mass)>int(chirpmass_min*20+j)/20.,np.concatenate(chirp_mass)<int(chirpmass_min*20+j+1)/20.)
    plt.plot(np.concatenate(q)[logic_in_chirpmass_range],np.concatenate(Lambda2Lambda1q6)[logic_in_chirpmass_range],'.',color=line_color[j],markersize=1)
    plt.plot([0],[0],'.',color=line_color[j],markersize=20,label='%.2f<M$_{ch}$/M$_\odot$<%.2f'%(int(chirpmass_min*20+j)/20.,int(chirpmass_min*20+j+1)/20.))
    fit_lower,fit_upper=Lambda1Lambda2_fitting_2018(q_fit,int(chirpmass_min*20+j)/20.)
    plt.plot(q_fit,q_fit**6/fit_lower,'--',lw=2,color=line_color[j])
    plt.plot(q_fit,q_fit**6/fit_upper,'--',lw=2,color=line_color[j])
plt.xlabel('q',fontsize=20)
plt.ylabel('$\Lambda_2/\Lambda1 q^6$',fontsize=20)
plt.xlim(0.7,1)
plt.ylim(0.5,1.5)
plt.legend(loc=1,frameon=False,fontsize=8)



color_list=['b','c','r','g','y','k','orange','purple','gold','pink','lime','violet']*2
def show_properity(ax,properity_array_x,properity_array_y,x_label,y_label,chirpmass_range,x_scale='linear',y_scale='linear',properity_array_z=1,properity_array_z_sort=[0,2],z_label=1,color_list=color_list):
    for i in range(len(properity_array_z_sort)-1):
        logic_sort=np.logical_and(properity_array_z>=np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),properity_array_z<np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]]))
        logic_in_chirpmass_range=np.logical_and(np.concatenate(chirpmass_range[0][logic_sort])>chirpmass_range[1],np.concatenate(chirpmass_range[0][logic_sort])<chirpmass_range[2])
        ax.plot(np.concatenate(properity_array_x[logic_sort])[logic_in_chirpmass_range],np.concatenate(properity_array_y[logic_sort])[logic_in_chirpmass_range],'.',color=color_list[i])#,label='%.1f<%s<%.1f'%(np.min([properity_array_z_sort[i],properity_array_z_sort[i+1]]),z_label,np.max([properity_array_z_sort[i],properity_array_z_sort[i+1]])))

import matplotlib as mpl   
fig, axes = plt.subplots(4, 2,figsize=(10,16),sharex=True,sharey=True)
chirpmass_min=1.0
q_fit=np.linspace(0.7,1.0,31)
p1_limit_array=np.logical_and(logic_p1_85,logic_p1_30)
maxmass_array=(maxmass_result[logic])[p1_limit_array,1]
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
            axes[ix,iy].set_ylabel('$\Lambda_2/\Lambda1 q^6$',fontsize=20)
        if(ix==3):
            axes[ix,iy].set_xlabel('q',fontsize=20)
fig.savefig(path+dir_name+'/Lambda1Lambda1beta6_Mass')


    
# =============================================================================
# from hull import hull,transform_trivial
# def plot_hull(points,hull_vertices,color,label_tex):
#     for i in range(len(hull_vertices)):
#         if(i==0):
#             plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1], color+'--', lw=2,label=label_tex)
#         plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1],color+'--', lw=2)
# 
# points=[]
# hull_vertices=[]
# for j in range(5):
#     points.append([])
#     for i in range(len(maxmass_result[logic])):
#         if(maxmass_result[logic][i,1]>2.4-0.1*j):
#             points[j]+=list(np.array([chirp_mass[i],Lambda_binary_beta6[i]]).transpose())
#     points[j]=np.array(points[j])
#     hull_vertices.append(hull(points[j],[0,2],transform_trivial,3))
#     plot_hull(points[j],hull_vertices[j],line_color[j],'$M_{max}/M_\odot>2.%d$'%(4-j))
# plt.xlabel('$M_{ch}/M_\odot$')
# plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$')
# plt.title('Hadronic binaries bound')
# plt.xlim(0.8,2.3)
# #plt.ylim(0.003,0.010)
# plt.legend(frameon=False)
# =============================================================================

# =============================================================================
# def Calculation_maxmass(eos):
#     result=[]
#     i=0
#     for eos_i in eos:
#         print(i)
#         i+=1
#         maxmass_result_i=Maxmass(Preset_Pressure_final,Preset_rtol,eos_i)[1:3]
#         result.append(maxmass_result_i+[eos_i.eosCs2(maxmass_result_i[0])])
#     return result
# f_maxmass_result='./'+dir_name+'/Lambda_PNM_calculation_maxmass.dat'
# maxmass_result=np.array(Calculation_maxmass(eos_flat))
# f_file=open(f_maxmass_result,'wb')
# cPickle.dump(maxmass_result,f_file)
# f_file.close()
# logic_maxmass=maxmass_result[:,1]>=2
# logic_causality=maxmass_result[:,2]<1
# logic=np.logical_and(logic_maxmass,logic_causality)
# 
# def show_PNM_eos(eos,x_index,y_index,baryon_density_range,N):#index baryon_density(0), pressure(1), energy density(2), energy per baryon(3), chempo(4)
#     pressure_density_energyPerBaryon_chempo=[]
#     for eos_i in eos:
#         eos_i=eos_i.eosPNM
#         baryon_density_i=np.linspace(baryon_density_range[0],np.min([baryon_density_range[1],eos_i.u_max*eos_i.baryon_density_s]),N)
#         pressure_density_energyPerBaryon_chempo_i=[]
#         pressure_density_energyPerBaryon_chempo_i.append(baryon_density_i)
#         pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosPressure_frombaryon(baryon_density_i))
#         pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_density_energyPerBaryon_chempo_i[1]))
#         pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/baryon_density_i)
#         pressure_density_energyPerBaryon_chempo_i.append((pressure_density_energyPerBaryon_chempo_i[1]+pressure_density_energyPerBaryon_chempo_i[2])/baryon_density_i)
#         pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
#         plt.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index])
#     pressure_density_energyPerBaryon_chempo=np.array(pressure_density_energyPerBaryon_chempo)
#     label_text=['Baryon density(fm$^{-3}$)','Pressure(MeV fm$^{-3}$)','Energy density(MeV fm$^{-3}$)','Energy per baryon(MeV)','Chemical potential(MeV)']
#     plt.xlabel(label_text[x_index])
#     plt.ylabel(label_text[y_index])
#     #plt.xlim(pressure_density_energyPerBaryon_chempo[:,x_index,:].min(),pressure_density_energyPerBaryon_chempo[:,x_index,:].max())
#     #plt.ylim(pressure_density_energyPerBaryon_chempo[:,y_index,:].min(),pressure_density_energyPerBaryon_chempo[:,y_index,:].max())
# 
# 
# show_PNM_eos(eos_flat[logic],2,1,[0.00016,1.85*0.16],100)
# from eos_class import BPS,EOS_BPSwithPoly
# BPSpoly1=EOS_BPSwithPoly([0.059259259259259255, 13.0, 1.85*0.16, 200, 0.5984, 500, 1.1840000000000002])
# BPSpoly2=EOS_BPSwithPoly([0.059259259259259255, 25.0, 1.85*0.16, 200, 0.5984, 500, 1.1840000000000002])
# plt.plot(BPS.eosDensity(np.linspace(0.001,25,100)),(np.linspace(0.001,25,100)),'k',label='SLY4')
# plt.plot(BPSpoly1.eosDensity(np.linspace(0.001,13,100)),(np.linspace(0.001,13,100)),'k--',label='SLY4_piecewisePoly1')
# plt.plot(BPSpoly2.eosDensity(np.linspace(0.001,25,100)),(np.linspace(0.001,25,100)),'k-.',label='SLY4_piecewisePoly2')
# plt.legend()
# 
# plt.figure()
# plt.imshow(np.reshape(logic,(N1,N2,N3))[5].transpose(),aspect='auto',origin='lower',extent=(K_pnm.min(),K_pnm.max(),Q_pnm.min(),Q_pnm.max()))
# plt.title('L=50 MeV')
# plt.ylabel('$Q_n$ MeV')
# plt.xlabel('$K_n$ MeV')
# 
# def Calculation_mass_beta_Lambda(eos,maxmass_result,pc_list=10**np.linspace(0,-1.5,10)):
#     result=[]
#     for i in range(len(eos)):
#         mass=[]
#         beta=[]
#         Lambda=[]
#         for j in range(len(pc_list)):
#             MR_result=MassRadius(maxmass_result[i][0]*pc_list[j],Preset_Pressure_final,Preset_rtol,'MRT',eos[i])
#             mass.append(MR_result[0])
#             beta.append(MR_result[2])
#             Lambda.append(MR_result[4])
#         result.append([mass,beta,Lambda])
#     return result
# f_mass_beta_Lambda_result='./'+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat'
# mass_beta_Lambda_result=np.array(Calculation_mass_beta_Lambda(eos_flat[logic],maxmass_result[logic]))
# f_file=open(f_mass_beta_Lambda_result,'wb')
# cPickle.dump(mass_beta_Lambda_result,f_file)
# f_file.close()
# 
# M_min=1.1
# M_max=1.6
# def mass_chirp(mass1,mass2):
#     return (mass1*mass2)**0.6/(mass1+mass2)**0.2
# def tidal_binary(q,tidal1,tidal2):
#     return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
# def Calculation_chirpmass_Lambdabeta6(args_list,i):
#     mass_beta_Lambda=list(args_list[:,0])
#     beta_onepointfour=args_list[:,1]
#     mass=np.array(mass_beta_Lambda)[:,0]
#     Lambda=np.array(mass_beta_Lambda)[:,2]
#     logic_mass=np.logical_and(mass[i]>M_min,mass[i]<M_max)
#     mass1,mass2 = np.meshgrid(mass[i][logic_mass],mass[i][logic_mass])
#     Lambda1,Lambda2 = np.meshgrid(Lambda[i][logic_mass],Lambda[i][logic_mass])
#     chirp_mass=mass_chirp(mass1,mass2).flatten()
#     Lambda_binary_beta6=(beta_onepointfour[i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2).flatten()
#     return [chirp_mass,Lambda_binary_beta6]
# 
# f_chirpmass_Lambdabeta6_result='./'+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
# main_parallel(Calculation_chirpmass_Lambdabeta6,np.array([list(mass_beta_Lambda_result),list(Properity_onepointfour[:,3])]).transpose(),f_chirpmass_Lambdabeta6_result,0)
# f_file=open(f_chirpmass_Lambdabeta6_result,'rb')
# chirpmass_Lambdabeta6_result=np.array(cPickle.load(f_file))
# f_file.close()
# chirp_mass=chirpmass_Lambdabeta6_result[:,0]
# Lambda_binary_beta6=chirpmass_Lambdabeta6_result[:,1]
# =============================================================================
