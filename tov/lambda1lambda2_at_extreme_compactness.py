#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:37:41 2018

@author: sotzee
"""

import pickle
import numpy as np
from tov_f import MassRadius_transition,MassRadius
from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass_transition,Maxmass
from eos_class import EOS_BPSwithPoly,EOS_BPSwithPolyCSS,EOS_CSS

# =============================================================================
# #the most compact solution in hadronic case, p1=3.75
# p1=3.74
# p2p3pc=[[101.21815222738961, 710.79698845375674, 942.72049519288259],
#         [121.21975283589097, 837.30085284760423, 873.24000207465804],
#         [144.53076103783107, 975.61506701822998, 820.54287640952191],
#         [171.74825820803059, 1125.3695389439354, 781.8923081137159],
#         [203.7369405017611, 1285.6457824339805, 757.58014316979109]]
# 
# #the most compact solution in hybrid case, p1=3.75
# p1=3.74
# pt=1.26989404186196
# det_density=[476.3644129763533,
#             417.87079745373524,
#             367.20898903066376,
#             323.050507686911,
#             284.3286532307534,]
# =============================================================================

#the most compact solution in hadronic case, p1=8.4
p1=8.4
p2p3pc=[[103.55029671995027, 737.42059473457505, 925.67431026853399],
        [123.84669331422306, 869.01175124385668, 856.62060993702289],
        [147.39665767976882, 1013.1473798583421, 802.40778750457184],
        [174.8503688381025, 1169.0191568187433, 760.73599576128572],
        [207.05212614544061, 1334.9104339044127, 734.06201018868228]]

#the most compact solution in hybrid case, p1=8.4
p1=8.4
pt=2.0930077407742105
det_density=[476.68042742146446,
            418.46469717856615,
            368.0284456994796,
            323.8088207494245,
            285.0386365985221]

# =============================================================================
# #the most compact solution in hadronic case, p2 and pc are causal maximum, tune p1 for maximum mass.
# #maximum mass=[2.3,2.4,2.5]
# p1p2p3pc=   [[7.504535755576949,174.42818350272577,1161.680436463556,764.7238529543927],
#             [12.495026542999506,208.12562484237642,1369.1284992868104,714.0889741061044]]
# =============================================================================


baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.7*0.16
baryon_density3 = 7.4*0.16
Preset_Pressure_final=1e-8
Preset_rtol=1e-4
eos_hadronic_compact=[]
eos_hybrid_compact=[]
for i in range(5):
    eos_hadronic_compact.append(EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2p3pc[i][0],baryon_density2,p2p3pc[i][1],baryon_density3]))
    eos_hybrid_compact.append(EOS_BPSwithPolyCSS([baryon_density0,p1,baryon_density1,100,baryon_density2,1000,baryon_density3,pt,det_density[i],1.0]))

maxmass=[2.0,2.1,2.2,2.3,2.4]
pc_ratio=10**np.linspace(0,-1.2,25)
Lambda_hadronic_compact=[]
beta_hadronic_compact=[]
mass_hadronic_compact=[]
Lambda_hybrid_compact=[]
beta_hybrid_compact=[]
mass_hybrid_compact=[]
pc_hadronic_max=[]
pc_hadronic=[]
pc_hybrid_max=[]
pc_hybrid=[]
for i in range(5):
    Lambda_hadronic_compact.append([])
    Lambda_hybrid_compact.append([])
    beta_hadronic_compact.append([])
    beta_hybrid_compact.append([])
    mass_hadronic_compact.append([])
    mass_hybrid_compact.append([])
    pc_hadronic_max.append(p2p3pc[i][2])
    pc_hadronic.append(pc_hadronic_max[i]*pc_ratio)
    pc_hybrid_max.append(Maxmass_transition(Preset_Pressure_final,Preset_rtol,eos_hybrid_compact[i])[1])
    pc_hybrid.append(pc_hybrid_max[i]*pc_ratio)


for i in range(5):
    print i
    for j in range(len(pc_ratio)):
        MRBIT_result=MassRadius(pc_hadronic[i][j],Preset_Pressure_final,Preset_rtol,'MRBIT',eos_hadronic_compact[i])
        Lambda_hadronic_compact[i].append(MRBIT_result[6])
        beta_hadronic_compact[i].append(MRBIT_result[2])
        mass_hadronic_compact[i].append(MRBIT_result[0])
        MRBIT_result=MassRadius_transition(pc_hybrid[i][j],Preset_Pressure_final,Preset_rtol,'MRBIT',eos_hybrid_compact[i])
        Lambda_hybrid_compact[i].append(MRBIT_result[6])
        beta_hybrid_compact[i].append(MRBIT_result[2])
        mass_hybrid_compact[i].append(MRBIT_result[0])

Lambda_hadronic_compact=np.array(Lambda_hadronic_compact)
beta_hadronic_compact=np.array(beta_hadronic_compact)
mass_hadronic_compact=np.array(mass_hadronic_compact)
pc_hadronic=np.array(pc_hadronic)
Lambda_hybrid_compact=np.array(Lambda_hybrid_compact)
beta_hybrid_compact=np.array(beta_hybrid_compact)
mass_hybrid_compact=np.array(mass_hybrid_compact)
pc_hybrid=np.array(pc_hybrid)



# =============================================================================
# f=open('./hadronic_hybrid_eos_at_extreme_compactness.dat','rb')
# eos_hadronic_lower,eos_hadronic_upper,eos_hybrid_lower,eos_hybrid_upper=pickle.load(f)
# f.close()
# =============================================================================


def bound_with_eos(m1,m2,pc_low,pc_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    tidal1=Properity_ofmass(m1,pc_low,pc_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,1,eos)[7]
    tidal2=Properity_ofmass(m2,pc_low,pc_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,1,eos)[7]
    return tidal1,tidal2
def get_bound(m1_grid,m2_grid,pc_low,Maxmass_function,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    tidal1=m1_grid.copy()
    tidal2=m1_grid.copy()
    pc_high=Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)[1]
    for i in range(len(m1_grid)):
        print i
        for j in range(len(m1_grid[0])):
            tidal1[i,j],tidal2[i,j]= bound_with_eos(m1_grid[i,j],m2_grid[i,j],pc_low,pc_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos)
    return tidal1,tidal2
def mass_binary(mc,q):
    return [mc*(1+q)**0.2*q**0.4,mc*(1+q)**0.2/q**0.6]
def tidal_binary(q,tidal1,tidal2):
    return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
chip_mass= np.linspace(1.05, 1.4,8)
q=np.linspace(0.7,1.,13)
chip_mass_grid,q_grid = np.meshgrid(chip_mass,q)

m2_grid,m1_grid=mass_binary(chip_mass_grid,q_grid)  #q=m2/m1

Preset_Pressure_final=1e-8
Preset_rtol=1e-4

tidal1_hadronic_lower,tidal2_hadronic_lower,tidal_binary_hadronic_lower = get_bound(m1_grid,m2_grid,50.,Maxmass,MassRadius,1e-8,1e-4,eos_hadronic_lower)
tidal1_hadronic_upper,tidal2_hadronic_upper = get_bound(m1_grid,m2_grid,10.,Maxmass,MassRadius,1e-8,1e-4,eos_hadronic_upper)
tidal1_hybrid_lower,tidal2_hybrid_lower = get_bound(m1_grid,m2_grid,100.,Maxmass_transition,MassRadius_transition,1e-8,1e-4,eos_hybrid_lower)
tidal1_hybrid_upper,tidal2_hybrid_upper = get_bound(m1_grid,m2_grid,10.,Maxmass_transition,MassRadius_transition,1e-8,1e-4,eos_hybrid_upper)

f=open('hadronic_at_extreme_compactness.dat','wb')
pickle.dump([[tidal1_hadronic_lower,tidal2_hadronic_lower],[tidal1_hadronic_upper,tidal2_hadronic_upper]],f)
f.close()
f=open('hybrid_at_extreme_compactness.dat.dat','wb')
pickle.dump([[tidal1_hybrid_lower,tidal2_hybrid_lower],[tidal1_hybrid_upper,tidal2_hybrid_upper]],f)
f.close()

import matplotlib.pyplot as plt
n=6
maxmass_min=2.0
pressure1_min=3.75
pressure1_max=30.
cmap = plt.cm.get_cmap('jet')
(np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int)
colors = cmap((np.array(cmap.N*np.linspace(0,1,len(chip_mass)))).astype(int))

f, axs= plt.subplots(1,2,figsize=(10, 6))
for i in range(len(q)):
    axs[0].plot(chip_mass,tidal_binary_hadronic_lower[i,:],'r')
    axs[0].plot(chip_mass,tidal_binary_hadronic_upper[i,:],'b')
    axs[0].plot(chip_mass,tidal_binary_hybrid_lower[i,:],'g')
    axs[0].plot(chip_mass,tidal_binary_hybrid_upper[i,:],'c')
axs[0].set_yscale('log')
axs[0].set_title('upper and lower bound of binary tidal deformability')
axs[0].set_xlabel('M$_{ch}$/M$_\odot$',fontsize=18)
axs[0].set_ylabel('$\\bar \Lambda$',fontsize=18)
axs[0].set_xlim(1.05,1.4)
axs[0].plot([1.188,1.188],[76,989],'k',label='1-$\sigma$ bound Soumi De et.al. 2018')
axs[0].plot([1.188],[800],marker='*',markersize=20,label='90% upper limit B.P. Abbott et.al. 2017')
axs[0].legend(loc=3,prop={'size':12},frameon=False)

def plot_MR(pressure_center,MassRadius_function,eos,subplot,COLOR,label_tex):
    result_mass=[]
    result_radius=[]
    for i in range(len(pressure_center)):
        result=MassRadius_function(pressure_center[i],1e-8,1e-4,'MR',eos)
        result_mass.append(result[0])
        result_radius.append(result[1])
    subplot.plot(np.array(result_radius)/1000,result_mass,color=COLOR,label=label_tex)
N=100
pressure_center=np.linspace(50,Maxmass(Preset_Pressure_final,Preset_rtol,eos_hadronic_lower)[1],N)
plot_MR(pressure_center,MassRadius,eos_hadronic_lower,axs[1],'r','hadronic star lower bound')
pressure_center=np.linspace(10,Maxmass(Preset_Pressure_final,Preset_rtol,eos_hadronic_upper)[1],N)
plot_MR(pressure_center,MassRadius,eos_hadronic_upper,axs[1],'b','hadronic star upper bound')
pressure_center=np.linspace(100,Maxmass_transition(Preset_Pressure_final,Preset_rtol,eos_hybrid_lower)[1],N)
plot_MR(pressure_center,MassRadius_transition,eos_hybrid_lower,axs[1],'g','hybrid star lower bound')
pressure_center=np.linspace(10,Maxmass_transition(Preset_Pressure_final,Preset_rtol,eos_hybrid_upper)[1],N)
plot_MR(pressure_center,MassRadius_transition,eos_hybrid_upper,axs[1],'c','hybrid star upper bound')
axs[1].legend(loc=2,prop={'size':15},frameon=False)
axs[1].set_ylim(1.0,2.75)
axs[1].set_title('M-R curves of corresponding configuration')
axs[1].set_xlabel('R(km)',fontsize=18)
axs[1].set_ylabel('M/M$_\odot$',fontsize=18)
plt.tight_layout()


for i in range(len(chip_mass)):
    axs[0].plot(list(q)+[1],list(q**n*tidal2_upper[:,i]/tidal1_upper[:,i])+[1],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[0].legend(loc=1,prop={'size':10},frameon=False)
    axs[0].set_title('$M_{max}>%.1f M_\odot$ upper bound'%(maxmass_min))
    axs[0].set_xlabel('q',fontsize=18)
    axs[0].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[0].set_ylim(0,180)
    #axs[0].set_yscale('log')
    
    axs[1].plot(list(q)+[1],list(q**n*tidal2_lower[:,i]/tidal1_lower[:,i])+[1],color=colors[i],label='$M_{ch}$=%.2f'%chip_mass[i])
    axs[1].legend(loc=4,prop={'size':10},frameon=False)
    axs[1].set_title('$p_1>%.2f$ MeV fm$^{-3}$ lower bound'%(pressure1_min))
    axs[1].set_xlabel('q',fontsize=18)
    axs[1].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    #axs[1].set_ylim(1,7)
    #axs[1].set_yscale('log')
    
    bound_log=axs[2].scatter(q,q*0.0,c=[chip_mass[i]]*len(q),cmap=plt.cm.jet)
    axs[2].plot(list(q)+[1],list(q**n*tidal2_upper[:,i]/tidal1_upper[:,i])+[1],color=colors[i])
    axs[2].plot(list(q)+[1],list(q**n*tidal2_lower[:,i]/tidal1_lower[:,i])+[1],'--',color=colors[i])
    if(i==0):
        axs[2].plot([0,0,0],[0,1,2],color='k',label='$M_{max}>%.1f M_\odot$ upper bound'%(maxmass_min))
        axs[2].plot([0,0,0],[0,1,2],'--',color='k',label='$p_1>%.2f$ MeV fm$^{-3}$ lower bound'%(pressure1_min))
    axs[2].legend(loc=1,prop={'size':8},frameon=False)
    axs[2].set_title('upper and lower bounds')
    axs[2].set_xlabel('q',fontsize=18)
    axs[2].set_ylabel('$q^6 \Lambda_2 /\Lambda_1$',fontsize=18)
    axs[2].set_ylim(0.5,22)
    axs[2].set_yscale('log')

f.colorbar(bound_log,ax=axs[2])
bound_log.set_clim(chip_mass[0], chip_mass[-1])
plt.xlim(0.7,1)
plt.show()
