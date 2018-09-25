#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 20:37:33 2018

@author: sotzee
"""

from eos_class import EOS_BPSwithPoly,EOS_BPS
import numpy as np
from tov_f import MassRadius
from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass
import scipy.optimize as opt

baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.7*0.16
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
Preset_Pressure_final=1e-8
Preset_rtol=1e-4

def Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i=np.log(pressure_i/pressure_i_minus)/np.log(baryon_density_i/baryon_density_i_minus)
    return gamma_i,(density_i_minus-pressure_i_minus/(gamma_i-1))*\
            (pressure_i/pressure_i_minus)**(1./gamma_i)+pressure_i/(gamma_i-1)

def causality_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i,density_i=Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus)
    return gamma_i*pressure_i/(density_i+pressure_i)-1

def causality_p2(p1):
    density1=Density_i(p1,baryon_density1,pressure0,baryon_density0,density0)[1]
    return opt.newton(causality_i,200.,args=(baryon_density2,p1,baryon_density1,density1))

N1=50
N2=50
N3=50
p1=np.linspace(3.74,30,N1)
p2=[]
p3=[]
Preset_Pressure_final=1e-8
Preset_rtol=1e-4
for i in range(len(p1)):
    p2.append([])
    p2[i]=np.linspace(100,causality_p2(p1[i]),N2)
for i in range(len(p1)):
    p3.append([])
    for j in range(len(p2[i])):
        p3[i].append([])
        density1=Density_i(p1[i],baryon_density1,pressure0,baryon_density0,density0)[1]
        density2=Density_i(p2[i][j],baryon_density2,p1[i],baryon_density1,density1)[1]
        gamma3_max=1+density2/p2[i][j]
        p3_max= p2[i][j]*(baryon_density3/baryon_density2)**gamma3_max
        p3[i][j]=np.exp(np.linspace(np.log(np.max([1.2*p2[i][j],250])),np.log(p3_max),N3))


eos=[]
p1p2p3=[]
for i in range(N1):
    eos.append([])
    p1p2p3.append([])
    for j in range(N2):
        eos[i].append([])
        p1p2p3[i].append([])
        for k  in range(N3):
            eos[i][j].append([])
            eos[i][j][k].append(EOS_BPSwithPoly([baryon_density0,p1[i],baryon_density1,p2[i][j],baryon_density2,p3[i][j][k],baryon_density3]))
            p1p2p3[i][j].append([])
            p1p2p3[i][j][k].append([p1[i],p2[i][j],p3[i][j][k]])
eos_flat=np.array(eos).flatten()
p1p2p3_flat=np.array(p1p2p3).flatten()
def Calculation_maxmass(eos,i):
    maxmass_result=Maxmass(Preset_Pressure_final,Preset_rtol,eos[i])[1:3]
    return maxmass_result+[eos[i].eosCs2(maxmass_result[0])]
def Calculation_onepointfour(args_list,i):
    eos=args_list[:,0]
    maxmass_result=args_list[:,1]
    Properity_onepointfour=Properity_ofmass(1.4,10,maxmass_result[i][0],MassRadius,Preset_Pressure_final,Preset_rtol,1,eos[i])
    return Properity_onepointfour

from Parallel_process import main_parallel
import cPickle
dir_name='Lambda_hadronic_calculation'
f_maxmass_result='./'+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
main_parallel(Calculation_maxmass,eos_flat,f_maxmass_result,0)
f=open(f_maxmass_result,'rb')
maxmass_result=cPickle.load(f)
f.close()

f_onepointfour_result='./'+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat'
main_parallel(Calculation_onepointfour,np.array([eos_flat,maxmass_result]).transpose(),f_onepointfour_result,0)
f=open(f_onepointfour_result,'rb')
Properity_onepointfour=np.array(cPickle.load(f))
f.close()


#M_max>2.0M_sun
logic1=np.array(maxmass_result)[:,1]>2.0
#cs2 at center < 1.0
logic2=np.array(maxmass_result)[:,2]<1.0

logic_eos=np.logical_and(logic1,logic2)
f=open('./'+dir_name+'/Lambda_hadronic_calculation_logic.dat','wb')
cPickle.dump([logic1,logic2,logic_eos],f)
f.close()
f=open('./'+dir_name+'/Lambda_hadronic_calculation_logic.dat','rb')
p1p2p3=np.array(cPickle.load(f))
f.close()

# =============================================================================
# def filter_eos(eos,bool_array):
#     result=[]
#     for i in range(len(bool_array)):
#         if(bool_array[i]):
#             result.append(eos[i])
#     return result
# 
# eos=filter_eos(eos,logic_eos)
# maxmass_result=list(np.array(maxmass_result)[logic_eos])
# Properity_onepointfour=Properity_onepointfour[logic_eos]
# p1p2p3=filter_eos(p1p2p3,logic_eos)
# f=open('./'+dir_name+'Lambda_hadronic_calculation_p1p2p3_filtered.dat','wb')
# cPickle.dump(p1p2p3,f)
# f.close()
# f=open('./'+dir_name+'Lambda_hadronic_calculation_p1p2p3_filtered.dat','rb')
# p1p2p3=np.array(cPickle.load(f))
# f.close()
# =============================================================================

pc_list=10**np.linspace(0,-1.5,40)
def Calculation_mass_beta_Lambda(args_list,i):
    eos=args_list[:,0]
    maxmass_result=args_list[:,1]
    mass=[]
    beta=[]
    Lambda=[]
    for j in range(len(pc_list)):
        MR_result=MassRadius(maxmass_result[i][0]*pc_list[j],Preset_Pressure_final,Preset_rtol,'MRT',eos[i])
        mass.append(MR_result[0])
        beta.append(MR_result[2])
        Lambda.append(MR_result[4])
    return [mass,beta,Lambda]
f_mass_beta_Lambda_result = './'+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
main_parallel(Calculation_mass_beta_Lambda,np.array([eos_flat,maxmass_result]).transpose(),f_mass_beta_Lambda_result,0)
f=open(f_mass_beta_Lambda_result,'rb')
mass_beta_Lambda_result=np.array(cPickle.load(f))
f.close()
mass=mass_beta_Lambda_result[:,0]
beta=mass_beta_Lambda_result[:,1]
Lambda=mass_beta_Lambda_result[:,2]


M_min=1.1
M_max=1.6
def mass_chirp(mass1,mass2):
    return (mass1*mass2)**0.6/(mass1+mass2)**0.2
def tidal_binary(q,tidal1,tidal2):
    return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
def Calculation_chirpmass_Lambdabeta6(args_list,i):
    mass_beta_Lambda=list(args_list[:,0])
    beta_onepointfour=args_list[:,1]
    mass=np.array(mass_beta_Lambda)[:,0]
    Lambda=np.array(mass_beta_Lambda)[:,2]
    logic_mass=np.logical_and(mass[i]>M_min,mass[i]<M_max)
    mass1,mass2 = np.meshgrid(mass[i][logic_mass],mass[i][logic_mass])
    Lambda1,Lambda2 = np.meshgrid(Lambda[i][logic_mass],Lambda[i][logic_mass])
    chirp_mass=mass_chirp(mass1,mass2).flatten()
    Lambda_binary_beta6=(beta_onepointfour[i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2).flatten()
    return [chirp_mass,Lambda_binary_beta6]

f_chirpmass_Lambdabeta6_result='./'+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
main_parallel(Calculation_chirpmass_Lambdabeta6,np.array([list(mass_beta_Lambda_result),list(Properity_onepointfour[:,3])]).transpose(),f_chirpmass_Lambdabeta6_result,0)
f=open(f_chirpmass_Lambdabeta6_result,'rb')
chirpmass_Lambdabeta6_result=np.array(cPickle.load(f))
f.close()
chirp_mass=chirpmass_Lambdabeta6_result[:,0]
Lambda_binary_beta6=chirpmass_Lambdabeta6_result[:,1]



# =============================================================================
# import matplotlib.pyplot as plt
# for i in range(len(eos_flat)):
#     plt.plot(chirp_mass[i],Lambda_binary_beta6[i],'o')
#     
#     
# from hull import hull,transform_trivial
# line_color=['b','c','r','g','y']
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
#     for i in range(len(eos_flat)):
#         if(maxmass_result[i][1]>2.4-0.1*j and eos_flat[i].args[1]>8.4):
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
# 
# 
# def plot_hull(points,hull_vertices,color,label_tex):
#     for i in range(len(hull_vertices)):
#         if(i==0):
#             plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1], color, lw=2,label=label_tex)
#         plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1],color, lw=2)
# points=[]
# hull_vertices=[]
# for j in range(5):
#     points.append([])
#     for i in range(len(eos_flat)):
#         if(maxmass_result[i][1]<2.6-0.1*j and eos_flat[i].args[1]>8.4):
#             points[j]+=list(np.array([chirp_mass[i],Lambda_binary_beta6[i]]).transpose())
#     points[j]=np.array(points[j])
#     hull_vertices.append(hull(points[j],[2],transform_trivial,3))
#     plot_hull(points[j],hull_vertices[j],line_color[j],'$M_{max}/M_\odot<2.%d$'%(6-j))
# plt.xlabel('$M_{ch}/M_\odot$')
# plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$')
# plt.title('Hadronic binaries bound')
# plt.xlim(0.8,2.3)
# #plt.ylim(0.003,0.010)
# plt.legend(frameon=False)
# =============================================================================


