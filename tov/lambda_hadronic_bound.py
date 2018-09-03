#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:14:27 2018

@author: sotzee
"""
from eos_class import EOS_BPSwithPoly,EOS_BPS
import numpy as np
from tov_f import MassRadius
from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass
import scipy.optimize as opt
import matplotlib.pyplot as plt

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
    return opt.newton(causality_i,200.,tol=1e-4,args=(baryon_density2,p1,baryon_density1,density1))

def causality_center(p3,p1,p2):
    a=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,p3,baryon_density3])
    return a.eosCs2(Maxmass(Preset_Pressure_final,Preset_rtol,a)[1])-1
def causality_p3(p1,p2):
    p3=opt.newton(causality_center,p2*6.5,tol=1e-1,args=(p1,p2))
    a=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,p3,baryon_density3])
    pc_max=Maxmass(Preset_Pressure_final,Preset_rtol,a)[1]
    return p3,pc_max
    
    
def func(x,a,b,c,d,e,f):
    return a+b*x+c*x**2+d*x**3+e*x**4+f*x**5


pc_list=10**np.linspace(0,-1.5,20)
p1=np.linspace(8.4,30,2)
beta_no_correction=[]
beta=[]
Lambda=[]
mass=[]
eos=[]
pc_eos=[]
import pickle
f=open('./hadronic_upper_bound_p1_p2p3pc_fit_result_2.0.dat','rb')
fit_result_p2,fit_result_p3,fit_result_pc_maxmass=pickle.load(f)
f.close()
for j in range(len(p1)):
    print j
    beta_no_correction.append([])
    beta.append([])
    Lambda.append([])
    mass.append([])
# =============================================================================
#     eos.append(EOS_BPSwithPoly([0.059259259259259255,
#      p1[j],
#      0.29600000000000004,
#      func(p1[j],*fit_result_p2[0]),
#      0.5984,
#      func(p1[j],*fit_result_p3[0]),
#      1.1840000000000002]))
#     pc_eos.append(Maxmass(Preset_Pressure_final,Preset_rtol,eos[j])[1])
# =============================================================================

    p2=causality_p2(p1[j])
    eos.append(EOS_BPSwithPoly([0.059259259259259255,
     p1[j],
     baryon_density1,
     p2,
     baryon_density2,
     1.2*p2,
     baryon_density3]))
    pc_eos.append(1.5*p2)
    for i in range(len(pc_list)):
        result_upper_bound_no_maxmass=MassRadius(pc_eos[j]*pc_list[i],Preset_Pressure_final,Preset_rtol,'MRBIT',eos[j])
        mass[j].append(result_upper_bound_no_maxmass[0])
        Lambda[j].append(result_upper_bound_no_maxmass[6])
        beta_no_correction[j].append(result_upper_bound_no_maxmass[2])
        beta[j].append(result_upper_bound_no_maxmass[2]/Radius_correction_ratio(pc_eos[j]*pc_list[i],Preset_Pressure_final,result_upper_bound_no_maxmass[2],eos[j]))
    
beta_no_correction=np.array(beta_no_correction)
beta=np.array(beta)
Lambda=np.array(Lambda)
mass=np.array(mass)
for j in range(len(eos)):
    plt.plot(mass[j],beta[j]**6*Lambda[j],label='p1=%d'%p1[j])
plt.xlabel('$M/M_\odot$')
plt.ylabel('$\Lambda \\beta^6$')
plt.xlim(1.,1.6)
plt.ylim(0.003,0.010)
plt.legend()




import pickle
f=open('./hadronic_upper_bound_p1_p2p3pc_fit_result_2.0.dat','rb')
fit_result_p2,fit_result_p3,fit_result_pc_maxmass=pickle.load(f)
f.close()
#func(x,*fit_result_p2[0])
def beta6lambda_ofmass(p1,ofmass):
    eos=EOS_BPSwithPoly([baryon_density0,
      p1,
      baryon_density1,
      func(p1,*fit_result_p2[0]),
      baryon_density2,
      func(p1,*fit_result_p3[0]),
      baryon_density3])
    ofmass_result=Properity_ofmass(ofmass,p1,func(p1,*fit_result_pc_maxmass[0]),MassRadius,Preset_Pressure_final,Preset_rtol,1.0,eos)
    print p1,(ofmass_result[3])**6*ofmass_result[7]
    return (ofmass_result[3])**6*ofmass_result[7]
print opt.minimize(beta6lambda_ofmass,20.,args=(1.6,),tol=1e-03,method='Nelder-Mead')

#Lambda*beta**6 bound
lower_bound_p1=[16.76171875, 16.31640625, 16.5234375, 16.875, 17.490234375]
lower_bound_p2=[105.14796003297099,125.37650780304907,148.83080036594265,176.08807511574906,207.87137636674274]
lower_bound_p3=[770.00099208777806,906.6066208960541,1059.2581321601538,1225.3575962054197,1405.0576563527995]
lower_bound_pc_maxmass=[891.94617460746917,823.54374309717764,764.7251041526473,721.2344048234047,690.03631320222746]
for i in range(5):
    f=open('./hadronic_upper_bound_p1_p2p3pc_fit_result_2.%d.dat'%i,'rb')
    fit_result_p2,fit_result_p3,fit_result_pc_maxmass=pickle.load(f)
    f.close()
    minimize_result=opt.minimize(beta6lambda_ofmass,20.,args=(1.6,),tol=1e-03,method='Nelder-Mead')
    lower_bound_p1.append(minimize_result.x[0])
    lower_bound_p2.append(func(minimize_result.x[0],*fit_result_p2[0]))
    lower_bound_p3.append(func(minimize_result.x[0],*fit_result_p3[0]))
    lower_bound_pc_maxmass.append(func(minimize_result.x[0],*fit_result_pc_maxmass[0]))
    
lower_bound_beta_one_point_four=[]
eos_lower_bound=[]
for i in range(len(lower_bound_p1)):
    eos_lower_bound.append(EOS_BPSwithPoly([baryon_density0,lower_bound_p1[i],baryon_density1,lower_bound_p2[i],baryon_density2,lower_bound_p3[i],baryon_density3]))
    ofmass_result=Properity_ofmass(1.4,lower_bound_p1[i],lower_bound_pc_maxmass[i],MassRadius,Preset_Pressure_final,Preset_rtol,1.0,eos_lower_bound[i])
    lower_bound_beta_one_point_four.append(ofmass_result[3])

upper_bound_p1=[3.74,4.06,7.51,8.4,12.49,30.]
upper_bound_p2=[141.46621614222298,144.76149521439103,174.46951849156656,181.0373156703981,208.09475251740099,298.98747443548757]
upper_bound_p3=[958.1,978.46880147527577,1161.897025044974,1203.2,1368.7681496008929,1949.2]
upper_bound_pc_maxmass=[826.751708984375,819.36279296875,764.66796875,753.30078125,717.2314453125,597.7783203125]
upper_bound_beta_one_point_four=[]
eos_upper_bound=[]
for i in range(len(upper_bound_p1)):
    eos_upper_bound.append(EOS_BPSwithPoly([baryon_density0,upper_bound_p1[i],baryon_density1,upper_bound_p2[i],baryon_density2,upper_bound_p3[i],baryon_density3]))
    ofmass_result=Properity_ofmass(1.4,upper_bound_p1[i],upper_bound_pc_maxmass[i],MassRadius,Preset_Pressure_final,Preset_rtol,1.0,eos_upper_bound[i])
    upper_bound_beta_one_point_four.append(ofmass_result[3])

pc_list=10**np.linspace(0,-1.5,20)
beta=[]
Lambda=[]
mass=[]
eos=[]
pc_eos=[]
beta_one_point_four=[]
for j in range(len(eos_lower_bound+eos_upper_bound)):
    print j
    beta.append([])
    Lambda.append([])
    mass.append([])
    eos.append((eos_lower_bound+eos_upper_bound)[j])
    pc_eos.append((lower_bound_pc_maxmass+upper_bound_pc_maxmass)[j])
    beta_one_point_four.append((lower_bound_beta_one_point_four+upper_bound_beta_one_point_four)[j])
    for i in range(len(pc_list)):
        result_upper_bound_no_maxmass=MassRadius(pc_eos[j]*pc_list[i],Preset_Pressure_final,Preset_rtol,'MRBIT',eos[j])
        mass[j].append(result_upper_bound_no_maxmass[0])
        Lambda[j].append(result_upper_bound_no_maxmass[6])
        beta[j].append(result_upper_bound_no_maxmass[2])
    
beta=np.array(beta)
Lambda=np.array(Lambda)
mass=np.array(mass)

plt.figure(figsize=(10,8))
for j in range(len(eos)-6):
    plt.plot(mass[j],beta[j]**6*Lambda[j],label='lower bound $M_{max}/M_\odot>2.%d$'%j)
j=len(eos)-6
plt.plot(mass[j],beta[j]**6*Lambda[j],'--',label='upper bound $p_1>%.1f MeV fm^{-3}$'%(upper_bound_p1[0]))
j=len(eos)-5
plt.plot(mass[j],beta[j]**6*Lambda[j],'--',label='upper bound $M_{max}/M_\odot>2.2$')
j=len(eos)-4
plt.plot(mass[j],beta[j]**6*Lambda[j],'--',label='upper bound $M_{max}/M_\odot>2.3$')
j=len(eos)-3
plt.plot(mass[j],beta[j]**6*Lambda[j],'--',label='upper bound $p_1>%.1f MeV fm^{-3}$'%(upper_bound_p1[3]))
j=len(eos)-2
plt.plot(mass[j],beta[j]**6*Lambda[j],'--',label='upper bound $M_{max}/M_\odot>2.4$')
j=len(eos)-1
plt.plot(mass[j],beta[j]**6*Lambda[j],'--',label='upper bound $p_1<%.1f MeV fm^{-3}$'%(upper_bound_p1[5]))

plt.xlabel('$M/M_\odot$')
plt.ylabel('$\Lambda \\beta^6$')
plt.title('Hadronic single star bound')
plt.xlim(1.0,2.65)
plt.ylim(0.003,0.011)
plt.legend(frameon=False,fontsize=1)

# =============================================================================
# def mass_chirp(mass1,mass2):
#     return (mass1*mass2)**0.6/(mass1+mass2)**0.2
# def tidal_binary(q,tidal1,tidal2):
#     return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
# 
# from hull import hull,transform_trivial
# line_color=['b','c','r','g','y','k','m','orange']
# def plot_hull(points,hull_vertices,color,label_tex):
#     for i in range(len(hull_vertices)):
#         if(i==0):
#             plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1], color+'--', lw=2,label=label_tex)
#         plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1],color+'--', lw=2)
# points=[]
# for i in range(8):
#     points=list(points)
#     mass1,mass2 = np.meshgrid(mass[i][mass[i]>1.0],mass[i][mass[i]>1.0])
#     Lambda1,Lambda2 = np.meshgrid(Lambda[i][mass[i]>1.0],Lambda[i][mass[i]>1.0])
#     chirp_mass=mass_chirp(mass1,mass2)
#         #plt.plot(chirp_mass.flatten(),((beta_one_point_four[j][i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2)).flatten(),'.',label='lower bound $M_{max}/M_\odot>2.%d$'%j)
#     points=list(np.array([chirp_mass.flatten(),((beta_one_point_four[i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2)).flatten()]).transpose())
#     points=np.array(points)
#     hull_vertices=hull(points,[0,2],transform_trivial,3)
#     plot_hull(points,hull_vertices,line_color[i],'$M_{max}/M_\odot>2.%d$'%i)
# plt.xlabel('$M_{ch}/M_\odot$')
# plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$')
# plt.title('Hadronic binaries bound')
# plt.xlim(0.8,2.3)
# #plt.ylim(0.003,0.010)
# plt.legend(frameon=False)
# =============================================================================







eos_lower_bound=[]
lower_bound_p1=[]
lower_bound_p2=[]
lower_bound_p3=[]
lower_bound_pc_maxmass=[]
lower_bound_beta_one_point_four=[]
for i in range(5):
    f=open('./hadronic_upper_bound_p1_p2p3pc_fit_result_2.%d.dat'%(i),'rb')
    fit_result_p2,fit_result_p3,fit_result_pc_maxmass=pickle.load(f)
    f.close()
    lower_bound_p1.append(np.array([3.74,8.4,16,24,30]))
    lower_bound_p2.append([])
    lower_bound_p3.append([])
    lower_bound_pc_maxmass.append([])
    eos_lower_bound.append([])
    lower_bound_beta_one_point_four.append([])
    for j in range(len(lower_bound_p1[i])):
        lower_bound_p2[i].append(func(lower_bound_p1[i][j],*fit_result_p2[0]))
        lower_bound_p3[i].append(func(lower_bound_p1[i][j],*fit_result_p3[0]))
        lower_bound_pc_maxmass[i].append(func(lower_bound_p1[i][j],*fit_result_pc_maxmass[0]))
        eos_lower_bound[i].append(EOS_BPSwithPoly([baryon_density0,lower_bound_p1[i][j],baryon_density1,lower_bound_p2[i][j],baryon_density2,lower_bound_p3[i][j],baryon_density3]))
        ofmass_result=Properity_ofmass(1.4,lower_bound_p1[i][j],lower_bound_pc_maxmass[i][j],MassRadius,Preset_Pressure_final,Preset_rtol,1.0,eos_lower_bound[i][j])
        lower_bound_beta_one_point_four[i].append(ofmass_result[3])

upper_bound_p1=[3.74,8.4,30.]
upper_bound_p2=[141.46621614222298,181.0373156703981,298.98747443548757]
upper_bound_p3=[958.1,1203.2,1949.2]
upper_bound_pc_maxmass=[826.751708984375,753.30078125,597.7783203125]
upper_bound_beta_one_point_four=[]
eos_upper_bound=[]
for i in range(len(upper_bound_p1)):
    eos_upper_bound.append(EOS_BPSwithPoly([baryon_density0,upper_bound_p1[i],baryon_density1,upper_bound_p2[i],baryon_density2,upper_bound_p3[i],baryon_density3]))
    ofmass_result=Properity_ofmass(1.4,upper_bound_p1[i],upper_bound_pc_maxmass[i],MassRadius,Preset_Pressure_final,Preset_rtol,1.0,eos_upper_bound[i])
    upper_bound_beta_one_point_four.append(ofmass_result[3])
eos_upper_bound=[]
pc_list=10**np.linspace(0,-1.5,20)
beta=[]
Lambda=[]
mass=[]
eos=[]
pc_eos=[]
beta_one_point_four=[]

for k in range(5):
    beta.append([])
    Lambda.append([])
    mass.append([])
    eos.append([])
    pc_eos.append([])
    beta_one_point_four.append([])
    for j in range(len(eos_lower_bound[k]+eos_upper_bound)):
        print j
        beta[k].append([])
        Lambda[k].append([])
        mass[k].append([])
        eos[k].append((eos_lower_bound[k]+eos_upper_bound)[j])
        pc_eos[k].append((lower_bound_pc_maxmass[k]+upper_bound_pc_maxmass)[j])
        beta_one_point_four[k].append((lower_bound_beta_one_point_four[k]+upper_bound_beta_one_point_four)[j])
        for i in range(len(pc_list)):
            result_upper_bound_no_maxmass=MassRadius(pc_eos[k][j]*pc_list[i],Preset_Pressure_final,Preset_rtol,'MRBIT',eos[k][j])
            mass[k][j].append(result_upper_bound_no_maxmass[0])
            Lambda[k][j].append(result_upper_bound_no_maxmass[6])
            beta[k][j].append(result_upper_bound_no_maxmass[2])
beta=np.array(beta)
Lambda=np.array(Lambda)
mass=np.array(mass)

def mass_chirp(mass1,mass2):
    return (mass1*mass2)**0.6/(mass1+mass2)**0.2
def tidal_binary(q,tidal1,tidal2):
    return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)

from hull import hull,transform_trivial
line_color=['b','c','r','g','y']
def plot_hull(points,hull_vertices,color,label_tex):
    for i in range(len(hull_vertices)):
        if(i==0):
            plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1], color+'--', lw=2,label=label_tex)
        plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1],color+'--', lw=2)
points=[]
for j in [4,3,2,1,0]:
    points=list(points)
    for i in range(len(eos[j])):
        mass1,mass2 = np.meshgrid(mass[j][i][mass[j][i]>1.0],mass[j][i][mass[j][i]>1.0])
        Lambda1,Lambda2 = np.meshgrid(Lambda[j][i][mass[j][i]>1.0],Lambda[j][i][mass[j][i]>1.0])
        chirp_mass=mass_chirp(mass1,mass2)
        #plt.plot(chirp_mass.flatten(),((beta_one_point_four[j][i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2)).flatten(),'.',label='lower bound $M_{max}/M_\odot>2.%d$'%j)
        points+=list(np.array([chirp_mass.flatten(),((beta_one_point_four[j][i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2)).flatten()]).transpose())
    points=np.array(points)
    hull_vertices=hull(points,[0,2],transform_trivial,3)
    plot_hull(points,hull_vertices,line_color[j],'$M_{max}/M_\odot>2.%d$'%j)
plt.xlabel('$M_{ch}/M_\odot$')
plt.ylabel('$\\bar \Lambda (M_{ch}/R_{1.4})^6$')
plt.title('Hadronic binaries bound')
plt.xlim(0.8,2.3)
#plt.ylim(0.003,0.010)
plt.legend(frameon=False)
