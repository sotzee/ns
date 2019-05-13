#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:34:31 2018

@author: sotzee
"""

from eos_class import EOS_BPSwithPoly,EOS_PiecewisePoly,EOS_BPS
import numpy as np
from tov_f import MassRadius
from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass
import scipy.optimize as opt
from scipy.misc import derivative
import matplotlib.pyplot as plt

baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
Preset_Pressure_final=1e-8
Preset_rtol=1e-6

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


def causality_central_pressure(pressure_center,density2,pressure2,gamma3):
    #print pressure_center>0,(gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)
    #print baryon_density1,baryon_density2
# =============================================================================
#     pressure3=pressure2*(baryon_density3/baryon_density2)**gamma3
#     eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
#     print pressure_center,eos.eosCs2(pressure_center)
# =============================================================================
    return np.where(pressure_center>0,(gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)-1,-1+pressure_center/1000.)

def caulality_central_pressure_at_peak(pressure3,pressure1,pressure2,Preset_Pressure_final,Preset_rtol):
    eos = EOS_PiecewisePoly([density0,pressure0,baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
    pressure_center=opt.newton(causality_central_pressure,0.3*pressure3,tol=0.1,args=(eos.density2,pressure2,eos.gamma3))    
    derivative_center_pressure=derivative(MassRadius,pressure_center,dx=1e-2,args=(Preset_Pressure_final,Preset_rtol,'M',eos))
    #print pressure3, pressure_center, derivative_center_pressure
    return derivative_center_pressure

def trial_p2(p1,of_maxmass):
    p2_standard=(360*(of_maxmass-2.0)**1.3+103)
    gamma2_standard=np.log(p2_standard/p1)/np.log(3.7*0.16/baryon_density1)
    return p1*(baryon_density2/baryon_density1)**gamma2_standard

def trial_p3(p1,p2):
    gamma2_standard=np.log(p2/p1)/np.log(baryon_density2/baryon_density1)
    p2_standard=p1*(3.7*0.16/baryon_density1)**gamma2_standard
    return 7.1*p2_standard

def p2p3_ofmaxmass(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1):
    print '==================Finding p2 of maxmass%.2f at p1=%.2f'%(ofmaxmass,p1)
    pressure3_result=[0]
    pressure_center_maxmass=[0]
    def Ofmaxmass(p2,ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,args_p1):
        print '==================Finding p3 at p2=%f'%p2
        pressure3=opt.newton(caulality_central_pressure_at_peak,trial_p3(args_p1,p2),tol=0.1,args=(args_p1,p2,Preset_Pressure_final,Preset_rtol))
        pressure3_result[0]=pressure3
        args=[baryon_density0,args_p1,baryon_density1,p2,baryon_density2,pressure3,baryon_density3]
        eos=EOS_BPSwithPoly(args)
        maxmass_result = Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)
        pressure_center_maxmass[0]=maxmass_result[1]
        print 'maxmass=%f'%maxmass_result[2]
        return -ofmaxmass+maxmass_result[2]
    Preset_p2=trial_p2(p1,ofmaxmass)
    result=opt.newton(Ofmaxmass,Preset_p2,tol=0.1,args=(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1))
    return result,pressure3_result[0],pressure_center_maxmass[0]

upper_bound_eos=[]
upper_bound_pc=[]
lower_bound_eos=[]
lower_bound_pc=[]

baryon_density1 = 1.85*0.16
baryon_density2 = 3.2*0.16
pressure1=30
pressure2=causality_p2(pressure1)
Preset_Pressure_final=1e-8
Preset_rtol=1e-6
pressure3=opt.newton(caulality_central_pressure_at_peak,trial_p3(pressure1,pressure2),tol=0.1,args=(pressure1,pressure2,Preset_Pressure_final,Preset_rtol))
upper_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
upper_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,upper_bound_eos[-1])[1])
print upper_bound_eos[-1].args
pressure1=8.4
pressure2,pressure3,pressure_center=p2p3_ofmaxmass(2.0,Maxmass,Preset_Pressure_final,Preset_rtol,pressure1)
lower_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
lower_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,lower_bound_eos[-1])[1])
print lower_bound_eos[-1].args

baryon_density1 = 1.85*0.16
baryon_density2 = 3.7*0.16
pressure1=30
pressure2=causality_p2(pressure1)
Preset_Pressure_final=1e-8
Preset_rtol=1e-8
pressure3=opt.newton(caulality_central_pressure_at_peak,trial_p3(pressure1,pressure2),tol=0.1,args=(pressure1,pressure2,Preset_Pressure_final,Preset_rtol))
upper_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
upper_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,upper_bound_eos[-1])[1])
print upper_bound_eos[-1].args
pressure1=8.4
pressure2,pressure3,pressure_center=p2p3_ofmaxmass(2.0,Maxmass,Preset_Pressure_final,Preset_rtol,pressure1)
lower_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
lower_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,lower_bound_eos[-1])[1])
print lower_bound_eos[-1].args


baryon_density1 = 1.85*0.16
baryon_density2 = 4.2*0.16
pressure1=30
pressure2=causality_p2(pressure1)
Preset_Pressure_final=1e-8
Preset_rtol=1e-8
pressure3=opt.newton(caulality_central_pressure_at_peak,trial_p3(pressure1,pressure2),tol=0.1,args=(pressure1,pressure2,Preset_Pressure_final,Preset_rtol))
upper_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
upper_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,upper_bound_eos[-1])[1])
print upper_bound_eos[-1].args
pressure1=8.4
pressure2,pressure3,pressure_center=p2p3_ofmaxmass(2.0,Maxmass,Preset_Pressure_final,Preset_rtol,pressure1)
lower_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
lower_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,lower_bound_eos[-1])[1])
print lower_bound_eos[-1].args

baryon_density1 = 1.7*0.16
baryon_density2 = 3.7*0.16
pressure1=23.179569511045678
pressure2=causality_p2(pressure1)
Preset_Pressure_final=1e-8
Preset_rtol=1e-8
pressure3=opt.newton(caulality_central_pressure_at_peak,trial_p3(pressure1,pressure2),tol=0.1,args=(pressure1,pressure2,Preset_Pressure_final,Preset_rtol))
upper_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
upper_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,upper_bound_eos[-1])[1])
print upper_bound_eos[-1].args
pressure1=6.9394800214143499
pressure2,pressure3,pressure_center=p2p3_ofmaxmass(2.0,Maxmass,Preset_Pressure_final,Preset_rtol,pressure1)
lower_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
lower_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,lower_bound_eos[-1])[1])
print lower_bound_eos[-1].args

baryon_density1 = 2.0*0.16
baryon_density2 = 3.7*0.16
pressure1=38.05392111496401
pressure2=causality_p2(pressure1)
Preset_Pressure_final=1e-8
Preset_rtol=1e-8
pressure3=opt.newton(caulality_central_pressure_at_peak,trial_p3(pressure1,pressure2),tol=0.1,args=(pressure1,pressure2,Preset_Pressure_final,Preset_rtol))
upper_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
upper_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,upper_bound_eos[-1])[1])
print upper_bound_eos[-1].args
pressure1=10.017537925119589
pressure2,pressure3,pressure_center=p2p3_ofmaxmass(2.0,Maxmass,Preset_Pressure_final,Preset_rtol,pressure1)
lower_bound_eos.append(EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3]))
lower_bound_pc.append(Maxmass(Preset_Pressure_final,1e-4,lower_bound_eos[-1])[1])
print lower_bound_eos[-1].args


for eos_i in lower_bound_eos:
    print eos_i.args
for eos_i in upper_bound_eos:
    print eos_i.args

#gamma1=2.2588784192626843
lower_bound_eos_args=[[0.059259259259259255, 8.4, 0.29600000000000004, 68.161110937142951, 0.512, 738.52598804194781, 1.1840000000000002],
[0.059259259259259255, 8.4, 0.29600000000000004, 103.56616205941582, 0.5920000000000001, 737.53456806740428, 1.1840000000000002],
[0.059259259259259255, 8.4, 0.29600000000000004, 149.49157528349323, 0.672, 737.79470441463366, 1.1840000000000002],
[0.059259259259259255, 6.93948002141435, 0.272, 103.57476787958687, 0.5920000000000001, 739.671922148676, 1.1840000000000002],
[0.059259259259259255, 10.017537925119589, 0.32, 103.60273144822639, 0.5920000000000001, 735.5549808413133, 1.1840000000000002]]
#gamma1=3.0503084533045279
upper_bound_eos_args=[[0.059259259259259255, 30, 0.29600000000000004, 218.54729186441736, 0.512, 2110.1402987649576, 1.1840000000000002],
[0.059259259259259255, 30, 0.29600000000000004, 298.98747443548757, 0.5920000000000001, 1949.035553443144, 1.1840000000000002],
[0.059259259259259255, 30, 0.29600000000000004, 386.9239253136754, 0.672, 1810.3314115401727, 1.1840000000000002],
[0.059259259259259255, 23.179569511045678, 0.272, 301.82393637730013, 0.5920000000000001, 1968.5642788279802, 1.1840000000000002],
[0.059259259259259255, 38.05392111496401, 0.32, 296.08342746016274, 0.5920000000000001, 1932.0239231893693, 1.1840000000000002]]



#lower_bound_eos=[]
#upper_bound_eos=[]
lower_bound_pc=[]
upper_bound_pc=[]
lower_cs2_pc_max=[]
upper_cs2_pc_max=[]
lower_maxmass=[]
upper_maxmass=[]
lower_cs2_p2=[]
upper_cs2_p2=[]
Preset_rtol=1e-4
for i in range(5):
    #lower_bound_eos.append(EOS_BPSwithPoly(lower_bound_eos_args[i]))
    #upper_bound_eos.append(EOS_BPSwithPoly(upper_bound_eos_args[i]))
    maxmass_result_lower=Maxmass(Preset_Pressure_final,Preset_rtol,lower_bound_eos[i])
    maxmass_result_upper=Maxmass(Preset_Pressure_final,Preset_rtol,upper_bound_eos[i])
    lower_bound_pc.append(maxmass_result_lower[1])
    upper_bound_pc.append(maxmass_result_upper[1])
    lower_maxmass.append(maxmass_result_lower[2])
    upper_maxmass.append(maxmass_result_upper[2])
    lower_cs2_p2.append(lower_bound_eos[i].eosCs2(0.99*lower_bound_eos[i].args[3]))
    upper_cs2_p2.append(upper_bound_eos[i].eosCs2(0.99*upper_bound_eos[i].args[3]))
    lower_cs2_pc_max.append(lower_bound_eos[i].eosCs2(lower_bound_pc[i]))
    upper_cs2_pc_max.append(upper_bound_eos[i].eosCs2(upper_bound_pc[i]))


from Parallel_process import main_parallel
import cPickle
pc_list=10**np.linspace(0,-1.5,20)
def Calculation_mass_beta_Lambda(args_list,i):
    eos=args_list[:,0]
    pc_max=args_list[:,1]
    mass=[]
    beta=[]
    Lambda=[]
    for j in range(len(pc_list)):
        MR_result=MassRadius(pc_max[i]*pc_list[j],Preset_Pressure_final,Preset_rtol,'MRBIT',eos[i])
        mass.append(MR_result[0])
        beta.append(MR_result[2])
        Lambda.append(MR_result[-1])
    return [mass,beta,Lambda]
f_mass_beta_Lambda_result='./out.dat'
main_parallel(Calculation_mass_beta_Lambda,np.array([upper_bound_eos,upper_bound_pc]).transpose(),f_mass_beta_Lambda_result,0)
f=open(f_mass_beta_Lambda_result,'rb')
mass_beta_Lambda_result=np.array(cPickle.load(f))
f.close()
mass_upper=mass_beta_Lambda_result[:,0]
beta_upper=mass_beta_Lambda_result[:,1]
Lambda_upper=mass_beta_Lambda_result[:,2]

f_mass_beta_Lambda_result='./out.dat'
main_parallel(Calculation_mass_beta_Lambda,np.array([lower_bound_eos,lower_bound_pc]).transpose(),f_mass_beta_Lambda_result,0)
f=open(f_mass_beta_Lambda_result,'rb')
mass_beta_Lambda_result=np.array(cPickle.load(f))
f.close()
mass_lower=mass_beta_Lambda_result[:,0]
beta_lower=mass_beta_Lambda_result[:,1]
Lambda_lower=mass_beta_Lambda_result[:,2]

eos_label=['n1=1.85ns,n2=3.2ns',	'n1=1.85ns,n2=3.7ns',	'n1=1.85ns,n2=4.2ns',	'n1=1.7ns,n2=3.7ns',	'n1=2.0ns,n2=3.7ns']
eos_color=['r','k','c','g','b']
for i in range(5):
    plt.plot(mass_upper[i],Lambda_upper[i],color=eos_color[i],label='upper bound '+eos_label[i])
    plt.plot(mass_lower[i],Lambda_lower[i],'--',color=eos_color[i],label='lower bound '+eos_label[i])
plt.legend()
plt.xlim(1.0,2.0)
plt.ylim(0,3000)
plt.xlabel('$M/M_\odot$')
plt.ylabel('$\Lambda$')

for i in range(5):
    plt.plot(mass_upper[i],beta_upper[i]**6*Lambda_upper[i],color=eos_color[i],label='upper bound '+eos_label[i])
    plt.plot(mass_lower[i],beta_lower[i]**6*Lambda_lower[i],'--',color=eos_color[i],label='lower bound '+eos_label[i])
plt.legend()
plt.xlim(1.0,2.0)
#plt.ylim(0,3000)
plt.xlabel('$M/M\odot$')
plt.ylabel('$\Lambda$')