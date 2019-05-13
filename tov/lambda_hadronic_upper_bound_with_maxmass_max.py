#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:23:52 2018

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
    return opt.newton(causality_i,200.,args=(baryon_density2,p1,baryon_density1,density1))


def p3_ofmaxmass(p1,ofmaxmass):
    def Ofmaxmass(p3):
        eos=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,causality_p2(p1),baryon_density2,p3,baryon_density3])
        maxmass=Maxmass(Preset_Pressure_final,Preset_rtol,eos)[2]
        print maxmass
        return maxmass-ofmaxmass
    return opt.newton(Ofmaxmass,3.*causality_p2(p1))

def p2_ofmaxmass(p1,ofmaxmass):
    def Ofmaxmass(p2):
        eos=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,2.2*p2,baryon_density3])
        print p2
        maxmass=MassRadius(p2,Preset_Pressure_final,Preset_rtol,'M',eos)
        print maxmass
        return maxmass-ofmaxmass
    return opt.newton(Ofmaxmass,200)

def p1_ofmaxmass(ofmaxmass):
    def Ofmaxmass(p1):
        p2=causality_p2(p1)
        eos=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,1000.,baryon_density3])
        maxmass=MassRadius(p2,Preset_Pressure_final,Preset_rtol,'M',eos)
        print maxmass
        return maxmass-ofmaxmass
    return opt.newton(Ofmaxmass,10)


p1=30.
p2=[298.98747443548757,280.42619530917631,246.62329988651169,217.41984574618286,191.78867337011866,169.14667443229015]
pc=[438.7646484375,280.42619530917631,246.62329988651169,217.41984574618286,191.78867337011866,169.14667443229015]
eos=[]
for i in range(len(p2)):
    eos.append(EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2[i],baryon_density2,932.73761329707713,baryon_density3]))

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
f_maxmass_result='./tmp/Lambda_hadronic_calculation_maxmass.dat'
main_parallel(Calculation_maxmass,eos,f_maxmass_result,0)
f=open(f_maxmass_result,'rb')
maxmass_result=cPickle.load(f)
f.close()

f_onepointfour_result='./tmp/Lambda_hadronic_calculation_onepointfour.dat'
main_parallel(Calculation_onepointfour,np.array([eos,maxmass_result]).transpose(),f_onepointfour_result,0)
f=open(f_onepointfour_result,'rb')
Properity_onepointfour=np.array(cPickle.load(f))
f.close()

pc_list=10**np.linspace(0,-1.5,50)
def Calculation_mass_beta_Lambda(args_list,i):
    eos=args_list[:,0]
    pc=args_list[:,1]
    mass=[]
    beta=[]
    Lambda=[]
    for j in range(len(pc_list)):
        MR_result=MassRadius(pc[i]*pc_list[j],Preset_Pressure_final,Preset_rtol,'MRT',eos[i])
        mass.append(MR_result[0])
        beta.append(MR_result[2])
        Lambda.append(MR_result[4])
    return [mass,beta,Lambda]
f_mass_beta_Lambda_result='./tmp/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
main_parallel(Calculation_mass_beta_Lambda,np.array([eos,pc]).transpose(),f_mass_beta_Lambda_result,0)
f=open(f_mass_beta_Lambda_result,'rb')
mass_beta_Lambda_result=np.array(cPickle.load(f))
f.close()
mass=mass_beta_Lambda_result[:,0]
beta=mass_beta_Lambda_result[:,1]
Lambda=mass_beta_Lambda_result[:,2]

M_min=1.1
def mass_chirp(mass1,mass2):
    return (mass1*mass2)**0.6/(mass1+mass2)**0.2
def tidal_binary(q,tidal1,tidal2):
    return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
def Calculation_chirpmass_Lambdabeta6(args_list,i):
    mass_beta_Lambda=list(args_list[:,0])
    beta_onepointfour=args_list[:,1]
    mass=np.array(mass_beta_Lambda)[:,0]
    Lambda=np.array(mass_beta_Lambda)[:,2]
    if(i==1):
        print mass_beta_Lambda[i]
        print beta_onepointfour[i]
        print mass[i]
        print Lambda[i]
    mass1,mass2 = np.meshgrid(mass[i][mass[i]>M_min],mass[i][mass[i]>M_min])
    if(i==1):
        print mass1
    #print np.min(mass1)
    Lambda1,Lambda2 = np.meshgrid(Lambda[i][mass[i]>M_min],Lambda[i][mass[i]>M_min])
    if(i==1):
        print Lambda1
    chirp_mass=mass_chirp(mass1,mass2).flatten()

    #print np.min(chirp_mass)
    Lambda_binary_beta6=(beta_onepointfour[i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2).flatten()
    if(i==1):
        print beta_onepointfour[i]
        print chirp_mass[i]
        print tidal_binary(mass2/mass1,Lambda1,Lambda2).flatten()
        print Lambda_binary_beta6
    return [chirp_mass,Lambda_binary_beta6]

f_chirpmass_Lambdabeta6_result='./tmp/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
main_parallel(Calculation_chirpmass_Lambdabeta6,np.array([list(mass_beta_Lambda_result),list(Properity_onepointfour[:,3])]).transpose(),f_chirpmass_Lambdabeta6_result,0)
f=open(f_chirpmass_Lambdabeta6_result,'rb')
chirpmass_Lambdabeta6_result=np.array(cPickle.load(f))
f.close()
chirp_mass=chirpmass_Lambdabeta6_result[:,0]
Lambda_binary_beta6=chirpmass_Lambdabeta6_result[:,1]