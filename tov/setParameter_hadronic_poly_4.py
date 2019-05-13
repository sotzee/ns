#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:18:34 2018

@author: sotzee
"""
import numpy as np
from eos_class import EOS_BPS,EOS_BPSwithPoly_4,EOS_PiecewisePoly_4
import scipy.optimize as opt
from scipy.misc import derivative
from tov_f import MassRadius
#from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass

def Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i=np.log(pressure_i/pressure_i_minus)/np.log(baryon_density_i/baryon_density_i_minus)
    return gamma_i,(density_i_minus-pressure_i_minus/(gamma_i-1))*\
            (pressure_i/pressure_i_minus)**(1./gamma_i)+pressure_i/(gamma_i-1)

def causality_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i,density_i=Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus)
    return gamma_i*pressure_i/(density_i+pressure_i)-1

baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density15= 2.63*0.16
baryon_density2 = 3.74*0.16
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
#eos=EOS_BPSwithPoly([0.059259259259259255, 10 , 0.29600000000000004,100, 0.5984, 800, 1.1840000000000002])

print opt.newton(causality_i,100.,args=(baryon_density1,pressure0,baryon_density0,density0,))

def causality_central_pressure(pressure_center,density2,pressure2,gamma3):
    #print pressure_center,np.where(pressure_center>0,(gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)-1,-1+pressure_center/1000.)
    return np.where(pressure_center>0,(gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)-1,-1+pressure_center/1000.)

def caulality_central_pressure_at_peak(pressure3,pressure1,pressure15,pressure2,Preset_Pressure_final,Preset_rtol):
    eos = EOS_PiecewisePoly_4([density0,pressure0,baryon_density0,pressure1,baryon_density1,pressure15,baryon_density15,pressure2,baryon_density2,pressure3,baryon_density3])
    gamma3=eos.gamma4
    density2=eos.density3
    pressure_center=opt.newton(causality_central_pressure,pressure3,tol=0.1,args=(density2,pressure2,gamma3))    
    derivative_center_pressure=derivative(MassRadius,pressure_center,dx=1e-2,args=(Preset_Pressure_final,Preset_rtol,'M',eos))
    return derivative_center_pressure

def p2p3_ofmaxmass(ofmaxmass,Preset_p2,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1,p15):
    print '==================Finding p2 of maxmass%.2f at p1=%.2f, p15=%.2f'%(ofmaxmass,p1,p15)
    pressure3_result=[0]
    pressure_center_maxmass=[0]
    def Ofmaxmass(p2,ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,args_p1,args_p15):
        print '==================Finding p3 at p2=%f'%p2
        print args_p1,args_p15,p2,Preset_Pressure_final,Preset_rtol
        pressure3=opt.newton(caulality_central_pressure_at_peak,750.0,tol=0.1,args=(args_p1,args_p15,p2,Preset_Pressure_final,Preset_rtol))
        pressure3_result[0]=pressure3
        args=[baryon_density0,args_p1,baryon_density1,args_p15,baryon_density15,p2,baryon_density2,pressure3,baryon_density3]
        eos=EOS_BPSwithPoly_4(args)
        maxmass_result = Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)
        pressure_center_maxmass[0]=maxmass_result[1]
        print 'maxmass=%f'%maxmass_result[2]
        return -ofmaxmass+maxmass_result[2]
    result=opt.newton(Ofmaxmass,Preset_p2,tol=0.1,args=(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1,p15))
    return result,pressure3_result[0],pressure_center_maxmass[0]


