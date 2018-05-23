#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:27:13 2018

@author: sotzee
"""
import numpy as np
from eos_class import EOS_BPS,EOS_BPSwithPoly,EOS_PiecewisePoly
from tov_f import MassRadius,Mass_formax
from Find_OfMass import Properity_ofmass
from FindMaxmass import Maxmass

import scipy.optimize as opt
from scipy.misc import derivative
baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.74*0.16
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
Preset_rtol=1e-4
Preset_Pressure_final=1e-8

# =============================================================================
# def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol):
#     result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol))
#     return result
# 
# def Ofmass(pressure_center,ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol):
#     pressure1=pressure1_max
#     gamma2=opt.newton(causality_trans,10,args=(pressure_center,))
#     pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
#     eos=EOS_BPSwithPoly([baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
#     return -ofmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'M',eos)
# 
# =============================================================================
def causality_central_pressure(pressure_center,density2,pressure2,gamma3):
    return (gamma3*pressure_center*(gamma3-1))/(((gamma3-1)*density2-pressure2)*(pressure_center/pressure2)**(1/gamma3)+gamma3*pressure_center)-1

def caulality_central_pressure_at_peak(pressure3,pressure1,pressure2,Preset_Pressure_final,Preset_rtol):
    eos = EOS_PiecewisePoly([density0,pressure0,baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3])
    print pressure1,pressure2
    pressure_center=opt.newton(causality_central_pressure,pressure3,tol=0.0005,args=(eos.density2,pressure2,eos.gamma3))
    return derivative(MassRadius,pressure_center,dx=1e-6,args=(Preset_Pressure_final,Preset_rtol,'M',eos))

def p2_ofmaxmass(ofmaxmass,Preset_p2,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1):
    def Ofmaxmass(p2,ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,args_p1):
        print '=================='
        print args_p1,p2
        print opt.newton(caulality_central_pressure_at_peak,1000.0,tol=0.001,args=(20.0,151.322998368,Preset_Pressure_final,Preset_rtol))
        pressure3=opt.newton(caulality_central_pressure_at_peak,1000.0,tol=0.001,args=(args_p1,p2,Preset_Pressure_final,Preset_rtol))
        args=[baryon_density0,args_p1,baryon_density1,p2,baryon_density2,pressure3,baryon_density3]
        eos=EOS_BPSwithPoly(args)
        maxmass_result = Maxmass_function(Preset_Pressure_final,Preset_rtol,eos)
        return -ofmaxmass+maxmass_result[2]
    result=opt.newton(Ofmaxmass,Preset_p2,tol=0.001,args=(ofmaxmass,Maxmass_function,Preset_Pressure_final,Preset_rtol,p1))
    return result

pressure3=opt.newton(caulality_central_pressure_at_peak,1000.0,tol=0.001,args=(20,150,Preset_Pressure_final,Preset_rtol))
eos_args=[baryon_density0,20.,baryon_density1,100.,baryon_density2,pressure3,baryon_density3]
p1=20.
print p2_ofmaxmass(2.2,150.,Maxmass,Preset_Pressure_final,Preset_rtol,p1)