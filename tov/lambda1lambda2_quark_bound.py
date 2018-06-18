#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:12:30 2018

@author: sotzee
"""

from eos_class import EOS_CSS
from tov_CSS import Mass_CSS_formax
from Find_OfMass import Properity_ofmass
import numpy as np
import scipy.optimize as opt

cs2_lower_bound=0.2
cs2_upper_bound=0.4

def Maxmass(eos):
    result=opt.minimize(Mass_CSS_formax,100.0,tol=0.001,args=(eos),method='Nelder-Mead')
    return [0,result.x[0],-result.fun,result.x[0],-result.fun,result.x[0],-result.fun]

def density_surface_ofmaxmass(ofmaxmass,density_surface_low,density_surface_high,Maxmass_function,cs2):
    def Ofmaxmass(density_surface,ofmaxmass,Maxmass_function):
        eos=EOS_CSS([density_surface,0.,1.,cs2])
        return -ofmaxmass+Maxmass_function(eos)[2]
    result=opt.brenth(Ofmaxmass,density_surface_low,density_surface_high,args=(ofmaxmass,Maxmass_function))
    return result

density_surface_ofmaxmass(2.0,50.,500.,Maxmass,0.3)

