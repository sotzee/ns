#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:46:31 2018

@author: sotzee
"""
from scipy.misc import derivative
from scipy.constants import c,G,e
from unitconvert import toMev4,toMevfm
import numpy as np
import matplotlib.pyplot as plt

def delta_shell(k_Fn,Lambda,kappa):
    delta=(Lambda**3/k_Fn**2+kappa*Lambda/9)
    return delta

def k_FQ(k_Fn,Lambda,kappa):
    delta=delta_shell(k_Fn,Lambda,kappa)
    k_FQ=(k_Fn-delta)/3
    return ((np.sign(k_FQ)+1)/2)*k_FQ

def n_B(k_Fn,Lambda,kappa):
    return (k_Fn**3-25.5*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)
    #return (k_Fn**3-24*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)

def Energy_density(k_Fn,n_s,a,b,Lambda,kappa,mass_args):
    m,m_u,m_d=mass_args
    k_FQ_=k_FQ(k_Fn,Lambda,kappa)
    return m**4*(chi(k_Fn/m)-chi(3*k_FQ_/m))+(3*m_u**4*chi(k_FQ_/(2.**(1./3)*m_u))+3*m_d**4*chi(k_FQ_/m_d))
    #return m**4*(chi(k_Fn/m)-chi(3*k_FQ_/m))+(3*m_u**4*chi(k_FQ_*(2.**(1./3)*m_u))+3*m_d**4*chi(k_FQ_/m_d))/2
def chi(x):
    return (x*(1+x**2)**0.5*(2*x**2+1)-np.log(x+(1+x**2)**0.5))/(8*np.pi**2)

def Pressure(k_Fn,n_s,a,b,Lambda,kappa,mass_args):
    det_Energy_density=Energy_density(k_Fn*1.001,n_s,a,b,Lambda,kappa,mass_args)-Energy_density(k_Fn*0.999,n_s,a,b,Lambda,kappa,mass_args)
    det_n_B=n_B(k_Fn*1.001,Lambda,kappa)-n_B(k_Fn*0.999,Lambda,kappa)
    return det_Energy_density/det_n_B*n_B(k_Fn,Lambda,kappa)-Energy_density(k_Fn,n_s,a,b,Lambda,kappa,mass_args)

def Cs2(k_Fn,n_s,a,b,Lambda,kappa,mass_args):
    det_Pressure=Pressure(k_Fn*1.001,n_s,a,b,Lambda,kappa,mass_args)-Pressure(k_Fn*0.999,n_s,a,b,Lambda,kappa,mass_args)
    det_Energy_density=Energy_density(k_Fn*1.001,n_s,a,b,Lambda,kappa,mass_args)-Energy_density(k_Fn*0.999,n_s,a,b,Lambda,kappa,mass_args)
    return det_Pressure/det_Energy_density


Lambda=380.
kappa=0.3
mass_args=(939.,313.,313.)

n_s=toMev4(0.16,'mevfm')
a=-28.8
b=10.
k_Fn=10**np.linspace(0,3.3,100)
k_FQ_array=k_FQ(k_Fn,Lambda,kappa)
n_B_array=toMevfm(n_B(k_Fn,Lambda,kappa),'mev4')
Energy_density_array=toMevfm(Energy_density(k_Fn,n_s,a,b,Lambda,kappa,mass_args),'mev4')
Pressure_array=toMevfm(Pressure(k_Fn,n_s,a,b,Lambda,kappa,mass_args),'mev4')
Cs2_array=Cs2(k_Fn,n_s,a,b,Lambda,kappa,mass_args)

plt.plot(n_B_array,Cs2_array)
plt.xlabel('n_B')
plt.ylabel('c$_s^2$')