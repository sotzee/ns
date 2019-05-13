#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:43:21 2019

@author: sotzee
"""

import numpy as np
def get_lower_bound_E_L(m,ns,u1,e1,p1,uc):
    A=(e1+p1)/(2*ns*u1**2)
    B=(e1-p1)/(2*ns)
    C=-A*uc**2+B
    D=2*A*uc
    u2=1
    e2,p2=get_ep_lower_bound_E_L(u2,ns,A,B,C,D,uc)
    return e2/ns-m,3*p2/ns
def get_ep_lower_bound_E_L(u,ns,A,B,C,D,uc):
    e=ns*np.where(u<uc,A*u**2+B,C+D*u)  #cs2=1 u<uc
    p=ns*np.where(u<uc,A*u**2-B,-C)     #cs2=0 u>uc
    return e,p
def get_upper_bound_E_L(m,ns,u1,e1,p1,uc):
    C=-p1/ns
    D=(e1+p1)/(ns*u1)
    A=D/(2*uc)
    B=A*uc**2+C
    u2=1
    e2,p2=get_ep_upper_bound_E_L(u2,ns,A,B,C,D,uc)
    return e2/ns-m,3*p2/ns

def get_ep_upper_bound_E_L(u,ns,A,B,C,D,uc):
    e=ns*np.where(u<uc,C+D*u,A*u**2+B)  #cs2=0 u<uc
    p=ns*np.where(u<uc,-C,A*u**2-B)     #cs2=1 u>uc
    return e,p
# =============================================================================
# from eos_class import EOS_BPS
# m=939
# ns=0.16
# n1=0.06
# u1=n1/ns
# p1=EOS_BPS.eosPressure_frombaryon(n1)
# e1=EOS_BPS.eosDensity(p1)
# import matplotlib.pyplot as plt
# uc_lower=np.linspace(1.0*u1,1.02*u1,100)
# uc_upper=np.linspace(0.85,1,100)
# E_L_array_lower=[]
# E_L_array_upper=[]
# for uc_i in zip(uc_lower,uc_upper):
#     E_L_array_lower.append(get_lower_bound_E_L(m,ns,u1,e1,p1,uc_i[0]))
#     E_L_array_upper.append(get_upper_bound_E_L(m,ns,u1,e1,p1,uc_i[1]))
# E_L_array_lower=np.array(E_L_array_lower)
# E_L_array_upper=np.array(E_L_array_upper)
# plt.plot(E_L_array_lower[:,0],E_L_array_lower[:,1],label='stiff-soft lower bound')
# plt.plot(E_L_array_upper[:,0],E_L_array_upper[:,1],label='soft-stiff upper bound')
# plt.xlabel('$\\frac{\\varepsilon_2}{n_s}-m$ (MeV)',fontsize=20)
# plt.ylabel('$3\\frac{p_2}{n_s}$ (MeV)',fontsize=20)
# plt.legend(frameon=False,fontsize=14)
# =============================================================================
