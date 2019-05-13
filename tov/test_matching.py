#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:22:54 2019

@author: sotzee
"""

from scipy.misc import derivative
import scipy.constants as const
from scipy.interpolate import interp1d
from unitconvert import toMev4#,toMevfm
import numpy as np
import scipy.optimize as opt
#import matplotlib.pyplot as plt

dlnx_cs2=1e-6

def energy_per_baryon_pnm(u,abcd_pnm):
    N=len(abcd_pnm)
    u_array=np.tile(u,(N,1)).transpose()**np.tile(np.linspace(0,(N-1),N)/3,(len(u),1))
    return np.dot(u_array,abcd_pnm)

def energy_per_baryon_pnm_jac(u,abcd_pnm):
    N=len(abcd_pnm)
    u_array=np.tile(u,(N,1)).transpose()**np.tile((np.linspace(0,(N-1),N))/3-1,(len(u),1))*np.tile(np.linspace(0,(N-1),N)/3,(len(u),1))
    return np.dot(u_array,abcd_pnm)

def get_parameters_tmp(parameter_array,ELKQ_array): #where E0,L0,K0,Q0 is for symmetric nuclear matter, and S,L,K,Q are for symmtry energy
    matrix=np.array([[120,-38,6,-1],[-270,90,-15,3],[216,-72,12,-3],[-60,20,-3,1]])
    #print(matrix,ELKQ_array,np.dot(matrix,ELKQ_array))
    return parameter_array+np.dot(matrix,ELKQ_array)/(6)

def get_parameters_pnm_around_vccume(m,T_223,ELKQ_array): #S,L,K,Q are for PNM(pure neutron matter).
    parameter_array=np.array([-4,6,-4,1])*T_223
    return np.array([m,0,T_223]+list(get_parameters_tmp(parameter_array,ELKQ_array)))

def get_parameters_pnm_margueron(m,T_223,mELKQZ_array): #S,L,K,Q are for PNM(pure neutron matter).
    m_eff,E,L,K,Q,Z=mELKQZ_array
    k=m/m_eff-1
    tmp=np.dot(np.array([[81,0,0,0,0],[-27,27,0,0,0],[9,-18,9,0,0],[-3,9,-9,3,0],[1,-4,6,-4,1]]).transpose()/81.,[0,(L-T_223*(2+5*k)),(K/2-T_223*(-1+5*k)),(Q/2-T_223*(4-5*k))/3,(Z/8-T_223*(-7+5*k))])
    return [m-T_223*(1+k)+E+tmp[0],0,T_223,tmp[1],0,T_223*k,tmp[2],0,0,tmp[3],0,0,tmp[4]]
    #m-(1+k)+E+   T*2**(2./3)*(u**(2./3))   +       T_223*(u**(5./3)*k  (L-T_223*(2+5*k))*x  (K/2-T_223*(-1+5*k))*x**2  (Q/2-(4-5*k))*x**3/3  (Z/8-T_223*(-7+5*k))*x**4/3)
    #return (a*u**2+(-a*2+b*3)*u+9*c+a-b*3+9*d*u**(-1./3)+9*e*u**(-4./3))/9

def get_baryon_density_u_max(abcd,defaut_u_max):
    N=len(abcd)
    coeff=(abcd*np.linspace(0,(N-1),N)/3*(np.linspace(0,(N-1),N)/3+1))[::-1]
    roots=np.roots(coeff)
    roots_real=roots.real[np.isreal(roots)]
    if(len(roots_real[roots_real>1.])==0):
        return defaut_u_max
    else:
        return np.min([roots_real[roots_real>1.].min()**3,defaut_u_max])

def get_eos_array(u_min,u_max,baryon_density_sat,m,T,abcd):
    baryon_density=baryon_density_sat*10**np.linspace(np.log10(u_min),np.log10(u_max),100)
    energy_dnnsity=np.concatenate((baryon_density*energy_per_baryon_pnm(baryon_density/baryon_density_sat,abcd),[10000]))
    pressure=np.concatenate((baryon_density**2/baryon_density_sat*energy_per_baryon_pnm_jac(baryon_density/baryon_density_sat,abcd),[10000]))
    baryon_density=np.concatenate((baryon_density,[1000*baryon_density_sat]))
    result=np.array([baryon_density,energy_dnnsity,pressure])
    #plt.plot(result[0],energy_per_baryon_pnm(baryon_density,baryon_density_sat,m,T,abcd))
    #plt.plot(result[0],result[1])
    #plt.plot(result[0][:-1],result[2][:-1])
    return result,[baryon_density_sat,baryon_density_sat*energy_per_baryon_pnm([1.],abcd)[0],baryon_density_sat*energy_per_baryon_pnm_jac([1.],abcd)[0]]

def match_eos(n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2,match_init):
    u1=n1/n_s
    u2=n2/n_s
    d=dpdn1
    b=p1/n_s-d*u1
    a=e1/n1+b/u1
    #c*(u2-u1)**gamma=e2/n2+b/u2-a-d*np.log(u2/u1)
    #c*gamma*(u2-u1)**(gamma)*u2**2=(p2/n_s-d*u2-b)*(u2-u1)
    gamma=(p2/n_s-d*u2-b)*(u2-u1)/((e2/n2+b/u2-a-d*np.log(u2/u1))*u2**2)
    c=(e2/n2+b/u2-a-d*np.log(u2/u1))/((u2/u1-1)**gamma)
    dpdn_match=(c*gamma*u2*(u2/u1-1)**(gamma-2))*(2*(u2-u1)+(gamma-1)*u2)/u1**2+d
    nep2_match=match_get_eos_array(u2,[a,b,c,gamma,d,u1,n_s])
    dedn2_match=(nep2_match[1]+nep2_match[2])/nep2_match[0]
    cs2_match=dpdn_match/dedn2_match
    return [a,b,c,gamma,d,u1,n_s], gamma>1 and cs2_match<1
def match_get_eos_array(u_array,args):
    a,b,c,gamma,d,u1,n_s=args
    e_array=(a-b/u_array+c*(u_array/u1-1)**gamma+d*np.log(u_array/u1))*n_s*u_array
    p_array=(b+c*gamma*(u_array/u1-1)**(gamma-1)*u_array**2/u1+d*u_array)*n_s
    return n_s*u_array,e_array,p_array

# =============================================================================
# def match_eos(n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2,match_init):
#     u1=n1/n_s
#     u2=n2/n_s
#     d=dpdn2
#     b=p2/n_s-d*u2
#     a=e2/n2+b/u2
#     #c*(u2-u1)**gamma=e2/n2+b/u2-a-d*np.log(u2/u1)
#     #c*gamma*(u2-u1)**(gamma)*u2**2=(p2/n_s-d*u2-b)*(u2-u1)
#     gamma=-(p1/n_s-d*u1-b)*(u2-u1)/((e1/n1+b/u1-a-d*np.log(u1/u2))*u1**2)
#     c=(e1/n1+b/u1-a-d*np.log(u1/u2))/((1-u1/u2)**gamma)
#     dpdn_match=(c*gamma*u1*(1-u1/u2)**(gamma-2))*(2*(u2-u1)-(gamma-1)*u1)/u2**2+d
#     nep1_match=match_get_eos_array(u1,[a,b,c,gamma,d,u1,n_s])
#     dedn1_match=(nep1_match[1]+nep1_match[2])/nep1_match[0]
#     cs2_match=dpdn_match/dedn1_match
#     return [a,b,c,gamma,d,u2,n_s], gamma>2 and cs2_match<1
# def match_get_eos_array(u_array,args):
#     a,b,c,gamma,d,u2,n_s=args
#     e_array=(a-b/u_array+c*(1-u_array/u2)**gamma+d*np.log(u_array/u2))*n_s*u_array
#     p_array=(b-c*gamma*(1-u_array/u2)**(gamma-1)*u_array**2/u2+d*u_array)*n_s
#     return n_s*u_array,e_array,p_array
# =============================================================================

# =============================================================================
# def match_eos(n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2,match_init):
#     a=p1/n_s
#     u1=n1/n_s
#     u2=n2/n_s
#     b=(p1+e1)/n1
#     gamma=(n2-n1)*(p2-p1)/(n2*(e2+p1)-n2**2*b)
#     d=((e2+p1)/n2-b)/(n2/n1-1)**gamma
#     dpdn_match=(d*gamma*u2*(u2/u1-1)**(gamma-2))*(2*(u2-u1)+(gamma-1)*u2)/u1**2
#     nep2_match=match_get_eos_array(u2,[a,b,d,gamma,u1,n_s])
#     dedn2_match=(nep2_match[1]+nep2_match[2])/nep2_match[0]
#     cs2_match=dpdn_match/dedn2_match
#     return [a,b,d,gamma,u1,n_s],gamma>2 and cs2_match<1
# def match_get_eos_array(u_array,args):
#     a,b,d,gamma,u1,n_s=args
#     e_array=n_s*u_array*(b-a/u_array+d*(u_array/u1-1)**gamma)
#     p_array=n_s*(a+d*gamma*u_array**2/u1*(u_array/u1-1)**(gamma-1))
#     return n_s*u_array,e_array,p_array
# =============================================================================

# =============================================================================
# def equations(x,args):
#     a,b,c,gamma=x
#     n_s,n1,p1,e1,n2,p2,e2=args
#     u1=n1/n_s
#     u2=n2/n_s
#     eq1=(a-b/u1+c*u1**gamma)*n1-e1
#     eq2=(a-b/u2+c*u2**gamma)*n2-e2
#     eq3=(b+c*gamma*u1**(gamma+1))*n_s-p1
#     eq4=(b+c*gamma*u2**(gamma+1))*n_s-p2
#     return eq1,eq2,eq3,eq4
# def match_eos(n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2):
#     return list(opt.root(equations,[955,3,3,5],tol=1e-8,args=[n_s,n1,p1,e1,n2,p2,e2]).x)+[n_s]
# def match_get_eos_array(u_array,args):
#     a,b,c,gamma,n_s=args
#     e_array=(a-b/u_array+c*u_array**gamma)*n_s*u_array
#     p_array=(b+c*gamma*u_array**(gamma+1))*n_s
#     return u_array,e_array,p_array
# =============================================================================

# =============================================================================
# def equations3(x,args):
#     a,b,c,d,gamma=x
#     n_s,n1,p1,e1,dpdn1,n2,p2,e2=args
#     u1=n1/n_s
#     u2=n2/n_s
#     eq1=(a-b/u1+c*u1**gamma+d*np.log(u1))*n1-e1
#     eq2=(a-b/u2+c*u2**gamma+d*np.log(u2))*n2-e2
#     eq3=(b+c*gamma*u1**(gamma+1)+d*u1)*n_s-p1
#     eq4=(b+c*gamma*u2**(gamma+1)+d*u2)*n_s-p2
#     eq5=(c*gamma*(gamma+1)*u1**(gamma)+d)-dpdn1
#     return eq1,eq2,eq3,eq4,eq5
# def match_eos(n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2,match_init):
#     try_solve=opt.root(equations3,match_init,tol=1e-8,args=[n_s,n1,p1,e1,dpdn1,n2,p2,e2])
#     if(try_solve.success):
#         EOS_EXPANSION_PNM.match_init=try_solve.x
#         print('matching success!!!')
#     else:
#         print('matching failed!!!')
#     return list(try_solve.x)+[n_s],try_solve.success
# def match_get_eos_array(u_array,args):
#     a,b,c,d,gamma,n_s=args
#     e_array=(a-b/u_array+c*u_array**gamma+d*np.log(u_array))*n_s*u_array
#     p_array=(b+c*gamma*u_array**(gamma+1)+d*u_array)*n_s
#     return n_s*u_array,e_array,p_array
# =============================================================================

from eos_class import EOS_BPS
class EOS_SLY4_match_EXPANSION_PNM(object):
    match_init=[955,3.1453328966256469,2.5839138055246758,0.1,5.2328896495712973]
    def __init__(self,args,PNM_EXPANSION_TYPE,defaut_u_min=1e-8,defaut_u_max=10):
        if(PNM_EXPANSION_TYPE=='around_vccume'):
            self.baryon_density_s,self.m,self.E_n,self.L_n,\
            self.K_n,self.Q_n=args
            self.args=args
            self.ELKQ_array=np.array(args[2:])
            get_parameters=get_parameters_pnm_around_vccume
        elif(PNM_EXPANSION_TYPE=='pnm_margueron'):
            self.baryon_density_s,self.m,self.m_eff,self.E_n,self.L_n,\
            self.K_n,self.Q_n,self.Z_n=args
            self.args=args
            self.ELKQ_array=np.array(args[2:])
            get_parameters=get_parameters_pnm_margueron
        self.n1_match=0.06
        self.n2_match=0.16
        self.T=.3*(1.5*np.pi**2*toMev4(self.baryon_density_s,'mevfm3'))**(2./3)/self.m
        self.abcd_array=get_parameters(self.m,self.T*2**(2./3),self.ELKQ_array)
        self.u_max=get_baryon_density_u_max(self.abcd_array,defaut_u_max)
        self.u_min=defaut_u_min
        eos_array_PNM,self.sol_saturation=get_eos_array(self.n2_match/self.baryon_density_s,self.u_max,self.baryon_density_s,self.m,self.T,self.abcd_array)
        p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
        e1=EOS_BPS.eosDensity(p1)
        dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
        p2=eos_array_PNM[2,0]
        e2=eos_array_PNM[1,0]
        dpdn2=EOS_BPS().eosCs2(p2)*EOS_BPS().eosChempo(p2)
        self.matching_args,self.matching_success=match_eos(self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init)
        if(self.matching_success):
            u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,100)
            eos_array_match=match_get_eos_array(u_array_low,self.matching_args)
        else:
            eos_array_match=[[self.n1_match,self.n2_match],[e1,e2],[p1,p2]]
        self.p_match=p1
        self.eos_array=np.concatenate((np.array([[0],[0],[0]]),np.array(eos_array_match)[:,:-1],eos_array_PNM),axis=1)
        self.pressure_s=self.sol_saturation[2]
        self.density_s=self.sol_saturation[1]
        self.unit_mass=const.c**4/(const.G**3*self.density_s*1e51*const.e)**0.5
        self.unit_radius=const.c**2/(const.G*self.density_s*1e51*const.e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        self.eosPressure_frombaryon_matchPNM = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosPressure_matchPNM = interp1d(self.eos_array[1],self.eos_array[2], kind='linear')
        self.eosDensity_matchPNM  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity_matchPNM = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon','eosPressure','eosDensity','eosBaryonDensity']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eosPressure_frombaryon = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosPressure = interp1d(self.eos_array[1],self.eos_array[2], kind='linear')
        self.eosDensity  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
    def eosPressure_frombaryon(self,baryon_density):
        return np.where(baryon_density<self.n1_match,EOS_BPS.eosPressure_frombaryon(baryon_density),self.eosPressure_frombaryon_matchPNM(baryon_density))
    def eosDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosDensity(pressure),self.eosDensity_matchPNM(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosBaryonDensity(pressure),self.eosBaryonDensity_matchPNM(pressure))
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

# =============================================================================
# a=EOS_EXPANSION_PNM([0.16,939.,32.-16,28,250,200],'around_vccume')
# print a.matching_args
# =============================================================================

n_s=0.16
m=939
N_E=91
N_L=150 #N_L=18
N_K=1
N_Q=1
N_Z=1
m_eff = 939*0.6
E_pnm = np.linspace(30-16,36.-16,N_E)
L_pnm = np.linspace(2,300,N_L)
K_pnm = np.linspace(200,200,N_K)#K_pnm = np.linspace(0,500,N_K)
Q_pnm = np.linspace(200,200,N_Q)#Q_pnm = np.linspace(-1000,1000,N_Q)
Z_pnm = np.linspace(-500,-500,N_Z)
Preset_Pressure_final=1e-8
Preset_rtol=1e-4
args=[]

eos_low=EOS_BPS()
eos_high=[]
eos =[]
eos_matching_success=[]
L_max=[]
L_min=[]
for i in range(len(E_pnm)):
    L_max_i=0
    L_min_i=10000
    for j in range(len(L_pnm)):
        for k in range(len(K_pnm)):
            for l in range(len(Q_pnm)):
                for s in range(len(Z_pnm)):
                    args.append([n_s,m,m_eff,E_pnm[i],L_pnm[j],K_pnm[k],Q_pnm[l],Z_pnm[s]])
                    eos.append(EOS_SLY4_match_EXPANSION_PNM(args[-1],'pnm_margueron'))
                    eos_matching_success.append(eos[-1].matching_success)
                    if(eos[-1].matching_success):
                        L_max_i=np.where(L_pnm[j]>L_max_i,L_pnm[j],L_max_i)
                        L_min_i=np.where(L_pnm[j]<L_min_i,L_pnm[j],L_min_i)
    L_max.append(L_max_i)
    L_min.append(L_min_i)
args=np.array(args).transpose().reshape((-1,N_E,N_L))
eos=np.array(eos).reshape((N_E,N_L))
eos_matching_success=np.array(eos_matching_success).reshape((N_E,N_L))
logic_labmda_over_2=[]
for i in range(len(eos)):
    for j in range(len(eos[i])):
        logic_labmda_over_2.append(eos[i,j].matching_args[3]>2)
logic_labmda_over_2=np.logical_and(np.array(logic_labmda_over_2).reshape((N_E,N_L)),eos_matching_success)
# =============================================================================
# for i in range(len(eos_matching_success.reshape((N_E,N_L)))):
#     L_pnm[eos_matching_success[i]].min()
#     L_pnm[eos_matching_success[i]].max()
#     matching_args
# =============================================================================

E_L_array_matching_lower=[]
E_L_array_matching_upper=[]
E_L_array_matching_lower_lambda_over_2=[]
for i in range(len(eos_matching_success)):
    E_L_array_matching_lower.append([E_pnm[i],L_pnm[eos_matching_success[i]].min()])
    E_L_array_matching_upper.append([E_pnm[i],L_pnm[eos_matching_success[i]].max()])
    E_L_array_matching_lower_lambda_over_2.append([E_pnm[i],L_pnm[logic_labmda_over_2[i]].min()])
E_L_array_matching_lower=np.array(E_L_array_matching_lower)
E_L_array_matching_upper=np.array(E_L_array_matching_upper)
E_L_array_matching_lower_lambda_over_2=np.array(E_L_array_matching_lower_lambda_over_2)

from test_matching_extreme import get_lower_bound_E_L,get_upper_bound_E_L
from eos_class import EOS_BPS
n1=0.06
u1=n1/n_s
p1=EOS_BPS.eosPressure_frombaryon(n1)
e1=EOS_BPS.eosDensity(p1)
uc_lower=np.linspace(1.0*u1,1.02*u1,100)
uc_upper=np.linspace(0.85,1,100)
E_L_array_lower=[]
E_L_array_upper=[]
for uc_i in zip(uc_lower,uc_upper):
    E_L_array_lower.append(get_lower_bound_E_L(m,n_s,u1,e1,p1,uc_i[0]))
    E_L_array_upper.append(get_upper_bound_E_L(m,n_s,u1,e1,p1,uc_i[1]))
E_L_array_lower=np.array(E_L_array_lower)
E_L_array_upper=np.array(E_L_array_upper)
np.savetxt('E_L_matching_bound.txt',np.array([E_L_array_matching_upper[:,0],E_L_array_matching_upper[:,1],E_L_array_matching_lower_lambda_over_2[:,1],E_L_array_matching_lower[:,1]]).transpose())
np.savetxt('E_L_absolute_bound.txt',np.array([E_L_array_upper[:,0],E_L_array_upper[:,1],E_L_array_lower[:,1]]).transpose())

dirctory='test_matching/'
import matplotlib.pyplot as plt
from plot_logic import plot_5D_logic
fig=plot_5D_logic(eos_matching_success,args[3:5],['J','L'],(0,1),figsize=(8,6))
plt.plot(E_L_array_upper[:,0],E_L_array_upper[:,1],label='MC upper bound',lw=3)
plt.plot(E_L_array_lower[:,0],E_L_array_lower[:,1],label='MP lower bound',lw=3)
plt.plot(E_L_array_matching_upper[:,0],E_L_array_matching_upper[:,1],label='Matching upper bound',lw=3)
plt.plot(E_L_array_matching_lower_lambda_over_2[:,0],E_L_array_matching_lower_lambda_over_2[:,1],label='Matching lower bound($\gamma>2$)',lw=3)
plt.plot(E_L_array_matching_lower[:,0],E_L_array_matching_lower[:,1],label='Matching lower bound($\gamma>1$)',lw=3)
plt.xlabel('$\\frac{\\varepsilon_2}{n_s}-m$ (MeV)',fontsize=20)
plt.ylabel('$3\\frac{p_2}{n_s}$ (MeV)',fontsize=20)
plt.xlim(11,20)
plt.ylim(0,300)
plt.legend(frameon=False,fontsize=14)
fig.savefig(dirctory+'success_matching_space')
# =============================================================================
# eos =[]
# for j in range(len(L_pnm)):
#     for k in range(len(K_pnm)):
#         for l in range(len(Q_pnm)):
#             args.append([n_s,m,E_pnm,L_pnm[j],K_pnm[k],Q_pnm[l]])
#             eos.append(EOS_EXPANSION_PNM(args[-1],'around_vccume'))
# =============================================================================


from show_properity import show_eos
fig, axes = plt.subplots(1, 1,sharex=True,sharey=True,figsize=(8,6))
show_eos(axes,eos[eos_matching_success][::103],0,3,200,pressure_range=[0.2,10,'linear'])
axes.plot(0.16*np.linspace(0.01,1.5,100),939+12.6*(np.linspace(0.01,1.5,100))**(2./3),label='UG limit')
axes.set_xlim(0.05,0.2)
axes.set_ylim(945,965)
plt.legend(frameon=False,fontsize=20)
fig.savefig(dirctory+'matching_eos_energy_per_baryon_spare')

fig, axes = plt.subplots(1, 1,sharex=True,sharey=True,figsize=(8,6))
show_eos(axes,eos[eos_matching_success][::103],0,1,200,pressure_range=[0.2,15,'linear'])
axes.plot(0.16*np.linspace(0.01,1.5,100),n_s*2./3.*12.6*(np.linspace(0.01,1.5,100))**(5./3),label='UG limit')
axes.set_xlim(0.05,0.2)
axes.set_ylim(0,15)
plt.legend(frameon=False,fontsize=20)
fig.savefig(dirctory+'matching_eos_pressure_spare')

fig, axes = plt.subplots(1, 1,sharex=True,sharey=True,figsize=(8,6))
show_eos(axes,eos[eos_matching_success][::103],0,5,200,pressure_range=[0.2,15,'linear'])
#axes.plot(0.16*np.linspace(0.01,1.5,100),n_s*2./3.*12.6*(np.linspace(0.01,1.5,100))**(5./3),label='UG limit')
axes.set_xlim(0.05,0.2)
#axes.set_ylim(0,15)
plt.legend(frameon=False,fontsize=20)
fig.savefig(dirctory+'matching_eos_cs2_spare')