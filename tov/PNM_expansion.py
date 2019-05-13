#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:17:36 2018

@author: sotzee
"""
from scipy.misc import derivative
import scipy.constants as const
from scipy.interpolate import interp1d
from unitconvert import toMev4#,toMevfm
import numpy as np
#import scipy.optimize as opt
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

def energy_per_baryon_pnm_jac_i(u,abcd_pnm,jac_n=1):
    N=len(abcd_pnm)
    u_array=np.tile(u,(N,1)).transpose()**np.tile((np.linspace(0,(N-1),N))/3-1,(len(u),1))
    for i in range(jac_n):
        u_array*=np.tile(np.linspace(0,(N-1),N)/3-i,(len(u),1))
    return np.dot(u_array,abcd_pnm)

def get_parameters_tmp(parameter_array,ELKQ_array): #where E0,L0,K0,Q0 is for symmetric nuclear matter, and S,L,K,Q are for symmtry energy
    matrix=np.array([[120,-38,6,-1],[-270,90,-15,3],[216,-72,12,-3],[-60,20,-3,1]])
    #print(matrix,ELKQ_array,np.dot(matrix,ELKQ_array))
    return parameter_array+np.dot(matrix,ELKQ_array)/6.

def get_parameters_pnm_around_vccume(m,T_223,ELKQ_array): #S,L,K,Q are for PNM(pure neutron matter).
    parameter_array=np.array([-4,6,-4,1])*T_223
    return np.array([m,0,T_223]+list(get_parameters_tmp(parameter_array,ELKQ_array)))

# =============================================================================
# def get_parameters_pnm_margueron(m,T_223,mELKQZ_array): #S,L,K,Q are for PNM(pure neutron matter).
#     m_eff,E,L,K,Q,Z=mELKQZ_array
#     k=1.*m/m_eff-1
#     tmp=np.dot(np.array([[81,0,0,0,0],[-27,27,0,0,0],[9,-18,9,0,0],[-3,9,-9,3,0],[1,-4,6,-4,1]]).transpose()/81.,[E-T_223*(1+k),(L-T_223*(2+5*k)),(K-2*T_223*(-1+5*k))/2.,(Q-2*T_223*(4-5*k))/6.,(Z-8*T_223*(-7+5*k))/24.])
#     return [m+tmp[0],0,T_223,tmp[1],0,T_223*k,tmp[2],0,0,tmp[3],0,0,tmp[4]]
# =============================================================================
def get_parameters_pnm_margueron(m,T_223,mELKQZ_array): #S,L,K,Q are for PNM(pure neutron matter).
    m_eff,E,L,K,Q,Z=mELKQZ_array
    k=1.*m/m_eff-1
    tmp=np.dot(np.array([[1944,0,0,0,0],[-648,648,0,0,0],[108,-216,108,0,0],[-12,36,-36,12,0],[1,-4,6,-4,1]]).transpose()/1944.,[E-T_223*(1+k),(L-T_223*(2+5*k)),(K-2*T_223*(-1+5*k)),(Q-2*T_223*(4-5*k)),(Z-8*T_223*(-7+5*k))])
    return [m+tmp[0],0,T_223,tmp[1],0,T_223*k,tmp[2],0,0,tmp[3],0,0,tmp[4]]

def get_baryon_density_u_max(abcd,defaut_u_max):
    N=len(abcd)
    coeff=(abcd*np.linspace(0,(N-1),N)/3*(np.linspace(0,(N-1),N)/3+1))[::-1]
    roots=np.roots(coeff)
    roots_real=roots.real[np.isreal(roots)]
    if(len(roots_real[roots_real>1.])==0):
        return defaut_u_max
    else:
        return np.min([roots_real[roots_real>1.].min()**3,defaut_u_max])

def get_eos_array(args):
    u_min,u_max,baryon_density_sat,m,T,abcd=args
    baryon_density=baryon_density_sat*10**np.linspace(np.log10(u_min),np.log10(u_max),200)
    energy_dnnsity=np.concatenate(([0],baryon_density*energy_per_baryon_pnm(baryon_density/baryon_density_sat,abcd),[10000]))
    pressure=np.concatenate(([0],baryon_density**2/baryon_density_sat*energy_per_baryon_pnm_jac(baryon_density/baryon_density_sat,abcd),[10000]))
    baryon_density=np.concatenate(([0],baryon_density,[1000*baryon_density_sat]))
    result=np.array([baryon_density,energy_dnnsity,pressure])
    #plt.plot(result[0],energy_per_baryon_pnm(baryon_density,baryon_density_sat,m,T,abcd))
    #plt.plot(result[0],result[1])
    #plt.plot(result[0][:-1],result[2][:-1])
    return result,[baryon_density_sat,baryon_density_sat*energy_per_baryon_pnm([1.],abcd)[0],baryon_density_sat*energy_per_baryon_pnm_jac([1.],abcd)[0]]

def match_eos(args):
    n_s,n1,p1,e1,dpdn1,n2,p2,e2,dpdn2,match_init=args
    u1=n1/n_s
    u2=n2/n_s
    d=dpdn1
    b=p1/n_s-d*u1
    a=e1/n1+b/u1
    gamma=(p2/n_s-d*u2-b)*(u2-u1)/((e2/n2+b/u2-a-d*np.log(u2/u1))*u2**2)
    c=(e2/n2+b/u2-a-d*np.log(u2/u1))/((u2-u1)**gamma)
    dpdn_match=(c*gamma*u2*(u2-u1)**(gamma-2))*(2*(u2-u1)+(gamma-1)*u2)+d
    nep2_match=match_get_eos_array(u2,[a,b,c,gamma,d,u1,n_s])
    dedn2_match=(nep2_match[1]+nep2_match[2])/nep2_match[0]
    cs2_match=dpdn_match/dedn2_match
    return [a,b,c,gamma,d,u1,n_s],gamma>1 and cs2_match<1
def match_get_eos_array(u_array,para):
    a,b,c,gamma,dpdn1,u1,n_s=para
    d=dpdn1
    e_array=(a-b/u_array+c*(u_array-u1)**gamma+d*np.log(u_array/u1))*n_s*u_array
    p_array=(b+c*gamma*(u_array-u1)**(gamma-1)*u_array**2+d*u_array)*n_s
    return np.array([n_s*u_array,e_array,p_array])

from eos_class import EOS_BPS
class EOS_SLY4_match_EXPANSION_PNM(object):
    match_init=[None,None] #not in use
    def __init__(self,args,PNM_EXPANSION_TYPE,defaut_u_max=12):
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
        self.get_eos_array_args=self.n2_match/self.baryon_density_s,self.u_max,self.baryon_density_s,self.m,self.T,self.abcd_array
        eos_array_PNM,self.sol_saturation=get_eos_array(self.get_eos_array_args)
        self.n_max,self.e_max,self.p_max=eos_array_PNM[:,-2]
        self.high_stability_success=self.p_max>100.
        self.cs2=1
        self.args_eosCSS=[self.e_max,self.p_max,self.n_max,self.cs2]
        self.eosCSS=EOS_CSS(self.args_eosCSS)
        p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
        e1=EOS_BPS.eosDensity(p1)
        dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
        p2=eos_array_PNM[2,1]
        e2=eos_array_PNM[1,1]
        dpdn2=dpdn1#this parameter is not used in match_eos, so it was trivially set to dpdn1
        self.matching_args=self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init
        self.matching_para,self.matching_success=match_eos(self.matching_args)
        self.eos_success=self.high_stability_success and self.matching_success
        if(self.matching_success):
            u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,81)
            eos_array_match=match_get_eos_array(u_array_low,self.matching_para)
        else:
            eos_array_match=np.array([[self.n1_match,self.n2_match],[e1,e2],[p1,p2]])
            eos_array_PNM=np.array([[self.n2_match,10*self.n2_match],[e2,10*e2],[p2,10*p2]])
        self.p_match=p1
        self.eos_array=np.concatenate((np.array([[0],[0],[0]]),eos_array_match[:,:-1],eos_array_PNM[:,1:]),axis=1)
        self.pressure_s=self.sol_saturation[2]
        self.density_s=self.sol_saturation[1]
        self.unit_mass=const.c**4/(const.G**3*self.density_s*1e51*const.e)**0.5
        self.unit_radius=const.c**2/(const.G*self.density_s*1e51*const.e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        self.eosPressure_frombaryon_matchPNM = interp1d(self.eos_array[0],self.eos_array[2], kind='quadratic')
        self.eosDensity_matchPNM  = interp1d(self.eos_array[2],self.eos_array[1], kind='quadratic')
        self.eosBaryonDensity_matchPNM = interp1d(self.eos_array[2],self.eos_array[0], kind='quadratic')
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon_matchPNM','eosDensity_matchPNM','eosBaryonDensity_matchPNM']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eosPressure_frombaryon_matchPNM = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosDensity_matchPNM  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity_matchPNM = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
    def eosPressure_frombaryon(self,baryon_density):
        return np.where(baryon_density<self.n1_match,EOS_BPS.eosPressure_frombaryon(baryon_density),np.where(baryon_density<self.n_max,self.eosPressure_frombaryon_matchPNM(baryon_density),self.eosCSS.eosPressure_frombaryon(baryon_density)))
    def eosDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosDensity(pressure),np.where(pressure<self.p_max,self.eosDensity_matchPNM(pressure),self.eosCSS.eosDensity(pressure)))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosBaryonDensity(pressure),np.where(pressure<self.p_max,self.eosBaryonDensity_matchPNM(pressure),self.eosCSS.eosBaryonDensity(pressure)))
    def eosCs2(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS().eosCs2(pressure),np.where(pressure<self.p_max,1.0/derivative(self.eosDensity_matchPNM,pressure,dx=pressure*dlnx_cs2),self.eosCSS.eosCs2(pressure)))
    def eosChempo(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS().eosChempo(pressure),np.where(pressure<self.p_max,(pressure+self.eosDensity_matchPNM(pressure))/self.eosBaryonDensity_matchPNM(pressure),self.eosCSS.eosChempo(pressure)))
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and self.mass_max>2
        self.eos_success_all=self.maxmass_success and self.eos_success
        return self.eos_success_all

class EOS_CSS(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryondensity_trans,self.cs2 = args
        self.B=(self.density0-self.pressure0/self.cs2)/(1.0+1.0/self.cs2)
    def eosPressure_frombaryon(self,baryon_density):
        return (baryon_density/self.baryondensity_trans)**(1.0+self.cs2)*(self.pressure0+self.B)-self.B
    def eosDensity(self,pressure):
        density = (pressure-self.pressure0)/self.cs2+self.density0
        return np.where(density>0,density,0)
    def eosBaryonDensity(self,pressure):
        tmp=(pressure+self.B)/(self.pressure0+self.B)
        baryondensity = np.where(tmp>0,self.baryondensity_trans*np.abs(tmp)**(1.0/(1.0+self.cs2)),0)
        return np.where(baryondensity>0,baryondensity,0)
    def eosCs2(self,pressure):
        return self.cs2
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

def Calculation_creat_eos_SLY4_match_EXPANSION_PNM(args,PNM_EXPANSION_TYPE):
    return EOS_SLY4_match_EXPANSION_PNM(args,PNM_EXPANSION_TYPE)

# =============================================================================
# def matching_eos(trial_pressure,eos_density1,eos_density2):
#     return eos_density1(trial_pressure)-eos_density2(trial_pressure)
# 
# def calculate_matching_pressure(trial_pressure,Preset_tol,eos_density1,eos_density2):
#     p_matching=opt.newton(matching_eos,trial_pressure,tol=Preset_tol,args=(eos_density1,eos_density2))
#     return p_matching
# 
# class EOS_EXPANSION_PNM(object):
#     def __init__(self,args,PNM_EXPANSION_TYPE,defaut_u_min=1e-8,defaut_u_max=12):
#         if(PNM_EXPANSION_TYPE=='around_vccume'):
#             self.baryon_density_s,self.m,self.E_n,self.L_n,\
#             self.K_n,self.Q_n=args
#             self.args=args
#             self.ELKQ_array=np.array(args[2:])
#             get_parameters=get_parameters_pnm_around_vccume
#         elif(PNM_EXPANSION_TYPE=='pnm_margueron'):
#             self.baryon_density_s,self.m,self.m_eff,self.E_n,self.L_n,\
#             self.K_n,self.Q_n,self.Z_n=args
#             self.args=args
#             self.ELKQ_array=np.array(args[2:])
#             get_parameters=get_parameters_pnm_margueron
#         self.T=.3*(1.5*np.pi**2*toMev4(self.baryon_density_s,'mevfm3'))**(2./3)/self.m
#         self.abcd_array=get_parameters(self.m,self.T*2**(2./3),self.ELKQ_array)
#         self.u_max=get_baryon_density_u_max(self.abcd_array,defaut_u_max)
#         self.u_min=defaut_u_min
#         self.eos_array,self.sol_saturation=get_eos_array(self.u_min,self.u_max,self.baryon_density_s,self.m,self.T,self.abcd_array)
#         self.pressure_s=self.sol_saturation[2]
#         self.density_s=self.sol_saturation[1]
#         self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#         self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#         self.eosPressure_frombaryon = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosPressure = interp1d(self.eos_array[1],self.eos_array[2], kind='linear')
#         self.eosDensity  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         for dict_intepolation in ['eosPressure_frombaryon','eosPressure','eosDensity','eosBaryonDensity']:
#             del state[dict_intepolation]
#         return state
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.eosPressure_frombaryon = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosPressure = interp1d(self.eos_array[1],self.eos_array[2], kind='linear')
#         self.eosDensity  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# class EOS_CSS(object):
#     def __init__(self,args):
#         self.density0,self.pressure0,self.baryondensity_trans,self.cs2 = args
#         self.B=(self.density0-self.pressure0/self.cs2)/(1.0+1.0/self.cs2)
#     def eosDensity(self,pressure):
#         density = (pressure-self.pressure0)/self.cs2+self.density0
#         return np.where(density>0,density,0)
#     def eosBaryonDensity(self,pressure):
#         
#         tmp=(pressure+self.B)/(self.pressure0+self.B)
#         baryondensity = np.where(tmp>0,self.baryondensity_trans*np.abs(tmp)**(1.0/(1.0+self.cs2)),0)
#         return np.where(baryondensity>0,baryondensity,0)
#     def eosCs2(self,pressure):
#         return self.cs2
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
#     
# class EOS_PnmCSS(object):
#     def __init__(self,args,PNM_EXPANSION_TYPE,cs2=1):
#         self.eosPNM=EOS_EXPANSION_PNM(args,PNM_EXPANSION_TYPE)
#         self.sol_saturation=self.eosPNM.sol_saturation
#         self.baryon_density_s=self.eosPNM.baryon_density_s
#         self.pressure_s=self.eosPNM.pressure_s
#         self.density_s=self.eosPNM.density_s
#         self.unit_mass=self.eosPNM.unit_mass
#         self.unit_radius=self.eosPNM.unit_radius
#         self.unit_N=self.eosPNM.unit_N
#         self.baryondensity_trans=self.eosPNM.u_max*self.eosPNM.baryon_density_s*0.9999999
#         self.pressure_trans=self.eosPNM.eosPressure_frombaryon(self.baryondensity_trans)
#         self.density_trans=self.eosPNM.eosDensity(self.pressure_trans)
#         self.cs2=cs2
#         args_eosCSS=[self.density_trans,self.pressure_trans\
#                      ,self.baryondensity_trans,self.cs2]
#         self.eosCSS=EOS_CSS(args_eosCSS)
#     def __getstate__(self):
#         state_PNM=self.eosPNM.__getstate__()
#         state = self.__dict__.copy()
#         return (state,state_PNM)
#     def __setstate__(self, state_):
#         state,state_PNM=state_
#         self.__dict__.update(state)
#         self.eosPNM.__setstate__(state_PNM)
# 
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosPNM.eosDensity(pressure),self.eosCSS.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosPNM.eosBaryonDensity(pressure),self.eosCSS.eosBaryonDensity(pressure))
#     def eosCs2(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosPNM.eosCs2(pressure),self.cs2)
#     def eosChempo(self,pressure):
#         return np.where(pressure<self.pressure_trans,self.eosPNM.eosChempo(pressure),self.eosCSS.eosChempo(pressure))
# =============================================================================

# =============================================================================
# Preset_tol_matching=1e-4
# class EOS_Sly4_match_PnmCSS(object):
#     def __init__(self,eos_low,eos_high):
#         self.eos_low=eos_low
#         self.eos_high=eos_high
#         flag=True
#         for trial_pressure in [0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1,1,0.01,10,0.001,100]:
#             if(flag==True):
#                 flag=False
#                 try:
#                     self.p_match=calculate_matching_pressure(trial_pressure,Preset_tol_matching,eos_low.eosDensity,eos_high.eosDensity)
#                 except:
#                     flag=True
#             else:
#                 break
#         if(flag):
#             if(eos_high.eosPNM.u_max<2):
#                 self.p_match=0
#             else:
#                 #print('Matching of low density EoS %s and hight density %s failed'%(self.eos_low,self.eos_high))
#                 print self.eos_high.eosPNM.args
#         if(self.p_match>100):
#             print('matching at exceptional high pressure, p_match=%f'%(self.p_match))
#         self.baryon_density_s=self.eos_high.baryon_density_s
#         self.pressure_s=self.eos_high.pressure_s
#         self.density_s=self.eos_high.density_s
#         self.unit_mass=self.eos_high.unit_mass
#         self.unit_radius=self.eos_high.unit_radius
#         self.unit_N=self.eos_high.unit_N
#     def __getstate__(self):
#         state_high=self.eos_high.__getstate__()
#         state = self.__dict__.copy()
#         return (state,state_high)
#     def __setstate__(self, state_):
#         state,state_high=state_
#         self.__dict__.update(state)
#         self.eos_high.__setstate__(state_high)
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.p_match,self.eos_low.eosDensity(pressure),self.eos_high.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.p_match,self.eos_low.eosBaryonDensity(pressure),self.eos_high.eosBaryonDensity(pressure))
#     def eosCs2(self,pressure):
#         return np.where(pressure<self.p_match,self.eos_low.eosCs2(pressure),self.eos_high.eosCs2(pressure))
#     def eosChempo(self,pressure):
#         return np.where(pressure<self.p_match,self.eos_low.eosChempo(pressure),self.eos_high.eosChempo(pressure))
# 
# from eos_class import EOS_BPS,EOS_BPSwithPoly
# class EOS_SLY4POLYwithPNM(object):
#     def __init__(self,args,PNM_EXPANSION_TYPE):
#         self.args=args
#         self.eos_PNM=EOS_PnmCSS(args,PNM_EXPANSION_TYPE)
#         self.sol_saturation=self.eos_PNM.sol_saturation
#         self.baryon_density_s=self.eos_PNM.baryon_density_s
#         self.pressure_s=self.eos_PNM.pressure_s
#         self.density_s=self.eos_PNM.density_s
#         self.unit_mass=self.eos_PNM.unit_mass
#         self.unit_radius=self.eos_PNM.unit_radius
#         self.unit_N=self.eos_PNM.unit_N
#         fix_crust_baryon_density=np.linspace(0.6,0.1,6)*self.sol_saturation[0]
#         self.fix_crust_logic=False
#         for fix_crust_baryon_density_i in fix_crust_baryon_density:
#             if(self.sol_saturation[2]>1.1*EOS_BPS.eosPressure_frombaryon(fix_crust_baryon_density_i)):
#                 self.eos_SLY4withPoly=EOS_BPSwithPoly([fix_crust_baryon_density_i,self.sol_saturation[2],self.sol_saturation[0],4*self.sol_saturation[2],2*self.sol_saturation[0],8*self.sol_saturation[2],3*self.sol_saturation[0]])
#                 self.fix_crust_logic=True
#                 break
#     def __getstate__(self):
#         state_PNM=self.eos_PNM.__getstate__()
#         state = self.__dict__.copy()
#         return (state,state_PNM)
#     def __setstate__(self, state_):
#         state,state_PNM=state_
#         self.__dict__.update(state)
#         self.eos_PNM.__setstate__(state_PNM)
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosDensity(pressure),self.eos_PNM.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosBaryonDensity(pressure),self.eos_PNM.eosBaryonDensity(pressure))
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#     def eosCs2(self,pressure):
#         return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosCs2(pressure),self.eos_PNM.eosCs2(pressure))
#     def eosChempo(self,pressure):
#         return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosChempo(pressure),self.eos_PNM.eosChempo(pressure))
# =============================================================================

import cPickle
import os
path = "./"
dir_name='Lambda_PNM_margueron_calculation_parallel'
#dir_name='Lambda_PNM_around_vacuum_calculation_parallel'
error_log=path+dir_name+'/error.log'
if __name__ == '__main__':
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)

    n_s=0.16
    m=939
    N_m=3
    N_E=3
    N_L=26
    N_K=17
    N_Q=17
    N_Z=19
    m_eff = np.linspace(0.6,1.0,N_m)*m
    E_pnm = np.linspace(30.-16,34.-16,N_E)
    L_pnm = np.linspace(20,120,N_L)
    K_pnm = np.linspace(0,400,N_K)
    Q_pnm = np.linspace(-1000,1000,N_Q)
    Z_pnm = np.linspace(-5000,-500,N_Z)
    args=[]
    for t in range(len(m_eff)):
        for i in range(len(E_pnm)):
            for j in range(len(L_pnm)):
                for k in range(len(K_pnm)):
                    for l in range(len(Q_pnm)):
                        for s in range(len(Z_pnm)):
                            args.append([n_s,m,m_eff[t],E_pnm[i],L_pnm[j],K_pnm[k],Q_pnm[l],Z_pnm[s]])
    args_flat=np.array(args)
    args=np.reshape(args_flat,(N_m,N_E,N_L,N_K,N_Q,N_Z,-1))

    from Parallel_process import main_parallel_unsave
    eos_flat=main_parallel_unsave(Calculation_creat_eos_SLY4_match_EXPANSION_PNM,args_flat,other_args='pnm_margueron')
    eos=eos_flat.reshape((N_m,N_E,N_L,N_K,N_Q,N_Z))
    eos_success=[]
    for eos_i in eos_flat:
        eos_success.append(eos_i.eos_success)
    eos_success=np.array(eos_success)

# =============================================================================
#     n_s=0.16
#     m=939
#     N_m=5
#     N_E=5
#     N_L=26 #N_L=18
#     N_K=31
#     N_Q=37
#     #N_Z=17
#     m_eff = np.linspace(0.6,1.0,N_m)*939
#     E_pnm = np.linspace(30.-16,34.-16,N_E)
#     L_pnm = np.linspace(20,120,N_L)
#     K_pnm = np.linspace(50,500,N_K)#K_pnm = np.linspace(0,500,N_K)
#     Q_pnm = np.linspace(-1000,800,N_Q)#Q_pnm = np.linspace(-1000,1000,N_Q)
#     #Z_pnm = np.linspace(-2500,-500,N_Z)
#     Preset_Pressure_final=1e-8
#     Preset_rtol=1e-4
#     args=[]
#     for i in range(len(E_pnm)):
#         for j in range(len(L_pnm)):
#             for k in range(len(K_pnm)):
#                 for l in range(len(Q_pnm)):
#                     args.append([n_s,m,E_pnm[i],L_pnm[j],K_pnm[k],Q_pnm[l]])
#                     eos.append(EOS_SLY4_match_EXPANSION_PNM(args[-1],'around_vccume'))
#                     eos_success.append(eos[-1].eos_success)
#     args_flat=np.array(args)
#     args=np.reshape(args_flat,(N_E,N_L,N_K,N_Q,-1))
#     
#     from Parallel_process import main_parallel_unsave
#     eos_flat=main_parallel_unsave(Calculation_creat_eos_SLY4_match_EXPANSION_PNM,args_flat,other_args='around_vccume')
#     eos=eos_flat.reshape((N_E,N_L,N_K,N_Q))
#     eos_success=[]
#     for eos_i in eos_flat:
#         eos_success.append(eos_flat.eos_success)
#     eos_success=np.array(eos_success)
# =============================================================================

    
    f_file=open(path+dir_name+'/Lambda_hadronic_calculation_args.dat','wb')
    cPickle.dump(args,f_file)
    f_file.close()
    
# =============================================================================
#     f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos.dat','wb')
#     cPickle.dump(eos,f_file)
#     f_file.close()
# =============================================================================
    

    print('%d EoS built with shape (E_n,L_n,K_n,Q_n)%s.'%(len(args_flat),np.shape(args)[:-1]))
    print('%d EoS are successful.'%(len(eos_success[eos_success])))
    
    from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
    from Parallel_process import main_parallel
    
    f_maxmass_result=path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
    maxmass_result=np.full((len(eos_flat),3),np.array([0,0,1]),dtype='float')
    maxmass_result[eos_success]=main_parallel(Calculation_maxmass,eos_flat[eos_success],f_maxmass_result,error_log)
    
    print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat)))
    logic_maxmass=maxmass_result[:,1]>=2
    print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat[eos_success]),len(eos_flat[logic_maxmass])))
    logic_causality=maxmass_result[:,2]<1
    print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat[eos_success]),len(eos_flat[logic_causality])))
    logic=np.logical_and(logic_maxmass,logic_causality)
    print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat[eos_success]),len(eos_flat[logic])))
    
    eos_success_maxmass=np.logical_and(logic,eos_success)
    for i in range(len(eos_flat)):
        if(eos_success[i]):
            eos_flat[i].setMaxmass(maxmass_result[i])
        else:
            eos_flat[i].setMaxmass([0,0,1])

    f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_success.dat','wb')
    cPickle.dump(eos_success,f_file)
    f_file.close()

    f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_success_maxmass.dat','wb')
    cPickle.dump(eos_success_maxmass,f_file)
    f_file.close()

    f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat','wb')
    cPickle.dump(eos_flat[eos_success_maxmass],f_file)
    f_file.close()
    
    print('Calculating properities of 1.4 M_sun star...')
    f_onepointfour_result=path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat'
    Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat[eos_success_maxmass],f_onepointfour_result,error_log)
    print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat[eos_success_maxmass])))
    
    print('Calculating mass, compactness and tidal Lambda...')
    f_mass_beta_Lambda_result=path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
    mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat[eos_success_maxmass],f_mass_beta_Lambda_result,error_log)
    print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat[eos_success_maxmass])))
    
    print('Calculating binary neutron star...')
    f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
    chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)

    

# =============================================================================
# import cPickle
# import os
# path = "./"
# #dir_name='Lambda_PNM_margueron_calculation_parallel'
# dir_name='Lambda_PNM_around_vacuum_calculation_parallel'
# error_log=path+dir_name+'/error.log'
# if __name__ == '__main__':
#     try:
#         os.stat(path+dir_name)
#     except:
#         os.mkdir(path+dir_name)
#     n_s=0.16
#     m=939
#     N_m=5
#     N_L=31 #N_L=18
#     N_K=36
#     N_Q=51
#     #N_Z=17
#     m_eff = np.linspace(0.6,1.0,N_m)*939
#     E_pnm = 32.-16
#     L_pnm = np.linspace(10,70,N_L)
#     K_pnm = np.linspace(50,400,N_K)#K_pnm = np.linspace(0,500,N_K)
#     Q_pnm = np.linspace(-400,600,N_Q)#Q_pnm = np.linspace(-1000,1000,N_Q)
#     #Z_pnm = np.linspace(-2500,-500,N_Z)
#     Preset_Pressure_final=1e-8
#     Preset_rtol=1e-4
#     args=[]
#     
# # =============================================================================
# #     eos_low=EOS_BPS()
# #     eos_high=[]
# # =============================================================================
# # =============================================================================
# #     eos =[]
# #     for i in range(len(m_eff)):
# #         for j in range(len(L_pnm)):
# #             for k in range(len(K_pnm)):
# #                 for l in range(len(Q_pnm)):
# #                     for s in range(len(Z_pnm)):
# #                         args.append([n_s,m,m_eff[i],E_pnm,L_pnm[j],K_pnm[k],Q_pnm[l],Z_pnm[s]])
# #                         eos.append(EOS_SLY4POLYwithPNM(args[-1],'pnm_margueron'))
# #     args=np.reshape(np.array(args),(N_m,N_L,N_K,N_Q,N_Z,-1))
# #     args_flat=np.reshape(np.array(args),(N_m*N_L*N_K*N_Q*N_Z,-1))
# #     eos =np.reshape(np.array(eos),(N_m,N_L,N_K,N_Q,N_Z))
# # =============================================================================
#     eos =[]
#     for j in range(len(L_pnm)):
#         for k in range(len(K_pnm)):
#             for l in range(len(Q_pnm)):
#                 args.append([n_s,m,E_pnm,L_pnm[j],K_pnm[k],Q_pnm[l]])
#                 eos.append(EOS_SLY4POLYwithPNM(args[-1],'around_vccume'))
#                         
#     args=np.reshape(np.array(args),(N_L,N_K,N_Q,-1))
#     args_flat=np.reshape(np.array(args),(N_L*N_K*N_Q,-1))
#     eos =np.reshape(np.array(eos),(N_L,N_K,N_Q))
#     eos_flat=np.array(eos).flatten()
#     f_file=open(path+dir_name+'/Lambda_PNM_calculation_args.dat','wb')
#     cPickle.dump(args,f_file)
#     f_file.close()
#     f_file=open(path+dir_name+'/Lambda_PNM_calculation_eos.dat','wb')
#     cPickle.dump(eos,f_file)
#     f_file.close()
#     print('%d EoS built with shape (L_n,K_n,Q_n)%s.'%(len(args_flat),np.shape(eos)))
#     
#     from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
#     from Parallel_process import main_parallel
#     
#     f_maxmass_result=path+dir_name+'/Lambda_PNM_calculation_maxmass.dat'
#     maxmass_result=main_parallel(Calculation_maxmass,eos_flat,f_maxmass_result,error_log)
#     print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat)))
#     logic_maxmass=maxmass_result[:,1]>=2
#     print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic_maxmass])))
#     logic_causality=maxmass_result[:,2]<1
#     print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic_causality])))
#     logic=np.logical_and(logic_maxmass,logic_causality)
#     print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic])))
# 
#     for i in range(len(eos_flat)):
#         eos_flat[i].setMaxmass(maxmass_result[i])
#         
#     f_file=open(path+dir_name+'/Lambda_PNM_calculation_eos_flat_logic.dat','wb')
#     cPickle.dump(eos_flat[logic],f_file)
#     f_file.close()
#     
#     f_onepointfour_result=path+dir_name+'/Lambda_PNM_calculation_onepointfour.dat'
#     Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat[logic],f_onepointfour_result,error_log)
#     print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat[logic])))
#     
#     f_mass_beta_Lambda_result=path+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat'
#     mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat[logic],f_mass_beta_Lambda_result,error_log)
#     print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat[logic])))
# 
#     f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
#     chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)
# =============================================================================


# =============================================================================
# tcheck=np.loadtxt('../../Downloads/tcheck.dat')
# tcheck_u=tcheck[0:22].flatten()
# tcheck_p=tcheck[22:44].flatten()
# tcheck_e=tcheck[44:66].flatten()
# tcheck_cs2=tcheck[66:88].flatten()
# #LOOK AT SOME EOS
# import matplotlib.pyplot as plt
# a=EOS_EXPANSION_PNM([0.16,939.,32.-16,70.,300.,-250.],'around_vccume')
# # =============================================================================
# # b1=EOS_EXPANSION_PNM([0.16,939.,939.,32.-16,50.,240.,100.,0],'pnm_margueron')
# # b2=EOS_EXPANSION_PNM([0.16,939.,0.8*939.,32.-16,50.,240.,100.,0],'pnm_margueron')
# # b3=EOS_EXPANSION_PNM([0.16,939.,939.,32.-16,50.,240.,100.,500.],'pnm_margueron')
# # =============================================================================
# plt.figure()
# plt.title('E=32-16, L=50, K=240, Q=100')
# plt.plot(a.eos_array[0,:-1],a.eos_array[2,:-1],label='around_vacuum')
# plt.plot(0.16*tcheck_u,tcheck_p)
# # =============================================================================
# # plt.plot(b1.eos_array[0,:-1],b1.eos_array[2,:-1],label='pnm_margueron, $m_e/m=1, Z=0$')
# # plt.plot(b2.eos_array[0,:-1],b2.eos_array[2,:-1],label='pnm_margueron, $m_e/m=0.8, Z=0$')
# # plt.plot(b3.eos_array[0,:-1],b3.eos_array[2,:-1],label='pnm_margueron, $m_e/m=1, Z=500$')
# # =============================================================================
# plt.legend()
# plt.xlim(0,0.6)
# plt.ylim(0.01,1000)
# plt.yscale('log')
# plt.xlabel('baryon density(fm$^{-3}$)')
# plt.ylabel('pressure(MeV fm$^{-3}$)')
# 
# from eos_class import EOS_BPS
# a_match=EOS_Sly4_match_PnmCSS(EOS_BPS(),EOS_PnmCSS([0.16,939.,32.-16,50.,240.,100.],'around_vccume'))
# b1_match=EOS_Sly4_match_PnmCSS(EOS_BPS(),EOS_PnmCSS([0.16,939.,939.,32.-16,50.,240.,100.,0],'pnm_margueron'))
# b2_match=EOS_Sly4_match_PnmCSS(EOS_BPS(),EOS_PnmCSS([0.16,939.,0.8*939.,32.-16,50.,240.,100.,0],'pnm_margueron'))
# b3_match=EOS_Sly4_match_PnmCSS(EOS_BPS(),EOS_PnmCSS([0.16,939.,939.,32.-16,50.,240.,100.,500.],'pnm_margueron'))
# def show_eos(eos,x_index,y_index,N,baryon_density_range=False,pressure_range=False):#index baryon_density(0), pressure(1), energy density(2), energy per baryon(3), chempo(4)
#     pressure_density_energyPerBaryon_chempo=[]
#     if(baryon_density_range):
#         for eos_i in eos:
#             baryon_density_i=np.linspace(baryon_density_range[0],baryon_density_range[1],N)
#             pressure_density_energyPerBaryon_chempo_i=[]
#             pressure_density_energyPerBaryon_chempo_i.append(baryon_density_i)
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosPressure_frombaryon(baryon_density_i))
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_density_energyPerBaryon_chempo_i[1]))
#             pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/baryon_density_i)
#             pressure_density_energyPerBaryon_chempo_i.append((pressure_density_energyPerBaryon_chempo_i[1]+pressure_density_energyPerBaryon_chempo_i[2])/baryon_density_i)
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosCs2(pressure_density_energyPerBaryon_chempo_i[1]))
#             pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
#             plt.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index])
#     elif(pressure_range):
#         for eos_i in eos:
#             pressure_i=np.linspace(pressure_range[0],pressure_range[1],N)
#             pressure_density_energyPerBaryon_chempo_i=[]
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosBaryonDensity(pressure_i))
#             pressure_density_energyPerBaryon_chempo_i.append(pressure_i)
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_i))
#             pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/pressure_density_energyPerBaryon_chempo_i[0])
#             pressure_density_energyPerBaryon_chempo_i.append((pressure_i+pressure_density_energyPerBaryon_chempo_i[2])//pressure_density_energyPerBaryon_chempo_i[0])
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosCs2(pressure_i))
#             pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
#             plt.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index])
#     pressure_density_energyPerBaryon_chempo=np.array(pressure_density_energyPerBaryon_chempo)
#     label_text=['Baryon density(fm$^{-3}$)','Pressure(MeV fm$^{-3}$)','Energy density(MeV fm$^{-3}$)','Energy per baryon(MeV)','Chemical potential(MeV)','$c_s^2/c^2$']
#     plt.xlabel(label_text[x_index])
#     plt.ylabel(label_text[y_index])
# plt.figure()
# show_eos([a_match,b1_match,b2_match,b3_match],0,1,100,pressure_range=[0.1,1000])
# plt.xlim(0,0.6)
# plt.ylim(0.1,1000)
# plt.yscale('log')
# plt.xlabel('baryon density(fm$^{-3}$)')
# plt.ylabel('pressure(MeV fm$^{-3}$)')
# =============================================================================



# =============================================================================
# def energy_per_baryon_pnm_margueron(n,n_s,m,T,kELKQZ_pnm):
#     u=n/n_s
#     x=(u-1)/3
#     k,E,L,K,Q,Z=kELKQZ_pnm
#     T_223=T*2**(2./3)
#     tmp=np.dot(np.array([[81,0,0,0,0],[-27,27,0,0,0],[9,-18,9,0,0],[-3,9,-9,3,0],[1,-4,6,-4,1]]).transpose(),[0,(L-T_223*(2+5*k)),(K/2-T_223*(-1+5*k)),(Q/2-T_223*(4-5*k))/3,(Z/8-T_223*(-7+5*k))]
#     return [m-T_223*(1+k)+E+tmp[0],0,T_223,tmp[1],0,T_223*k,tmp[2],0,0,tmp[3],0,0,tmp[4]]
#     print m-T_223*(1+k)+E+  T_223*(u**(2./3))   +       T_223*u**(5./3)*k  +(L-T_223*(2+5*k))*x  +(K/2-T_223*(-1+5*k))*x**2  +(Q/2-T_223*(4-5*k))*x**3/3  +(Z/8-T_223*(-7+5*k))*x**4/3
#     print m-T_223*(1+k)+E+  T_223*(u**(2./3))   +       T_223*u**(5./3)*k   +(L-T_223*(2+5*k))*x  +(K/2-T_223*(-1+5*k))*x**2  +(Q/2-T_223*(4-5*k))*x**3/3  +(Z/8-T_223*(-7+5*k))*x**4/3
#     print m-T_223*(1+k)+E ,m-T_223*(1+k)+E
#     print T*2**(2./3)*(u**(2./3)*(1+k*u)) , T_223*(u**(2./3))   +     T_223*u**(5./3)*k 
#     print L*x           -T_223*(2+5*k)*x        , (L-T_223*(2+5*k))*x
#     print K*x**2/2      -T_223*(-1+5*k)*x**2    , (K/2-T_223*(-1+5*k))*x**2
#     print Q*x**3/6      -T_223*(4-5*k)*x**3/3   , (Q/2-T_223*(4-5*k))*x**3/3
#     print Z*x**4/24     -T_223*(-7+5*k)*x**4/3  , (Z/8-T_223*(-7+5*k))*x**4/3
#     
#     return m+T*2**(2./3)*(u**(2./3)*(1+k*u)-(1+k)-(2+5*k)*x-(-1+5*k)*x**2-(4-5*k)*x**3/3-(-7+5*k)*x**4/3)   +E+L*x+K*x**2/2+Q*x**3/6+Z*x**4/24
# 
# def energy_per_baryon_pnm_jac_margueron(n,n_s,T,kELKQZ_pnm):
#     u=n/n_s
#     x=(u-1)/3
#     k,E,L,K,Q,Z=kELKQZ_pnm
#     return T*2**(2./3)*(u**(-1./3)*(2+5*k*u)/3-(2+5*k)/3-(-1+5*k)*x*2./3-(4-5*k)*x**2/3-(-7+5*k)*x**3*4./9)   +L/3+K*x/3+Q*x**2/6+Z*x**3/18
# 
# def energy_per_baryon_pnm_jac_jac_margueron(n,n_s,T,kELKQZ_pnm):
#     u=n/n_s
#     x=(u-1)/3
#     k,E,L,K,Q,Z=kELKQZ_pnm
#     #print T*2**(2./3)*(u**(-4./3)*(-1+5*k*u)*2/9-(-1+5*k)*2./9-(4-5*k)*x*2/9-(-7+5*k)*x**2*4./9)   +K/9+Q*x/9+Z*x**2/18
#     #print (Z/18-T*2**(2./3)*(-7+5*k)*4./9)*x**2+(Q/9-T*2**(2./3)*(4-5*k)*2/9)*x+K/9-T*2**(2./3)*(-1+5*k)*2./9+T*2**(2./3)*(u**(-4./3)*(-1+5*k*u)*2/9)
#     #print (Z/18-T*2**(2./3)*(-7+5*k)*4./9)*u**2/9+(-(Z/18-T*2**(2./3)*(-7+5*k)*4./9)*2/9+(Q/9-T*2**(2./3)*(4-5*k)*2/9)/3)*u+((Z/18-T*2**(2./3)*(-7+5*k)*4./9)/9-(Q/9-T*2**(2./3)*(4-5*k)*2/9)/3+K/9-T*2**(2./3)*(-1+5*k)*2./9)+T*2**(2./3)*(10*k/9)*u**(-1./3)-T*2**(2./3)*2/9*u**(-4./3)
#     T_223=T*2**(2./3)
#     a=(Z-T_223*(-7+5*k)*8.)/18
#     b=(Q-T_223*(4-5*k)*2)/9
#     c=(K-T_223*(-1+5*k)*2.)/9
#     d=T_223*(10*k/9)
#     e=-T_223*2/9
#     #print (a*u**2+(-a*2+b*3)*u+9*c+a-b*3+9*d*u**(-1./3)+9*e*u**(-4./3))/9
#     return a*x**2+b*x+c+d*u**(-1./3)+e*u**(-4./3)
# 
# def get_parameters_pnm_margueron(T,kELKQZ_array): #S,L,K,Q are for PNM(pure neutron matter).
#     k,E,L,K,Q,Z=kELKQZ_array
#     T_223=T*2**(2./3)
#     a=(Z-T_223*(-7+5*k)*8.)/18
#     b=(Q-T_223*(4-5*k)*2)/9
#     c=(K-T_223*(-1+5*k)*2.)/9
#     d=T_223*(10*k/9)
#     e=-T_223*2/9
#     return (a*u**2+(-a*2+b*3)*u+9*c+a-b*3+9*d*u**(-1./3)+9*e*u**(-4./3))/9
# =============================================================================
