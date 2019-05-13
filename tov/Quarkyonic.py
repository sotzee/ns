#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:40:54 2018

@author: sotzee
"""

from scipy.misc import derivative
import scipy.constants as const
from scipy.interpolate import interp1d
from unitconvert import toMev4,toMevfm
import numpy as np
import scipy.optimize as opt
import pickle as cPickle

factor=3 #k_FQ=(k_Fn-Delta)/3
#import matplotlib.pyplot as plt

# =============================================================================
# def phi(x):
#     return (x*(1+x**2)**0.5*(2*x**2-3)+3*np.log(x+(1+x**2)**0.5))/(24*np.pi**2)
# =============================================================================

def chi(x):
    return (x*(1+x**2)**0.5*(2*x**2+1)-np.log(x+(1+x**2)**0.5))/(8*np.pi**2)

def k_FQ(k_Fn,Lambda,kappa):
    return (np.sign(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3)+1)*((9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3)/(9*2*factor*k_Fn**2))

# =============================================================================
# def k_FQ_jac(k_Fn,Lambda,kappa):
#     print 'bec'
#     print (np.sign(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3)+1)
#     print ((27*k_Fn**2-2*kappa*Lambda*k_Fn)*(54*k_Fn**2)-108*k_Fn*(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3))
#     print (54*k_Fn**2)**2
#     print (np.sign(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3)+1)*((27*k_Fn**2-2*kappa*Lambda*k_Fn)*(54*k_Fn**2)-108*k_Fn*(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3))/(54*k_Fn**2)**2
#     return (np.sign(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3)+1)*((27*k_Fn**2-2*kappa*Lambda*k_Fn)*(54*k_Fn**2)-108*k_Fn*(9*k_Fn**3-kappa*Lambda*k_Fn**2-9*Lambda**3))/(54*k_Fn**2)**2
# =============================================================================

def n_B(k_Fn,Lambda,kappa):
    #return (k_Fn**3-25.5*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)
    return (k_Fn**3-(factor**3-1.5)*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)

# =============================================================================
# def n_B_jac(k_Fn,Lambda,kappa):
#     return (3*k_Fn**2-72*k_FQ(k_Fn,Lambda,kappa)**2*k_FQ_jac(k_Fn,Lambda,kappa))/(3*np.pi**2)
# =============================================================================

def n_B_for_newton(k_Fn,n_B,Lambda,kappa):
    #return (k_Fn**3-25.5*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)-n_B
    return (k_Fn**3-(factor**3-1.5)*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)-n_B

def n_n(k_Fn,Lambda,kappa):
    return (k_Fn**3-(factor**3)*k_FQ(k_Fn,Lambda,kappa)**3)/(3*np.pi**2)

# =============================================================================
# def n_n_jac(k_Fn,Lambda,kappa):
#     #print 'abc'
#     #print k_FQ_jac(k_Fn,Lambda,kappa)
#     #print (3*k_Fn**2-81*k_FQ(k_Fn,Lambda,kappa)**2*k_FQ_jac(k_Fn,Lambda,kappa))/(3*np.pi**2)
#     return (3*k_Fn**2-81*k_FQ(k_Fn,Lambda,kappa)**2*k_FQ_jac(k_Fn,Lambda,kappa))/(3*np.pi**2)
# =============================================================================


# =============================================================================
# def neutron_mean_potential(u,a,b):
#     return a*u+b*u**2
# 
# def neutron_mean_potential_jac(u,a,b):
#     return a+2*b*u
# =============================================================================

def Energy_density(k_Fn,n_s,a,b,Lambda,kappa,mass_args):
    m,m_u,m_d=mass_args
    k_FQ_=k_FQ(k_Fn,Lambda,kappa)
    u_n=n_n(k_Fn,Lambda,kappa)/n_s
    V_n=a*u_n+b*u_n**2
    return m**4*(chi(k_Fn/m)-chi(factor*k_FQ_/m))+(3*m_u**4*chi(k_FQ_/(2.**(1./3)*m_u))+3*m_d**4*chi(k_FQ_/m_d))+u_n*n_s*V_n

def Pressure(k_Fn,n_s,a,b,Lambda,kappa,mass_args):
    det_Energy_density=Energy_density(k_Fn*1.0001,n_s,a,b,Lambda,kappa,mass_args)-Energy_density(k_Fn*0.9999,n_s,a,b,Lambda,kappa,mass_args)
    det_n_B=n_B(k_Fn*1.0001,Lambda,kappa)-n_B(k_Fn*0.9999,Lambda,kappa)
    return det_Energy_density/det_n_B*n_B(k_Fn,Lambda,kappa)-Energy_density(k_Fn,n_s,a,b,Lambda,kappa,mass_args)

# =============================================================================
# def Pressure(k_Fn,n_s,a,b,Lambda,kappa,mass_args):
#     m,m_u,m_d=mass_args
#     k_FQ_=k_FQ(k_Fn,Lambda,kappa)
#     #n_n_=n_n(k_Fn,Lambda,kappa)
#     #n_B_=n_B(k_Fn,Lambda,kappa)
#     #u=n_n_/n_s
#     return m**4*(phi(k_Fn/m)-phi(3*k_FQ_/m))+3*m_u**4*phi(2.**(1./3)*k_FQ_/m_u)+3*m_d**4*phi(k_FQ_/m_d)#+n_n_*u*neutron_mean_potential_jac(u,a,b)#+(n_B_*neutron_mean_potential(u,a,b)+n_n_*n_B_*neutron_mean_potential_jac(u,a,b))*n_n_jac(k_Fn,Lambda,kappa)/n_B_jac(k_Fn,Lambda,kappa)-n_n_*neutron_mean_potential(u,a,b)
# =============================================================================

def get_eos_array(u_min,u_max,eos_args):
    n_s,a,b,Lambda,kappa,mass_args=eos_args
    n_s_mev4=toMev4(n_s,'mevfm')
    k_Fn_max=opt.newton(n_B_for_newton,1000.,args=(n_s_mev4*u_max,Lambda,kappa))
    k_Fn_min=(3*np.pi**2*n_s_mev4*u_min)**(1./3)
    k_Fn_array=k_Fn_min*np.exp(np.linspace(0,np.log(k_Fn_max/k_Fn_min),500))
    baryon_density=np.concatenate(([0],n_B(k_Fn_array,Lambda,kappa)))
    energy_density=np.concatenate(([0],Energy_density(k_Fn_array,n_s_mev4,a,b,Lambda,kappa,mass_args)))
    pressure=np.concatenate(([0],Pressure(k_Fn_array,n_s_mev4,a,b,Lambda,kappa,mass_args)))
    eos_array=np.array([baryon_density,energy_density,pressure])
    k_Fs=(3*np.pi**2*n_s_mev4)**(1./3)
    sol_saturation=np.array([n_B(k_Fs,Lambda,kappa),Energy_density(k_Fs,n_s_mev4,a,b,Lambda,kappa,mass_args),Pressure(k_Fs,n_s_mev4,a,b,Lambda,kappa,mass_args)])
    stability=np.logical_and(eos_array[1,1:]-eos_array[1,:-1]>0,eos_array[2,1:]-eos_array[2,:-1]>0)
    causality=eos_array[1,1:]-eos_array[1,:-1]>eos_array[2,1:]-eos_array[2,:-1]
    return toMevfm(eos_array,'mev4'),toMevfm(sol_saturation,'mev4'),bool(stability.min() and causality.min())

def get_ab(n_s_mev4,E,L,mass_args):
    k_Fs=(3*np.pi**2*n_s_mev4)**(1./3)
    m,m_u,m_d=mass_args
    a=2.*E-L/3.-3*m**4/n_s_mev4*chi(k_Fs/m)+np.sqrt(k_Fs**2+m**2)
    b=-E+L/3.+2*m**4/n_s_mev4*chi(k_Fs/m)-np.sqrt(k_Fs**2+m**2)
    return a,b

dlnx_cs2=1e-6
from eos_class import EOS_BPS,match_eos,match_get_eos_array
class EOS_Quarkyonic(object):
    match_init=[None,None] #not in use
    def __init__(self,args,defaut_u_min=1e-8,defaut_u_max=20):
        self.baryon_density_s,self.E,self.L,self.Lambda,self.kappa,self.mass_args=args
        self.a,self.b=get_ab(toMev4(self.baryon_density_s,'mevfm'),self.E,self.L,self.mass_args)
        self.args=self.baryon_density_s,self.a,self.b,self.Lambda,self.kappa,self.mass_args
        self.u_max=defaut_u_max
        self.u_min=1
        eos_array_quarkyonic,self.sol_saturation,self.causality_stability=get_eos_array(self.u_min,self.u_max,self.args)
        self.pressure_s=self.sol_saturation[2]
        self.density_s=self.sol_saturation[1]
        
        self.n1_match=0.06
        self.n2_match=self.baryon_density_s
        p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
        e1=EOS_BPS.eosDensity(p1)
        dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
        p2=self.pressure_s
        e2=self.density_s
        dpdn2=dpdn1#this parameter is not used in match_eos, so it was trivially set to dpdn1
        self.p_match=p1
        self.matching_args=self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init
        self.matching_para,self.matching_success=match_eos(self.matching_args)
        self.eos_success=self.matching_success and self.causality_stability
        if(self.matching_success):
            u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,81)
            eos_array_match=match_get_eos_array(u_array_low,self.matching_para)
        else:
            eos_array_match=[[self.n1_match,self.n2_match],[e1,e2],[p1,p2]]
        self.eos_array=np.concatenate((np.array([[0],[0],[0]]),np.array(eos_array_match)[:,:-1],eos_array_quarkyonic),axis=1)
        self.unit_mass=const.c**4/(const.G**3*self.density_s*1e51*const.e)**0.5
        self.unit_radius=const.c**2/(const.G*self.density_s*1e51*const.e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        self.eosPressure_frombaryon_match_Quarkyonic = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosDensity_match_Quarkyonic  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity_match_Quarkyonic = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')

    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon_match_Quarkyonic','eosDensity_match_Quarkyonic','eosBaryonDensity_match_Quarkyonic']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eosPressure_frombaryon_match_Quarkyonic = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosDensity_match_Quarkyonic  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity_match_Quarkyonic = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
    def eosPressure_frombaryon(self,baryon_density):
        return np.where(baryon_density<self.n1_match,EOS_BPS.eosPressure_frombaryon(baryon_density),self.eosPressure_frombaryon_match_Quarkyonic(baryon_density))
    def eosDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosDensity(pressure),self.eosDensity_match_Quarkyonic(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosBaryonDensity(pressure),self.eosBaryonDensity_match_Quarkyonic(pressure))
    def eosCs2(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS().eosCs2(pressure),1.0/derivative(self.eosDensity_match_Quarkyonic,pressure,dx=pressure*dlnx_cs2))
    def eosChempo(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS().eosChempo(pressure),(pressure+self.eosDensity_match_Quarkyonic(pressure))/self.eosBaryonDensity_match_Quarkyonic(pressure))
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and self.mass_max>2
        self.eos_success_all=self.maxmass_success and self.eos_success
        return self.eos_success_all

def Calculation_creat_eos_Quarkyonic(eos_args_args_array,mass_args):
    return EOS_Quarkyonic(list(eos_args_args_array)+[mass_args])

# =============================================================================
# aa=EOS_Quarkyonic([0.16,968.,66,380,0.3,(939,313,313)])
# bb=EOS_Quarkyonic([0.16,955.,40.,380,0.3,(939,313,313)])
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# import show_properity
# show_properity.show_eos(axes,[aa,bb],0,5,500,baryon_density_range=[0.01,3,'log'])
# =============================================================================

# =============================================================================
# aa=EOS_Quarkyonic([0.16,-8.988391755337943e-07  , 8.988392892206321e-07,380,0.3,(939,313,313)])
# bb=EOS_Quarkyonic([0.16,-28.600001855905134     , 9.900001855905316,    380, 0.3,(939,313,313)])
# =============================================================================

# =============================================================================
# aa=EOS_Quarkyonic([0.16, 973.3454881322934, 67.2679569190962, 380,0.3,(939,313,313)])
# bb=EOS_Quarkyonic([0.16, 954.6454881322935, 40.86795979029441,380,0.3,(939,313,313)])
# 
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# import show_properity
# show_properity.show_eos(axes,[aa,bb],0,5,500,baryon_density_range=[0.01,3,'log'])
# =============================================================================

import os
path = "./"
dir_name='Lambda_Quarkyonic_calculation_parallel'
error_log=path+dir_name+'/error.log'
if __name__ == '__main__':
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)

    show_plot=False
    
    baryon_density_s=0.16
    m=939
    BindE=16
    mass_args=(939,313,313)
    #args=np.mgrid[(m-BindE+30):(m-BindE+36):5j,0:220:23j,340:440:21j,0.0:1.5:31j]
    args=np.mgrid[(m-BindE+30):(m-BindE+36):3j,0:220:11j,340:440:6j,0.0:1.5:6j]
    args_flat=args.reshape((-1,np.prod(np.shape(args)[1:]))).transpose()
    f_file=open(path+dir_name+'/Quarkyonic_args.dat','wb')
    cPickle.dump(args,f_file)
    f_file.close()
    J,L,Lambda,kappa=args
    args_shape=np.shape(J)

    f_eos_RMF='./'+dir_name+'/Quarkyonic_eos.dat'
    from Parallel_process import main_parallel_unsave
    eos_flat=main_parallel_unsave(Calculation_creat_eos_Quarkyonic,np.concatenate((np.full((len(args_flat),1),baryon_density_s),args_flat),axis=1),other_args=mass_args)
    eos=eos_flat.reshape(args_shape)

    logic_causality_stability=[]
    eos_success=[]
    for i in range(len(args_flat)):
        logic_causality_stability.append(eos_flat[i].causality_stability)
        eos_success.append(eos_flat[i].eos_success)
    logic_causality_stability=np.array(logic_causality_stability).reshape(args_shape)
    eos_success=np.array(eos_success).reshape(args_shape)
    
    f_file=open(path+dir_name+'/Quarkyonic_eos_logic.dat','wb')
    cPickle.dump(logic_causality_stability,f_file)
    f_file.close()
    
    f_file=open(path+dir_name+'/Quarkyonic_eos_success.dat','wb')
    cPickle.dump(eos_success,f_file)
    f_file.close()
    
    f_file=open(path+dir_name+'/Quarkyonic_eos.dat','wb')
    cPickle.dump(eos[logic_causality_stability],f_file)
    f_file.close()
    
    eos_flat_success=eos[logic_causality_stability]
    eos_success_logic=logic_causality_stability
    
    from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
    from Parallel_process import main_parallel
    f_maxmass_result=path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
    maxmass_result=np.full(eos_success_logic.shape+(3,),np.array([0,0,1]),dtype='float')
    maxmass_result[eos_success_logic]=main_parallel(Calculation_maxmass,eos_flat_success,f_maxmass_result,error_log)
    
    print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat_success)))
    logic_maxmass=maxmass_result[:,:,:,:,1]>=2
    print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_maxmass[eos_success_logic]])))
    logic_causality=maxmass_result[:,:,:,:,2]<1
    print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_causality[eos_success_logic]])))
    logic=np.logical_and(logic_maxmass,logic_causality)
    print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic[eos_success_logic]])))
    
    eos_success_maxmass=np.logical_and(logic,eos_success_logic)
    for eos_flat_success_i,maxmass_result_i in zip(eos_flat_success.flatten(),maxmass_result[eos_success_logic]):
        eos_flat_success_i.setMaxmass(maxmass_result_i)
    
    
    f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_success_maxmass.dat','wb')
    cPickle.dump(eos_success_maxmass,f_file)
    f_file.close()
    
    f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat','wb')
    cPickle.dump(eos_flat_success[eos_success_maxmass[eos_success_logic]],f_file)
    f_file.close()
    
    print('Calculating properities of 1.4 M_sun star...')
    f_onepointfour_result=path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat'
    Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_onepointfour_result,error_log)
    print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))
    
    print('Calculating mass, compactness and tidal Lambda...')
    f_mass_beta_Lambda_result=path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
    mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_mass_beta_Lambda_result,error_log)
    print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))
    
    print('Calculating binary neutron star...')
    f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
    chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)

    if(show_plot==True):
        import matplotlib.pyplot as plt
        import show_properity
        fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
        show_properity.show_eos(axes,eos[eos_success],0,5,500,baryon_density_range=[0.01,2,'log'])
    
        from plot_logic import plot_5D_logic
        import matplotlib.pyplot as plt
        plot_5D_logic(np.logical_not(logic_causality_stability[:,:,:,:]),args[:,:,:,:,:],['E','L','$\Lambda$','$\kappa$'],(0,2,1,3),figsize=(16,15))
        plot_5D_logic(np.logical_not(eos_success[:,:,:,:]),args[:,:,:,:,:],['E','L','$\Lambda$','$\kappa$'],(0,2,1,3),figsize=(16,15))