#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:36:17 2018

@author: sotzee
"""

from scipy.misc import derivative
from scipy.constants import c,G,e
from scipy.interpolate import interp1d
from unitconvert import toMev4#,toMevfm
import numpy as np
import scipy.optimize as opt
#import matplotlib.pyplot as plt

dlnx_cs2=1e-6

def energy_per_baryon_sym(n,n_s,m,T,abcd_sym):
    u=n/n_s
    a_sym,b_sym,c_sym,d_sym=abcd_sym
    return m+T*(u**(2./3)+a_sym*u+b_sym*u**(4./3)+c_sym*u**(5./3)+d_sym*u**2)

def energy_per_baryon_sym_jac(n,n_s,T,abcd_sym):
    u=n/n_s
    a_sym,b_sym,c_sym,d_sym=abcd_sym
    return T*(2.*u**(-1./3)+3*a_sym+4.*b_sym*u**(1./3)+5.*c_sym*u**(2./3)+6.*d_sym*u)/3

def energy_per_baryon_pnm(n,n_s,m,T,abcd_pnm):
    u=n/n_s
    a_pnm,b_pnm,c_pnm,d_pnm=abcd_pnm
    return m+T*((2*u)**(2./3)+a_pnm*u+b_pnm*u**(4./3)+c_pnm*u**(5./3)+d_pnm*u**2)

def energy_per_baryon_pnm_jac(n,n_s,T,abcd_pnm):
    u=n/n_s
    a_pnm,b_pnm,c_pnm,d_pnm=abcd_pnm
    return T*(4.*(2*u)**(-1./3)+3*a_pnm+4.*b_pnm*u**(1./3)+5.*c_pnm*u**(2./3)+6.*d_pnm*u)/3

def get_parameters_tmp(parameter_array,T,ELKQ_array): #where E0,L0,K0,Q0 is for symmetric nuclear matter, and S,L,K,Q are for symmtry energy
    matrix=np.array([[120,-38,6,-1],[-270,90,-15,3],[216,-72,12,-3],[-60,20,-3,1]])
    #print(matrix,ELKQ_array,np.dot(matrix,ELKQ_array))
    return parameter_array+np.dot(matrix,ELKQ_array)/(6*T)

def get_parameters_sym(T,ELKQ_array): #S,L,K,Q are for PNM(pure neutron matter).
    parameter_array=np.array([-4,6,-4,1])
    return get_parameters_tmp(parameter_array,T,ELKQ_array)

def get_parameters_pnm(T,ELKQ_array): #S,L,K,Q are for PNM(pure neutron matter).
    parameter_array=np.array([-4,6,-4,1])*2**(2./3)
    return get_parameters_tmp(parameter_array,T,ELKQ_array)

def get_baryon_density_u_max(abcd,defaut_u_max):
    coeff=[54*abcd[3],40*abcd[2],28*abcd[1],18*abcd[0],10*2**(2./3)]
    roots=np.roots(coeff)
    roots_real=roots.real[np.isreal(roots)]
    if(len(roots_real[roots_real>0])==0):
        return defaut_u_max
    else:
        return np.min([roots_real[roots_real>0].min()**3,defaut_u_max])

def get_eos_array(u_min,u_max,baryon_density_sat,m,T,abcd):
    baryon_density=baryon_density_sat*10**np.linspace(np.log10(u_min),np.log10(u_max),501)
    energy_dnnsity=np.concatenate(([0],baryon_density*energy_per_baryon_pnm(baryon_density,baryon_density_sat,m,T,abcd),[10000]))
    pressure=np.concatenate(([0],baryon_density**2/baryon_density_sat*energy_per_baryon_pnm_jac(baryon_density,baryon_density_sat,T,abcd),[10000]))
    baryon_density=np.concatenate(([0],baryon_density,[1000*baryon_density_sat]))
    result=np.array([baryon_density,energy_dnnsity,pressure])
    #plt.plot(result[0],energy_per_baryon_pnm(baryon_density,baryon_density_sat,m,T,abcd))
    #plt.plot(result[0],result[1])
    #plt.plot(result[0][:-1],result[2][:-1])
    return result,result[:,int(len(baryon_density)/2)]

def matching_eos(trial_pressure,eos_density1,eos_density2):
    return eos_density1(trial_pressure)-eos_density2(trial_pressure)

def calculate_matching_pressure(trial_pressure,Preset_tol,eos_density1,eos_density2):
    p_matching=opt.newton(matching_eos,trial_pressure,tol=Preset_tol,args=(eos_density1,eos_density2))
    return p_matching

class EOS_EXPANSION_PNM(object):
    def __init__(self,args,defaut_u_min=1e-8,defaut_u_max=12):
        self.baryon_density_s,self.m,self.E_n,self.L_n,\
        self.K_n,self.Q_n=args
        self.args=args
        self.ELKQ_array=np.array(args[2:])
        self.T=.3*(1.5*np.pi**2*toMev4(self.baryon_density_s,'mevfm3'))**(2./3)/self.m
        self.abcd_array=get_parameters_pnm(self.T,self.ELKQ_array)
        self.u_max=get_baryon_density_u_max(self.abcd_array,defaut_u_max)
        self.u_min=defaut_u_min
        self.eos_array,self.sol_saturation=get_eos_array(self.u_min,self.u_max,self.baryon_density_s,self.m,self.T,self.abcd_array)
        self.pressure_s=self.sol_saturation[2]
        self.density_s=self.sol_saturation[1]
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        self.eosPressure_frombaryon = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosPressure = interp1d(self.eos_array[1],self.eos_array[2], kind='linear')
        self.eosDensity  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
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
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_CSS(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryondensity_trans,self.cs2 = args
        self.B=(self.density0-self.pressure0/self.cs2)/(1.0+1.0/self.cs2)
    def eosDensity(self,pressure):
        density = (pressure-self.pressure0)/self.cs2+self.density0
        return np.where(density>0,density,0)
    def eosBaryonDensity(self,pressure):
        baryondensity_trans = self.baryondensity_trans*((pressure+self.B)/(self.pressure0+self.B))**(1.0/(1.0+self.cs2))
        return np.where(baryondensity_trans>0,baryondensity_trans,0)
    def eosCs2(self,pressure):
        return self.cs2
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    
class EOS_PnmCSS(object):
    def __init__(self,args,cs2=1):
        self.eosPNM=EOS_EXPANSION_PNM(args)
        self.baryon_density_s=self.eosPNM.baryon_density_s
        self.pressure_s=self.eosPNM.pressure_s
        self.density_s=self.eosPNM.density_s
        self.unit_mass=self.eosPNM.unit_mass
        self.unit_radius=self.eosPNM.unit_radius
        self.unit_N=self.eosPNM.unit_N
        self.baryondensity_trans=self.eosPNM.u_max*self.eosPNM.baryon_density_s*0.9999999
        self.pressure_trans=self.eosPNM.eosPressure_frombaryon(self.baryondensity_trans)
        self.density_trans=self.eosPNM.eosDensity(self.pressure_trans)
        self.cs2=cs2
        args_eosCSS=[self.density_trans,self.pressure_trans\
                     ,self.baryondensity_trans,self.cs2]
        self.eosCSS=EOS_CSS(args_eosCSS)
    def __getstate__(self):
        state_PNM=self.eosPNM.__getstate__()
        state = self.__dict__.copy()
        return (state,state_PNM)
    def __setstate__(self, state_):
        state,state_PNM=state_
        self.__dict__.update(state)
        self.eosPNM.__setstate__(state_PNM)

    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
    def eosDensity(self,pressure):
        return np.where(pressure<self.pressure_trans,self.eosPNM.eosDensity(pressure),self.eosCSS.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.pressure_trans,self.eosPNM.eosBaryonDensity(pressure),self.eosCSS.eosBaryonDensity(pressure))
    def eosCs2(self,pressure):
        return np.where(pressure<self.pressure_trans,self.eosPNM.eosCs2(pressure),self.cs2)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

Preset_tol_matching=1e-4
class EOS_Sly4_match_PnmCSS(object):
    def __init__(self,eos_low,eos_high):
        self.eos_low=eos_low
        self.eos_high=eos_high
        flag=True
        for trial_pressure in [0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1]:
            if(flag==True):
                flag=False
                try:
                    self.p_match=calculate_matching_pressure(trial_pressure,Preset_tol_matching,eos_low.eosDensity,eos_high.eosDensity)
                except:
                    self.p_match=False
                    flag=True
            else:
                break
        if(not self.p_match and eos_high.eosPNM.u_max<1):
            self.p_match=0
        else:
            print('Matching of low density EoS %s and hight density %s failed'%(self.eos_low,self.eos_high))
        self.baryon_density_s=self.eos_high.baryon_density_s
        self.pressure_s=self.eos_high.pressure_s
        self.density_s=self.eos_high.density_s
        self.unit_mass=self.eos_high.unit_mass
        self.unit_radius=self.eos_high.unit_radius
        self.unit_N=self.eos_high.unit_N
    def __getstate__(self):
        state_high=self.eos_high.__getstate__()
        state = self.__dict__.copy()
        return (state,state_high)
    def __setstate__(self, state_):
        state,state_high=state_
        self.__dict__.update(state)
        self.eos_high.__setstate__(state_high)
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
    def eosDensity(self,pressure):
        return np.where(pressure<self.p_match,self.eos_low.eosDensity(pressure),self.eos_high.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.p_match,self.eos_low.eosBaryonDensity(pressure),self.eos_high.eosBaryonDensity(pressure))
    def eosCs2(self,pressure):
        return np.where(pressure<self.p_match,self.eos_low.eosCs2(pressure),self.eos_high.eosCs2(pressure))
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    
import cPickle
import os
path = "./"
dir_name='Lambda_PNM_calculation_parallel'
error_log=path+dir_name+'/error.log'
if __name__ == '__main__':
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)
    N1=6
    N2=15
    N3=21
    n_s=0.16
    m=939
    E_pnm = 32-16
    L_pnm = np.linspace(30,70,N1)
    K_pnm = np.linspace(50,400,N2)
    Q_pnm = np.linspace(-400,600,N3)
    Preset_Pressure_final=1e-8
    Preset_rtol=1e-4
    args=[]
    
    from eos_class import EOS_BPS
    eos_low=EOS_BPS()
    eos_high=[]
    eos =[]
    for i in range(len(L_pnm)):
        for j in range(len(K_pnm)):
            for k in range(len(Q_pnm)):
                args.append([n_s,m,E_pnm,L_pnm[i],K_pnm[j],Q_pnm[k]])
                eos_high.append(EOS_PnmCSS(args[-1]))
                #print args[-1]
                eos.append(EOS_Sly4_match_PnmCSS(eos_low,eos_high[-1]))
    args=np.reshape(np.array(args),(N1,N2,N3,6))
    args_flat=np.reshape(np.array(args),(N1*N2*N3,6))
    eos =np.reshape(np.array(eos),(N1,N2,N3))
    eos_flat=np.array(eos).flatten()
    f_file=open(path+dir_name+'/Lambda_PNM_calculation_args.dat','wb')
    cPickle.dump(args,f_file)
    f_file.close()
    f_file=open(path+dir_name+'/Lambda_PNM_calculation_eos.dat','wb')
    cPickle.dump(eos,f_file)
    f_file.close()
    print('%d EoS built with shape (L_n,K_n,Q_n)%s.'%(len(args_flat),np.shape(eos)))
    
    from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
    from Parallel_process import main_parallel
    
    f_maxmass_result=path+dir_name+'/Lambda_PNM_calculation_maxmass.dat'
    maxmass_result=main_parallel(Calculation_maxmass,eos_flat,f_maxmass_result,error_log)
    print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat)))
    logic_maxmass=maxmass_result[:,1]>=2
    print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic_maxmass])))
    logic_causality=maxmass_result[:,2]<1
    print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic_causality])))
    logic=np.logical_and(logic_maxmass,logic_causality)
    print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic])))

    for i in range(len(eos_flat)):
        eos_flat[i].setMaxmass(maxmass_result[i])
    
    f_onepointfour_result=path+dir_name+'/Lambda_PNM_calculation_onepointfour.dat'
    Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat[logic],f_onepointfour_result,error_log)
    print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat[logic])))
    
    f_mass_beta_Lambda_result=path+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat'
    mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat[logic],f_mass_beta_Lambda_result,error_log)
    print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat[logic])))

    f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
    chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)

#else:
def read_file(file_name):
    f_file=open(file_name,'rb')
    content=np.array(cPickle.load(f_file))
    f_file.close()
    return content
args=read_file(path+dir_name+'/Lambda_PNM_calculation_args.dat')
eos=read_file(path+dir_name+'/Lambda_PNM_calculation_eos.dat')
maxmass_result=read_file(path+dir_name+'/Lambda_PNM_calculation_maxmass.dat')
Properity_onepointfour=read_file(path+dir_name+'/Lambda_PNM_calculation_onepointfour.dat')
mass_beta_Lambda_result=read_file(path+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat')
chirp_q_Lambdabeta6_Lambda1Lambda2=read_file(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat')