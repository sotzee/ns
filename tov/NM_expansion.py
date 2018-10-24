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
    baryon_density=baryon_density_sat*10**np.linspace(np.log10(u_min),np.log10(u_max),201)
    energy_dnnsity=np.concatenate(([0],baryon_density*energy_per_baryon_pnm(baryon_density,baryon_density_sat,m,T,abcd),[10000]))
    pressure=np.concatenate(([0],baryon_density**2/baryon_density_sat*energy_per_baryon_pnm_jac(baryon_density,baryon_density_sat,T,abcd),[10000]))
    baryon_density=np.concatenate(([0],baryon_density,[1000*baryon_density_sat]))
    result=np.array([baryon_density,energy_dnnsity,pressure])
    #plt.plot(result[0],energy_per_baryon_pnm(baryon_density,baryon_density_sat,m,T,abcd))
    #plt.plot(result[0],result[1])
    #plt.plot(result[0][:-1],result[2][:-1])
    return result,result[:,int(len(baryon_density)/2)]

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
# =============================================================================
#     def __getstate__(self):
#         # Copy the object's state from self.__dict__ which contains
#         # all our instance attributes. Always use the dict.copy()
#         # method to avoid modifying the original state.
#         state = self.__dict__.copy()
#         # Remove the unpicklable entries.
#         print(state)
#         for dict_intepolation in ['eosPressure_frombaryon','eosPressure','eosDensity','eosBaryonDensity']:
#             del state[dict_intepolation]
#         print(state)
#         return state
#     def __setstate__(self, state):
#         # Restore instance attributes (i.e., filename and lineno).
#         self.__dict__.update(state)
#         # Restore the previously opened file's state.
#         self.eosPressure_frombaryon = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosPressure = interp1d(self.eos_array[1],self.eos_array[2], kind='linear')
#         self.eosDensity  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
# =============================================================================
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
    
class EOS_PnmCSS(EOS_EXPANSION_PNM,EOS_CSS):
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


#a=EOS_EXPANSION_PNM([0.16,939,32-16,50,100,0])
#import matplotlib.pyplot as plt
#plt.plot(a.eosDensity(a.eosPressure_frombaryon(np.linspace(0.1,0.6,100))),a.eosPressure_frombaryon(np.linspace(0.1,0.6,100)))
#plt.plot(np.linspace(0.1,0.16,100),a.eosDensity(a.eosPressure_frombaryon(np.linspace(0.1,0.16,100))))
#print(energy_per_baryon_pnm(0.16,0.16,939,22.1,get_parameters_pnm(22.1,[939-16+32,50,100,0])))
#print(get_baryon_density_pnm_max(2.,a.abcd_array))


import cPickle
import os
path = "./"
dir_name='Lambda_PNM_calculation_parallel'
error_log='./'+dir_name+'/error.log'
if __name__ == '__main__':
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)
    N1=26
    N2=101
    N3=121
# =============================================================================
#     N1=11
#     N2=61
#     N3=51
# =============================================================================
    n_s=0.16
    m=939
    E_pnm = 32-16
    L_pnm = np.linspace(30,70,N1)
    K_pnm = np.linspace(0,500,N2)
    Q_pnm = np.linspace(-200,1000,N3)
    Preset_Pressure_final=1e-8
    Preset_rtol=1e-4
    args=[]
    eos =[]
    for i in range(len(L_pnm)):
        for j in range(len(K_pnm)):
            for k in range(len(Q_pnm)):
                args.append([n_s,m,E_pnm,L_pnm[i],K_pnm[j],Q_pnm[k]])
                eos.append(EOS_PnmCSS(args[-1]))
    args=np.reshape(np.array(args),(N1,N2,N3,6))
    args_flat=np.reshape(np.array(args),(N1*N2*N3,6))
    eos =np.reshape(np.array(eos),(N1,N2,N3))
    eos_flat=np.array(eos).flatten()
    f_file=open('./'+dir_name+'/Lambda_PNM_calculation_args.dat','wb')
    cPickle.dump(args,f_file)
    f_file.close()
# =============================================================================
#     f_file=open('./'+dir_name+'/Lambda_PNM_calculation_eos.dat','wb')
#     cPickle.dump(eos,f_file)
#     f_file.close()
# =============================================================================
    print('%d EoS built with shape (L_n,K_n,Q_n)%s.'%(len(args_flat),np.shape(eos)))
    
    from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour
    from Parallel_process import main_parallel
    
    f_maxmass_result='./'+dir_name+'/Lambda_PNM_calculation_maxmass.dat'
    main_parallel(Calculation_maxmass,eos_flat,f_maxmass_result,error_log)
    f_file=open(f_maxmass_result,'rb')
    maxmass_result=cPickle.load(f_file)
    f_file.close()
    print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat)))
    logic_maxmass=maxmass_result[:,1]>=2
    print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic_maxmass])))
    logic_causality=maxmass_result[:,2]<1
    print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic_causality])))
    logic=np.logical_and(logic_maxmass,logic_causality)
    print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat),len(eos_flat[logic])))

    for i in range(len(eos_flat)):
        eos_flat[i].setMaxmass(maxmass_result[i])
    
    f_onepointfour_result='./'+dir_name+'/Lambda_PNM_calculation_onepointfour.dat'
    main_parallel(Calculation_onepointfour,eos_flat[logic],f_onepointfour_result,error_log)
    f_file=open(f_onepointfour_result,'rb')
    Properity_onepointfour=np.array(cPickle.load(f_file))
    f_file.close()
    
    
# =============================================================================
#     def Calculation_error(args,i,pc_list=10**np.linspace(0,-1.5,40)):
#         eos_i=args[i][0]
#         maxmass_pc_i=args[i][1]
#         mass=[]
#         beta=[]
#         Lambda=[]
#         for pc_i in pc_list:
#             mass.append(1*pc_i)
#             beta.append(1*pc_i)
#             Lambda.append(1*pc_i)
#         return [mass,beta,Lambda]
#     print(np.shape(np.array([eos_flat[logic],maxmass_result[logic][:,0]]).transpose()))
# =============================================================================
    f_mass_beta_Lambda_result='./'+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat'
    main_parallel(Calculation_mass_beta_Lambda,eos_flat[logic],f_mass_beta_Lambda_result,error_log)
    f_file=open(f_mass_beta_Lambda_result,'rb')
    mass_beta_Lambda_result=cPickle.load(f_file)
    f_file.close()
    print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos)))


else:
    f_file=open('./'+dir_name+'/Lambda_hadronic_calculation_args.dat','rb')
    args=np.array(cPickle.load(f_file))
    f_file.close()
    f_file=open('./'+dir_name+'/Lambda_hadronic_calculation_eos.dat','rb')
    eos=np.array(cPickle.load(f_file))
    f_file.close()

    f_maxmass_result='./'+dir_name+'/Lambda_PNM_calculation_maxmass.dat'
    f_file=open(f_maxmass_result,'rb')
    maxmass_result=cPickle.load(f_file)
    f_file.close()
    logic_maxmass=maxmass_result[:,1]>=2
    logic_causality=maxmass_result[:,2]<1
    logic=np.logical_and(logic_maxmass,logic_causality)

    f_mass_beta_Lambda_result='./'+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat'
    f_file=open(f_mass_beta_Lambda_result,'rb')
    mass_beta_Lambda_result=cPickle.load(f_file)
    f_file.close()


# =============================================================================
# def Calculation_maxmass(eos):
#     result=[]
#     i=0
#     for eos_i in eos:
#         print(i)
#         i+=1
#         maxmass_result_i=Maxmass(Preset_Pressure_final,Preset_rtol,eos_i)[1:3]
#         result.append(maxmass_result_i+[eos_i.eosCs2(maxmass_result_i[0])])
#     return result
# f_maxmass_result='./'+dir_name+'/Lambda_PNM_calculation_maxmass.dat'
# maxmass_result=np.array(Calculation_maxmass(eos_flat))
# f_file=open(f_maxmass_result,'wb')
# cPickle.dump(maxmass_result,f_file)
# f_file.close()
# logic_maxmass=maxmass_result[:,1]>=2
# logic_causality=maxmass_result[:,2]<1
# logic=np.logical_and(logic_maxmass,logic_causality)
# 
# def show_PNM_eos(eos,x_index,y_index,baryon_density_range,N):#index baryon_density(0), pressure(1), energy density(2), energy per baryon(3), chempo(4)
#     pressure_density_energyPerBaryon_chempo=[]
#     for eos_i in eos:
#         eos_i=eos_i.eosPNM
#         baryon_density_i=np.linspace(baryon_density_range[0],np.min([baryon_density_range[1],eos_i.u_max*eos_i.baryon_density_s]),N)
#         pressure_density_energyPerBaryon_chempo_i=[]
#         pressure_density_energyPerBaryon_chempo_i.append(baryon_density_i)
#         pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosPressure_frombaryon(baryon_density_i))
#         pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_density_energyPerBaryon_chempo_i[1]))
#         pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/baryon_density_i)
#         pressure_density_energyPerBaryon_chempo_i.append((pressure_density_energyPerBaryon_chempo_i[1]+pressure_density_energyPerBaryon_chempo_i[2])/baryon_density_i)
#         pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
#         plt.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index])
#     pressure_density_energyPerBaryon_chempo=np.array(pressure_density_energyPerBaryon_chempo)
#     label_text=['Baryon density(fm$^{-3}$)','Pressure(MeV fm$^{-3}$)','Energy density(MeV fm$^{-3}$)','Energy per baryon(MeV)','Chemical potential(MeV)']
#     plt.xlabel(label_text[x_index])
#     plt.ylabel(label_text[y_index])
#     #plt.xlim(pressure_density_energyPerBaryon_chempo[:,x_index,:].min(),pressure_density_energyPerBaryon_chempo[:,x_index,:].max())
#     #plt.ylim(pressure_density_energyPerBaryon_chempo[:,y_index,:].min(),pressure_density_energyPerBaryon_chempo[:,y_index,:].max())
# 
# 
# show_PNM_eos(eos_flat[logic],2,1,[0.00016,1.85*0.16],100)
# from eos_class import BPS,EOS_BPSwithPoly
# BPSpoly1=EOS_BPSwithPoly([0.059259259259259255, 13.0, 1.85*0.16, 200, 0.5984, 500, 1.1840000000000002])
# BPSpoly2=EOS_BPSwithPoly([0.059259259259259255, 25.0, 1.85*0.16, 200, 0.5984, 500, 1.1840000000000002])
# plt.plot(BPS.eosDensity(np.linspace(0.001,25,100)),(np.linspace(0.001,25,100)),'k',label='SLY4')
# plt.plot(BPSpoly1.eosDensity(np.linspace(0.001,13,100)),(np.linspace(0.001,13,100)),'k--',label='SLY4_piecewisePoly1')
# plt.plot(BPSpoly2.eosDensity(np.linspace(0.001,25,100)),(np.linspace(0.001,25,100)),'k-.',label='SLY4_piecewisePoly2')
# plt.legend()
# 
# plt.figure()
# plt.imshow(np.reshape(logic,(N1,N2,N3))[5].transpose(),aspect='auto',origin='lower',extent=(K_pnm.min(),K_pnm.max(),Q_pnm.min(),Q_pnm.max()))
# plt.title('L=50 MeV')
# plt.ylabel('$Q_n$ MeV')
# plt.xlabel('$K_n$ MeV')
# 
# def Calculation_mass_beta_Lambda(eos,maxmass_result,pc_list=10**np.linspace(0,-1.5,10)):
#     result=[]
#     for i in range(len(eos)):
#         mass=[]
#         beta=[]
#         Lambda=[]
#         for j in range(len(pc_list)):
#             MR_result=MassRadius(maxmass_result[i][0]*pc_list[j],Preset_Pressure_final,Preset_rtol,'MRT',eos[i])
#             mass.append(MR_result[0])
#             beta.append(MR_result[2])
#             Lambda.append(MR_result[4])
#         result.append([mass,beta,Lambda])
#     return result
# f_mass_beta_Lambda_result='./'+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat'
# mass_beta_Lambda_result=np.array(Calculation_mass_beta_Lambda(eos_flat[logic],maxmass_result[logic]))
# f_file=open(f_mass_beta_Lambda_result,'wb')
# cPickle.dump(mass_beta_Lambda_result,f_file)
# f_file.close()
# 
# M_min=1.1
# M_max=1.6
# def mass_chirp(mass1,mass2):
#     return (mass1*mass2)**0.6/(mass1+mass2)**0.2
# def tidal_binary(q,tidal1,tidal2):
#     return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
# def Calculation_chirpmass_Lambdabeta6(args_list,i):
#     mass_beta_Lambda=list(args_list[:,0])
#     beta_onepointfour=args_list[:,1]
#     mass=np.array(mass_beta_Lambda)[:,0]
#     Lambda=np.array(mass_beta_Lambda)[:,2]
#     logic_mass=np.logical_and(mass[i]>M_min,mass[i]<M_max)
#     mass1,mass2 = np.meshgrid(mass[i][logic_mass],mass[i][logic_mass])
#     Lambda1,Lambda2 = np.meshgrid(Lambda[i][logic_mass],Lambda[i][logic_mass])
#     chirp_mass=mass_chirp(mass1,mass2).flatten()
#     Lambda_binary_beta6=(beta_onepointfour[i]/1.4*chirp_mass)**6*tidal_binary(mass2/mass1,Lambda1,Lambda2).flatten()
#     return [chirp_mass,Lambda_binary_beta6]
# 
# f_chirpmass_Lambdabeta6_result='./'+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
# main_parallel(Calculation_chirpmass_Lambdabeta6,np.array([list(mass_beta_Lambda_result),list(Properity_onepointfour[:,3])]).transpose(),f_chirpmass_Lambdabeta6_result,0)
# f_file=open(f_chirpmass_Lambdabeta6_result,'rb')
# chirpmass_Lambdabeta6_result=np.array(cPickle.load(f_file))
# f_file.close()
# chirp_mass=chirpmass_Lambdabeta6_result[:,0]
# Lambda_binary_beta6=chirpmass_Lambdabeta6_result[:,1]
# =============================================================================
