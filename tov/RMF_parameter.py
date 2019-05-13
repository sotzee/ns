#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:20:42 2019

@author: sotzee
"""

import numpy as np
from unitconvert import toMevfm,toMev4
from solver_equations import solve_equations,explore_solutions,Calculation_unparallel
from Parallel_process import main_parallel,main_parallel_unsave
from scipy.misc import derivative
import scipy.constants as const
from scipy.interpolate import interp1d
from eos_class import EOS_BPS,match_eos,match_get_eos_array
import pickle as cPickle

baryon_density_s=0.15
m=939
BindE=16
K=240
mass_args=(0.5109989461,939,550,783,763)
baryon_density_s_MeV4=toMev4(baryon_density_s,'mevfm')
equations_extra_args=(baryon_density_s_MeV4,m-BindE,K,mass_args)

def equations(x,args,args_extra):
    m_eff,J,L,self_W=args
    baryon_density_sat,bd_energy,incompressibility,mass_args=args_extra
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda=x
    k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    W_0=bd_energy-E_F
    n_scalar_sym=(m_eff/np.pi**2)*(E_F*k_F-m_eff**2*np.log((k_F+E_F)/m_eff))
    eq1=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar_sym)
    eq2=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 - baryon_density_sat)
    tmp_2=g2_W/(m_W**2+self_W/2*g2_W*W_0**2)
    tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
    eq3=m**2*(E_F**2*tmp_1*((m_Phi)**2/g2_Phi+2*b*m*Phi_0+3*c*Phi_0**2+(1/np.pi**2)*(k_F/E_F*(E_F**2+2*m_eff**2)-3*m_eff**2*np.log((k_F+E_F)/m_eff)))+m_eff**2)
    eq4=(1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi_0*m_Phi)**2/(2*g2_Phi) + (m*b)*Phi_0**3/3 + c*Phi_0**4/4 -(W_0*m_W)**2/(2*g2_W) - (self_W/6)*W_0**4/4 - baryon_density_sat*E_F
    tmp_J_0=k_F**2/(6*E_F)
    tmp_J_1=baryon_density_sat*g2_rho/(8*(m_rho)**2+16*Lambda*(W_0)**2*g2_rho) #origin fomula was baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*(W_0)**2)
    eq5=m**3*(tmp_J_0+tmp_J_1-J)
    eq6=m**3*(tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)-L)
    return eq1,eq2,eq3,eq4,eq5,eq6

def logic_sol(sol_x,init_i,args):
    return sol_x[:3].min()>0 and sol_x[2].max()<250 #this 250 is tuned in order to make PNM energy density stay in range. using 1000 with make Max(PNM energy density) go to 


def eos_J_L_around_sym(baryon_density_sat,bd_energy,incompressibility,_args):
    mass_args,eos_args,args=_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    baryon_density_sat=baryon_density_s_MeV4
    bd_energy=m-BindE
    incompressibility=K
    m_eff=args[0]
    k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    W_0=bd_energy-E_F
    tmp_2=g2_W/(m_W**2+self_W/2*g2_W*W_0**2)
    tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
    tmp_J_0=k_F**2/(6*E_F)
    tmp_J_1=baryon_density_sat*g2_rho/(8*(m_rho)**2+16*Lambda*(W_0)**2*g2_rho) #origin fomula was baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*(W_0)**2)
    J=tmp_J_0+tmp_J_1
    L=tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)
    return J,L
def Calculation_J_L_around_sym(eos_args_args):
    return eos_J_L_around_sym(baryon_density_s_MeV4,m-BindE,K,(mass_args,eos_args_args[:7],eos_args_args[7:]))

# =============================================================================
# def eos_equations_PNM_get_m_eff(m_eff,args,args_extra):
#     n,mass_args,eos_args=args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
#     n_n=n
#     k_F_n=(3*np.pi**2*n_n)**(1./3)
#     E_F_n=(k_F_n**2+m_eff**2)**0.5
#     n_scalar=(m_eff/(2*np.pi**2))*((E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
#     Phi_0=m-m_eff
#     eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
#     return eq2
# def eos_equations_PNM_get_W_0(W_0,args,args_extra):
#     m_eff,n,mass_args,eos_args=args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
#     n_n=n
#     n3=n_n
#     rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
#     eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
#     return eq3
# def eos_pressure_density_PNM(n,Preset_tol,_args):
#     mass_args,eos_args,args=_args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
#     sol1_success,sol1 = solve_equations(eos_equations_PNM_get_m_eff,[[args[0]]],vary_list=np.array([1.,0.95,1.05]),tol=Preset_tol,args=[n,mass_args,eos_args])
#     m_eff=sol1[0]
#     k_F=(3*np.pi**2*n)**(1./3.)
#     E_F=(k_F**2+m_eff**2)**0.5
#     W_0=m-BindE-E_F
#     sol2_success,sol2 = solve_equations(eos_equations_PNM_get_W_0,[[W_0]],vary_list=np.array([1.,0.95,1.05]),tol=Preset_tol,args=[m_eff,n,mass_args,eos_args])
#     W_0=sol2[0]
#     n_n=n
#     n3=n_n
#     k_F_n=(3*np.pi**2*n_n)**(1./3)
#     E_F_n=(k_F_n**2+m_eff**2)**0.5
#     Phi_0=m-m_eff
#     rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
#     chempo_n=E_F_n+W_0+rho_0/2
#     energy_density=((E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
#     pressure=chempo_n*n_n-energy_density
#     return [sol1_success and sol2_success,energy_density,pressure]
# 
# def Calculation_PNM_saturation(eos_args_args):
#     return eos_pressure_density_PNM(baryon_density_s_MeV4,Preset_tol,(mass_args,eos_args_args[:7],eos_args_args[7:]))
# 
# f_PNM_saturation='./'+dir_name+'/RMF_PNM_saturation.dat'
# error_log=path+dir_name+'/error.log'
# PNM_saturation=main_parallel(Calculation_PNM_saturation,np.concatenate((eos_args_flat,args_flat),axis=1)[eos_args_logic_flat],f_PNM_saturation,error_log)
# =============================================================================

def equations_PNM(x,eos_args,args_extra):
    m_eff,W_0=x
    n,mass_args=args_extra
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    n_n=n
    n3=-n_n
    k_F_n=(3*np.pi**2*n_n)**(1./3)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    n_scalar=(m_eff/(2*np.pi**2))*((E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    return eq2,eq3

def logic_sol_PNM(sol_x,init_i,args):
    return sol_x[1]>0 and sol_x[0]<m  and np.abs(sol_x[1]-init_i[1])/init_i[1]<0.10

def pressure_density_PNM(PNM_args,eos_args,extra_args):
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    n,mass_args=extra_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    m_eff,W_0=PNM_args
    n_n=n
    n3=-n_n
    k_F_n=(3*np.pi**2*n_n)**(1./3)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_n=E_F_n+W_0-rho_0/2
    energy_density=((E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_n*n_n-energy_density
    return toMevfm(np.array([energy_density,pressure]),'mev4')


# =============================================================================
# equations_185PNM_extra_args=(1.85*baryon_density_s_MeV4,mass_args)
# init_args=(700.,400)
# PNM_185saturation_logic,PNM_185args=explore_solutions(equations_PNM,eos_args,((5,7,10,0),),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=logic_sol_PNM,Calculation_routing=Calculation_unparallel,equations_extra_args=equations_PNM_extra_args)
# PNM_185saturation_logic=np.logical_and(PNM_185saturation_logic,eos_args_logic)
# PNM_185saturation=pressure_density_PNM(PNM_185args,eos_args,equations_185PNM_extra_args)
# PNM_185saturation_constrain_L30=np.logical_and(PNM_185saturation[1]<30.,PNM_185saturation_logic)
# PNM_185saturation_constrain_L840=np.logical_and(PNM_185saturation[1]>8.4,PNM_185saturation_logic)
# PNM_185saturation_constrain_L374=np.logical_and(PNM_185saturation[1]>3.74,PNM_185saturation_logic)
# plt.figure()
# plt.plot(PNM_185saturation[0][PNM_185saturation_logic],PNM_185saturation[1][PNM_185saturation_logic],'.')
# plt.plot(PNM_185saturation[0][PNM_185saturation_logic],PNM_185saturation[1][PNM_185saturation_logic],'.')
# plt.xlabel('$\\varepsilon _s$(MeV fm$^{-3}$)')
# plt.ylabel('$p_s$(MeV fm$^{-3}$)')
# plt.title('Neutron matter at saturation density')
# 
# plot_5D_logic(np.logical_not((PNM_185saturation_constrain_L374[:,::3,:,::4])),args[:,:,::3,:,::4],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))
# =============================================================================

def eos_equations(y,eos_args,equations_extra_args):
    mass_args=equations_extra_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    n,g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    m_eff,W_0,k_F_n=y
    
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
    n3=n_p-n_n
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5

# =============================================================================
#     if(m_eff<=0):
#         n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p)+(E_F_n*k_F_n))
#     else:
#         n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p-m_eff**2*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
# =============================================================================
    n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p-m_eff**2*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    #eq5=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0/g_Phi)**2/2+(m_W*W_0/g_W)**2/2+(m_rho*rho_0/g_rho)**2/2+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2-energy_density
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0+rho_0/2
    chempo_n=E_F_n+W_0-rho_0/2
    eq6=chempo_e+chempo_p-chempo_n
    #eq7=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density-pressure
    eq6=chempo_e+chempo_p-chempo_n
    return eq2,eq3,eq6

def eos_logic_sol(sol_x,init_i,args):
    return sol_x[1]>0 and sol_x[0]<m and sol_x[2]**3/(3*np.pi**2)<args[0] #and (np.abs(sol_x-init_i)/init_i).max()<0.1 # and np.abs((sol_x-init_i)/init_i).max()<0.1

def Calculation_parallel_sub(args_i,other_args):
    logic_calculatable_array_i,init_array_i,args_array_i=args_i
    #equations,vary_list,tol,logic_success_f=other_args
    vary_list,tol,equations_extra_args=other_args
    success_i,result_i=solve_equations(eos_equations,init_array_i[logic_calculatable_array_i],vary_list=vary_list,tol=tol,args=args_array_i,logic_success_f=eos_logic_sol,equations_extra_args=equations_extra_args)
    return [success_i]+result_i
    
def Calculation_parallel(equations,logic_calculatable_array,init_array,args_array,vary_list=np.linspace(1.,1.,1),tol=1e-12,logic_success_f=eos_logic_sol,equations_extra_args=[]):
    main_parallel_result=main_parallel_unsave(Calculation_parallel_sub,zip(logic_calculatable_array,init_array,args_array),other_args=(vary_list,tol,equations_extra_args))
    return main_parallel_result[:,0].astype('bool'),main_parallel_result[:,1:]

def eos_pressure_density(eos_array_args,eos_args_with_baryon,mass_args):
    m_e,m,m_Phi,m_W,m_rho=mass_args
    n,g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args_with_baryon
    m_eff,W_0,k_F_n=eos_array_args
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n3=n_p-n_n
    n_e=n_p
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0+rho_0/2
    chempo_n=E_F_n+W_0-rho_0/2
    energy_density=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density
    return toMevfm(np.array([n,energy_density,pressure]),'mev4')

dlnx_cs2=1e-6
class EOS_SLY4_match_RMF(object):
    match_init=[None,None] #not in use
    def __init__(self,args,eos_args,eos_array_rmf):
        self.eos_args=eos_args
        self.args=args
        self.baryon_density_s=baryon_density_s
        self.bd_energy=m-BindE
        self.incompressibility=K
        self.mass_args=mass_args
        
        self.eos_array_rmf=eos_array_rmf[:,eos_array_rmf[2]!=0.]
        self.n_max,self.e_max,self.p_max=self.eos_array_rmf[:,-1]
        self.eos_array_rmf=np.concatenate((self.eos_array_rmf,self.eos_array_rmf[:,-1][:,np.newaxis]+np.multiply((self.eos_array_rmf[:,-1]-self.eos_array_rmf[:,-2])[:,np.newaxis],eos_array_rmf[0,self.eos_array_rmf.shape[1]:]-self.eos_array_rmf[0,-1])/(self.eos_array_rmf[0,-1]-self.eos_array_rmf[0,-2])),axis=1)
        self.positive_pressure=self.eos_array_rmf[2,:].min()>0
        
        self.n1_match=0.06
        self.n2_match,self.density_s,self.pressure_s=self.eos_array_rmf[:,0]
        p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
        e1=EOS_BPS.eosDensity(p1)
        dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
        p2=self.pressure_s
        e2=self.density_s
        dpdn2=dpdn1#this parameter is not used in match_eos, so it was trivially set to dpdn1
        self.p_match=p1
        self.matching_args=self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init
        self.matching_para,self.matching_success=match_eos(self.matching_args)
        self.eos_success=self.matching_success and self.positive_pressure
        if(self.matching_success):
            u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,81)
            eos_array_match=match_get_eos_array(u_array_low,self.matching_para)
        else:
            eos_array_match=[[self.n1_match,self.n2_match],[e1,e2],[p1,p2]]
        self.eos_array=np.concatenate((np.array([[0],[0],[0]]),np.array(eos_array_match)[:,:-1],self.eos_array_rmf),axis=1)
        self.unit_mass=const.c**4/(const.G**3*self.density_s*1e51*const.e)**0.5
        self.unit_radius=const.c**2/(const.G*self.density_s*1e51*const.e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        self.eosPressure_frombaryon_matchRMF = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosDensity_matchRMF  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity_matchRMF = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon_matchRMF','eosDensity_matchRMF','eosBaryonDensity_matchRMF']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eosPressure_frombaryon_matchRMF = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
        self.eosDensity_matchRMF  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
        self.eosBaryonDensity_matchRMF = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
    def eosPressure_frombaryon(self,baryon_density):
        return np.where(baryon_density<self.n1_match,EOS_BPS.eosPressure_frombaryon(baryon_density),self.eosPressure_frombaryon_matchRMF(baryon_density))
    def eosDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosDensity(pressure),self.eosDensity_matchRMF(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS.eosBaryonDensity(pressure),self.eosBaryonDensity_matchRMF(pressure))
    def eosCs2(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS().eosCs2(pressure),1.0/derivative(self.eosDensity_matchRMF,pressure,dx=pressure*dlnx_cs2))
    def eosChempo(self,pressure):
        return np.where(pressure<self.p_match,EOS_BPS().eosChempo(pressure),(pressure+self.eosDensity_matchRMF(pressure))/self.eosBaryonDensity_matchRMF(pressure))
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and self.mass_max>2 and self.pc_max<self.p_max
        self.eos_success_all=self.maxmass_success and self.eos_success
        return self.eos_success_all

def Calculation_creat_eos_RMF(eos_args_args_array):
    return EOS_SLY4_match_RMF(*eos_args_args_array)


import os
path = "./"
dir_name='Lambda_RMF_calculation_parallel_test_matching'
error_log=path+dir_name+'/error.log'
if __name__ == '__main__':
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)

    show_plot=False
    
    #args=np.mgrid[0.5*939:0.8*939:31j,30:36:13j,0:120:49j,0:0.03:7j]
    args=np.mgrid[0.5*939:0.8*939:16j,30:36:4j,0:120:13j,0:0.03:4j]
    args_flat=args.reshape((-1,np.prod(np.shape(args)[1:]))).transpose()
    f_file=open(path+dir_name+'/RMF_args.dat','wb')
    cPickle.dump(args,f_file)
    f_file.close()
    J,m_eff,self_W,L=args
    args_shape=np.shape(m_eff)
    
    init_args= (90.,90.,90.,0.001,0.001,0.)
    init_index=tuple(np.array(args_shape)/2)
    eos_args_logic,eos_args=explore_solutions(equations,args,(init_index,),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=logic_sol,Calculation_routing=Calculation_unparallel,equations_extra_args=equations_extra_args)
    eos_args=np.concatenate((eos_args,args[[3]]))
    f_file=open(path+dir_name+'/RMF_eos_args.dat','wb')
    cPickle.dump([eos_args_logic,eos_args],f_file)
    f_file.close()
    eos_args_flat=eos_args.reshape((-1,np.prod(args_shape))).transpose()
    eos_args_logic_flat=eos_args_logic.astype(bool).flatten()
    
    f_J_L_around_sym='./'+dir_name+'/RMF_J_L_around_sym.dat'
    error_log=path+dir_name+'/error.log'
    J_L_around_sym=main_parallel(Calculation_J_L_around_sym,np.concatenate((eos_args_flat,args_flat),axis=1)[eos_args_logic_flat],f_J_L_around_sym,error_log)
    
    init_index=(0,-1,-1,0)
    init_args=(args[0][init_index],384.5)
    equations_PNM_extra_args=(baryon_density_s_MeV4,mass_args)
    PNM_saturation_logic,PNM_args=explore_solutions(equations_PNM,eos_args,(init_index,),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=logic_sol_PNM,Calculation_routing=Calculation_unparallel,equations_extra_args=equations_PNM_extra_args)
    PNM_saturation_logic=np.logical_and(PNM_saturation_logic,eos_args_logic)
    PNM_saturation=pressure_density_PNM(PNM_args,eos_args,equations_PNM_extra_args)
    
    N=100
    baryon_density=baryon_density_s_MeV4*np.exp(np.linspace(0,np.log(12),N))
    baryon_density=np.tile(baryon_density,np.concatenate(([1],args_shape,[1]))).transpose(np.concatenate(([0,len(args_shape)+1],np.array(range(1,len(args_shape)+1)))))
    eos_args_with_baryon=np.tile(eos_args,np.concatenate(([N],np.full(len(args_shape)+1,1)))).transpose(np.concatenate(([1,0],np.array(range(2,len(args_shape)+2)))))
    eos_args_with_baryon=np.concatenate((baryon_density,eos_args_with_baryon))
    
    #init_index=tuple(np.array(args_shape)/2)
    init_index=tuple(np.array(args_shape)/2)
    init_args=(PNM_args[0][init_index],PNM_args[1][init_index],((3*np.pi**2)*baryon_density_s_MeV4)**0.33)
    init_index=(0,)+init_index
    eos_equations_extra_args=mass_args
    eos_array_logic_,eos_array_args=explore_solutions(eos_equations,eos_args_with_baryon,(init_index,),(init_args,),vary_list=np.array([1.]),tol=1e-12,logic_success_f=eos_logic_sol,Calculation_routing=Calculation_parallel,equations_extra_args=eos_equations_extra_args)
    eos_check_discontinuity=np.logical_or((np.abs((eos_array_args[:,1:]-eos_array_args[:,:-1])/eos_array_args[:,:-1]))<0.1,eos_array_args[:,1:]==0).min(axis=(0,1))
    eos_array_logic=np.logical_and(PNM_saturation_logic,eos_check_discontinuity)
    eos_array=eos_pressure_density(eos_array_args,eos_args_with_baryon,mass_args)#.reshape((3,N,-1)).transpose((2,1,0))
    
    
    f_eos_RMF='./'+dir_name+'/RMF_eos.dat'
    error_log=path+dir_name+'/error.log'
    eos_flat=main_parallel(Calculation_creat_eos_RMF,zip(args[:,eos_array_logic].transpose(),eos_args[:,eos_array_logic].transpose(),eos_array[:,:,eos_array_logic].transpose((2,0,1))),f_eos_RMF,error_log)
    
    eos_success=[]
    matching_success=[]
    positive_pressure=[]
    for eos_i in eos_flat:
        matching_success.append(eos_i.matching_success)
        eos_success.append(eos_i.eos_success)
        positive_pressure.append(eos_i.positive_pressure)
    eos_success=np.array(eos_success)
    matching_success=np.array(matching_success)
    positive_pressure=np.array(positive_pressure)
    print('len(eos)=%d'%len(eos_success))
    print('len(eos[positive_pressure])=%d'%len(positive_pressure[positive_pressure]))
    print('len(eos[matching_success])=%d'%len(matching_success[matching_success]))
    print('len(eos[eos_success])=%d'%len(eos_success[eos_success]))
    
    # =============================================================================
    # f_file=open(path+dir_name+'/RMF_eos_logic.dat','wb')
    # cPickle.dump(eos_array_logic,f_file)
    # f_file.close()
    # 
    # f_file=open(path+dir_name+'/RMF_eos_success.dat','wb')
    # cPickle.dump(eos_success,f_file)
    # f_file.close()
    # 
    # f_file=open(path+dir_name+'/RMF_eos.dat','wb')
    # cPickle.dump(eos_flat,f_file)
    # f_file.close()
    # 
    # eos_flat_success=eos_flat
    # eos_success_logic=eos_array_logic
    # 
    # from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
    # from Parallel_process import main_parallel
    # 
    # f_maxmass_result=path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
    # maxmass_result=np.full(eos_success_logic.shape+(3,),np.array([0,0,1]),dtype='float')
    # maxmass_result[eos_success_logic]=main_parallel(Calculation_maxmass,eos_flat_success,f_maxmass_result,error_log)
    # 
    # print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat_success)))
    # logic_maxmass=maxmass_result[:,:,:,:,1]>=2
    # print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_maxmass[eos_success_logic]])))
    # logic_causality=maxmass_result[:,:,:,:,2]<1
    # print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_causality[eos_success_logic]])))
    # logic=np.logical_and(logic_maxmass,logic_causality)
    # print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic[eos_success_logic]])))
    # 
    # eos_success_maxmass=np.logical_and(logic,eos_success_logic)
    # for eos_flat_success_i,maxmass_result_i in zip(eos_flat_success.flatten(),maxmass_result[eos_success_logic]):
    #     eos_flat_success_i.setMaxmass(maxmass_result_i)
    # 
    # 
    # 
    # f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_success_maxmass.dat','wb')
    # cPickle.dump(eos_success_maxmass,f_file)
    # f_file.close()
    # 
    # f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat','wb')
    # cPickle.dump(eos_flat_success[eos_success_maxmass[eos_success_logic]],f_file)
    # f_file.close()
    # 
    # print('Calculating properities of 1.4 M_sun star...')
    # f_onepointfour_result=path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat'
    # Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_onepointfour_result,error_log)
    # print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))
    # 
    # print('Calculating mass, compactness and tidal Lambda...')
    # f_mass_beta_Lambda_result=path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
    # mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_mass_beta_Lambda_result,error_log)
    # print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))
    # 
    # print('Calculating binary neutron star...')
    # f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
    # chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)
    # 
    # 
    # =============================================================================
    if(show_plot==True):
        from plot_logic import plot_5D_logic
        import matplotlib.pyplot as plt
        plot_5D_logic(np.logical_not(eos_args_logic[:,::3,:,::4]),args[:,:,::3,:,::4],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))
        plot_5D_logic(np.logical_not((PNM_saturation_logic[:,::3,:,::4])),args[:,:,::3,:,::4],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))
        
        
        for L_i_min in [0,2,3]:
            plt.figure(figsize=(10,6))
            plt.plot(PNM_saturation[0,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]],PNM_saturation[1,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]],'.',label='PNM')
            plt.plot((m-BindE+args[1,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]])*baryon_density_s,args[2,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]]*baryon_density_s/3,'.',label='SNM')
            plt.xlabel('$(m-BE+J)n_s$ or $\\varepsilon _s$(MeV fm$^{-3}$)',fontsize=20)
            plt.ylabel('$Ln_s/3$ or $p_s$(MeV fm$^{-3}$)',fontsize=20)
            #plt.xlim(142.5,145)
            #plt.ylim(-4,6)
            plt.title('SNM expansion vs PNM at saturation \n(%.2f MeV<L<%.2f MeV)'%(args[2,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]].min(),args[2,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]].max()),fontsize=25)
            plt.legend(fontsize=20)
        
        
        plot_5D_logic(np.logical_not((eos_array_logic[:,::3,:,::4])),args[:,:,::3,:,::4],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))
        
        
        plt.figure()
        for plot_i in eos_array[:,:,eos_array_logic].transpose((2,0,1))[::100]:
            plt.plot(plot_i[0],plot_i[2])
        
        plt.figure()
        for plot_i in eos_array_args[:,:,eos_array_logic].transpose((2,0,1))[::100]:
            plt.plot(0.16*np.exp(np.linspace(0,np.log(12),N)),plot_i[0])
        plt.ylim(0,900)
        
        
        for L_i_min in [0,2,3]:
            plt.figure(figsize=(10,6))
            plt.plot(PNM_saturation[0,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]],PNM_saturation[1,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]],'.',label='PNM')
            plt.plot((eos_array[1,0,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]]),eos_array[2,0,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]],'.',label='$\\beta$-eqlibrium')
            plt.plot((m-BindE+args[1,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]])*baryon_density_s,args[2,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]]*baryon_density_s/3,'.',label='SNM')
            plt.xlabel('$(m-BE+J)n_s$ or $\\varepsilon _s$(MeV fm$^{-3}$)',fontsize=20)
            plt.ylabel('$Ln_s/3$ or $p_s$(MeV fm$^{-3}$)',fontsize=20)
            plt.xlim(142.5,145)
            plt.ylim(0,6)
            plt.title('SNM expansion vs PNM at saturation \n(%.2f MeV<L<%.2f MeV)'%(args[2,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]].min(),args[2,:,:,L_i_min:][PNM_saturation_logic[:,:,L_i_min:]].max()),fontsize=25)
            plt.legend(fontsize=20)
        
        fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
        from show_properity import show_eos
        show_eos(axes,eos_flat[eos_success],0,1,500,pressure_range=[0.01,30,'log'])
        
        eos_success_logic=np.copy(eos_array_logic)
        eos_success_logic[eos_success_logic]=eos_success
        plot_5D_logic(np.logical_not((eos_success_logic[:,::3,:,::4])),args[:,:,::3,:,::4],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))


# =============================================================================
# def eos_pressure_density(n,init,Preset_tol,args):
#     mass_args,eos_args=args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
#     sol_success,sol_x = solve_equations(eos_equations,(init,),tol=Preset_tol,vary_list=np.array([1,0.9,1.1]),args=[n]+list(eos_args),logic_success_f=eos_logic_sol,equations_extra_args=mass_args)
#     m_eff,W_0,k_F_n=sol_x
#     n_n=k_F_n**3/(3*np.pi**2)
#     n_p=n-n_n
#     #n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
#     n3=n_n-n_p
#     n_e=n_p
#     k_F_p=(3*np.pi**2*n_p)**(1./3)
#     k_F_e=k_F_p
#     E_F_e=(k_F_e**2+m_e**2)**0.5
#     E_F_p=(k_F_p**2+m_eff**2)**0.5
#     E_F_n=(k_F_n**2+m_eff**2)**0.5
#     Phi_0=m-m_eff
#     rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
#     chempo_e=E_F_e
#     chempo_p=E_F_p+W_0-rho_0/2
#     chempo_n=E_F_n+W_0+rho_0/2
#     energy_density=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
#     pressure=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density
#     return [sol_success,sol_x,energy_density,pressure]
# 
# def get_eos_array(init0,Preset_tol,baryon_density_sat,mass_args,eos_args):
#     baryon_density=baryon_density_sat/1.05**np.linspace(0,100,201)
#     eos_array=[]
#     init=init0
#     success=True
#     for i in range(len(baryon_density)):
#         tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
#         eos_array.append([baryon_density[i]]+tmp[2:])
#         init=tmp[1]
#         success=success and tmp[0]
#     eos_array.append([0.,0.,0.])
#     eos_array.append(list(2*np.array(eos_array[-1])-np.array(eos_array[-2])))
#     eos_array=list(reversed(eos_array))
# 
#     sol_saturation=np.array(eos_array[-1])
#     init = init0
#     baryon_density=baryon_density_sat*1.05**np.linspace(0,50,201)
#     #init_prev = eos_pressure_density(baryon_density[0]**2/baryon_density[1],init,Preset_tol,[mass_args,eos_args])[0]
#     for i in range(1,len(baryon_density)):
#         #tmp=eos_pressure_density(baryon_density[i],2*init-init_prev,Preset_tol,[mass_args,eos_args])
#         tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
#         eos_array.append([baryon_density[i]]+tmp[2:])
#         #init_prev=init
#         init=tmp[1]
#         success=success and tmp[0]
#         if(not success):
#             break
#     
#     eos_array=np.array(eos_array).transpose()
#     #print eos_array
#     #positive_pressure=eos_array[2][102:].min()>0
#     positive_pressure=eos_array[2][2:].min()>0
#     #if(positive_pressure and not stability):
#     #plt.plot(toMevfm(eos_array[0],'mev4'),toMevfm(eos_array[1],'mev4'))
#     #plt.xlim(0.0,0.3)
#     #plt.ylim(-2,40)
#     return toMevfm(eos_array,'mev4'),toMevfm(sol_saturation,'mev4'),success,positive_pressure
# 
# 
# 
# 
# from scipy.misc import derivative
# import scipy.constants as const
# from scipy.interpolate import interp1d
# from eos_class import EOS_BPS,match_eos,match_get_eos_array
# dlnx_cs2=1e-6
# class EOS_SLY4_match_RMF(object):
#     match_init=[None,None] #not in use
#     def __init__(self,eos_args_args):
#         self.eos_args,args=eos_args_args
#         self.m_eff,self.J,self.self_W,self.L=args
#         self.baryon_density_s=baryon_density_s_MeV4
#         self.bd_energy=m-BindE
#         self.incompressibility=K
#         self.mass_args=mass_args
#         self.args=self.baryon_density_s,self.bd_energy,self.incompressibility,\
#         self.m_eff,self.J,self.L,self.self_W,self.mass_args
#         k_F=(3*np.pi**2*self.baryon_density_s)**(1./3.)
#         W_0=self.bd_energy-(k_F**2+self.m_eff**2)**0.5
#         init_sat=(self.m_eff,W_0,k_F)
#         self.eos_array_rmf,self.sol_saturation,self.success,self.positive_pressure=get_eos_array(init_sat,Preset_tol,self.baryon_density_s,self.mass_args,self.eos_args)
#         self.baryon_density_s=self.baryon_density_s
#         self.pressure_s=self.sol_saturation[2]
#         self.density_s=self.sol_saturation[1]
#         self.n1_match=0.06
#         self.n2_match=0.16
#         p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
#         e1=EOS_BPS.eosDensity(p1)
#         dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
#         p2=self.pressure_s
#         e2=self.density_s
#         dpdn2=dpdn1#this parameter is not used in match_eos, so it was trivially set to dpdn1
#         self.p_match=p1
#         self.matching_args=self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init
#         self.matching_para,self.matching_success=match_eos(self.matching_args)
#         self.eos_success=self.success and self.positive_pressure and self.matching_success
#         if(self.matching_success):
#             u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,81)
#             eos_array_match=match_get_eos_array(u_array_low,self.matching_para)
#         else:
#             eos_array_match=[[self.n1_match,self.n2_match],[e1,e2],[p1,p2]]
#         self.eos_array=np.concatenate((np.array([[0],[0],[0]]),np.array(eos_array_match)[:,:-1],self.eos_array_rmf[:,203:]),axis=1)
#         self.unit_mass=const.c**4/(const.G**3*self.density_s*1e51*const.e)**0.5
#         self.unit_radius=const.c**2/(const.G*self.density_s*1e51*const.e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#         self.eosPressure_frombaryon_matchRMF = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosDensity_matchRMF  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity_matchRMF = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         for dict_intepolation in ['eosPressure_frombaryon_matchRMF','eosDensity_matchRMF','eosBaryonDensity_matchRMF']:
#             del state[dict_intepolation]
#         return state
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.eosPressure_frombaryon_matchRMF = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosDensity_matchRMF  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity_matchRMF = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
#     def eosPressure_frombaryon(self,baryon_density):
#         return np.where(baryon_density<self.n1_match,EOS_BPS.eosPressure_frombaryon(baryon_density),self.eosPressure_frombaryon_matchRMF(baryon_density))
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS.eosDensity(pressure),self.eosDensity_matchRMF(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS.eosBaryonDensity(pressure),self.eosBaryonDensity_matchRMF(pressure))
#     def eosCs2(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS().eosCs2(pressure),1.0/derivative(self.eosDensity_matchRMF,pressure,dx=pressure*dlnx_cs2))
#     def eosChempo(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS().eosChempo(pressure),(pressure+self.eosDensity_matchRMF(pressure))/self.eosBaryonDensity_matchRMF(pressure))
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#         self.maxmass_success=self.cs2_max<1 and self.mass_max>2
#         self.eos_success_all=self.maxmass_success and self.eos_success
#         return self.eos_success_all
# 
# def Calculation_creat_eos_RMF(eos_args_args):
#     return EOS_SLY4_match_RMF((eos_args_args[:7],eos_args_args[7:]))
# 
# f_eos_RMF='./'+dir_name+'/RMF_eos.dat'
# error_log=path+dir_name+'/error.log'
# eos_flat=main_parallel(Calculation_creat_eos_RMF,np.concatenate((eos_args_flat,args_flat),axis=1),f_eos_RMF,error_log)
# 
# eos_success=[]
# success=[]
# matching_success=[]
# positive_pressure=[]
# for eos_i in eos_flat:
#     success.append(eos_i.success)
#     matching_success.append(eos_i.matching_success)
#     eos_success.append(eos_i.eos_success)
#     positive_pressure.append(eos_i.positive_pressure)
# eos_success=np.array(eos_success)
# success=np.array(success)
# matching_success=np.array(matching_success)
# positive_pressure=np.array(positive_pressure)
# print('len(eos)=%d'%len(eos_success))
# print('len(eos[success])=%d'%len(success[success]))
# print('len(eos[positive_pressure])=%d'%len(positive_pressure[positive_pressure]))
# print('len(eos[matching_success])=%d'%len(matching_success[matching_success]))
# print('len(eos[eos_success])=%d'%len(eos_success[eos_success]))
# 
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# from show_properity import show_eos
# show_eos(axes,eos_flat,0,1,500,pressure_range=[0.01,30,'log'])
# =============================================================================

