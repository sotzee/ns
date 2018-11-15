#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:53:12 2018

@author: sotzee
"""
import scipy.optimize as opt
import numpy as np
from unitconvert import toMevfm,toMev4
Preset_tol=1e-30

Preset_Pressure_final=1e-8
Preset_rtol=1e-4

# =============================================================================
# def equations_isoscalar(x,args):
#     baryon_density_sat,bd_energy,incompressibility,m_eff,mass_args=args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g_Phi,g_W,b,c,self_W=x
#     k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
#     E_F=(k_F**2+m_eff**2)**0.5
#     Phi_0=m-m_eff
#     W_0=bd_energy-E_F
#     n_scalar_sym=(m_eff/np.pi**2)*(E_F*k_F-m_eff**2*np.log((k_F+E_F)/m_eff))
#     eq1=m*((m_Phi/g_Phi)**2*Phi_0 + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar_sym)
#     eq2=m*((m_W/g_W)**2*W_0 + (self_W/6)*W_0**3 - baryon_density_sat)
#     tmp_2=(g_W**2/(m_W**2+self_W/2*g_W**2*baryon_density_iW_0**2))
#     tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
#     eq3=m**2*(E_F**2*tmp_1*((m_Phi/g_Phi)**2+2*b*m*Phi_0+3*c*Phi_0**2+(1/np.pi**2)*(k_F/E_F*(E_F**2+2*m_eff**2)-3*m_eff**2*np.log((k_F+E_F)/m_eff)))+m_eff**2)
#     eq4=(1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi_0*m_Phi/g_Phi)**2/2 + (m*b)*Phi_0**3/3 + c*Phi_0**4/4 -(W_0*m_W/g_W)**2/2 - (self_W/6)*W_0**4/4 - baryon_density_sat*E_F
#     eq5=self_W
#     return eq1,eq2,eq3,eq4,eq5
# 
# def energy_density_sym(n,mass_args,isoscalar_args):
#     g_Phi,g_W,b,c,self_W=isoscalar_args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     k_F=(3*np.pi**2*n/2)**(1./3.)
#     def equations_sym(x):
#         m_eff,W=x
#         E_F=(k_F**2+m_eff**2)**0.5
#         n_scalar_sym=(m_eff/np.pi**2)*(E_F*k_F-m_eff**2*np.log((k_F+E_F)/m_eff))
#         Phi = m-m_eff
#         eq1=m*((m_Phi/g_Phi)**2*Phi + (m*b)*Phi**2 + c*Phi**3 - n_scalar_sym)
#         eq2=m*((m_W/g_W)**2*W + (self_W/6)*W**3 - n)
#         return eq1,eq2baryon_density_i
#     init=[0.7*m,200,]
#     sol = opt.root(equations_sym,init,tol=1e-10)
#     m_eff,W=sol
#     E_F=(k_F**2+m_eff**2)**0.5
#     Phi = m-m_eff
#     return (1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi*m_Phi/g_Phi)**2/2 + (m*b)*Phi**3/3 + c*Phi**4/4 -(W*m_W/g_W)**2/2 - (self_W/6)*W**4/4 + n*W
# 
# def energy_density_neu(n,mass_args,isoscalar_args,isovector_args):
#     g_Phi,g_W,b,c,self_W=isoscalar_args
#     g_rho,Lambda=isovector_args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     k_F=(3*np.pi**2*n/2)**(1./3.)
#     def equations_sym(x):
#         m_eff,W=x
#         E_F=(k_F**2+m_eff**2)**0.5
#         n_scalar_sym=(m_eff/np.pi**2)*(E_F*k_F-m_eff**2*np.log((k_F+E_F)/m_eff))
#         Phi = m-m_eff
#         eq1=m*((m_Phi/g_Phi)**2*Phi + (m*b)*Phi**2 + c*Phi**3 - n_scalar_sym)
#         eq2=m*((m_W/g_W)**2*W + (self_W/6)*W**3 - n)
#         return eq1,eq2
#     init=[0.7*m,200,]
#     sol = opt.root(equations_sym,init,tol=1e-10)
#     m_eff,W=sol
#     E_F=(k_F**2+m_eff**2)**0.5
#     Phi = m-m_eff
#     return (1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi*m_Phi/g_Phi)**2/2 + (m*b)*Phi**3/3 + c*Phi**4/4 -(W*m_W/g_W)**2/2 - (self_W/6)*W**4/4 + n*W
# =============================================================================



def equations(x,args):
    baryon_density_sat,bd_energy,incompressibility,m_eff,J,L,self_W,mass_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g_Phi,g_W,g_rho,b,c,Lambda=x
    k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    W_0=bd_energy-E_F
    n_scalar_sym=(m_eff/np.pi**2)*(E_F*k_F-m_eff**2*np.log((k_F+E_F)/m_eff))
    eq1=m*((m_Phi/g_Phi)**2*Phi_0 + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar_sym)
    eq2=m*((m_W/g_W)**2*W_0 + (self_W/6)*W_0**3 - baryon_density_sat)
    tmp_2=(g_W**2/(m_W**2+self_W/2*g_W**2*W_0**2))
    tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
    eq3=m**2*(E_F**2*tmp_1*((m_Phi/g_Phi)**2+2*b*m*Phi_0+3*c*Phi_0**2+(1/np.pi**2)*(k_F/E_F*(E_F**2+2*m_eff**2)-3*m_eff**2*np.log((k_F+E_F)/m_eff)))+m_eff**2)
    eq4=(1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi_0*m_Phi/g_Phi)**2/2 + (m*b)*Phi_0**3/3 + c*Phi_0**4/4 -(W_0*m_W/g_W)**2/2 - (self_W/6)*W_0**4/4 - baryon_density_sat*E_F
    tmp_J_0=k_F**2/(6*E_F)
    tmp_J_1=baryon_density_sat*g_rho**2/(8*(m_rho)**2+16*Lambda*(W_0*g_rho)**2) #origin fomula was baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*(W_0)**2)
    eq5=m**3*(tmp_J_0+tmp_J_1-J)
    eq6=m**3*(tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)-L)
    return eq1,eq2,eq3,eq4,eq5,eq6

# =============================================================================
# def equations(x,args):#correct the error
#     baryon_density_sat,bd_energy,incompressibility,m_eff_sym,J,L,mass_args=args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g_Phi,g_W,g_rho,b,c,Lambda,self_W=x
# # =============================================================================
# #     baryon_density_sat,bd_energy,incompressibility,m_eff,mass_args=args
# #     m,m_Phi,m_W=mass_args
# #     g_Phi,g_W,b,c=x
# # =============================================================================
#     k_F_sym=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
#     k_F_n  =(3*np.pi**2*baryon_density_sat)**(1./3.)
#     E_F_sym=(k_F_sym**2+m_eff_sym**2)**0.5
#     E_F_n  =(k_F_n  **2+m_eff_n  **2)**0.5
#     Phi_0_sym=m-m_eff_sym
#     Phi_0_n  =m-m_eff_n
#     W_0_sym=bd_energy-E_F_sym
#     #print(W_0,bd_energy,E_F,k_F,baryon_density_sat)
#     n_scalar_sym=(m_eff_sym/np.pi**2)*(E_F_sym*k_F_sym-m_eff_sym**2*np.log((k_F_sym+E_F_sym)/m_eff_sym))
#     eq1=m*((m_Phi/g_Phi)**2*Phi_0 + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar_sym)
#     eq2=m*((m_W/g_W)**2*W_0 + (self_W/6)*W_0**3 - baryon_density_sat)
#     tmp_2=(g_W**2/(m_W**2+self_W/2*g_W**2*W_0**2))
#     tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
#     eq3=m**2*(E_F**2*tmp_1*((m_Phi/g_Phi)**2+2*b*m*Phi_0+3*c*Phi_0**2+(1/np.pi**2)*(k_F/E_F*(E_F**2+2*m_eff**2)-3*m_eff**2*np.log((k_F+E_F)/m_eff)))+m_eff**2)
#     
#     eq4=(1./(4*np.pi**2))*(k_F*E_F*(E_F**2+k_F**2)-m_eff**4*np.log((k_F+E_F)/m_eff))+ (Phi_0*m_Phi/g_Phi)**2/2 + (m*b)*Phi_0**3/3 + c*Phi_0**4/4 -(W_0*m_W/g_W)**2/2 - (self_W/6)*W_0**4/4 - baryon_density_sat*E_F
#     tmp_J_0=k_F**2/(6*E_F)
#     tmp_J_1=baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*W_0**2)
#     eq5=m**3*(tmp_J_0+tmp_J_1-J)
#     eq6=m**3*(tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)-L)
#     eq7=m**4*self_W
#     return eq1,eq2,eq3,eq4,eq5,eq6,eq7
# =============================================================================

#check with W.C. Chen and J. Piekarewicz 2014   NL3 sets
sol = opt.root(equations,[10.,12.,9.,0.002,-0.002,0],tol=1e-30,args=[toMev4(0.1481,'mevfm'),939-16.24,271.5,0.595*939,37.28,118.2,0.,(0.5109989461,939,508.194,782.501,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.1481,'mevfm'),939-16.24,271.5,0.595*939,37.28,118.2,0.,(0.5109989461,939,508.194,782.501,763)]))

#check with W.C. Chen and J. Piekarewicz 2014   FSU sets
sol = opt.root(equations,[10.5,14.,12.,0.002,-0.002,0],tol=1e-30,args=[toMev4(0.1484,'mevfm'),939-16.3,230.0,0.61*939,32.59,60.5,0.06,(0.5109989461,939,491.5,782.500,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.1484,'mevfm'),939-16.3,230.0,0.61*939,32.59,60.5,0.06,(0.5109989461,939,491.5,782.500,763)]))

#check with Nadine Hornick et. al. 2018
sol = opt.root(equations,[10.,12.,9.,0.002,-0.002,0],tol=1e-30,args=[toMev4(0.15,'mevfm'),939-16,240,0.65*939,30,50,0.,(0.5109989461,939,550,783,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.15,'mevfm'),939-16,240,0.65*939,30,50,0.,(0.5109989461,939,550,783,763)]))
eos_args=sol.x

sol = opt.root(equations,[10.,12.,9.,0.002,-0.002,0],tol=1e-30,args=[toMev4(0.15,'mevfm'),939-16,240,0.60*939,30,50,0.,(0.5109989461,939,550,783,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.15,'mevfm'),939-16,240,0.60*939,30,50,0.,(0.5109989461,939,550,783,763)]))
eos_args=sol.x

# =============================================================================
# def eos_equations(y,args):
#     n,mass_args,eos_args=args
#     m_e,m,m_Phi,m_W,m_rho=mass_args
#     g_Phi,g_W,g_rho,b,c,Lambda,self_W=eos_args
#     m_eff,Phi_0,W_0,rho_0,k_F_n,energy_density,pressure=y
#     
#     n_n=k_F_n**3/(3*np.pi**2)
#     n_p=n-n_n
#     n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
#     n3=n_n-n_p
#     n_e=n_p
#     k_F_p=(3*np.pi**2*n_p)**(1./3)
#     k_F_e=k_F_p
#     E_F_e=(k_F_e**2+m_e**2)**0.5
#     E_F_p=(k_F_p**2+m_eff**2)**0.5
#     E_F_n=(k_F_n**2+m_eff**2)**0.5
#     n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p-m_eff**2*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
#     eq1=m_eff-m+Phi_0
#     eq2=m*((m_Phi/g_Phi)**2*Phi_0 + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
#     eq3=m*((m_W/g_W)**2*W_0 + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
#     eq4=m*((m_rho/g_rho)**2*rho_0 + 2*Lambda*W_0**2*rho_0 - n3/2)
#     eq5=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0/g_Phi)**2/2+(m_W*W_0/g_W)**2/2+(m_rho*rho_0/g_rho)**2/2+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2-energy_density
#     chempo_e=E_F_e
#     chempo_p=E_F_p+W_0-rho_0/2
#     chempo_n=E_F_n+W_0+rho_0/2
#     eq6=chempo_e+chempo_p-chempo_n
#     eq7=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density-pressure
#     return eq1,eq2,eq3,eq4,eq5,eq6,eq7
# =============================================================================

def eos_equations(y,args):
    n,mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g_Phi,g_W,g_rho,b,c,Lambda,self_W=eos_args
    m_eff,W_0,k_F_n=y
    
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
    n3=n_n-n_p
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5

    if(m_eff<=0):
        n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p)+(E_F_n*k_F_n))
    else:
        n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p-m_eff**2*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi/g_Phi)**2*Phi_0 + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho/g_rho)**2 + 2*Lambda*W_0**2)
    eq3=m*((m_W/g_W)**2*W_0 + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    #eq5=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0/g_Phi)**2/2+(m_W*W_0/g_W)**2/2+(m_rho*rho_0/g_rho)**2/2+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2-energy_density
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0-rho_0/2
    chempo_n=E_F_n+W_0+rho_0/2
    eq6=chempo_e+chempo_p-chempo_n
    #eq7=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density-pressure
    return eq2,eq3,eq6

def eos_pressure_density(n,init,Preset_tol,args):
    mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g_Phi,g_W,g_rho,b,c,Lambda,self_W=eos_args
    sol = opt.root(eos_equations,init,tol=Preset_tol,args=[n,mass_args,eos_args])
    m_eff,W_0,k_F_n=sol.x
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n_e=n_p
    n3=n_n-n_p
    n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho/g_rho)**2 + 2*Lambda*W_0**2)
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0-rho_0/2
    chempo_n=E_F_n+W_0+rho_0/2
    energy_density=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0/g_Phi)**2/2+(m_W*W_0/g_W)**2/2+(m_rho*rho_0/g_rho)**2/2+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density
    return [sol.x,energy_density,pressure]

def get_eos_array(init0,Preset_tol,baryon_density_sat,mass_args,eos_args):
    baryon_density=baryon_density_sat/1.05**np.linspace(0,100,101)
    eos_array=[]
    init_sat=eos_pressure_density(baryon_density_sat,init0,Preset_tol,[mass_args,eos_args])[0]
    init=init_sat
    for i in range(len(baryon_density)):
        tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
        eos_array.append([baryon_density[i]]+tmp[1:])
        init=tmp[0]
        #print eos_array[i][2],eos_array[i-1][2]
    eos_array.append([0.,0.,0.])
    eos_array.append(list(2*np.array(eos_array[-1])-np.array(eos_array[-2])))
    eos_array=list(reversed(eos_array))

    sol_saturation=eos_array[-1]
    init = init_sat
    baryon_density=baryon_density_sat*1.05**np.linspace(0,50,101)
    stability=True
    for i in range(1,len(baryon_density)):
        tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
        eos_array.append([baryon_density[i]]+tmp[1:])
        init=tmp[0]
        stability=stability and eos_array[-2][2]<eos_array[-1][2]<2*eos_array[-2][2]
        #print init
    
    eos_array=np.array(eos_array).transpose()
    #print eos_array
    positive_pressure=eos_array[2][102:].min()>0
    #if(positive_pressure and not stability):
    #plt.plot(toMevfm(eos_array[0],'mev4'),toMevfm(eos_array[1],'mev4'))
    #plt.xlim(0.0,0.3)
    #plt.ylim(-2,40)
    return init_sat,eos_array,sol_saturation,stability,positive_pressure

from scipy.misc import derivative
from scipy.constants import c,G,e
from scipy.interpolate import interp1d
dlnx_cs2=1e-6
class EOS_RMF(object):
    def __init__(self,init_args,init_sat,args):
        self.baryon_density_sat,self.bd_energy,self.incompressibility,\
        self.m_eff,self.J,self.L,self.self_W,self.mass_args=args
        self.args=args
        sol_parameters = opt.root(equations,init_args[:6],tol=Preset_tol,args=self.args)
        self.eos_args=list(sol_parameters.x)+[self.self_W]
        #init_saturation = [6.2e+02,3.2e+02,2.6e+02,4.9e+01,3.2e+02,1.1e+09,1.8e+07]
        self.init_sat,self.eos_array,self.sol_saturation,self.stability,self.positive_pressure=get_eos_array(init_sat,Preset_tol,self.baryon_density_sat,self.mass_args,self.eos_args)
        print(self.stability,self.positive_pressure)
# =============================================================================
#         plt.plot(toMevfm(self.eos_array[6],'mev4'),toMevfm(self.eos_array[7],'mev4'))
#         print(toMevfm(self.eos_array[7],'mev4'))
#         plt.xlim(0,500)
#         plt.ylim(0,5)
# =============================================================================
        self.baryon_density_s=toMevfm(self.baryon_density_sat,'mev4')
        self.pressure_s=toMevfm(self.sol_saturation[2],'mev4')
        self.density_s=toMevfm(self.sol_saturation[1],'mev4')
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        self.eosPressure_frombaryon = interp1d(toMevfm(self.eos_array[0],'mev4'),toMevfm(self.eos_array[2],'mev4'), kind='linear')
        self.eosPressure = interp1d(toMevfm(self.eos_array[1],'mev4'),toMevfm(self.eos_array[2],'mev4'), kind='linear')
        self.eosDensity  = interp1d(toMevfm(self.eos_array[2],'mev4'),toMevfm(self.eos_array[1],'mev4'), kind='linear')
        self.eosBaryonDensity = interp1d(toMevfm(self.eos_array[2],'mev4'),toMevfm(self.eos_array[0],'mev4'), kind='linear')
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon','eosPressure','eosDensity','eosBaryonDensity']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eosPressure_frombaryon = interp1d(toMevfm(self.eos_array[0],'mev4'),toMevfm(self.eos_array[2],'mev4'), kind='linear')
        self.eosPressure = interp1d(toMevfm(self.eos_array[1],'mev4'),toMevfm(self.eos_array[2],'mev4'), kind='linear')
        self.eosDensity  = interp1d(toMevfm(self.eos_array[2],'mev4'),toMevfm(self.eos_array[1],'mev4'), kind='linear')
        self.eosBaryonDensity = interp1d(toMevfm(self.eos_array[2],'mev4'),toMevfm(self.eos_array[0],'mev4'), kind='linear')
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

from eos_class import EOS_BPS,EOS_BPSwithPoly
class EOS_SLY4withRMF(object):
    def __init__(self,init_args,init_sat,args):
        self.baryon_density_sat,self.bd_energy,self.incompressibility,\
        self.m_eff,self.J,self.L,self.self_W,self.mass_args=args
        self.args=args
        self.eos_RMF=EOS_RMF(init_args,init_sat,args)
        self.eos_args=self.eos_RMF.eos_args
        self.init_sat=self.eos_RMF.init_sat
        self.eos_array=self.eos_RMF.eos_array
        self.sol_saturation=toMevfm(np.array(self.eos_RMF.sol_saturation),'mev4')
        fix_crust_baryon_density=np.linspace(0.6,0.3,4)*self.sol_saturation[0]
        self.fix_crust_logic=False
        for fix_crust_baryon_density_i in fix_crust_baryon_density:
            if(self.sol_saturation[2]>1.1*EOS_BPS.eosPressure_frombaryon(fix_crust_baryon_density_i)):
                self.eos_SLY4withPoly=EOS_BPSwithPoly([fix_crust_baryon_density_i,self.sol_saturation[2],self.sol_saturation[0],4*self.sol_saturation[2],2*self.sol_saturation[0],8*self.sol_saturation[2],3*self.sol_saturation[0]])
                self.fix_crust_logic=True
                break
            
        self.stability=self.eos_RMF.stability
        self.positive_pressure=self.eos_RMF.positive_pressure
        self.baryon_density_s=self.eos_RMF.baryon_density_s
        self.pressure_s=self.eos_RMF.pressure_s
        self.density_s=self.eos_RMF.density_s
        self.unit_mass=self.eos_RMF.unit_mass
        self.unit_radius=self.eos_RMF.unit_radius
        self.unit_N=self.eos_RMF.unit_N
    def __getstate__(self):
        state_RMF=self.eos_RMF.__getstate__()
        state = self.__dict__.copy()
        return (state,state_RMF)
    def __setstate__(self, state_):
        state,state_RMF=state_
        self.__dict__.update(state)
        self.eos_RMF.__setstate__(state_RMF)
    def eosDensity(self,pressure):
        return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosDensity(pressure),self.eos_RMF.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosBaryonDensity(pressure),self.eos_RMF.eosBaryonDensity(pressure))
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)


# =============================================================================
# m_eff=[0.55*939]
# J=[30]
# L=[40]
# EOS_RMF([toMev4(0.15,'mevfm'),939-16,240,m_eff[0],J[0],L[0],(0.5109989461,939,550,783,763)])
# =============================================================================

from Lambda_hadronic_calculation import Maxmass,MassRadius,Properity_ofmass
def Calculation_maxmass(eos_i):
    try:
        if(eos_i.positive_pressure):
            maxmass_result=Maxmass(Preset_Pressure_final,Preset_rtol,eos_i)[1:3]
            maxmass_result+=[eos_i.eosCs2(maxmass_result[0])]
        else:
            maxmass_result=[0,3.,1.]
    except RuntimeWarning:
        print('Runtimewarning happens at calculating max mass:')
        print(eos_i.args)
    return maxmass_result

def Calculation_onepointfour(eos_i):
    try:
        if(eos_i.positive_pressure):
            Properity_onepointfour=Properity_ofmass(1.4,10,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
        else:
            Properity_onepointfour=[0,1.4,15,1,0,1400]
    except RuntimeWarning:
        print('Runtimewarning happens at calculating Calculation_onepointfour:')
        print(eos_i.args)
    return Properity_onepointfour



path = "./"
dir_name='Lambda_RMF_calculation_parallel'
import cPickle,os
if __name__ == '__main__':
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)
    print('main calculation starts here:')
    J=30
    eos_rmf=[]
    baryon_density_s=0.15
    args=np.mgrid[0.5*939:0.8*939:4j,0:0.05:6j,80:30:6j]
    init_args= [12.124200434658492, 14.358205178844612, 9.2022483593647681, 0.0018220954984504475, -0.0027122507224894791, 0.028295688189677305, 0]
    init_sat = [484.27504990105365, 378.79614094279168, 319.23744720471598]
    m_eff,self_W,L=args
    args_shape=np.shape(m_eff)
    f_file=open(path+dir_name+'/Lambda_RMF_calculation_args.dat','wb')
    cPickle.dump(args,f_file)
    f_file.close()

    for i in range(args_shape[0]):
        eos_rmf.append([])
        for j in range(args_shape[1]):
            eos_rmf[i].append([])
            for k in range(args_shape[2]):
                #print m_eff[i],L[k],self_W[j]
                #print m_eff_[i,j,k],L_[i,j,k],self_W_[i,j,k]
                try:
                    eos_rmf[i][j].append(EOS_SLY4withRMF(eos_rmf[i][j][k-1].eos_args,eos_rmf[i][j][k-1].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
                except:
                    try:
                        eos_rmf[i][j].append(EOS_SLY4withRMF(eos_rmf[i][j-1][k].eos_args,eos_rmf[i][j-1][k].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
                    except:
                        try:
                            eos_rmf[i][j].append(EOS_SLY4withRMF(eos_rmf[i-1][j][k].eos_args,eos_rmf[i-1][j][k].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
                        except:
                            eos_rmf[i][j].append(EOS_SLY4withRMF(init_args,init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))

    eos_flat=np.array(eos_rmf).flatten  ()        
    eos_rmf=np.reshape(np.array(eos_rmf),args_shape)
    print('%d EoS built with shape (m_eff,self_W,L)%s.'%(np.size(eos_rmf),np.shape(eos_rmf)))
    
    error_log=path+dir_name+'/error.log'
    from Parallel_process import main_parallel
    f_maxmass_result='./'+dir_name+'/Lambda_RMF_calculation_maxmass.dat'
    maxmass_result=main_parallel(Calculation_maxmass,eos_flat,f_maxmass_result,error_log)
    print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat)))
    
    logic_maxmass=maxmass_result[:,1]>=2
    for i in range(len(eos_flat)):
        eos_flat[i].setMaxmass(maxmass_result[i])
    
    
# =============================================================================
#     import matplotlib.pyplot as plt
#     def show_eos(eos,x_index,y_index,baryon_density_range,N):#index baryon_density(0), pressure(1), energy density(2), energy per baryon(3), chempo(4)
#         pressure_density_energyPerBaryon_chempo=[]
#         for eos_i in eos:
#             baryon_density_i=np.linspace(baryon_density_range[0],baryon_density_range[1],N)
#             pressure_density_energyPerBaryon_chempo_i=[]
#             pressure_density_energyPerBaryon_chempo_i.append(baryon_density_i)
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosPressure_frombaryon(baryon_density_i))
#             pressure_density_energyPerBaryon_chempo_i.append(eos_i.eosDensity(pressure_density_energyPerBaryon_chempo_i[1]))
#             pressure_density_energyPerBaryon_chempo_i.append(pressure_density_energyPerBaryon_chempo_i[2]/baryon_density_i)
#             pressure_density_energyPerBaryon_chempo_i.append((pressure_density_energyPerBaryon_chempo_i[1]+pressure_density_energyPerBaryon_chempo_i[2])/baryon_density_i)
#             pressure_density_energyPerBaryon_chempo.append(pressure_density_energyPerBaryon_chempo_i)
#             plt.plot(pressure_density_energyPerBaryon_chempo_i[x_index],pressure_density_energyPerBaryon_chempo_i[y_index])
#         pressure_density_energyPerBaryon_chempo=np.array(pressure_density_energyPerBaryon_chempo)
#         label_text=['Baryon density(fm$^{-3}$)','Pressure(MeV fm$^{-3}$)','Energy density(MeV fm$^{-3}$)','Energy per baryon(MeV)','Chemical potential(MeV)']
#         plt.xlabel(label_text[x_index])
#         plt.ylabel(label_text[y_index])
#         #plt.xlim(pressure_density_energyPerBaryon_chempo[:,x_index,:].min(),pressure_density_energyPerBaryon_chempo[:,x_index,:].max())
#         #plt.ylim(pressure_density_energyPerBaryon_chempo[:,y_index,:].min(),pressure_density_energyPerBaryon_chempo[:,y_index,:].max())
# 
#     logic_stability=[]
#     logic_positive_pressure=[]
#     radius_onepointfour=[]
#     ii=0
#     for i in range(len(m_eff)):
#         for j in range(len(self_W)):
#             for k in range(len(L)):
#                 logic_stability.append(eos_rmf[i][j][k].stability)
#                 logic_positive_pressure.append(eos_rmf[i][j][k].positive_pressure)
#     logic_stability=np.reshape(np.array(logic_stability),(len(m_eff),len(self_W),len(L)))
#     logic_positive_pressure=np.reshape(np.array(logic_positive_pressure),(len(m_eff),len(self_W),len(L)))
#     maximum_mass=np.reshape(np.array(maxmass_result[:,1]),(len(m_eff),len(self_W),len(L)))
#     logic_maximum_mass=maximum_mass>2.0
#     show_eos(eos_rmf[np.logical_and(logic_stability,np.logical_and(logic_positive_pressure,logic_maximum_mass))].flatten(),2,1,[0.00016,1.*0.16],100)
# =============================================================================

    f_onepointfour_result=path+dir_name+'/Lambda_RMF_calculation_onepointfour.dat'
    Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat[logic_maxmass],f_onepointfour_result,error_log)
    print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat[logic_maxmass])))

    f_file=open(path+dir_name+'/Lambda_RMF_calculation_eos.dat','wb')
    cPickle.dump(eos_rmf,f_file)
    f_file.close()
    
# =============================================================================
# p=10**np.linspace(-1,2,500)
# baryon_density_rmf=[[],[],[]]
# energy_density_rmf=[[],[],[]]
# for i in range(len(m_eff)):
#     for j in range(len(self_W)):
#         for k in range(len(L)):
#             for l in range(len(p)):
#                 baryon_density_rmf.append(eos_rmf[-1].eosBaryonDensity(p[l]))
#                 energy_density_rmf.append(eos_rmf[-1].eosDensity(p[l]))
# baryon_density_rmf=np.reshape(np.array(baryon_density_rmf),(len(m_eff),len(self_W),len(L),len(p)))
# energy_density_rmf=np.reshape(np.array(energy_density_rmf),(len(m_eff),len(self_W),len(L),len(p)))
# =============================================================================



# =============================================================================
# init_args= [12.,14.,12.,0.002,-0.003,0.000]
# init_sat = [6.2e+02,2.6e+02,3.2e+02]
# init_sat = [753.29362749, 134.09525771, 319.39600576]
# for i in range(len(m_eff)):
#     eos_rmf.append([])
#     p_s.append([])
#     p1.append([])
#     p2.append([])
#     p3.append([])
#     baryon_density_rmf.append([])
#     energy_density_rmf.append([])
#     for j in range(len(self_W)):
#         eos_rmf[i].append([])
#         p_s[i].append([])
#         p1[i].append([])
#         p2[i].append([])
#         p3[i].append([])
#         baryon_density_rmf[i].append([])
#         energy_density_rmf[i].append([])
#         for k in range(len(L)):
#             try:
#                 eos_rmf[i][j].append(EOS_RMF(eos_rmf[i][j][k-1].init_args,eos_rmf[i][j][k-1].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i],J,L[k],self_W[j],(0.5109989461,939,550,783,763)]))
#             except:
#                 try:
#                     eos_rmf[i][j].append(EOS_RMF(eos_rmf[i][j-1][k].init_args,eos_rmf[i][j-1][k].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i],J,L[k],self_W[j],(0.5109989461,939,550,783,763)]))
#                 except:
#                     try:
#                         eos_rmf[i][j].append(EOS_RMF(eos_rmf[i-1][j][k].init_args,eos_rmf[i-1][j][k].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i],J,L[k],self_W[j],(0.5109989461,939,550,783,763)]))
#                     except:
#                         eos_rmf[i][j].append(EOS_RMF(init_args,init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i],J,L[k],self_W[j],(0.5109989461,939,550,783,763)]))
#             #eos_rmf[i][j].append(EOS_RMF(init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i],J[j],L[k],0,(0.5109989461,939,550,783,763)]))
#             p_s[i][j].append(eos_rmf[i][j][k].eosPressure_frombaryon(baryon_density_s))
#             p1[i][j].append(eos_rmf[i][j][k].eosPressure_frombaryon(baryon_density1))
#             p2[i][j].append(eos_rmf[i][j][k].eosPressure_frombaryon(baryon_density2))
#             p3[i][j].append(eos_rmf[i][j][k].eosPressure_frombaryon(baryon_density3))
#             baryon_density_rmf[i][j].append([])
#             energy_density_rmf[i][j].append([])
#             for l in range(len(p)):
#                 baryon_density_rmf[i][j][k].append(eos_rmf[i][j][k].eosBaryonDensity(p[l]))
#                 energy_density_rmf[i][j][k].append(eos_rmf[i][j][k].eosDensity(p[l]))
#             baryon_density_rmf[i][j][k]=np.array(baryon_density_rmf[i][j][k])
#             energy_density_rmf[i][j][k]=np.array(energy_density_rmf[i][j][k])
# baryon_density_rmf=np.array(baryon_density_rmf)
# energy_density_rmf=np.array(energy_density_rmf)
#
# from eos_class import EOS_BPSwithPoly
# for i in range(len(m_eff)):
#     eos_poly.append([])
#     baryon_density_poly.append([])
#     energy_density_poly.append([])
#     for j in range(len(self_W)):
#         eos_poly[i].append([])
#         baryon_density_poly[i].append([])
#         energy_density_poly[i].append([])
#         for k in range(len(L)):
#             #eos_poly[i][j].append(EOS_BPSwithPoly([baryon_density0,p1[i][j][k],baryon_density1,p2[i][j][k],baryon_density2,p3[i][j][k],baryon_density3]))
#             gamma1=np.log(p_s[i][j][k]/pressure0)/np.log(baryon_density_s/baryon_density0)
#             eos_poly[i][j].append(EOS_BPSwithPoly([baryon_density0,p_s[i][j][k]*(baryon_density1/baryon_density_s)**gamma1,baryon_density1,p2[i][j][k],baryon_density2,p3[i][j][k],baryon_density3]))
#             baryon_density_poly[i][j].append([])
#             energy_density_poly[i][j].append([])
#             for l in range(len(p)):
#                 baryon_density_poly[i][j][k].append(eos_poly[i][j][k].eosBaryonDensity(p[l]))
#                 energy_density_poly[i][j][k].append(eos_poly[i][j][k].eosDensity(p[l]))
#             baryon_density_poly[i][j][k]=np.array(baryon_density_poly[i][j][k])                
#             energy_density_poly[i][j][k]=np.array(energy_density_poly[i][j][k])          
# baryon_density_poly=np.array(baryon_density_poly)                
# energy_density_poly=np.array(energy_density_poly)
# 
# plt.figure(figsize=(7,7))
# color_list=[['r','g','b'],['c','y','k']]
# plt.plot([], [], ' ', label='m$_{eff}$, $\\xi$, L (MeV)')
# for i in range(len(m_eff)):
#     for j in range(len(self_W)):
#         for k in range(len(L)):
#             if(i+j+k==0):
#                 plt.plot(baryon_density_poly[i][j][k],p,'k',label='piecewise poly')
#             else:
#                 plt.plot(baryon_density_poly[i][j][k],p,'k')
#             if(i==0):
#                 plt.plot(baryon_density_rmf[i][j][k],p,':',color=color_list[j][k],label='%.2f,%.2f,%d'%(m_eff[i]/939,self_W[j],L[k]))
#             elif(i==1):
#                 plt.plot(baryon_density_rmf[i][j][k],p,'--',color=color_list[j][k],label='%.2f,%.2f,%d'%(m_eff[i]/939,self_W[j],L[k]))
#             elif(i==2):
#                 plt.plot(baryon_density_rmf[i][j][k],p,'-.',color=color_list[j][k],label='%.2f,%.2f,%d'%(m_eff[i]/939,self_W[j],L[k]))
#             plt.plot([0.15],[toMevfm(np.array(eos_rmf[i][j][k].sol_saturation)[2],'mev4')],'o',color=color_list[j][k])
# plt.xlim(0.075,0.3)
# plt.ylim(-2,40)
# plt.xlabel('baryon density(fm$^{-3}$)',size=20)
# plt.ylabel('pressure(MeV fm$^{-3}$)',size=20)
# plt.legend(frameon=False,loc=2)
# 
# plt.figure(figsize=(7,7))
# color_list=[['r','g','b'],['c','y','k']]
# plt.plot([], [], ' ', label='m$_{eff}/m$, $\\xi$, L (MeV)')
# for i in range(len(m_eff)):
#     for j in range(len(self_W)):
#         for k in range(len(L)):
#             if(i+j+k==0):
#                 plt.plot(baryon_density_poly[i][j][k],energy_density_poly[i][j][k]/baryon_density_poly[i][j][k],'k',label='piecewise poly')
#             else:
#                 plt.plot(baryon_density_poly[i][j][k],energy_density_poly[i][j][k]/baryon_density_poly[i][j][k],'k')
#             if(i==0):
#                 plt.plot(baryon_density_rmf[i][j][k],energy_density_rmf[i][j][k]/baryon_density_rmf[i][j][k],':',color=color_list[j][k],label='%.2f,%.2f,%d'%(m_eff[i]/939,self_W[j],L[k]))
#             elif(i==1):
#                 plt.plot(baryon_density_rmf[i][j][k],energy_density_rmf[i][j][k]/baryon_density_rmf[i][j][k],'--',color=color_list[j][k],label='%.2f,%.2f,%d'%(m_eff[i]/939,self_W[j],L[k]))
#             elif(i==2):
#                 plt.plot(baryon_density_rmf[i][j][k],energy_density_rmf[i][j][k]/baryon_density_rmf[i][j][k],'-.',color=color_list[j][k],label='%.2f,%.2f,%d'%(m_eff[i]/939,self_W[j],L[k]))
#             plt.plot([0.15],[toMevfm(np.array(eos_rmf[i][j][k].sol_saturation)[1],'mev4')/0.15],'o',color=color_list[j][k])
# plt.plot(np.zeros(10)+0.2,np.linspace(954,966,10),lw=10)
# plt.plot(np.zeros(10)+0.5,np.linspace(984,1059,10),lw=10)
# plt.xlim(0.075,0.3)
# plt.ylim(945,980)
# plt.xlabel('baryon density(fm$^{-3}$)',size=20)
# plt.ylabel('energy per baryon(MeV)',size=20)
# plt.legend(frameon=False,loc=2)
# =============================================================================

# =============================================================================
# from tov_f import MassRadius
# pc=np.linspace(10,500,500)
# mr_rmf=[[],[],[]]
# mr_poly=[[],[],[]]
# color_list=['r','g','b']
# for i in range(len(pc)):
#     print(pc[i])
#     for j in range(len(mr_rmf)):
#         mr_rmf[j].append(MassRadius(pc[i],1e-5,1e-4,'MRBIT',eos_rmf[j][1][2]))
# mr_rmf=np.array(mr_rmf)
# 
# for i in range(len(pc)):
#     for k in range(len(mr_poly)):
#         mr_poly[k].append(MassRadius(pc[i],1e-5,1e-4,'MRBIT',eos_poly[k][1][2]))
# mr_poly=np.array(mr_poly)
# 
# for i in range(len(mr_rmf)):
#     plt.plot(np.array(mr_rmf[i])[:,1],np.array(mr_rmf[i])[:,0],color=color_list[i],label='m_$_{eff}$/m=%.1f'%(m_eff[i]/939))
# for i in range(len(mr_poly)):
#     plt.plot(np.array(mr_poly[i])[:,1],np.array(mr_poly[i])[:,0],':',color=color_list[i],label='mimic poly')
# plt.legend()
# plt.xlabel('radius(km)')
# plt.ylabel('mass(M$_\odot)')
# 
# for i in [0,2]:
#     plt.plot(np.array(mr_rmf[i])[:,0],np.array(mr_rmf[i])[:,-1]*np.array(mr_rmf[i])[:,2]**6,color=color_list[i],label='m_$_{eff}$/m=%.1f'%(m_eff[i]/939))
# plt.legend()
# plt.xlabel('mass(M$_\odot)')
# plt.ylabel('$\Lambda \\beta^6$')
# plt.xlim(1,1.8)
# =============================================================================
