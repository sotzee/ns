#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:20:55 2019

@author: sotzee
"""
import scipy.optimize as opt
import numpy as np
from unitconvert import toMevfm,toMev4
Preset_tol=1e-10

Preset_Pressure_final=1e-8
Preset_rtol=1e-4

def equations(x,args):
    baryon_density_sat,bd_energy,incompressibility,m_eff,J,L,self_W,mass_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda=x
    g2_Phi=np.max([0,g2_Phi])
    g2_W=np.max([0,g2_W])
    g2_rho=np.max([0,g2_rho])
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

init_args=np.array([10.,12.,9.,0.002,-0.002,0])**2
#check with W.C. Chen and J. Piekarewicz 2014   NL3 sets
sol = opt.root(equations,init_args,tol=1e-30,args=[toMev4(0.1481,'mevfm'),939-16.24,271.5,0.595*939,37.28,118.2,0.,(0.5109989461,939,508.194,782.501,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.1481,'mevfm'),939-16.24,271.5,0.595*939,37.28,118.2,0.,(0.5109989461,939,508.194,782.501,763)]))

#check with W.C. Chen and J. Piekarewicz 2014   FSU sets
sol = opt.root(equations,init_args,tol=1e-30,args=[toMev4(0.1484,'mevfm'),939-16.3,230.0,0.61*939,32.59,60.5,0.06,(0.5109989461,939,491.5,782.500,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.1484,'mevfm'),939-16.3,230.0,0.61*939,32.59,60.5,0.06,(0.5109989461,939,491.5,782.500,763)]))

#check with Nadine Hornick et. al. 2018
sol = opt.root(equations,init_args,tol=1e-30,args=[toMev4(0.15,'mevfm'),939-16,240,0.65*939,30,50,0.,(0.5109989461,939,550,783,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.15,'mevfm'),939-16,240,0.65*939,30,50,0.,(0.5109989461,939,550,783,763)]))
eos_args=sol.x

sol = opt.root(equations,init_args,tol=1e-30,args=[toMev4(0.15,'mevfm'),939-16,240,0.60*939,30,50,0.,(0.5109989461,939,550,783,763)])
print(sol.x)
print(equations(sol.x,[toMev4(0.15,'mevfm'),939-16,240,0.60*939,30,50,0.,(0.5109989461,939,550,783,763)]))
eos_args=sol.x

def power_with_sign(x,n):
    return np.sign(x)*np.abs(x)**n

def eos_equations(y,args):
    n,mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    m_eff,W_0,k_F_n=y
    
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    #n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
    n3=n_n-n_p
    #k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_p=power_with_sign(3*np.pi**2*n_p,1./3)
    k_F_e=k_F_p
    E_F_e=power_with_sign(power_with_sign(k_F_e,2)+m_e**2,0.5)
    E_F_p=power_with_sign(power_with_sign(k_F_p,2)+m_eff**2,0.5)
    E_F_n=power_with_sign(power_with_sign(k_F_n,2)+m_eff**2,0.5)

    if(m_eff<=0):
        n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p)+(E_F_n*k_F_n))
    else:
        n_scalar=(m_eff/(2*np.pi**2))*((E_F_p*k_F_p)+(E_F_n*k_F_n)-m_eff**2*(np.log((k_F_p+E_F_p)*(k_F_n+E_F_n))-2*np.log(m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0-rho_0/2
    chempo_n=E_F_n+W_0+rho_0/2
    eq6=(chempo_e+chempo_p-chempo_n)
    return eq2,eq3,eq6

def eos_pressure_density(n,init,Preset_tol,args):
    mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    sol = opt.root(eos_equations,init,tol=Preset_tol,args=[n,mass_args,eos_args],method='hybr')
    if(not (sol.success and 0.5*n<sol.x[2]**3/(3*np.pi**2)<n)): #0.5*n<sol.x[2]**3/(3*np.pi**2)<n
        for init_modify in np.mgrid[0.9:1.1:5j,0.9:1.1:5j,0.9:1.1:5j].reshape(3,125).transpose():
            sol = opt.root(eos_equations,init_modify*init,tol=Preset_tol,args=[n,mass_args,eos_args],method='hybr')
            if((sol.success and 0.5*n<sol.x[2]**3/(3*np.pi**2)<n)):
                break
    m_eff,W_0,k_F_n=sol.x
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
    n3=n_n-n_p
    n_e=n_p
    k_F_p=(3*np.pi**2*n_p)**(1./3)
    k_F_e=k_F_p
    E_F_e=(k_F_e**2+m_e**2)**0.5
    E_F_p=(k_F_p**2+m_eff**2)**0.5
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0-rho_0/2
    chempo_n=E_F_n+W_0+rho_0/2
    energy_density=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density
# =============================================================================
#     print n_e,n_p,n_n
#     print chempo_e*n_e,chempo_p*n_p,chempo_n*n_n
# =============================================================================
# =============================================================================
#     if(sol.success):
#         plt.plot([toMevfm(n,'mev4')],[toMevfm(pressure,'mev4')],'.')
# =============================================================================
    success=sol.success and 0.5*n<sol.x[2]**3/(3*np.pi**2)<n
    return [sol.x,success,energy_density,pressure]

def get_eos_array(init0,Preset_tol,baryon_density_sat,mass_args,eos_args):
    baryon_density=baryon_density_sat/1.05**np.linspace(0,100,201)
    eos_array=[]
    init_sat=eos_pressure_density(baryon_density_sat,init0,Preset_tol,[mass_args,eos_args])[0]
    init=init_sat
    for i in range(len(baryon_density)):
        tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
        eos_array.append([baryon_density[i]]+tmp[2:])
        init=tmp[0]
        #print eos_array[i][2],eos_array[i-1][2]
    eos_array.append([0.,0.,0.])
    eos_array.append(list(2*np.array(eos_array[-1])-np.array(eos_array[-2])))
    eos_array=list(reversed(eos_array))

    sol_saturation=eos_array[-1]
    init = init_sat
    baryon_density=baryon_density_sat*1.05**np.linspace(0,50,101)
    init_prev = eos_pressure_density(baryon_density[0]**2/baryon_density[1],init,Preset_tol,[mass_args,eos_args])[0]
    stability=True
    for i in range(1,len(baryon_density)):
        tmp=eos_pressure_density(baryon_density[i],2*init-init_prev,Preset_tol,[mass_args,eos_args])
        #print baryon_density[i],tmp
        eos_array.append([baryon_density[i]]+tmp[2:])
        init_prev=init
        init=tmp[0]
        success=stability and tmp[1]
# =============================================================================
#         if(not stability):
#             break
# =============================================================================
    
    eos_array=np.array(eos_array).transpose()
    #print eos_array
    #positive_pressure=eos_array[2][102:].min()>0
    positive_pressure=eos_array[2][2:].min()>0
    #if(positive_pressure and not stability):
    #plt.plot(toMevfm(eos_array[0],'mev4'),toMevfm(eos_array[1],'mev4'))
    #plt.xlim(0.0,0.3)
    #plt.ylim(-2,40)
    return init_sat,eos_array,sol_saturation,success,positive_pressure

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

# =============================================================================
# init_args= [12.124200434658492**2, 14.358205178844612**2, 9.2022483593647681**2, 0.0018220954984504475, -0.0027122507224894791, 0.028295688189677305, 0]
# init_sat = [484.27504990105365, 378.79614094279168, 319.23744720471598]
# aa=EOS_RMF(init_args,init_sat,[1152525.807543879,
#  923,
#  240,
#  657.30000000000007,
#  32.0,
#  100.0,
#  0.0,
#  (0.5109989461, 939, 550, 783, 763)])
# 
# from show_properity import show_eos
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
# show_eos(axes,[aa],0,1,500,pressure_range=[0.01,250,'log'])
# axes.plot([toMevfm(3454696.40485,'mev4'),toMevfm(3454696.40485,'mev4')],[toMevfm(1113320804.4081149,'mev4'),toMevfm(1189096186.4187522,'mev4')],'.')
# =============================================================================

# =============================================================================
# eos_pressure_density(3371438.4735,[ 307.25801137,  731.00479943,  403.4388082 ],Preset_tol,[(0.5109989461, 939, 550, 783, 763),aa.eos_args])
# 3454696.40485
# 
# True
# 3371438.4735 [array([ 307.25801137,  731.00479943,  403.4388082 ]), 3599410778.2397461, 1001421816.1980724]
# True
# 3454696.40485 [array([ 297.34117556,  779.97899961,  394.38970782]), 3723490411.3275251, 1189096192.5216742]
# =============================================================================

def Calculation_creat_eos(eos_args):
    Properity_onepointfour=Properity_ofmass(1.4,10,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
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
    eos_success=[]
    baryon_density_s=0.15

    args=np.mgrid[30:34:5j,0.5*939:0.8*939:4j,0:0.03:4j,80:100:9j]
    init_args= [12.124200434658492**2, 14.358205178844612**2, 9.2022483593647681**2, 0.0018220954984504475, -0.0027122507224894791, 0.028295688189677305, 0]
    init_sat = [484.27504990105365, 378.79614094279168, 319.23744720471598]
    J,m_eff,self_W,L=args
    args_shape=np.shape(m_eff)

    for i in range(args_shape[0]):
        eos_rmf.append([])
        for j in range(args_shape[1]):
            eos_rmf[i].append([])
            for k in range(args_shape[2]):
                eos_rmf[i][j].append([])
                for l in range(args_shape[3]):
                    print i,j,k,l
                    try:
                        eos_rmf[i][j][k].append(EOS_RMF(eos_rmf[i][j][k][l-1].eos_args,eos_rmf[i][j][k][l-1].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],(0.5109989461,939,550,783,763)]))
                    except:
                        try:
                            eos_rmf[i][j][k].append(EOS_RMF(eos_rmf[i][j][k-1][l].eos_args,eos_rmf[i][j][k-1][l].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],(0.5109989461,939,550,783,763)]))
                        except:
                            try:
                                eos_rmf[i][j][k].append(EOS_RMF(eos_rmf[i][j-1][k][l].eos_args,eos_rmf[i][j-1][k][l].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],(0.5109989461,939,550,783,763)]))
                            except:
                                try:
                                    eos_rmf[i][j][k].append(EOS_RMF(eos_rmf[i-1][j][k][l].eos_args,eos_rmf[i-1][j][k][l].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],(0.5109989461,939,550,783,763)]))
                                except:
                                    eos_rmf[i][j][k].append(EOS_RMF(init_args,init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],(0.5109989461,939,550,783,763)]))
                    eos_success.append(eos_rmf[i][j][k][l].stability)
    eos_rmf=np.array(eos_rmf)
    eos_flat=eos_rmf.flatten()
    eos_success=np.array(eos_success)
    from show_properity import show_eos
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
    show_eos(axes,eos_flat[eos_success],0,1,500,pressure_range=[0.01,250,'log'])


# =============================================================================
#     J=30
#     eos_rmf=[]
#     baryon_density_s=0.15
#     args=np.mgrid[0.5*939:0.8*939:4j,0:0.04:5j,80:30:6j]
#     init_args= [12.124200434658492**2, 14.358205178844612**2, 9.2022483593647681**2, 0.0018220954984504475, -0.0027122507224894791, 0.028295688189677305, 0]
#     init_sat = [484.27504990105365, 378.79614094279168, 319.23744720471598]
#     m_eff,self_W,L=args
#     args_shape=np.shape(m_eff)
# 
#     for i in range(args_shape[0]):
#         eos_rmf.append([])
#         for j in range(args_shape[1]):
#             eos_rmf[i].append([])
#             for k in range(args_shape[2]):
#                 try:
#                     eos_rmf[i][j].append(EOS_RMF(eos_rmf[i][j][k-1].eos_args,eos_rmf[i][j][k-1].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
#                 except:
#                     try:
#                         eos_rmf[i][j].append(EOS_RMF(eos_rmf[i][j-1][k].eos_args,eos_rmf[i][j-1][k].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
#                     except:
#                         try:
#                             eos_rmf[i][j].append(EOS_RMF(eos_rmf[i-1][j][k].eos_args,eos_rmf[i-1][j][k].init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
#                         except:
#                             eos_rmf[i][j].append(EOS_RMF(init_args,init_sat,[toMev4(baryon_density_s,'mevfm'),939-16,240,m_eff[i][j][k],J,L[i][j][k],self_W[i][j][k],(0.5109989461,939,550,783,763)]))
# =============================================================================

        
# =============================================================================
# import cPickle
# import os
# path = "./"
# dir_name='Lambda_RMF_calculation_parallel'
# error_log=path+dir_name+'/error.log'
# if __name__ == '__main__':
#     try:
#         os.stat(path+dir_name)
#     except:
#         os.mkdir(path+dir_name)
# 
#     n_s=0.15
#     m=939
#     mass_args=(0.5109989461,939,550,783,763)
#     BindE=16
#     K=240 #incompressibility of symmetric nuclear matter
#     N_m=1#N_m=31
#     N_J=1
#     N_L=6#N_L=26
#     N_W=5
#     m_eff = np.linspace(0.5,0.5,N_m)*m
#     J = np.linspace(30.,30.,N_J)
#     L = np.linspace(80,30,N_L)
#     self_W = np.linspace(0,0.4,N_W)
#     init_args= [12.124200434658492**2, 14.358205178844612**2, 9.2022483593647681**2, 0.0018220954984504475, -0.0027122507224894791, 0.028295688189677305, 0]
#     init_sat = [484.27504990105365, 378.79614094279168, 319.23744720471598]
#     args=[]
#     eos =[]
#     eos_success=[]
#     for t in range(len(m_eff)):
#         eos.append([])
#         for i in range(len(self_W)):
#             eos[t].append([])
#             for j in range(len(L)):
#                 eos[t][i].append([])
#                 for k in range(len(J)):
#                     print t,i,j,k
#                     
#                     args.append([n_s,m,m_eff[t],J[k],L[j],self_W[i]])
#                     try:
#                         print '1'
#                         eos[t][i][j].append(EOS_RMF(eos[t][i][j][k-1].eos_args,eos[t][i][j][k-1].init_sat,[toMev4(n_s,'mevfm'),m-BindE,K,m_eff[t],J[k],L[j],self_W[i],mass_args]))
#                     except:
#                         try:
#                             print '2'
#                             eos[t][i][j].append(EOS_RMF(eos[t][i][j-1][k].eos_args,eos[t][i][j-1][k].init_sat,[toMev4(n_s,'mevfm'),m-BindE,K,m_eff[t],J[k],L[j],self_W[i],mass_args]))
#                         except:
#                             try:
#                                 print '3'
#                                 eos[t][i][j].append(EOS_RMF(eos[t][i-1][j][k].eos_args,eos[t][i-1][j][k].init_sat,[toMev4(n_s,'mevfm'),m-BindE,K,m_eff[t],J[k],L[j],self_W[i],mass_args]))
#                             except:
#                                 try:
#                                     print '4'
#                                     eos[t][i][j].append(EOS_RMF(eos[t-1][i][j][k].eos_args,eos[t-1][i][j][k].init_sat,[toMev4(n_s,'mevfm'),m-BindE,K,m_eff[t],J[k],L[j],self_W[i],mass_args]))
#                                 except:
#                                     print '5'
#                                     eos[t][i][j].append(EOS_RMF(init_args,init_sat,[toMev4(n_s,'mevfm'),m-BindE,K,m_eff[t],J[k],L[j],self_W[i],mass_args]))
#                     
#                     print eos[t][i][j][k].eos_args
#                     #eos_success.append(eos[-1].eos_success)
#     eos_success=np.array(eos_success)
#     args=np.reshape(np.array(args),(N_m,N_J,N_L,N_W,-1))
#     args_flat=np.reshape(np.array(args),(N_m*N_J*N_L*N_W,-1))
#     eos =np.array(eos)
#     eos_flat=eos.flatten()
# =============================================================================
    