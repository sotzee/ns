#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:55:11 2019

@author: sotzee
"""

import scipy.optimize as opt
import numpy as np
from unitconvert import toMevfm,toMev4
from solver_equations import solve_equations,explore_solutions,Calculation_unparallel
Preset_tol=1e-6

Preset_Pressure_final=1e-6
Preset_rtol=1e-4

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

def logic_sol(sol_x):
    return sol_x[:3].min()>0 and sol_x[:3].max()<1e8

def equations1234(x,args):
    baryon_density_sat,bd_energy,incompressibility,m_eff,J,L,self_W,mass_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,b,c=x
# =============================================================================
#     g2_Phi=np.max([0,g2_Phi])
#     g2_W=np.max([0,g2_W])
#     g2_rho=np.max([0,g2_rho])
# =============================================================================
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
    return eq1,eq2,eq3,eq4

def equations56(x,args):
    equation1234_sol,baryon_density_sat,bd_energy,incompressibility,m_eff,J,L,self_W,mass_args=args
    g2_Phi,g2_W,b,c=equation1234_sol
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_rho,Lambda=x
# =============================================================================
#     g2_Phi=np.max([0,g2_Phi])
#     g2_W=np.max([0,g2_W])
#     g2_rho=np.max([0,g2_rho])
# =============================================================================
    k_F=(3*np.pi**2*baryon_density_sat/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    W_0=bd_energy-E_F
    tmp_2=g2_W/(m_W**2+self_W/2*g2_W*W_0**2)
    tmp_1=(incompressibility/(9*baryon_density_sat)-np.pi**2/(2*k_F*E_F)-tmp_2)
    tmp_J_0=k_F**2/(6*E_F)
    tmp_J_1=baryon_density_sat*g2_rho/(8*(m_rho)**2+16*Lambda*(W_0)**2*g2_rho) #origin fomula was baryon_density_sat/(8*(m_rho/g_rho)**2+16*Lambda*(W_0)**2)
    eq5=m**3*(tmp_J_0+tmp_J_1-J)
    eq6=m**3*(tmp_J_0*(1+(m_eff**2-3*baryon_density_sat*E_F*tmp_1)/E_F**2)+3*tmp_J_1*(1-32*tmp_2*W_0*Lambda*tmp_J_1)-L)
    return eq5,eq6

def equations5678(x,args):
    equation1234_sol,baryon_density_sat,bd_energy,incompressibility,m_eff_sym,J,L,self_W,mass_args=args
    g2_Phi,g2_W,b,c=equation1234_sol
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_rho,Lambda,W_0,m_eff=x
# =============================================================================
#     g2_Phi=np.max([0,g2_Phi])
#     g2_W=np.max([0,g2_W])
#     g2_rho=np.max([0,g2_rho])
# =============================================================================
    n_n=baryon_density_sat
    n3=baryon_density_sat
    k_F_n=(3*np.pi**2*n_n)**(1./3.)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    n_scalar=(m_eff/(2*np.pi**2))*((E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_n=E_F_n+W_0+rho_0/2
    energy_density=((E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_n*n_n-energy_density
    eq5=energy_density-n_n*(bd_energy+J)
    eq6=pressure - n_n*L/3.
    eq7=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    eq8=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - baryon_density_sat)
    return eq5,eq6,eq7,eq8

# =============================================================================
# init_args=np.array([10.,12.,9.,0.002,-0.002,0])**2
# #check with W.C. Chen and J. Piekarewicz 2014   NL3 sets
# sol = solve_equations(equations,init_args,tol=1e-30,vary_N=5,args=[toMev4(0.1481,'mevfm'),939-16.24,271.5,0.595*939,37.28,118.2,0.,(0.5109989461,939,508.194,782.501,763)])
# print(sol.x,sol.success)
# print(equations(sol.x,[toMev4(0.1481,'mevfm'),939-16.24,271.5,0.595*939,37.28,118.2,0.,(0.5109989461,939,508.194,782.501,763)]))
# 
# #check with W.C. Chen and J. Piekarewicz 2014   FSU sets
# sol = solve_equations(equations,init_args,tol=1e-30,vary_N=5,args=[toMev4(0.1484,'mevfm'),939-16.3,230.0,0.61*939,32.59,60.5,0.06,(0.5109989461,939,491.5,782.500,763)])
# print(sol.x,sol.success)
# print(equations(sol.x,[toMev4(0.1484,'mevfm'),939-16.3,230.0,0.61*939,32.59,60.5,0.06,(0.5109989461,939,491.5,782.500,763)]))
# 
# #check with Nadine Hornick et. al. 2018
# sol = solve_equations(equations,init_args,tol=1e-30,vary_N=5,args=[toMev4(0.15,'mevfm'),939-16,240,0.65*939,30,50,0.,(0.5109989461,939,550,783,763)])
# print(sol.x,sol.success)
# print(equations(sol.x,[toMev4(0.15,'mevfm'),939-16,240,0.65*939,30,50,0.,(0.5109989461,939,550,783,763)]))
# eos_args=sol.x
# 
# sol = solve_equations(equations,init_args,tol=1e-30,vary_N=5,args=[toMev4(0.15,'mevfm'),939-16,240,0.60*939,30,50,0.,(0.5109989461,939,550,783,763)])
# print(sol.x,sol.success)
# print(equations(sol.x,[toMev4(0.15,'mevfm'),939-16,240,0.60*939,30,50,0.,(0.5109989461,939,550,783,763)]))
# eos_args=sol.x
# 
# #check 1234-5678 calibration with NL3 paper,
# sol = solve_equations(equations1234,[10.,12.,0.002,-0.002],tol=1e-12,vary_N=5,args=[toMev4(0.148,'mevfm'),939-16.299,271.76,0.60*939,37.4,118.2,0.,(0.5109989461,939,508.194,782.501,763)])
# print(sol.x,sol.success)
# eos_args1234=sol.x
# W_0_init=939-16.299-((3*np.pi**2*toMev4(0.148,'mevfm'))**(2./3.)+(0.60*939)**2)**0.5
# sol = solve_equations(equations5678,[9.,0.,W_0_init,0.60*939],tol=1e-12,vary_N=5,args=[eos_args1234,toMev4(0.148,'mevfm'),939-16.299,271.76,0.60*939,37.4,118.2,0.,(0.5109989461,939,508.194,782.501,763)])
# print(sol.x,sol.success)
# eos_args5678=sol.x
# =============================================================================

# =============================================================================
# def Precalculaltion_eos_args(init,Preset_tol,args):
#     J,m_eff,self_W,L=args
#     sol = solve_equations(equations,init[:6],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#     if not sol.success:
#         print('Calculation_calculate_eos_args failed at following parameter:')
#         print('J,m_eff,self_W,L=%s'%args)
#         print('args=%s'%[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#         print(equations(sol.x,[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args]))
#     print [sol.success],sol.x,self_W
#     return np.concatenate(([sol.success],sol.x,[self_W]))
# 
# def Precalculaltion_eos_args_1234_56(init,Preset_tol,args):
#     J,m_eff,self_W,L=args
#     sol_1234 = solve_equations(equations1234,init[[0,1,3,4]],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#     sol_56 = solve_equations(equations56,init[[2,5]],tol=Preset_tol,vary_N=11,args=[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#     if not sol_56.success:
#         print('Calculation_calculate_eos_args failed at following parameter:')
#         print('J,m_eff,self_W,L=%s'%args)
#         print('args=%s'%[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#         print(equations56(sol_56.x,[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args]))
#     return np.concatenate(([sol_1234.success and sol_56.success],sol_1234.x[:2],(sol_56.x[0],),sol_1234.x[2:],(sol_56.x[1],self_W)))
# 
# def Precalculaltion_eos_args_1234_5678(init,Preset_tol,args):
#     J,m_eff,self_W,L=args
#     sol_1234 = solve_equations(equations1234,init[[0,1,3,4]],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#     k_F=(3*np.pi**2*baryon_density_s_MeV4/2)**(1./3.)
#     E_F=(k_F**2+m_eff**2)**0.5
#     W_0=(m-BindE)-E_F
#     sol_5678 = solve_equations(equations5678,list(init[[2,5]])+[W_0,m_eff],tol=Preset_tol,vary_N=11,args=[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#     if not sol_5678.success:
#         print('Calculation_calculate_eos_args failed at following parameter:')
#         print('J,m_eff,self_W,L=%s'%args)
#         print('args=%s'%[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
#         print(equations5678(sol_5678.x,[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args]))
#     return np.concatenate(([sol_1234.success and sol_5678.success],sol_1234.x[:2],(sol_5678.x[0],),sol_1234.x[2:],(sol_5678.x[1],self_W)))
# eos_args=[]
# for i in range(args_shape[0]):
#     eos_args.append([])
#     for j in range(args_shape[1]):
#         eos_args[i].append([])
#         for k in range(args_shape[2]):
#             eos_args[i][j].append([])
#             for l in range(args_shape[3]):
#                 print i,j,k,l
#                 try:
#                     print 1
#                     sol = Precalculaltion_eos_args(eos_args[i][j][k][l-1],Preset_tol,args[:,i,j,k,l])
#                 except:
#                     try:
#                         print 2
#                         sol = Precalculaltion_eos_args(eos_args[i][j][k-1][l],Preset_tol,args[:,i,j,k,l])
#                     except:
#                         try:
#                             print 3
#                             sol = Precalculaltion_eos_args(eos_args[i][j-1][k][l],Preset_tol,args[:,i,j,k,l])
#                         except:
#                             try:
#                                 print 4
#                                 sol = Precalculaltion_eos_args(eos_args[i-1][j][k][l],Preset_tol,args[:,i,j,k,l])
#                             except:
#                                 sol = Precalculaltion_eos_args(init_args,Preset_tol,args[:,i,j,k,l])
#                 eos_args[i][j][k].append(sol[1:])
#                 if not sol[0]:
#                     print i,j,k,l
#                     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# eos_args=np.array(eos_args)
# 
# from scipy.interpolate import RegularGridInterpolator
# eos_args_int=[]
# for i in range(len(eos_args[0,0,0,0])):
#     eos_args_int.append(RegularGridInterpolator((J[:,0,0,0],m_eff[0,:,0,0],self_W[0,0,:,0],L[0,0,0,:]),eos_args[:,:,:,:,i]))
# 
# f_file=open(path+dir_name+'/RMF_eos_args_interpolation_from_bulk_properity.dat','wb')
# cPickle.dump(eos_args_int,f_file)
# f_file.close()
# 
# 
# path = "./"
# dir_name='Lambda_RMF_calculation_parallel'
# import cPickle
# 
# baryon_density_s=0.15
# m=939
# BindE=16
# K=240
# mass_args=(0.5109989461,939,550,783,763)
# baryon_density_s_MeV4=toMev4(baryon_density_s,'mevfm')
# args=np.mgrid[30:31:2j,0.5*939:0.6*939:2j,0:0.03:4j,20:120:11j]
# args_flat=args.reshape((-1,np.prod(np.shape(args)[1:]))).transpose()
# init_args= [  7.22771445e+01,   6.92175306e+01,   1.84292144e+02,
#          8.22065307e-03,   8.46232278e-03,   1.74361065e-01]
# init_sat = [484.27504990105365, 378.79614094279168, 319.23744720471598]
# J,m_eff,self_W,L=args
# args_shape=np.shape(m_eff)
# 
# eos_args=[]
# for i in range(args_shape[0]):
#     eos_args.append([])
#     for j in range(args_shape[1]):
#         eos_args[i].append([])
#         for k in range(args_shape[2]):
#             eos_args[i][j].append([])
#             for l in range(args_shape[3]):
#                 print i,j,k,l
#                 try:
#                     sol = solve_equations(equations,eos_args[i][j][k][l-1],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],mass_args])
#                 except:
#                     try:
#                         sol = solve_equations(equations,eos_args[i][j][k-1][l],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],mass_args])
#                     except:
#                         try:
#                             sol = solve_equations(equations,eos_args[i][j-1][k][l],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],mass_args])
#                         except:
#                             try:
#                                 sol = solve_equations(equations,eos_args[i-1][j][k][l],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],mass_args])
#                             except:
#                                 sol = solve_equations(equations,init_args,tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],mass_args])
#                 eos_args[i][j][k].append(sol.x)
#                 if not (sol.success and np.abs(sol.x).max()<1e7):
#                     print i,j,k,l
#                     print equations(sol.x,[baryon_density_s_MeV4,m-BindE,K,m_eff[i][j][k][l],J[i][j][k][l],L[i][j][k][l],self_W[i][j][k][l],mass_args])
# eos_args=np.array(eos_args)
# 
# from scipy.interpolate import RegularGridInterpolator
# eos_args_int=[]
# for i in range(len(eos_args[0,0,0,0])):
#     eos_args_int.append(RegularGridInterpolator((J[:,0,0,0],m_eff[0,:,0,0],self_W[0,0,:,0],L[0,0,0,:]),eos_args[:,:,:,:,i]))
# 
# f_file=open(path+dir_name+'/RMF_eos_args_interpolation_from_bulk_properity.dat','wb')
# cPickle.dump(eos_args_int,f_file)
# f_file.close()
# =============================================================================


path = "./"
dir_name='Lambda_RMF_calculation_parallel'
import cPickle

baryon_density_s=0.16
m=939
BindE=16
K=240
mass_args=(0.5109989461,939,550,783,763)
baryon_density_s_MeV4=toMev4(baryon_density_s,'mevfm')
equations_extra_args=(baryon_density_s_MeV4,m-BindE,K,mass_args)
args=np.mgrid[0.5*939:0.8*939:13j,28:36:17j,0:100:21j,0:0.03:13j]
args_flat=args.reshape((-1,np.prod(np.shape(args)[1:]))).transpose()
init_args= (100.,100.,100.,0.001,0.001,0.)
J,m_eff,self_W,L=args
args_shape=np.shape(m_eff)

calculated_logic,result=explore_solutions(equations,args,((5,7,10,0),),(init_args,),vary_list=np.array([1.]),tol=1e-10,logic_success_f=logic_sol,Calculation_routing=Calculation_unparallel,equations_extra_args=equations_extra_args)

from plot_logic import plot_5D_logic
plot_5D_logic(np.logical_not(calculated_logic[::3,::3,::3,::3]),args[:,::3,::3,::3,::3],['$m_{eff}$','J','L','$\zeta$'],(1,3,2,0),figsize=(16,15))

# =============================================================================
# fig, axes = plt.subplots(3, 5,sharex=True,sharey=True,figsize=(10,6))
# plt.title('M_eff=0.5*939MeV, J=30 MeV, K=240 MeV, $\zeta=0$')
# for i in range(3):
#     for j in range(5):
#         if(i==0):
#             axes[i,j].imshow(logic_positive_pressure_RMF_subnuclear[:,2*j,:].transpose(),aspect='auto',origin='upper',extent=(args[0].min(),args[0].max(),args[2].min(),args[2].max()))
#         elif(i==1):
#             axes[i,j].imshow(logic_maximum_mass[:,2*j,:].transpose(),aspect='auto',origin='upper',extent=(args[0].min(),args[0].max(),args[2].min(),args[2].max()))
#             #axes[i,j].imshow(np.logical_and(logic_positive_pressure,logic_maximum_mass)[:,j,:].transpose(),aspect='auto',origin='upper',extent=(args[0].min(),args[0].max(),args[2].min(),args[2].max()))
#         elif(i==2):
#             axes[i,j].imshow((radius_onepointfour<13500)[:,2*j,:].transpose(),aspect='auto',origin='upper',extent=(args[0].min(),args[0].max(),args[2].min(),args[2].max()))
# 
#         axes[i,j].set_title('self_W=%.2f MeV'%(args[1,0,2*j,0]))
#         if(j==0):
#             axes[i,j].set_ylabel('$L$ MeV')
#         if(i==2):
#             axes[i,j].set_xlabel('$m_{eff}$ MeV')
# =============================================================================

f_file=open(path+dir_name+'/RMF_eos_args_interpolation_from_bulk_properity.dat','rb')
eos_args_int=cPickle.load(f_file)
f_file.close()

def get_eos_args(args):#args=[[J,m_eff,self_W,L],...,[J,m_eff,self_W,L]]
    result=[]
    for eos_args_int_i in eos_args_int:
        result.append(eos_args_int_i(args))
    return np.array(result).transpose()

def Calculation_calculate_eos_args(args):
    J,m_eff,self_W,L=args
    sol = solve_equations(equations,get_eos_args(args)[0],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
    if not sol.success:
        print('Calculation_calculate_eos_args failed at following parameter:')
        print('J,m_eff,self_W,L=%s'%args)
        print('args=%s'%[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
        print(equations(sol.x,[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args]))
    return np.concatenate(([sol.success],sol.x,self_W))

def Calculation_calculate_eos_args_1234_56(args):
    J,m_eff,self_W,L=args
    sol_1234 = solve_equations(equations1234,get_eos_args(args)[0,[0,1,3,4]],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
    sol_56 = solve_equations(equations56,get_eos_args(args)[0,[2,5]],tol=Preset_tol,vary_N=11,args=[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
    if not sol_56.success:
        print('Calculation_calculate_eos_args failed at following parameter:')
        print('J,m_eff,self_W,L=%s'%args)
        print('args=%s'%[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
        print(equations56(sol_56.x,[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args]))
    return np.concatenate(([sol_1234.success and sol_56.success],sol_1234.x[:2],(sol_56.x[0],),sol_1234.x[2:],(sol_56.x[1],self_W)))

def Calculation_calculate_eos_args_1234_5678(args):
    J,m_eff,self_W,L=args
    sol_1234 = solve_equations(equations1234,get_eos_args(args)[0,[0,1,3,4]],tol=Preset_tol,vary_N=7,args=[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
    k_F=(3*np.pi**2*baryon_density_s_MeV4/2)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    W_0=(m-BindE)-E_F
    sol_5678 = solve_equations(equations5678,list(get_eos_args(args)[0,[2,5]])+[W_0,m_eff],tol=Preset_tol,vary_N=11,args=[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
    if not sol_5678.success:
        print('Calculation_calculate_eos_args failed at following parameter:')
        print('J,m_eff,self_W,L=%s'%args)
        print('args=%s'%[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args])
        print(equations56(sol_5678.x,[sol_1234.x]+[baryon_density_s_MeV4,m-BindE,K,m_eff,J,L,self_W,mass_args]))
    return np.concatenate(([sol_1234.success and sol_5678.success],sol_1234.x[:2],(sol_5678.x[0],),sol_1234.x[2:],(sol_5678.x[1],self_W)))

from Parallel_process import main_parallel
f_eos_args='./'+dir_name+'/RMF_eos_args.dat'
error_log=path+dir_name+'/error.log'
eos_args_flat=main_parallel(Calculation_calculate_eos_args_1234_5678,args_flat,f_eos_args,error_log)
eos_args_success=eos_args_flat[:,0].astype('bool')
eos_args_flat=eos_args_flat[:,1:]
eos_args=(eos_args_flat.transpose()).reshape((-1,)+np.shape(args)[1:])

print np.shape(eos_args)
# =============================================================================
# def Calculation_creat_eos(eos_args):
#     sol = solve_equations(equations,eos_args[i][j][k-1][l],tol=Preset_tol,vary_N=9,args=[baryon_density_s_MeV4,m-BindE]+args+[mass_args])
#     return Properity_onepointfour
# =============================================================================

def eos_equations(y,args):
    n,mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
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
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    #eq5=((E_F_e*k_F_e**3+E_F_e**3*k_F_e-m_e**4*np.log((k_F_e+E_F_e)/m_e))+(E_F_p*k_F_p**3+E_F_p**3*k_F_p-m_eff**4*np.log((k_F_p+E_F_p)/m_eff))+(E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0/g_Phi)**2/2+(m_W*W_0/g_W)**2/2+(m_rho*rho_0/g_rho)**2/2+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2-energy_density
    chempo_e=E_F_e
    chempo_p=E_F_p+W_0-rho_0/2
    chempo_n=E_F_n+W_0+rho_0/2
    eq6=chempo_e+chempo_p-chempo_n
    #eq7=chempo_e*n_e+chempo_p*n_p+chempo_n*n_n-energy_density-pressure
    eq6=chempo_e+chempo_p-chempo_n
    return eq2,eq3,eq6


def eos_equations_PNM_get_m_eff(m_eff,args):
    n,mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    n_n=n
    k_F_n=(3*np.pi**2*n_n)**(1./3)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    n_scalar=(m_eff/(2*np.pi**2))*((E_F_n*k_F_n-m_eff**2*np.log((k_F_n+E_F_n)/m_eff)))
    Phi_0=m-m_eff
    eq2=m*((m_Phi)**2*Phi_0/g2_Phi + (m*b)*Phi_0**2 + c*Phi_0**3 - n_scalar)
    return eq2
def eos_equations_PNM_get_W_0(W_0,args):
    m_eff,n,mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    n_n=n
    n3=n_n
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    eq3=m*((m_W)**2*W_0/g2_W + (self_W/6)*W_0**3 + 2*Lambda*W_0*rho_0**2 - n)
    return eq3
def eos_pressure_density_PNM(n,Preset_tol,_args):
    mass_args,eos_args,args=_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    sol1 = solve_equations(eos_equations_PNM_get_m_eff,args[1],tol=Preset_tol,vary_N=5,args=[n,mass_args,eos_args])
    m_eff=sol1.x
    k_F=(3*np.pi**2*n)**(1./3.)
    E_F=(k_F**2+m_eff**2)**0.5
    W_0=m-BindE-E_F
    sol2 = solve_equations(eos_equations_PNM_get_W_0,W_0,tol=Preset_tol,vary_N=5,args=[m_eff,n,mass_args,eos_args])
    W_0=sol2.x
    n_n=n
    n3=n_n
    k_F_n=(3*np.pi**2*n_n)**(1./3)
    E_F_n=(k_F_n**2+m_eff**2)**0.5
    Phi_0=m-m_eff
    rho_0=0.5*n3/((m_rho)**2/g2_rho + 2*Lambda*W_0**2)
    chempo_n=E_F_n+W_0+rho_0/2
    energy_density=((E_F_n*k_F_n**3+E_F_n**3*k_F_n-m_eff**4*np.log((k_F_n+E_F_n)/m_eff)))/(8*np.pi**2)+(m_Phi*Phi_0)**2/(2*g2_Phi)+(m_W*W_0)**2/(2*g2_W)+(m_rho*rho_0)**2/(2*g2_rho)+b*m*Phi_0**3/3+c*Phi_0**4/4+self_W*W_0**4/8+3*Lambda*(W_0*rho_0)**2
    pressure=chempo_n*n_n-energy_density
    return [sol1.success and sol2.success,energy_density,pressure]

def Calculation_PNM_saturation(eos_args_args):
    return eos_pressure_density_PNM(baryon_density_s_MeV4,Preset_tol,(mass_args,eos_args_args[:7],eos_args_args[7:]))

f_PNM_saturation='./'+dir_name+'/RMF_PNM_saturation.dat'
error_log=path+dir_name+'/error.log'
PNM_saturation=main_parallel(Calculation_PNM_saturation,np.concatenate((eos_args_flat,args_flat),axis=1),f_PNM_saturation,error_log)

def eos_J_L_around_sym(baryon_density_sat,bd_energy,incompressibility,_args):
    mass_args,eos_args,args=_args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    baryon_density_sat=baryon_density_s_MeV4
    bd_energy=m-BindE
    incompressibility=K
    m_eff=args[1]
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

f_J_L_around_sym='./'+dir_name+'/RMF_J_L_around_sym.dat'
error_log=path+dir_name+'/error.log'
J_L_around_sym=main_parallel(Calculation_J_L_around_sym,np.concatenate((eos_args_flat,args_flat),axis=1),f_J_L_around_sym,error_log)

# =============================================================================
# eos_args_flat_old=main_parallel(Calculation_calculate_eos_args_1234_56,args_flat,f_eos_args,error_log)
# eos_args_success_old=eos_args_flat_old[:,0].astype('bool')
# eos_args_flat_old=eos_args_flat_old[:,1:]
# eos_args_old=(eos_args_flat_old.transpose()).reshape((-1,)+np.shape(args)[1:])
# PNM_saturation_old=main_parallel(Calculation_calculate_PNM_saturation,np.concatenate((eos_args_flat_old[eos_args_success_old],args_flat[eos_args_success_old]),axis=1),f_PNM_saturation,error_log)
# import matplotlib.pyplot as plt
# plt.plot(PNM_saturation[:,1]/baryon_density_s_MeV4-939+16,PNM_saturation[:,2]/baryon_density_s_MeV4*3,'.',label='isovector calibration 4+4')
# plt.plot(PNM_saturation_old[:,1]/baryon_density_s_MeV4-939+16,PNM_saturation_old[:,2]/baryon_density_s_MeV4*3,'.',label='isovector calibration 4+2')
# plt.xlim(952-939+16,962-939+16)
# plt.xlabel('$S_0$')
# plt.ylim(0,160)
# plt.ylabel('$L$')
# plt.legend(frameon=False)
# =============================================================================

def eos_pressure_density(n,init,Preset_tol,args):
    mass_args,eos_args=args
    m_e,m,m_Phi,m_W,m_rho=mass_args
    g2_Phi,g2_W,g2_rho,b,c,Lambda,self_W=eos_args
    sol = solve_equations(eos_equations,init,tol=Preset_tol,vary_N=5,args=[n,mass_args,eos_args])
    m_eff,W_0,k_F_n=sol.x
    n_n=k_F_n**3/(3*np.pi**2)
    n_p=n-n_n
    #n_p=(np.sign(n_p)+1)/2*np.abs(n_p)
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
    return [sol.success,sol.x,energy_density,pressure]

def get_eos_array(init0,Preset_tol,baryon_density_sat,mass_args,eos_args):
    baryon_density=baryon_density_sat/1.05**np.linspace(0,100,201)
    eos_array=[]
    init=init0
    success=True
    for i in range(len(baryon_density)):
        tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
        eos_array.append([baryon_density[i]]+tmp[2:])
        init=tmp[1]
        success=success and tmp[0]
    eos_array.append([0.,0.,0.])
    eos_array.append(list(2*np.array(eos_array[-1])-np.array(eos_array[-2])))
    eos_array=list(reversed(eos_array))

    sol_saturation=np.array(eos_array[-1])
    init = init0
    baryon_density=baryon_density_sat*1.05**np.linspace(0,50,201)
    #init_prev = eos_pressure_density(baryon_density[0]**2/baryon_density[1],init,Preset_tol,[mass_args,eos_args])[0]
    for i in range(1,len(baryon_density)):
        #tmp=eos_pressure_density(baryon_density[i],2*init-init_prev,Preset_tol,[mass_args,eos_args])
        tmp=eos_pressure_density(baryon_density[i],init,Preset_tol,[mass_args,eos_args])
        eos_array.append([baryon_density[i]]+tmp[2:])
        #init_prev=init
        init=tmp[1]
        success=success and tmp[0]
        if(not success):
            break
    
    eos_array=np.array(eos_array).transpose()
    #print eos_array
    #positive_pressure=eos_array[2][102:].min()>0
    positive_pressure=eos_array[2][2:].min()>0
    #if(positive_pressure and not stability):
    #plt.plot(toMevfm(eos_array[0],'mev4'),toMevfm(eos_array[1],'mev4'))
    #plt.xlim(0.0,0.3)
    #plt.ylim(-2,40)
    return toMevfm(eos_array,'mev4'),toMevfm(sol_saturation,'mev4'),success,positive_pressure




from scipy.misc import derivative
import scipy.constants as const
from scipy.interpolate import interp1d
from eos_class import EOS_BPS,match_eos,match_get_eos_array
dlnx_cs2=1e-6
class EOS_SLY4_match_RMF(object):
    match_init=[None,None] #not in use
    def __init__(self,eos_args_args):
        self.eos_args,args=eos_args_args
        self.J,self.m_eff,self.self_W,self.L=args
        self.baryon_density_s=baryon_density_s_MeV4
        self.bd_energy=m-BindE
        self.incompressibility=K
        self.mass_args=mass_args
        self.args=self.baryon_density_s,self.bd_energy,self.incompressibility,\
        self.m_eff,self.J,self.L,self.self_W,self.mass_args
        k_F=(3*np.pi**2*self.baryon_density_s)**(1./3.)
        W_0=self.bd_energy-(k_F**2+self.m_eff**2)**0.5
        init_sat=(self.m_eff,W_0,k_F)
        self.eos_array_rmf,self.sol_saturation,self.success,self.positive_pressure=get_eos_array(init_sat,Preset_tol,self.baryon_density_s,self.mass_args,self.eos_args)
        self.baryon_density_s=self.baryon_density_s
        self.pressure_s=self.sol_saturation[2]
        self.density_s=self.sol_saturation[1]
        self.n1_match=0.06
        self.n2_match=0.16
        p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
        e1=EOS_BPS.eosDensity(p1)
        dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
        p2=self.pressure_s
        e2=self.density_s
        dpdn2=dpdn1#this parameter is not used in match_eos, so it was trivially set to dpdn1
        self.p_match=p1
        self.matching_args=self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init
        self.matching_para,self.matching_success=match_eos(self.matching_args)
        self.eos_success=self.success and self.positive_pressure and self.matching_success
        if(self.matching_success):
            u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,81)
            eos_array_match=match_get_eos_array(u_array_low,self.matching_para)
        else:
            eos_array_match=[[self.n1_match,self.n2_match],[e1,e2],[p1,p2]]
        self.eos_array=np.concatenate((np.array([[0],[0],[0]]),np.array(eos_array_match)[:,:-1],self.eos_array_rmf[:,203:]),axis=1)
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
        self.maxmass_success=self.cs2_max<1 and self.mass_max>2
        self.eos_success_all=self.maxmass_success and self.eos_success
        return self.eos_success_all

def Calculation_creat_eos_RMF(eos_args_args):
    return EOS_SLY4_match_RMF((eos_args_args[:7],eos_args_args[7:]))

f_eos_RMF='./'+dir_name+'/RMF_eos.dat'
error_log=path+dir_name+'/error.log'
eos_flat=main_parallel(Calculation_creat_eos_RMF,np.concatenate((eos_args_flat,args_flat),axis=1),f_eos_RMF,error_log)

eos_success=[]
success=[]
matching_success=[]
positive_pressure=[]
for eos_i in eos_flat:
    success.append(eos_i.success)
    matching_success.append(eos_i.matching_success)
    eos_success.append(eos_i.eos_success)
    positive_pressure.append(eos_i.positive_pressure)
eos_success=np.array(eos_success)
success=np.array(success)
matching_success=np.array(matching_success)
positive_pressure=np.array(positive_pressure)
print('len(eos)=%d'%len(eos_success))
print('len(eos[success])=%d'%len(success[success]))
print('len(eos[positive_pressure])=%d'%len(positive_pressure[positive_pressure]))
print('len(eos[matching_success])=%d'%len(matching_success[matching_success]))
print('len(eos[eos_success])=%d'%len(eos_success[eos_success]))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 1,figsize=(8,6),sharex=True,sharey=True)
from show_properity import show_eos
show_eos(axes,eos_flat,0,1,500,pressure_range=[0.01,30,'log'])


# =============================================================================
# from eos_class import EOS_BPS,EOS_BPSwithPoly
# class EOS_SLY4_match_RMF(object):
#     def __init__(self,init_args,init_sat,args):
#         self.n1_match=0.06
#         self.n2_match=0.16
#         self.g2_Phi,self.g2_W,self.g2_rho,self.b,self.c,self.Lambda,self.self_W=args
#         self.args=args
#         
#         self.eos_RMF=EOS_RMF(init_args,init_sat,args)
#         self.eos_args=self.eos_RMF.eos_args
#         self.init_sat=self.eos_RMF.init_sat
#         self.eos_array=self.eos_RMF.eos_array
#         self.sol_saturation=toMevfm(np.array(self.eos_RMF.sol_saturation),'mev4')
#         fix_crust_baryon_density=np.linspace(0.6,0.3,4)*self.sol_saturation[0]
#         self.fix_crust_logic=False
#         for fix_crust_baryon_density_i in fix_crust_baryon_density:
#             if(self.sol_saturation[2]>1.1*EOS_BPS.eosPressure_frombaryon(fix_crust_baryon_density_i)):
#                 self.eos_SLY4withPoly=EOS_BPSwithPoly([fix_crust_baryon_density_i,self.sol_saturation[2],self.sol_saturation[0],4*self.sol_saturation[2],2*self.sol_saturation[0],8*self.sol_saturation[2],3*self.sol_saturation[0]])
#                 self.fix_crust_logic=True
#                 break
#             
#         self.stability=self.eos_RMF.stability
#         self.positive_pressure=self.eos_RMF.positive_pressure
#         self.baryon_density_s=self.eos_RMF.baryon_density_s
#         self.pressure_s=self.eos_RMF.pressure_s
#         self.density_s=self.eos_RMF.density_s
#         self.unit_mass=self.eos_RMF.unit_mass
#         self.unit_radius=self.eos_RMF.unit_radius
#         self.unit_N=self.eos_RMF.unit_N
#     def __getstate__(self):
#         state_RMF=self.eos_RMF.__getstate__()
#         state = self.__dict__.copy()
#         return (state,state_RMF)
#     def __setstate__(self, state_):
#         state,state_RMF=state_
#         self.__dict__.update(state)
#         self.eos_RMF.__setstate__(state_RMF)
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosDensity(pressure),self.eos_RMF.eosDensity(pressure))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.sol_saturation[2],self.eos_SLY4withPoly.eosBaryonDensity(pressure),self.eos_RMF.eosBaryonDensity(pressure))
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=pressure*dlnx_cs2)
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# 
# 
# from eos_class import EOS_BPS
# class EOS_SLY4_match_EXPANSION_PNM(object):
#     match_init=[None,None] #not in use
#     def __init__(self,args,PNM_EXPANSION_TYPE,defaut_u_max=12):
# 
#         self.n1_match=0.06
#         self.n2_match=0.16
#         
#         self.T=.3*(1.5*np.pi**2*toMev4(self.baryon_density_s,'mevfm3'))**(2./3)/self.m
#         self.abcd_array=get_parameters(self.m,self.T*2**(2./3),self.ELKQ_array)
#         self.u_max=get_baryon_density_u_max(self.abcd_array,defaut_u_max)
#         self.get_eos_array_args=self.n2_match/self.baryon_density_s,self.u_max,self.baryon_density_s,self.m,self.T,self.abcd_array
#         eos_array_PNM,self.sol_saturation=get_eos_array(self.get_eos_array_args)
#         self.n_max,self.e_max,self.p_max=eos_array_PNM[:,-2]
#         self.high_stability_success=self.p_max>100.
#         self.cs2=1
#         self.args_eosCSS=[self.e_max,self.p_max,self.n_max,self.cs2]
#         self.eosCSS=EOS_CSS(self.args_eosCSS)
#         p1=EOS_BPS.eosPressure_frombaryon(self.n1_match)
#         e1=EOS_BPS.eosDensity(p1)
#         dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
#         p2=eos_array_PNM[2,1]
#         e2=eos_array_PNM[1,1]
#         dpdn2=EOS_BPS().eosCs2(p2)*EOS_BPS().eosChempo(p2)
#         self.matching_args=self.baryon_density_s,self.n1_match,p1,e1,dpdn1,self.n2_match,p2,e2,dpdn2,self.match_init
#         self.matching_para,self.matching_success=match_eos(self.matching_args)
#         self.eos_success=self.high_stability_success and self.matching_success
#         if(self.matching_success):
#             u_array_low=np.linspace(self.n1_match/self.baryon_density_s,self.n2_match/self.baryon_density_s,81)
#             eos_array_match=match_get_eos_array(u_array_low,self.matching_para)
#         else:
#             eos_array_match=[[self.n1_match,self.n2_match],[e1,e2],[p1,p2]]
#         self.p_match=p1
#         self.eos_array=np.concatenate((np.array([[0],[0],[0]]),np.array(eos_array_match)[:,:-1],eos_array_PNM[:,1:]),axis=1)
#         self.pressure_s=self.sol_saturation[2]
#         self.density_s=self.sol_saturation[1]
#         self.unit_mass=const.c**4/(const.G**3*self.density_s*1e51*const.e)**0.5
#         self.unit_radius=const.c**2/(const.G*self.density_s*1e51*const.e)**0.5
#         self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#         self.eosPressure_frombaryon_matchPNM = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosDensity_matchPNM  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity_matchPNM = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         for dict_intepolation in ['eosPressure_frombaryon_matchPNM','eosDensity_matchPNM','eosBaryonDensity_matchPNM']:
#             del state[dict_intepolation]
#         return state
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.eosPressure_frombaryon_matchPNM = interp1d(self.eos_array[0],self.eos_array[2], kind='linear')
#         self.eosDensity_matchPNM  = interp1d(self.eos_array[2],self.eos_array[1], kind='linear')
#         self.eosBaryonDensity_matchPNM = interp1d(self.eos_array[2],self.eos_array[0], kind='linear')
#     def eosPressure_frombaryon(self,baryon_density):
#         return np.where(baryon_density<self.n1_match,EOS_BPS.eosPressure_frombaryon(baryon_density),np.where(baryon_density<self.n_max,self.eosPressure_frombaryon_matchPNM(baryon_density),self.eosCSS.eosPressure_frombaryon(baryon_density)))
#     def eosDensity(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS.eosDensity(pressure),np.where(pressure<self.p_max,self.eosDensity_matchPNM(pressure),self.eosCSS.eosDensity(pressure)))
#     def eosBaryonDensity(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS.eosBaryonDensity(pressure),np.where(pressure<self.p_max,self.eosBaryonDensity_matchPNM(pressure),self.eosCSS.eosBaryonDensity(pressure)))
#     def eosCs2(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS().eosCs2(pressure),np.where(pressure<self.p_max,1.0/derivative(self.eosDensity_matchPNM,pressure,dx=pressure*dlnx_cs2),self.eosCSS.eosCs2(pressure)))
#     def eosChempo(self,pressure):
#         return np.where(pressure<self.p_match,EOS_BPS().eosChempo(pressure),np.where(pressure<self.p_max,(pressure+self.eosDensity_matchPNM(pressure))/self.eosBaryonDensity_matchPNM(pressure),self.eosCSS.eosChempo(pressure)))
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#         self.maxmass_success=self.cs2_max<1 and self.mass_max>2
#         self.eos_success_all=self.maxmass_success and self.eos_success
#         return self.eos_success_all
# =============================================================================
