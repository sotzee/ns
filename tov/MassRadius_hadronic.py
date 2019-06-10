#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:31:33 2018

@author: sotzee
"""
import numpy as np
from tov_f import f,f_baryon_number,f_tidal,f_MRI,f_complete,lsoda_ode,lsoda_ode_array
from astropy.constants import M_sun
from scipy.constants import m_n

def Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos):
    X=(eos.eosChempo(pc*Preset_Pressure_final)/(eos.chempo_surface))**2-1
    return beta/(beta+beta*X-0.5*X)
# =============================================================================
# def Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos):
#     X=(eos.eosChempo(pc*Preset_Pressure_final)/(931.171))**2-1
#     return beta/(beta+beta*X-0.5*X)
# =============================================================================
def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos,Radius_correction_ratio=Radius_correction_ratio):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    if(MRorMRBIT=='M'):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        return M
    elif(MRorMRBIT=='B'):
        r = lsoda_ode(f_baryon_number,Preset_rtol,[0.,0.,0,],x0,xf,eos)
        M_binding=r.y[2]*eos.unit_N*m_n/M_sun.value
        return M_binding
    elif(MRorMRBIT=='MR'):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        R=R*Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        return [M,R]
    elif(MRorMRBIT=='MRT'):
        r = lsoda_ode(f_tidal,Preset_rtol,[0.,0.,2.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        R=R*Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        yR=r.y[2]
        tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
        k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
        tidal=2.0/3.0*(k2/beta**5)
        beta=beta/Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        return [M,R,beta,k2,tidal]
    elif(MRorMRBIT=='MRBIT'):
        r = lsoda_ode(f_complete,Preset_rtol,[0.,0.,0.,0.,2.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        R=R*Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        N=r.y[2]*eos.unit_N
        M_binding=N*m_n/M_sun.value
        momentofinertia=r.y[3]/(6.0+2.0*r.y[3])/beta**3
        yR=r.y[4]
        tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
        k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
        tidal=2.0/3.0*(k2/beta**5)
        beta=beta/Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        return [M,R,beta,M_binding,momentofinertia,k2,tidal]

# =============================================================================
# def Tidal_corrected(pc,Preset_Pressure_final,beta,yR,eos):
#     radius_correction=Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos)
#     a=(5-8*beta)/(1-2*beta)
#     b=(beta/(0.5-beta))**2-4*beta/(0.5-beta)
#     print '+++++++++++',a,b,radius_correction
#     zb=yR-2
#     check_solution_type=a**2-4*b
#     check_solution_type_sqrt=np.abs(check_solution_type)**0.5
#     print zb
#     zR=np.where(check_solution_type>0,
#                 ((2*check_solution_type_sqrt/(1-radius_correction**(1/check_solution_type**0.5)*(2*zb-check_solution_type_sqrt+a)/(2*zb+check_solution_type_sqrt+a)))-check_solution_type_sqrt-a)/2.,
#                 (check_solution_type_sqrt*np.tan(np.log(radius_correction)*(check_solution_type_sqrt)+np.arctan((2*zb+a)/check_solution_type_sqrt))-a)/2.)
#     yR=zR+2
#     print zR
#     beta=beta/Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos)
#     tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
#     k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
#     tidal=2.0/3.0*(k2/beta**5)
#     return tidal
# =============================================================================

def get_radius_corr(zR,zb,a,b):
    return ((2*zR-(a**2-4*b)**0.5+a)/(2*zR+(a**2-4*b)**0.5+a)*(2*zb+(a**2-4*b)**0.5+a)/(2*zb-(a**2-4*b)**0.5+a))**((a**2-4*b)**0.5)

def Mass_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):#(this function is used for finding maxmass in FindMaxmass.py ONLY!!
    if(pressure_center[0]<=0):
        return 0
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
    return -r.y[0]*eos.unit_mass/M_sun.value

def f_momentofinertia(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        dmdx=0#np.sqrt(y[1])*eps*den
        dr2dx=0.5/np.pi*den
        djdx=0
        dzdx=(4+y[3])/(eps/p/3.0+1)
        dwdx=0
    else:
        r=y[1]**0.5
        den=p/((y[0]+4*np.pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        dmdx=4*np.pi*eps*y[1]**2*rel*den
        dr2dx=2*y[1]*r*rel*den
        djdx=-4*np.pi*p*y[1]*r/(y[0]+4*np.pi*y[1]*r*p)
        dzdx=((4+y[3])*4*np.pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*r*den
        dwdx=y[3]/2/y[1]*dr2dx
    return np.array([dmdx,dr2dx,djdx,dzdx,dwdx])

def MomentOfInertia_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    xf_array = np.linspace(x0,xf,N)
    p=np.exp(-xf_array)*eos.density_s
    bds=eos.eosBaryonDensity(p)*939.5654
    y_array = lsoda_ode_array(f_momentofinertia,Preset_rtol,[0.,0.,0.,0.,0.],x0,xf_array,eos)
    # runtime warning due to zeros at star center
    M_array=y_array[:,0]*eos.unit_mass/M_sun.value
    r_array=y_array[:,1]**0.5
    beta_array=y_array[:,0]/(r_array)
    R_array=r_array*eos.unit_radius
    j_array=np.exp(y_array[:,2]-y_array[-1,2])
    z_array=y_array[:,3]
    w_array=3./(3.+z_array[-1])*np.exp(y_array[:,4]-y_array[-1,4])
    I_array=r_array*y_array[:,1]*j_array*z_array*w_array/(6*y_array[-1,0]**3)
    beta=y_array[-1,0]/r_array[-1]
    I_array_Lattimer=28*np.pi*p*r_array[-1]**3*(1-1.67*beta-0.6*beta**2)*beta/(3*eos.density_s*y_array[-1,0]*(beta**2+2*p*(1+7*beta)*(1-2*beta)/bds))
    I_array_Lattimer=(1-I_array_Lattimer)*I_array[-1]
    I_array_Lattimer2=I_array[-1]-(6*np.pi*p*r_array[-1]**6/(eos.density_s*y_array[-1,0]**4))*(4+z_array[-1])/(3+z_array[-1])**2
    
    I_array_z3z=r_array**3*z_array/(z_array+3)
    I_array_test1=I_array[-1]-(0.5/(beta_array[-1]**3)-I_array)*16*np.pi*p*r_array**3/(3.*eos.density_s*y_array[:,0])
    I_array_test2=I_array[-1]-(0.5/(beta_array**3)-I_array)*16*np.pi*p*r_array**3/(3.*eos.density_s*y_array[:,0])
    I_array_test3=0.5/(beta_array**3)-(0.5/(beta_array**3)-I_array[-1])/(1-16*np.pi*p*r_array**3/(3.*eos.density_s*y_array[:,0]))
    return  [eos.density_s*np.exp(-xf_array),M_array,R_array,j_array,z_array,w_array,I_array,I_array_Lattimer,I_array_Lattimer2,I_array_z3z,I_array_test1,I_array_test2,I_array_test3]
