#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:31:38 2018

@author: sotzee
"""
import numpy as np
from tov_f import f,f_baryon_number,f_tidal,f_MRI,f_complete,lsoda_ode
from astropy.constants import M_sun
from scipy.constants import m_n
from tov_CSS import Integration_CSS
from MassRadius_hadronic import MassRadius
def MassRadius_transition(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
    if(pressure_center<=eos.pressure_trans):
        return MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos)
    else:
        x0 = -np.log(pressure_center/eos.density_s)
        xt = -np.log(eos.pressure_trans/eos.density_s)
        #xf = xt-np.log(Preset_Pressure_final)
        xf = x0-np.log(Preset_Pressure_final)
        if(MRorMRBIT=='M'):
            yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:3]
            yt[0:2]=[yt[0]*eos.eosCSS.unit_mass/eos.unit_mass,yt[1]*(eos.eosCSS.unit_radius/eos.unit_radius)**2]
            r = lsoda_ode(f,Preset_rtol,yt,xt,xf,eos)
            M=r.y[0]*eos.unit_mass/M_sun.value
            return M
        if(MRorMRBIT=='B'):
            yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:4]
            yt[0:3]=[yt[0]*eos.eosCSS.unit_mass/eos.unit_mass,yt[1]*(eos.eosCSS.unit_radius/eos.unit_radius)**2,yt[2]*eos.eosCSS.unit_N/eos.unit_N]
            r = lsoda_ode(f_baryon_number,Preset_rtol,yt,xt,xf,eos)
            M_binding=r.y[2]*eos.unit_N*m_n/M_sun.value
            return M_binding
        elif(MRorMRBIT=='MR'):
            yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:3]
            yt[0:2]=[yt[0]*eos.eosCSS.unit_mass/eos.unit_mass,yt[1]*(eos.eosCSS.unit_radius/eos.unit_radius)**2]
            r = lsoda_ode(f,Preset_rtol,yt,xt,xf,eos)
            M=r.y[0]*eos.unit_mass/M_sun.value
            R=r.y[1]**0.5*eos.unit_radius
            return [M,R]
        elif(MRorMRBIT=='MRBIT'):
            yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:6]
            yt[0:3]=[yt[0]*eos.eosCSS.unit_mass/eos.unit_mass,yt[1]*(eos.eosCSS.unit_radius/eos.unit_radius)**2,yt[2]*eos.eosCSS.unit_N/eos.unit_N]
            yt[4]-=eos.det_density/eos.density_s/yt[0]*4*np.pi*yt[1]**1.5
            r = lsoda_ode(f_complete,Preset_rtol,yt,xt,xf,eos)
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
            beta=beta*Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
            return [M,R,beta,M_binding,momentofinertia,k2,tidal]

def Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos):
    X=(eos.eosChempo(pc*Preset_Pressure_final)/(931.171))**2-1
    return beta/(beta+beta*X-0.5*X)

def Mass_transition_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):
    if(pressure_center[0]<=0):
        return 0
    x0 = -np.log(pressure_center[0]/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    if(pressure_center[0]<=eos.pressure_trans):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
    else:
        xt = -np.log(eos.pressure_trans/eos.density_s)
        yt=Integration_CSS(-np.log(pressure_center[0]/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:3]
        yt[0:2]=[yt[0]*eos.eosCSS.unit_mass/eos.unit_mass,yt[1]*(eos.eosCSS.unit_radius/eos.unit_radius)**2]
        r = lsoda_ode(f,Preset_rtol,yt,xt,xf,eos)
    return -r.y[0]*eos.unit_mass/M_sun.value