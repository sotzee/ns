
from scipy.integrate import ode
from numpy import pi
import numpy as np
from astropy.constants import M_sun
from scipy.constants import m_n

def f(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        return np.array([np.sqrt(y[1])*eps*den,0.5/pi*den])
    else:
        den=p/((y[0]+4*pi*y[1]**1.5*p)*(eps+p))
        rel=1-2*y[0]/y[1]**0.5
# =============================================================================
#         print p,y,rel,den
#         print [4*pi*eps*y[1]**2*rel*den,2*y[1]**1.5*rel*den]
# =============================================================================
        return np.array([4*pi*eps*y[1]**2*rel*den,2*y[1]**1.5*rel*den])
    
def f_complete(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    baryondensity=eos.eosBaryonDensity(p_dimentionful)/eos.baryon_density_s
    cs2=eos.eosCs2(p_dimentionful)
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)-(8*pi*np.sqrt(y[1])*(eps/3.0+p))**2
        dmdx=np.sqrt(y[1])*eps*den
        dr2dx=0.5/pi*den
        dNdx=np.sqrt(y[1])*baryondensity*den
        dzdx=(4+y[3])/(eps/p/3.0+1)
        dydx=-Q*den/(4*pi)
    else:
        den=p/((y[0]+4*pi*y[1]**1.5*p)*(eps+p))
        rel=1-2*y[0]/y[1]**0.5
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)/rel-(2*p/(den*(eps+p)*y[1]*rel))**2
        dmdx=4*pi*eps*y[1]**2*rel*den
        dr2dx=2*y[1]**1.5*rel*den
        dNdx=4*pi*y[1]**2*baryondensity*np.sqrt(rel)*den
        dzdx=((4+y[3])*4*pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*np.sqrt(y[1])*den
        dydx=-(y[4]**2+(y[4]-6)/rel+y[1]*Q)*np.sqrt(y[1])*rel*den
    return np.array([dmdx,dr2dx,dNdx,dzdx,dydx])



# =============================================================================
# from eos_class import EOS_BPSwithPolyCSS
# baryon_density0=0.16/2.7
# baryon_density1=1.85*0.16
# baryon_density2=3.74*0.16
# baryon_density3=7.4*0.16
# pressure1=10.0
# pressure2=250.
# pressure3=1000.
# pressure_trans=250
# det_density=100
# cs2=1.0/3.0
# args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# a=EOS_BPSwithPolyCSS(args)
# 
# from rk4 import rk4
# print 'xxxxxxxxxxxxxx'
# pressure_center=300.
# Preset_Pressure_final=1e-7
# value=500
# x0 = -np.log(pressure_center/a.density_s)
# xf = x0-np.log(Preset_Pressure_final)
# y0 = [0,0]
# vx, vy = rk4(f, x0, y0, xf, value, a.eosBPSwithPoly)
# print [vy[value][0]*a.eosBPSwithPoly.unit_mass,np.sqrt(vy[value][1])*a.eosBPSwithPoly.unit_radius]
# 
# y0 = [0,0,0,0,2]
# vx, vy = rk4(f_complete, x0, y0, xf, value, a.eosBPSwithPoly)
# print [vy[value][0]*a.eosBPSwithPoly.unit_mass,np.sqrt(vy[value][1])*a.eosBPSwithPoly.unit_radius,vy[value][2],vy[value][3],vy[value][4]]
# 
# =============================================================================


def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)    
    if(MRorMRBIT=='MR'):
        r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value([0,0], x0).set_f_params(eos)
        r.integrate(xf)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        return [M,R]
    elif(MRorMRBIT=='MRBIT'):
        r = ode(f_complete).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value([0,0,0,0,2], x0).set_f_params(eos)
        r.integrate(xf)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        N=r.y[2]*eos.unit_N
        M_binding=N*m_n/M_sun.value
        momentofinertia=r.y[3]/(6.0+2.0*r.y[3])/beta**3
        yR=r.y[4]
        tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
        k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
        tidal=2.0/3.0*(k2/beta**5)
        return [M,R,beta,M_binding,momentofinertia,yR,tidal]

def Mass_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):#(this function is used for finding maxmass in FindMaxmass.py ONLY!!
    if(pressure_center[0]<=0):
        return 0
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
    r.set_initial_value([0,0], x0).set_f_params(eos)
    r.integrate(xf)
    return -r.y[0]*eos.unit_mass/M_sun.value

from tov_CSS import Integration_CSS
def MassRadius_transition(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
    if(pressure_center<=eos.pressure_trans):
        return MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos)
    else:
        x0 = -np.log(pressure_center/eos.density_s)
        xt = -np.log(eos.pressure_trans/eos.density_s)
        #xf = xt-np.log(Preset_Pressure_final)
        xf = x0-np.log(Preset_Pressure_final)
        if(MRorMRBIT=='MR'):
            yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:3]
            r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
            r.set_initial_value(yt, xt).set_f_params(eos)
            r.integrate(xf)
            M=r.y[0]*eos.unit_mass/M_sun.value
            R=r.y[1]**0.5*eos.unit_radius
            return [M,R]
        elif(MRorMRBIT=='MRBIT'):
            yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:6]
            yt[4]-=eos.det_density/eos.density_s/yt[0]*4*np.pi*yt[1]**1.5
            r = ode(f_complete).set_integrator('lsoda',rtol=Preset_rtol)
            r.set_initial_value(yt, xt).set_f_params(eos)
            r.integrate(xf)
            M=r.y[0]*eos.unit_mass/M_sun.value
            R=r.y[1]**0.5*eos.unit_radius
            beta=r.y[0]/R*eos.unit_radius
            N=r.y[2]*eos.unit_N
            M_binding=N*m_n/M_sun.value
            momentofinertia=r.y[3]/(6.0+2.0*r.y[3])/beta**3
            yR=r.y[4]
            tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
            k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
            tidal=2.0/3.0*(k2/beta**5)
            return [M,R,beta,M_binding,momentofinertia,yR,tidal]

def Mass_transition_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):
    if(pressure_center[0]<=0):
        return 0
    print pressure_center[0],eos.density_s
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    if(pressure_center<=eos.pressure_trans):
        r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value([0,0], x0).set_f_params(eos)
        r.integrate(xf)
        print r.y[0]*eos.unit_mass/M_sun.value
        return -r.y[0]*eos.unit_mass/M_sun.value
    else:
        xt = -np.log(eos.pressure_trans/eos.density_s)
        #xf = xt-np.log(Preset_Pressure_final)
        yt=Integration_CSS(-np.log(pressure_center/eos.eosCSS.density_s),-np.log(eos.pressure_trans/eos.eosCSS.density_s),eos.eosCSS)[1:3]
        r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value(yt, xt).set_f_params(eos)
        r.integrate(xf)
        print r.y[0]*eos.unit_mass/M_sun.value
        return -r.y[0]*eos.unit_mass/M_sun.value

# =============================================================================
# from eos_class import EOS_BPSwithPolyCSS
# baryon_density0=0.16/2.7
# baryon_density1=1.85*0.16
# baryon_density2=3.74*0.16
# baryon_density3=7.4*0.16
# pressure1=20.0
# pressure2=600.
# pressure3=1000.
# pressure_trans=40.
# det_density=200.
# cs2=3./12
# args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# a=EOS_BPSwithPolyCSS(args)
# #print MassRadius_transition(800,1e-7,1e-5,'MRBIT',a)
# print Mass_transition_formax([800],1e-7,1e-5,a)
# =============================================================================


# =============================================================================
# #print MassRadius(300,1e-7,1e-2,'MRBIT',a)150
# from eos_class import EOS_CSS
# abb=EOS_CSS([150,0,0.16,12./12])
# from tov_CSS import MassRadius_CSS
# pc=3.034*150
# print('xxxxxxxxxxxxx')
# abc=MassRadius(pc,1e-7,1e-5,'MRBIT',abb)
# print(abc)
# print('xxxxxxxxxxxxx')
# abc=MassRadius_CSS(pc,'MRBIT',abb)
# print(abc)
# print('xxxxxxxxxxxxx')
# #print(Integration_CSS(-np.log(pc/abb.density_s),np.log(2)-np.log(pc/abb.density_s),abb))
# 
# from rk4 import rk4
# Preset_Pressure_final=1e-7
# value=200
# x0 = -np.log(pc/abb.density_s)
# xf = x0-np.log(Preset_Pressure_final)
# y0 = [0,0,0,0,2]
# vx, vy = rk4(f_complete, x0, y0, xf, value, abb)
# print [vy[value][0]*abb.unit_mass/M_sun.value,np.sqrt(vy[value][1])*abb.unit_radius,vy[value][2],vy[value][3],vy[value][4]]
# =============================================================================
