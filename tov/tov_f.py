
from scipy.integrate import ode
from numpy import pi
import numpy as np
from astropy.constants import M_sun
from scipy.constants import m_n

def f(x, y, eos):
    p=np.exp(-x)
    eps=eos.eosDensity(p)/eos.density_s
    p=p/eos.density_s
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
    eps=eos.eosDensity(p)/eos.density_s
    baryondensity=eos.eosBaryonDensity(p)/eos.baryon_density_s
    cs2=eos.eosCs2(p)
    p=p/eos.density_s
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



from eos_class import EOS_BPSwithPolyCSS
baryon_density0=0.16/2.7
baryon_density1=1.85*0.16
baryon_density2=3.74*0.16
baryon_density3=7.4*0.16
pressure1=10.0
pressure2=150.
pressure3=1000.
pressure_trans=1000
det_density=100
cs2=1.0/3.0
args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
a=EOS_BPSwithPolyCSS(args)

# =============================================================================
# from rk4 import rk4
# print 'xxxxxxxxxxxxxx'
# pressure_center=300.
# Preset_Pressure_final=1e-7
# value=500
# x0 = -np.log(pressure_center)
# xf = x0-np.log(Preset_Pressure_final)
# y0 = [0,0]
# vx, vy = rk4(f, x0, y0, xf, value, a)
# print [vy[value][0]*a.unit_mass,np.sqrt(vy[value][1])*a.unit_radius]
# 
# y0 = [0,0,0,0,2]
# vx, vy = rk4(f_complete, x0, y0, xf, value, a)
# print [vy[value][0]*a.unit_mass,np.sqrt(vy[value][1])*a.unit_radius,vy[value][2],vy[value][3],vy[value][4]]
# =============================================================================
def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
    pressure_center=300.
    Preset_Pressure_final=1e-7
    x0 = -np.log(pressure_center)
    xf = x0-np.log(Preset_Pressure_final)    
    if(MRorMRBIT=='MR'):
        r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value([0,0], x0).set_f_params(a)
        r.integrate(xf)
        M=r.y[0]*a.unit_mass/M_sun.value
        R=r.y[1]**0.5*a.unit_radius
        return [M,R]
    elif(MRorMRBIT=='MRBIT'):
        r = ode(f_complete).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value([0,0,0,0,2], x0).set_f_params(a)
        r.integrate(xf)
        M=r.y[0]*a.unit_mass/M_sun.value
        R=r.y[1]**0.5*a.unit_radius
        N=r.y[1]*a.unit_N
        M_binding=N*m_n
        beta=r.y[0]/R*a.unit_radius
        momentofinertia=r.y[3]/(6.0+2.0*r.y[3])/beta**3
        yR=r.y[4]
        tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
        k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
        tidal=2.0/3.0*(k2/beta**5)
        return [M,R,beta,M_binding,momentofinertia,yR,tidal]
# =============================================================================
#     while r.successful() and r.t < xf:
#         r.integrate(r.t+1)
#         print("%s %s" % (r.t, r.y))
# =============================================================================
