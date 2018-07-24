
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
        return np.array([0,0.5/pi*den])
    else:
        r=y[1]**0.5
        den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        return np.array([4*pi*eps*y[1]**2*rel*den,2*y[1]*r*rel*den])

def f_baryon_number(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    baryondensity=eos.eosBaryonDensity(p_dimentionful)/eos.baryon_density_s
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        dmdx=0
        dr2dx=0.5/pi*den
        dNdx=0
    else:
        r=y[1]**0.5
        r4=y[1]**2
        den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        dmdx=4*pi*eps*r4*rel*den
        dr2dx=2*y[1]*r*rel*den
        dNdx=4*pi*r4*baryondensity*np.sqrt(rel)*den
    return np.array([dmdx,dr2dx,dNdx])

def f_MRI(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        dmdx=0
        dr2dx=0.5/pi*den
        dzdx=(4+y[2])/(eps/p/3.0+1)
    else:
        r=y[1]**0.5
        r4=y[1]**2
        den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        dmdx=4*pi*eps*r4*rel*den
        dr2dx=2*y[1]*r*rel*den
        dzdx=((4+y[2])*4*pi*(eps+p)*y[1]-rel*y[2]*(3+y[2]))*r*den
    print p,eps,[dmdx,dr2dx,dzdx]
    return np.array([dmdx,dr2dx,dzdx])

def f_complete(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    baryondensity=eos.eosBaryonDensity(p_dimentionful)/eos.baryon_density_s
    cs2=eos.eosCs2(p_dimentionful)
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)#-(8*pi*np.sqrt(y[1])*(eps/3.0+p))**2
        dmdx=0#np.sqrt(y[1])*eps*den
        dr2dx=0.5/pi*den
        dNdx=0#np.sqrt(y[1])*baryondensity*den
        dzdx=(4+y[3])/(eps/p/3.0+1)
        dydx=-Q*den/(4*pi)
    else:
        r=y[1]**0.5
        r4=y[1]**2
        den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)/rel-(2*p/(den*(eps+p)*y[1]*rel))**2
        dmdx=4*pi*eps*r4*rel*den
        dr2dx=2*y[1]*r*rel*den
        dNdx=4*pi*r4*baryondensity*np.sqrt(rel)*den
        dzdx=((4+y[3])*4*pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*r*den
        dydx=-(y[4]**2+(y[4]-6)/rel+y[1]*Q)*r*rel*den
    return np.array([dmdx,dr2dx,dNdx,dzdx,dydx])

def lsoda_ode(function,Preset_rtol,y0,x0,xf,para):
    r = ode(function).set_integrator('lsoda',rtol=Preset_rtol,nsteps=1000)
    r.set_initial_value(y0, x0).set_f_params(para)
    r.integrate(xf)
    i=0
    while(i<5 and not r.successful()):
        r.set_initial_value(r.y, r.t).set_f_params(para)
        r.integrate(xf)
        i+=1
    return r

def lsoda_ode_array(function,Preset_rtol,y0,x0,xf_array,para):
    r = ode(function).set_integrator('lsoda',rtol=Preset_rtol,nsteps=1000)
    r.set_initial_value(y0, x0).set_f_params(para)
    y_array=[]
    for xf in xf_array:
        r.integrate(xf)
        i=0
        while(i<5 and not r.successful()):
            r.set_initial_value(r.y, r.t).set_f_params(para)
            r.integrate(xf)
            i+=1
        y_array.append(r.y)
    return np.array(y_array)

def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    if(MRorMRBIT=='M'):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        return M
    if(MRorMRBIT=='B'):
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
        return [M,R,beta,M_binding,momentofinertia,yR,tidal]

def Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos):
    X=(eos.eosChempo(pc*Preset_Pressure_final)/(931.171))**2-1
    return beta/(beta+beta*X-0.5*X)

def Tidal_corrected(pc,Preset_Pressure_final,beta,yR,eos):
    radius_correction=Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos)
    a=(5-8*beta)/(1-2*beta)
    b=(beta/(0.5-beta))**2-4*beta/(0.5-beta)
    print '+++++++++++',a,b,radius_correction
    zb=yR-2
    check_solution_type=a**2-4*b
    check_solution_type_sqrt=np.abs(check_solution_type)**0.5
    print zb
    zR=np.where(check_solution_type>0,
                ((2*check_solution_type_sqrt/(1-radius_correction**(1/check_solution_type**0.5)*(2*zb-check_solution_type_sqrt+a)/(2*zb+check_solution_type_sqrt+a)))-check_solution_type_sqrt-a)/2.,
                (check_solution_type_sqrt*np.tan(np.log(radius_correction)*(check_solution_type_sqrt)+np.arctan((2*zb+a)/check_solution_type_sqrt))-a)/2.)
    yR=zR+2
    print zR
    beta=beta/Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos)
    tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
    k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
    tidal=2.0/3.0*(k2/beta**5)
    return tidal

def get_radius_corr(zR,zb,a,b):
    return ((2*zR-(a**2-4*b)**0.5+a)/(2*zR+(a**2-4*b)**0.5+a)*(2*zb+(a**2-4*b)**0.5+a)/(2*zb-(a**2-4*b)**0.5+a))**((a**2-4*b)**0.5)

def Mass_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):#(this function is used for finding maxmass in FindMaxmass.py ONLY!!
    if(pressure_center[0]<=0):
        return 0
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
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
            return [M,R,beta,M_binding,momentofinertia,yR,tidal]

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

# =============================================================================
# def f_momentofinertia(x, y, eos):
#     p=np.exp(-x)
#     p_dimentionful=p*eos.density_s
#     eps=eos.eosDensity(p_dimentionful)/eos.density_s
#     if(y[1]==0):
#         den=p/((eps+p)*(eps/3.0+p))
#         dmdx=0#np.sqrt(y[1])*eps*den
#         dr2dx=0.5/pi*den
#         djdx=0
#         dzdx=(4+y[3])/(eps/p/3.0+1)
#         dwdx=0
#     else:
#         r=y[1]**0.5
#         den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
#         rel=1-2*y[0]/r
#         dmdx=4*pi*eps*y[1]**2*rel*den
#         dr2dx=2*y[1]*r*rel*den
#         djdx=-4*pi*p*y[1]*r/(y[0]+4*pi*y[1]*r*p)
#         dzdx=((4+y[3])*4*pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*r*den
#         dwdx=y[3]/2/y[1]*dr2dx
#     return np.array([dmdx,dr2dx,djdx,dzdx,dwdx])
# 
# def MomentOfInertia_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
#     x0 = -np.log(pressure_center/eos.density_s)
#     xf = x0-np.log(Preset_Pressure_final)
#     xf_array = np.linspace(x0,xf,N)
#     y_array = lsoda_ode_array(f_momentofinertia,Preset_rtol,[0.,0.,0.,0.,0.],x0,xf_array,eos)
# 
#     M_array=y_array[:,0]*eos.unit_mass/M_sun.value
#     r_array=y_array[:,1]**0.5
#     R_array=r_array*eos.unit_radius
#     j_array=np.exp(y_array[:,2]-y_array[-1,2])
#     z_array=y_array[:,3]
#     w_array=3./(3.+z_array[-1])*np.exp(y_array[:,4]-y_array[-1,4])
#     return  [eos.density_s*np.exp(-xf_array),M_array,R_array,j_array,z_array,w_array,r_array*y_array[:,1]*j_array*z_array*w_array/(6*y_array[-1,0]**3)]
# mri_result=MomentOfInertia_profile(pc,Preset_Pressure_final,1e-4,100,eos)
# 
# def MassRadius_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
#     x0 = -np.log(pressure_center/eos.density_s)
#     xf = x0-np.log(Preset_Pressure_final)
#     xf_array = np.linspace(xf,x0,N)
#     r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
#     print r.y
#     r = lsoda_ode(f,Preset_rtol,r.y,xf,x0,eos)
#     print r.y
#     y_array = lsoda_ode_array(f,Preset_rtol,[r.y[0],r.y[1]],xf,xf_array,eos)
#     M_array=y_array[:,0]*eos.unit_mass/M_sun.value
#     R_array=y_array[:,1]**0.5*eos.unit_radius
#     return [M_array,R_array]
# mri_result=MassRadius_profile(pc,Preset_Pressure_final,1e-4,100,eos)
# 
# 
# def f_MRI(x, y, eos):
#     p=np.exp(-x)
#     p_dimentionful=p*eos.density_s
#     eps=eos.eosDensity(p_dimentionful)/eos.density_s
#     if(y[1]==0):
#         den=p/((eps+p)*(eps/3.0+p))
#         dmdx=0
#         dr2dx=0.5/pi*den
#         dzdx=(4+y[2])/(eps/p/3.0+1)
#     else:
#         r=y[1]**0.5
#         r4=y[1]**2
#         den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
#         rel=1-2*y[0]/r
#         dmdx=4*pi*eps*r4*rel*den
#         dr2dx=2*y[1]*r*rel*den
#         dzdx=((4+y[2])*4*pi*(eps+p)*y[1]-rel*y[2]*(3+y[2]))*r*den
#     #print p,eps,[dmdx,dr2dx,dzdx]
#     return np.array([dmdx,dr2dx,dzdx])
# def MomentOfInertia_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
#     x0 = -np.log(pressure_center/eos.density_s)
#     xf = x0-np.log(Preset_Pressure_final)
#     xf_array = np.linspace(xf,x0,N)
#     r = lsoda_ode(f_MRI,Preset_rtol,[0.,0.,0],x0,xf,eos)
#     print r.y
#     r = lsoda_ode(f_MRI,Preset_rtol,r.y,xf,x0,eos)
#     print r.y
#     beta=r.y[0]/r.y[1]**0.5 
#     I=r.y[2]/(6.0+2.0*r.y[2])/beta**3
#     y_array = lsoda_ode_array(f_momentofinertia,Preset_rtol,[r.y[0],r.y[1],1.,r.y[2],1.],xf,xf_array,eos)
#     M_array=y_array[:,0]*eos.unit_mass/M_sun.value
#     R_array=y_array[:,1]**0.5*eos.unit_radius
#     I_array=y_array[:,2]*y_array[:,3]*y_array[:,4]
#     I_array=I*I_array/I_array.max()
#     #print y_array[:,2]
#     #print y_array[:,3]
#     #print y_array[:,4]
#     return [M_array,R_array,I_array]
# mri_result=MomentOfInertia_profile(pc,Preset_Pressure_final,1e-4,100,eos)
# =============================================================================



# =============================================================================
# from eos_class import EOS_BPSwithPolyCSS
# #args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# from fractions import Fraction
# a=EOS_BPSwithPolyCSS([0.059259259259259255, 16.0, 0.29600000000000004, 267.2510854860387, 0.5984, 5000.0, 1.1840000000000002, 51.02970970539573, 410.826065399642474, Fraction(1, 1)])
# print MassRadius_transition(52.02970970539573,1e-8,1e-4,'MRBIT',a)
# =============================================================================

#print MassRadius(61.37455071231708,1e-7,1e-5,'MRBIT',a)
#print MassRadius(61.37455071231708,1e-7,1e-5,'MR',a)


# =============================================================================
# #print MassRadius(300,1e-7,1e-2,'MRBIT',a)150
# from eos_class import EOS_CSS
# abb=EOS_CSS([150,0,0.16,4./12])
# from tov_CSS import MassRadius_CSS
# pc=3.034*150
# print('xxxxxxxxxxxxx')
# abc=MassRadius(pc,1e-7,1e-5,'MRBIT',abb)
# print(abc)
# print('xxxxxxxxxxxxx')
# abc=MassRadius_CSS(pc,'MRBIT',abb)
# print(abc)
# print('xxxxxxxxxxxxx')
# print(Integration_CSS(-np.log(pc/abb.density_s),np.log(2)-np.log(pc/abb.density_s),abb))
# =============================================================================

# =============================================================================
# from rk4 import rk4
# Preset_Pressure_final=1e-7
# value=200
# x0 = -np.log(pc/abb.density_s)
# xf = x0-np.log(Preset_Pressure_final)
# y0 = [0,0,0,0,2]
# vx, vy = rk4(f_complete, x0, y0, xf, value, abb)
# print [vy[value][0]*abb.unit_mass/M_sun.value,np.sqrt(vy[value][1])*abb.unit_radius,vy[value][2],vy[value][3],vy[value][4]]
# =============================================================================
