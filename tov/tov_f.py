
from scipy.integrate import ode
from numpy import pi
import numpy as np

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

def f_tidal(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    cs2=eos.eosCs2(p_dimentionful)
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        Q=4*pi*((5-y[2])*eps+(9+y[2])*p+(eps+p)/cs2)#-(8*pi*np.sqrt(y[1])*(eps/3.0+p))**2
        dmdx=0#np.sqrt(y[1])*eps*den
        dr2dx=0.5/pi*den
        dydx=-Q*den/(4*pi)
    else:
        r=y[1]**0.5
        r4=y[1]**2
        den=p/((y[0]+4*pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        Q=4*pi*((5-y[2])*eps+(9+y[2])*p+(eps+p)/cs2)/rel-(2*p/(den*(eps+p)*y[1]*rel))**2
        dmdx=4*pi*eps*r4*rel*den
        dr2dx=2*y[1]*r*rel*den
        dydx=-(y[2]**2+(y[2]-6)/rel+y[1]*Q)*r*rel*den
    return np.array([dmdx,dr2dx,dydx])

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

# =============================================================================
# import matplotlib.pyplot as plt
# label_list=['M','R','j','y','$\omega$','I','I_Lattimer1999','I_Lattimer_TOV_notes','$\\frac{yr^3}{y+3}$','I_APPROX_I','I_APPROX_II','I_APPROX_III']
# mri_result=MomentOfInertia_profile(379.908447265625,1e-20,1e-8,100,eos)
# for i in range(5):
#     plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],label=label_list[i])
# i=5
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],'k',label=label_list[i])
# i=6
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],':',label=label_list[i])
# i=7
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],':',label=label_list[i])
# i=8
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],'--',label=label_list[i])
# i=9
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],'--',label=label_list[i])
# i=10
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],'--',label=label_list[i])
# i=11
# plt.plot(mri_result[0],mri_result[i+1]/mri_result[i+1][-1],'-.',label=label_list[i])
# #plt.plot([0.5006],[0.986],'o')
# #plt.xscale('log')
# plt.xlim(0,2)
# plt.ylim(0.96,1.02)
# plt.xlabel('Crustal pressure(MeV fm$^{-3}$)')
# plt.ylabel('normalized to surface value')
# plt.legend(frameon=False,fontsize=8)
# plt.savefig('CrustalI.eps', format='eps', dpi=1000)
# =============================================================================



# =============================================================================
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
