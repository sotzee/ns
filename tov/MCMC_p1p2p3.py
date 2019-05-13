#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:20:56 2018

@author: sotzee
"""

import pymc as pm
import numpy as np
from astropy.constants import M_sun
import matplotlib.pyplot as plt
from eos_class import EOS_BPSwithPoly
import scipy.optimize as opt
from scipy.integrate import ode

baryon_density0=0.16/2.7
baryon_density1=1.85*0.16
baryon_density2=3.7*0.16
baryon_density3=7.4*0.16
Preset_Pressure_final=1e-8
Preset_rtol=1e-4
Preset_pressure_center_low=10
Preset_Pressure_final_index=1

from numpy import pi
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

def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    if(MRorMRBIT=='M'):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        return M
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
        return [M,R,beta,yR,tidal]

def Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos):
    X=(eos.eosChempo(pc*Preset_Pressure_final)/(931.171))**2-1
    return beta/(beta+beta*X-0.5*X)
def Mass_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):#(this function is used for finding maxmass in FindMaxmass.py ONLY!!
    if(pressure_center[0]<=0):
        return 0
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
    return -r.y[0]*eos.unit_mass/M_sun.value
def Maxmass(Preset_Pressure_final,Preset_rtol,eos):
    result=opt.minimize(Mass_formax,100.0,tol=0.001,args=(Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
    return [0,result.x[0],-result.fun,result.x[0],-result.fun,result.x[0],-result.fun]

def Properity_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,eos):
    pressure_center=pressure_center_ofmass(ofmass,Preset_pressure_center_low,MaximumMass_pressure_center,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos)
    [M,R,beta,yR,tidal]=MassRadius_function(pressure_center,Preset_Pressure_final**Preset_Pressure_final_index,Preset_rtol,'MRT',eos)
    return [pressure_center,M,R,beta,yR,tidal]

def pressure_center_ofmass(ofmass,Preset_pressure_center_low,Preset_pressure_center_high,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    result=opt.brenth(Ofmass,Preset_pressure_center_low,Preset_pressure_center_high,args=(ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos))
    return result

def Ofmass(pressure_center,ofmass,MassRadius_function,Preset_Pressure_final,Preset_rtol,eos):
    return -ofmass+MassRadius_function(pressure_center,Preset_Pressure_final,Preset_rtol,'M',eos)


from joblib import Parallel, delayed
from multiprocessing import cpu_count

def processInput(Calculation,parameter_list,i,args,num_cores,complete_set):
    result=list()
    for ii in range(complete_set):
        result.append(Calculation(parameter_list,i+num_cores*ii,args))
    return result

def main_parallel(Calculation,parameter_list,args):
    num_cores = cpu_count()-1
    total_num = len(parameter_list)
    complete_set=np.int(total_num/num_cores)
    leftover_num=total_num-complete_set*num_cores

    Output=Parallel(n_jobs=num_cores)(delayed(processInput)(Calculation,parameter_list,i,args,num_cores,complete_set) for i in range(num_cores))
    Output_leftover=Parallel(n_jobs=num_cores)(delayed(Calculation)(parameter_list,i+complete_set*num_cores,args) for i in range(leftover_num))
    result=list()
    for i in range(complete_set):
        for ii in range(num_cores):
            result.append(Output[ii][i])
    for i in range(leftover_num):
            result.append(Output_leftover[i])
    return result

def Calculation_Lambda(parameter_list,x,args):
    if(parameter_list[x]>args[2]):
        return 0
    else:
        ofmass_result=Properity_ofmass(parameter_list[x],Preset_pressure_center_low,args[1],MassRadius,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,args[0])
        return ofmass_result[5]

def Lambda(mass,p1,p2,p3):
    eos=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,p3,baryon_density3])
    maxmass_result=Maxmass(Preset_Pressure_final,Preset_rtol,eos)
    args=[eos,maxmass_result[1],maxmass_result[2]]
    return main_parallel(Calculation_Lambda,mass,args)


from pycbc import cosmology
distance=40.7 # in Mpc
redshift=cosmology.redshift(distance)

import h5py
filename = 'uniform_mass_prior_common_eos_20hz_lowfreq_posteriors.hdf'
fp = h5py.File(filename, "r")
print fp.attrs['variable_args']
m1 = (fp['samples/mass1'][:100]/(1+redshift)).flatten()
m2 = (fp['samples/mass2'][:100]/(1+redshift)).flatten()
Lambdasym = (fp['samples/lambdasym'][:100]).flatten()
q = m2/m1
fp.close()
Lambda1=Lambdasym*(m2/m1)**3
Lambda2=Lambdasym*(m1/m2)**3
array_mass=np.concatenate((m1,m2))
array_log10_Lambda=np.log10(np.concatenate((Lambda1,Lambda2)))

p1_dist=pm.Uniform("p1", 10, 30)
p2_dist=pm.Uniform("p2", 100, 200)
p3_dist=pm.Uniform("p3", 300, 1000)
@pm.deterministic
def center_i(array_mass=array_mass,p1_dist=p1_dist,p2_dist=p2_dist,p3_dist=p3_dist):
    return np.log10(Lambda(array_mass,p1_dist,p2_dist,p3_dist))

@pm.deterministic
def tau_i(centers=array_mass):
    return 0.1*centers

observations = pm.Normal("obs", center_i, tau_i, value=array_log10_Lambda, observed=True)
model = pm.Model([observations, p1_dist,p2_dist,p3_dist])

mcmc = pm.MCMC(model)

from cPickle import dump
mcmc.sample(10)
file_data=open('./p1p2p3_trace.dat1')
dump(np.array([mcmc.trace("p1"),mcmc.trace("p2"),mcmc.trace("p3")]),file_data)
file_data.close()
mcmc.sample(10)
file_data=open('./p1p2p3_trace.dat2')
dump(np.array([mcmc.trace("p1"),mcmc.trace("p2"),mcmc.trace("p3")]),file_data)
file_data.close()
mcmc.sample(10)
file_data=open('./p1p2p3_trace.dat3')
dump(np.array([mcmc.trace("p1"),mcmc.trace("p2"),mcmc.trace("p3")]),file_data)
file_data.close()


plt.figure(figsize=(12.5, 9))
plt.subplot(311)
lw = 1
p1_trace = mcmc.trace("p1")[:]
plt.plot(p1_trace, label="p_1", lw=lw)
# =============================================================================
# plt.title("Traces of unknown parameters")
# leg = plt.legend(loc="upper right")
# leg.get_frame().set_alpha(0.7)
# =============================================================================

plt.subplot(312)
p2_trace = mcmc.trace("p2")[:]
plt.plot(p2_trace, label="p_2", lw=lw)

plt.subplot(313)
p3_trace = mcmc.trace("p3")[:]
plt.plot(p3_trace, label="p_3", lw=lw)



plt.figure(figsize=(11.0, 4))
plt.subplot(2, 2, 1)
plt.hist(p1_trace, bins=30,
             histtype="stepfilled")
plt.subplot(2, 2, 2)
plt.hist(p2_trace, bins=30,
             histtype="stepfilled")
plt.subplot(2, 2, 3)
plt.hist(p3_trace, bins=30,
             histtype="stepfilled")