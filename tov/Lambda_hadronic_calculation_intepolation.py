#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:43:21 2018

@author: sotzee
"""

from eos_class import EOS_BPS,EOS_BPSwithPoly
#import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
baryon_density_s= 0.16
baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.7*0.16
baryon_density3 = 7.4*0.16
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
Preset_rtol=1e-4
Preset_Pressure_final=1e-8

def Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i=np.log(pressure_i/pressure_i_minus)/np.log(baryon_density_i/baryon_density_i_minus)
    return gamma_i,(density_i_minus-pressure_i_minus/(gamma_i-1))*\
            (pressure_i/pressure_i_minus)**(1./gamma_i)+pressure_i/(gamma_i-1)

def causality_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i,density_i=Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus)
    return gamma_i*pressure_i/(density_i+pressure_i)-1

def causality_p2(p1):
    density1=Density_i(p1,baryon_density1,pressure0,baryon_density0,density0)[1]
    return opt.newton(causality_i,200.,args=(baryon_density2,p1,baryon_density1,density1))

def p3_max(pressure1,pressure2):
    density1=Density_i(pressure1,baryon_density1,pressure0,baryon_density0,density0)[1]
    density2=Density_i(pressure2,baryon_density2,pressure1,baryon_density1,density1)[1]
    gamma3_max=1+density2/pressure2
    return pressure2*(baryon_density3/baryon_density2)**gamma3_max

import cPickle
dir_name='Lambda_hadronic_calculation'
f_file=open('./'+dir_name+'/Lambda_hadronic_calculation_p1p2p3_eos.dat','rb')
p1p2p3,eos=np.array(cPickle.load(f_file))
f_file.close()
shape=np.shape(p1p2p3)[0:3]+(-1,)

f_maxmass_result='./'+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
f_file=open(f_maxmass_result,'rb')
maxmass_result=np.reshape(cPickle.load(f_file),shape)
f_file.close()

f_mass_beta_Lambda_result = './'+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
f_file=open(f_mass_beta_Lambda_result,'rb')
mass_beta_Lambda_result=np.array(cPickle.load(f_file))
f_file.close()
mass=np.reshape(mass_beta_Lambda_result[:,0],shape)
beta=np.reshape(mass_beta_Lambda_result[:,1],shape)
Lambda=np.reshape(mass_beta_Lambda_result[:,2],shape)

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
p1_range=(p1p2p3[:,0,0,0].min()[0],p1p2p3[:,0,0,0].max()[0])
p2_range=(100,'causal maximum')
p3_range=('np.max([1.2*p2,250])','p3_max')

p2_causal=[]
for p_i in np.linspace(*p1_range):
    p2_causal.append(causality_p2(p_i))
causality_p2_int=interp1d(np.linspace(*p1_range),p2_causal)
#plt.plot(np.linspace(*p1_range),p2_causal,'.')
#plt.plot(np.linspace(*(p1_range)+(200,)),causality_p2_int(np.linspace(*(p1_range)+(200,))))

maxmass_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2])), maxmass_result[:,:,:,1])
pc_max_int  = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2])), maxmass_result[:,:,:,0])
cs2_max_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2])), maxmass_result[:,:,:,2])

def get_p1p2p3_i(p1,p2,p3):
    p1_i=(p1-p1_range[0])/(p1_range[1]-p1_range[0])*(shape[0]-1)
    p2_max=causality_p2_int(p1)
    p2_i=(p2-p2_range[0])/(p2_max-p2_range[0])*(shape[1]-1)
    log_p3_min=np.log(np.max([1.2*p2,250]))
    log_p3_max=np.log(p3_max(p1,p2))
    p3_i=(np.log(p3)-log_p3_min)/(log_p3_max-log_p3_min)*(shape[2]-1)
    return [p1_i,p2_i,p3_i]

def get_p1p2p3(p1_i,p2_i,p3_i):
    p1=p1_i/(shape[0]-1)*(p1_range[1]-p1_range[0])+p1_range[0]
    p2_max=causality_p2_int(p1)
    p2=p2_i/(shape[1]-1)*(p2_max-p2_range[0])+p2_range[0]
    log_p3_min=np.log(np.max([1.2*p2,250]))
    log_p3_max=np.log(p3_max(p1,p2))
    p3=np.exp(p3_i/(shape[2]-1)*(log_p3_max-log_p3_min)+log_p3_min)
    return [p1,p2,p3]

def get_maxmass(p1,p2,p3):
    p1p2p3_i=get_p1p2p3_i(p1,p2,p3)
    return pc_max_int(p1p2p3_i),maxmass_int(p1p2p3_i),cs2_max_int(p1p2p3_i)

star_N=np.shape(mass)[-1]
mass_int  = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2]), range(star_N)), mass)
log_tidal_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2]), range(star_N)), np.log(Lambda))

def Ofmass(star_i,ofmass,p1p2p3_i):
    return mass_int(p1p2p3_i+[star_i])[0]-ofmass
def log_tidal_ofmass(ofmass,p1p2p3_i):
    if(ofmass>mass_int(p1p2p3_i+[0])):
        return 0
    else:
        star_i=opt.brenth(Ofmass,0.,star_N-1,rtol=Preset_rtol,args=(ofmass,p1p2p3_i,))
        return log_tidal_int(p1p2p3_i+[star_i])[0]

mass_grid=np.linspace(1.0,2.0,37)
def Calculation_log_Lambda_mass_grid(mass_beta_Lambda,ii):
    log_Lambda_mass_grid_ii=[]
    k=ii%shape[2]
    j=((ii-k)/shape[2])%shape[1]
    i=((ii-k)/shape[2]-j)/shape[1]
    #print(i,j,k)
    for mass_i in mass_grid:
        #print(mass_i)
        log_Lambda_mass_grid_ii.append(log_tidal_ofmass(mass_i,[i,j,k]))
    return log_Lambda_mass_grid_ii

f_log_Lambda_mass_grid_result='./'+dir_name+'/Lambda_hadronic_calculation_log_Lambda_mass_grid.dat'
#from Parallel_process import main_parallel
#main_parallel(Calculation_log_Lambda_mass_grid,mass_beta_Lambda_result,f_log_Lambda_mass_grid_result,0)
f_file=open(f_log_Lambda_mass_grid_result,'rb')
log_Lambda_mass_grid=np.reshape(np.array(cPickle.load(f_file)),shape)
f_file.close()
log_Lambda_mass_grid_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2]), mass_grid), log_Lambda_mass_grid)


#test intepolation accuracy
from Lambda_hadronic_calculation import MassRadius,Properity_ofmass

ijk_rand=(np.random.rand(5,5,5,3)*39)
log_lambda_exact=[]
log_lambda_int=[]
mass_rand=[]
for i in range(np.shape(ijk_rand)[0]):
    for j in range(np.shape(ijk_rand)[1]):
        for k in range(np.shape(ijk_rand)[2]):
            mass_rand.append(1+np.min([maxmass_result[i][j][k][1]-1,1])*np.random.rand())
            p1,p2,p3=get_p1p2p3(*ijk_rand[i,j,k])
            print ijk_rand[i,j,k]
            print p1,p2,p3,mass_rand[-1]
            log_lambda_exact.append(np.log(Properity_ofmass(mass_rand[-1], 10, maxmass_result[i][j][k][0], MassRadius, Preset_Pressure_final, Preset_rtol,1 ,EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,p3,baryon_density3]))[-1]))
            log_lambda_int.append(log_Lambda_mass_grid_int([ijk_rand[i,j,k,0],ijk_rand[i,j,k,1],ijk_rand[i,j,k,2],mass_rand[-1]])[0])
            print(log_lambda_exact[-1],log_lambda_int[-1])
log_lambda_exact=np.array(log_lambda_exact)
log_lambda_int=np.array(log_lambda_int)
mass_rand=np.array(mass_rand)
import matplotlib.pyplot as plt
plt.plot(mass_rand,(log_lambda_int-log_lambda_exact)/log_lambda_exact)

#log_Lambda_mass_grid=[]
#for i in range(shape[0]):
#    log_Lambda_mass_grid.append([])
#    for j in range(shape[1]):
#        print[i,j]
#        log_Lambda_mass_grid[i].append([])
#        for k in range(shape[2]):
#            print[i,j,k]
#            log_Lambda_mass_grid[i][j].append([])
#            for mass_i in mass_grid:
#                #print mass_i
#                log_Lambda_mass_grid[i][j][k].append(log_tidal_ofmass(mass_i,[i,j,k]))

                
# =============================================================================
# def get_mass(p1,p2,p3,star_i):
#     maxmass_result_int=get_maxmass(p1,p2,p3)
# =============================================================================
    
