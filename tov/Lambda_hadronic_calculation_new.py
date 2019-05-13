#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:17:17 2019

@author: sotzee
"""

import cPickle
import numpy as np




f_eos_RMF='./'+dir_name+'/RMF_eos.dat'
error_log=path+dir_name+'/error.log'
eos_flat=main_parallel(Calculation_creat_eos_RMF,zip(args[:,eos_array_logic].transpose(),eos_args[:,eos_array_logic].transpose(),eos_array[:,:,eos_array_logic].transpose((2,0,1))),f_eos_RMF,error_log)

eos_success=[]
matching_success=[]
positive_pressure=[]
for eos_i in eos_flat:
    matching_success.append(eos_i.matching_success)
    eos_success.append(eos_i.eos_success)
    positive_pressure.append(eos_i.positive_pressure)
eos_success=np.array(eos_success)
matching_success=np.array(matching_success)
positive_pressure=np.array(positive_pressure)
print('len(eos)=%d'%len(eos_success))
print('len(eos[positive_pressure])=%d'%len(positive_pressure[positive_pressure]))
print('len(eos[matching_success])=%d'%len(matching_success[matching_success]))
print('len(eos[eos_success])=%d'%len(eos_success[eos_success]))

f_file=open(path+dir_name+'/RMF_eos_logic.dat','wb')
cPickle.dump(eos_array_logic,f_file)
f_file.close()

f_file=open(path+dir_name+'/RMF_eos_success.dat','wb')
cPickle.dump(eos_success,f_file)
f_file.close()

f_file=open(path+dir_name+'/RMF_eos.dat','wb')
cPickle.dump(eos_flat,f_file)
f_file.close()

eos_flat_success=eos_flat
eos_success_logic=eos_array_logic

from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
from Parallel_process import main_parallel

f_maxmass_result=path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
maxmass_result=np.full(eos_success_logic.shape+(3,),np.array([0,0,1]),dtype='float')
maxmass_result[eos_success_logic]=main_parallel(Calculation_maxmass,eos_flat_success,f_maxmass_result,error_log)

print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat_success)))
logic_maxmass=maxmass_result[:,:,:,:,1]>=2
print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_maxmass[eos_success_logic]])))
logic_causality=maxmass_result[:,:,:,:,2]<1
print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_causality[eos_success_logic]])))
logic=np.logical_and(logic_maxmass,logic_causality)
print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic[eos_success_logic]])))

eos_success_maxmass=np.logical_and(logic,eos_success_logic)
for eos_flat_success_i,maxmass_result_i in zip(eos_flat_success.flatten(),maxmass_result[eos_success_logic]):
    eos_flat_success_i.setMaxmass(maxmass_result_i)



f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_success_maxmass.dat','wb')
cPickle.dump(eos_success_maxmass,f_file)
f_file.close()

f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat','wb')
cPickle.dump(eos_flat_success[eos_success_maxmass[eos_success_logic]],f_file)
f_file.close()

print('Calculating properities of 1.4 M_sun star...')
f_onepointfour_result=path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat'
Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_onepointfour_result,error_log)
print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))

print('Calculating mass, compactness and tidal Lambda...')
f_mass_beta_Lambda_result=path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_mass_beta_Lambda_result,error_log)
print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))

print('Calculating binary neutron star...')
f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)










path = "./"
dir_name='Lambda_RMF_calculation_parallel'
error_log=path+dir_name+'/error.log'

f_file=open(path+dir_name+'/RMF_eos_args.dat','rb')
eos_args_logic,args=cPickle.load(f_file)
f_file.close()
args_flat=args.flatten()

f_file=open(path+dir_name+'/RMF_eos_success.dat','rb')
eos_success_logic=cPickle.load(f_file)
f_file.close()

f_file=open(path+dir_name+'/RMF_eos.dat','rb')
eos_flat_success=cPickle.load(f_file)
f_file.close()

    
print('%d EoS built with shape (E_n,L_n,K_n,Q_n)%s.'%(args.size,args.shape[:-1]))
print('%d EoS are successful.'%(len(eos_success_logic[eos_success_logic])))

from Lambda_hadronic_calculation import Calculation_maxmass,Calculation_mass_beta_Lambda,Calculation_onepointfour,Calculation_chirpmass_Lambdabeta6
from Parallel_process import main_parallel

f_maxmass_result=path+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
maxmass_result=np.full((eos_success_logic.shape,3),np.array([0,0,1]),dtype='float')
maxmass_result[eos_success_logic]=main_parallel(Calculation_maxmass,eos_flat_success,f_maxmass_result,error_log)

print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat_success)))
logic_maxmass=maxmass_result[:,1]>=2
print('Maximum mass constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_maxmass[eos_success_logic]])))
logic_causality=maxmass_result[:,2]<1
print('Causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic_causality[eos_success_logic]])))
logic=np.logical_and(logic_maxmass,logic_causality)
print('Maximum mass and causality constrain of %d EoS calculated, %d EoS satisfied.' %(len(eos_flat_success),len(eos_flat_success[logic[eos_success_logic]])))

eos_success_maxmass=np.logical_and(logic,eos_success_logic)
for i in range(len(eos_flat_success)):
    if(eos_success_logic[i]):
        eos_flat_success[i].setMaxmass(maxmass_result[i])
    else:
        eos_flat_success[i].setMaxmass([0,0,1])


f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_success_maxmass.dat','wb')
cPickle.dump(eos_success_maxmass,f_file)
f_file.close()

f_file=open(path+dir_name+'/Lambda_hadronic_calculation_eos_flat_logic.dat','wb')
cPickle.dump(eos_flat_success[eos_success_maxmass[eos_success_logic]],f_file)
f_file.close()

print('Calculating properities of 1.4 M_sun star...')
f_onepointfour_result=path+dir_name+'/Lambda_hadronic_calculation_onepointfour.dat'
Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_onepointfour_result,error_log)
print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))

print('Calculating mass, compactness and tidal Lambda...')
f_mass_beta_Lambda_result=path+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat_success[eos_success_maxmass[eos_success_logic]],f_mass_beta_Lambda_result,error_log)
print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat_success[eos_success_maxmass[eos_success_logic]])))

print('Calculating binary neutron star...')
f_chirpmass_Lambdabeta6_result=path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat'
chirp_q_Lambdabeta6_Lambda1Lambda2=main_parallel(Calculation_chirpmass_Lambdabeta6,np.concatenate((mass_beta_Lambda_result,np.tile(Properity_onepointfour[:,3],(40,1,1)).transpose()),axis=1),f_chirpmass_Lambdabeta6_result,error_log)


# =============================================================================
# 
# Preset_Pressure_final=1e-8
# Preset_rtol=1e-4
# from Lambda_hadronic_calculation import Maxmass,MassRadius,Properity_ofmass
# def Calculation_maxmass(eos_i):
#     try:
#         if(eos_i.positive_pressure):
#             maxmass_result=Maxmass(Preset_Pressure_final,Preset_rtol,eos_i)[1:3]
#             maxmass_result+=[eos_i.eosCs2(maxmass_result[0])]
#         else:
#             maxmass_result=[0,3.,1.]
#     except RuntimeWarning:
#         print('Runtimewarning happens at calculating max mass:')
#         print(eos_i.args)
#     return maxmass_result
# 
# def Calculation_onepointfour(eos_i):
#     try:
#         if(eos_i.positive_pressure):
#             Properity_onepointfour=Properity_ofmass(1.4,10,eos_i.pc_max,MassRadius,Preset_Pressure_final,Preset_rtol,1,eos_i)
#         else:
#             Properity_onepointfour=[0,1.4,15,1,0,1400]
#     except RuntimeWarning:
#         print('Runtimewarning happens at calculating Calculation_onepointfour:')
#         print(eos_i.args)
#     return Properity_onepointfour
# error_log=path+dir_name+'/error.log'
# from Parallel_process import main_parallel
# f_maxmass_result='./'+dir_name+'/Lambda_RMF_calculation_maxmass.dat'
# maxmass_result=main_parallel(Calculation_maxmass,eos_flat,f_maxmass_result,error_log)
# print('Maximum mass configuration of %d EoS calculated.' %(len(eos_flat)))
# 
# eos_success_all=[]
# for i in range(len(eos_flat)):
#     eos_success_all.append(eos_flat[i].setMaxmass(maxmass_result[i]))
# eos_success_all=np.array(eos_success_all)
# eos_array_logic_maxmass=np.copy(eos_array_logic)
# eos_array_logic_maxmass[eos_array_logic]=eos_success_all
# 
# f_onepointfour_result=path+dir_name+'/Lambda_RMF_calculation_onepointfour.dat'
# Properity_onepointfour=main_parallel(Calculation_onepointfour,eos_flat[logic_maxmass],f_onepointfour_result,error_log)
# print('properities of 1.4 M_sun star of %d EoS calculated.' %(len(eos_flat[logic_maxmass])))
# 
# f_file=open(path+dir_name+'/Lambda_RMF_calculation_eos.dat','wb')
# cPickle.dump(eos_rmf,f_file)
# f_file.close()
# 
# f_mass_beta_Lambda_result=path+dir_name+'/Lambda_RMF_calculation_mass_beta_Lambda.dat'
# mass_beta_Lambda_result=main_parallel(Calculation_mass_beta_Lambda,eos_flat[logic],f_mass_beta_Lambda_result,error_log)
# print('mass, compactness and tidal Lambda of %d EoS calculated.' %(len(eos_flat[logic])))
# =============================================================================
