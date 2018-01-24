#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:20:02 2018

@author: sotzee
"""

import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from time import time
import pickle
#from tov_f import Mass_transition_formax
#import scipy.optimize as opt
from distribution_centerdensity import centerdensity
import warnings

Preset_rtol = 1e-4
number_per_parameter=10

def Calculation(x):
    eos=config.eos_config(parameter[x].args)
    warnings.filterwarnings('error')

# =============================================================================
#     if(parameter[x].properity[2]>2):
#         result1=opt.minimize(Mass_transition_formax,800.0,tol=0.001,args=(config.Preset_Pressure_final,Preset_rtol,eos),method='Nelder-Mead')
#         properity_new=parameter[x].properity[0:3]+[result1.x[0]]+parameter[x].properity[3:]
#         parameter[x].set_properity(properity_new)
#         Maximum_pressure_center=result1.x[0]
#     else:
#         Maximum_pressure_center=parameter[x].properity[0]
#         properity_new=parameter[x].properity[0:3]+[Maximum_pressure_center]+parameter[x].properity[3:]
#         parameter[x].set_properity(properity_new)
#     ofpc_array=centerdensity(10,parameter[x].properity[3],Maximum_pressure_center,config.concentration,number_per_parameter)
# =============================================================================

    ofpc_array=centerdensity(10,parameter[x].properity[4],parameter[x].properity[3],config.concentration,number_per_parameter)
 
    for i in range(np.size(ofpc_array)):
        try:
            parameter[x].add_star([ofpc_array[i]]+config.eos_MassRadius(ofpc_array[0],config.Preset_Pressure_final,Preset_rtol,'MRBIT',eos))
        except RuntimeWarning:
            print('Runtimewarning happens at addstars: '+str(ofpc_array[i]))
            print(parameter[x].args)
    return parameter[x]

def processInput(i,num_cores,complete_set):
    timenow=time()
    timebegin=timenow
    timeprev=timenow
    result=list()
    for ii in range(complete_set):
        result.append(Calculation(i+num_cores*ii))
        timeprev=remainingTime(timebegin,timeprev,ii,config.start_from,complete_set)
    return result


def remainingTime(timebegin,timeprev,ii,start_from,complete_set):
    timenow=time()
    print('Completeness: %.2f%%#################################'%(100.0*(1.0*ii/complete_set)))
    print('Estimate time remaining: %.2f hours (overall)'%((timenow-timebegin)*(complete_set-ii)/(ii+1-start_from*complete_set)/3600))
    print('                         %.2f hours (instant)'%((timenow-timeprev)*(complete_set-ii)/3600))
    return timenow

def main(processInput):
    num_cores = cpu_count()-1
    total_num = np.size(parameter)
    complete_set=np.int(total_num/num_cores)
    leftover_num=total_num-complete_set*num_cores
    timebegin=time()

    f_log=open('./'+dir_name+'/'+name_log,'wb')
    f_log.write("Calculation_mode: " + Calculation_mode)
    f_log.write('%d cores are being used.\n'% num_cores)
    f_log.close()

    Output=Parallel(n_jobs=num_cores)(delayed(processInput)(i,num_cores,complete_set) for i in range(num_cores))
    Output_leftover=Parallel(n_jobs=num_cores)(delayed(Calculation)(i+complete_set*num_cores) for i in range(leftover_num))
    result=list()
    for i in range(complete_set):
        for ii in range(num_cores):
            result.append(Output[ii][i])
    for i in range(leftover_num):
            result.append(Output_leftover[i])

    print('Completeness: 100%%#################################')
    print('Total time cost: %.2f hours'%((time()-timebegin)/3600))

    f1=open('./'+dir_name+'/'+name_dat_main+'_addstars','wb')
    pickle.dump(result,f1)
    f1.close()
    print('Congratulation! %s successfully saved!!!!!!!!!!!!!'%name_dat_main)

if __name__ == '__main__':
    import sys
    print("Running Program: " + sys.argv[0])
    print("Configuration file: " + sys.argv[1])
    config=__import__(sys.argv[1])
    Calculation_mode=config.Calculation_mode
    Hybrid_sampling=sys.argv[2]
    print("Calculation_mode: " + Calculation_mode)
    print("Hybrid_sampling: " + Hybrid_sampling)
    sampling = Calculation_mode+'_'+Hybrid_sampling
    name_dat_main='sorted_'+sampling+'_2.0-2.4.dat_addofmass'
    name_log='main_'+sampling+'_addstars.log'
    if(Calculation_mode=='hybrid'):
        if(Hybrid_sampling=='low_trans_complete'):
            dir_name='data_low_trans_complete'
        else:
            dir_name='data_p1='+str(config.pressure1)+'_cs2='+str(1.0*config.cs2)
    elif(Calculation_mode=='hadronic'):
        dir_name='data_hadronic'
    else:
        print 'Calculation_mode not found!'
    f1=open('./'+dir_name+'/'+name_dat_main,'rb')
    parameter=pickle.load(f1)
    f1.close()
    main(processInput)