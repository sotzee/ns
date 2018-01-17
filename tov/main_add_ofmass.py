#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:57:06 2018

@author: sotzee
"""

import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from time import time
import pickle
from Find_OfMass import Properity_ofmass
import warnings

Preset_rtol = 1e-4

#################################################
#setParameter(xxx) returns parameter[x][x]
#parameter[x][0] = pressure1
#parameter[x][1] = pressure2
#parameter[x][2] = gamma2
#parameter[x][3] = pressure_trans
#parameter[x][4] = density_trans
#parameter[x][5] = pressure_trans/density_trans
#parameter[x][6] = det_density_reduced*density_trans
#parameter[x][7] = det_density_reduced
#parameter[x][8] = cs2
#################################################
#hardronic parameter
#parameter.append([pressure1,pressure2,gamma2,pressure3,gamma3,0.0,0.0,0.0,0.0])

def processInput(x):
    eos=config.eos_config(parameter[x].args)
    warnings.filterwarnings('error')
    MaximumMass_pressure_center=parameter[x].properity[0]
    for i in range(np.size(ofmass_array)):
        try:
            parameter[x].add_star(Properity_ofmass(ofmass_array[i],config.Preset_pressure_center_low,MaximumMass_pressure_center,config.eos_MassRadius,config.Preset_Pressure_final,Preset_rtol,config.Preset_Pressure_final_index,eos))
        except RuntimeWarning:
            print('Runtimewarning happens at OfMass: '+str(ofmass_array[i]))
            print(parameter[x].args)

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
    timenow=time()
    timeprev=timenow
    timebegin=timenow
    f_log=open('./'+dir_name+'/'+name_log,'wb')
    f_log.write("Calculation_mode: " + Calculation_mode)
    f_log.write('%d cores are being used.\n'% num_cores)
    f_log.write("OfMass array: " + sys.argv[3])
    f_log.write(ofmass_array)
    f_log.close()
    for ii in range(int(config.start_from*complete_set),complete_set):
        Parallel(n_jobs=num_cores)(delayed(processInput)(i+num_cores*ii) for i in range(num_cores))
        timeprev=remainingTime(timebegin,timeprev,ii,config.start_from,complete_set)
    if(leftover_num>0):
        Parallel(n_jobs=leftover_num)(delayed(processInput)(i+num_cores*complete_set) for i in range(leftover_num))
    timenow=time()
    print('Completeness: 100%%#################################')
    print('Total time cost: %.2f hours'%((timenow-timebegin)/3600))

    f1=open('./'+dir_name+'/'+name_dat_main+'_try','wb')
    pickle.dump(parameter,f1)
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
    print("OfMass array: " + sys.argv[3])
    sampling = Calculation_mode+'_'+Hybrid_sampling
    name_dat_para='parameter_'+sampling+'.dat'
    name_dat_main='parameter_'+sampling+'_2.0-2.4.dat'
    name_log='main_'+sampling+'_add_ofmass.log'
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
    f2=open('./'+sys.argv[3],'rb')
    ofmass_array=pickle.load(f2)
    f2.close()
    main(processInput)