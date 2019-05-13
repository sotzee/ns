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
import cPickle

def Calculation(parameter_list,x):
    return np.array(parameter_list[x])

def Calculation_i(parameter_i):
    return np.array(parameter_i)

def processInput(Calculation,parameter_list,i,num_cores,complete_set,preset_start_from):
    timenow=time()
    timebegin=timenow
    timeprev=timenow
    result=list()
    for ii in range(complete_set):
        result.append(Calculation(parameter_list,i+num_cores*ii))
        timeprev=remainingTime(timebegin,timeprev,ii,preset_start_from,complete_set)
    return np.array(result)

def remainingTime(timebegin,timeprev,ii,start_from,complete_set):
    timenow=time()
    print('Completeness: %.2f%%#################################'%(100.0*(1.0*ii/complete_set)))
    print('Estimate time remaining: %.2f hours (overall)'%((timenow-timebegin)*(complete_set-ii)/(ii+1-start_from*complete_set)/3600))
    print('                         %.2f hours (instant)'%((timenow-timeprev)*(complete_set-ii)/3600))
    return timenow

def main_parallel_largest_batch(Calculation,parameter_list,result_file_name,preset_start_from):
    num_cores = cpu_count()-1
    total_num = len(parameter_list)
    complete_set=np.int(total_num/num_cores)
    leftover_num=total_num-complete_set*num_cores
    timebegin=time()

    Output=Parallel(n_jobs=num_cores,verbose=2)(delayed(processInput)(Calculation,parameter_list,i,num_cores,complete_set,preset_start_from) for i in range(num_cores))
    Output_leftover=Parallel(n_jobs=num_cores,verbose=2,batch_size=complete_set)(delayed(Calculation)(parameter_list,i) for i in range(total_num))
    result=list()
    for i in range(complete_set):
        for ii in range(num_cores):
            result.append(Output[ii][i])
    for i in range(leftover_num):
            result.append(Output_leftover[i])
    
    print('Completeness: 100%%#################################')
    print('Total time cost: %.2f hours'%((time()-timebegin)/3600))
    
    f=open(result_file_name,'wb')
    cPickle.dump(result,f)
    f.close()
    
import warnings
def main_parallel(Calculation_i,parameter_list,result_file_name,error_log_file_name):
    num_cores = cpu_count()-1
    try:
        Output=Parallel(n_jobs=num_cores,verbose=10)(delayed(Calculation_i)(parameter_i) for parameter_i in parameter_list)
# =============================================================================
#         try:
#             Output=Parallel(n_jobs=num_cores,verbose=10)(delayed(Calculation_i)(parameter_i) for parameter_i in parameter_list)
#         except:
#             Output=Parallel(n_jobs=num_cores,verbose=10)(delayed(Calculation_i)(parameter_list,i) for i in range(len(parameter_list)))
# =============================================================================
    except RuntimeWarning:
        print('Runtimewarning happens at calculating max mass:')
        f = open(error_log_file_name, "a")
        f.write('Runtimewarning happens at calculating max mass:')
        f.close()
    f=open(result_file_name,'wb')
    cPickle.dump(np.array(Output),f)
    f.close()
    return np.array(Output)

def main_parallel_unsave(Calculation_i,parameter_list,other_args=[]):
    num_cores = cpu_count()-1
    Output=Parallel(n_jobs=num_cores,verbose=1)(delayed(Calculation_i)(parameter_i,other_args) for parameter_i in parameter_list)
    return np.array(Output)

if __name__ == '__main__':
    #import sys
    #print("Running Program: " + sys.argv[0])
    #print("Configuration file: " + sys.argv[1])
    parameter=np.reshape(np.linspace(0,100,50001*6),(50001,6))
    result_file_name='parallel_result.dat'
# =============================================================================
#     dir_name=
#     name_dat_main=
#     f1=open('./'+dir_name+'/'+name_dat_main,'rb')
#     parameter=pickle.load(f1)
#     f1.close()
# ============================================================================
    main_parallel(Calculation_i,parameter,result_file_name)