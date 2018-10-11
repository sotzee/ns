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
    return parameter_list[x]

def processInput(Calculation,parameter_list,i,num_cores,complete_set,preset_start_from):
    timenow=time()
    timebegin=timenow
    timeprev=timenow
    result=list()
    for ii in range(complete_set):
        result.append(Calculation(parameter_list,i+num_cores*ii))
        timeprev=remainingTime(timebegin,timeprev,ii,preset_start_from,complete_set)
    return result

def remainingTime(timebegin,timeprev,ii,start_from,complete_set):
    timenow=time()
    print('Completeness: %.2f%%#################################'%(100.0*(1.0*ii/complete_set)))
    print('Estimate time remaining: %.2f hours (overall)'%((timenow-timebegin)*(complete_set-ii)/(ii+1-start_from*complete_set)/3600))
    print('                         %.2f hours (instant)'%((timenow-timeprev)*(complete_set-ii)/3600))
    return timenow

def main_parallel(Calculation,parameter_list,result_file_name,preset_start_from):
    num_cores = cpu_count()-1
    total_num = len(parameter_list)
    complete_set=np.int(total_num/num_cores)
    leftover_num=total_num-complete_set*num_cores
    timebegin=time()

    Output=Parallel(n_jobs=num_cores)(delayed(processInput)(Calculation,parameter_list,i,num_cores,complete_set,preset_start_from) for i in range(num_cores))
    Output_leftover=Parallel(n_jobs=num_cores)(delayed(Calculation)(parameter_list,i+complete_set*num_cores) for i in range(leftover_num))
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
# =============================================================================
#     print Output
#     print Output_leftover
#     print result
# =============================================================================

# =============================================================================
#     f1=open('./'+dir_name+'/'+name_dat_main+'_try','wb')
#     pickle.dump(parameter,f1)
#     f1.close()
#     print('Congratulation! %s successfully saved!!!!!!!!!!!!!'%name_dat_main)
# =============================================================================

if __name__ == '__main__':
    import sys
    #print("Running Program: " + sys.argv[0])
    #print("Configuration file: " + sys.argv[1])
    preset_start_from=0
    parameter=np.linspace(0,100,101)
    result_file_name='parallel_result.dat'
# =============================================================================
#     dir_name=
#     name_dat_main=
#     f1=open('./'+dir_name+'/'+name_dat_main,'rb')
#     parameter=pickle.load(f1)
#     f1.close()
# ============================================================================
    main_parallel(Calculation,parameter,result_file_name,preset_start_from)