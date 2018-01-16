#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:27:48 2018

@author: sotzee
"""
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from time import time
import pickle

def processInput(x):
    t0=time()
    flag=config.constrain(parameter[x])
    t1=time()
    print t1-t0
    return flag

def remainingTime(timebegin,timeprev,ii,start_from,complete_set):
    timenow=time()
    print('Completeness: %.2f%%#################################'%(100.0*(1.0*ii/complete_set)))
    print('Estimate time remaining: %.2f hours (overall)'%((timenow-timebegin)*(complete_set-ii)/(ii+1-start_from*complete_set)/3600))
    print('                         %.2f hours (instant)'%((timenow-timeprev)*(complete_set-ii)/3600))
    return timenow

def insert_words_dat(dat_filename,words_to_insert):
    return dat_filename[:-4]+words_to_insert+dat_filename[-4:]

def main(processInput):
    num_cores = cpu_count()-1
    total_num = np.size(parameter)
    complete_set=np.int(total_num/num_cores)
    leftover_num=total_num-complete_set*num_cores
    timenow=time()
    timeprev=timenow
    timebegin=timenow
    for ii in range(int(config.start_from*complete_set),complete_set):
        t0=time()
        Output=Parallel(n_jobs=num_cores)(delayed(processInput)(i+num_cores*ii) for i in range(num_cores))
        t1=time()
        timeprev=remainingTime(timebegin,timeprev,ii,config.start_from,complete_set)
        for i in range(num_cores):
            n=i+num_cores*ii
            print n
            if(Output[i]):
                parameter_satisfied.append(parameter[n])
            else:
                parameter_unsatisfied.append(parameter[n])
        t2=time()
        print t1-t0,t2-t1
    if(leftover_num>0):
        Parallel(n_jobs=leftover_num)(delayed(processInput)(i+num_cores*complete_set) for i in range(leftover_num))
        for i in range(leftover_num):
            n=i+num_cores*ii
            print n
            if(Output[i]):
                parameter_satisfied.append(parameter[n])
            else:
                parameter_unsatisfied.append(parameter[n])
    timenow=time()
    print('Completeness: 100%%#################################')
    print('Total time cost: %.2f hours'%((timenow-timebegin)/3600))

if __name__ == '__main__':
    import sys
    print("Running Program: " + sys.argv[0])
    print("Configuration file for sorting: " + sys.argv[1])
    config=__import__(sys.argv[1])
    fire_to_sort=sys.argv[2]
    print("Parameter for sorting: " + fire_to_sort)
    
    f1=open(fire_to_sort,'rb')
    parameter=pickle.load(f1)
    f1.close()
    parameter_satisfied=list()
    parameter_unsatisfied=list()
    main(processInput)

    name_satisfied=insert_words_dat(fire_to_sort,config.name_satisfied)
    f2=open(name_satisfied,'wb')
    pickle.dump(parameter_satisfied,f2)
    f2.close()
    print('Congratulation! %s successfully saved!!!!!!!!!!!!!'%name_satisfied)
    print('Number of eos included: ' + str(np.size(parameter_satisfied)))

    name_unsatisfied=insert_words_dat(fire_to_sort,config.name_unsatisfied)
    f2=open(name_unsatisfied,'wb')
    pickle.dump(parameter_unsatisfied,f2)
    f2.close()
    print('Congratulation! %s successfully saved!!!!!!!!!!!!!'%name_unsatisfied)
    print('Number of eos included: ' + str(np.size(parameter_unsatisfied)))