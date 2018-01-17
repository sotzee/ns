#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:27:48 2018

@author: sotzee
"""
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import pickle
from time import time

def processInput(i0,i1):
    parameter_satisfied_core=list()
    parameter_unsatisfied_core=list()
    for i in range(i0,i1):
        print i
        if(config.constrain(parameter[i])):
            parameter_satisfied_core.append(parameter[i])
        else:
            parameter_unsatisfied_core.append(parameter[i])
    return [parameter_satisfied_core,parameter_unsatisfied_core]

def insert_words_dat(dat_filename,words_to_insert):
    return dat_filename[:-4]+words_to_insert+dat_filename[-4:]

def main(processInput):
    num_cores = cpu_count()-1
    total_num = np.size(parameter)
    complete_set=np.int(total_num/num_cores)
    leftover_num=total_num-complete_set*num_cores
    
    Output=Parallel(n_jobs=num_cores)(delayed(processInput)(i*complete_set,(i+1)*complete_set) for i in range(num_cores))
    if(leftover_num>0):
        Output_leftover=processInput(complete_set*num_cores,total_num)
    else:
        Output_leftover=[[],[]]
    parameter_satisfied=list()
    parameter_unsatisfied=list()
    for i in range(num_cores):
        parameter_satisfied+=Output[i][0]
        parameter_unsatisfied+=Output[i][1]
    parameter_satisfied+=Output_leftover[0]
    parameter_unsatisfied+=Output_leftover[1]
    
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
    
    print 'start'
    t0=time()
    main(processInput)
    t1=time()
    print t1-t0