# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:53:22 2016

@author: sotzee
"""

import numpy as np
import pickle

def main(parameter,sortmass):
    num_sort = np.size(sortmass)
    file_sort = range(num_sort-1)
    for i in range(num_sort-1):
        file_sort[i]=name_dat_sort+'_'+str(sortmass[i])+'-'+str(sortmass[i+1])+'.dat'
    num=np.size(parameter)/np.size(parameter[0])
    parameter_sorted=list()
    for ii in range(num_sort-1):
        parameter_sorted.append(list())
    for i in range(num):
        for ii in range(num_sort-1):        
            if(sortmass[ii]<parameter[i].properity[2]<sortmass[ii+1]):
                parameter_sorted[ii].append(parameter[i])
                break
    check_num=0
    for ii in range(num_sort-1):
        f1=open('./'+dir_name+'/'+file_sort[ii],'wb')
        pickle.dump(parameter_sorted[ii],f1)
        f1.close()
        if(np.size(parameter_sorted[ii])==0):
            num_in_sort=0
        else:
            num_in_sort=np.size(parameter_sorted[ii])
            print('%d sets of parameter is in range (%f,%f), being stored in %s\n'%(num_in_sort,sortmass[ii],sortmass[ii+1],file_sort[ii]))
            check_num+=num_in_sort
    if(num==check_num):
        print('number of parameter sets in '+name_dat_main+' is equal to total number of parameter sets in '+name_dat_sort+' *.dat\n')
    else:
        print('ERROR!!!!\n')
        print('number of parameter sets in '+name_dat_main+' is NOT equal to total number of parameter sets in '+name_dat_sort+' *.dat\n')
        print('%d is within range (%f,%f), while %d is at outside\n'%(check_num,sortmass[0],sortmass[num_sort-1],num-check_num))

def split_parameter(parameter):
    a=[]
    b=[]
    for i in range(np.size(parameter)/np.size(parameter[0])):
        a.append(parameter[i][0])
        b.append(parameter[i][1])
    return a,b

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
    name_dat_main='parameter_'+sampling+'_main.dat'
    name_dat_sort='sorted_'+sampling
    file_sort_namelist='file_sort_namelist_'+sampling
    Maxmass_step=0.1
    if(Calculation_mode=='hybrid'):
        if(Hybrid_sampling=='low_trans_complete'):
            dir_name='data_low_trans_complete'
        else:
            dir_name='data_p1='+str(config.pressure1)+'_cs2='+str(1.0*config.cs2)
    elif(Calculation_mode=='hadronic'):
        dir_name='data_hadronic'
    f=open('./'+dir_name+'/'+name_dat_main,'rb')
    parameter=pickle.load(f)
    f.close()
# =============================================================================
#     parameter_eos,parameter_properity = split_parameter(parameter)
#     parameter_properity=np.array(parameter_properity)
#     parameter_properity=parameter_properity.transpose()
#     sortmass=np.arange(0.1*int(10*parameter_properity[1].min()),parameter_properity[1].max()+Maxmass_step,Maxmass_step)
#     main(parameter,sortmass)
#     num_sort = np.size(sortmass)
#     file_sort = range(num_sort-1)
#     for i in range(num_sort-1):
#         file_sort[i]=name_dat_sort+'_'+str(sortmass[i])+'-'+str(sortmass[i+1])+'.dat'
#     f2=open('./'+dir_name+'/'+file_sort_namelist,'wb')
#     pickle.dump(file_sort,f2)
#     f2.close()
# =============================================================================
    main(parameter,[1.4,2.0,2.4,5.0])
