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
from eos_class import EOS_item_with_binary
import warnings

Preset_rtol = 1e-4
number_per_parameter=5

def mass_chirp(m1,m2):
    return (m1*m2)**0.6/(m1+m2)**0.2

def tidal_binary(m1,m2,tidal1,tidal2):
    return 16*((m1+12*m2)*m1**4*tidal1+(m2+12*m1)*m2**4*tidal2)/(13*(m1+m2)**5)

def Calculation(x,random_chisquare):
    eos=config.eos_config(parameter[x].args)
    warnings.filterwarnings('error')
    binaries=[]
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
    try:
        if(parameter[x].properity[35]==0):
            ofpc_array=centerdensity(10,parameter[x].properity[3],parameter[x].properity[1],config.concentration,random_chisquare[x])
            for i in range(np.size(ofpc_array)):
                processOutput_ofpc = config.eos_MassRadius(ofpc_array[i],config.Preset_Pressure_final,Preset_rtol,'MRBIT',eos)
                if(ofpc_array[i]<parameter[x].args[7]):
                    parameter[x].add_star([0,ofpc_array[i]]+processOutput_ofpc)
                else:
                    parameter[x].add_star([1,ofpc_array[i]]+processOutput_ofpc)
        else:
            if(parameter[x].properity[11]==0):
                ofpc_array=centerdensity(10,parameter[x].properity[3],parameter[x].properity[19],config.concentration,random_chisquare[x])
            else:
                ofpc_array=centerdensity(10,parameter[x].properity[11],parameter[x].properity[19],config.concentration,random_chisquare[x])
            for i in range(np.size(ofpc_array)):
                processOutput_ofpc = config.eos_MassRadius(ofpc_array[i],config.Preset_Pressure_final,Preset_rtol,'MRBIT',eos)
                if(ofpc_array[i]<parameter[x].args[7]):#hadronic
                    parameter[x].add_star([0,ofpc_array[i]]+processOutput_ofpc)
                elif(ofpc_array[i]<parameter[x].properity[27]):#continuous hybrid
                    parameter[x].add_star([1,ofpc_array[i]]+processOutput_ofpc)
                elif(ofpc_array[i]<parameter[x].properity[35]):#unstable hybrid   [35]used to be [19] as a bug 04/10/2018
                    parameter[x].add_star([2,ofpc_array[i]]+processOutput_ofpc)
                else:#discontinuous hybrid
                    parameter[x].add_star([3,ofpc_array[i]]+processOutput_ofpc)
                for j in range(np.size(ofpc_array)):
                    m1=parameter[x].stars[i+5][2]
                    tidal1=parameter[x].stars[i+5][8]
                    m2=parameter[x].stars[j+5][2]
                    tidal2=parameter[x].stars[j+5][8]
                    binaries.append([i,j,mass_chirp(m1,m2),tidal_binary(m1,m2,tidal1,tidal2)])
    except RuntimeWarning:
        print('Runtimewarning happens at addstars: '+str(ofpc_array[i]))
        print(parameter[x].args)
    return EOS_item_with_binary(parameter[x].args,parameter[x].properity,parameter[x].stars,binaries)

def processInput(i,num_cores,complete_set,random_chisquare):
    timenow=time()
    timebegin=timenow
    timeprev=timenow
    result=list()
    for ii in range(complete_set):
        result.append(Calculation(i+num_cores*ii,random_chisquare))
        timeprev=remainingTime(timebegin,timeprev,ii,config.start_from,complete_set)
    return result

def centerdensity(begin,mid,end,concentration,random_array):
    ratio=(mid-begin)/concentration/(1-2.0/9/concentration)**3
    random_array=random_array*ratio
    #array=array[(array >= 0) & (array <= end)]
    #array=(array*array+begin*end)/(begin+end)
    random_array=random_array[(random_array >= begin) & (random_array <= end)]
    return random_array[0:number_per_parameter]

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
    
    random_chisquare=np.random.chisquare(config.concentration,[total_num,5*number_per_parameter])
    Output=Parallel(n_jobs=num_cores)(delayed(processInput)(i,num_cores,complete_set,random_chisquare) for i in range(num_cores))
    Output_leftover=Parallel(n_jobs=num_cores)(delayed(Calculation)(i+complete_set*num_cores,random_chisquare) for i in range(leftover_num))
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