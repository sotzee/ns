# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:09:42 2016

@author: Sotzee
"""

import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from time import time
import pickle
from Find_OfMass import Properity_ofmass
import warnings

Preset_rtol = 1e-4

from eos_class import EOS_item
def Calculation(x):
    eos=config.eos_config(parameter[x].args)
    warnings.filterwarnings('error')
    try:    
        processOutput_maxmass = config.eos_Maxmass(config.Preset_Pressure_final,Preset_rtol,eos)
    except RuntimeWarning:
        print 'Runtimewarning happens at:'
        print parameter[x].args
        processOutput_maxmass=[0,0,0,0]
    [MaximumMass_pressure_center,MaximumMass,transition_type,Maximum_pressure_center,Mass_transition]=processOutput_maxmass
    
    if(config.TurnOn_radius_onepointfour):
        if(Mass_transition==0): #hadronic eos with no transition
            processOutput_onepointfour = Properity_ofmass(1.4,config.Preset_pressure_center_low,MaximumMass_pressure_center,config.eos_MassRadius,config.Preset_Pressure_final,Preset_rtol,config.Preset_Pressure_final_index,eos)
        if(Mass_transition>1.4):
            processOutput_onepointfour = Properity_ofmass(1.4,config.Preset_pressure_center_low,eos.pressure_trans,config.eos_MassRadius,config.Preset_Pressure_final,Preset_rtol,config.Preset_Pressure_final_index,eos)
        elif(MaximumMass>1.4):
            processOutput_onepointfour = Properity_ofmass(1.4,eos.pressure_trans,MaximumMass_pressure_center,config.eos_MassRadius,config.Preset_Pressure_final,Preset_rtol,config.Preset_Pressure_final_index,eos)
        else:
            [0,0,0,0,0,0,0]
    else:
        processOutput_onepointfour=[0,0,0,0,0,0,0]
    processOutput=processOutput_maxmass+processOutput_onepointfour
    return processOutput

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
    if(Calculation_mode=='hybrid'):
        if(Hybrid_sampling=='low_trans_complete'):
            f_log.write('Preset_rtol=%s\n Preset_pressure1=%s    Preset_cs2=%s\n'% (Preset_rtol,config.Preset_pressure1,config.Preset_cs2))
        else:
            f_log.write('Preset_rtol=%s\n pressure1=%f    cs2=%f\n'% (Preset_rtol,config.pressure1,config.cs2))
    elif(Calculation_mode=='hadronic'):
        f_log.write('Preset_rtol=%s\n Preset_pressure1=%s\n Preset_pressure2=%s \n Preset_pressure3=%s \n'% (Preset_rtol,config.Preset_pressure1,config.Preset_pressure2,config.Preset_pressure3))
    f_log.write('%d cores are being used.\n'% num_cores)
    f_log.close()

    Output=Parallel(n_jobs=num_cores)(delayed(processInput)(i,num_cores,complete_set) for i in range(num_cores))
    Output_leftover=Parallel(n_jobs=num_cores)(delayed(Calculation)(i+complete_set*num_cores) for i in range(leftover_num))
    parameter_main=list()
    for i in range(complete_set):
        for ii in range(num_cores):
            if(Output[ii][i][1]>1.4):
                n=ii+i*num_cores
                parameter[n].set_properity(Output[ii][i])
                parameter_main.append(parameter[n])
    for i in range(leftover_num):
        if(Output_leftover[i][1]>1.4):
            n=i+complete_set*num_cores
            parameter[n].set_properity(Output_leftover[i])
            parameter_main.append(parameter[n])

    print('Completeness: 100%%#################################')
    print('Total time cost: %.2f hours'%((time()-timebegin)/3600))

    f1=open('./'+dir_name+'/'+name_dat_main,'wb')
    pickle.dump(parameter_main,f1)
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
    name_dat_para='parameter_'+sampling+'.dat'
    name_dat_main='parameter_'+sampling+'_main.dat'
    name_log='main_'+sampling+'.log'
    from OStools import ensure_dir
    if(Calculation_mode=='hybrid'):
        from setParameter import setParameter,setParameter_low_trans_complete
        if(Hybrid_sampling=='normal_trans'):
            Preset_gamma2 = config.Preset_gamma2 #[num,low,high]polytropic index between pressure1 and pressure_trans.
        elif(Hybrid_sampling=='hardest_before_trans'):
            Preset_gamma2 = [0,2.0,10.0]
        elif(Hybrid_sampling=='softest_before_trans'):
            Preset_gamma2 = [0,2.0,10.0]
        elif(Hybrid_sampling=='low_trans'):
            Preset_gamma2 = [1,2.0,2.0]
        elif(Hybrid_sampling=='low_trans_complete'):
            Preset_gamma2 = [1,2.0,2.0]
        else:
            Preset_gamma2 = [1,2.0,2.0]
            print('Hybrid_sampling==%s, which is not valid.'%Hybrid_sampling)

        if(Hybrid_sampling=='low_trans_complete'):
            dir_name='data_low_trans_complete'
            ensure_dir('./'+dir_name+'/')
            parameter=setParameter_low_trans_complete(config.baryon_density0,config.Preset_pressure1,config.baryon_density1,config.baryon_density2,Preset_gamma2,config.Preset_num_pressure_trans,config.Preset_det_density,config.Preset_cs2,Hybrid_sampling)
        else:
            dir_name='data_p1='+str(config.pressure1)+'_cs2='+str(1.0*config.cs2)
            ensure_dir('./'+dir_name+'/')
            parameter=setParameter(config.baryon_density0,config.pressure1,config.baryon_density1,config.baryon_density2,Preset_gamma2,config.Preset_num_pressure_trans,config.Preset_det_density,config.cs2,Hybrid_sampling)
        for i in range(np.size(parameter)/np.size(parameter[0])):
            parameter[i]=EOS_item([config.baryon_density0,parameter[i][0],config.baryon_density1,parameter[i][1],config.baryon_density2,config.pressure3,config.baryon_density3,parameter[i][3],parameter[i][6],parameter[i][8]])
    elif(Calculation_mode=='hadronic'):
        from setParameter import setParameter_hadronic
        dir_name='data_hadronic'
        ensure_dir('./'+dir_name+'/')
        parameter=setParameter_hadronic(config.baryon_density0,config.Preset_pressure1,config.baryon_density1,config.Preset_pressure2,config.baryon_density2,config.Preset_pressure3,config.baryon_density3)
        for i in range(np.size(parameter)/np.size(parameter[0])):
            parameter[i]=EOS_item([config.baryon_density0,parameter[i][0],config.baryon_density1,parameter[i][1],config.baryon_density2,parameter[i][3],config.baryon_density3])
    else:
        print('Calculation_mode not found!')
    main(processInput)
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
#parameter[x][9] = MaximumMass_pressure_center
#parameter[x][10]= MaximumMass
#parameter[x][11]= twopeak_forsure
#parameter[x][12]= radius_onepointfour_pressure_center
#parameter[x][13]= radius_onepointfour
#parameter[x][14]= mass_onepointfour
#################################################

