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
from eos_class import EOS_item
def processInput(x):
    eos=config.eos_config(parameter[x].args)
    warnings.filterwarnings('error')
    try:    
        processOutput_maxmass = config.eos_Maxmass(config.Preset_Pressure_final,Preset_rtol,eos)
    except RuntimeWarning:
        print 'Runtimewarning happens at:'
        print parameter[x].args
        processOutput_maxmass=[0,0,0]
    [MaximumMass_pressure_center,MaximumMass,transition_type]=processOutput_maxmass
    if(config.TurnOn_radius_onepointfour & (MaximumMass>1.4)):
        processOutput_onepointfour = Properity_ofmass(1.4,config.Preset_pressure_center_low,MaximumMass_pressure_center,config.eos_MassRadius,config.Preset_Pressure_final,Preset_rtol,config.Preset_Pressure_final_index,eos)
    else:
        processOutput_onepointfour=[0,0,0,0,0,0,0]
    processOutput=processOutput_maxmass+processOutput_onepointfour
    return processOutput

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
    parameter_main=list()
    timenow=time()
    timeprev=timenow
    timebegin=timenow
    f_log=open('./'+dir_name+'/'+name_log,'wb')
    if(Calculation_mode=='hybrid'):
        if(Hybrid_sampling=='low_trans_complete'):
            f_log.write('Preset_rtol=%s\n Preset_pressure1=%s    Preset_cs2=%s\n'% (Preset_rtol,config.Preset_pressure1,config.Preset_cs2))
        else:
            f_log.write('Preset_rtol=%s\n pressure1=%f    cs2=%f\n'% (Preset_rtol,config.pressure1,config.cs2))
    elif(Calculation_mode=='hadronic'):
        f_log.write('Preset_rtol=%s\n Preset_pressure1=%s\n Preset_pressure2=%s \n Preset_pressure3=%s \n'% (Preset_rtol,config.Preset_pressure1,config.Preset_pressure2,config.Preset_pressure3))
    f_log.write('%d cores are being used.\n'% num_cores)
    for ii in range(int(config.start_from*complete_set),complete_set):
        Output=Parallel(n_jobs=num_cores)(delayed(processInput)(i+num_cores*ii) for i in range(num_cores))
        timeprev=remainingTime(timebegin,timeprev,ii,config.start_from,complete_set)
        for i in range(num_cores):
            if(Output[i][1]>1.4):
                n=i+num_cores*ii
                parameter[n].set_properity(Output[i])
                parameter_main.append(parameter[n])
                f_log.write(str(parameter[n].args)+str(parameter[n].properity)+'\n')
    if(leftover_num>0):
        Output=Parallel(n_jobs=leftover_num)(delayed(processInput)(i+num_cores*complete_set) for i in range(leftover_num))
        for i in range(leftover_num):
            if(Output[i][1]>1.4):
                n=i+num_cores*ii
                parameter[n].set_properity(Output[i])
                parameter_main.append(parameter[n])
                f_log.write(str(parameter[n].args)+str(parameter[n].properity)+'\n')
    f_log.close()
    timenow=time()
    print('Completeness: 100%%#################################')
    print('Total time cost: %.2f hours'%((timenow-timebegin)/3600))

    f1=open('./'+dir_name+'/'+name_dat_main,'wb')
    pickle.dump(parameter_main,f1)
    f1.close()
    print('Congratulation! %s successfully saved!!!!!!!!!!!!!'%name_dat_main)

if __name__ == '__main__':
    import sys
    print("Running Program: " + str(sys.argv[0]))
    print("Configuration file: " + str(sys.argv[1]))
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


