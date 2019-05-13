#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:28:08 2019

@author: sotzee
"""

from PNM_expansion import EOS_EXPANSION_PNM,EOS_CSS,EOS_PnmCSS,EOS_Sly4_match_PnmCSS,EOS_SLY4POLYwithPNM
import matplotlib.pyplot as plt
import numpy as np
import cPickle

path = "./"
dir_name='Lambda_PNM_margueron_calculation_parallel'

def read_file(file_name):
    f_file=open(file_name,'rb')
    content=np.array(cPickle.load(f_file))
    f_file.close()
    return content

args=read_file(path+dir_name+'/Lambda_PNM_calculation_args.dat')
eos=read_file(path+dir_name+'/Lambda_PNM_calculation_eos.dat')
maxmass_result=read_file(path+dir_name+'/Lambda_PNM_calculation_maxmass.dat')
Properity_onepointfour=read_file(path+dir_name+'/Lambda_PNM_calculation_onepointfour.dat')
mass_beta_Lambda_result=read_file(path+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat')
chirp_q_Lambdabeta6_Lambda1Lambda2=read_file(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat')
logic_maxmass=maxmass_result[:,1]>=2
logic_causality=maxmass_result[:,2]<1
logic=np.logical_and(logic_maxmass,logic_causality)

path = "./"
dir_name='Lambda_PNM_margueron_calculation_parallel_lowL'
args_lowL=read_file(path+dir_name+'/Lambda_PNM_calculation_args.dat')
eos_lowL=read_file(path+dir_name+'/Lambda_PNM_calculation_eos.dat')
maxmass_result_lowL=read_file(path+dir_name+'/Lambda_PNM_calculation_maxmass.dat')
Properity_onepointfour_lowL=read_file(path+dir_name+'/Lambda_PNM_calculation_onepointfour.dat')
mass_beta_Lambda_result_lowL=read_file(path+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat')
chirp_q_Lambdabeta6_Lambda1Lambda2_lowL=read_file(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat')
logic_maxmass_lowL=maxmass_result_lowL[:,1]>=2
logic_causality_lowL=maxmass_result_lowL[:,2]<1
logic_lowL=np.logical_and(logic_maxmass_lowL,logic_causality_lowL)

args_new=np.concatenate((args_lowL,args),axis=1)
eos_new=np.concatenate((eos_lowL,eos),axis=1)
maxmass_result_new=np.concatenate((np.reshape(maxmass_result_lowL,np.shape(eos_lowL)+(-1,)),np.reshape(maxmass_result,np.shape(eos)+(-1,))),axis=1)
logic_extend=np.concatenate((np.reshape(np.full(len(logic_lowL),False,dtype=bool),np.shape(eos_lowL)+(-1,)),np.reshape(logic,np.shape(eos)+(-1,))),axis=1).flatten()
logic_lowL_extend=np.concatenate((np.reshape(logic_lowL,np.shape(eos_lowL)+(-1,)),np.reshape(np.full(len(logic),False,dtype=bool),np.shape(eos)+(-1,))),axis=1).flatten()
logic_new=np.logical_or(logic_lowL_extend,logic_extend)
Properity_onepointfour_new=np.full((len(logic_new),6),1,dtype=np.ndarray)
Properity_onepointfour_new[logic_extend]=Properity_onepointfour
Properity_onepointfour_new[logic_lowL_extend]=Properity_onepointfour_lowL
Properity_onepointfour_new=Properity_onepointfour_new[logic_new]

mass_beta_Lambda_result_new=np.full((len(logic_new),3,40),1,dtype=np.ndarray)
mass_beta_Lambda_result_new[logic_extend]=mass_beta_Lambda_result
mass_beta_Lambda_result_new[logic_lowL_extend]=mass_beta_Lambda_result_lowL
mass_beta_Lambda_result_new=mass_beta_Lambda_result_new[logic_new]

chirp_q_Lambdabeta6_Lambda1Lambda2_new=np.full((len(logic_new),4),1,dtype=np.ndarray)
chirp_q_Lambdabeta6_Lambda1Lambda2_new[logic_extend]=chirp_q_Lambdabeta6_Lambda1Lambda2
chirp_q_Lambdabeta6_Lambda1Lambda2_new[logic_lowL_extend]=chirp_q_Lambdabeta6_Lambda1Lambda2_lowL
chirp_q_Lambdabeta6_Lambda1Lambda2_new=chirp_q_Lambdabeta6_Lambda1Lambda2_new[logic_new]



path = "./"
dir_name='Lambda_PNM_margueron_calculation_parallel_new'
f=open(path+dir_name+'/Lambda_PNM_calculation_args.dat','wb')
cPickle.dump(args_new,f)
f.close()
f=open(path+dir_name+'/Lambda_PNM_calculation_eos.dat','wb')
cPickle.dump(eos_new,f)
f.close()
f=open(path+dir_name+'/Lambda_PNM_calculation_eos_flat_logic.dat','wb')
cPickle.dump(eos_new.flatten()[logic_new],f)
f.close()
f=open(path+dir_name+'/Lambda_PNM_calculation_maxmass.dat','wb')
cPickle.dump(maxmass_result_new,f)
f.close()
f=open(path+dir_name+'/Lambda_PNM_calculation_onepointfour.dat','wb')
cPickle.dump(Properity_onepointfour_new,f)
f.close()
f=open(path+dir_name+'/Lambda_PNM_calculation_mass_beta_Lambda.dat','wb')
cPickle.dump(mass_beta_Lambda_result_new,f)
f.close()
f=open(path+dir_name+'/Lambda_hadronic_calculation_chirpmass_Lambdabeta6.dat','wb')
cPickle.dump(chirp_q_Lambdabeta6_Lambda1Lambda2_new,f)
f.close()