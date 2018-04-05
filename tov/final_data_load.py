#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:07:44 2018

@author: sotzee
"""

import numpy as np
import pickle
from eos_class import EOS_BPSwithPoly

def mass_chirp(m1,m2):
    return (m1*m2)**0.6/(m1+m2)**0.2

def tidal_binary(m1,m2,tidal1,tidal2):
    return 16*((m1+12*m2)*m1**4*tidal1+(m2+12*m1)*m2**4*tidal2)/(13*(m1+m2)**5)

class final_data_BPSwithPolyCSS(object):
    def __init__(self,parameter):
        self.parameter=parameter
        self.N=np.size(parameter)
        args_matrix=list()
        properity_matrix=list()
        stars_type0=list()
        stars_type1=list()
        stars_type2=list()
        stars_type3=list()
        tidal_binary_matrix=list()
        density_trans=list()
        baryondensity_trans=list()
        gamma2=list()
        for i in range(self.N):
            eos=EOS_BPSwithPoly(parameter[i].args[0:7])
            density_trans.append(eos.eosDensity(parameter[i].args[7]))
            baryondensity_trans.append(eos.eosBaryonDensity(parameter[i].args[7]))
            gamma2.append(eos.eosPiecewisePoly.gamma2)
            tidal_binary_matrix.append([parameter[i].stars[0][-1],parameter[i].stars[1][-1],parameter[i].stars[2][-1],parameter[i].stars[3][-1],parameter[i].stars[4][-1]])
            args_matrix.append(parameter[i].args)
            properity_matrix.append(parameter[i].properity)
            for j in range(np.size(parameter[i].stars)/9-5):
                if(parameter[i].stars[j+5][0]==0):
                    stars_type0.append(parameter[i].stars[j+5])
                if(parameter[i].stars[j+5][0]==1):
                    stars_type1.append(parameter[i].stars[j+5])
                if(parameter[i].stars[j+5][0]==2):
                    stars_type2.append(parameter[i].stars[j+5])
                if(parameter[i].stars[j+5][0]==3):
                    stars_type3.append(parameter[i].stars[j+5])

        args_matrix=np.array(args_matrix)
        self.args_matrix=args_matrix.transpose()
        properity_matrix=np.array(properity_matrix)
        self.properity_matrix=properity_matrix.transpose()
        
        stars_type0=np.array(stars_type0)
        self.stars_type0=stars_type0.transpose()
        stars_type1=np.array(stars_type1)
        self.stars_type1=stars_type1.transpose()
        stars_type2=np.array(stars_type2)
        self.stars_type2=stars_type2.transpose()
        stars_type3=np.array(stars_type3)
        self.stars_type3=stars_type3.transpose()
        
        tidal_binary_matrix=np.array(tidal_binary_matrix)
        tidal_binary_matrix=tidal_binary_matrix.transpose()
        self.density_trans=np.array(density_trans)
        self.baryondensity_trans=np.array(baryondensity_trans)
        self.gamma2=np.array(gamma2)
        if(self.N>0):
            self.tidal_binary1=tidal_binary(1.364653646,1.364653646,tidal_binary_matrix[2],tidal_binary_matrix[2])
            self.tidal_binary2=tidal_binary(1.243697915,1.5,tidal_binary_matrix[1],tidal_binary_matrix[3])
            self.tidal_binary3=tidal_binary(1.049578992,1.8,tidal_binary_matrix[0],tidal_binary_matrix[4])
            tidal_binay123=np.array([self.tidal_binary1,self.tidal_binary2,self.tidal_binary3])
            self.tidal_binary_max=tidal_binay123.max(0)
            self.tidal_binary_min=tidal_binay123.min(0)
            self.pressure2=self.args_matrix[3]
            self.pressure_trans=self.args_matrix[7]
            self.det_density=self.args_matrix[8]
            self.Maximum_mass=self.properity_matrix[2]
            [self.R_opf,self.beta_opf,self.M_binding_opf,self.momentofinertia_opf,self.yR_opf,self.tidal_opf]=self.properity_matrix[5:11]
            [self.R_opf_quark,self.beta_opf_quark,self.M_binding_opf_quark,self.momentofinertia_opf_quark,self.yR_opf_quark,self.tidal_opf_quark]=self.properity_matrix[13:19]
        else:
            self.tidal_binary1=[]
            self.tidal_binary2=[]
            self.tidal_binary3=[]
            self.tidal_binary_max=[]
            self.tidal_binary_min=[]
            self.pressure2=[]
            self.pressure_trans=[]
            self.det_density=[]
            self.Maximum_mass=[]
            [self.R_opf,self.beta_opf,self.M_binding_opf,self.momentofinertia_opf,self.yR_opf,self.tidal_opf]=[[],[],[],[],[],[]]
            [self.R_opf_quark,self.beta_opf_quark,self.M_binding_opf_quark,self.momentofinertia_opf_quark,self.yR_opf_quark,self.tidal_opf_quark]=[[],[],[],[],[],[]]

    def tidal_binary_max_restriction(self,tidal_binary_devide):
        within_tidal_binary_devide=list()
        beyond_tidal_binary_devide=list()
        for i in range(self.N):
            if (self.tidal_binary_min[i]<tidal_binary_devide):
                within_tidal_binary_devide.append(self.parameter[i])
            else:
                beyond_tidal_binary_devide.append(self.parameter[i])
        return final_data_BPSwithPolyCSS(within_tidal_binary_devide),final_data_BPSwithPolyCSS(beyond_tidal_binary_devide)

def merge_final_data_BPSwithPolyCSS(final_data1,final_data2):
    return final_data_BPSwithPolyCSS(final_data1.parameter+final_data2.parameter)

unit_MeV4_to_MeVfm3=1.302e-7
m0=939.5654
n_s=0.16
A0=m0**4/np.pi**2*unit_MeV4_to_MeVfm3
def stable_final_data_BPSwithPolyCSS(final_data):
    stable_parameter=[]
    for i in range(final_data.N):
        eos=EOS_BPSwithPoly(final_data.parameter[i].args[0:7])
        pressure_trans=final_data.parameter[i].args[7]
        det_density=final_data.parameter[i].args[8]
        cs2=final_data.parameter[i].args[9]
        density_trans=eos.eosDensity(pressure_trans)
        baryondensity_trans=eos.eosBaryonDensity(pressure_trans)
        chempo=(density_trans+pressure_trans)/baryondensity_trans
        if(1-cs2-4.*A0*((chempo/m0)**2-1)**2.5/(45*chempo*(density_trans+det_density+pressure_trans)/m0)>0):
            stable_parameter.append(final_data.parameter[i])
    return final_data_BPSwithPolyCSS(stable_parameter)
    
def read_parameter(p1,cs2,Calculation_mode,Hybrid_sampling):
    if(Calculation_mode=='hybrid'):
        print('%s'%Hybrid_sampling)
        print('pressure1=%.1f'%p1)
        print('cs2=%.2f'%cs2)
        if(cs2==1.):
            dir_name='data_p1=%.1f_cs2=%.1f'%(p1,cs2)
        else:
            dir_name='data_p1=%.1f_cs2=%.12f'%(p1,cs2)
        f2=open('./'+dir_name+'/sorted_%s_%s_2.0-2.4.dat_addofmass_addstars'%(Calculation_mode,Hybrid_sampling),'rb')
        parameter=pickle.load(f2)
        f2.close()
        return parameter
    
    elif(Calculation_mode=='hadronic'):
        dir_name='data_hadronic'
        f2=open('./'+dir_name+'/sorted_%s_2.0-2.4.dat_addofmass_addstars'%(Calculation_mode),'rb')
        parameter=pickle.load(f2)
        f2.close()
        return parameter

def read_parameter_normal_and_low_trans(p1,cs2):
    Calculation_mode='hybrid'
    normal_trans=read_parameter(p1,cs2,Calculation_mode,'normal_trans')
    low_trans=read_parameter(p1,cs2,Calculation_mode,'low_trans')
    return final_data_BPSwithPolyCSS(normal_trans),final_data_BPSwithPolyCSS(low_trans)