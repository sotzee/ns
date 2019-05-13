#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:03:29 2019

@author: sotzee
"""

import scipy.optimize as opt
import numpy as np

def logic_no_over(sol_x,init_i,args):
    return np.abs(sol_x).max()<1e6

def solve_equations(equations,init_list,args,vary_list=np.linspace(1.,1.,1),tol=1e-20,logic_success_f=logic_no_over,equations_extra_args=[]):
    shape_init=np.array(init_list).shape
    init_vary_list=np.multiply(init_list,np.tile(np.array(np.meshgrid(*([vary_list]*shape_init[1]))), (shape_init[0],)+(1,)*(shape_init[1]+1)).transpose(list(range(2,shape_init[1]+2))+list(range(2)))).reshape((len(vary_list)**shape_init[1]*shape_init[0],shape_init[1]))
    for init_i in init_vary_list:
        sol = opt.root(equations,init_i,tol=tol,args=(args,equations_extra_args),method='hybr')
        if (sol.success and logic_success_f(sol.x,init_i,args)):
            return True,list(sol.x)
    return False,list(sol.x)


def get_nearby_logic(logic,false_array):
    shape=np.shape(logic)
    N=len(shape)
    result=np.full(shape,False,bool)
    for i in range(len(shape)):
        transpose_axis=list(range(N))
        #print transpose_axis
        transpose_axis[0]=i
        transpose_axis[i]=0
        logic_i=np.transpose(logic,transpose_axis)
        #print transpose_axis
        logic_i_plus=np.concatenate(([false_array[i]],logic_i[:(shape[i]-1)]))
        logic_i_minus=np.concatenate((logic_i[1:(shape[i])],[false_array[i]]))
        logic_i_plus_minus=np.transpose(np.logical_or(logic_i_plus,logic_i_minus),transpose_axis)
        result=np.logical_or(result,logic_i_plus_minus)
    result[logic]=False
    return result
# =============================================================================
# #to test 'get_nearby_logic'
# print 0,len(logic_test[logic_test])
# logic_result=get_nearby_logic(logic_test,false_array_list)
# print len(np.logical_and(logic_result,logic_test)[np.logical_and(logic_result,logic_test)]),len(logic_result[logic_result])
# #print logic_result
# for i in range(8):
#     logic_test=np.logical_or(logic_result,logic_test)
#     logic_result=get_nearby_logic(logic_test,false_array_list)
#     print len(np.logical_and(logic_result,logic_test)[np.logical_and(logic_result,logic_test)]),len(logic_result[logic_result])
#     #print logic_result
# =============================================================================

def get_false_array_list(shape):
    false_array_list=[]
    for i in range(len(shape)):
        shape_i=list(shape)
        shape_i[i]=shape_i[0]
        false_array_list.append(np.full(shape_i[1:],False,bool))
    return false_array_list

def get_result_near(logic,false_array,logic_calculated,result):
    shape=np.shape(logic)
    N=len(shape)
    i=0
    result_nearby=[]
    logic_calculated_nearby=[]
    for i in range(len(shape)):
        transpose_axis=list(range(N+1))
        transpose_axis[0]=i
        transpose_axis[i]=0
        result_i=np.transpose(result,transpose_axis)
        result_i_plus=np.transpose(np.concatenate(([result_i[shape[i]-1]],result_i[:(shape[i]-1)])),transpose_axis)
        result_i_minus=np.transpose(np.concatenate((result_i[1:(shape[i])],[result_i[0]])),transpose_axis)
        result_nearby+=[result_i_plus,result_i_minus]
        logic_calculated_i=np.transpose(logic_calculated,transpose_axis[:-1])
        logic_calculated_plus=np.transpose(np.concatenate(([false_array[i]],logic_calculated_i[:(shape[i]-1)])),transpose_axis[:-1])
        logic_calculated_minus=np.transpose(np.concatenate((logic_calculated_i[1:(shape[i])],[false_array[i]])),transpose_axis[:-1])
        logic_calculated_nearby+=[np.logical_and(logic_calculated_plus,np.logical_not(logic_calculated)),np.logical_and(logic_calculated_minus,np.logical_not(logic_calculated))]
    result_nearby=np.array(result_nearby)
    logic_calculated_nearby=np.array(logic_calculated_nearby)
    #print result_nearby[:,logic,:],logic_calculated_nearby[:,logic]
    return result_nearby,logic_calculated_nearby

def Calculation_unparallel(equations,logic_calculatable_array,init_array,args_array,vary_list=np.linspace(1.,1.,1),tol=1e-10,logic_success_f=logic_no_over,equations_extra_args=[]):
    result_to_be_calculated=[]
    success_to_be_calculated=[]
    for logic_calculatable_array_i,init_array_i,args_array_i in zip(logic_calculatable_array,init_array,args_array):
        #print init_array_i[logic_calculatable_array_i]
        success_i,result_i=solve_equations(equations,init_array_i[logic_calculatable_array_i],args_array_i,vary_list=vary_list,tol=tol,logic_success_f=logic_success_f,equations_extra_args=equations_extra_args)
        success_to_be_calculated.append(success_i)
        result_to_be_calculated.append(result_i)
    return np.array(success_to_be_calculated),np.array(result_to_be_calculated)

def get_init(shape,index_list,result_list):
    init_calculated_logic=np.full(shape,False,bool)
    init_result=np.full(shape+(len(result_list[0]),),0.0)
    init_record_calculation=np.full((2*len(shape),)+shape,True,bool)
    init_logic=np.full(shape,False,bool)
    init_logic_nearby=np.full((2*len(shape),)+shape,False,bool)
    init_result_nearby=np.full((2*len(shape),)+shape+(len(result_list[0]),),0.0)
    for index_i,result_i in zip(index_list,result_list):
        init_logic[index_i]=True
        init_logic_nearby[(0,)+index_i]=True
        #print (0,)+index_i
        init_result_nearby[(0,)+index_i]=result_i
    return init_calculated_logic,init_result,init_record_calculation,init_logic,init_logic_nearby,init_result_nearby

def explore_solutions(equations,args,init_index_tadpole,init_result_tadpole,vary_list=np.linspace(1.,1.,1),tol=1e-10,logic_success_f=logic_no_over,Calculation_routing=Calculation_unparallel,equations_extra_args=[]):
    shape_args_space=args.shape[1:]
    args_total_number=np.prod(shape_args_space)
    false_array_list=get_false_array_list(shape_args_space)
    print('Setting initial value for index %s'%list(init_index_tadpole))
    print(init_result_tadpole)
    init_stuff=get_init(shape_args_space,init_index_tadpole,init_result_tadpole)
    calculated_logic=init_stuff[0]
    result=init_stuff[1]
    record_calculation=init_stuff[2]
    to_be_calculated_logic=init_stuff[3]
    logic_calculatable_nearby=init_stuff[4]
    result_nearby=init_stuff[5]
    print result_nearby[logic_calculatable_nearby]
    i=0
    print('################%d'%i)
    print('calculated comfiguration number: %d'%calculated_logic[calculated_logic].shape[0])
    print('initial comfiguration number: %d'%to_be_calculated_logic[to_be_calculated_logic].shape[0])
    while(len(to_be_calculated_logic[to_be_calculated_logic])>0):
        logic_calculatable_array=logic_calculatable_nearby[:,to_be_calculated_logic].transpose()
        init_array=np.transpose(result_nearby[:,to_be_calculated_logic,:],(1,0,2))
        args_array=args[:,to_be_calculated_logic].transpose()
        print('calculatable trial directions number: %d'%logic_calculatable_nearby[logic_calculatable_nearby].shape[0])
        
        print('begin calculating...')
        success_to_be_calculated,result_to_be_calculated=Calculation_routing(equations,logic_calculatable_array,init_array,args_array,vary_list=vary_list,tol=tol,logic_success_f=logic_success_f,equations_extra_args=equations_extra_args)
        print('successful comfiguration number: %d \t(%.2f%%)'%(success_to_be_calculated[success_to_be_calculated].shape[0],100.*success_to_be_calculated[success_to_be_calculated].shape[0]/to_be_calculated_logic[to_be_calculated_logic].shape[0]))
        #print  success_to_be_calculated,result_to_be_calculated
        calculated_logic[to_be_calculated_logic]=success_to_be_calculated
        print('calculated comfiguration number: %d \t(%.2f%%)'%(calculated_logic[calculated_logic].shape[0],100.*calculated_logic[calculated_logic].shape[0]/args_total_number))
        result[to_be_calculated_logic]=result_to_be_calculated
        #print record_to_be_altered,record_calculation[logic_calculatable_nearby]
        record_calculation[logic_calculatable_nearby]=False
        #print record_to_be_altered,record_calculation[logic_calculatable_nearby]
    
        i+=1
        print('################%d'%i)
        to_be_calculated_logic=get_nearby_logic(calculated_logic,false_array_list)
        print('nearby comfiguration number: %d'%to_be_calculated_logic[to_be_calculated_logic].shape[0])
        result_nearby,logic_calculated_nearby=get_result_near(to_be_calculated_logic,false_array_list,calculated_logic,result)
        
        logic_calculatable_nearby = np.logical_and(logic_calculated_nearby,record_calculation)
        to_be_calculated_logic=logic_calculatable_nearby.sum(axis=0).astype(bool)
        print('calculatable nearby comfiguration number: %d'%to_be_calculated_logic[to_be_calculated_logic].shape[0])
    print('Calculated ratio is %.2f%%'%(100.*calculated_logic[calculated_logic].shape[0]/np.size(calculated_logic)))
    return calculated_logic,result.transpose([len(shape_args_space)]+list(range(len(shape_args_space))))

# =============================================================================
# #This is a test problem solve coordinate transformation from cartesian to polar
# def equations_example(para,args,args_extra):
#     x,y,z=args
#     r,theta,phi=para
#     return r*np.sin(theta)*np.cos(phi)-x,r*np.sin(theta)*np.sin(phi)-y,r*np.cos(theta)-z
# 
# def logic_example(sol_x,init_i,args):
#     return sol_x[0]>=0 and 0<=sol_x[1]<=np.pi and 0<=sol_x[2]<=2*np.pi
# 
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# args=np.mgrid[-1:1:21j,-1:1:21j,-1:1:21j]
# calculated_logic,result=explore_solutions(equations_example,args,((20,10,10),(0,10,10)),((1,np.pi/2,0),(1,np.pi/2,np.pi)),vary_list=np.array([1.]),tol=1e-10,logic_success_f=logic_example,Calculation_routing=Calculation_unparallel)
# x = args[0]
# y = args[1]
# z = args[2]
# r = result[:,:,:,0]
# theta = result[:,:,:,1]
# phi = result[:,:,:,2]
# 
# fig = plt.figure(figsize=(18,5))
# title_list=['r','$\\theta$','$\phi$']
# for i in [0,1,2]:
#     ax =fig.add_subplot(131+i, projection='3d')
#     ax.set_title(title_list[i])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     cs=ax.scatter(x, y, z, c=result[:,:,:,i].flatten(),edgecolors='face', cmap=plt.jet())
#     plt.colorbar(cs)
# plt.show()
# =============================================================================


# =============================================================================
# #This is tempelate for Paralle Exploring
# #Paralle process is unable to accept equations and logic_success_f as parameters.
# #NEED TO DEFINE EXPLICTLY IN SIDE!!!!!!!!!
# from Parallel_process import main_parallel_unsave  
# def Calculation_parallel_sub(args_i,other_args):
#     logic_calculatable_array_i,init_array_i,args_array_i=args_i
#     #equations,vary_list,tol,logic_success_f=other_args
#     equations=equations_example
#     logic_success_f=logic_example
#     vary_list=other_args[1]
#     tol=other_args[2]
#     success_i,result_i=solve_equations(equations,init_array_i[logic_calculatable_array_i],args_array_i,vary_list=vary_list,tol=tol,logic_success_f=logic_success_f)
#     return [success_i]+result_i
#     
# def Calculation_parallel(equations,logic_calculatable_array,init_array,args_array,vary_list=np.linspace(1.,1.,1),tol=1e-10,logic_success_f=logic_no_over):
#     main_parallel_result=main_parallel_unsave(Calculation_parallel_sub,zip(logic_calculatable_array,init_array,args_array),other_args=(equations,vary_list,tol,logic_success_f))
#     return main_parallel_result[:,0].astype('bool'),main_parallel_result[:,1:]
# =============================================================================