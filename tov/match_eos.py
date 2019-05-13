#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 09:00:20 2019

@author: sotzee
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def equations(x,args):
    a,b,c,gamma=x
    n_s,n1,p1,e1,n2,p2,e2=args
    u1=n1/n_s
    u2=n2/n_s
    eq1=(a-b/u1+c*u1**gamma)*n1-e1
    eq2=(a-b/u2+c*u2**gamma)*n2-e2
    eq3=(b+c*gamma*u1**(gamma+1))*n_s-p1
    eq4=(b+c*gamma*u2**(gamma+1))*n_s-p2
    return eq1,eq2,eq3,eq4

def match_get_eos_array_equations(u_array,args):
    a,b,c,gamma,n_s=args
    e_array=(a-b/u_array+c*u_array**gamma)*n_s*u_array
    p_array=(b+c*gamma*u_array**(gamma+1))*n_s
    return u_array,e_array,p_array

def equations3(x,args):
    a,b,c,d,gamma=x
    n_s,n1,p1,e1,dpdn1,n2,p2,e2=args
    u1=n1/n_s
    u2=n2/n_s
    eq1=(a-b/u1+c*u1**gamma+d*np.log(u1))*n1-e1
    eq2=(a-b/u2+c*u2**gamma+d*np.log(u2))*n2-e2
    eq3=(b+c*gamma*u1**(gamma+1)+d*u1)*n_s-p1
    eq4=(b+c*gamma*u2**(gamma+1)+d*u2)*n_s-p2
    eq5=(c*gamma*(gamma+1)*u1**(gamma)+d)-dpdn1
    return eq1,eq2,eq3,eq4,eq5

def match_get_eos_array_equations3(u_array,args):
    a,b,c,d,gamma,n_s=args
    e_array=(a-b/u_array+c*u_array**gamma+d*np.log(u_array))*n_s*u_array
    p_array=(b+c*gamma*u_array**(gamma+1)+d*u_array)*n_s
    return u_array,e_array,p_array

def equations4(x,args):
    a,b,c,gamma=x
    n_s,n1,p1,e1,dpdn1,n2,p2,e2=args
    d=dpdn1
    u1=n1/n_s
    u2=n2/n_s
    eq1=(a-b/u1)*n1-e1
    eq2=(a-b/u2+c*(u2-u1)**gamma+d*np.log(u2/u1))*n2-e2
    eq3=(b+d*u1)*n_s-p1
    eq4=(b+c*gamma*(u2-u1)**(gamma-1)*u2**2+d*u2)*n_s-p2
    return eq1,eq2,eq3,eq4

def match_get_eos_array_equations4(u_array,args):
    a,b,c,gamma,dpdn1,u1,n_s=args
    d=dpdn1
    e_array=(a-b/u_array+c*(u_array-u1)**gamma+d*np.log(u_array/u1))*n_s*u_array
    p_array=(b+c*gamma*(u_array-u1)**(gamma-1)*u_array**2+d*u_array)*n_s
    return u_array,e_array,p_array

def match_eos4(args):
    n_s,n1,p1,e1,dpdn1,n2,p2,e2=args
    u1=n1/n_s
    u2=n2/n_s
    d=dpdn1
    b=p1/n_s-d*u1
    a=e1/n1+b/u1
    #c*(u2-u1)**gamma=e2/n2+b/u2-a-d*np.log(u2/u1)
    #c*gamma*(u2-u1)**(gamma)*u2**2=(p2/n_s-d*u2-b)*(u2-u1)
    gamma=(p2/n_s-d*u2-b)*(u2-u1)/((e2/n2+b/u2-a-d*np.log(u2/u1))*u2**2)
    c=(e2/n2+b/u2-a-d*np.log(u2/u1))/((u2-u1)**gamma)
    return a,b,c,gamma,d,u1,n_s

# =============================================================================
# def equations(x,args):
#     a,b,c,d,gamma=x
#     n_s,n1,p1,e1,n2,p2,e2=args
#     u1=n1/n_s
#     u2=n2/n_s
#     eq1=(a-b/u1+c*np.log(u1))*n1-e1
#     eq2=(a-b/u2+c*np.log(u2)+d*(u2-u1)**gamma)*n2-e2
#     eq3=(b+c*u1+d*gamma*u1**(gamma+1))*n_s*u1**2-p1
#     eq4=(b+c*u2+d*gamma*u2**(gamma+1))*n_s*u2**2-p2
#     return eq1,eq2,eq3,eq4
# 
# def match_get_eos_array_equations(u_array,args):
#     a,b,c,gamma,n_s=args
#     e_array=(a-b/u_array+c*u_array*gamma)*n_s*u_array
#     p_array=(b+c*gamma*u_array**(gamma+1))*n_s*u_array**2
#     return u_array,e_array,p_array
# 
# n_s=0.16
# n1=0.1
# n2=1*0.16
# p1=0.618833146813
# e1=95.0749758461
# p2=2.66666666667
# e2=152.8
# args=opt.root(equations,[1000,5,50,1],tol=1e-8,args=[n_s,n1,p1,e1,n2,p2,e2])
# eos_array=match_get_eos_array_equations(np.linspace(n1/n_s,n2/n_s,100),list(args.x)+[n_s])
# plt.figure()
# plt.plot(eos_array[0],eos_array[1]/(n_s*eos_array[0]))
# u_low=np.linspace(0.8*n1/n_s,1.1*n1/n_s,100)
# plt.plot(u_low,p1/n1**2*(n_s*u_low-n1)+e1/n1)
# u_high=np.linspace(0.8*n2/n_s,1.1*n2/n_s,100)
# plt.plot(u_high,p2/n2**2*(n_s*u_high-n2)+e2/n2)
# 
# plt.figure()
# plt.plot(eos_array[0],eos_array[2])
# u_low=np.linspace(0.8*n1/n_s,1.1*n1/n_s,100)
# plt.plot(u_low,p1*(u_low/u_low))
# u_high=np.linspace(0.8*n2/n_s,1.1*n2/n_s,100)
# plt.plot(u_high,p2*(u_high/u_high))
# plt.ylim(0,1.5*p2)
# =============================================================================

def match_get_eos_array(u_array,args):
    a,b,c,gamma,u1,n_s=args
    e_array=n_s*u_array*(b-a/u_array+c*(u_array/u1-1)**gamma)
    p_array=n_s*(a+c*gamma*u_array**2/u1*(u_array/u1-1)**(gamma-1))
    return u_array,e_array,p_array
def match_eos(n_s,n1,p1,e1,n2,p2,e2):
    a=p1/n_s
    u1=n1/n_s
    b=(p1+e1)/n1
    gamma=(n2-n1)*(p2-p1)/(n2*(e2+p1)-n2**2*b)
    c=((e2+p1)/n2-b)/(n2/n1-1)**gamma
    return a,b,c,gamma,u1,n_s

#0.16 0.1 0.618833146813 95.0749758461 0.16 2.66666666667 152.8
from eos_class import EOS_BPS
n_s=0.16
n1=0.1
n2=1*0.16
p1=EOS_BPS.eosPressure_frombaryon(n1)
e1=EOS_BPS.eosDensity(p1)
dpdn1=EOS_BPS().eosCs2(p1)*EOS_BPS().eosChempo(p1)
p2=2.666667
e2=152.8
eos_array_list=[]
eos_label_list=[]
args_1=match_eos(n_s,n1,p1,e1,n2,p2,e2)
eos_array_list.append(match_get_eos_array(np.linspace(n1/n_s,n2/n_s,100),args_1))
eos_label_list.append('type 1')
args_2=list(opt.root(equations,[955,3,3,5],tol=1e-8,args=[n_s,n1,p1,e1,n2,p2,e2]).x)+[n_s]
eos_array_list.append(match_get_eos_array_equations(np.linspace(n1/n_s,n2/n_s,100),args_2))
eos_label_list.append('type 2')
args_3=list(opt.root(equations3,[955,3.1453328966256469,2.5839138055246758,0.1,5.2328896495712973],tol=1e-8,args=[n_s,n1,p1,e1,dpdn1,n2,p2,e2]).x)+[n_s]
eos_array_list.append(match_get_eos_array_equations3(np.linspace(n1/n_s,n2/n_s,100),args_3))
eos_label_list.append('type 3')
args_4=list(opt.root(equations4,[955,3.1453328966256469,2.5839138055246758,5.2328896495712973],tol=1e-8,args=[n_s,n1,p1,e1,dpdn1,n2,p2,e2]).x)+[dpdn1,n1/n_s,n_s]
eos_array_list.append(match_get_eos_array_equations4(np.linspace(n1/n_s,n2/n_s,100),args_4))
eos_label_list.append('type 4')
args_5=match_eos4([n_s,n1,p1,e1,dpdn1,n2,p2,e2])
eos_array_list.append(match_get_eos_array_equations4(np.linspace(n1/n_s,n2/n_s,100),args_5))
eos_label_list.append('type 5')

plt.figure()
for eos_array,label_tex in zip(eos_array_list,eos_label_list):
    plt.plot(eos_array[0],eos_array[1]/(n_s*eos_array[0]),label=label_tex)
u_low=np.linspace(0.8*n1/n_s,1.1*n1/n_s,100)
plt.plot(u_low,p1/n1**2*(n_s*u_low-n1)+e1/n1)
u_high=np.linspace(0.8*n2/n_s,1.1*n2/n_s,100)
plt.plot(u_high,p2/n2**2*(n_s*u_high-n2)+e2/n2)
plt.legend()
plt.figure()
for eos_array,label_tex in zip(eos_array_list,eos_label_list):
    plt.plot(eos_array[0],eos_array[2],label=label_tex)
u_low=np.linspace(0.8*n1/n_s,1.1*n1/n_s,100)
plt.plot(u_low,p1*(u_low/u_low))
u_high=np.linspace(0.8*n2/n_s,1.1*n2/n_s,100)
plt.plot(u_high,p2*(u_high/u_high))
plt.ylim(0,1.5*p2)
plt.legend()

