#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:10:01 2018

@author: sotzee
"""

import pickle
import numpy as np

def get_input(remid_words,default):
    try:
        receive=input(remid_words)
    except SyntaxError:
        receive=default
    return receive

path=get_input('Enter the path of output array file(\'./\'): ','./')
file_name=get_input('Enter the file name of output array file(\'OfMass.dat\'): ','OfMass.dat')

flag_onebyone=input('Enter elements one by one?(\'y\' or \'n\'): ')
if(flag_onebyone=='y'):
    i=1
    array=list()
    while(True):
        element=input('Enter element %d(Enter \'done\' to finish): '%(i))
        print element
        if(element=='done'):
            break
        else:
            print element
            array.append(element)
        i+=1
        print array
    array=np.array(array)
else:
    flag_lin=input('Enter elements as linear array?(\'y\' or \'n\'): ')
    if(flag_onebyone=='y'):
        x0=input('Enter first elements: ')
        xf=input('Enter last element: ')
        N=input('Enter number of elements total(>2): ')
        array=np.linspace(x0,xf,N)

print('array: '+str(array))
print('was written into file: '+path+file_name)
f=open(path+file_name,'wb')
pickle.dump(array,f)
f.close()