#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:31:45 2018

@author: sotzee
"""


name_satisfied='_2.0-'
name_unsatisfied='_1.4-2.0'

def constrain(eos_item):
    if(eos_item.properity[1]>2):
        return True
    else:
        return False



# =============================================================================
# name_satisfied='2.4'
# name_unsatisfied='_2.4-'
# 
# def constrain(eos_item):
#     if(eos_item.properity[1]<2.4):
#         return True
#     else:
#         return False
# =============================================================================
