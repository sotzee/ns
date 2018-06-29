# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:03:02 2016

@author: sotzee
"""

unitMeVfm=1.302e-7
unitPressure=2.085e26
unitDensity=2.32e5
unitBaryonDensity=1e-39
unitMass=2.0e33

def toPressure(pressure_before,unit_before):
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return pressure_before/unitMeVfm*unitPressure
    if(unit_before=='mev4'):
        return pressure_before*unitPressure
    if(unit_before=='mev'):
        return pressure_before**4.0*unitPressure
    if(unit_before=='fm-4'):
        return pressure_before/unitMeVfm**(4./3)*unitPressure
        
def toDensity(density_before,unit_before):
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return density_before/unitMeVfm*unitDensity
    if(unit_before=='mev4'):
        return density_before*unitDensity
    if(unit_before=='mev'):
        return density_before**4.0*unitDensity
    if(unit_before=='fm-4'):
        return density_before/unitMeVfm**(4/3)*unitDensity

def toBaryonDensity(density_before,unit_before):
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return density_before/unitBaryonDensity
        
def toMevfm(before,unit_before):
    if(unit_before=='pressure'):
        return before*unitMeVfm/unitPressure
    if(unit_before=='density'):
        return before*unitMeVfm/unitDensity
    if(unit_before=='baryondensity'):
        return before*unitBaryonDensity
    if(unit_before=='mev4'):
        return before*unitMeVfm
        

def toMev4(before,unit_before):
    if(unit_before=='pressure'):
        return before/unitPressure
    if(unit_before=='density'):
        return before/unitDensity
        
def toMev(before,unit_before):
    if(unit_before=='pressure'):
        return (before/unitPressure)**0.25
    if(unit_before=='density'):
        return (before/unitDensity)**0.25
        
#test:
#print('test:')
#print(toMevfm(toPressure(10,'mevfm'),'pressure'))
#print(toMevfm(toDensity(10,'mevfm'),'density'))
#print(toMev4(toPressure(10,'mev4'),'pressure'))
#print(toMev4(toDensity(10,'mev4'),'density'))
#print(toMev(toPressure(10,'mev'),'pressure'))
#print(toMev(toDensity(10,'mev'),'density'))