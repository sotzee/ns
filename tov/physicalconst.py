# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:05:08 2016

@author: sotzee
"""

#cgs unit

global c,G,mass_sun
import scipy.constants as const
from astropy.constants import M_sun
c=100*const.c
G=1000*const.G
e=1e7*const.e
hbar=const.hbar*1e7
mass_sun=1000*M_sun.value
mass_per_baryon=const.m_n/10000