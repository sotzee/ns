#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:59:22 2018

@author: sotzee
"""

import numpy as np
def I_from_lambda(lambda_array):
    coeff=[1.47,0.0817,0.0149,0.000287,-0.0000364]
    log_lambda_array=np.log(lambda_array)
    return np.exp(coeff[0]+coeff[1]*log_lambda_array+coeff[2]*log_lambda_array**2+coeff[3]*log_lambda_array**3+coeff[4]*log_lambda_array**4)
