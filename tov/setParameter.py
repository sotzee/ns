# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:27:25 2016

@author: Sotzee
"""

from scipy.optimize import brenth,newton
import numpy as np

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
import BPS
def setParameter(baryon_density0,pressure1,baryon_density1,baryon_density2,Preset_gamma2,Preset_num_pressure_trans,Preset_det_density,cs2,Hybrid_sampling):
    pressure0=BPS.eosPressure_frombaryon(baryon_density0)
    density0=BPS.eosDensity(pressure0)

    Preset_num_gamma2=Preset_gamma2[0]
    Preset_min_gamma2=Preset_gamma2[1]
    Preset_max_gamma2=Preset_gamma2[2]
    Preset_num_det_density = Preset_det_density[0]
    Preset_min_det_density = Preset_det_density[1]
    Preset_max_det_density = Preset_det_density[2]
    
    gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
    density1=(density0-pressure0/(gamma1-1.))*(pressure1/pressure0)**(1./gamma1)+pressure1/(gamma1-1.)
    def causality2(pressure2):
        gamma2=np.log(pressure2/pressure1)/np.log(baryon_density2/baryon_density1)
        density2=(density1-pressure1/(gamma2-1.))*(pressure2/pressure1)**(1./gamma2)+pressure2/(gamma2-1.)
        return gamma2*pressure2/(density2+pressure2)-1.
    def causality_trans(gamma2,pressure_trans):
        density_trans=(density1-pressure1/(gamma2-1.))*(pressure_trans/pressure1)**(1./gamma2)+pressure_trans/(gamma2-1.)
        return gamma2*pressure_trans/(density_trans+pressure_trans)-1.
    
    parameter=list()
    if(Hybrid_sampling=='hardest_before_trans'):
        max_pressure_trans = brenth(causality2,1,1000)
        min_pressure_trans = pressure1
        for i_pressure_trans in range(Preset_num_pressure_trans):
            pressure_trans = (i_pressure_trans+1)*(max_pressure_trans-min_pressure_trans)/(Preset_num_pressure_trans)+min_pressure_trans
            gamma2=newton(causality_trans,10,args=(pressure_trans,))
            density_trans=(density1-pressure1/(gamma2-1.))*(pressure_trans/pressure1)**(1./gamma2)+pressure_trans/(gamma2-1.)
            pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
            for i_det_density in range(Preset_num_det_density):
                det_density_reduced=Preset_min_det_density+1.0*i_det_density*(Preset_max_det_density-Preset_min_det_density)/(Preset_num_det_density-1.)
                parameter.append([pressure1,pressure2,gamma2,pressure_trans,density_trans,pressure_trans/density_trans,det_density_reduced*density_trans,det_density_reduced,cs2])

    elif(Hybrid_sampling=='softest_before_trans'):
        max_pressure_trans = brenth(causality2,1,1000)
        min_pressure_trans = pressure1
        for i_pressure_trans in range(Preset_num_pressure_trans):
            pressure_trans = (i_pressure_trans+1)*(max_pressure_trans-min_pressure_trans)/(Preset_num_pressure_trans)+min_pressure_trans
            gamma2=np.log(pressure_trans/pressure1)/np.log(baryon_density2/baryon_density1)
            density_trans=(density1-pressure1/(gamma2-1.))*(pressure_trans/pressure1)**(1./gamma2)+pressure_trans/(gamma2-1.)
            pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
            for i_det_density in range(Preset_num_det_density):
                det_density_reduced=Preset_min_det_density+1.0*i_det_density*(Preset_max_det_density-Preset_min_det_density)/(Preset_num_det_density-1.)
                print gamma2
                parameter.append([pressure1,pressure2,gamma2,pressure_trans,density_trans,pressure_trans/density_trans,det_density_reduced*density_trans,det_density_reduced,cs2])
    elif(Hybrid_sampling=='low_trans'):
        max_pressure_trans = pressure1
        min_pressure_trans = pressure1*(0.16/baryon_density1)**(gamma1)
        PREVENT_ZERO = 1
    elif(Hybrid_sampling=='normal_trans'):
        max_pressure_trans = brenth(causality2,1,1000)
        min_pressure_trans = pressure1
        PREVENT_ZERO = 0
    else:
        print('Hybrid_sampling==%s, which is not valid.'%Hybrid_sampling)

    if(Hybrid_sampling=='low_trans'):
        for i_pressure_trans in range(Preset_num_pressure_trans):
            pressure_trans = i_pressure_trans*(max_pressure_trans-min_pressure_trans)/(Preset_num_pressure_trans-1.)+min_pressure_trans
            for i_det_density in range(Preset_num_det_density):
                det_density_reduced=Preset_min_det_density+1.0*i_det_density*(Preset_max_det_density-Preset_min_det_density)/(Preset_num_det_density-1.)
                for i_gamma2 in range(Preset_num_gamma2):
                    gamma2=1.0*i_gamma2*(Preset_max_gamma2-Preset_min_gamma2)/(Preset_num_gamma2-1+PREVENT_ZERO)+Preset_min_gamma2
                    pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
                    density_trans=(density0-pressure0/(gamma1-1))*(pressure_trans/pressure0)**(1./gamma1)+pressure_trans/(gamma1-1.)
                    if(((baryon_density2/baryon_density1)**gamma2*pressure1>=pressure_trans) & (gamma2*pressure_trans/(density_trans+pressure_trans)<=1)):
                        parameter.append([pressure1,pressure2,gamma2,pressure_trans,density_trans,pressure_trans/density_trans,det_density_reduced*density_trans,det_density_reduced,cs2])

    elif(Hybrid_sampling=='normal_trans'):
        for i_pressure_trans in range(Preset_num_pressure_trans):
            pressure_trans = i_pressure_trans*(max_pressure_trans-min_pressure_trans)/(Preset_num_pressure_trans-1.)+min_pressure_trans
            for i_det_density in range(Preset_num_det_density):
                det_density_reduced=Preset_min_det_density+1.0*i_det_density*(Preset_max_det_density-Preset_min_det_density)/(Preset_num_det_density-1.)
                for i_gamma2 in range(Preset_num_gamma2):
                    gamma2=1.0*i_gamma2*(Preset_max_gamma2-Preset_min_gamma2)/(Preset_num_gamma2-1+PREVENT_ZERO)+Preset_min_gamma2
                    pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
                    density_trans=(density1-pressure1/(gamma2-1))*(pressure_trans/pressure1)**(1./gamma2)+pressure_trans/(gamma2-1.)
                    if(((baryon_density2/baryon_density1)**gamma2*pressure1>=pressure_trans) & (gamma2*pressure_trans/(density_trans+pressure_trans)<=1)):
                        parameter.append([pressure1,pressure2,gamma2,pressure_trans,density_trans,pressure_trans/density_trans,det_density_reduced*density_trans,det_density_reduced,cs2])

    return parameter



def setParameter_hadronic(baryon_density0,Preset_pressure1,baryon_density1,Preset_pressure2,baryon_density2,Preset_pressure3,baryon_density3):
    pressure0=BPS.eosPressure_frombaryon(baryon_density0)
    density0=BPS.eosDensity(pressure0)
    
    num_pressure1=Preset_pressure1[0]
    min_pressure1=Preset_pressure1[1]
    max_pressure1=Preset_pressure1[2]
    num_pressure2=Preset_pressure2[0]
    min_pressure2=Preset_pressure2[1]
    num_pressure3=Preset_pressure3[0]
    min_pressure3=Preset_pressure3[1]
    gamma1=np.log(max_pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
    def causality2(pressure2):
        gamma2=np.log(pressure2/max_pressure1)/np.log(baryon_density2/baryon_density1)
        density1=(density0-pressure0/(gamma1-1.))*(max_pressure1/pressure0)**(1./gamma1)+max_pressure1/(gamma1-1.)
        density2=(density1-max_pressure1/(gamma2-1.))*(pressure2/max_pressure1)**(1./gamma2)+pressure2/(gamma2-1.)
        return gamma2*pressure2/(density2+pressure2)-1.
    max_pressure2 = brenth(causality2,1,1000)
    def causality3(pressure3):
        density1=(density0-pressure0/(gamma1-1.))*(max_pressure1/pressure0)**(1./gamma1)+max_pressure1/(gamma1-1.)
        gamma2=np.log(max_pressure2/max_pressure1)/np.log(baryon_density2/baryon_density1)
        gamma3=np.log(pressure3/max_pressure2)/np.log(baryon_density3/baryon_density2)
        density2=(density1-max_pressure1/(gamma2-1.))*(max_pressure2/max_pressure1)**(1./gamma2)+max_pressure2/(gamma2-1.)
        density3=(density2-max_pressure2/(gamma3-1.))*(pressure3/max_pressure2)**(1./gamma3)+pressure3/(gamma3-1.)
        return gamma3*pressure3/(density3+pressure3)-1.
    max_pressure3 = brenth(causality3,min_pressure3,3000)-1.
    parameter=list()
    for i_p1 in range(num_pressure1):
        pressure1 = i_p1*(max_pressure1-min_pressure1)/(num_pressure1-1.)+min_pressure1
        for i_p2 in range(num_pressure2):
            pressure2 = i_p2*(max_pressure2-min_pressure2)/(num_pressure2-1.)+min_pressure2
            for i_p3 in range(num_pressure3):
                pressure3 = (max_pressure3/min_pressure3)**(i_p3/(num_pressure3-1.))*min_pressure3
                gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
                gamma2=np.log(pressure2/pressure1)/np.log(baryon_density2/baryon_density1)
                gamma3=np.log(pressure3/pressure2)/np.log(baryon_density3/baryon_density2)
                density1=(density0-pressure0/(gamma1-1.))*(pressure1/pressure0)**(1./gamma1)+pressure1/(gamma1-1.)
                density2=(density1-pressure1/(gamma2-1.))*(pressure2/pressure1)**(1./gamma2)+pressure2/(gamma2-1.)
                density3=(density2-pressure2/(gamma3-1.))*(pressure3/pressure2)**(1./gamma3)+pressure3/(gamma3-1.)
                if((gamma2*pressure2/(density2+pressure2)<=1) & (gamma3*pressure3/(density3+pressure3)<=1)&(pressure3>=1.05*pressure2)):
                    parameter.append([pressure1,pressure2,gamma2,pressure3,gamma3,0.0,0.0,0.0,0.0])
    return parameter


def setParameter_low_trans_complete(baryon_density0,Preset_pressure1,baryon_density1,baryon_density2,Preset_gamma2,Preset_num_pressure_trans,Preset_det_density,Preset_cs2,Hybrid_sampling):
    pressure0=BPS.eosPressure_frombaryon(baryon_density0)
    density0=BPS.eosDensity(pressure0)
    
    Preset_num_pressure1=Preset_pressure1[0]
    Preset_min_pressure1=Preset_pressure1[1]
    Preset_max_pressure1=Preset_pressure1[2]
    Preset_min_trans=(pressure0*Preset_min_pressure1)**0.5
    Preset_max_trans=Preset_max_pressure1
    Preset_num_det_density = Preset_det_density[0]
    Preset_min_det_density = Preset_det_density[1]
    Preset_max_det_density = Preset_det_density[2]
    Preset_num_cs2=Preset_cs2[0]
    Preset_min_cs2=Preset_cs2[1]
    Preset_max_cs2=Preset_cs2[2]
    gamma2=(Preset_gamma2[1]+Preset_gamma2[2])/2.
    
    parameter=list()
    if(Hybrid_sampling=='low_trans_complete'):
        for i_pressure1 in range(Preset_num_pressure1):
            pressure1=i_pressure1*(Preset_max_pressure1-Preset_min_pressure1)/(Preset_num_pressure1-1.)+Preset_min_pressure1
            gamma1=np.log(pressure1/pressure0)/np.log(baryon_density1/baryon_density0)
            pressure2=pressure1*(baryon_density2/baryon_density1)**gamma2
            for i_pressure_trans in range(Preset_num_pressure_trans):
                pressure_trans = i_pressure_trans*(Preset_max_trans-Preset_min_trans)/(Preset_num_pressure_trans-1.)+Preset_min_trans
                if(pressure_trans<=pressure1):
                    density_trans=(density0-pressure0/(gamma1-1.))*(pressure_trans/pressure0)**(1./gamma1)+pressure_trans/(gamma1-1.)
                    for i_det_density in range(Preset_num_det_density):
                        det_density_reduced=Preset_min_det_density+1.0*i_det_density*(Preset_max_det_density-Preset_min_det_density)/(Preset_num_det_density-1.)
                        for i_cs2 in range(Preset_num_cs2):
                            cs2=i_cs2*(Preset_max_cs2-Preset_min_cs2)/(Preset_num_cs2-1.)+Preset_min_cs2
                            parameter.append([pressure1,pressure2,gamma2,pressure_trans,density_trans,pressure_trans/density_trans,det_density_reduced*density_trans,det_density_reduced,cs2])
    return parameter


# =============================================================================
# Preset_pressure1=[11,10.,20.]
# Preset_det_density = [11,0.0,1.0]
# Preset_num_pressure_trans=11
# Preset_cs2=[9,1.0/3.0,1.0]
# Hybrid_sampling='low_trans_complete'
# baryon_density0 = 0.16/2.7
# baryon_density1 = 1.85*0.16
# baryon_density2 = 3.74*0.16
# baryon_density3 = 7.4*0.16 
# Preset_gamma2 = [1,2.0,2.0]
# parameter=setParameter_low_trans_complete(baryon_density0,Preset_pressure1,baryon_density1,baryon_density2,Preset_gamma2,Preset_num_pressure_trans,Preset_det_density,Preset_cs2,Hybrid_sampling)
# =============================================================================
