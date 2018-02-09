import numpy as np
from numpy import pi
from scipy.integrate import ode
import pickle
from astropy.constants import M_sun
from scipy.constants import m_n

N_x0=1000
x0=-np.linspace(np.log(200),np.log(0.01),N_x0)
N_cs2=10
cs2=np.linspace(3,12,N_cs2)/12
Preset_rtol=1e-5
N=1000
Preset_Pressure_final=1e-7
dp=np.exp(-x0)/N
#dx=-np.log(Preset_Pressure_final)/N

def f_CSS(x, y, cs2):
    p=np.exp(-x)
    eps=(p+cs2+1)/cs2
    baryondensity=(p+1)**(1/(cs2+1))
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)-(8*pi*np.sqrt(y[1])*(eps/3.0+p))**2
        dmdx=np.sqrt(y[1])*eps*den
        dr2dx=0.5/pi*den
        dNdx=np.sqrt(y[1])*baryondensity*den
        dzdx=(4+y[3])/(eps/p/3.0+1)
        dydx=-Q*den/(4*pi)
    else:
        den=p/((y[0]+4*pi*y[1]**1.5*p)*(eps+p))
        rel=1-2*y[0]/y[1]**0.5
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)/rel-(2*p/(den*(eps+p)*y[1]*rel))**2
        dmdx=4*pi*eps*y[1]**2*rel*den
        dr2dx=2*y[1]**1.5*rel*den
        dNdx=4*pi*y[1]**2*baryondensity*np.sqrt(rel)*den
        dzdx=((4+y[3])*4*pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*np.sqrt(y[1])*den
        dydx=-(y[4]**2+(y[4]-6)/rel+y[1]*Q)*np.sqrt(y[1])*rel*den
    return np.array([dmdx,dr2dx,dNdx,dzdx,dydx])

def MassRadius_CSS(pressure_center,MRorMRBIT,eos):
    x0_=-np.log(pressure_center/eos.density_s)
    cs2_=eos.cs2
    x0_index_f=(N_x0-1)*(x0_-x0[0])/(x0[-1]-x0[0])
    x0_index=int(x0_index_f)
    x0_weight=1+x0_index-x0_index_f
    cs2_index_f=(N_cs2-1)*(12*cs2_-12*cs2[0])/(12*cs2[-1]-12*cs2[0])
    cs2_index=int(cs2_index_f)
    cs2_weight=1+cs2_index-cs2_index_f
    if(x0_index<N_x0-1):
        if(cs2_index==cs2_index_f):
            y=x0_weight*result[cs2_index][x0_index][-1]\
            +(1-x0_weight)*result[cs2_index][x0_index+1][-1]
        else:
            y=x0_weight*cs2_weight*result[cs2_index][x0_index][-1]\
            +(1-x0_weight)*cs2_weight*result[cs2_index][x0_index+1][-1]\
            +x0_weight*(1-cs2_weight)*result[cs2_index+1][x0_index][-1]\
            +(1-x0_weight)*(1-cs2_weight)*result[cs2_index+1][x0_index+1][-1]
    else:
        print('use canaonical way')
    y=y[1:6]
    M=y[0]*eos.unit_mass/M_sun.value
    R=y[1]**0.5*eos.unit_radius
    beta=y[0]/R*eos.unit_radius
    N=y[2]*eos.unit_N
    M_binding=N*m_n/M_sun.value
    momentofinertia=y[3]/(6.0+2.0*y[3])/beta**3
    yR=y[4]
    tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
    k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
    tidal=2.0/3.0*(k2/beta**5)
    if(MRorMRBIT=='MRBIT'):
        return [M,R,beta,M_binding,momentofinertia,yR,tidal]
    elif(MRorMRBIT=='MR'):
        return [M,R]

def Integration_CSS(x0_,xf_,eos):
    cs2_=eos.cs2
    x0_index_f=(N_x0-1)*(x0_-x0[0])/(x0[-1]-x0[0])
    x0_index=int(x0_index_f)
    x0_weight=1+x0_index-x0_index_f
    cs2_index_f=(N_cs2-1)*(12*cs2_-12*cs2[0])/(12*cs2[-1]-12*cs2[0])
    cs2_index=int(cs2_index_f)
    #cs2_weight=1+cs2_index-cs2_index_f
    if(0<x0_index<N_x0-1 and cs2_index==cs2_index_f):
        xf_index_f=(np.exp(-result[cs2_index][x0_index][0][0])-np.exp(-xf_))/dp[x0_index]
        xf_index=int(xf_index_f)
        if(xf_index<N-1):
            xf_weight=(result[cs2_index][x0_index][xf_index+1][0]-xf_)/(result[cs2_index][x0_index][xf_index+1][0]-result[cs2_index][x0_index][xf_index][0])
            xf1_index_f=(np.exp(-result[cs2_index][x0_index+1][0][0])-np.exp(-xf_))/dp[x0_index+1]
            xf1_index=int(xf1_index_f)
            xf1_weight=(result[cs2_index][x0_index+1][xf1_index+1][0]-xf_)/(result[cs2_index][x0_index+1][xf1_index+1][0]-result[cs2_index][x0_index+1][xf1_index][0])
            if(xf1_weight>1 or xf1_index<0):
                xf1_weight=1.
                xf1_index=0
            y=x0_weight*xf_weight*result[cs2_index][x0_index][xf_index]\
            +x0_weight*(1-xf_weight)*result[cs2_index][x0_index][xf_index+1]\
            +(1-x0_weight)*xf1_weight*result[cs2_index][x0_index+1][xf1_index]\
            +(1-x0_weight)*(1-xf1_weight)*result[cs2_index][x0_index+1][xf1_index+1]
# =============================================================================
#             xf2_index_f=(np.exp(-result[cs2_index+1][x0_index][0][0])-np.exp(-xf_))/dp[x0_index]
#             xf2_index=int(xf2_index_f)
#             xf2_weight=1+xf_index-xf_index_f
#             xf2_weight=(result[cs2_index+1][x0_index][xf_index+1][0]-xf_)/(result[cs2_index+1][x0_index][xf_index+1][0]-result[cs2_index+1][x0_index][xf_index][0])
#             xf12_index_f=(np.exp(-result[cs2_index+1][x0_index+1][0][0])-np.exp(-xf_))/dp[x0_index+1]
#             xf12_index=int(xf12_index_f)
#             xf12_weight=1+xf_index-xf1_index_f
#             xf12_weight=(result[cs2_index+1][x0_index+1][xf1_index+1][0]-xf_)/(result[cs2_index+1][x0_index+1][xf1_index+1][0]-result[cs2_index+1][x0_index+1][xf1_index][0])
#             y=x0_weight*cs2_weight*xf_weight*result[cs2_index][x0_index][xf_index]\
#             +x0_weight*cs2_weight*(1-xf_weight)*result[cs2_index][x0_index][xf_index+1]\
#             +(1-x0_weight)*cs2_weight*xf1_weight*result[cs2_index][x0_index+1][xf1_index]\
#             +(1-x0_weight)*cs2_weight*(1-xf1_weight)*result[cs2_index][x0_index+1][xf1_index+1]\
#             +x0_weight*(1-cs2_weight)*xf2_weight*result[cs2_index+1][x0_index][xf2_index]\
#             +x0_weight*(1-cs2_weight)*(1-xf2_weight)*result[cs2_index+1][x0_index][xf2_index+1]\
#             +(1-x0_weight)*(1-cs2_weight)*xf12_weight*result[cs2_index+1][x0_index+1][xf12_index]\
#             +(1-x0_weight)*(1-cs2_weight)*(1-xf12_weight)*result[cs2_index+1][x0_index+1][xf12_index+1]
# =============================================================================
        else:
            print('Warning!!! tov_CSS intepolation solver failed, use ordinary integration which decreases performance.')
            print('transition at too skin!!!')
            r = ode(f_CSS).set_integrator('lsoda',rtol=Preset_rtol)
            r.set_initial_value([0,0,0,0,2], x0_).set_f_params(cs2_)
            r.integrate(xf_)
            y=[r.t,r.y[0],r.y[1],r.y[2],r.y[3],r.y[4]]
    else:
        print('Warning!!! tov_CSS intepolation solver failed, use ordinary integration which decreases performance.')
        print('x0_=%f  is out side range [%f,%f]'%(x0_,x0[0],x0[-1]))
        r = ode(f_CSS).set_integrator('lsoda',rtol=Preset_rtol)
        r.set_initial_value([0,0,0,0,2], x0_).set_f_params(cs2_)
        r.integrate(xf_)
        y=[r.t,r.y[0],r.y[1],r.y[2],r.y[3],r.y[4]]
    return y

# =============================================================================
# def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos):
#     x0 = -np.log(pressure_center)
#     xf = x0-np.log(Preset_Pressure_final)    
#     if(MRorMRBIT=='MR'):
#         r = ode(f).set_integrator('lsoda',rtol=Preset_rtol)
#         r.set_initial_value([0,0], x0).set_f_params(a)
#         r.integrate(xf)
#         M=r.y[0]*a.unit_mass/M_sun.value
#         R=r.y[1]**0.5*a.unit_radius
#         return [M,R]
#     elif(MRorMRBIT=='MRBIT'):
#         r = ode(f_complete).set_integrator('lsoda',rtol=Preset_rtol)
#         r.set_initial_value([0,0,0,0,2], x0).set_f_params(a)
#         r.integrate(xf)
#         M=r.y[0]*a.unit_mass/M_sun.value
#         R=r.y[1]**0.5*a.unit_radius
#         beta=r.y[0]/R*a.unit_radius
#         N=r.y[2]*a.unit_N
#         M_binding=N*m_n/M_sun.value
#         momentofinertia=r.y[3]/(6.0+2.0*r.y[3])/beta**3
#         yR=r.y[4]
#         tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
#         k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
#         tidal=2.0/3.0*(k2/beta**5)
#         return [M,R,beta,M_binding,momentofinertia,yR,tidal]
# =============================================================================

if __name__ == '__main__':
    result=np.zeros([N_cs2,N_x0,N+1,6])
    r = ode(f_CSS).set_integrator('lsoda',rtol=Preset_rtol)
    for i in range(N_cs2):
        for j in range(N_x0):
            print('%.2f /100'%(100.*(i*N_x0+j)/(N_x0*N_cs2)))
            r.set_initial_value([0,0,0,0,2], x0[j]).set_f_params(cs2[i])
            result[i][j][0]=[x0[j],0,0,0,0,2]
            for k in range(N-1):
                #r.integrate(r.t+dx)
                r.integrate(-np.log(np.exp(-r.t)-dp[j]))
                result[i][j][k+1]=[r.t,r.y[0],r.y[1],r.y[2],r.y[3],r.y[4]]
            #print x0[j],Preset_Pressure_final
            r.integrate(x0[j]-np.log(Preset_Pressure_final))
            result[i][j][N]=[r.t,r.y[0],r.y[1],r.y[2],r.y[3],r.y[4]]
    f1=open('./tov_CSS_result.dat','wb')
    pickle.dump(result,f1)
    f1.close()
else:
    f2=open('./tov_CSS_result.dat','rb')
    result=pickle.load(f2)
    f2.close()