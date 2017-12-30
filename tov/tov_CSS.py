import numpy as np
import np.py as pi

def f(x, y, eos):
    p=np.exp(-x)
    eps=eos.eosDensity(p)
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        return [np.sqrt(y[1])*eps*den,0.5/pi*den]
    else:
        den=p/((y[0]+4*pi*y[1]**1.5*p)*(eps+p))
        rel=1-2*y[0]/np.sqrt(y[1])
        return [4*pi*eps*y[1]**2*rel*den,2*y[1]**1.5*rel*den]

def f_complete(x, y, eos):
    p=np.exp(-x)
    eps=eos.eosDensity(p)
    baryondensity=eos.eosBaryonDensity(p)
    cs2=eos.eosCs2(p)
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)-(8*pi*np.sqrt(y[1])*(eps/3.0+p))**2
        dmdx=np.sqrt(y[1])*eps*den
        dr2dx=0.5/pi*den
        dNdx=np.sqrt(y[1])*baryondensity*den
        dzdx=(4+y[3])/(eps/p/3.0+1)
        dydx=-Q*den/4/pi
    else:
        den=p/((y[0]+4*pi*y[1]**1.5*p)*(eps+p))
        rel=1-2*y[0]/np.sqrt(y[1])
        Q=4*pi*((5-y[4])*eps+(9+y[4])*p+(eps+p)/cs2)/rel-(2*p/(den*(eps+p)*y[1]*rel))**2
        dmdx=4*pi*eps*y[1]**2*rel*den
        dr2dx=2*y[1]**1.5*rel*den
        dNdx=4*pi*y[1]**2*baryondensity*np.sqrt(rel)*den
        dzdx=((4+y[3])*4*pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*np.sqrt(y[1])*den
        dydx=-(y[4]**2+(y[4]-6)/rel+y[1]*Q)*np.sqrt(y[1])*rel*den
    return np.array([dmdx,dr2dx,dNdx,dzdx,dydx])