from scipy.interpolate import interp1d
import numpy as np
from unitconvert import toMevfm
from scipy.misc import derivative
from scipy.constants import c,G,e

class EOS_item(object):
    def __init__(self,para):
        self.args=para
        self.stars=[]
    def set_properity(self,properity):
        self.properity=properity
    def add_star(self,star_properity):
        self.stars.append(star_properity)
    
class EOS_BPS(object):
    #density in units: g/cm3
    densityBPS = np.array([0.0,7.861E0,  7.900E0,  8.150E0,  1.160E1,  1.640E1,  4.510E1,  2.120E2,  1.150E3,  1.044E4,  2.622E4,  6.587E4,  1.654E5,  4.156E5,  1.044E6,  2.622E6,  6.588E6,  8.294E6,  1.655E7,  3.302E7,  6.590E7,  1.315E8,  2.624E8,  3.304E8,  5.237E8,  8.301E8,  1.045E9,  1.316E9,  1.657E9,  2.626E9,  4.164E9,  6.602E9,  8.313E9,  1.046E10,  1.318E10,  1.659E10,  2.090E10,  2.631E10,  3.313E10,  4.172E10,  5.254E10,  6.617E10,  8.333E10,  1.049E11,  1.322E11,  1.664E11,  1.844E11,  2.096E11,  2.640E11,  3.325E11,  4.188E11,  4.299E11,  4.460E11,  5.228E11,  6.610E11,  7.964E11,  9.728E11,  1.196E12,  1.471E12,  1.805E12,  2.202E12,  2.930E12,  3.833E12,  4.933E12,  6.248E12,  7.801E12,  9.612E12,  1.246E13,  1.496E13, 1.778E13, 2.210E13, 2.988E13, 3.767E13, 5.081E13, 6.193E13, 7.732E13, 9.826E13, 1.262E14])
    #pressure in units: g*cm/s
    pressureBPS = np.array([0.0, 1.010E9,  1.010E10,  1.010E11,  1.210E12,  1.400E13,  1.700E14,  5.820E15,  1.900E17,  9.744E18,  4.968E19,  2.431E20,  1.151E21,  5.266E21,  2.318E22,  9.755E22,  3.911E23,  5.259E23,  1.435E24,  3.833E24,  1.006E25,  2.604E25,  6.676E25,  8.738E25,  1.629E26,  3.029E26,  4.129E26,  5.036E26,  6.860E26,  1.272E27,  2.356E27,  4.362E27,  5.662E27,  7.702E27,  1.048E28,  1.425E28,  1.938E28,  2.503E28,  3.404E28,  4.628E28,  5.949E28,  8.089E28,  1.100E29,  1.495E29,  2.033E29,  2.597E29,  2.892E29,  3.290E29,  4.473E29,  5.816E29,  7.538E29,  7.805E29,  7.890E29,  8.352E29,  9.098E29,  9.831E29,  1.083E30,  1.218E30,  1.399E30,  1.638E30,  1.950E30,  2.592E30,  3.506E30,  4.771E30,  6.481E30,  8.748E30,  1.170E31, 1.695E31, 2.209E31, 2.848E31, 3.931E31, 6.178E31, 8.774E31, 1.386E32, 1.882E32, 2.662E32, 3.897E32, 5.861E32])
    #baryon density in units: 1/cm3
    baryondensityBPS = np.array([0.0, 4.73E24, 4.76E24, 4.91E24, 6.990E24,  9.900E24,  2.720E25,  1.270E26,  6.930E26,  6.295E27,  1.581E28,  3.972E28,  9.976E28,  2.506E29,  6.294E29,  1.581E30,  3.972E30,  5.000E30,  9.976E30,  1.990E31,  3.972E31,  7.924E31,  1.581E32,  1.990E32,  3.155E32,  5.000E32,  6.294E32,  7.924E32,  9.976E32,  1.581E33,  2.506E33,  3.972E33,  5.000E33,  6.294E33,  7.924E33,  9.976E33,  1.256E34,  1.581E34,  1.990E34,  2.506E34,  3.155E34,  3.972E34,  5.000E34,  6.294E34,  7.924E34,  9.976E34,  1.105E35,  1.256E35,  1.581E35,  1.990E35,  2.506E35,  2.572E35,  2.670E35,  3.126E35,  3.951E35,  4.759E35,  5.812E35,  7.143E35,  8.786E35,  1.077E36,  1.314E36,  1.748E36,  2.287E36,  2.942E36,  3.726E36,  4.650E36,  5.728E36, 7.424E36, 8.907E36, 1.059E37, 1.315E37, 1.777E37, 2.239E37, 3.017E37, 3.675E37, 4.585E37, 5.821E37, 7.468E37])
    #density in units: Mevfm3
    densityBPS = toMevfm(densityBPS,'density')
    #pressure in units: Mevfm3
    pressureBPS = toMevfm(pressureBPS,'pressure')
    #baryon density in units: 1/fm3
    baryondensityBPS = toMevfm(baryondensityBPS,'baryondensity')
    eosPressure_frombaryon = interp1d(baryondensityBPS,pressureBPS, kind='linear')
    eosPressure = interp1d(densityBPS,pressureBPS, kind='linear')
    eosDensity  = interp1d(pressureBPS,densityBPS, kind='linear')
    eosBaryonDensity = interp1d(pressureBPS,baryondensityBPS, kind='linear')
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_PiecewisePoly(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryon_density0,self.pressure1\
        ,self.baryon_density1,self.pressure2,self.baryon_density2\
        ,self.pressure3,self.baryon_density3 = args
        self.gamma1=np.log(self.pressure1/self.pressure0)\
        /np.log(self.baryon_density1/self.baryon_density0)
        self.gamma2=np.log(self.pressure2/self.pressure1)\
        /np.log(self.baryon_density2/self.baryon_density1)
        self.gamma3=np.log(self.pressure3/self.pressure2)\
        /np.log(self.baryon_density3/self.baryon_density2)
        self.density1=(self.density0-self.pressure0/(self.gamma1-1))\
        *(self.pressure1/self.pressure0)**(1/self.gamma1)\
        +self.pressure1/(self.gamma1-1)
        self.density2=(self.density1-self.pressure1/(self.gamma2-1))\
        *(self.pressure2/self.pressure1)**(1/self.gamma2)\
        +self.pressure2/(self.gamma2-1)
        self.baryon_density_s=0.16
        self.pressure_s=self.pressure0*(self.baryon_density_s/self.baryon_density0)**self.gamma1
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosDensity(self,pressure):
        if(pressure<self.pressure1):
            density = ((self.density0-self.pressure0/(self.gamma1-1))\
                       *(pressure/self.pressure0)**(1/self.gamma1)\
                       +pressure/(self.gamma1-1))
        elif(pressure<self.pressure2):
            density = ((self.density1-self.pressure1/(self.gamma2-1))\
                       *(pressure/self.pressure1)**(1/self.gamma2)\
                       +pressure/(self.gamma2-1))
        else:
            density = ((self.density2-self.pressure2/(self.gamma3-1))\
                       *(pressure/self.pressure2)**(1/self.gamma3)\
                       +pressure/(self.gamma3-1))
        return density
    def eosBaryonDensity(self,pressure):
        if(pressure<self.pressure1):
            density = self.baryon_density0*(pressure/self.pressure0)\
            **(1.0/self.gamma1)
        elif(pressure<self.pressure2):
            density = self.baryon_density1*(pressure/self.pressure1)\
            **(1.0/self.gamma2)
        else:
            density = self.baryon_density2*(pressure/self.pressure2)\
            **(1.0/self.gamma3)
        return density
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=1e-10)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_BPSwithPoly(EOS_BPS):
    def __init__(self,args):
        self.baryon_density0,self.pressure1,self.baryon_density1\
        ,self.pressure2,self.baryon_density2,self.pressure3\
        ,self.baryon_density3=args
        self.pressure0=EOS_BPS.eosPressure_frombaryon(self.baryon_density0)
        self.density0=EOS_BPS.eosDensity(self.pressure0)
        args_eosPiecewisePoly=[self.density0,self.pressure0\
                                           ,self.baryon_density0,self.pressure1\
                                           ,self.baryon_density1,self.pressure2\
                                           ,self.baryon_density2,self.pressure3\
                                           ,self.baryon_density3]
        self.eosPiecewisePoly=EOS_PiecewisePoly(args_eosPiecewisePoly)
        self.baryon_density_s=self.eosPiecewisePoly.baryon_density_s
        self.pressure_s=self.eosPiecewisePoly.pressure_s
        self.density_s=self.eosPiecewisePoly.density_s
        self.unit_mass=self.eosPiecewisePoly.unit_mass
        self.unit_radius=self.eosPiecewisePoly.unit_radius
        self.unit_N=self.eosPiecewisePoly.unit_N
    def eosDensity(self,pressure):
        if(pressure<self.pressure0):
            return EOS_BPS.eosDensity(pressure)
        else:
            return self.eosPiecewisePoly.eosDensity(pressure)
    def eosBaryonDensity(self,pressure):
        if(pressure<self.pressure0):
            return EOS_BPS.eosBaryonDensity(pressure)
        else:
            return self.eosPiecewisePoly.eosBaryonDensity(pressure)
    def eosCs2(self,pressure): #it is a step function at BPS region, since I use Linear intepolation
        return 1.0/derivative(self.eosDensity,pressure,dx=1e-10)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_CSS(object):
    def __init__(self,args):
        self.density0,self.pressure0,self.baryondensity_trans,self.cs2 = args
        self.B=(self.density0-self.pressure0/self.cs2)/(1.0+1.0/self.cs2)
        if(self.B>0):
            self.baryon_density_s=self.baryondensity_trans/(self.pressure0/self.B+1)**(1/(self.cs2+1))
            self.pressure_s=self.B
            self.density_s=self.B
        else:
            self.baryon_density_s=0.16
            self.pressure_s=-self.B
            self.density_s=-self.B
            print('Warning!!! ESS equation get negative Bag constant!!!')
            print('args=%s'%(args))
            print('B=%f MeVfm-3'%(self.B))
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosDensity(self,pressure):
        return np.max((pressure-self.pressure0)/self.cs2+self.density0,0)
    def eosBaryonDensity(self,pressure):
        return np.max(self.baryondensity_trans\
                       *((pressure+self.B)/(self.pressure0+self.B))\
                       **(1.0/(1.0+self.cs2)),0)
    def eosCs2(self,pressure):
        return self.cs2
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

class EOS_BPSwithPolyCSS(EOS_BPSwithPoly,EOS_CSS):
    def __init__(self,args):
        self.baryon_density0,self.pressure1,self.baryon_density1\
        ,self.pressure2,self.baryon_density2,self.pressure3\
        ,self.baryon_density3,self.pressure_trans,self.det_density\
        ,self.cs2=args
        self.args=args
        self.eosBPSwithPoly=EOS_BPSwithPoly(args[0:7])
        self.baryon_density_s=self.eosBPSwithPoly.baryon_density_s
        self.pressure_s=self.eosBPSwithPoly.pressure_s
        self.density_s=self.eosBPSwithPoly.density_s
        self.unit_mass=self.eosBPSwithPoly.unit_mass
        self.unit_radius=self.eosBPSwithPoly.unit_radius
        self.unit_N=self.eosBPSwithPoly.unit_N
        self.density_trans=self.eosBPSwithPoly.eosDensity(self.pressure_trans)
        self.baryondensity_trans=self.eosBPSwithPoly.eosBaryonDensity\
        (self.pressure_trans)/(self.density_trans+self.pressure_trans)\
        *(self.density_trans+self.pressure_trans+self.det_density)
        args_eosCSS=[self.density_trans+self.det_density,self.pressure_trans\
                     ,self.baryondensity_trans,self.cs2]
        self.eosCSS=EOS_CSS(args_eosCSS)
    def eosDensity(self,pressure):
        if(pressure<self.pressure_trans):
            return self.eosBPSwithPoly.eosDensity(pressure)
        else:
            return self.eosCSS.eosDensity(pressure)
    def eosBaryonDensity(self,pressure):
        if(pressure<self.pressure_trans):
            return self.eosBPSwithPoly.eosBaryonDensity(pressure)
        else:
            return self.eosCSS.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        if(pressure<self.pressure_trans):
            return self.eosBPSwithPoly.eosCs2(pressure)
        else:
            return self.cs2
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)

# =============================================================================
# baryon_density0=0.16/2.7
# baryon_density1=1.85*0.16
# baryon_density2=3.74*0.16
# baryon_density3=7.4*0.16
# pressure1=10.0
# pressure2=150.
# pressure3=1000.
# pressure_trans=120
# det_density=100
# cs2=1.0
# args=[baryon_density0,pressure1,baryon_density1,pressure2,baryon_density2,pressure3,baryon_density3,pressure_trans,det_density,cs2]
# a=EOS_BPSwithPolyCSS(args)
# =============================================================================

class EOS_MIT(EOS_BPS):
    def __init__(self,args):
        self.ms,self.bag,self.delta,self.a4=args
        self.baryon_density_s=0.16
        self.pressure_s=self.bag
        self.density_s=self.bag
        self.pressureMIT,self.densityMIT,self.baryondensityMIT\
        =self.eos_for_intepolation(np.linspace(1,2000,1000))
        self.eosPressure = interp1d(self.densityMIT,self.pressureMIT, kind='linear')
        self.eosDensity  = interp1d(self.pressureMIT,self.densityMIT, kind='linear')
        self.eosBaryonDensity = interp1d(self.pressureMIT,self.baryondensityMIT, kind='linear')
    def Omega(self,chempo):
        return -3./4/np.pi**2*self.a4*chempo**4+3.*self.ms**2*chempo**2/4\
    /np.pi**2-3.*self.delta**2*chempo**2/np.pi**2\
    +(12.*np.log(self.ms/2/chempo)-1)*self.ms**4/32/np.pi**2+self.bag
    def dOmega_dchempo(self,chempo):
        return -3./np.pi**2*self.a4*chempo**3+3.*self.ms**2*chempo/2/np.pi**2\
    -6.*self.delta**2*chempo/np.pi**2-3.*self.ms**4/8/np.pi**4/chempo
    def eos_for_intepolation(self,chempo):
        pressure=toMevfm(-self.Omega(chempo),'mev4')
        baryondensity=toMevfm(-self.dOmega_dchempo(chempo),'mev4')
        energydensity=chempo*baryondensity-pressure
        return pressure,energydensity,baryondensity
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=1e-10)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
# =============================================================================
# a=EOS_MIT([100.,133**4,0,0.61])
# =============================================================================

def show_eos(eos,pressure):
    N=np.size(pressure)
    density=np.linspace(0,200,N)
    chempo=np.linspace(0,200,N)
    baryondensity=np.linspace(0,200,N)
    cs2=np.linspace(0,200,N)
    for i in range(N):
        density[i]=eos.eosDensity(pressure[i])/eos.density_s
        chempo[i]=eos.eosChempo(pressure[i])*eos.baryon_density_s/(eos.density_s+eos.pressure_s)
        baryondensity[i]=eos.eosBaryonDensity(pressure[i])/eos.baryon_density_s
        cs2[i]=eos.eosCs2(pressure[i])
    import matplotlib.pyplot as plt
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True,figsize=(10, 10))
    ax1.plot(pressure,density)
    ax1.set_ylabel('density(MeV fm$^{-3}$)')
    ax2.plot(pressure,chempo)
    ax2.set_ylabel('chemical potential(MeV)')
    ax3.plot(pressure,baryondensity)
    ax3.set_xlabel('pressure(MeV fm$^{-3}$)')
    ax3.set_ylabel('baryon density(fm$^{-3}$)')
    ax4.plot(pressure,cs2)
    ax4.set_xlabel('pressure(MeV fm$^{-3}$)')
    ax4.set_ylabel('sound speed square')
    