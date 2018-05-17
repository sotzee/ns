

from eos_class import EOS_BPSwithPolyCSS,EOS_CSS
from tov_f import MassRadius_transition
from tov_CSS import MassRadius_CSS
import numpy as np
from fractions import Fraction

import matplotlib.pyplot as plt
f, ((ax1, ax2)) = plt.subplots(1,2, sharex=False,figsize=(10, 5))

a=EOS_BPSwithPolyCSS([0.059259259259259255, 20.0, 0.29600000000000004, 81.73907962016068, 0.5984, 5000.0, 1.1840000000000002, 3., 0, Fraction(1, 4)])
N=500
pressure=np.linspace(1,200,N)
density=a.eosDensity(pressure)
chempo=a.eosChempo(pressure)
baryondensity=a.eosBaryonDensity(pressure)
cs2=a.eosCs2(pressure)

pressure_center = np.linspace(10,1000,N)
mass_radius=[]
for i in range(N):
    mass_radius.append ( MassRadius_transition(pressure_center[i],1e-7,1e-4,'MR',a))
mass_radius=np.array(mass_radius).transpose()
ax1.plot(pressure,density)
ax1.set_xlabel('pressure (MeV fm$^{-3}$)')
ax1.set_ylabel('density (MeV fm$^{-3}$)')
ax2.plot(mass_radius[1]/1000,mass_radius[0])
ax2.set_xlabel('Radius(km)')
ax2.set_ylabel('$M/M_{\odot}')



# =============================================================================
# a=EOS_CSS([627.675282406, 0.0, 0.16, Fraction(3, 3)])
# N=500
# pressure=np.linspace(1,200,N)
# density=a.eosDensity(pressure)
# chempo=a.eosChempo(pressure)
# baryondensity=a.eosBaryonDensity(pressure)
# cs2=a.eosCs2(pressure)
# 
# pressure_center = np.linspace(20,1000,N)
# mass_radius=[]
# for i in range(N):
#     mass_radius.append ( MassRadius_CSS(pressure_center[i],'MR',a))
# mass_radius=np.array(mass_radius).transpose()
# ax1.plot(pressure,density)
# ax1.set_xlabel('pressure (MeV fm$^{-3}$)')
# ax1.set_ylabel('density (MeV fm$^{-3}$)')
# ax2.plot(mass_radius[1]/1000,mass_radius[0])
# ax2.set_xlabel('Radius(km)')
# ax2.set_ylabel('$M/M_{\odot}')
# =============================================================================
