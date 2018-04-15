#tov integral over log-p is used to calculate mass, momentum of initial and tidal
Preset_Pressure_final = 1e-8
Preset_Pressure_final_MR = 1e-8
start_from = 0.0 #start_from=0 for calcuate all the parameter space

#tov integral over r is used to calculate radius only, since it uses build in RK45 method which cost time
TurnOn_radius_onepointfour = True
Preset_pressure_center_low = 1.0         # this parameter is only for setting boundary for searching 1.4 solar mass stars
Preset_Pressure_final_index = 1          # equal 2 means Preset_Pressure_final for solving radius is Preset_Pressure_final**2, which include more low density area
number_per_parameter = 5
concentration = 2

baryon_density0 = 0.16/2.7
baryon_density1 = 1.85*0.16
baryon_density2 = 3.74*0.16
baryon_density3 = 7.4*0.16

#EoS parameter for Hadronic star
Calculation_mode='hadronic'
Preset_pressure1=[13*1+1,7.,20.] ##[num,lower_bound,up_bound] 
Preset_pressure2=[5*5,100.] #[num,lower_bound] 
Preset_pressure3=[7*5,150.] #[num,lower_bound]

from eos_class import EOS_BPSwithPoly
eos_config=EOS_BPSwithPoly
from FindMaxmass import Maxmass
eos_Maxmass=Maxmass
from tov_f import MassRadius
eos_MassRadius=MassRadius
