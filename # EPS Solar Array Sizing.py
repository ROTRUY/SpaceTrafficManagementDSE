# EPS Solar Array Sizing
import math
import numpy as np


P_sun = 1367  #Solar flux at 1 au in W/m^2
t_mis = 6  #Mission duration in months

#Power Requirements

P_sc_day = 3.5  #Power requirement of the SC during day in W
P_sc_ecl = 3.5  #Power requirement of the SC during eclipse in W

P_pl = 0  #Power requirement of the PL in W
t_pl = 12  #Measurement duration of the PL in ms
f_pl = 10  #Measurement frequency of the PL in min (i.e. one snapshot every x minutes)

P_day_peak = P_sc_day + P_pl  #Peak power during day in W
P_ecl_peak = P_sc_ecl + P_pl  #Peak power during eclipse in W

P_day_avg = P_sc_day + P_pl * ((t_pl / 100) / (f_pl * 60))  #Average power during day in W
P_ecl_avg = P_sc_ecl + P_pl * ((t_pl / 100) / (f_pl * 60))  #Average power during day in W

#Orbit Values 

t_day = 100  #Day time in minutes
t_ecl = 0  #Eclipse time in minutes

theta = 0  #Incidence angle of the solar arrays (worst case)

#Solar Cell Properties

eta_SC = 0.32  #Efficiency of the solar cell
degradation_rate = 0.5  #Degradation/year of the solar cell in %
P_sp_opt = 125  #Specific power of the solar cell in W/kg at 0 degree incidence
I_d = 0.77  #Inherent degradation of solar cells

#Power Regulation Values

eta_PPT_day = 0.8  #Day efficiency for Peak Power Tracking
eta_PPT_ecl = 0.6  #Eclipse efficiency for Peak Power Tracking

eta_DET_day = 0.85  #Day efficiency for Direct Energy Transfer
eta_DET_ecl = 0.65  #Eclipse efficiency for Direct Energy Transfer

power_regulation_type = 1  #1=PPT, 2=DET

# ----------------------------------------------------------------------

if power_regulation_type == 1:
    eta_day = eta_PPT_day
    eta_ecl = eta_PPT_ecl

    P_d = P_day_avg
    P_e = P_ecl_avg

elif power_regulation_type == 2:
    eta_day = eta_DET_day
    eta_ecl = eta_DET_ecl

    P_d = P_day_peak
    P_e = P_ecl_peak


P_req = (P_d * t_day / eta_day + P_e * t_ecl / eta_ecl) / t_day  #Total power that the EPS should deliver

P_delta_opt = P_sun * eta_SC

P_BOL_delta = P_delta_opt * math.cos(math.radians(theta)) * I_d  #Power delivered by a m^2 of a solar cell at BOL
P_BOL_sp = P_sp_opt * math.cos(math.radians(theta)) * I_d  #Power delivered by a g of a solar cell at BOL

L_d = (1 - degradation_rate/100)**(t_mis / 12)

P_EOL_delta = P_BOL_delta * L_d
P_EOL_sp = P_BOL_sp * L_d

A_SA = P_req / P_EOL_delta
M_SA = P_req / P_EOL_sp

print("To generate " + str(P_req) + " W, a solar array of "  + str(A_SA * 10**4) + " cm^2 which weighs " + str(M_SA) + " kg is required")