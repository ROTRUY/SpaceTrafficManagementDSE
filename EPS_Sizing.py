import math
import numpy as np


#--------------------------------------------------------------------
# SOLAR ARRAY SIZING
#--------------------------------------------------------------------


P_sun = 1367  #Solar flux at 1 au in W/m^2
t_mis = 6  #Mission duration in months

#Orbit Values 

t_day = 56.364  #Day time in minutes
t_ecl = 36.036  #Eclipse time in minutes
t_orb = t_day + t_ecl  #Orbit time in minutes

theta = 0  #Incidence angle of the solar arrays 

#Power Requirements

P_day_avg = 1.04525
P_ecl_avg = 1.04525

#Solar Cell Properties

eta_SC = 0.30  #Efficiency of the solar cell
degradation_rate = 0.5  #Degradation/year of the solar cell in %
P_sp_opt = 70  #Power generate by 1 kg solar array at optimal conditions & BOL in W
I_d = 0.72  #Inherent degradation of solar cells

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

    P_d = 0
    P_e = 0


P_req = (P_d * t_day / eta_day + P_e * t_ecl / eta_ecl) / t_day  #Total power that the EPS should deliver

P_delta_opt = P_sun * eta_SC

P_BOL_delta = P_delta_opt * math.cos(math.radians(theta)) * I_d  #Power delivered by a m^2 of a solar cell at BOL
P_BOL_sp = P_sp_opt * math.cos(math.radians(theta)) * I_d  #Power delivered by 1 kg of a solar cell at BOL

L_d = (1 - degradation_rate/100)**(t_mis / 12)

P_EOL_delta = P_BOL_delta * L_d  #Power delivered by a m^2 of a solar cell at EOL
P_EOL_sp = P_BOL_sp * L_d  #Power delivered by 1 kg of a solar cell at EOL

A_SA = P_req / P_EOL_delta  #Area of the solar arrays in m^2.
M_SA = P_req / P_EOL_sp  #Mass of the solar arrays in kg.

print("To generate " + str(P_req) + " W, a solar array of "  + str(A_SA * 10**4) + " cm^2 which weighs " + str(M_SA) + " kg is required")


#--------------------------------------------------------------------
# BATTERY SIZING
#--------------------------------------------------------------------


#Battery Data

eta_bat = 0.90  #Total efficiency of the battery and the discharge electronics
DOD = 0.6  #Depth of discharge of the battery (Obtained from the graphs)

E_delta_bat = 321  #Specific power of the battery in Wh/L
E_sp_bat = 133  #Specific power of the battery in Wh/kg

#Battery Sizing

E_bat = (P_e * (t_ecl / 60)) / (DOD * eta_bat)  #Energy stored in battery in Wh.

V_bat = E_bat / E_delta_bat  #Volume of the battery in L.
M_bat = E_bat / E_sp_bat  #Mass of the battery in kg.

print("To store " + str(E_bat) + " Wh, a battery of "  + str(V_bat * 10**3) + " cm^3 which weighs " + str(M_bat) + " kg is required")


#--------------------------------------------------------------------
# POWER CONTROL & DISTRIBUTION UNIT SIZING
#--------------------------------------------------------------------

#Power control & distribution unit is selected from commercially available PCU, PCD, and PCDUs (current pick: )

M_PCDU = 0.148  #Mass of the PCDU in kg
V_PCDU = 180.02  #Volume of the battery in cm^3 (95.89mm x 90.17mm x 20.82mm) - ACC Clyde NanoPlus PCDU

print("The PCDU is able to handle " + str(10) + " W, weighing "  + str(M_PCDU) + " kg and having a volume of " + str(V_PCDU) + " cm^3")


#--------------------------------------------------------------------
# EPS
#--------------------------------------------------------------------

EPS_mass = M_SA + M_bat + M_PCDU

print("The total mass of the EPS is " +str(EPS_mass) + " kg")