import math
import numpy as np


#--------------------------------------------------------------------
# SOLAR ARRAY SIZING
#--------------------------------------------------------------------


P_sun = 1367  #Solar flux at 1 au in W/m^2
t_mis = 6  #Mission duration in months

#Orbit Values 

t_day = 60  #Day time in minutes
t_ecl = 36  #Eclipse time in minutes
t_orb = t_day + t_ecl  #Orbit time in minutes

theta = 0  #Incidence angle of the solar arrays 

#Power Requirements

P_sc_day = 5  #Power requirement of the SC during day in W
P_sc_ecl = 5  #Power requirement of the SC during eclipse in W

P_pl_snap = 0.1  #Extra power requirement due to snapshot in W
t_pl_snap = 1  #Measurement duration of the PL in s
f_pl_snap = 10  #Measurement frequency of the PL in min (i.e. one snapshot every x minutes)

P_pl_eph = 0.3  #Extra power requirement due to ephemeris data gathering in W
t_pl_eph = 30  #Measurement duration of the ephemeris data in s
f_pl_eph = 30  #Ephemeris receiving frequency of the PL in min (i.e. one ephemeris capturing every x hr)

P_pl_alm = 0.5  #Extra power requirement due to almanac data gathering in W
t_pl_alm = 15  #Measurement duration of the almanac data in min

P_pl_max = np.max([P_pl_snap, P_pl_eph, P_pl_alm])  #Peak power need due to PL

P_day_peak = P_sc_day + P_pl_max  #Peak power during day in W
P_ecl_peak = P_sc_ecl + P_pl_max  #Peak power during eclipse in W

P_day_avg = P_sc_day + P_pl_snap * (t_pl_snap / (f_pl_snap * 60)) + P_pl_eph * (t_pl_eph / (f_pl_eph * 60)) + P_pl_alm * (t_pl_alm / t_orb)  #Average power during day in W
P_ecl_avg = P_sc_ecl + P_pl_snap * (t_pl_snap / (f_pl_snap * 60)) + P_pl_eph * (t_pl_eph / (f_pl_eph * 60)) + P_pl_alm * (t_pl_alm / t_orb)  #Average power during eclipse in W

#Solar Cell Properties

eta_SC = 0.30  #Efficiency of the solar cell
degradation_rate = 0.5  #Degradation/year of the solar cell in %
P_sp_EOL = 125  #Specific power of the solar cell in W/kg at EOL in normal operation conditions
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

    P_d = P_day_peak
    P_e = P_ecl_peak


P_req = (P_d * t_day / eta_day + P_e * t_ecl / eta_ecl) / t_day  #Total power that the EPS should deliver

P_delta_opt = P_sun * eta_SC

P_BOL_delta = P_delta_opt * math.cos(math.radians(theta)) * I_d  #Power delivered by a m^2 of a solar cell at BOL

L_d = (1 - degradation_rate/100)**(t_mis / 12)

P_EOL_delta = P_BOL_delta * L_d

A_SA = P_req / P_EOL_delta  #Area of the solar arrays in m^2.
M_SA = P_req / P_sp_EOL  #Mass of the solar arrays in kg.

print("To generate " + str(P_req) + " W, a solar array of "  + str(A_SA * 10**4) + " cm^2 which weighs " + str(M_SA) + " kg is required")


#--------------------------------------------------------------------
# BATTERY SIZING
#--------------------------------------------------------------------


#Battery Data

eta_bat = 0.9  #Total efficiency of the battery and the discharge electronics
DOD = 0.7  #Depth of discharge of the battery (Obtained from the graphs)

E_delta_bat = 71  #Specific power of the battery in Wh/L
E_sp_bat = 48  #Specific power of the battery in Wh/kg

#Battery Sizing

E_bat = (P_e * (t_ecl / 60)) / (DOD * eta_bat)  #Energy stored in battery in Wh.

V_bat = E_bat / E_delta_bat  #Volume of the battery in L.
M_bat = E_bat / E_sp_bat  #Mass of the battery in kg.

print("To store " + str(E_bat) + " Wh, a battery of "  + str(V_bat * 10**3) + " cm^3 which weighs " + str(M_bat) + " kg is required")


#--------------------------------------------------------------------
# POWER CONTROL & DISTRIBUTION UNIT SIZING
#--------------------------------------------------------------------

#Power control & distribution unit is selected from commercially available PCU, PCD, and PCDUs (current pick: )

M_PCDU = 0.086  #Mass of the PCDU in kg
V_PCDU = 140.1  #Volume of the battery in cm^3 (95.89mm x 90.17mm x 1.62mm)

print("The PCDU is able to handle " + str(P_day_peak) + " W, weighing "  + str(M_PCDU) + " kg and having a volume of " + str(V_PCDU) + " cm^3")


#--------------------------------------------------------------------
# EPS
#--------------------------------------------------------------------

EPS_mass = M_SA + M_bat + M_PCDU

print("The total mass of the EPS is " +str(EPS_mass) + " kg")