import pandas as pd
from Read_GMAT_Files import *
import numpy as np
import matplotlib.pyplot as plt

##################### Variable Initialization ########################################

ADCS_power_nominal = 0.0        # ADCS nominal consumption (W) when NOT in ground pass
CDH_power_nominal = 0.3        # C&DH nominal consumption outside ground pass (W)
CDH_power_ground = 6.0          # C&DH consumption during a ground pass (W)
verification_power = 0.125      # Verification‐payload nominal consumption (W)
payload_power = 0.032
solar_generation_power = 1
Battery_size = 5 #Watt hours
Battery_Capacity = Battery_size * 3600 # Joules

Constant_Power = CDH_power_nominal + verification_power + payload_power

####### Function to go from a time array to check and a value array to a boolean array ######

def getQueryTimeBooleanArray(query_array, check_array):
    starts = check_array['Start Time (UTC)'].values
    ends   = check_array['Stop Time (UTC)'].values
    indexArray = np.searchsorted(starts, query_array, side='right') - 1

    in_check_array = np.zeros(len(query_times), dtype=bool)
    valid = indexArray >= 0
    in_check_array[valid] = ends[indexArray[valid]] >= query_array[valid]
    
    return in_check_array

##################### Dataset Initialization ########################################

e_data = Get_Eclipse_Data("GSCData-500km-21mar/EclipseData500.txt")
gp_data = Get_GroundPass_Data("GSCData-500km-21mar/delft60.txt")
query_times = pd.date_range('2029-03-21 0:00:00', periods=2678000, freq='s').values


########################### Find array how much the power is charging or draining ###############
inEclipseArray = getQueryTimeBooleanArray(query_array=query_times,check_array=e_data)
OverGroundStationArray = getQueryTimeBooleanArray(query_array=query_times,check_array=gp_data)

EPSChargingArray = np.where(inEclipseArray, 0, solar_generation_power)
GroundStationDischargeArray = np.where(OverGroundStationArray, -CDH_power_ground, 0)

Power_Total = -Constant_Power + EPSChargingArray + GroundStationDischargeArray

############################## Find Array with battery capacity #################################

#add battery size in joules. Because the timestep is one second, the wattages is the same as the energy per timestep
#Use np.clip to make sure the battery charge stays between 0 and max
Battery_Charge_Raw = Battery_Capacity + np.cumsum(Power_Total)
overshoot = np.maximum.accumulate(np.maximum(Battery_Charge_Raw - Battery_Capacity, 0.0))

Battery_Charge_No_Overshoot = Battery_Charge_Raw - overshoot

undershoot = np.minimum.accumulate(np.minimum(Battery_Charge_No_Overshoot, 0.0))

Battery_Charge_Final = Battery_Charge_No_Overshoot - undershoot

Battery_Charge_Data = pd.DataFrame({
    'Query_Times':query_times,
    'Charge_Rate':Power_Total,
    'Battery_Charge':Battery_Charge_Final
})

######################### Find lowest value, and plot around that time ##########################

Battery_Charge_Data = Battery_Charge_Data.set_index('Query_Times')
worst_charge_time = Battery_Charge_Data['Battery_Charge'].idxmin()

window_start = worst_charge_time - pd.Timedelta(seconds=70000)
window_end   = worst_charge_time + pd.Timedelta(seconds=40000)

df_window = Battery_Charge_Data.loc[window_start : window_end]

plt.figure(figsize=(10,4))
plt.plot(df_window.index, df_window['Battery_Charge'], lw=1)
plt.axvline(worst_charge_time, color='r', ls='--', label=f"min at {worst_charge_time.strftime('%Y-%m-%d %H:%M:%S')}")
plt.xlabel("Time")
plt.ylabel("Battery_Charge (J)")
plt.ylim((0.0, Battery_Capacity * 1.1))
plt.title(f"Power around lowest‐power point ({worst_charge_time})")
plt.legend()
plt.tight_layout()
plt.show()