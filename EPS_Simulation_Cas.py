import pandas as pd
from Read_GMAT_Files import *
import time
import numpy as np

P_sun = 1322  # Solar flux at 1 AU in W/m^2
t_mis = 6     # Mission duration in months

ADCS_power_nominal = 0.0        # ADCS nominal consumption (W) when NOT in ground pass
CDH_power_nominal = 0.25        # C&DH nominal consumption outside ground pass (W)
CDH_power_ground = 5.0          # C&DH consumption during a ground pass (W)
verification_power = 0.125      # Verification‐payload nominal consumption (W)
payload_power = 0.032

# def IsInGroundPass_simple(time_to_check, groundpass_data):
#     mask = (groundpass_data['Start Time (UTC)'] <= time_to_check) & (groundpass_data['Stop Time (UTC)'] >= time_to_check)
#     return mask.any()

# def IsInEclipse_simple(time_to_check, eclipse_data):
#     mask = (eclipse_data['Start Time (UTC)'] <= time_to_check) & (eclipse_data['Stop Time (UTC)'] >= time_to_check)
#     return mask.any()

# start = time.time()
# for i in range(1000):
#     ttc= pd.to_datetime('2029-03-21 01:47:45.682').tz_localize('UTC')
#     void = IsInEclipse_simple(time_to_check=ttc, eclipse_data=e_data)
#     void = IsInGroundPass_simple(time_to_check=ttc, groundpass_data=e_data)

#     ttc= pd.to_datetime('2029-02-21 01:47:45.682').tz_localize('UTC')
#     void = IsInEclipse_simple(time_to_check=ttc, eclipse_data=e_data)
#     void = IsInGroundPass_simple(time_to_check=ttc, groundpass_data=e_data)

# end = time.time()
# elapsed = end - start
# print(f'Time taken: {elapsed:.6f} seconds')

e_data = Get_Eclipse_Data("GSCData-500km-21mar/EclipseData500.txt")
gp_data = Get_GroundPass_Data("GSCData-500km-21mar/delft60.txt")

# print(IsInEclipse(time_to_check=ttc, eclipse_data=gp_data))

def getQueryTimeBooleanArray(query_array, check_array):
    starts = check_array['Start Time (UTC)'].values
    ends   = check_array['Stop Time (UTC)'].values
    indexArray = np.searchsorted(starts, query_array, side='right') - 1

    in_check_array = np.zeros(len(query_times), dtype=bool)
    valid = indexArray >= 0
    in_check_array[valid] = ends[indexArray[valid]] >= query_array[valid]
    
    return in_check_array

query_times = pd.date_range('2029-03-21 0:00:00', periods=2678000, freq='s').values

print(np.count_nonzero(getQueryTimeBooleanArray(query_array=query_times,check_array=e_data)))
print(np.count_nonzero(getQueryTimeBooleanArray(query_array=query_times,check_array=gp_data)))
print(query_times[-1])