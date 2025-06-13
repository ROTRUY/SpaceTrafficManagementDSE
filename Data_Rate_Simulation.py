import pandas as pd
from Read_GMAT_Files import *
import numpy as np
import matplotlib.pyplot as plt

##################### Variable Initialization ########################################

HK_Snapshot_Rate = 1./20 #how many time is housekeeping data measured
ADCS_HK_Data_Size = 200        # Bits
OBC_HK_Data_Size = 600        # Bits
Transceiver_HK_Data_Size = 424       #Bits
EPS_HK_Data_Size = 1000       #Bits
SP_HK_Data_Size = 200        #Bits
VP_HK_Data_Size = 200        #Bits

HK_Data_Size_Total = ADCS_HK_Data_Size + OBC_HK_Data_Size + Transceiver_HK_Data_Size + EPS_HK_Data_Size + SP_HK_Data_Size + VP_HK_Data_Size
# HK_Data_Size_Total = 3250

Payload_Snapshot_Rate = 1./60           #Time between snapshots
GNSS_Size = 1000                 #Bits

Data_Generation_Total = HK_Data_Size_Total * HK_Snapshot_Rate + GNSS_Size * Payload_Snapshot_Rate #Data per second

Baud_Rate = 4800                        #Symbols that can be downlinked per second
Encoding_Overhead_Ratio = 0.4           #Bits have to be encoded, error correction, encryption
Compression_Ratio = 4             #How much data can be compressed, essentially increasing data rate
Safety_Factor = 0.9
Downlink_Data_Rate = Baud_Rate * Encoding_Overhead_Ratio * Compression_Ratio * Safety_Factor #Data Downlink Bits per second
Storage_Start = 1000000 #Storage start of 1 Mb 

# Data_Generation_Total = 250


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

# e_data = Get_Eclipse_Data("GSCData-500km-21mar/EclipseData500.txt")
# gp_data = Get_GroundPass_Data("GSCData-500km-21mar/delft60.txt") #delft ground station, 60 degrees incl.
# query_times = pd.date_range('2029-03-21 0:00:00', periods=2500000, freq='s').values

gp_data = Get_GroundPass_Data("GSCData500km/delft80.txt") # delft ground station, sun synchronous orbit
query_times = pd.date_range('2029-01-02 0:00:00', periods=2500000, freq='s').values


########################### Find array how much the data is being added or removed ###############
OverGroundStationArray = getQueryTimeBooleanArray(query_array=query_times,check_array=gp_data)

GroundStationDischargeArray = np.where(OverGroundStationArray, -Downlink_Data_Rate, 0)

Data_Total = Data_Generation_Total + GroundStationDischargeArray

############################## Find Array with battery capacity #################################

#add storage size in bits. Because the timestep is one second, the bits added is the same as the bits per timestep
#undershoot is needed, to clamp the storage at 0

Storage_Raw = Storage_Start + np.cumsum(Data_Total)

undershoot = np.minimum.accumulate(np.minimum(Storage_Raw, 0.0))

Storage_Final = Storage_Raw - undershoot

Storage_Data = pd.DataFrame({
    'Query_Times':query_times,
    'Storage_Rate':Storage_Raw,
    'Storage_Size':Storage_Final
})

######################### Find lowest value, and plot around that time ##########################

Storage_Data = Storage_Data.set_index('Query_Times')
worst_storage_time = Storage_Data['Storage_Size'].idxmax()

window_start = worst_storage_time - pd.Timedelta(seconds=370000)
window_end   = worst_storage_time + pd.Timedelta(seconds=340000)

df_window = Storage_Data.loc[window_start : window_end]

plt.figure(figsize=(10,4))
plt.plot(df_window.index, df_window['Storage_Size'], lw=1)
plt.axvline(worst_storage_time, color='r', ls='--', label=f"max at {worst_storage_time.strftime('%Y-%m-%d %H:%M:%S')}")
plt.xlabel("Time")
plt.ylabel("Storage Size (bits)")
plt.ylim((0.0, Storage_Data['Storage_Size'].max() * 1.1))
plt.title(f"Storage around highest‐storage point ({worst_storage_time})")
plt.legend()
plt.tight_layout()
plt.show()