import pandas as pd
from Read_GMAT_Files import *

P_sun = 1322  # Solar flux at 1 AU in W/m^2
t_mis = 6     # Mission duration in months

ADCS_power_nominal = 0.0        # ADCS nominal consumption (W) when NOT in ground pass
CDH_power_nominal = 0.25        # C&DH nominal consumption outside ground pass (W)
CDH_power_ground = 5.0          # C&DH consumption during a ground pass (W)
verification_power = 0.125      # Verification‐payload nominal consumption (W)
payload_power = 0.032



# print(Get_Eclipse_Data("GSCData-500km-21mar/EclipseData500.txt").dtypes)
print(Get_GroundPass_Data("GSCData-500km-21mar/delft60.txt").dtypes)
