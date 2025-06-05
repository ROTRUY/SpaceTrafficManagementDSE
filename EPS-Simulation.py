import math
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# SOLAR ARRAY SIZING
#--------------------------------------------------------------------

P_sun = 1322  # Solar flux at 1 AU in W/m^2
t_mis = 6     # Mission duration in months

#--------------------------------------------------------------------
# USER‐DEFINED ORBIT AND SUBSYSTEM PARAMETERS (no interactive inputs)
#--------------------------------------------------------------------

# Orbit Values (minutes)
# (These are now just placeholders; actual sun/penumbra/umbra timing will be read from "daytime.txt".)
t_day = 58.768   # Daylight time per orbit (minutes)  (unused for solar‐power in the loop)
t_ecl = 35.699   # Eclipse time per orbit (minutes)   (unused for solar‐power in the loop)
t_orb = t_day + t_ecl

theta = 0  # Incidence angle of the solar arrays (degrees)

# Subsystem nominal powers (Watts)
ADCS_power_nominal = 0.0        # ADCS nominal consumption (W) when NOT in ground pass
ADCS_power_ground = 1.0         # ADCS consumption (W) DURING a ground pass
CDH_power_nominal = 0.25        # C&DH nominal consumption outside ground pass (W)
CDH_power_ground = 5.0          # C&DH consumption during a ground pass (W)
verification_power = 0.125      # Verification‐payload nominal consumption (W)
payload_power = 0.032           # Payload power during snapshot (W)

# Snapshot event timing: duration (seconds) and interval (seconds)
snapshot_duration = 0.02        # Duration of each snapshot (seconds)
snapshot_interval = 600         # Interval between snapshots (seconds)

# Ground‐pass event timing: duration (minutes) and interval (minutes)
ground_pass_duration = 10.0     # Duration of each ground pass (minutes)
ground_pass_interval = 180.0    # Interval between ground passes (minutes)

# Ephemeris‐update event: power draw (W), duration (minutes), interval (minutes)
ephemeris_power = 0.032         # Power draw during each ephemeris update (W)
ephemeris_duration = 0.75       # Duration of each ephemeris update (minutes)
ephemeris_interval = 120.0      # Interval between ephemeris updates (minutes)

#--------------------------------------------------------------------
# 24-HOUR BATTERY CHARGE/DISCHARGE SIMULATION
#--------------------------------------------------------------------
P_sa = 2.85             # Solar-array output during FULL sunlight (W)
E_bat_capacity = 10.0  # Battery capacity (Wh)

# Simulation parameters
total_minutes = 24 * 60
SoC = np.zeros(total_minutes + 1)  # State of Charge (Wh)
SoC[0] = E_bat_capacity            # Start fully charged

# Precompute intervals (in minutes) for snapshot and ephemeris, but convert snapshot from seconds to minutes
snapshot_duration_min = snapshot_duration / 60.0
snapshot_interval_min = snapshot_interval / 60.0

orbit_period = t_day + t_ecl   # (unused in sunlight logic, but kept for reference)
gp_interval = ground_pass_interval
sp_interval = snapshot_interval_min
ep_interval = ephemeris_interval

for minute in range(total_minutes):
    # 1) Determine solar power based on illumination state
    state = illumination[minute]  # "full", "penumbra", or "umbra"
    if state == "full":
        solar_power = P_sa
    else:
        # penumbra or umbra → no usable power
        solar_power = 0.0

    # 2) Determine ADCS power depending on ground-pass flag
    if gp_flags[minute] == 1:
        ADCS_power = ADCS_power_ground
        CDH_power = CDH_power_ground
    else:
        ADCS_power = ADCS_power_nominal
        CDH_power = CDH_power_nominal

    # 3) Build the total load for this minute
    #    a) Continuous loads: ADCS + verification payload
    load = ADCS_power + verification_power

    #    b) C&DH load (ground-pass vs. nominal already folded into CDH_power)
    load += CDH_power

    #    c) Payload snapshot load (only if within a snapshot window)
    if (minute % sp_interval) < snapshot_duration_min:
        load += payload_power

    #    d) Ephemeris update load (only if within an ephemeris window)
    if (minute % ep_interval) < ephemeris_duration:
        load += ephemeris_power

    # 4) Compute net power to/from battery (W)
    net_power = solar_power - load

    # 5) Convert net power into Wh over one minute
    delta_energy = net_power * (1.0 / 60.0)

    # 6) Update SoC, ensuring it remains between 0 and full capacity
    SoC[minute + 1] = np.clip(SoC[minute] + delta_energy, 0.0, E_bat_capacity)

# Time axis in hours
time_hours = np.arange(total_minutes + 1) / 60.0

# Plot SoC over 24 hours
plt.figure(figsize=(10, 5))
plt.plot(time_hours, SoC, linewidth=2)
plt.xlabel("Time (hours)")
plt.ylabel("Battery State of Charge (Wh)")
plt.title("24-Hour Battery State of Charge Simulation")
plt.grid(True)
plt.xlim(0, 24)
plt.ylim(0, E_bat_capacity * 1.05)
plt.tight_layout()
plt.show()