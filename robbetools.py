"""
Toolbox with various functions for various things
"""
### IMPORTS
from math import *
import matplotlib.pyplot as plt

### GLOBALS
g0 = 9.80665  # Standard gravity [m/s²]
mu_earth = 3.986004418e+14  # Standard gravitational parameter μ (GM) [m3 s−2]
r_earth = 6378.137  # Radius of earth [km]
c = 299792458  # speed of light [m/s]

### FUNCTIONS / TOOLS
def propellant_mass(m_dry: int|float, dv: int|float, I_sp: int|float) -> float:
    """
    This function calculates the required propellant mass based on Tsiolovsky's rocket equation.

    Inputs
    ------
    - `m_dry`: Dry mass of the spacecraft in [kg].
    - `dv`: Required change in velocity delta v in [m/s].
    - `I_sp`: Specific impulse of the thruster in [s].

    Outputs
    -------
    - `m_prop`: Required propellant mass to give the required delta v in [kg].
    """
    return m_dry * (exp(dv / (g0 * I_sp)) - 1)

def ground_station_contact(h: int|float, fov: int|float=180) -> int|float:
    """
    This function estimates the contact time between a ground station and a satellite in a circular orbit.

    Assumptions
    ---
    - Satellite is in a circular orbit!
    - Nothing is blocking the view of the ground station in any direction.

    Inputs
    ---
    - `h`: Altitude of the satellite in [km].
    - `fov`: Ground station FOV in [deg]. Default value is 180° (unobstructed ideal ground station).

    Outputs
    ---
    - `t`: Time the satellite is visible to a ground station [min].
    """
    a = (r_earth + h)  # Semi-major axis [km]
    T = 2 * pi * sqrt((a * 1000)**3 / mu_earth) / 60  # Orbital period [min]
    print(T)
    fov_rad = fov * pi / 180  # FOV in [rad].
    frac_vis = acos(r_earth / a) / fov_rad  # Fraction of the orbit the satellite is visible to the ground station.
    t = T * frac_vis  # Time the satellite is visible to a ground station [min].
    return t

def power_usage_sp(T: int|float=3600) -> int|float:
    """
    This function calculates the power usage of the standardized payload for option A.

    Accounts for
    ---
    - Power usage while receiver sleeps.
    - Power usage for each snapshot taken.
    - Power usage of the antenna

    Inputs
    ---
    - `T`: Sampling period in [s].

    Outputs
    ---
    - `p`: Power usage of the standardized payload in [mW].
    """
    pu_snap = 1e-3 # Power usage of a single snapshot [mWh]
    pu_sleep = 5.3e-3  # Power usage of microcontroller in sleeping mode [mW]
    pu_antenna = 37  # Power usage of antenna [mW]

    return pu_antenna + pu_sleep + pu_snap * 3600 / T

def data_generated(option: str='A') -> int|float:
    """
    Function to calculate data generated with a single snapshot, based on option.

    Inputs
    ---
    - `Option:` A, B or C

    Outputs
    ---
    - `D:` Data generated per snapshot
    """
    snap_duration = 12  # [ms]
    sampling_freq = 4.092  # [MHz]
    quantization = 2  # [bits]
    match option:
        case 'A':
            return snap_duration * 1e-3 * sampling_freq * 1e+6 * quantization
        case 'B':
            return 400
        case 'C':
            return 256

def decibels(x: int|float) -> float:
   """
   Function to convert to decibel scale.

   Inputs
   ---
   - `x:` Number to convert.

   Output
   ---
   - `x [dB]:` Number in decibels.
   """
   return 10 * log10(x)

def distance_sc_ground_earth(h: int|float) -> float:
    """
    Function to calculate distance between spacecraft and earth.

    Assumptions
    ---
    - Circular orbit.

    Inputs
    ---
    - `h:` Altitude of the spacecraft in [km].

    Outputs
    ---
    - `d:` Distance between spacecraft and earth in [km].
    """
    return sqrt((r_earth + h) ** 2 - r_earth ** 2)

### CLASSES / OBJECTS

### MAIN
if __name__ == "__main__":
    periodlst = []
    powerlst = []
    dataAlst = []
    dataBlst = []
    dataClst = []
    for T in range(30, 24*60*60+1):
        periodlst.append(T)
        powerlst.append(power_usage_sp(T))
        dataAlst.append(data_generated('A') * 24*60*60/T)
        dataBlst.append(data_generated('B') * 24*60*60/T)
        dataClst.append(data_generated('C') * 24*60*60/T)
    print(data_generated('A'), data_generated('B'), data_generated('C'))
    
    pltpower = True
    if pltpower:
        plt.plot(periodlst, powerlst)
        plt.axvline(x = 60, linestyle='dashed', color='blue', label = '1min sampling period')
        plt.axvline(x = 600, linestyle='dashed', color='red', label = '10min sampling period')
        plt.axvline(x = 60*60, linestyle='dashed', color='purple', label = '1hr sampling period')
        plt.axvline(x = 60*60*8, linestyle='dashed', color='orange', label = '8hr sampling period')
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Sampling period [s]')
        plt.ylabel('Power usage [mW]')
        plt.grid()
        plt.tight_layout()
        plt.savefig("plots/powerplot")
        plt.close()

    pltdata = True
    if pltdata:
        plt.plot(periodlst, dataAlst, label='Option A')
        plt.plot(periodlst, dataBlst, label='Option B')
        plt.plot(periodlst, dataClst, label='Option C')
        plt.axvline(x = 60, linestyle='dashed', color='blue', label = '1min sampling period')
        plt.axvline(x = 600, linestyle='dashed', color='red', label = '10min sampling period')
        plt.axvline(x = 60*60, linestyle='dashed', color='purple', label = '1hr sampling period')
        plt.axvline(x = 60*60*8, linestyle='dashed', color='orange', label = '8hr sampling period')
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Sampling period [s]')
        plt.ylabel('Data generated per day [bits]')
        plt.tight_layout()
        plt.grid()
        plt.savefig("plots/dataplot")