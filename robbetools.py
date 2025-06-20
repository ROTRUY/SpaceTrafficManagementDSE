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

def solar_radiation_pressure_torque(A_s: int|float, q: int|float, phi: int|float, d: int|float, Phi: int|float=1366) -> float:
    """
    Function to calculate disturbance torque on a satellite by solar radiation pressure.

    Inputs
    ---
    - `A_s:` Sunlit surface area in [m²].
    - `q:` Unitless reflectance factor (0 for perfect absorption to 1 for perfect reflection).
    - `phi:` Angle of incidence of the sun in [deg].
    - `d:` Distance between centre of mass and centre of solar radiation pressure in [m].
    - `Phi:` Solar constant adjusted for distance from the sun in [W/m²]. Default: for Earth, 1366 [W/m²]

    Outputs
    ---
    - `T_srp:` Disturbance torque due to solar radiation pressure in [Nm].
    """
    return Phi / c * A_s * (1 + q) * d * cos(phi*pi/180)

def gravity_gradient_torque_worst_case(L: int|float, m: int|float, h: int|float, theta: int|float|None = None) -> float:
    """
    Function to calculate the worst case gravity gradient torque on a satellite.

    Inputs
    ---
    - `L:` Length of the satellite in [m].
    - `m:` Mass of the satellite in [kg].
    - `h:` Altitude of the satellite in [km].
    - `theta:` Angle between the satellite axis and the center of the earth in [rad].

    Outputs
    ---
    - `T_gg:` Worst case gravity gradient torque in [Nm].
    """
    r = (r_earth + h) * 1000  # Distance from the center of the earth to the satellite in [m]
    
    if theta is None:
        return max(gravity_gradient_torque_worst_case(L, m, h, theta=theta) for theta in [n/10 for n in range(0, 70)])  # Calculate the maximum torque over all angles
        
    F1 = mu_earth * m / 2 / (r - L * cos(theta) / 2)**2
    F2 = mu_earth * m / 2 / (r + L * cos(theta) / 2)**2
    
    return (F1 - F2) * L * sin(theta) / 2  # Torque in [Nm]. The factor of 2 is because the torque is calculated for the center of mass, not the center of the satellite.

def gravity_gradient_torque_alt(h: int|float, m: int|float, r: int|float) -> float:
    """
    Function to calculate worst-case gravity gradient torque

    Inputs
    ---
    - `h:` Orbit altitude in [km].
    - `m:` Spacecraft mass in [kg].
    - `r:` Distance for moment of inertia in [m].

    Outputs
    ---
    - `T_g:` Disturbance torque in [Nm].
    """
    return 3 * mu_earth / (2 * 1000 * (h + r_earth))**3 * m * r**2

def aero_drag_torque(A: int|float, delta_cp: int|float, h: int|float) -> float:
    """
    Function to calculate the aerodynamic drag torque on a satellite.

    Inputs
    ---
    - `A:` Cross-sectional area of the satellite in [m²].
    - `h:` Altitude of the satellite in [km].

    Outputs
    ---
    - `T_ad:` Aerodynamic drag torque in [Nm].
    """
    r = (r_earth + h) * 1000
    V = sqrt(mu_earth / r)# Orbital velocity in [m/s]
    Cd = 2.5
    rho1 = 7.22e-12 #ISA value for 300 km altitude in [kg/m³]
    rho2 = 5.68e-13 #ISA value for 400 km altitude in [kg/m³]
    rho = 1.51E-12 #ISA value for 360 km altitude in [kg/m³]
    #print(V)
    return 0.5 * rho * V**2 * A * Cd * delta_cp

def min_dipol_moment(beta_min: int|float) -> float:
    """
    Function to calculate the minimum dipole moment of a magnetometer.

    Inputs
    ---
    - `beta_min:` Desired pointing accuracy [rad].
    
    Outputs
    ---
    - `m_min:` Minimum dipole moment in [A m²].
    """
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ts = solar_radiation_pressure_torque(0.02, 0.2, 0, 0.02)
    print("Ts: " + str(Ts))
    Tg = gravity_gradient_torque_alt(400, 2, 0.1)
    print("Tg: " + str(Tg))
    Ta = aero_drag_torque(0.02, 0.02, 360)
    print("Ta: " + str(Ta))
    
    Trms = sqrt(Ts**2 + Tg**2 + Ta**2)  # Total disturbance torque in [Nm]
    print("Trms: " + str(Trms))
    Bmin = 2.44e-5 # Minimum magnetic field strength in [T] (worst case)
    
    m_min = 15 * Trms / Bmin / sin(beta_min)  # Minimum dipole moment in [A m²]
    Vmin = m_min * mu0 / 1.45 #1.45 for N52 magnets
    print("Vmin: " + str(Vmin))
    V = pi * 0.004 ** 2 * 0.006
    print("V: " + str(V))
    m = 1.45 * V /mu0 # 1.45 for N52 magnet
    print("m: " + str(m))
    
    rho = 7500 #kg/m³ for N52 magnet
    mass = V * rho
    print("mass: " + str(mass))
    return m_min

def sensi_min_dipol_moment(beta_max: int|float, n_trms) -> float:
    """
    Function to calculate the minimum dipole moment of a magnetometer.

    Inputs
    ---
    - `beta_min:` Desired pointing accuracy [rad].
    
    Outputs
    ---
    - `m_min:` Minimum dipole moment in [A m²].
    """
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ts = solar_radiation_pressure_torque(0.02, 0.2, 0, 0.02)
    Tg = gravity_gradient_torque_alt(400, 2, 0.1)
    Ta = aero_drag_torque(0.02, 0.02, 360)

    Trms = sqrt(Ts**2 + Tg**2 + Ta**2)  * n_trms# Total disturbance torque in [Nm]
    
    Bmin = 2.44e-5 # Minimum magnetic field strength in [T] (worst case)
    
    m_min = 15 * Trms / Bmin / sin(beta_max)  # Minimum dipole moment in [A m²]
    print("m_min: " + str(m_min))
    Vmin = m_min * mu0 / 1.45 #1.45 for N52 magnets
    
    V = pi * 0.004 ** 2 * 0.006
    
    m = 1.45 * V /mu0 # 1.45 for N52 magnet
    
    
    rho = 7500 #kg/m³ for N52 magnet
    mass = V * rho
    
    expected_mass = Vmin * rho  # Expected mass of the magnet in [kg]
    return Trms, expected_mass, m_min

def performe_sensi_permanent_magnet():
    """
    Function to perform sensitivity analysis for permanent magnet.
    """
    beta_max = pi/180 * 10  # Maximum pointing accuracy in [rad]
    n_trms = 1.1  # Number of disturbance torque values to average
    n_betamax = 0.9  # Number of beta_max values to average
    
    Trms, expected_mass, m_min = sensi_min_dipol_moment(beta_max, 1)
    
    Trms2, expected_mass2, m_min2 = sensi_min_dipol_moment(beta_max, n_trms)
    
    print("Trms change =========================================")
    print("Trms diff: " + str(Trms2 - Trms))
    print("Expected mass diff: " + str(expected_mass2 - expected_mass))
    print("mass percentage diff: " + str((expected_mass2 - expected_mass) / expected_mass * 100) + "%")
    print("m_min diff: " + str(m_min2 / m_min))
    print("==============================================")
    
    Trms, expected_mass, m_min = sensi_min_dipol_moment(beta_max, 1)
    
    Trms2, expected_mass2, m_min2 = sensi_min_dipol_moment(beta_max * n_betamax, 1)
    
    print("Beta_max change =========================================")
    print("beta max diff: " + str( beta_max * n_betamax - beta_max))
    print("Expected mass diff: " + str(expected_mass2 - expected_mass))
    print("mass percentage diff: " + str((expected_mass2 - expected_mass) / expected_mass * 100) + "%")
    print("m_min diff: " + str(m_min2 / m_min))
    print("==============================================")
    
    
    return 

def natural_frequency():
    rho = 2.70  # [g/cm³]
    m = 120  # [g]
    Lt = 10  # [cm]
    t = (Lt - sqrt(Lt**2 - m / rho / Lt)) / 2 *1e-3  # [m]

    Em = 71e+9  # Young's modulus of structural material [Pa]
    L = 0.1  # [m]
    A_lat = L**2  # [m²]
    As = A_lat - (L - 2 * t)**2  # [m²]
    ms = 1.2  # [kg]
    I = (L**4 - (L - 2 * t)**4) / 12

    Es = Em * As / A_lat

    fnlong = sqrt(Es * A_lat /( ms * L)) / (2 * pi)
    fnlat = sqrt(3 * Es * I / (ms * L**3)) / (2 * pi)
    return t, A_lat, As, Es, I, fnlong, fnlat

### MAIN
if __name__ == "__main__":
    # print(solar_radiation_pressure_torque(0.02, .2, 0, 0.02))
    # print(gravity_gradient_torque_worst_case(0.1, 2, 400, theta=pi/4))
    # print(gravity_gradient_torque_alt(400, 2, 0.1))
    # print(aero_drag_torque(0.02, 0.1, 360))

    # magneticstuff_gerhard()
    # print(natural_frequency())
    performe_sensi_permanent_magnet()
