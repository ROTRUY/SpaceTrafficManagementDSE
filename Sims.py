import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
OrbitalHeight = 500  # [km]
R = 6371             # [km]
OrbitalPeriod = 5667 # [s]
betaCrit = np.arcsin(R/(R+OrbitalHeight))  # [rad]

# Functions
def FractionTime(beta, betaCrit=betaCrit, OrbitalHeight=OrbitalHeight):
    if np.abs(beta) < betaCrit:
        return 1/np.pi * np.arccos(np.sqrt(OrbitalHeight**2 + 2*R*OrbitalHeight)/((R+OrbitalHeight)*np.cos(beta)))
    return 0

def albedo(beta):
    return 0.23 if beta < np.pi/6 else 0.265

def IR(beta):
    return 250 if beta < np.pi/6 else 237.5

def s(fe, t, OrbitalPeriod=OrbitalPeriod):
    if t < OrbitalPeriod/2*(1-fe) or t > OrbitalPeriod/2*(1+fe):
        return 1
    return 0

# Geometry/Material Parameters
AreaAlb = 6*0.1*0.1*np.cos(np.pi/4)
emAlb = 0.825
aAlb = 0.805
AreaRad = 6*0.1*0.1*np.cos(np.pi/4)
emRad = 0.825
aRad = 0.805
AreaSol = 6*0.1*0.1*np.cos(np.pi/4)
emSol = 0.825
aSol = 0.805
AreaRem = 0.1*0.1*2
emRem  = 0.90
aRem = 0.09

# Energy and material constants
S = 1367   # Solar constant [W/m^2]
Qgen = 5    # Internal heat generation
SBC = 5.6051e-8  # Stefan-Boltzmann Constant [W/m^2K^4]
cp = 896     # Specific heat [J/kg*K]
m = 5       # Mass [kg]
timestep = 1  # Time step [s]
timerange = np.arange(0, 7 * OrbitalPeriod + 1, timestep)
F = (6371)**2/(500+6371)**2
FP = 1/(2*np.pi)*( np.pi - 2*np.arcsin(np.sqrt(1-F)) - np.sin( 2*np.arcsin(np.sqrt(1-F))))

# Beta range (in radians)
betarange_deg = np.arange(-30,-25, 1)
betarange_rad = np.deg2rad(betarange_deg)

# Prepare data storage
temperature_data = []  # Will be 2D: len(beta) x len(time)

# Loop over each beta angle
for beta in betarange_rad:
    Ti1 = 293.15  # Initial temperature
    temps = []    # Store temperature over time for current beta
    for t in timerange:
        fe = FractionTime(beta)
        Qin = F * IR(beta) * AreaRad * emRad \
            + S * AreaSol * s(fe, t % OrbitalPeriod) * aSol \
            + F * albedo(beta) * S * AreaAlb * s(fe, t % OrbitalPeriod) * aAlb \
            + Qgen \
            + FP * albedo(beta) * S * AreaRem * s(fe, t % OrbitalPeriod) * aRem
        Qrad = (AreaSol * SBC * emSol + AreaAlb * SBC * emAlb +
                AreaRad * SBC * emRad + AreaRem * SBC * emRem) * Ti1**4
        Qdoti = Qin - Qrad
        T = Ti1 + timestep / (cp * m) * Qdoti
        temps.append(T)
        Ti1 = T
    temperature_data.append(temps)

temperature_data = np.array(temperature_data)

# Convert beta to degrees for plotting
BetaGrid, TimeGrid = np.meshgrid(timerange, betarange_deg)

# 3D Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(TimeGrid, BetaGrid/ OrbitalPeriod, temperature_data - 273.15, cmap='inferno')
ax.set_ylabel("Time [orbits]")
ax.set_xlabel("Beta [deg]")
ax.set_zlabel("Temperature [°C]")
#ax.set_title("Temperature Evolution vs Beta and Time")
ax.invert_xaxis() 
plt.tight_layout()
plt.show()
