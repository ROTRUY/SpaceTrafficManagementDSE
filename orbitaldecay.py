import math
from datetime import datetime

# Get input from the user
name = "Patra"
mass = 1.2  # Mass in kg
area = 0.01  # Area in m^2
height = 500  # Initial height in km
f10 = 35  # F10.7 index
ap = 13.8  # Geomagnetic A index

# Print header information
now = datetime.now()
print("\nSATELLITE ORBITAL DECAY - Model date/time", now.strftime("%Y-%m-%d @ %H:%M:%S"))
print("\nSatellite -", name)
print(f" Mass = {mass:8.1f} kg")
print(f" Area = {area:7.1f} m^2")
print(f" Initial height = {height:6.1f} km")
print(f" F10.7 = {int(f10)} Ap = {int(ap)}\n")

# Print column headings
print(" TIME    HEIGHT   PERIOD   MEAN MOTION   DECAY")
print("(days)    (km)     (mins)    (rev/day)   (rev/day^2)")

# Constants
Re = 6371000      # Earth radius (m)
Me = 5.9722e24        # Earth mass (kg)
G = 6.6743015e-11        # Gravitational constant
pi = math.pi
T = 0               # Time (days)
dT = 0.1            # Time increment (days)
D9 = dT * 86400     # Time increment in seconds
H1 = 10             # Print height step (km)
H2 = height         # Initialize print height (km)

# Initial orbital radius and period
R = Re + height * 1000  # radius in meters
P = 2 * pi * math.sqrt(R**3 / (G * Me))  # period in seconds

# Iterate over time
while True:
    SH = (900 + 2.5 * (f10 - 70) + 1.5 * ap) / (27 - 0.012 * (height - 200)) # scale height in km
    DN = 6e-10 * math.exp(-(height - 175) / SH)     # exponential density starting from 175 km
    dP = 3 * pi * area / mass * R * DN * D9 # change in period (seconds)

    if height <= H2:
        Pm = P / 60 # period in minutes
        MM = 1440 / Pm  # mean motion (rev/day)
        nMM = 1440 / ((P - dP) / 60) # new mean motion after decay
        decay = dP / dT / P * MM # decay rate (rev/day^2)
        print(f"{T:6.1f} {height:8.1f} {Pm:8.1f} {MM:10.4f} {decay:13.2e}")
        H2 -= H1 # update print height

    if height < 180: # Re-entry condition
        break

    P -= dP
    T += dT
    R = (G * Me * P**2 / (4 * pi**2))**(1/3)
    height = (R - Re) / 1000

# Print estimated lifetime
print(f"\nRe-entry after {T:.0f} days ({T / 365:.2f} years)")
