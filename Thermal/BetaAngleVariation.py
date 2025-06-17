import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from datetime import datetime, timedelta
# Constants
earth_radius_km = 6378.1
altitude_km = 500
inclination_deg = 60
semi_major_axis_km = earth_radius_km + altitude_km
# Load ephemeris and timescale
ts = load.timescale()
eph = load('de421.bsp')
# Time range: one year from today
start_time = datetime.utcnow()
times = ts.utc(start_time.year, start_time.month, start_time.day, range(0, 365))
# Get Sun position
sun = eph['sun']
earth = eph['earth']
sun_pos = earth.at(times).observe(sun).position.km  # Sun vector in ECI (km)
# Normalize sun vectors
sun_unit = sun_pos / np.linalg.norm(sun_pos, axis=0)
# Calculate orbital plane normal vector (ECI)
# Using simplified constant RAAN = 0 for demonstration
raan = 0
inc = np.radians(inclination_deg)
n_hat = np.array([
    np.sin(inc) * np.sin(raan),
    -np.sin(inc) * np.cos(raan),
    np.cos(inc)
])[:, np.newaxis]  # shape (3,1)
# Dot product to get sin(beta)
dot_products = np.dot(n_hat.T, sun_unit)
beta_rad = np.arcsin(dot_products).flatten()
beta_deg = np.degrees(beta_rad)
# Plot
min = min(beta_deg)
max = max(beta_deg)
plt.figure(figsize=(10, 5))
plt.plot(range(365), beta_deg)
plt.hlines(min,0,365,linestyles='--' , colors='r',label ='Minima :{:.2f}'.format(min))
plt.hlines(max,0,365,linestyles='--' ,colors='r',label ='Maxima :{:.2f}'.format(max))
plt.xlim(0,365)
#plt.title('Beta Angle Over One Year (55° Inclination, 400 km Altitude)')
plt.xlabel('Day of Year')
plt.ylabel('Beta Angle [°]')
plt.grid(True)
plt.legend(loc= (0.8,0.8))
plt.tight_layout()
plt.savefig('Beta Angle Variation.png')
plt.show()