import numpy as np
import matplotlib.pyplot as plt

Tdot = np.deg2rad(360/365) # degrees/Half a day
J2 = 1082.62*10**(-6) 
req = 6378 # km
mu = 0.3986*10**6 #km³/s²
r = (6378 + 500) # km
i = np.deg2rad(60) # degrees

wdot = -3/2 * J2 * (req/r)**2 * np.sqrt(mu/r**3)*np.cos(i)
wdotd = wdot * 60 * 60 * 24
timerange = np.arange(0,365,0.25)
betacrit = np.arcsin(6378/(6378+500))

def FractionTime(beta : float, betaCrit=betacrit, OrbitalHeight=500 ):
    if np.abs(beta) < betaCrit:
        return 1/np.pi * np.arccos(np.sqrt(OrbitalHeight**2 + 2*req*OrbitalHeight)/((req+OrbitalHeight)*np.cos(beta)))
    return 0


T = 0
e = np.deg2rad(23.45)
Ts = timerange*Tdot
RAANs = np.arange(-np.deg2rad(180),-np.deg2rad(180) + wdotd*365, wdotd/4)
beta_deg = np.arcsin(np.cos(Ts)*np.sin(RAANs)*np.sin(i) - np.sin(Ts) * np.cos(e) * np.cos(RAANs) * np.sin(i) + np.sin(Ts) * np.sin(e) * np.cos(i))

bmin = min(np.rad2deg(beta_deg[0:183*2:]))
bmax = max(np.rad2deg(beta_deg))
plt.figure(figsize=(10, 5))
plt.plot(timerange, np.rad2deg(np.array(beta_deg)))
plt.hlines(bmin,0,365,linestyles='--' , colors='r',label ='Minima :{:.2f}°'.format(bmin))
plt.hlines(bmax,0,365,linestyles='--' ,colors='r',label ='Maxima :{:.2f}°'.format(bmax))
plt.hlines(np.rad2deg(np.arcsin(6378.1/(6378.1+500))),0,365,colors='b', linestyles='dotted')
plt.hlines(np.rad2deg(-np.arcsin(6378.1/(6378.1+500))),0,365,colors='b', linestyles='dotted',label='Critical β :±{:.2f}°'.format(np.rad2deg(np.arcsin(6378.1/(6378.1+500)))))
#plt.title('Beta Angle Over One Year (60° Inclination, 500 km Altitude)')
plt.xlabel('Day Since Vernal Equinox', fontsize = 14)
plt.ylabel('β Angle [°]', fontsize = 14)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.grid(True)
plt.xlim(0,183)
plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.savefig('Beta Angle Variation.png')
plt.show()

plt.figure(figsize=(10, 5))
bmax = FractionTime(min(np.abs(beta_deg)))
plt.hlines(bmax*100,0,365,linestyles='--' ,colors='r',label ='Maxima :{:.2f}'.format(bmax*100))
plt.plot(timerange, [FractionTime(beta)*100 for beta in beta_deg])
#plt.title('Beta Angle Over One Year (60° Inclination, 500 km Altitude)')
plt.xlabel('Day Since Vernal Equinox', fontsize = 14)
plt.ylabel('Percentage of Orbit Spend in Eclipse [%]', fontsize = 14)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.grid(True)
plt.xlim(0,183)
plt.tight_layout()
plt.legend(loc='best', fontsize=14)
plt.savefig('EclipseFraction.png')
plt.show()