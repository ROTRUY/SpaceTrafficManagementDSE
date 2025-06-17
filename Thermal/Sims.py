import numpy as np
import plotly.graph_objects as go

#--------------------------------------------------------------
# Mode of operation
#--------------------------------------------------------------

mode = 'M'  # 'H' for High, 'M' for Medium, 'C' for Cold
plot = True  # Set to True to plot the temperature surface

#--------------------------------------------------------------
# Orbital parameters
#--------------------------------------------------------------

OrbitalHeight = 400  # [km] 
R = 6371             # [km] Radius of Earth
OrbitalPeriod = 2 * np.pi * np.sqrt(((OrbitalHeight+R)*1000)**3 / (3.986004418*10**14)) # [s]
betaCrit = np.arcsin(R/(R+OrbitalHeight))  # [rad]
# Note: betaCrit is the critical angle where the orbital height is tangent to the Earth's surface

#--------------------------------------------------------------
# Material properties
#--------------------------------------------------------------

AreaSol = 0.1*0.1#*2*np.cos(np.pi/4) # Area of the solar surface [m^2]
AreaAlb = 4*0.1*0.1#2*np.cos(np.pi/4) # Area of the albedo surface [m^2]
AreaRad = 0.1*0.1#*2*np.cos(np.pi/4) # Area of the radiating surface [m^2]
AreaRem = 0 #0.1*0.1*2 # Area of the remaining surface [m^2]

# Emissivity and absorptivity of the solar surface
aSol = 0.805 
emSol = 0.825
# Emissivity and absorptivity of the albedo surface
emAlb = emSol
aAlb = aSol
# Emissivity and absorptivity of the radiating surface
emRad = emSol
aRad = aSol
# Emissivity and absorptivity of the remaining surface
emRem  = emSol
aRem = aSol

# Heat Transfer parameters
cp = 896     # Specific heat [J/kg*K]
m = 1.105      # Mass [kg]

#--------------------------------------------------------------
# Stefan-Boltzmann constant
#--------------------------------------------------------------

SBC = 5.6051 * 10**(-8)  # [W/m^2*K^4]
#--------------------------------------------------------------
# Functions
#--------------------------------------------------------------'

def FractionTime(beta : float, betaCrit=betaCrit, OrbitalHeight=OrbitalHeight ):
    if np.abs(beta) < betaCrit:
        return 1/np.pi * np.arccos(np.sqrt(OrbitalHeight**2 + 2*R*OrbitalHeight)/((R+OrbitalHeight)*np.cos(beta)))
    return 0

def albedo(beta, mode = 'H'):
    if mode == 'H':
        return 0.28 if np.abs(beta) < np.pi/6 else 0.30  
    elif mode == 'M':
        return 0.23 if np.abs(beta) < np.pi/6 else 0.265
    elif mode == 'C':
        return 0.18 if np.abs(beta) < np.pi/6 else 0.23

def IR(beta , mode = 'H'):
    if mode == 'H':
        return 275 if np.abs(beta) < np.pi/6 else 257.5
    elif mode == 'M':
        return 250 if np.abs(beta) < np.pi/6 else 237.5
    elif mode == 'C':
        return 228 if np.abs(beta) < np.pi/6 else 218

def s(fe, t, OrbitalPeriod=OrbitalPeriod):
    if t < OrbitalPeriod/2*(1-fe) or t > OrbitalPeriod/2*(1+fe):
        return 1
    return 0

def plot_temperature_surface(timerange, betarange, OrbitalPeriod, temperature_data):
    """
    Plots the temperature surface as a 3D plot with min and max markers.
    """
    import plotly.graph_objects as go
    import numpy as np

    # Convert beta to degrees for plotting
    BetaGrid, TimeGrid = np.meshgrid(timerange, betarange * 180 / np.pi)

    fig = go.Figure(data=go.Surface(
        x=TimeGrid[::-1],
        y=BetaGrid / OrbitalPeriod,
        z=temperature_data,
        colorscale='inferno',
        showscale=False
    ))

    # Min and Max Points
    min_idx = np.unravel_index(np.argmin(temperature_data), temperature_data.shape)
    max_idx = np.unravel_index(np.argmax(temperature_data), temperature_data.shape)

    fig.add_trace(go.Scatter3d(
        x=[TimeGrid[::-1][min_idx], TimeGrid[::-1][max_idx]],
        y=[BetaGrid[min_idx] / OrbitalPeriod, BetaGrid[max_idx] / OrbitalPeriod],
        z=[temperature_data[min_idx], temperature_data[max_idx]],
        mode='markers+text',
        text=['Min', 'Max'],
        textposition='top center',
        marker=dict(size=6, color=['blue', 'red'], symbol='circle')
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='Beta Angle [°]', font=dict(size=30)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='Time [Orbits]', font=dict(size=30)),
                tickfont=dict(size=14),
                autorange='reversed'
            ),
            zaxis=dict(
                title=dict(text='Temperature [K]', font=dict(size=30)),
                tickfont=dict(size=14)
            ),
            aspectratio=dict(x=1, y=2, z=1)
        )
    )
    fig.show()

#--------------------------------------------------------------
# Energy balance calculations
#--------------------------------------------------------------	

if mode == 'H':
    S  = 1414 # Solar constant [W/m^2] H1414
    Qgen = 1.05 # Internal heat generation [W]
elif mode == 'M':
    S = 1367 # Solar constant [W/m^2] M1367
    Qgen = 1.05 # Internal heat generation [W]
elif mode == 'C':
    C = 1322 # Solar constant [W/m^2] C1322
    Qgen = 0


timestep = 10 # Time step [s]
timerange = np.arange(0, 10 * OrbitalPeriod + 1, timestep) # Time range for simulation [s]


#Fraction Factors
F = (6371)**2/(500+6371)**2
FP = 1/(2*np.pi)*( np.pi - 2*np.arcsin(np.sqrt(1-F)) - np.sin( 2*np.arcsin(np.sqrt(1-F))))

# Beta range (in radians)
if mode == 'M':
    betarange = np.deg2rad(np.arange(-35,-20,1))
else:
    betarange = np.deg2rad(np.arange(0, 90, 1))

# Prepare data storage
temperature_data = []  # Will be 2D: len(beta) x len(time)

# Loop over each beta angle
for beta in betarange:
    Ti1 = 273.15  # Initial temperature
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
Max = np.max(temperature_data)
Min = np.min(temperature_data)
print(f"Max: {Max - 273.15:.2f} °C, Min: {Min - 273.15:.2f} °C")
if plot == True:
    print("Plotting temperature surface...")
    plot_temperature_surface(timerange, betarange, OrbitalPeriod, temperature_data)



class Node:
    def __init__(self, k : float, emissivity : float, absorptivity : float, Ntemps  : list, T : float, m : float, cp : float , F : float, Area : float):
       
        self.ContactConductance = k # W/m²*K
        self.emissivity = emissivity 
        self.absorptivity = absorptivity
        self.m = m
        self.cp = cp
        self.NeighborTemperatures = Ntemps
        self.T = T
        self.Qdoti = 0.0
        self.F = F
        self.Area = Area
        self.Toggle = [0,0,0,0,0]  # Toggle for different heat sources: [Solar, Albedo, Infrared, Internal, Radiation]

    
    def WriteTemps (self,Ntemps):
        self.NeighborTemperatures = Ntemps

    def HeatTransfer(self, T, Qdoti : float = 0 ):
        self.T = T
        for i in self.NeighborTemperatures:
            self.Qdoti += self.ContactConductance * (i - self.T)

    def SolarHeat(self, S : float, Area : float ):        
        self.Qdoti +=  S * Area * s(fe, t % OrbitalPeriod) * self.absorptivity * self.F
    
    def AlbedoHeat(self, S : float, Area : float, albedo : float):
        self.Qdoti += F * albedo * S * Area * s(fe, t % OrbitalPeriod) * self.absorptivity
    
    def InfraredHeat(self, IR : float, Area : float):
        self.Qdoti += F * IR * Area * self.emissivity

    def RadiationHeat(self, Area : float):
        self.Qdoti -= (Area * SBC * self.emissivity) * self.T**4

    def InternalHeat(self, Qgen : float):
        self.Qdoti += Qgen

    def TemperatureChange(self, timestep : float, cp : float, m : float):
        if self.Toggle[0] == 1:
            self.SolarHeat(S, self.Area)
        if self.Toggle[1] == 1: 
            self.AlbedoHeat(S, self.Area, albedo(beta, mode))
        if self.Toggle[2] == 1:
            self.InfraredHeat(IR(beta, mode), self.Area)
        if self.Toggle[3] == 1:
            self.InternalHeat(Qgen)
        if self.Toggle[4] == 1:
            self.RadiationHeat(self.Area)    
        self.T += timestep / (cp * m) * self.Qdoti
        self.Qdoti = 0.0  # Reset for next calculation
    def __call__(self):
        return self.T