import numpy as np
#--------------------------------------------------------------
# Mode of operation
# --------------------------------------------------------------`
mode = 'H' # 'H' for High, 'M' for Medium, 'C' for Cold
PLOT = False
#--------------------------------------------------------------
# Sun parameters
#--------------------------------------------------------------
semimajor = 149.598e9
ecc = 0.0167
astunit = 1.495978707*10**11

def theta(t):
    # Mean anomaly increases linearly with time'
    M = 2 * np.pi * (t + (365.256 * 24 * 60 * 60/2)) / (365.256 * 24 * 60 * 60) 
    return M+(2*ecc*np.sin(M))
def r(theta):
    return semimajor * (1 - ecc**2) / (1 + ecc * np.cos(theta))

#--------------------------------------------------------------
# Orbital parameters
#--------------------------------------------------------------

OrbitalHeight = 500  # [km] 
R = 6371             # [km] Radius of Earth
OrbitalPeriod = 2 * np.pi * np.sqrt(((OrbitalHeight+R)*1000)**3 / (3.986004418*10**14)) # [s]
F = (R)**2/(OrbitalHeight+R)**2
FP = 1/(2*np.pi)*( np.pi - 2*np.arcsin(np.sqrt(1-F)) - np.sin( 2*np.arcsin(np.sqrt(1-F))))
betaCrit = np.arcsin(R/(R+OrbitalHeight))  # [rad]
beta = 0 #* np.pi / 180  # [rad] beta angle, can be adjusted

#--------------------------------------------------------------
# Material properties
#--------------------------------------------------------------

#Solar surface properties
emSol = 0.825 # Emissivity of the solar surface
alSol= 0.805 # Absorptivity of the solar surface
Area = 0.1*0.1  # m² of the solar surface

#Frame properties
AreaFrame = 0.008*0.008*8 + 0.008*0.112*8  # m², area of the frame
CASFrame = (0.0015*0.0684)*2 + (0.0828*0.0122)*2 # m², contact area of the frame with side panels
CATFrame = (0.0825*0.005)*4  # m², contact area of the frame with top panels

# Rod properties
CARod = 0.0036*np.pi*0.0025 # m², contact area of the rod with the frame
MassRod = (0.0032/2)**2*0.083*2712  # kg, mass of the rod
CASub = 0.0032*np.pi*0.001623 # m², contact area of the rod with the subsystems

# Subsystem properties
CABS = 0.066*0.01 # m², area of the Pin Connectors
AreaAnt = 0.163 * 0.003 *np.pi * 4
emAnt = 0.2
aAnt = 0.4

#--------------------------------------------------------------
# Stefan-Boltzmann constant
#--------------------------------------------------------------

SBC = 5.67e-8  # Stefan-Boltzmann constant [W/m²*K⁴]

#--------------------------------------------------------------
# Functions
#--------------------------------------------------------------

# Fraction of time the satellite is in sunlight based on beta angle
def FractionTime(beta : float, betaCrit=betaCrit, OrbitalHeight=OrbitalHeight ):
    if np.abs(beta) < betaCrit:
        return 1/np.pi * np.arccos(np.sqrt(OrbitalHeight**2 + 2*R*OrbitalHeight)/((R+OrbitalHeight)*np.cos(beta)))
    return 0

# Albedo function based on beta angle and mode
def albedo(beta, mode = 'H'):
    if mode == 'H':
        return 0.28 if np.abs(beta) < np.pi/6 else 0.30  
    elif mode == 'M':
        return 0.23 if np.abs(beta) < np.pi/6 else 0.265
    elif mode == 'C':
        return 0.18 if np.abs(beta) < np.pi/6 else 0.23
    
# Function to determine if the satellite is in sunlight based on fraction of time and orbital period
def s(fe, t, OrbitalPeriod=OrbitalPeriod):
    if t < OrbitalPeriod/2*(1-fe) or t > OrbitalPeriod/2*(1+fe):
        return 1
    return 0

# Function to calculate orbital angle based on time
def Theta(t, OrbitalPeriod=OrbitalPeriod):
        return 2*np.pi/OrbitalPeriod*(t%OrbitalPeriod)

# Function to turn on the albedo heat based on time and beta angle
def sAlbedo(t, OrbitalPeriod=OrbitalPeriod):
    if t < OrbitalPeriod/4 or t > 3*OrbitalPeriod/4:
        return 1
    return 0

#--------------------------------------------------------------
# Node class for thermal model
#--------------------------------------------------------------

class Node:
    def __init__(self, emissivity : float, absorptivity : float, Neigbors : list, ID : float, m : float, \
                 cp : float, Area : float, Qgen : float, F : float, Adjust : bool = 1, BetaAdjust : bool = 1):
        
        #Initialize node properties
        self.emissivity = emissivity 
        self.absorptivity = absorptivity
        self.m = m # Mass of the node [kg]
        self.cp = cp # Specific heat capacity [J/kg*K]
        self.Neighbors = Neigbors # List of neighbors in the format [(ID, contact area, thermal conductivity)]
        self.T = 273.15  # Initial temperature in Kelvin
        self.ID = ID # Node ID
        self.F = F # Form Factor
        self.Area = Area #  Radiative area of the node [m²]
        self.Qgen = Qgen # Internal heat generation [W]

        # Simulation parameters
        self.NeighborTemperatures = [273.15] * len(self.Neighbors)  # Initialize neighbor temperatures to 273.15 K
        self.Toggle = [0,0,0,0,0]  # Toggle for different heat sources: [Solar, Albedo, Infrared, Internal, Radiation]
        self.Adjust = Adjust  # Adjustment for solar angle
        self.BetaAdjust = BetaAdjust  # Adjustment for beta angle
        self.Qdoti = 0.0 # Heat flux for a certain timestep

    # Method to write temperatures of neighboring nodes	
    def WriteTemps (self,Nodetemps):
        self.NeighborTemperatures = [Nodetemps[j] for j in [i[0] for i in self.Neighbors]]  # Update neighbor temperatures based on current node temperatures

    # Method to calculate heat transfer from neighbors
    def HeatTransfer(self):
        inter = 0.0
        for i,e in enumerate(self.NeighborTemperatures):
            inter += self.Neighbors[i][2] * (e - self.T) * self.Neighbors[i][1]  # Heat transfer from neighbors
        return inter

    # Method to calculate solar heat based on solar flux, time, and beta angle
    def SolarHeat(self, S : float,t : float, beta : float = beta):
        if self.BetaAdjust == 1:
            if self.Adjust == 1:
                return np.abs((S * self.Area * self.absorptivity ) *np.cos(Theta(t)) * (np.cos(beta)) *  s(fe, t % OrbitalPeriod))
            else:
                return np.abs((S * self.Area * self.absorptivity ) *np.sin(Theta(t)) *  s(fe, t % OrbitalPeriod)) # * (np.cos(beta)) 
        if self.Adjust == 1:
            return np.abs((S * self.Area * self.absorptivity ) *np.cos(Theta(t)) * (np.sin(beta)) *  s(fe, t % OrbitalPeriod))
        else:
            return np.abs((S * self.Area * self.absorptivity ) *np.sin(Theta(t)) * (np.sin(beta)) *  s(fe, t % OrbitalPeriod))

    # Method to calculate albedo heat based on solar flux, albedo, and beta angle
    def AlbedoHeat(self, S : float, albedo : float):
        ksi = np.arccos(np.cos(beta) * np.cos(Theta(t)))
        return self.F * (albedo + 4.9115*10**(-9)*ksi**4 + 6.0372*10**(-8)*ksi**3 - 2.1793*10**(-5)*ksi**2 + 1.3798*10**(-3)*ksi)* S * self.Area * self.absorptivity * sAlbedo(t % OrbitalPeriod) * (np.cos(beta) * np.cos(Theta(t)))
    
    # Method to calculate infrared heat based on emissivity, albedo, and solar flux
    def InfraredHeat(self, S : float, albedo : float):
        ksi = np.arccos(np.cos(beta) * np.cos(Theta(t)))
        return self.F * S * (1-(albedo + 4.9115*10**(-9)*ksi**4 + 6.0372*10**(-8)*ksi**3 - 2.1793*10**(-5)*ksi**2 + 1.3798*10**(-3)*ksi)) / 4 * self.Area * self.emissivity
    
    # Method to calculate radiation heat based on area, emissivity, and temperature
    def RadiationHeat(self):
        return (self.Area * SBC * self.emissivity) * self.T**4

    # Method to calculate internal heat generation
    def InternalHeat(self):
        return self.Qgen

    # Method to calculate temperature change based on various heat sources
    def TemperatureChange(self, timestep : float, cp : float, m : float, S : float, beta : float, t:float,   albedo : float):
        self.Qdot = 0
        if self.Toggle[1] == 1: 
            self.Qdoti += self.AlbedoHeat(S, albedo)
        if self.Toggle[0] == 1:
            self.Qdoti += self.SolarHeat(S,t, beta)
        if self.Toggle[2] == 1:
           self.Qdoti +=  self.InfraredHeat(S,albedo)
        if self.Toggle[3] == 1:
            self.Qdoti += self.InternalHeat()
        if self.Toggle[4] == 1:
            self.Qdoti -= self.RadiationHeat()  
        self.Qdoti += self.HeatTransfer()   # Store the heat flux for this timestep
        self.Qdot = self.Qdoti  # Store the heat flux for this timestep
        self.T += timestep / (cp * m) * self.Qdoti
        self.Qdoti = 0.0  # Reset for next calculation

    # Method to return the current temperature of the node
    def __call__(self):
        return self.T
    
#--------------------------------------------------------------
#SolarPanel Nodes
#--------------------------------------------------------------

#Node(emissivity=emissivity,absorptivity=absorptivity,Neigbors=[ID, contact area, thermal conductivity], ID=ID, m=mass, cp=specific heat capacity, Area=area, F=form factor, Qgen=internal heat generation)
#Node.Toggle = [Solar, Albedo, Infrared, Internal, Radiation]

NodeEarth = Node(emissivity=emSol, absorptivity=alSol, Neigbors=[(6, CASFrame, 400)], ID=0, m=0.045 + 0.0056 + 0.003, cp=896, Area=Area, F=F, Qgen=0, Adjust=1) #Qgen = 0.007
NodeEarth.Toggle = [0, 1, 1, 0, 1]

NodeSun  = Node(emissivity=emSol, absorptivity=alSol, Neigbors=[(6, CATFrame, 400)], ID=1, m=0.045 + 0.0056 + 0.003, cp=896, Area=Area, F=0, Qgen=0, Adjust=1)
NodeSun.Toggle = [1, 0, 0, 0, 1]

Nodev = Node(emissivity=emSol, absorptivity=alSol, Neigbors=[(6, CASFrame, 400)], ID=2, m=0.045 + 0.0056*2 + 0.003, cp=896, Area=Area, F=FP, Qgen=0, Adjust=0)
Nodev.Toggle = [0, 1, 1, 0, 1]

Nodenv = Node(emissivity=emSol, absorptivity=alSol, Neigbors=[(6, CASFrame, 400)], ID=3, m=0.045 + 0.0056*2 + 0.003, cp=896, Area=Area, F=FP, Qgen=0, Adjust=0)
Nodenv.Toggle = [1, 1, 1, 0, 1]

NodeS = Node(emissivity=emSol, absorptivity=alSol, Neigbors=[(6, CASFrame, 400)], ID=4, m=0.045 + 0.0056*2 + 0.003, cp=896, Area=Area, F=FP, Qgen=0,BetaAdjust=0)
NodeS.Toggle = [0, 1, 1, 0, 1]

NodeN = Node(emissivity=emSol, absorptivity=alSol, Neigbors=[(15, 0.1*0.1, 400)], ID=5, m=0.045 + 0.0056*2 + 0.003, cp=896, Area=Area, F=FP, Qgen=0)
NodeN.Toggle = [0, 1, 1, 0, 1]

#--------------------------------------------------------------
# Connecting Rods
#--------------------------------------------------------------	

# NodeRod1 is initialized with emissivity=0 and absorptivity=0, making it thermally inert intentionally.
NodeRod1 = Node(emissivity=0, absorptivity=0, Neigbors=[
    (6, CARod*2, 400), (11, CASub, 666), (12, CASub, 666), (13, CASub, 666), (14, CASub, 666)
], ID=7, m=MassRod, cp=896, Area=0, F=FP, Qgen=0)
NodeRod1.Toggle = [0, 0, 0, 0, 0]

NodeRod2 = Node(emissivity=0, absorptivity=0, Neigbors=[
    (6, CARod*2, 400), (11, CASub, 666), (12, CASub, 666), (13, CASub, 666), (14, CASub, 666)
], ID=8, m=MassRod, cp=896, Area=0, F=FP, Qgen=0)
NodeRod2.Toggle = [0, 0, 0, 0, 0]

NodeRod3 = Node(emissivity=0, absorptivity=0, Neigbors=[
    (6, CARod*2, 400), (11, CASub, 666), (12, CASub, 666), (13, CASub, 666), (14, CASub, 666)
], ID=9, m=MassRod, cp=896, Area=0, F=FP, Qgen=0)
NodeRod3.Toggle = [0, 0, 0, 0, 0]

NodeRod4 = Node(emissivity=0, absorptivity=0, Neigbors=[
    (6, CARod*2, 400), (11, CASub, 666), (12, CASub, 666), (13, CASub, 666), (14, CASub, 666)
], ID=10, m=MassRod, cp=896, Area=0, F=FP, Qgen=0)
NodeRod4.Toggle = [0, 0, 0, 0, 0]

#--------------------------------------------------------------
# Subsystems
#--------------------------------------------------------------

NodeAnt = Node(emissivity=emAnt, absorptivity=aAnt, Neigbors=[
    (6, CATFrame, 400), (5, 0.1*0.1, 400)
], ID=15, m=0.030, cp=896, Area=AreaAnt, F=FP, Qgen=0)
NodeAnt.Toggle = [1, 0, 0, 0, 1]

NodeOBC = Node(emissivity=0, absorptivity=0, Neigbors=[
    (7, CASub, 666), (8, CASub, 666), (9, CASub, 666), (10, CASub, 666), (13,CABS, 666)
], ID=14, m=0.203, cp=750, Area=0, F=FP, Qgen=0.17*1.25)
NodeOBC.Toggle = [0, 0, 0, 1, 0]

NodePayloadII = Node(emissivity=0, absorptivity=0, Neigbors=[
    (7, CASub, 666), (8, CASub, 666), (9, CASub, 666), (10, CASub, 666), (14, CABS, 666), (12, CABS, 666)
], ID=13, m=0.074, cp=750, Area=0, F=FP, Qgen=0)#0.100)
NodePayloadII.Toggle = [0, 0, 0, 1, 0]

NodeEPS = Node(emissivity=0, absorptivity=0, Neigbors=[
    (7, CASub, 666), (8, CASub, 666), (9, CASub, 666), (10, CASub, 666), (11, CABS, 666), (13, CABS, 666)
], ID=12, m=0.086+0.005, cp=750, Area=0, F=FP, Qgen=0)
NodeEPS.Toggle = [0, 0, 0, 0, 0]

NodePayloadI = Node(emissivity=0, absorptivity=0, Neigbors=[
    (7, CASub, 666), (8, CASub, 666), (9, CASub, 666), (10, CASub, 666), (12, CABS, 666)
], ID=11, m=0.030 + 0.00226 + 0.051, cp=750, Area=0, F=FP, Qgen=0)#0.00026)
NodePayloadI.Toggle = [0, 0, 0, 1, 0]

NodeFrame = Node(emissivity=0, absorptivity=0.0, Neigbors=[
    (1, CATFrame, 400), (2, CASFrame, 400), (3, CASFrame, 400), (4, CASFrame, 400), (0, CASFrame, 400),
    (7, 2*CARod, 400), (8, 2*CARod, 400), (9, 2*CARod, 400), (10, 2*CARod, 400), (15, CATFrame, 400)
], ID=6, m=0.120 + 0.00169*2, cp=896, Area=AreaFrame, F=FP, Qgen=0)
NodeFrame.Toggle = [0, 0, 0, 0, 0]
Nodes = [NodeEarth, NodeSun, Nodev, Nodenv, NodeS, NodeN,NodeFrame, NodeRod1, NodeRod2, NodeRod3, NodeRod4,NodePayloadI,NodeEPS, NodePayloadII, NodeOBC, NodeAnt]
NodeTemps = {Node.ID: Node.T for Node in Nodes}
timestep = 1 # Time step [s]
timerange = np.arange(0,16*OrbitalPeriod, timestep) # Time range for simulation [s]

solar_toggles = [[] for _ in Nodes]

# Prepare storage for each node's temperature history
temps_all_nodes = [[] for _ in Nodes]
flux_all_nodes = [[] for _ in Nodes]
times = []
fe = FractionTime(beta)
trc_heating_intervals = [
    (4972.085, 5349.094),
    (10743.010, 10994.061),
    (65219.267, 65373.743),
    (70764.411, 71144.768),
    (77738.189, 78065.974),
    (82331.300, 82624.021)
]
betarange = np.linspace(-np.pi/4, np.pi/4, 20)  # Example: 20 beta angles from -45° to 45°
panel_count = 6  # Number of panels you want to track
activation_cube = np.zeros((len(betarange), len(times), panel_count))

for t in timerange:
    NodeOBC.Qgen = 0.17*1.25
    # Antenna heating logic
    for start, end in trc_heating_intervals:
        if start <= t <= end:
            NodeOBC.Qgen = 0.17*1.25#4
            NodeAnt.Qgen = 0#1
            break
    # Update toggles based on time
        if t % OrbitalPeriod > 0 :
            Nodev.Toggle[0] = 0
            NodeS.Toggle[0] = 1
            NodeSun.Toggle[0] = 1
            Nodenv.Toggle[0] = 1
        if t % OrbitalPeriod > OrbitalPeriod/4 :
            NodeS.Toggle[0] = 0
            NodeSun.Toggle[0] = 0
            NodeEarth.Toggle[0] = 1
        if t % OrbitalPeriod > OrbitalPeriod /2 :
            Nodenv.Toggle[0] = 0
            Nodev.Toggle[0] = 1
        if t % OrbitalPeriod > 3*OrbitalPeriod /4 :
            NodeEarth.Toggle[0] = 0
            NodeS.Toggle[0] = 1
            NodeSun.Toggle[0] = 1
    for idx, node in enumerate(Nodes[:6:]):
        if s(fe, t%OrbitalPeriod):
            solar_toggles[idx].append(node.Toggle[0])
        else:
            solar_toggles[idx].append(0)
    if t%(24*60*60) == 0:
            print(t//(24*60*60))
    # Update temperatures and heat transfer for each node
    for node in Nodes:
        node.WriteTemps([n() for n in Nodes])  # Update neighbor temperatures
    for idx, node in enumerate(Nodes):
        node.TemperatureChange(timestep, cp=node.cp, m=node.m, S=(1367/(r(theta(t))/astunit)**2), beta=beta, albedo=albedo(beta,mode), t=t)  # Update temperature
        temps_all_nodes[idx].append(node.T)
        flux_all_nodes[idx].append(node.Qdot)
    times.append(t)

# Give each node a name for plotting
node_names = [
    "[2] Earth (P)",      # NodeSun
    "[1] Sun (P)",       # NodeEarth
    "[6] Forward (P)",          # Nodev
    "[5] Backward (P)",          # Nodenv
    "[4] South (P)",       # NodeS
    "[3] North (P)",       # NodeN
    "Frame",       # NodeFrame
    "Rod UpLeft",        # NodeRod1
    "Rod UpRight",        # NodeRod2
    "Rod DownLeft",        # NodeRod3
    "Rod DownRight",        # NodeRod4
    "Standardized Payload",         # NodeEPS
    "EPS",        # NodeOBC
    "Validation Payload",   # NodePayloadI
    "OBC/TRC",  # NodePayloadII
    "Antenna"          # NodeTRC
]

# Choose which nodes to display by their indices or names
# Example: display_nodes = [0, 1, 2]
import matplotlib.pyplot as plt
import matplotlib as mp
display_nodes = [i for i in np.arange(0,16,1)]  # Change this list to select nodes by index
cmap = mp.colormaps.get_cmap('tab20')  # 20 unique colors

if not PLOT:
    plt.figure()
    for idx, node_idx in enumerate(display_nodes):
        label = node_names[node_idx] if node_idx < len(node_names) else f'Node {node_idx}'
        plt.plot(times/OrbitalPeriod, np.array(temps_all_nodes[node_idx])-273.15, label=label, color=cmap(idx))
    plt.xlabel('Orbits [-]',fontsize= 20)
    plt.ylabel('Temperature [°C]',fontsize= 20)
    #plt.title('Selected Node Temperatures Over Time')
    plt.xlim(6,7) 
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=14)  # 14 is the font size
    plt.savefig('Temperature_Variation.png')
    plt.show()

if PLOT:
    plt.figure()
    inter = np.zeros_like(times)
    for idx in display_nodes:
        label = node_names[idx] if idx < len(node_names) else f'Node {idx}'
        plt.plot(times/OrbitalPeriod, np.array(flux_all_nodes[idx]), label=label,color=cmap(idx))
        inter += np.array(flux_all_nodes[idx])
    plt.plot(times/OrbitalPeriod, inter, label='Total Heat Transfer', color='black', linestyle='--')
    plt.xlabel('Orbits [-]',fontsize= 20)
    plt.ylabel('Heat Transfer [W]',fontsize= 20)
    #plt.title('Selected Node Temperatures Over Time')
    plt.tick_params(axis='both', labelsize=14)  # 14 is the font size
    plt.xlim(6,7) 
    plt.legend()
    plt.grid(True)
    plt.savefig('Verification.png')
    plt.show()

total_mass = sum(node.m for node in Nodes)
print(f"Total mass of all components: {total_mass:.4f} kg")

if not PLOT:
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create a graph
    G = nx.Graph() # gay boy


    # Add nodes with labels
    for idx, node in enumerate(Nodes):
        G.add_node(idx, label=node_names[idx])

    # Add edges for each neighbor connection
    for idx, node in enumerate(Nodes):
        for neighbor in node.Neighbors:
            neighbor_id = neighbor[0]
            # Add edge only once (undirected graph)
            if not G.has_edge(idx, neighbor_id):
                G.add_edge(idx, neighbor_id)
    node_colors = [cmap(i) for i in range(len(Nodes))]

    pos = nx.circular_layout(G)
    # Assign a unique color to each edge based on the surce node
    edge_colors = []
    for u, v in G.edges(): 
        edge_colors.append(node_colors[u])

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos,
        with_labels=False,
        node_color=node_colors,
        node_size=1000,
        font_size=14,
        font_weight='bold',
        edge_color=edge_colors,
        width=2
    )

    for idx, name in enumerate(node_names):
        plt.scatter([], [], color=node_colors[idx], label=name, s=150)

    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
        title="Nodes",
        title_fontsize=15
    )
    #plt.title("Thermal Node Connection Graph (Unique Node & Edge Colors)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('Network')
    plt.show()

if PLOT:
    import matplotlib.pyplot as plt
    # Plot as a timeline
    plt.figure(figsize=(12, 6))
    for idx, node_idx in enumerate(display_nodes[:6:]):
        # Find intervals where the panel is activated
        active = np.array(solar_toggles[node_idx])
        # Find start and end indices of activation periods
        on_periods = np.where(active == 1)[0]
        if len(on_periods) == 0:
            continue
        # Group consecutive indices
        splits = np.split(on_periods, np.where(np.diff(on_periods) != 1)[0]+1)
        for group in splits:
            plt.hlines(idx, times[group[0]], times[group[-1]], colors=cmap(idx), linewidth=8)
    plt.yticks(range(len(display_nodes[:6:])), [node_names[i] for i in display_nodes[:6:]])
    plt.xlabel('Time [s]', fontsize = 14)
    #plt.title('Panel Solar Activation Timeline Over Orbit')
    plt.xlim(0, 1*OrbitalPeriod)  # Adjust x-axis limits as needed
    plt.grid(True, axis='x')
    plt.tick_params(axis='both', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)  # 14 is the font size
    plt.savefig('IloveGantt')
    plt.show()