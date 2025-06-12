import numpy as np
import plotly.graph_objects as go

#--------------------------------------------------------------
# Orbital parameters
#--------------------------------------------------------------

mode = 'M'
OrbitalHeight = 400  # [km] 
R = 6371             # [km] Radius of Earth
OrbitalPeriod = 2 * np.pi * np.sqrt(((OrbitalHeight+R)*1000)**3 / (3.986004418*10**14)) # [s]
betaCrit = np.arcsin(R/(R+OrbitalHeight))  # [rad]
# Note: betaCrit is the critical angle where the orbital height is tangent to the Earth's surface

def FractionTime(beta : float, betaCrit=betaCrit, OrbitalHeight=OrbitalHeight ):
    if np.abs(beta) < betaCrit:
        return 1/np.pi * np.arccos(np.sqrt(OrbitalHeight**2 + 2*R*OrbitalHeight)/((R+OrbitalHeight)*np.cos(beta)))
    return 0
beta = 0 #-26.5 * np.pi / 180  # Example beta angle in radians
fe = FractionTime(beta)
SBC = 5.67e-8  # Stefan-Boltzmann constant [W/m²*K⁴]

class Node:
    def __init__(self, k : float, emissivity : float, absorptivity : float, Neigbors : list, ID : float, m : float, cp : float, Area : float, Qgen :float, FE : float = 1.0, Adjust : bool = 1):
        self.ContactConductance = k # W/m²*K
        self.emissivity = emissivity 
        self.absorptivity = absorptivity
        self.m = m
        self.cp = cp
        self.Neighbors = Neigbors
        self.T = 273.15  # Initial temperature from neighbor temperatures
        self.ID = ID
        self.Qdoti = 0.0
        self.FE = FE
        self.Area = Area
        self.Qgen = Qgen
        self.Toggle = [0,0,0,0,0]  # Toggle for different heat sources: [Solar, Albedo, Infrared, Internal, Radiation]
        self.Adjust = Adjust  # Adjustment for solar angle

    
    def WriteTemps (self,Nodetemps):
        self.NeighborTemperatures = [Nodetemps[j] for j in [i[0] for i in self.Neighbors]]  # Update neighbor temperatures from the provided list

    def HeatTransfer(self):
        inter = 0.0
        for i,e in enumerate(self.NeighborTemperatures):
            inter += self.ContactConductance * (e - self.T) * self.Neighbors[i][1]  # Heat transfer from neighbors
        return inter

    def SolarHeat(self, S : float,t : float, beta : float = beta):
        if self.Adjust == 1:
            return np.abs((S * self.Area * self.absorptivity ) *np.cos(Theta(t)) * (np.cos(beta)) *  s(fe, t % OrbitalPeriod))
        else:
            return np.abs((S * self.Area * self.absorptivity ) *np.sin(Theta(t)) * (np.cos(beta)) *  s(fe, t % OrbitalPeriod))


    def AlbedoHeat(self, S : float, albedo : float):
        ksi = np.arccos(np.cos(beta) * np.cos(Theta(t)))
        return self.FE * (albedo + 4.9115*10**(-9)*ksi**4 + 6.0372*10**(-8)*ksi**3 - 2.1793*10**(-5)*ksi**2 + 1.3798*10**(-3)*ksi)* S * self.Area * self.absorptivity * sAlbedo(t % OrbitalPeriod) * (np.cos(beta) * np.cos(Theta(t)))
    
    def InfraredHeat(self, albedo : float):
        return self.FE * S * (1-albedo) / 4 * self.Area * self.emissivity
    

    def RadiationHeat(self):
        return (self.Area * SBC * self.emissivity) * self.T**4

    def InternalHeat(self):
        return self.Qgen

    def TemperatureChange(self, timestep : float, cp : float, m : float, S : float, beta : float, t:float,   albedo : float, IR: float):
        self.Qdot = 0
        if self.Toggle[1] == 1: 
            self.Qdoti += self.AlbedoHeat(S, albedo)
        if self.Toggle[0] == 1:
            self.Qdoti += self.SolarHeat(S,t, beta)
        if self.Toggle[2] == 1:
           self.Qdoti +=  self.InfraredHeat(albedo)
        if self.Toggle[3] == 1:
            self.Qdoti += self.InternalHeat()
        if self.Toggle[4] == 1:
            self.Qdoti -= self.RadiationHeat()  
        self.Qdot = self.Qdoti
        self.Qdoti += self.HeatTransfer()   # Store the heat flux for this timestep
        self.T += timestep / (cp * m) * self.Qdoti
        self.Qdoti = 0.0  # Reset for next calculation

    def __call__(self):
        return self.T


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
def Theta(t, OrbitalPeriod=OrbitalPeriod):
        return 2*np.pi/OrbitalPeriod*(t%OrbitalPeriod)
def sAlbedo(t, OrbitalPeriod=OrbitalPeriod):
    if t < OrbitalPeriod/4 or t > 3*OrbitalPeriod/4:
        return 1
    return 0

# Constants
emSol = 0.825
alSol= 0.805
Area = 0.1*0.1  # m²
AreaFrame = 0.008*0.008*8 + 0.008*0.112*8  # m², area of the frame
CAFrame = 0.08*(0.08-0.005)
CARod = 0.0032*np.pi*0.0023
MassRod = (0.0032/2)**2*0.083*2712  # kg, mass of the rod
CASub = 0.0032*np.pi*0.001623 # m², area of the substructure
CABS = 0.066*0.01 # m², area of the Pin Connectors


F = (R)**2/(OrbitalHeight+R)**2
FP = 1/(2*np.pi)*( np.pi - 2*np.arcsin(np.sqrt(1-F)) - np.sin( 2*np.arcsin(np.sqrt(1-F))))
  # Example factor for solar flux reduction

#SolarPanel Nodes

NodeSun  = Node(k=237, emissivity=emSol, absorptivity=alSol, Neigbors=[(6,CAFrame)], ID = 0, m=0.044, cp=896, Area=Area,FE=0,Qgen=0,Adjust = 1)
NodeSun.Toggle = [1, 0, 0, 0, 1]  # Toggle for different heat sources: [Solar, Albedo, Infrared, Internal, Radiation]

NodeEarth = Node(k=237, emissivity=emSol, absorptivity=alSol,Neigbors=[(7,0.1*0.1)], ID  = 1, m=0.044, cp=896, Area=Area,FE=F,Qgen=0,Adjust= 1)
NodeEarth.Toggle = [0, 1, 1, 0, 1] 

Nodev = Node(k=237, emissivity=emSol, absorptivity=alSol,Neigbors=[(6,CAFrame)], ID = 2, m=0.044, cp=896, Area=Area,FE=FP,Qgen=0,Adjust= 0)
Nodev.Toggle = [0, 1, 1, 0, 1]

Nodenv = Node(k=237, emissivity=emSol, absorptivity=alSol,Neigbors=[(6,CAFrame)], ID = 3, m=0.044, cp=896, Area=Area,FE=FP,Qgen=0,Adjust = 0)
Nodenv.Toggle = [1, 1, 1, 0, 1]

NodeS = Node(k=237, emissivity=emSol, absorptivity=alSol, Neigbors=[(6,CAFrame)], ID = 4, m=0.044, cp=896, Area=Area,FE=FP,Qgen=0)
NodeS.Toggle = [0, 1, 1, 0, 1]

NodeN = Node(k=237, emissivity=emSol, absorptivity=alSol,Neigbors=[(6,CAFrame)], ID  = 5, m=0.044, cp=896, Area=Area,FE=FP,Qgen=0)
NodeN.Toggle = [0, 1, 1, 0, 1]

NodeAnt = Node(k = 237, emissivity=0, absorptivity=0, Neigbors=[
(6, CAFrame), (1, 0.1*0.1)], ID=7, m=0.089, cp=896, Area=0, FE=FP, Qgen=0.040)
NodeAnt.Toggle = [0, 0, 0, 1, 0]

NodeFrame = Node(k=237, emissivity=0, absorptivity=0.0, Neigbors=[(7,CAFrame),(0,CAFrame),(1,CAFrame),(2,CAFrame),(3,CAFrame),(4,CAFrame),(5,CAFrame)], ID = 6, m=(1.1-0.044*6), cp=896, Area=AreaFrame,FE=FP,Qgen=1.05)
#(5,CAFrame)
NodeFrame.Toggle = [0, 0, 0, 1, 0]

Nodes = [NodeSun, NodeEarth, Nodev, Nodenv, NodeS, NodeN,NodeFrame,NodeAnt]
NodeTemps = {Node.ID: Node.T for Node in Nodes}
timestep = 1 # Time step [s]
timerange = np.arange(0,4*OrbitalPeriod + 1, timestep) # Time range for simulation [s]

import matplotlib.pyplot as plt

# Prepare storage for each node's temperature history
temps_all_nodes = [[] for _ in Nodes]
flux_all_nodes = [[] for _ in Nodes]
times = []
S = 1367
for t in timerange:
    # Update toggles based on time
    if t % OrbitalPeriod > 0 :
        Nodenv.Toggle[0] = 1
        Nodev.Toggle[0] = 0
        NodeSun.Toggle[0] = 1
    if t % OrbitalPeriod > OrbitalPeriod/4 :
        NodeSun.Toggle[0] = 0
        NodeEarth.Toggle[0] = 1
    if t % OrbitalPeriod > OrbitalPeriod /2 :
        Nodenv.Toggle[0] = 0
        Nodev.Toggle[0] = 1
    if t % OrbitalPeriod > 3*OrbitalPeriod /4 :
        NodeEarth.Toggle[0] = 0
        NodeSun.Toggle[0] = 1

    # Update temperatures and heat transfer for each node
    for node in Nodes:
        node.WriteTemps([n() for n in Nodes])  # Update neighbor temperatures
    for idx, node in enumerate(Nodes):
        node.TemperatureChange(timestep, cp=node.cp, m=node.m, S=S, beta=beta, albedo=albedo(beta), IR=IR(beta), t=t)  # Update temperature
        temps_all_nodes[idx].append(node.T)
        flux_all_nodes[idx].append(node.Qdot)
    times.append(t)

# Give each node a name for plotting
node_names = [
    "Sun",      # NodeSun
    "Earth",       # NodeEarth
    "+V",          # Nodev
    "-V",          # Nodenv
    "South",       # NodeS
    "North",       # NodeN
    "Frame",       # NodeFrame
    "Rod1",        # NodeRod1
    "Rod2",        # NodeRod2
    "Rod3",        # NodeRod3
    "Rod4",        # NodeRod4
    "EPS",         # NodeEPS
    "OBC",         # NodeOBC
    "Payload I",   # NodePayloadI
    "Payload II",  # NodePayloadII
    "Antenna",     # NodeAnt
    "TRC"          # NodeTRC
]

# Choose which nodes to display by their indices or names
# Example: display_nodes = [0, 1, 2] or display_nodes = ["Zenith", "Earth"]
display_nodes = [i for i in np.arange(0,8,1)]  # Change this list to select nodes by index

# If you want to select by name, uncomment below and comment the above line:
# display_node_names = ["Zenith", "Earth", "Frame"]
# display_nodes = [node_names.index(name) for name in display_node_names]

plt.figure()
cmap = plt.cm.get_cmap('tab20', len(display_nodes))  # 20 unique colors, or increase as needed
for idx, node_idx in enumerate(display_nodes):
    label = node_names[node_idx] if node_idx < len(node_names) else f'Node {node_idx}'
    plt.plot(times, np.array(temps_all_nodes[node_idx])-273.15, label=label, color=cmap(idx))
plt.xlabel('Time [s]')
plt.ylabel('Temperature [°C]')
plt.title('Selected Node Temperatures Over Time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
inter = np.zeros_like(times)
for idx in display_nodes:
    label = node_names[idx] if idx < len(node_names) else f'Node {idx}'
    plt.plot(times, np.array(flux_all_nodes[idx]), label=label,color=cmap(idx))
    inter += np.array(flux_all_nodes[idx])
plt.plot(times, inter, label='Total Heat Transfer', color='black', linestyle='--')
print(sum(inter))
plt.xlabel('Time [s]')
plt.ylabel('Heat Transfer [W]')
plt.title('Selected Node Temperatures Over Time')
plt.xlim(0,4*OrbitalPeriod)   
#plt.axvline(x=OrbitalPeriod/2*(1-fe))
#plt.axvline(x=OrbitalPeriod/4)
#plt.axvline(x=3*OrbitalPeriod/4)
#plt.axvline(x=OrbitalPeriod/2*(1+fe))
plt.legend()
plt.grid(True)
plt.show()