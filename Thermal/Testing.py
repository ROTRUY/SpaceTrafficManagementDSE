import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Mode of operation
# --------------------------------------------------------------
mode = 'C'  # 'H' for High, 'M' for Medium, 'C' for Cold

# --------------------------------------------------------------
# Sun parameters
# --------------------------------------------------------------
semimajor = 149.598e9
ecc = 0.0167
astunit = 1.495978707 * 10 ** 11


def theta(t):
    """True‐anomaly approximation (first-order in e)."""
    M = 2 * np.pi * (
        t + (365.256 * 24 * 60 * 60 / 2)
    ) / (365.256 * 24 * 60 * 60)
    return M + (2 * ecc * np.sin(M))


def r_fcn(th):
    return semimajor * (1 - ecc**2) / (1 + ecc * np.cos(th))


# --------------------------------------------------------------
# Orbital parameters
# --------------------------------------------------------------
OrbitalHeight = 500  # [km]
R = 6371  # [km] Earth radius
OrbitalPeriod = 2 * np.pi * np.sqrt(
    ((OrbitalHeight + R) * 1000) ** 3 / (3.986004418 * 10**14)
)  # [s]

F = (R) ** 2 / (OrbitalHeight + R) ** 2
FP = 1 / (2 * np.pi) * (
    np.pi
    - 2 * np.arcsin(np.sqrt(1 - F))
    - np.sin(2 * np.arcsin(np.sqrt(1 - F)))
)
betaCrit = np.arcsin(R / (R + OrbitalHeight))  # [rad]

# --------------------------------------------------------------
# Material & geometry constants
# --------------------------------------------------------------
emSol = 0.825
alSol = 0.805
Area = 0.1 * 0.1

AreaFrame = 0.008 * 0.008 * 8 + 0.008 * 0.112 * 8
CASFrame = (0.0015 * 0.0684) * 2 + (0.0828 * 0.0122) * 2
CATFrame = (0.0825 * 0.005) * 4

CARod = 0.0032 * np.pi * 0.0023
MassRod = (0.0032 / 2) ** 2 * 0.083 * 2712
CASub = 0.0032 * np.pi * 0.001623

CABS = 0.066 * 0.01
AreaAnt = 0.163 * 0.003 * np.pi * 4
emAnt = 0.2
aAnt = 0.4

SBC = 5.67e-8  # Stefan–Boltzmann

# --------------------------------------------------------------
# Helper functions that depend on beta (β) – β will be supplied
# --------------------------------------------------------------


def FractionTime(beta: float):
    """Fraction of the orbit in shadow."""
    if np.abs(beta) < betaCrit:
        return 1 / np.pi * np.arccos(
            np.sqrt(OrbitalHeight**2 + 2 * R * OrbitalHeight)
            / ((R + OrbitalHeight) * np.cos(beta))
        )
    return 0.0


def albedo(beta, mode=mode):
    if mode == "H":
        return 0.28 if np.abs(beta) < np.pi / 6 else 0.30
    elif mode == "M":
        return 0.23 if np.abs(beta) < np.pi / 6 else 0.265
    elif mode == "C":
        return 0.18 if np.abs(beta) < np.pi / 6 else 0.23


def s(fe, t, Orb=OrbitalPeriod):
    """Sun-in/out binary flag."""
    if t < Orb / 2 * (1 - fe) or t > Orb / 2 * (1 + fe):
        return 1
    return 0


def Theta(t, Orb=OrbitalPeriod):
    return 2 * np.pi / Orb * (t % Orb)


def sAlbedo(t, Orb=OrbitalPeriod):
    return 1 if (t < Orb / 4 or t > 3 * Orb / 4) else 0


# --------------------------------------------------------------
# Thermal node class (unchanged)
# --------------------------------------------------------------
class Node:
    def __init__(
        self,
        emissivity: float,
        absorptivity: float,
        Neigbors: list,
        ID: float,
        m: float,
        cp: float,
        Area: float,
        Qgen: float,
        F: float,
        Adjust: bool = 1,
        BetaAdjust: bool = 1,
    ):
        self.emissivity = emissivity
        self.absorptivity = absorptivity
        self.m = m
        self.cp = cp
        self.Neighbors = Neigbors
        self.T = 273.15
        self.ID = ID
        self.F = F
        self.Area = Area
        self.Qgen = Qgen

        # bookkeeping
        self.NeighborTemperatures = [273.15] * len(self.Neighbors)
        self.Toggle = [0, 0, 0, 0, 0]
        self.Adjust = Adjust
        self.BetaAdjust = BetaAdjust
        self.Qdoti = 0.0

    # --- helper methods (unchanged except for beta argument) ---
    def WriteTemps(self, Nodetemps):
        self.NeighborTemperatures = [
            Nodetemps[j] for j in [i[0] for i in self.Neighbors]
        ]

    def HeatTransfer(self):
        inter = 0.0
        for i, e in enumerate(self.NeighborTemperatures):
            inter += (
                self.Neighbors[i][2]
                * (e - self.T)
                * self.Neighbors[i][1]
            )
        return inter

    def SolarHeat(self, S: float, t: float, beta: float):
        if self.BetaAdjust == 1:
            if self.Adjust == 1:
                return abs(
                    (S * self.Area * self.absorptivity)
                    * np.cos(Theta(t))
                    * np.cos(beta)
                    * s(fe, t % OrbitalPeriod)
                )
            else:
                return abs(
                    (S * self.Area * self.absorptivity)
                    * np.sin(Theta(t))
                    * s(fe, t % OrbitalPeriod)
                )
        # opposite side
        if self.Adjust == 1:
            return abs(
                (S * self.Area * self.absorptivity)
                * np.cos(Theta(t))
                * np.sin(beta)
                * s(fe, t % OrbitalPeriod)
            )
        else:
            return abs(
                (S * self.Area * self.absorptivity)
                * np.sin(Theta(t))
                * np.sin(beta)
                * s(fe, t % OrbitalPeriod)
            )

    def AlbedoHeat(self, S: float, alb: float, t: float, beta: float):
        ksi = np.arccos(np.cos(beta) * np.cos(Theta(t)))
        return (
            self.F
            * (
                alb
                + 4.9115e-9 * ksi**4
                + 6.0372e-8 * ksi**3
                - 2.1793e-5 * ksi**2
                + 1.3798e-3 * ksi
            )
            * S
            * self.Area
            * self.absorptivity
            * sAlbedo(t % OrbitalPeriod)
            * (np.cos(beta) * np.cos(Theta(t)))
        )

    def InfraredHeat(self, S: float, alb: float, t: float, beta: float):
        ksi = np.arccos(np.cos(beta) * np.cos(Theta(t)))
        return (
            self.F
            * S
            * (
                1
                - (
                    alb
                    + 4.9115e-9 * ksi**4
                    + 6.0372e-8 * ksi**3
                    - 2.1793e-5 * ksi**2
                    + 1.3798e-3 * ksi
                )
            )
            / 4
            * self.Area
            * self.emissivity
        )

    def RadiationHeat(self):
        return (self.Area * SBC * self.emissivity) * self.T**4

    def InternalHeat(self):
        return self.Qgen

    def TemperatureChange(
        self,
        timestep: float,
        cp: float,
        m: float,
        S: float,
        beta: float,
        alb: float,
        t: float,
    ):
        self.Qdoti = 0
        if self.Toggle[1]:
            self.Qdoti += self.AlbedoHeat(S, alb, t, beta)
        if self.Toggle[0]:
            self.Qdoti += self.SolarHeat(S, t, beta)
        if self.Toggle[2]:
            self.Qdoti += self.InfraredHeat(S, alb, t, beta)
        if self.Toggle[3]:
            self.Qdoti += self.InternalHeat()
        if self.Toggle[4]:
            self.Qdoti -= self.RadiationHeat()
        self.Qdoti += self.HeatTransfer()

        self.T += timestep / (cp * m) * self.Qdoti

    def __call__(self):
        return self.T


# --------------------------------------------------------------
# Build an entire satellite (fresh nodes each run)
# --------------------------------------------------------------
def build_nodes():
    # Panels
    NodeEarth = Node(
        emSol,
        alSol,
        [(6, CASFrame, 400)],
        0,
        0.035,
        896,
        Area,
        0,
        F,
        Adjust=1,
    )
    NodeEarth.Toggle = [0, 1, 1, 0, 1]

    NodeSun = Node(
        emSol, alSol, [(6, CATFrame, 400)], 1, 0.035, 896, Area, 0, 0, Adjust=1
    )
    NodeSun.Toggle = [1, 0, 0, 0, 1]

    Nodev = Node(
        emSol, alSol, [(6, CASFrame, 400)], 2, 0.035, 896, Area, 0, FP, Adjust=0
    )
    Nodev.Toggle = [0, 1, 1, 0, 1]

    Nodenv = Node(
        emSol, alSol, [(6, CASFrame, 400)], 3, 0.035, 896, Area, 0, FP, Adjust=0
    )
    Nodenv.Toggle = [1, 1, 1, 0, 1]

    NodeS = Node(
        emSol,
        alSol,
        [(6, CASFrame, 400)],
        4,
        0.038,
        896,
        Area,
        0,
        FP,
        BetaAdjust=0,
    )
    NodeS.Toggle = [0, 1, 1, 0, 1]

    NodeN = Node(
        emSol,
        alSol,
        [(15, 0.1 * 0.1, 400)],
        5,
        0.038,
        896,
        Area,
        0,
        FP,
    )
    NodeN.Toggle = [0, 1, 1, 0, 1]

    # Frame
    NodeFrame = Node(
        0,
        0.0,
        [
            (1, CASFrame, 400),
            (2, CASFrame, 400),
            (3, CASFrame, 400),
            (4, CASFrame, 400),
            (0, CASFrame, 400),
        ],
        6,
        0.130,
        896,
        AreaFrame,
        0,
        FP,
    )
    NodeFrame.Toggle = [0, 0, 0, 0, 0]

    # Rods
    RodNeigh = [
        (6, CARod * 2, 400),
        (11, CASub, 400),
        (12, CASub, 400),
        (13, CASub, 400),
        (14, CASub, 400),
    ]
    NodeRod1 = Node(0, 0, RodNeigh, 7, MassRod, 896, 0, 0, FP)
    NodeRod2 = Node(0, 0, RodNeigh, 8, MassRod, 896, 0, 0, FP)
    NodeRod3 = Node(0, 0, RodNeigh, 9, MassRod, 896, 0, 0, FP)
    NodeRod4 = Node(0, 0, RodNeigh, 10, MassRod, 896, 0, 0, FP)
    for r in (NodeRod1, NodeRod2, NodeRod3, NodeRod4):
        r.Toggle = [0, 0, 0, 0, 0]

    # Sub-systems
    NodePayloadI = Node(
        0,
        0,
        [
            (7, CASub, 400),
            (8, CASub, 400),
            (9, CASub, 400),
            (10, CASub, 400),
            (12, CABS, 400),
        ],
        11,
        0.031,
        896,
        0,
        0.00026,
        FP,
    )
    NodePayloadI.Toggle = [0, 0, 0, 1, 0]

    NodeEPS = Node(
        0,
        0,
        [
            (7, CASub, 400),
            (8, CASub, 400),
            (9, CASub, 400),
            (10, CASub, 400),
            (11, CABS, 400),
            (13, CABS, 400),
        ],
        12,
        0.086,
        896,
        0,
        0,
        FP,
    )

    NodePayloadII = Node(
        0,
        0,
        [
            (7, CASub, 400),
            (8, CASub, 400),
            (9, CASub, 400),
            (10, CASub, 400),
            (14, CABS, 400),
            (12, CABS, 400),
        ],
        13,
        0.024,
        896,
        0,
        0.125,
        FP,
    )
    NodePayloadII.Toggle = [0, 0, 0, 1, 0]

    NodeOBC = Node(
        0,
        0,
        [
            (7, CASub, 400),
            (8, CASub, 400),
            (9, CASub, 400),
            (10, CASub, 400),
            (15, 2 * CABS, 400),
            (13, CABS, 400),
        ],
        14,
        0.159,
        896,
        0,
        0.17 * 1.25,
        FP,
    )
    NodeOBC.Toggle = [0, 0, 0, 1, 0]

    NodeAnt = Node(
        emAnt,
        aAnt,
        [(6, CATFrame, 400), (5, 0.1 * 0.1, 400), (14, 2 * CABS, 400)],
        15,
        0.030,
        896,
        AreaAnt,
        0,
        FP,
    )
    NodeAnt.Toggle = [1, 0, 0, 0, 1]

    # final list
    return [
        NodeEarth,
        NodeSun,
        Nodev,
        Nodenv,
        NodeS,
        NodeN,
        NodeFrame,
        NodeRod1,
        NodeRod2,
        NodeRod3,
        NodeRod4,
        NodePayloadI,
        NodeEPS,
        NodePayloadII,
        NodeOBC,
        NodeAnt,
    ]


# readable names (unchanged order)
node_names = [
    "Earth (P)",
    "Sun (P)",
    "+V (P)",
    "-V (P)",
    "South (P)",
    "North (P)",
    "Frame",
    "Rod1",
    "Rod2",
    "Rod3",
    "Rod4",
    "Payload I",
    "EPS",
    "Payload II",
    "OBC/TRC",
    "Antenna",
]

# --------------------------------------------------------------
# Single-orbit simulator (returns histories)
# --------------------------------------------------------------
timestep = 1.0  # [s]
timerange = np.arange(0, 16 * OrbitalPeriod, timestep)


def run_orbit(beta: float):
    global fe  # required by s()
    fe = FractionTime(beta)
    alb = albedo(beta)

    Nodes = build_nodes()
    temps_all = [[] for _ in Nodes]
    flux_all = [[] for _ in Nodes]
    times = []

    # Antenna heating profile
    trc_heating_intervals = [
        (4972.085, 5349.094),
        (10743.010, 10994.061),
        (65219.267, 65373.743),
        (70764.411, 71144.768),
        (77738.189, 78065.974),
        (82331.300, 82624.021),
    ]

    for t in timerange:
        # TRC heater logic (mutated Qgen live)
        Nodes[14].Qgen = 0.17 * 1.25  # OBC default
        Nodes[15].Qgen = 0
        for start, end in trc_heating_intervals:
            if start <= t <= end:
                Nodes[14].Qgen = 4
                break

        # Solar toggle schedule (identical to original, kept concise)
        if t % OrbitalPeriod > 0:
            Nodes[2].Toggle[0] = 0
            Nodes[4].Toggle[0] = 1
            Nodes[1].Toggle[0] = 1
            Nodes[3].Toggle[0] = 1
        if t % OrbitalPeriod > OrbitalPeriod / 4:
            Nodes[4].Toggle[0] = 0
            Nodes[1].Toggle[0] = 0
            Nodes[0].Toggle[0] = 1
        if t % OrbitalPeriod > OrbitalPeriod / 2:
            Nodes[3].Toggle[0] = 0
            Nodes[2].Toggle[0] = 1
        if t % OrbitalPeriod > 3 * OrbitalPeriod / 4:
            Nodes[0].Toggle[0] = 0
            Nodes[4].Toggle[0] = 1
            Nodes[1].Toggle[0] = 1

        # neighbour temps
        for node in Nodes:
            node.WriteTemps([n() for n in Nodes])

        # update temps
        for idx, node in enumerate(Nodes):
            S = 1367 / (r_fcn(theta(t)) / astunit) ** 2
            node.TemperatureChange(
                timestep,
                cp=node.cp,
                m=node.m,
                S=S,
                beta=beta,
                alb=alb,
                t=t,
            )
            temps_all[idx].append(node.T)
            flux_all[idx].append(node.Qdoti)
        times.append(t)

    return np.array(times), temps_all, flux_all
# --------------------------------------------------------------
#  INPUTS you can easily tweak
# --------------------------------------------------------------
betarange = np.linspace(0,betaCrit, 20)    # β sweep
display_nodes = [15]      # indices in node_names you want to see

# --------------------------------------------------------------
#  SWEEP over β – store temps into a single cube
# --------------------------------------------------------------
times, dummy_temps, _ = run_orbit(betarange[0])       # just to size arrays
nTimes   = len(times)
nNodes   = len(dummy_temps)
nBetas   = len(betarange)

temp_cube = np.zeros((nBetas, nNodes, nTimes))        # β × node × time

for bidx, beta in enumerate(betarange):
    _, temps_list, _ = run_orbit(beta)
    temp_cube[bidx, :, :] = np.array(temps_list)      # K

# --------------------------------------------------------------
#  3-D plotting helpers
# --------------------------------------------------------------
import plotly.graph_objects as go

# Assume 'temp_cube' is shape (n_beta, n_time) in °C
# 'betarange' is in radians, 'times' is in seconds

# Convert for plotting
Tg, Bg = np.meshgrid(times / 3600, np.degrees(betarange))  # Time [h], β [deg]

node_idx = display_nodes[0]  # or any node you want to plot
z = temp_cube[:, node_idx, :] - 273.15  # K to °C
fig = go.Figure(data=[go.Surface(
    z=z,
    x=Tg,
    y=Bg,
    colorscale='Viridis',
    colorbar=dict(title='T [°C]'),
)])

fig.update_layout(
    title="Panel temperature – β sweep (Plotly)",
    scene=dict(
        xaxis_title='Time [h]',
        yaxis_title='β [deg]',
        zaxis_title='Temperature [°C]',
    ),
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=40)
)

fig.show()
