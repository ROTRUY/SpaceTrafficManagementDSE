import numpy as np

U = 0.1*0.1 #m³
PassiveCooling =  0 #0.1*0.02 #m²
A_N = 9*U #m²
A_Z = 9*U #m²
A_S = 8*U #m²
alphaZ = 0.85
alphaN = 0.09
alphaS = 0.09
#  Emissivity
epsilonZ = 0.9
epsilonN = 0.90
epsilonS = 0.03
S = 1414 # W/m²
QEH = 257
QEC = 218
QA = 0.35*S
F = (6371)**2/(500+6371)**2
rho = np.arcsin(np.sqrt(F))
Ka = 0.664 + 0.521*rho - 0.203*rho**2
FP = 1/(2*np.pi)*( np.pi - 2*np.arcsin(np.sqrt(1-F)) - np.sin( 2*np.arcsin(np.sqrt(1-F))))
SBC = 5.6051*10**(-8) #W/m^2*K^4


SolarHeat = np.array([0,A_Z*alphaZ*S,0])
Albedo = np.array([A_N*F*(alphaN*QA)*Ka,0,A_S*FP*(alphaS*QA)])
InfraredH = np.array([A_N*F*(epsilonN*QEH),0,0])
InfraredC= np.array([A_N*F*(epsilonN*QEC),0,0])
Pint = np.array([3.3,3.3,3.3])

QinM = np.sum(SolarHeat+Albedo+InfraredH+Pint) - 0.3*S*A_Z
QinL = np.sum(InfraredC)

TemperatureMax = (QinM/((epsilonZ*A_Z+epsilonN*A_N + epsilonS*A_S)*SBC))**(1/4) -273.15
TemperatureMin = (QinL/((epsilonZ*A_Z+epsilonN*A_N + epsilonS*A_S)*SBC))**(1/4) -273.15
print(TemperatureMax)
print(TemperatureMin)