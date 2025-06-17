import numpy as np
import matplotlib.pyplot as plt

OrbitalHeight = 400
R = 6371
OrbitalPeriod = 5667
betaCrit = np.arcsin(R/(R+OrbitalHeight))



def FractionTime (beta,betaCrit = betaCrit,OrbitalHeight = OrbitalHeight):
    if  np.abs(beta) < betaCrit:
        return 1/np.pi*np.arccos(np.sqrt(OrbitalHeight**2 + 2*R*OrbitalHeight)/((R+OrbitalHeight)*np.cos(beta)))
    return 0

def albedo(beta):
    if np.abs(beta) < np.pi/6 :
        return 0.28
    return 0.30

def IR(beta):
    if np.abs(beta) < np.pi/6 :
        return 275
    return 257
def s(fe,t,OrbitalPeriod = OrbitalPeriod):
    if t < OrbitalPeriod/2*(1-fe) or t > OrbitalPeriod/2*(1+fe):
        return 1
    return 0

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
#AreaSp = 4*np.pi*EqRadius**2
beta = 0#-28 *np.pi/180
S = 1414
Qgen = 8
SBC = 5.6051*10**(-8) #W/m^2*K^4
Ti1 = 293.15
timestep = 1
cp = 896
m = 5
timerange = np.arange(0,16*OrbitalPeriod+1,timestep)
Temps = []

F = (6371)**2/(500+6371)**2
FP = 1/(2*np.pi)*( np.pi - 2*np.arcsin(np.sqrt(1-F)) - np.sin( 2*np.arcsin(np.sqrt(1-F))))


for i in timerange:
    Qin = F*IR(beta)*AreaRad*emRad + S*AreaSol*s(FractionTime(beta),i%OrbitalPeriod)*aSol  + F*(albedo(beta))*S*AreaAlb*s(FractionTime(beta),i%OrbitalPeriod)*aAlb + Qgen + FP*(albedo(beta))*S*AreaRem*s(FractionTime(beta),i%OrbitalPeriod)*aRem
    Qrad =  AreaSol*SBC*emSol*(Ti1)**4 + AreaAlb*SBC*emAlb*(Ti1)**4 + AreaRad*SBC*emRad*(Ti1)**4 + AreaRem*SBC*emRem*(Ti1)**4
    Qdoti = Qin - Qrad
    T = Ti1 + timestep/(cp*m) * Qdoti
    Temps.append(T)
    Ti1 = T


plt.plot(timerange/OrbitalPeriod, np.array(Temps)-273.15, color='r', label='Hot Case')

# Second simulation
def albedo(beta):
    if beta < np.pi/6 :
        return 0.18
    return 0.23

def IR(beta):
    if beta < np.pi/6 :
        return 228
    return 218

S = 1322
Ti1 = 293.15
Qgen = 0
Temps = []

for i in timerange:
    Qin = F*IR(beta)*AreaRad*emRad + S*AreaSol*s(FractionTime(beta), i%OrbitalPeriod)*aSol + F*(albedo(beta))*S*AreaAlb*s(FractionTime(beta), i%OrbitalPeriod)*aAlb + Qgen + FP*(albedo(beta))*S*AreaRem*s(FractionTime(beta), i%OrbitalPeriod)*aRem 
    Qrad = AreaSol*SBC*emSol*(Ti1)**4 + AreaAlb*SBC*emAlb*(Ti1)**4 + AreaRad*SBC*emRad*(Ti1)**4 + AreaRem*SBC*emRem*(Ti1)**4
    Qdoti = Qin - Qrad
    T = Ti1 + timestep/(cp*m) * Qdoti
    Temps.append(T)
    Ti1 = T

plt.plot(timerange/OrbitalPeriod, np.array(Temps)-273.15, color='b', label='Cold Case')

# Third simulation
def albedo(beta):
    if beta < np.pi/6 :
        return 0.23
    return 0.265

def IR(beta):
    if beta < np.pi/6 :
        return 250
    return 237.5

S = 1367
Ti1 = 293.15
Qgen = 3
Temps = []

for i in timerange:
    Qin = F*IR(beta)*AreaRad*emRad + S*AreaSol*s(FractionTime(beta), i%OrbitalPeriod)*aSol + F*(albedo(beta))*S*AreaAlb*s(FractionTime(beta), i%OrbitalPeriod)*aAlb + Qgen + FP*(albedo(beta))*S*AreaRem*s(FractionTime(beta), i%OrbitalPeriod)*aRem 
    Qrad = AreaSol*SBC*emSol*(Ti1)**4 + AreaAlb*SBC*emAlb*(Ti1)**4 + AreaRad*SBC*emRad*(Ti1)**4 + AreaRem*SBC*emRem*(Ti1)**4
    Qdoti = Qin - Qrad
    T = Ti1 + timestep/(cp*m) * Qdoti
    Temps.append(T)
    Ti1 = T

plt.plot(timerange/OrbitalPeriod, np.array(Temps)-273.15, color='g', label='Mild Case')

# Add labels and legend
plt.xlabel('Orbits')
plt.ylabel('Temperature [°C]')
#plt.title('Temperature Evolution Over 24 Hours')
plt.legend()
plt.grid(True)
plt.show()


