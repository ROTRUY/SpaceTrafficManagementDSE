
#Imports
import numpy as np

#Inputs
# Number of U Units
N = 1 
#Iradiance From Planet
IR = 500
#Area Exposed to IR
AIR = 0.01*N #m²
# Solar irradiance reflected by Earth
R = 0.5
# U Units
AR = 0.01*N #m² 
# Solar Constant
S = 1366 # W/m²
# Absorptivity
alpha = 0.85
#  Emissivity
epsilon = 0.9
# Projected Area
Ap = 0.01*N #m²
# Radiating Surface Area
Ar = 0.01*N*6 #m² must be corrected
# Stefan-Boltzmann Constant
SBC = 5.6051*10**(-8) #W/m^2*K^4

#Calculation
#Absolute Temperature
#T = ((S*(alpha/epsilon)*(Ap/Ar)/SBC) )**(1/4) - 273.15

T = ((S*alpha*Ap+5)/(epsilon*Ar*SBC)) ** (1/4) -273.15
# Environmental Heat
Qenv = alpha*S*(Ap + R*AR) + epsilon*IR*AIR

print(T)
