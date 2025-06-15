from math import *
import matplotlib.pyplot as plt

def magneticstuff_Quetzal():
    Bs = 0.45  # From paper
    a0 = 1.02  # From paper
    k0 = 5.0*1000  # From paper
    eta = 12  # From paper
    m = 1.97  # From paper
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ha = 25  # Magnetic field strength of earth max
    kw = 0.6  # From paper

    # ASSUMED VALUES
    e = 95  # elongation
    L = 9.5e-2  # m
    D = 1.00e-3  # m
    V = L * pi * (D/2)**2
    n = 1 #number of rods

    Nd = (4.02 * log10(e) - 0.185) / 2 / e**2
    Hmax = (-(Bs - mu0 / Nd * Ha) + sqrt((Bs - mu0 / Nd * Ha)**2 + 4 * (k0 + mu0 / Nd) * a0 * Bs)) / (2 * (k0 + mu0 / Nd))
    
    Bmax1 = Bs * (1 - a0 / Hmax) + k0 * Hmax
    Bmax2 = (Ha - Hmax) * mu0 / Nd

    Whm1 = eta * Bmax1**m
    Whm2 = eta * Bmax2**m

    Wh1 = kw * Whm1 * V * n
    Wh2 = kw * Whm2 * V * n

    omega0 = 25*pi/180  
    omega = 1.833*pi/180  
    I = 0.0018  # Moment of inertia Assumption

    td1 = 2 * pi * I / Wh1 * (omega0 - omega) / 60 / 60 / 24  # days
    td2 = 2 * pi * I / Wh2 * (omega0 - omega) / 60 / 60 / 24 # days

    IDomega = I * (omega0 - omega) 
    print("===== Quetzal magnetic stuff =====")
    print(f"{IDomega=}")
    print(f"{Bmax1=}, {Bmax2=}")
    print(f"{Wh1=}, {Wh2=}")
    print(f"{Whm1=}, {Whm2=}")
    print(f"{td1=}, {td2=}")
    print("================================")
    return #should be within 4 to 6 days

def magneticstuff_gerhard():
    Bs = 0.45  # From paper
    a0 = 1.02  # From paper
    k0 = 5.0*1000  # From paper
    eta = 12  # From paper
    m = 1.97  # From paper
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ha = 25  # Magnetic field strength of earth max
    kw = 0.6  # From paper

    # ASSUMED VALUES
    e = 95  # elongation
    L = 9.5e-2  # [m]
    D = 1.00e-3  # [m]
    V = L * pi * (D/2)**2
    n = 2 #number of rods

    Nd = (4.02 * log10(e) - 0.185) / 2 / e**2
    Hmax = (-(Bs - mu0 / Nd * Ha) + sqrt((Bs - mu0 / Nd * Ha)**2 + 4 * (k0 + mu0 / Nd) * a0 * Bs)) / (2 * (k0 + mu0 / Nd))
    
    Bmax1 = Bs * (1 - a0 / Hmax) + k0 * Hmax
    Bmax2 = (Ha - Hmax) * mu0 / Nd

    Whm1 = eta * Bmax1**m
    Whm2 = eta * Bmax2**m

    Wh1 = kw * Whm1 * V * n
    Wh2 = kw * Whm2 * V * n

    omega0 = 12.5*pi/180  
    omega = 2.5*pi/180  
    I = 0.019  # Moment of inertia ! Assumption !

    td1 = 2 * pi * I / Wh1 * (omega0 - omega) / 60 / 60 / 24  # days
    td2 = 2 * pi * I / Wh2 * (omega0 - omega) / 60 / 60 / 24 # days

    IDomega = I * (omega0 - omega) 
    print("===== Gerhard magnetic stuff =====")
    print(f"{IDomega=}")
    print(f"{Bmax1=}, {Bmax2=}")
    print(f"{Wh1=}, {Wh2=}")
    print(f"{Whm1=}, {Whm2=}")
    print(f"{td1=}, {td2=}")
    print("================================")
    return #should be within 5 to 7 days

def magneticstuff_robbe():
    Bs = 0.45  # From paper
    a0 = 1.02  # From paper
    k0 = 5.0*1000  # From paper
    eta = 12  # From paper
    m = 1.97  # From paper
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ha = 25  # Magnetic field strength of earth max
    kw = 0.6  # From paper

    # ASSUMED VALUES
    e = 100  # elongation
    L = 8.5e-2  # m
    D = 0.85e-3  # m
    V = L * pi * (D / 2)**2
    
    Nd = (4.02 * log10(e) - 0.185) / 2 / e**2
    Hmax = (-(Bs - mu0 / Nd * Ha) + sqrt((Bs - mu0 / Nd * Ha)**2 + 4 * (k0 + mu0 / Nd) * a0 * Bs)) / (2 * (k0 + mu0 / Nd))
    
    Bmax1 = Bs * (1 - a0 / Hmax) + k0 * Hmax
    Bmax2 = (Ha - Hmax) * mu0 / Nd

    Whm1 = eta * Bmax1**m
    Whm2 = eta * Bmax2**m

    Wh1 = kw * Whm1 * V
    Wh2 = kw * Whm2 * V

    omega0 = 25*pi/180  # rad/s --> 25 deg/s ~ 4.2 RPM
    omega = 0.1  # rad/s --> ~1 RPM
    I = 1/6 * 1.2 * 0.1 ** 2   # Moment of inertia Assumption
    print(f"Moment of inertia: {I}")    
    

    td1 = 2 * pi * I / Wh1 * (omega0 - omega) / 60 / 60 / 24  # days
    td2 = 2 * pi * I / Wh2 * (omega0 - omega) / 60 / 60 / 24 # days

    IDomega = I * (omega0 - omega) 
    print("===== robbe magnetic stuff =====")
    print(f"{Hmax=}")
    print(f"{IDomega=}")
    print(f"{Bmax1=}, {Bmax2=}")
    print(f"{Wh1=}, {Wh2=}")
    print(f"{Whm1=}, {Whm2=}")
    print(f"{td1=}, {td2=}")
    print("================================")
    return

def magneticstuff_sensitivity():
    Bs = 0.45  # From paper
    a0 = 1.02  # From paper
    k0 = 5.0*1000  # From paper
    eta = 12  # From paper
    m = 1.97  # From paper
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ha = 25  # Magnetic field strength of earth max
    kw = 0.6  # From paper

    # ASSUMED VALUES
    e = 100  # elongation
    L = 8.5e-2  # m
    D = 0.85e-3 * 1.5 # m
    V = L * pi * (D/2)**2

    Nd = (4.02 * log10(e) - 0.185) / 2 / e**2
    Hmax = (-(Bs - mu0 / Nd * Ha) + sqrt((Bs - mu0 / Nd * Ha)**2 + 4 * (k0 + mu0 / Nd) * a0 * Bs)) / (2 * (k0 + mu0 / Nd))
    
    Bmax1 = Bs * (1 - a0 / Hmax) + k0 * Hmax
    Bmax2 = (Ha - Hmax) * mu0 / Nd

    Whm1 = eta * Bmax1**m
    Whm2 = eta * Bmax2**m

    Wh1 = kw * Whm1 * V
    Wh2 = kw * Whm2 * V

    omega0 = 25*pi/180  # rad/s --> 25 deg/s ~ 4.2 RPM
    omega = 0.1  # rad/s --> ~1 RPM
    I = 1/6 * 1.2 * 0.1 ** 2   # Moment of inertia Assumption
    

    td1 = 2 * pi * I / Wh1 * (omega0 - omega) / 60 / 60 / 24  # days
    td2 = 2 * pi * I / Wh2 * (omega0 - omega) / 60 / 60 / 24 # days

    IDomega = I * (omega0 - omega) 
    print("===== sensitivity magnetic stuff =====")
    print(f"{Hmax=}")
    print(f"{IDomega=}")
    print(f"{Bmax1=}, {Bmax2=}")
    print(f"{Wh1=}, {Wh2=}")
    print(f"{Whm1=}, {Whm2=}")
    print(f"{td1=}, {td2=}")
    print("================================")
    return

def Wh1calc(factor: int|float):
    Bs = 0.45  # From paper
    a0 = 1.02  # From paper
    k0 = 5.0*1000  # From paper
    eta = 12  # From paper
    m = 1.97  # From paper
    mu0 = 4 * pi * 1e-7  # Permeability of free space [T m/A]
    Ha = 25  # Magnetic field strength of earth max
    kw = 0.6  # From paper

    # ASSUMED VALUES
    L = 8.5e-2  # m
    D = 0.85e-3 * factor # m
    e = L / D
    V = L * pi * (D / 2)**2

    Nd = (4.02 * log10(e) - 0.185) / 2 / e**2
    Hmax = (-(Bs - mu0 / Nd * Ha) + sqrt((Bs - mu0 / Nd * Ha)**2 + 4 * (k0 + mu0 / Nd) * a0 * Bs)) / (2 * (k0 + mu0 / Nd))
    
    Bmax1 = Bs * (1 - a0 / Hmax) + k0 * Hmax
    Bmax2 = (Ha - Hmax) * mu0 / Nd

    Whm1 = eta * Bmax1**m
    Whm2 = eta * Bmax2**m

    Wh1 = kw * Whm1 * V
    Wh2 = kw * Whm2 * V
    return Wh1, Wh2

def plot_Wh1():
    """
    Function to plot the Wh1 values for different factors.
    """
    factors = [1, 1.25, 1.5, 2.0]
    Wh1_values = []
    Wh2_values = []
    
    for factor in factors:
        Wh1, Wh2 = Wh1calc(factor)
        Wh1_values.append(Wh1)
        Wh2_values.append(Wh2)
        
    
    
    plt.plot(factors, [w / Wh1_values[0] for w in Wh1_values], label='Wh1')
    plt.plot(factors, [f ** 2 for f in factors], label='quadratic')
    plt.xlabel('Factor')
    plt.ylabel('Wh (J)')
    plt.title('Wh vs Factor')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    magneticstuff_Quetzal()
    magneticstuff_gerhard()
    magneticstuff_robbe()
    # magneticstuff_sensitivity()
    # plot_Wh1()
    # Wh1, Wh2 = Wh1calc(1.5)
    # print(f"Wh1: {Wh1}, Wh2: {Wh2}")