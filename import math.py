import math

# --- Input (given data) ---
P_watt = 0.01  # Spacecraft Transmitter Power [W]
f_hz = 1.575e9  # downlink frequency [Hz]
L_l = 0.8  # Loss factor transmitter
L_r = 0.7  # Loss factor receiver
D_t = 0.02  # Spacecraft antenna diameter [m]
D_r = 5  # Ground Station antenna diameter [m]
eta = 0.55  # Antenna efficiency
mission_type = 1
h = 2000e3  # Spacecraft orbit altitude [m]
e_t_t = 5  # Pointing offset spacecraft [deg]
e_t_r = -0.012  # Pointing offset ground station [deg]
T_DL = 0.03 * 3600  # Downlink time ratio in seconds
snapshots = 48
bits_per_snapshot = 256
orbital_period = 5760  # seconds
D_c = 0.5
L_a = -0.5  # dB
T_s = 221  # K
required_SNR_dB = 15
voltage = 3.7  # V

# Constants
k = 1.38e-23  # Boltzmann constant [J/K]
c = 3e8  # Speed of light [m/s]
AU_km = 149597870.7
planet_radius = 6378e3  # [m]

# --- Interim Results ---
lambda_m = c / f_hz
R = bits_per_snapshot * snapshots / (orbital_period * T_DL / 86400)  # bits/s

# Transmitting and Receiving antenna gains (simplified)
G_t_dB = 10 * math.log10(eta * (math.pi * D_t / lambda_m) ** 2)
G_r_dB = 10 * math.log10(eta * (math.pi * D_r / lambda_m) ** 2)

# Space loss
d = planet_radius + h  # worst-case distance [m]
L_s_dB = 20 * math.log10(4 * math.pi * d / lambda_m)

# Losses
L_l_dB = 10 * math.log10(L_l)
L_r_dB = 10 * math.log10(L_r)

# Pointing losses (simplified)
L_pr_dB = -0.000675 + -0.12

# Total system noise factor
Ts_dB = 10 * math.log10(T_s)
k_dB = 10 * math.log10(k)
R_dB = 10 * math.log10(R)

# Transmit power in dB
P_dB = 10 * math.log10(P_watt)

# SNR calculation (link budget equation)
Eb_No_dB = (
    P_dB + G_t_dB + G_r_dB - L_s_dB - L_a + L_pr_dB - L_l_dB - L_r_dB
    - k_dB - Ts_dB - R_dB
)

# SNR margin
margin = Eb_No_dB - required_SNR_dB

# Power usage
current_mA = P_watt / voltage * 1000  # [mA]
P_mAh = current_mA * (T_DL / 3600)  # per day
P_y = P_mAh * 365
P_5y = P_y * 5

# --- Output ---
print(f"lambda: {lambda_m:.8f} m")
print(f"Data rate R: {R:.2f} bit/s")
print(f"G_t: {G_t_dB:.2f} dB")
print(f"G_r: {G_r_dB:.2f} dB")
print(f"L_s: {-L_s_dB:.2f} dB")
print(f"L_l: {L_l_dB:.2f} dB")
print(f"L_r: {L_r_dB:.2f} dB")
print(f"L_pr: {L_pr_dB:.2f} dB")
print(f"1/k: {-k_dB:.2f} dB")
print(f"1/Ts: {-Ts_dB:.2f} dB")
print(f"1/R: {-R_dB:.2f} dB")
print(f"Eb/No: {Eb_No_dB:.2f} dB")
print(f"Required Eb/No: {required_SNR_dB:.2f} dB")
print(f"Margin: {margin:.2f} dB")
print(f"Power Usage per Downlink: {P_mAh:.6f} mAh")
print(f"Power Usage per Year: {P_y:.2f} mAh")
print(f"Power Usage over 5 Years: {P_5y:.2f} mAh")
