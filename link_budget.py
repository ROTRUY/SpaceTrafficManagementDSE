import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt

class LinkBudget:
    def __init__(self, distance, transmission_power,gain_r, gain_t,frequency, bandwidth, temp, antenna_diameter, is_uplink=False):
        self.d = distance
        self.transmission_power = transmission_power
        self.gain_r = gain_r
        self.gain_t = gain_t
        self.b = bandwidth
        self.c = 3*10**8
        self.f, self.name = frequency
        self.lmbda = self.c / self.f
        self.boltzman = 1.38 * 10**-23
        self.temp = temp
        self.antenna_d = antenna_diameter
        self.snr = 0
        self.pl = 0
        self.A_r_gain = 0
        self.tn = 0
        self.receiver_efficiency = 1
        self.receiver_efficiency = 0.6 # delfi
        self.is_uplink = is_uplink

    def antenna_receiver_gain(self):
        if self.is_uplink:
            self.A_r_gain = self.gain_r     #[dB]
        else:
            gain = self.receiver_efficiency * (np.pi * self.antenna_d / self.lmbda) ** 2
            self.A_r_gain = LinkBudget.decimal_to_db(gain)  #[dB]

    def pathloss(self,atm, misc):
        # path_loss = LinkBudget.decimal_to_db((self.lmbda/(self.d*4*np.pi))**2)
        path_loss = 20 * np.log10(4 * np.pi * self.d / self.lmbda) # [dB]
        print(self.name,"free space pathloss", path_loss)
        atmospheric_loss = atm  # [dB]
        # atmospheric_loss = -1 # delfi
        miscellaneous_loss = misc   # [dB]
        # miscellaneous_loss = -2 #delfi
        self.pl = path_loss + atmospheric_loss + miscellaneous_loss # [dB]

    def thermal_noise(self):
        """
        thermal noise floor of a radio receiver.
        Noise FloordBm = 10*Log10(k T B/(1 mW)) + NF
        """
        self.tn = LinkBudget.decimal_to_db((self.boltzman *self.temp* self.b)) + 30 # [dBm]

    def signal_to_noise(self, atm_loss=0, misc_loss=0, plots = False):
        self.antenna_receiver_gain() # [dB]
        self.pathloss(atm = atm_loss, misc = misc_loss) # [dB]
        self.thermal_noise() # [dB]
        # rc_p = self.transmission_power + self.pl + self.A_r_gain + self.gain_t - 30
        print(
            f"{self.name}: {self.transmission_power} {self.pl} {self.A_r_gain} {self.gain_t}"
        )
        rc_p = self.transmission_power - self.pl + self.A_r_gain + self.gain_t  # [dB]
        self.snr = rc_p - self.tn # [dB]
        if not plots:
            print(f"{self.name}-band, Antenna receiver gain = {self.A_r_gain} dB,\n \
                pathloss = {self.pl} dB, noise floor = {self.tn} dBm,\n \
                received power = {rc_p} dBm, sign to noise ratio = {self.snr} dB \n")

    def eb_no(self, data_rate):
        s_no_dbhz = self.snr + 10 * np.log10(self.b)
        eb_no = s_no_dbhz - 10 * np.log10(data_rate)
        return eb_no

    def link_margin(self, required_eb_no, data_rate):
        actual_eb_no = self.eb_no(data_rate)
        margin = actual_eb_no - required_eb_no
        print(f"{self.name}-band, Eb/No = {actual_eb_no:.2f} dB, Required = {required_eb_no} dB, Link Margin = {margin:.2f} dB")
        return margin

    def shannon_theorem(self):
        return self.b * np.log2(1 + self.snr)

    @staticmethod
    def decimal_to_db(value):
        return 10*np.log10(value)

def compare_links(uplink, downlink, data_rate, required_eb_no):
    print("----- UPLINK -----")
    uplink.signal_to_noise(atm_loss=5, misc_loss=5)
    uplink.link_margin(required_eb_no, data_rate)

    print("\n----- DOWNLINK -----")
    downlink.signal_to_noise(atm_loss=5, misc_loss=5)
    downlink.link_margin(required_eb_no, data_rate)

if __name__ == "__main__":
    plot = True
    # plot = False
    downlink_check = True
    downlink_check = False

    downlink_uhf = True
    downlink_uhf = False
    uplink_uhf = True
    uplink_uhf = False

    gomgom = True
    # gomgom = False
    gommission= True
    # gommission = False
    if plot:
        distances = range(300000, 5000000 + 50000, 50000)
        transmission_power = 30.0 # dBm
        antenna_diameter = 4.0
        gain_r = 0  # groundstation gain is calculated
        gain_t = -5 # gain from the omni antenna
        # band = [((1.45*10**8, "VHF"), 12500),((4*10**8, "UHF"), 25000),((3*10**9, "S"), 2000000),((1*10**10,"X"), 10000000)]
        frequency, bandwidth = ((4.375 * 10**8, "UHF"), 25000)
        temp = 1000  # kelvin
        distance_band = np.zeros(len(distances))

        # modulation = [10.5, 12]
        # for i , mod in enumerate(modulation):
        for i, distance in enumerate(distances):
            link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter)
            link.signal_to_noise(atm_loss=5, misc_loss=5, plots=plot)
            distance_band[i] = link.eb_no(9600)
        distances_km = np.array(distances) / 1000
        hz, name = frequency
        # Plot SNR vs distance (x = SNR, y = distance)
        uhf_frequency = plt.plot(distance_band, distances_km, label=f"{name}")

        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel('Distance (km)')
        # plt.title('Distance vs Eb/N0 for UHF')

        snr_GMSK = 10.5
        snr_GFSK = 12
        # snr_16QAM = 19.5

        plt.axvline(x=snr_GMSK, color="blue", linestyle="--", label="GMSK")
        # plt.text(snr_GFSK, plt.ylim()[0] - 250, f"{snr_GFSK}", color="black", ha="center")
        plt.axvline(x=snr_GFSK, color="purple", linestyle="--", label="GFSK ")
        # plt.axvline(x=snr_16QAM, color='pink', linestyle='--', label='16QAM')


        plt.legend(["Frequency", "GMSK Modulation (10.5 dB)", "GFSK Modulation (12 dB)"])
        plt.grid(True)
        plt.tight_layout()
        plt.show()






        RE = 6371          # Earth radius in km
        H0 = 1000           # Satellite height in km
        HGS = 0            # Ground station height in km
        eps_deg = np.linspace(0.1, 90, 500)  # Elevation angle in degrees (avoid 0° to prevent sqrt of negative)
        eps_rad = np.radians(eps_deg)       # Convert to radians

        # Equation 14 for slant distance L
        term1 = ((RE + HGS) * np.sin(eps_rad))**2
        term2 = 2 * (H0 - HGS) * (RE + HGS)
        term3 = (H0 - HGS)**2
        L = np.sqrt(term1 + term2 + term3) - (RE + HGS) * np.sin(eps_rad)

        # Relative Free Space Loss (FSL)
        L_min = np.min(L)
        FSL_rel = 20 * np.log10(L / L_min)  # in dB

        # Plotting
        fig, ax1 = plt.subplots()

        # Distance plot
        ax1.plot(eps_deg, L, 'b-', label='Distance')
        ax1.set_xlabel('Elevation  [°]')
        ax1.set_ylabel('Distance  [km]', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # # FSL plot (second y-axis)
        # ax2 = ax1.twinx()
        # ax2.plot(eps_deg, FSL_rel, 'r--', label='Relative FSL')
        # ax2.set_ylabel('FSL relative to zenith / dB', color='r')
        # ax2.tick_params(axis='y', labelcolor='r')

        # Legends and grid
        fig.legend(loc="upper right")
        # plt.title(f'Distance vs Elevation Angle at {H0}km Orbit')
        plt.grid(True)
        plt.show()






        # Elevation angles (in degrees) from 5 to 90
        # # Frequency range from 300 MHz to 3 GHz
        # frequencies_mhz = np.linspace(300, 3000, 500)
        # frequencies_ghz = frequencies_mhz / 1000.0

        # # Simplified linear model for atmospheric attenuation in dB/km
        # # Starting at 0.001 dB/km at 300 MHz up to ~0.016 dB/km at 3 GHz
        # attenuation_db_per_km = 0.001 + (frequencies_ghz - 0.3) * (0.015 / (3.0 - 0.3))

        # # Path length in kilometers
        # path_length_km = 10

        # # Total atmospheric loss (in dB)
        # total_loss_db = attenuation_db_per_km * path_length_km

        # # Plot the results
        # plt.figure(figsize=(10, 6))
        # plt.plot(frequencies_mhz, total_loss_db, color='darkblue')
        # plt.title("Atmospheric Loss vs Frequency (300 MHz to 3 GHz over 10 km)")
        # plt.xlabel("Frequency (MHz)")
        # plt.ylabel("Atmospheric Loss (dB)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    if downlink_check:
        distance = 2000000
        transmission_power = 30 # dB
        antenna_diameter = 4.5
        gain_r = 0  # groundstation gain is calculated
        gain_t = -5  # gain from the omni antenna
        band = [((1.45*10**8, "VHF"), 12500),((4*10**8, "UHF"), 25000),((3*10**9, "S"), 2000000),((1*10**10,"X"), 10000000)]
        # (frequencies bandwidths )VHF UHF S X
        temp = 300  # kelvin
        for frequency, bandwidth in band:
            link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter)
            link.signal_to_noise(atm_loss=5, misc_loss=5)

        # verification delfi
        distance = 635000
        transmission_power = -10
        antenna_diameter = 25.0
        gain_r = 0
        gain_t = 0 # offset the large margin of loss for other calculation, it is not the actual t gain
        frequency, bandwidth = [(9.15 * 10**8, "Delfi"), 100000]
        temp = 200
        link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter)
        link.signal_to_noise(atm_loss=1, misc_loss=2)

        # performance analysis
        distancess: Iterable = [400000, 2000000]
        transmission_power = 27 # dBm
        antenna_diameter = 4.5
        gain_r = 0  # groundstation gain is calculated
        gain_t = -5  # gain from the omni antenna
        frequency, bandwidth = [(4 * 10**8, "Prelim option"), 25000]
        temp = 300
        BER = 1*10^-5
        snr: list = []
        for distance in distancess:
            link = LinkBudget(distance, transmission_power, gain_r,gain_t, frequency, bandwidth, temp, antenna_diameter)
            link.signal_to_noise(atm_loss=5, misc_loss=5)
            snr.append(link.snr)

    if downlink_uhf:
        distance = 2000000
        transmission_power = 30
        antenna_diameter = 4 # Ground station dish
        gain_r = 0  # groundstation gain is calculated
        gain_t = -5  # gain from the omni antenna
        frequency, bandwidth = [(4.375* 10**8, "UHF_downlink"), 25000]
        temp = 400
        link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter)
        link.signal_to_noise(atm_loss=5, misc_loss=5)

    if uplink_uhf:
        distance = 2000000
        transmission_power = 48 # Ground station, higher power
        antenna_diameter = 0  # spacecraft dish
        gain_r = 0  # Satellite typically has little gain
        gain_t = 15  # gain from the ground staiton
        frequency, bandwidth = [(4*10**8, "UHF_uplink"), 25000]
        temp = 400
        link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter, is_uplink=True)
        link.signal_to_noise(atm_loss=5, misc_loss=5)

    if gomgom:
        distance = 1900000 #700000
        transmission_power = 30    # S/C                   [dBm]
        antenna_diameter = 0        # spacecraft dish       [m]
        gain_r = 0   # Satellite typically has little gain  [dB]
        gain_t = 17  # gain from the ground staiton         [dB]
        frequency, bandwidth = [(4.375*10**8, "GOM_downlink"), 25000] #[hz]
        temp = 1003 # [K]
        link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter)
        link.signal_to_noise(atm_loss=2.1, misc_loss=4.9)

        distance = 1900000 #700000
        transmission_power = 44  # G/S                  [dBm]
        antenna_diameter = 0        # spacecraft dish       [m]
        gain_r = 17   # gain from the ground staiton         [dB]
        gain_t = 0  # Satellite typically has little gain  [dB]
        frequency, bandwidth = [(4.375*10**8, "GOM_uplink"), 25000] #[hz]
        temp = 234 # [K]
        link = LinkBudget(distance, transmission_power, gain_r, gain_t, frequency, bandwidth, temp, antenna_diameter, is_uplink=True)
        link.signal_to_noise(atm_loss=2.1, misc_loss=5.7)

    if gommission:
        uplink = LinkBudget(
            distance=2000000,
            transmission_power=44,
            gain_r=0,
            gain_t=15,
            frequency=(437.5e6, "GOM_uplink"),
            bandwidth=25000,
            temp=234,
            antenna_diameter=0,
            is_uplink=True,
            )

        downlink = LinkBudget(
            distance=2500000,
            transmission_power=30,  # satellite Tx power in dBm
            gain_r=15,  # G/S antenna gain
            gain_t=-5,  # spacecraft antenna gain
            frequency=(437.5e6, "GOM_downlink"),
            bandwidth=25000,
            temp=1003,
            antenna_diameter=4,
            is_uplink=False,
        )

        compare_links(uplink, downlink, data_rate=9600, required_eb_no=10.5)

