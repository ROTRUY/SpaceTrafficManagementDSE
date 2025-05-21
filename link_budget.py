import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt

class LinkBudget:
    def __init__(self, distance, transmission_power,gain_r, frequency, bandwidth, temp, antenna_diameter, is_uplink=False):
        self.d = distance
        self.transmission_power = transmission_power
        self.gain_r = gain_r
        self.f, self.name = frequency
        self.b = bandwidth
        self.c = 3*10**8
        self.lmbda = self.c / self.f
        self.boltzman = 1.38 * 10**-23
        self.temp = temp
        self.antenna_d = antenna_diameter
        self.snr = 0
        self.pl = 0
        self.A_r_gain = 0
        self.tn = 0
        # self.receiver_efficiency = 0.5
        self.receiver_efficiency = 0.6 # delfi
        self.is_uplink = is_uplink

    def antenna_receiver_gain(self):
        if self.is_uplink:
            self.A_r_gain = self.gain_r
        else:
            gain = self.receiver_efficiency * (np.pi * self.antenna_d / self.lmbda) ** 2
            self.A_r_gain = LinkBudget.decimal_to_db(gain)

    def pathloss(self):
        path_loss = LinkBudget.decimal_to_db((self.lmbda/(self.d*4*np.pi))**2)
        print(self.name,"free space pathloss", path_loss)
        atmospheric_loss = -10
        # atmospheric_loss = -1 # delfi
        miscellaneous_loss = -10
        # miscellaneous_loss = -2 #delfi
        self.pl = path_loss + atmospheric_loss + miscellaneous_loss

    def thermal_noise(self):
        """
        thermal noise floor of a radio receiver.
        Noise FloordBm = 10*Log10(k T B/(1 mW)) + NF
        """
        self.tn = LinkBudget.decimal_to_db((self.boltzman *self.temp* self.b)) + 30

    def signal_to_noise(self, plots = False):
        self.antenna_receiver_gain()
        self.pathloss()
        self.thermal_noise()
        rc_p = self.transmission_power + self.pl + self.A_r_gain + self.gain_r
        self.snr = rc_p - self.tn
        if not plots:
            print(f"{self.name}-band, Antenna receiver gain = {self.A_r_gain} dB,\n \
                free space pathloss = {self.pl} dB, noise floor = {self.tn} dBW,\n \
                received power = {rc_p} dBm, sign to noise ratio = {self.snr} dB \n")

    def shannon_theorem(self):
        return self.b * np.log2(1 + self.snr)

    @staticmethod
    def decimal_to_db(value):
        return 10*np.log10(value)



if __name__ == "__main__":
    # plot = True
    plot = False
    downlink = False
    # downlink = True

    downlink_uhf = True
    uplink_uhf = True
    if plot:
        distances = range(300000, 2000000 + 50000, 50000)
        transmission_power = 30 # dBm
        antenna_diameter = 25
        gain_r = 0
        band = [((1.45*10**8, "VHF"), 12500),((4*10**8, "UHF"), 25000),((3*10**9, "S"), 2000000),((1*10**10,"X"), 10000000)]
        # (frequencies bandwidths )VHF UHF S X
        temp = 300  # kelvin
        distance_band = np.zeros([len(band), len(distances)])
        print(distance_band.shape)
        for i , (frequency, bandwidth) in enumerate(band):
            for j, distance in enumerate(distances):
                link = LinkBudget(distance, transmission_power, gain_r, frequency, bandwidth, temp, antenna_diameter)
                link.signal_to_noise(plots=plot)
                distance_band[i, j] = link.snr
        print(distance_band)
        distances_km = np.array(distances) / 1000
        for i, (frequency, bandwidth) in enumerate(band):
            hz, name = frequency
            # Plot SNR vs distance (x = SNR, y = distance)
            plt.plot(distance_band[i], distances_km, label=f"{name}")

        plt.xlabel('SNR (dB)')
        plt.ylabel('Distance (km)')
        plt.title('Distance vs SNR for Different Frequency Bands')

        snr_QPSK = 17
        snr_8PSK = 19
        snr_16QAM = 19.5

        plt.axvline(x=snr_QPSK, color='blue', linestyle='--', label='QPSK')
        plt.axvline(x=snr_8PSK, color='purple', linestyle='--', label='8PSK')
        plt.axvline(x=snr_16QAM, color='pink', linestyle='--', label='16QAM')


        plt.legend(title='Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if downlink:
        distance = 400000
        transmission_power = 30 # dBm
        antenna_diameter = 25
        gain_r = 0
        band = [((1.45*10**8, "VHF"), 12500),((4*10**8, "UHF"), 25000),((3*10**9, "S"), 2000000),((1*10**10,"X"), 10000000)]
        # (frequencies bandwidths )VHF UHF S X
        temp = 300  # kelvin
        for frequency, bandwidth in band:
            link = LinkBudget(distance, transmission_power, gain_r, frequency, bandwidth, temp, antenna_diameter)
            link.signal_to_noise()

        # verification delfi
        distance = 635000
        transmission_power = -10
        antenna_diameter = 25
        gain_r = 0
        frequency, bandwidth = [(9.15 * 10**8, "Delfi"), 100000]
        temp = 200
        link = LinkBudget(distance, transmission_power, gain_r, frequency, bandwidth, temp, antenna_diameter)
        link.signal_to_noise()

        # performance analysis
        distancess: Iterable = [400000, 2000000]
        transmission_power = 27 # dBm
        antenna_diameter = 15
        gain_r = 0
        frequency, bandwidth = [(4 * 10**8, "Prelim option"), 25000]
        temp = 300
        BER = 1*10^-5
        snr: list = []
        for distance in distancess:
            link = LinkBudget(distance, transmission_power, gain_r, frequency, bandwidth, temp, antenna_diameter)
            link.signal_to_noise()
            snr.append(link.snr)

    if downlink_uhf:
        distance = 400000
        transmission_power = 27
        antenna_diameter = 25
        gain_r = 0
        frequency, bandwidth = [(4*10**8, "UHF_downlink"), 25000]
        temp = 300
        link = LinkBudget(distance, transmission_power, gain_r, frequency, bandwidth, temp, antenna_diameter)
        link.signal_to_noise()

    if uplink_uhf:
        distance = 400000
        transmission_power = 48  # Ground station, higher power
        antenna_diameter = 3  # Ground station big dish
        gain_r = 0  # Satellite typically has little gain
        frequency, bandwidth = [(4*10**8, "UHF_uplink"), 25000]
        temp = 300
        link = LinkBudget(distance, transmission_power, gain_r, frequency, bandwidth, temp, antenna_diameter, is_uplink=True)
        link.signal_to_noise()
