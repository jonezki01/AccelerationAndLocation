import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
from geopy.distance import geodesic
from scipy.signal import welch

df_step = pd.read_csv('Linear Acceleration.csv')
df_loc = pd.read_csv('Location.csv')

def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

T = df_step['Time (s)'][len(df_step['Time (s)'])-1] - df_step['Time (s)'][0]
n = len(df_step['Time (s)'])
fs = n/T
nyq = fs/2
order = 3
cutoff = 1/(0.5)

filtered_signal = butter_lowpass_filter(df_step['Linear Acceleration z (m/s^2)'], cutoff, nyq, order)

plt.figure(figsize=(12,4))
plt.plot(df_step['Time (s)'], df_step['Linear Acceleration z (m/s^2)'])
plt.grid()
plt.axis([0,20,-5,9])

jaksot = 0
for i in range(len(filtered_signal)-1):
    if filtered_signal[i]/filtered_signal[i+1] < 0:
        jaksot += 1
print('Askelmäärä on ', np.floor(jaksot/2))

yf = fft(filtered_signal)
xf = fftfreq(n, 1 / fs)

idx = np.argmax(np.abs(yf))
dominant_freq = np.abs(xf[idx])

step_count = dominant_freq * T

print("Askelmäärä Fourier-analyysillä on: ", np.floor(step_count))

time = df_step['Time (s)']
acceleration = df_step['Linear Acceleration z (m/s^2)']

velocity = np.cumsum(acceleration) * (time[1] - time[0])

average_velocity_m_s = trapezoid(velocity, time) / (time.iloc[-1] - time.iloc[0])
average_velocity_km_h = abs(average_velocity_m_s * 3.6)

print("Keskinopeus on: ", average_velocity_km_h, "km/h")

coordinates = df_loc[['Latitude (°)', 'Longitude (°)']].values.tolist()
total_distance = 0.0
for i in range(1, len(coordinates)):
    total_distance += geodesic(coordinates[i-1], coordinates[i]).kilometers
print("Kokonaismatka on: ", total_distance, "km")

askelpituus = (total_distance * 1000) / step_count

askelpituus_senteissa = askelpituus * 100

print("Askelpituus on: ", askelpituus_senteissa, "senttimetriä")

time = df_step['Time (s)']
acceleration = df_step['Linear Acceleration z (m/s^2)']

# Laske näytteenottotaajuus
fs = 1 / (time[1] - time[0])

# Laske tehospektritiheys Welch-menetelmällä
frequencies, psd = welch(acceleration, fs)

# Luo DataFrame tehospektritiheydelle
chart_data = pd.DataFrame(np.transpose(np.array([frequencies, psd])), columns=["freq", "psd"])

# Piirrä tehospektritiheys Matplotlibillä
plt.figure(figsize=(10, 6))
plt.plot(chart_data['freq'], chart_data['psd'])
plt.xlabel('Taajuus [Hz]')
plt.ylabel('Teho')
plt.grid()
plt.show()