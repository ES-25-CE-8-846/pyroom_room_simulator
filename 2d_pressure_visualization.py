import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from tools.phone import Phone
from scipy.io import wavfile


room_dim = [10, 7, 3]
room = pra.ShoeBox(room_dim, fs=16000, max_order=3, absorption=0.4)

phone_position = [3, 4, 1.5]
phone_orientation = [0, -90, 0]
phone = Phone(position=phone_position, orientation=phone_orientation, unit="m")
phone_speakers = phone.get_speaker_positions()

# Load the base signal
fs, base_signal = wavfile.read(r"wav_files\president-is-moron.wav")
if base_signal.ndim > 1:
    base_signal = base_signal[:, 0]  # Use first channel if stereo

# Load filters from combined_filters.npz
filters = np.load(r"filters\combined_filters.npz")
#print(np.shape(filters))
#print(filters.files)

filtered_signals = []
for i, speaker_pos in enumerate(phone_speakers):
    filter = filters["q_vast"][i]
    filtered = fftconvolve(base_signal, filter)[:len(base_signal)]
    filtered_signals.append(filtered)
    room.add_source(speaker_pos, signal=filtered)

# Microphone grid at z=1.5
x = np.linspace(0, room_dim[0], 1000)
y = np.linspace(0, room_dim[1], 1000)
xx, yy = np.meshgrid(x, y)
mic_positions = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 1.5)].T
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
room.compute_rir()

pressure = np.zeros(mic_positions.shape[1])
for mic_idx, mic_rirs in enumerate(room.rir):
    pressure[mic_idx] = sum(np.sum(np.abs(rir)) for rir in mic_rirs)
pressure = pressure.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, 20 * np.log10(np.abs(pressure) + 1e-6), levels=50, cmap="viridis")
plt.colorbar(label="Sound Pressure Level (dB)")
plt.scatter(phone_position[0], phone_position[1], color="red", label="Phone")
plt.title("2D Pressure Visualization")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.show()