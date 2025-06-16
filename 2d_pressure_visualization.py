import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import lfilter, butter
from tools.phone import Phone
from scipy.io import wavfile

room_dim = [10, 7, 3]

room = pra.ShoeBox(
    room_dim,
    fs=16000,  # Sampling frequency
    max_order=3,  # Maximum reflection order
    absorption=0.1,  # Absorption coefficient
)

phone_position = [3, 4, 1.5]  # X, Y, Z in meters
phone_orientation = [0, -90, 0]  # Euler angles (roll, pitch, yaw) in degrees

# Initialize the phone
phone = Phone(position=phone_position, orientation=phone_orientation, unit="m")

# Add phone's speakers to the room
phone_speakers = phone.get_speaker_positions()

fs, base_signal = wavfile.read(r"wav_files\president-is-moron.wav")

filtered_signals = []
for i, speaker_pos in enumerate(phone_speakers):
    # TODO: Apply filters to the base signal
    filtered_signals.append(base_signal) # Placeholder for filtered signals
    room.add_source(speaker_pos, signal=filtered_signals[i], delay=0)

# Define the microphone grid (heatmap resolution)
x = np.linspace(0, room_dim[0], 100)  # 100 points along the x-axis
y = np.linspace(0, room_dim[1], 100)  # 100 points along the y-axis
xx, yy = np.meshgrid(x, y)
mic_positions = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 1.5)].T  # shape (3, N)

# Add all grid points as microphones at z=1.5
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

# Compute all RIRs
room.compute_rir()

# Calculate pressure as sum of RIR amplitudes for each mic and all sources
pressure = np.zeros(mic_positions.shape[1])
for mic_idx, mic_rirs in enumerate(room.rir):
    pressure[mic_idx] = sum(np.sum(np.abs(rir)) for rir in mic_rirs)

# Reshape pressure data to match the grid
pressure = pressure.reshape(xx.shape)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, 20 * np.log10(np.abs(pressure) + 1e-6), levels=50, cmap="viridis")
plt.colorbar(label="Sound Pressure Level (dB)")
plt.scatter(phone_position[0], phone_position[1], color="red", label="Phone")
plt.title("2D Pressure Visualization")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.show()