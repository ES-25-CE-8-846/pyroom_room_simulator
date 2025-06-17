import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from tools.phone import Phone
from scipy.io import wavfile

def compute_pressure_field(absorption=0.4, room_dim=[10, 7, 3], filter_type="q_vast"):
    room = pra.ShoeBox(room_dim, fs=16000, max_order=3, absorption=absorption)

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

    # Prepare signals for each speaker
    speaker_signals = []
    for i, speaker_pos in enumerate(phone_speakers):
        if filter_type:
            filt = filters[filter_type][i]
            filtered = fftconvolve(base_signal, filt)[:len(base_signal)]
            speaker_signals.append(filtered)
            room.add_source(speaker_pos, signal=filtered)
        else:
            speaker_signals.append(base_signal)
            room.add_source(speaker_pos, signal=base_signal)

    # Microphone grid at z=1.5
    x = np.linspace(0, room_dim[0], 100)
    y = np.linspace(0, room_dim[1], 100)
    xx, yy = np.meshgrid(x, y)
    mic_positions = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 1.5)].T
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    room.compute_rir()

    # Compute pressure at each mic by convolving RIR with source signal and taking RMS
    pressure = np.zeros(mic_positions.shape[1])
    for mic_idx, mic_rirs in enumerate(room.rir):
        mic_signal = np.zeros(1)
        for src_idx, rir in enumerate(mic_rirs):
            # Convolve RIR with the corresponding speaker signal
            received = fftconvolve(speaker_signals[src_idx], rir)
            # Sum contributions from all sources
            if len(received) > len(mic_signal):
                mic_signal = np.pad(mic_signal, (0, len(received) - len(mic_signal)))
            mic_signal[:len(received)] += received
        # Compute RMS of the received signal at this mic
        pressure[mic_idx] = np.sqrt(np.mean(mic_signal**2))
    pressure = pressure.reshape(xx.shape)
    return xx, yy, pressure, phone_position

print("Computing pressure field...")
# Get filtered pressure field
xx, yy, pressure_filtered, phone_position = compute_pressure_field(
    absorption=0.4, room_dim=[10, 7, 3], filter_type="q_vast"
)
print("Computing unfiltered pressure field...")
# Get unfiltered pressure field
_, _, pressure_unfiltered, _ = compute_pressure_field(
    absorption=0.4, room_dim=[10, 7, 3], filter_type=False
)
print("Plotting results...")
plt.figure(figsize=(16, 6))

# Plot filtered
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, 20 * np.log10(np.abs(pressure_filtered) + 1e-12), levels=50, cmap="viridis")
plt.scatter(phone_position[0], phone_position[1], color="red", label="Phone")
plt.title("Filtered")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.colorbar(label="Sound Pressure Level (dB RMS)")
plt.legend()

# Plot unfiltered
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, 20 * np.log10(np.abs(pressure_unfiltered) + 1e-12), levels=50, cmap="viridis")
plt.scatter(phone_position[0], phone_position[1], color="red", label="Phone")
plt.title("Unfiltered")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.colorbar(label="Sound Pressure Level (dB RMS)")
plt.legend()

plt.tight_layout()
plt.show()