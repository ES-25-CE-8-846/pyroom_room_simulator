import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from tools import RoomGenerator
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
from scipy.linalg import toeplitz
from numpy.linalg import inv


def main():
    # specify signal source
    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono
    signal = signal / np.max(np.abs(signal) + 1e-8)  # Normalize

    # Define room properties
    material_properties = {'energy_absorption': 0.3, 'scattering': 0.5}
    ray_tracing_params = {'receiver_radius': 0.5, 'n_rays': 10000, 'energy_thres': 1e-5}

    # Generate the room
    width, length = 3, 5
    corners = None #np.array([[0, 0], [0, width], [length, width], [length, 0]]).T
    room_generator = RoomGenerator(corners=corners, material_properties_bounds=None, fs=fs, ray_tracing_params=ray_tracing_params, extrude_height=np.random.uniform(2.0, 3.0))
    room_bounds={
        "min_width": 4.0,
        "max_width": 8.0,
        "min_length": 5.0,
        "max_length": 10.0,
        "min_extrude": 2.5,
        "max_extrude": 4.0
        }
    room, _ = room_generator.generate_room(room_bounds=room_bounds)

    # Define the mic array
    from tools import MicrophoneCircle
    from tools import Phone

    RADIUS = 0.5
    N_MICS = 12

    room_bbox = room.get_bbox() # in m
    phone_pos = [
        np.random.uniform(room_bbox[0,0]+RADIUS, room_bbox[0,1]-RADIUS),
        np.random.uniform(room_bbox[1,0]+RADIUS, room_bbox[1,1]-RADIUS),
        np.random.uniform(room_bbox[2,0]+0.1, room_bbox[2,1]-0.1),
    ]

    microphone_circle = MicrophoneCircle(center=phone_pos, radius=RADIUS, n_mics=N_MICS)
    dark_zone_mics = microphone_circle.get_microphone_positions()

    phone_speaker = Phone(position=phone_pos, unit="m")
    bright_zone_mics = phone_speaker.get_mic_positions()
    phone_speakers = phone_speaker.get_speaker_positions()

    # Add the speakers to the room as sources
    #for pos in phone_speakers:
    #    room.add_source(pos, signal=signal/len(phone_speakers))
    room.add_source(phone_speakers[0], signal=signal)

    # Add the microphone arrays to the room
    all_mics = np.concatenate([bright_zone_mics, dark_zone_mics], axis=0)
    room.add_microphone_array(all_mics.T)

    # compute image sources
    room.image_source_model()

    # plot
    fig, ax = room.plot()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])

    plt.show()

    # Simulate the room acoustics (necessary for impulse response calculation with ray tracing)
    print("Simulating room acoustics...")
    room.simulate()
    print("Simulation complete")

    print("Calculating impulse responses...")
    impulse_responses = room.rir  # List of lists: [mic][source] -> array of IR

    # == Prepare impulse responses for VAST ==
    target_length = 2048  # IR length
    filter_length = 2048  # Filter length for VAST
    alpha = 100          # Trade-off between zones

    num_sources = len(room.sources)
    num_mics_total = len(room.mic_array.R.T)

    # Split mic indexes
    num_mics_bright = bright_zone_mics.shape[0]
    num_mics_dark = dark_zone_mics.shape[0]

    assert num_mics_total == num_mics_bright + num_mics_dark

    # Init IR containers
    ir_bright = np.zeros((num_mics_bright, num_sources, target_length))
    ir_dark = np.zeros((num_mics_dark, num_sources, target_length))


    for m in range(num_mics_total):
        for s in range(num_sources):
            ir = impulse_responses[m][s][:target_length]
            if m < num_mics_bright:
                ir_bright[m, s, :len(ir)] = ir
            else:
                ir_dark[m - num_mics_bright, s, :len(ir)] = ir

    original_mic_signals = room.mic_array.signals  # shape: [num_mics, signal_len]
    original_bright_mic_signal = original_mic_signals[0]  # First bright mic
    original_dark_mic_signal = original_mic_signals[num_mics_bright]  # First dark mic

    plt.figure(figsize=(12, 6))
    plt.plot(original_bright_mic_signal, label="Bright Zone Mic 1 Signal")
    plt.plot(original_dark_mic_signal, label="Dark Zone Mic 1 Signal")
    plt.title("Signal Comparison Between original Bright and Dark Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Normalize and clip original mic signals
    original_bright_mic_signal = original_bright_mic_signal / (np.max(np.abs(original_bright_mic_signal)) + 1e-8)
    original_bright_mic_signal = np.clip(original_bright_mic_signal, -1, 1)
    sf.write("bright_mic_original.wav", original_mic_signals[0], fs)
    original_dark_mic_signal = original_dark_mic_signal / (np.max(np.abs(original_dark_mic_signal)) + 1e-8)
    original_dark_mic_signal = np.clip(original_dark_mic_signal, -1, 1)
    sf.write("dark_mic_original.wav", original_mic_signals[num_mics_bright], fs)
    
    ### GPT suggested code for VAST filter generation ###

    def create_conv_matrix(ir, filter_length):
        ir_padded = np.concatenate([ir, np.zeros(filter_length - 1)])
        return toeplitz(ir_padded, np.r_[ir[0], np.zeros(filter_length - 1)]) # Create Toeplitz matrix to convolve with matrix multiplication

    def vast(ir_bright, ir_dark, L=256, span_idx=None, alpha=1e-2):
        M_b, S, _ = ir_bright.shape
        M_d = ir_dark.shape[0]
        if span_idx is None:
            span_idx = np.arange(S)

        S_active = len(span_idx)
        A_b = []
        A_d = []

        for m in range(M_b):
            row = [create_conv_matrix(ir_bright[m, s], L) for s in span_idx]
            A_b.append(np.hstack(row))
        A_b = np.vstack(A_b)

        for m in range(M_d):
            row = [create_conv_matrix(ir_dark[m, s], L) for s in span_idx]
            A_d.append(np.hstack(row))
        A_d = np.vstack(A_d)

        d = np.zeros((M_b * (L + ir_bright.shape[2] - 1), 1))
        for m in range(M_b):
            d[m * (L + ir_bright.shape[2] - 1):(m+1)*(L + ir_bright.shape[2] - 1)] = 1.0 / M_b


        H = inv(A_b.T @ A_b + alpha * A_d.T @ A_d) @ (A_b.T @ d)
        return H

    print("Designing VAST filters...")
    filters = vast(ir_bright, ir_dark, L=filter_length, span_idx=[0], alpha=alpha)
    print("Filters designed")

    # Plot VAST filters
    plt.figure(figsize=(12, 6))
    for i in range(num_sources):
        start = i * filter_length
        end = (i + 1) * filter_length
        plt.plot(filters[start:end], label=f"Filter for Speaker {i+1}")
    plt.title("VAST Filters for Each Speaker")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    # Apply filters to original signal
    # Apply filters: assume filters are stacked [spk1_filter, spk2_filter, ...]
    num_sources = len(room.sources)
    filters = filters.reshape(-1, 1)  # [num_sources * filter_len, 1]

    filtered_signals = []
    for i in range(num_sources):
        start = i * filter_length
        end = (i + 1) * filter_length
        h = filters[start:end].flatten()
        filtered = fftconvolve(signal, h)[:len(signal)]
        filtered_signals.append(filtered)

    # Inject filtered signals into room
    for i, src in enumerate(room.sources):
        src.signal = filtered_signals[i]
    
    print("Simulating room with filtered signal...")
    room.simulate()
    print("Simulation done")

    # Test playback of mic signals
    VAST_mic_signals = room.mic_array.signals  # shape: [num_mics, signal_len]

    VAST_bright_mic_signal = VAST_mic_signals[0]  # First bright mic
    VAST_dark_mic_signal = VAST_mic_signals[num_mics_bright]  # First dark mic
    
    # Compare the filtered signals from bright and dark zone
    plt.figure(figsize=(12, 6))
    plt.plot(VAST_bright_mic_signal, label="Bright Zone Mic 1 Signal")
    plt.plot(VAST_dark_mic_signal, label="Dark Zone Mic 1 Signal")
    plt.title("Signal Comparison Between Bright and Dark Zone Mics after VAST")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    #
    plt.figure(figsize=(12, 6))
    plt.plot(original_bright_mic_signal, label="Original Bright Zone Mic 1 Signal")
    plt.plot(VAST_bright_mic_signal, label="VAST Bright Zone Mic 1 Signal")
    plt.title("Signal Comparison Between orignal Bright and VAST Bright Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(original_dark_mic_signal, label="Original Dark Zone Mic 1 Signal")
    plt.plot(VAST_dark_mic_signal, label="VAST Dark Zone Mic 1 Signal")
    plt.title("Signal Comparison Between orignal Dark and VAST Dark Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()



    # Optionally save to WAV
    sf.write("bright_mic_VAST.wav", VAST_bright_mic_signal, fs)
    sf.write("dark_mic_VAST.wav", VAST_dark_mic_signal, fs)


if __name__ == "__main__":
    main()