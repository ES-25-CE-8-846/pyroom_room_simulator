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

def pressure_matching(ir_bright, ir_dark, ir_length, lambd):
    """
    PM filter design with least-squares minimization.
    Returns filters [num_sources, L]
    """
    M_bright, S, N = ir_bright.shape
    M_dark, _, _ = ir_dark.shape
    
    # Compute convolution matrices for bright and dark zones
    H_bright = []
    for s in range(S): # Go through each source in bright zone
        Hs = []
        for m in range(M_bright): # Go through each mic in bright zone
            Hs.append(toeplitz(np.r_[ir_bright[m, s], np.zeros(ir_length - 1)], np.zeros(ir_length)))
        H_bright.append(np.vstack(Hs))  # [M_bright*signal_len, length]
    
    H_dark = []
    for s in range(S):
        Hs = []
        for m in range(M_dark):
            Hs.append(toeplitz(np.r_[ir_dark[m, s], np.zeros(ir_length - 1)], np.zeros(ir_length)))
        H_dark.append(np.vstack(Hs))


    filters = []
    for s in range(S): # Compute filter for each source
        H_b = H_bright[s]
        H_d = H_dark[s]

        d = np.zeros(H_b.shape[0])
        d[len(d)//2] = 1.0  # Dirac at center (desired pressure)

        A = H_b.T @ H_b + lambd * (H_d.T @ H_d) # Cost function
        b = H_b.T @ d # Desired pressure in bright zone

        h = np.linalg.solve(A, b) # Solve for filter coefficients
        filters.append(h)

    return np.stack(filters)  # shape [num_sources, L]

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
    for pos in phone_speakers:
        room.add_source(pos, signal=signal)

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

    ir_length = 2048

    num_sources = len(room.sources)
    num_mics_total = len(room.mic_array.R.T)

    # Split mic indexes
    num_mics_bright = bright_zone_mics.shape[0]
    num_mics_dark = dark_zone_mics.shape[0]

    assert num_mics_total == num_mics_bright + num_mics_dark

    # Init IR containers
    ir_bright = np.zeros((num_mics_bright, num_sources, ir_length))
    ir_dark = np.zeros((num_mics_dark, num_sources, ir_length))


    for m in range(num_mics_total):
        for s in range(num_sources):
            ir = impulse_responses[m][s][:ir_length]
            if m < num_mics_bright:
                ir_bright[m, s, :len(ir)] = ir
            else:
                ir_dark[m - num_mics_bright, s, :len(ir)] = ir

    original_mic_signals = room.mic_array.signals

    global_norm = np.max(np.abs(original_mic_signals)) + 1e-8  # USE THIS FOR ALL SCALING FROM HERE ON OUT
    original_mic_signals = original_mic_signals / global_norm

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

    sf.write("bright_mic_original.wav", original_mic_signals[0], fs)
    sf.write("dark_mic_original.wav", original_mic_signals[num_mics_bright], fs)
    
    # Apply PM filtering

    print("Designing filters...")
    filters = pressure_matching(ir_bright, ir_dark, ir_length, lambd=1e-8) # Increase lambd for more dark zone influence
    print("Filters designed")
    filter_length = filters.shape[1]

    # Apply filters to original signal

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
    PM_mic_signals = room.mic_array.signals  # shape: [num_mics, signal_len]
    PM_mic_signals = PM_mic_signals / global_norm  # Normalize
    
    PM_bright_mic_signal = PM_mic_signals[0]  # First bright mic
    PM_dark_mic_signal = PM_mic_signals[num_mics_bright]  # First dark mic

    # Compare the filtered signals from bright and dark zone
    plt.figure(figsize=(12, 6))
    plt.plot(PM_bright_mic_signal, label="PM Bright Zone Mic 1 Signal")
    plt.plot(PM_dark_mic_signal, label="PM Dark Zone Mic 1 Signal")
    plt.title("Signal Comparison Between PM Bright and Dark Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Compare filtered bright with original bright
    plt.figure(figsize=(12, 6))
    plt.plot(original_bright_mic_signal, label="Original Bright Zone Mic 1 Signal")
    plt.plot(PM_bright_mic_signal, label="PM Bright Zone Mic 1 Signal")
    plt.title("Signal Comparison Between Original and PM Bright Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Compare filtered dark with original dark
    plt.figure(figsize=(12, 6))
    plt.plot(original_dark_mic_signal, label="Original Dark Zone Mic 1 Signal")
    plt.plot(PM_dark_mic_signal, label="PM Dark Zone Mic 1 Signal")
    plt.title("Signal Comparison Between Original and PM Dark Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    sf.write("bright_mic_PM.wav", PM_bright_mic_signal, fs)
    sf.write("dark_mic_PM.wav", PM_dark_mic_signal, fs)

if __name__ == "__main__":
    main()