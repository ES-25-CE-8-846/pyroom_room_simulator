import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
from tools import MicrophoneArray
from tools import SpeakerArray
from tools import RoomGenerator

# specify signal source
fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
if signal.ndim > 1:
    signal = np.mean(signal, axis=1)  # Average channels to convert to mono


# Define room properties
corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
material_properties = {'energy_absorption': 0.3, 'scattering': 0.5}
ray_tracing_params = {'receiver_radius': 0.5, 'n_rays': 10000, 'energy_thres': 1e-5}

# Generate the room
room_generator = RoomGenerator(corners, material_properties, fs, ray_tracing_params)
room, _ = room_generator.generate_room()

# Define the speaker array
n_speakers = 8
speaker1_orientation = [-1, -0.2, 0]  # Along x,y,z axis
speaker_array_distance = 0.2

speaker_array1_position = [2.5, 0.5, 0.3]  # (x, y, z)

speaker_array1 = SpeakerArray(n_speakers, speaker_array1_position, speaker_array_distance, speaker1_orientation)
speaker_array1_positions = speaker_array1.get_speaker_positions()

# Add the speaker array to the room as sources
for pos in speaker_array1_positions:
    room.add_source(pos, signal=signal)

# Define the microphone array
mic_array_rows = 4
mic_array_cols = 4
mic_array_distance = 0.1

mic_array_1_position = [1.5, 1.5, 1]  # (x, y, z)


# add microphone array 1
mic_array_1 = MicrophoneArray(rows=mic_array_rows, cols=mic_array_cols, start_position=mic_array_1_position, distance=mic_array_distance)
mic_1_positions = mic_array_1.get_microphone_positions()
mic_1_positions = mic_1_positions.T
room.add_microphone_array(mic_1_positions)

# compute image sources
room.image_source_model()

# plot
fig, ax = room.plot()

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])

plt.show()
print("Simulating room acoustics...")
room.simulate()
print("Simulation complete")

mic_signals = room.mic_array.signals

print("Calculating impulse responses...")
impulse_responses = []
for mic_signal in mic_signals:
    impulse_response = fftconvolve(signal, mic_signal[::-1])
    impulse_responses.append(impulse_response)

print("Plotting impulse responses...")
# Plot impulse response for microphones
for i, impulse_response in enumerate(impulse_responses):
    plt.figure()
    plt.plot(impulse_response)
    plt.title(f"Impulse Response for Microphone {i+1}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
