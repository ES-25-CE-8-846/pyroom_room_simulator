import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
from microphone_array import MicrophoneArray
from microphone_circle import MicrophoneCircle
from speaker_array import SpeakerArray
from room_generator import RoomGenerator


# specify signal source
fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
if signal.ndim > 1:
    signal = np.mean(signal, axis=1)  # Average channels to convert to mono


# Define room properties
#corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
material_properties = {'energy_absorption': 0.3, 'scattering': 0.5}
ray_tracing_params = {'receiver_radius': 0.5, 'n_rays': 10000, 'energy_thres': 1e-5}

# Generate the room
room_generator = RoomGenerator(material_properties=material_properties, fs=fs, ray_tracing_params=ray_tracing_params)
room, room_dimensions = room_generator.generate_room()

# Define microphone circle params
radius = 0.5  # radius of the circle in meters
n_mics = 20
center = [random.uniform(0 + radius, room_dimensions['length']-radius), # x
          random.uniform(0 + radius, room_dimensions['width']-radius), # y
          random.uniform(0 + radius, room_dimensions['height']-radius) # z
          ]

mic_array = MicrophoneCircle(center=center, radius=radius, n_mics=n_mics, sphere=False)
room.add_microphone_array(mic_array.get_microphone_positions().T)

"""
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
"""

# compute image sources
room.image_source_model()

# plot
fig, ax = room.plot()

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

plt.show()
print("Simulating room acoustics...")
room.simulate()
print("Simulation complete")

mic_signals = room.mic_array.signals

