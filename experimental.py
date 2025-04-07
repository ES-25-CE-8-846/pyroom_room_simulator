import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
from microphone_array import MicrophoneArray
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
room_generator = RoomGenerator(material_properties=material_properties, fs=fs, ray_tracing_params=ray_tracing_params, extrude_height=np.random.uniform(2.0, 3.0))
room = room_generator.generate_room()

# Define the mic array
from microphone_circle import MicrophoneCircle
from phone_speaker import PhoneSpeaker

radius = 0.5
n_mics = 12

print(room.get_bbox())

room_bbox = room.get_bbox() # in m
phone_pos_x = np.random.uniform(room_bbox[0,0]+0.6, room_bbox[0,1]-0.6)
phone_pos_y = np.random.uniform(room_bbox[1,0]+0.6, room_bbox[1,1]-0.6)
phone_pos_z = np.random.uniform(room_bbox[2,0]+0.1, room_bbox[2,1]+0.1)
phone_pos = [phone_pos_x, phone_pos_y, phone_pos_z] # mm
#phone_pos = [pos * 0.001 for pos in [phone_pos_x, phone_pos_y, phone_pos_z]]  # Convert to meters

print(phone_pos)

microphone_circle = MicrophoneCircle(center=phone_pos, radius=radius, n_mics=n_mics)
bright_zone_mics = microphone_circle.get_microphone_positions()

phone_speaker = PhoneSpeaker(position=phone_pos)#, orientation=[0, -90, 0])
dark_zone_mics = phone_speaker.get_mic_positions()
phone_speakers = phone_speaker.get_speaker_positions()

# Add the speaker array to the room as sources
for pos in phone_speakers:
    #print(pos)
    room.add_source(pos, signal=signal)
    #print(f"added source at {pos}")

# Add the microphone array to the room
print(bright_zone_mics)
room.add_microphone_array(bright_zone_mics.T)
room.add_microphone_array(dark_zone_mics.T)

# compute image sources
#room.image_source_model()

# plot
fig, ax = room.plot()
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

plt.show()


# print("Simulating room acoustics...")
# room.simulate()
# print("Simulation complete")

# mic_signals = room.mic_array.signals

# print("Calculating impulse responses...")
# impulse_responses = []
# for mic_signal in mic_signals:
#     impulse_response = fftconvolve(signal, mic_signal[::-1])
#     impulse_responses.append(impulse_response)

# print("Plotting impulse responses...")
# # Plot impulse response for microphones
# for i, impulse_response in enumerate(impulse_responses):
#     plt.figure()
#     plt.plot(impulse_response)
#     plt.title(f"Impulse Response for Microphone {i+1}")
#     plt.xlabel("Samples")
#     plt.ylabel("Amplitude")
#     plt.show()
