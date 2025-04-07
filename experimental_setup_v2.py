import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from tools import RoomGenerator


def main():
    # specify signal source
    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono


    # Define room properties
    material_properties = {'energy_absorption': 0.3, 'scattering': 0.5}
    ray_tracing_params = {'receiver_radius': 0.5, 'n_rays': 10000, 'energy_thres': 1e-5}

    # Generate the room
    width, length = 3, 5
    corners = None #np.array([[0, 0], [0, width], [length, width], [length, 0]]).T
    room_generator = RoomGenerator(corners=corners, material_properties=material_properties, fs=fs, ray_tracing_params=ray_tracing_params, extrude_height=np.random.uniform(2.0, 3.0))
    room, _ = room_generator.generate_room()

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

    print(phone_pos)

    microphone_circle = MicrophoneCircle(center=phone_pos, radius=RADIUS, n_mics=N_MICS)
    dark_zone_mics = microphone_circle.get_microphone_positions()

    phone_speaker = Phone(position=phone_pos, unit="m")#, orientation=[0, -90, 0])
    bright_zone_mics = phone_speaker.get_mic_positions()
    phone_speakers = phone_speaker.get_speaker_positions()

    # Add the speakers to the room as sources
    for pos in phone_speakers:
        room.add_source(pos, signal=signal)

    # Add the microphone array to the room
    print(phone_speakers)
    print(bright_zone_mics)
    room.add_microphone_array(bright_zone_mics.T)
    room.add_microphone_array(dark_zone_mics.T)

    # compute image sources
    room.image_source_model()

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
    # print(len(impulse_responses))
    # print(impulse_responses)


    # # Plot impulse response for microphones
    # for i, impulse_response in enumerate(impulse_responses):
    #     plt.figure()
    #     plt.plot(impulse_response)
    #     plt.title(f"Impulse Response for Microphone {i+1}")
    #     plt.xlabel("Samples")
    #     plt.ylabel("Amplitude")
    #     plt.show()

if __name__ == "__main__":
    main()