import numpy as np
import pyroomacoustics as pra
from speaker_array import SpeakerArray

class RoomGenerator:
    def __init__(self, corners, material_properties, fs, ray_tracing_params, extrude_height=2.0):
        """
        Initialize the RoomGenerator.

        Parameters:
        corners (np.array): Array of room corners [x, y].
        material_properties (dict): Dictionary with 'energy_absorption' and 'scattering'.
        fs (int): Sampling frequency.
        ray_tracing_params (dict): Dictionary with 'receiver_radius', 'n_rays', and 'energy_thres'.
        extrude_height (float): Height to extrude the room.
        """
        self.corners = corners
        self.material_properties = material_properties
        self.fs = fs
        self.ray_tracing_params = ray_tracing_params
        self.extrude_height = extrude_height

    def generate_room(self):
        """
        Generate the room with the specified properties.
        """
        material = pra.Material(energy_absorption=self.material_properties['energy_absorption'],
                                scattering=self.material_properties['scattering'])
        room = pra.Room.from_corners(self.corners, materials=material, fs=self.fs, ray_tracing=True, air_absorption=True)
        room.extrude(self.extrude_height)
        room.set_ray_tracing(receiver_radius=self.ray_tracing_params['receiver_radius'],
                             n_rays=self.ray_tracing_params['n_rays'],
                             energy_thres=self.ray_tracing_params['energy_thres'])
        return room
    
    def insert_speaker_array_randomly(self, room):
        """
        Insert a speaker array into the room.

        Parameters:
        room (pyroomacoustics.room.Room): Room object.
        """
        # Define the speaker array
        n_speakers = 8
        speaker_array_distance = 0.2

        x_min, xmax = np.min(self.corners[0]), np.max(self.corners[0]) - n_speakers * speaker_array_distance
        y_min, ymax = np.min(self.corners[1]), np.max(self.corners[1]) - n_speakers * speaker_array_distance
        z_min, zmax = 0, self.extrude_height

        # Randomly choose a position within the adjusted range
        speaker_array_position = [
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ]

        speaker_orientation = np.random.uniform(-1, 1, 3)
        speaker_orientation[2] = 0  # Ensure the speaker is flat
        speaker_orientation /= np.linalg.norm(speaker_orientation)

        speaker_array = SpeakerArray(n_speakers, speaker_array_position, speaker_array_distance, speaker_orientation)
        speaker_array_positions = speaker_array.get_speaker_positions()
        
        for pos in speaker_array_positions:
            room.add_source(pos)

        return room