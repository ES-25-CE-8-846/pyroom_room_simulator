import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

class RoomGenerator:
    def __init__(self, corners=None, material_properties=None, fs=16000, ray_tracing_params=None, extrude_height=None):
        """
        Initialize the RoomGenerator.

        Parameters:
        corners (np.array): Array of room corners [x, y].
        material_properties (dict): Dictionary with 'energy_absorption' and 'scattering'.
        fs (int): Sampling frequency.
        ray_tracing_params (dict): Dictionary with 'receiver_radius', 'n_rays', and 'energy_thres'.
        extrude_height (float): Height to extrude the room.
        max_corners (int): Maximum number of corners for the room.
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
        if self.corners is None:
            width = np.random.uniform(3.0, 30.0)
            length = np.random.uniform(3.0, 30.0) 
            self.corners = np.array([[0, 0], [0, width], [length, width], [length, 0]])

        if self.material_properties is None:
            self.material_properties = {
                'energy_absorption': np.random.uniform(0.1, 0.9),
                'scattering': np.random.uniform(0.1, 0.9)
            }
        if self.ray_tracing_params is None:
            self.ray_tracing_params = {
                'receiver_radius': 0.1,
                'n_rays': 1000,
                'energy_thres': 1e-5
            }
        if self.extrude_height is None:
            self.extrude_height = np.random.uniform(min(2.0, width, length), max(width, length)) #Ensure that the room is at least 2m high unless width or length are lower

        material = pra.Material(energy_absorption=self.material_properties['energy_absorption'],
                                scattering=self.material_properties['scattering'])
        room = pra.Room.from_corners(self.corners.T, materials=material, fs=self.fs, ray_tracing=True, air_absorption=True)
        room.extrude(self.extrude_height)
        room.set_ray_tracing(receiver_radius=self.ray_tracing_params['receiver_radius'],
                             n_rays=self.ray_tracing_params['n_rays'],
                             energy_thres=self.ray_tracing_params['energy_thres'])

        # Reset attributes to None
        self.corners = None
        self.material_properties = None
        self.ray_tracing_params = None
        self.extrude_height = None

        return room
    
if __name__ == "__main__":
    room_generator = RoomGenerator()
    room1 = room_generator.generate_room()
    fig, ax = room1.plot()
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_zlim([0, 30])
    plt.show()
    
    room2 = room_generator.generate_room()
    fig, ax = room2.plot()
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_zlim([0, 30])
    plt.show()
