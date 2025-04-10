import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

class RoomGenerator:
    def __init__(self, corners=None, shape: str = "shoebox", material_properties=None, fs=48000, ray_tracing_params=None, extrude_height=None):
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
        assert shape in ["shoebox", "t_room", "l_room"], f"Invalid room shape: {shape}. Must be one of ['shoebox', 't_room', 'l_room']."
        self.shape = shape
        self.material_properties = material_properties
        self.fs = fs
        self.ray_tracing_params = ray_tracing_params
        self.extrude_height = extrude_height

    # TODO: Needs a v2
    def _compose_random_shoebox(self, np_random_generator: np.random.Generator, min_width = 3.0, max_width = 10.0, min_length = 3.0, max_length = 10.0, min_extrude = 2.0) -> tuple[np.ndarray, float]:
        """
        Compose a shoebox room with the specified corners and material properties.
        """
        logger.debug("Generating random shoebox room")
        
        # Generate random width and length
        width = np_random_generator.uniform(min_width, max_width)
        length = np_random_generator.uniform(min_length, max_length)
        corners = np.array([[0, 0], [0, width], [length, width], [length, 0]])
        
        if self.extrude_height is None:
            extrude_height = np_random_generator.uniform(max(min_extrude, min(width, length)), min(width, length)) # Ensure that the room is at least 2m high
            return corners, extrude_height
        
        return corners, self.extrude_height
    
    
    # FIXME: Make this
    def _compose_random_troom(self,) -> tuple[np.ndarray, float]:
        """
        Compose a T-shaped room with the specified corners and material properties.
        """
        raise NotImplementedError("T-shaped room composition is not implemented.")
        return corners, extrude_height
    
    
    # FIXME: Make this
    def _compose_random_lroom(self,) -> tuple[np.ndarray, float]:
        """
        Compose a L-shaped room with the specified corners and material properties.
        """
        raise NotImplementedError("L-shaped room composition is not implemented.")
        return corners, extrude_height
    
    
    def generate_room(self, seed=None):
        """
        Generate the room with the specified properties. If ´corners´ is not set, a room with random width and length is generated.
        """
        
        # Set randomness generator seed
        random_gen = np.random.default_rng(seed=seed)
        
        
        # If corners is not set, generate random corners
        if self.corners is None:
            if self.shape == "shoebox":
                self.corners, self.extrude_height = self._compose_random_shoebox(np_random_generator=random_gen)
            elif self.shape == "t_room":
                self.corners, self.extrude_height = self._compose_random_troom()
            elif self.shape == "l_room":
                self.corners, self.extrude_height = self._compose_random_lroom()
        
        
        # If corners is set, use them to generate the room
        else:
            width = np.max(self.corners[1, :]) - np.min(self.corners[1, :])
            length = np.max(self.corners[0, :]) - np.min(self.corners[0, :])
            self.corners = np.array([[0, 0], [0, width], [length, width], [length, 0]])
            self.extrude_height = random_gen.uniform(min(2.0, width, length), min(width, length))

        if self.material_properties is None:
            self.material_properties = {
                'energy_absorption': random_gen.uniform(0.1, 0.9),
                'scattering': random_gen.uniform(0.1, 0.9)
            }
        if self.ray_tracing_params is None:
            self.ray_tracing_params = {
                'receiver_radius': 0.1,
                'n_rays': 1000,
                'energy_thres': 1e-5
            }
        

        material = pra.Material(energy_absorption=self.material_properties['energy_absorption'],
                                scattering=self.material_properties['scattering'])
        room = pra.Room.from_corners(self.corners.T, materials=material, fs=self.fs, ray_tracing=True, air_absorption=True)
        room.extrude(self.extrude_height)
        room.set_ray_tracing(receiver_radius=self.ray_tracing_params['receiver_radius'],
                             n_rays=self.ray_tracing_params['n_rays'],
                             energy_thres=self.ray_tracing_params['energy_thres'])
        dimensions = {
            'width': room.get_bbox()[0,1],
            'length': room.get_bbox()[1,1],
            'height': self.extrude_height
        }
        # Reset attributes to None
        self.corners = None
        self.material_properties = None
        self.ray_tracing_params = None
        self.extrude_height = None

        return room , dimensions
    
if __name__ == "__main__":
    room_generator = RoomGenerator()
    room1, room1_dimensions = room_generator.generate_room()
    print(room1_dimensions['width'], room1_dimensions['length'], room1_dimensions['height'])
    fig, ax = room1.plot()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    plt.show()
    
    room2, room2_dimensions = room_generator.generate_room()
    print(room2_dimensions['width'], room2_dimensions['length'], room2_dimensions['height'])
    fig, ax = room2.plot()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    plt.show()
