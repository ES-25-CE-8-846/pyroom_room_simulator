import numpy as np
import pyroomacoustics as pra
from tools import RoomGenerator
from tools import MicrophoneCircle
from tools import Phone
import matplotlib.pyplot as plt


class RoomSimulator():
    def __init__(
            self, fs: int, 
            signal: np.ndarray,
            corners: np.ndarray | None = None,
            material_properties: dict | None = None,
            ray_tracing_params: dict | None = None,
            extrude_height: float | None = None,
        ):
        """Simulate room :)

        Args:
            fs (int): Sample rate of signal
            signal (np.ndarray): Sound signal to be broadcast in the room. Ensure it is mono?
        """
        
        self.fs = fs
        self.signal = signal
        
        # Define room properties
        self.corners = corners
        
        if extrude_height is None: self.extrude_height = np.random.uniform(2.0, 3.0)
        else: self.extrude_height = extrude_height
        
        if material_properties is None: self.material_properties = {'energy_absorption': 0.3, 'scattering': 0.5}
        else: self.material_properties = material_properties
        
        if ray_tracing_params is None: self.ray_tracing_params = {'receiver_radius': 0.5, 'n_rays': 10000, 'energy_thres': 1e-5}
        else: self.ray_tracing_params = ray_tracing_params
        
        # Compose an initial room
        self.compose_room(
            corners=self.corners,
            material_properties=self.material_properties,
            ray_tracing_params=self.ray_tracing_params,
            extrude_height=self.extrude_height,
        )
        
        
    def compose_room(
            self, 
            n_mics: int = 12,
            mic_radius: float = 0.5,
            phone_rotation: np.ndarray = np.array([0, -90, 0]),
            corners: np.ndarray = None, 
            material_properties: dict = None, 
            ray_tracing_params: dict = None,
            extrude_height: float = None,
        ):
        """Generate a room with Phone and Microphone Circle
            Sets the `self.room` and `self.room_dims` variables

        Args:
            n_mics (int, optional): Number of mics in cicle. Defaults to 12.
            mic_radius (float, optional): The radius of the mic circle. Defaults to 0.5.
            phone_rotation (np.ndarray, optional): The rotation of the phone in Euler angles (XYZ). Defaults to np.array(0, -90, 0) (Standing).
            corners (np.ndarray, optional): The corners of the room. Defaults to None (Randomized).
            material_properties (dict, optional): _description_. Defaults to check yo self >:).
            ray_tracing_params (dict, optional): _description_. Defaults to check yo self >:).
            extrude_height (float, optional): Height of the room. Defaults to None (Random between 2->3).
        """
        
        # Create the empty room
        room_gen = RoomGenerator(corners=corners, material_properties=material_properties, ray_tracing_params=ray_tracing_params, fs=self.fs, extrude_height=extrude_height)
        room, room_dims = room_gen.generate_room()

        
        # Add mic circle
        room_bbox = room.get_bbox() # in m
        phone_pos = [
            np.random.uniform(room_bbox[0,0]+mic_radius, room_bbox[0,1]-mic_radius),
            np.random.uniform(room_bbox[1,0]+mic_radius, room_bbox[1,1]-mic_radius),
            np.random.uniform(room_bbox[2,0]+0.1, room_bbox[2,1]-0.1),
        ]
        
        mic_circle_gen = MicrophoneCircle(center=phone_pos, n_mics=n_mics, radius=mic_radius)
        dark_zone_mics = mic_circle_gen.get_microphone_positions()
        room.add_microphone_array(dark_zone_mics.T)
        
        
        # Add phone mics and speakers
        phone_gen = Phone(position=phone_pos, orientation=phone_rotation, unit="m")
        bright_zone_mics = phone_gen.get_mic_positions()
        phone_speakers = phone_gen.get_speaker_positions()
        
        room.add_microphone_array(bright_zone_mics.T)
        
        for pos in phone_speakers:
            room.add_source(pos, signal=self.signal)
        
        # Set the room and its dims
        self.room, self.room_dims = room, room_dims
        
        return True
    
    
    def plot_room(self):
        fig, ax = self.room.plot()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([0, 10])
        plt.show()
        return True
    
    
    def simulate_room(self, plot=False):
        print("Simulating room acoustics...") # FIXME: logging
        self.room.simulate()
        print("Simulation complete!") # FIXME: logging
        
        #mic_signals = self.room.mic_array.signals
        print("Computing room impulse response...") # FIXME: logging
        self.room.compute_rir()
        print("Computed room impulse response!")
        
        return self.room.rir
        # if plot: 
        #     # for x in ["ir", "tf", "spec"]:
        #     #     self.room.plot_rir(kind=x)
        #     self.room.plot_rir()
        #     # self.room.plot_rir(FD=True)
                
        





if __name__ == "__main__":
    from scipy.io import wavfile
    
    # specify signal source
    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono
        
    room_sim = RoomSimulator(fs=fs, signal=signal)
    room_sim.plot_room()
    room_sim.simulate_room(plot=False)
    print(len(room_sim.room.rir))
    print(len(room_sim.room.rir[0]))