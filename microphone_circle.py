import numpy as np

class MicrophoneCircle:
    def __init__(self, center: list | np.ndarray, radius: float, n_mics: int) -> np.ndarray:
        """
        Initialize a circle of microphones around a defined center.

        Parameters:
            center (list or np.array): Position of the microhpne circle center [x, y, z]
            radius (float): Distance from center to microphones
            n_mics (int): Number of microphones in the circle
        """
        self.center = center
        self.radius = radius
        self.n_mics = n_mics

        # Generate microphone positions as a matrix
        self.microphone_array = self._generate_microphone_positions()

    def _generate_microphone_positions(self):
        """
        Generate a circle of microphone positions as a (rows*cols, 3) matrix.
        """
        positions = []
        
        # Add the center microphone as the first index
        positions.append(self.center)
        
        # Calculate the positions of microphones in a circle
        angles = np.linspace(0, 2 * np.pi, self.n_mics, endpoint=False) # Endpoint ensures the last mic is not sampled at the same position as the first, 0deg = 360deg
        for angle in angles:
            x = self.center[0] + self.radius * np.cos(angle)
            y = self.center[1] + self.radius * np.sin(angle)
            z = self.center[2]
            positions.append([x, y, z])
        
        return np.array(positions)

    def plot_microphone_positions(self):
        """
        Plot the microphone positions in 3D.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the microphones
        ax.scatter(self.microphone_array[:, 0], self.microphone_array[:, 1], self.microphone_array[:, 2], c='r', marker='o')
        
        # Set labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()
    
    def get_microphone_positions(self):
        """
        Return the microphone positions as a NumPy matrix.
        """
        return self.microphone_array

    def __repr__(self):
        return (f"MicrophoneCircle(center={self.center}, radius={self.radius}, n_mics={self.n_mics})")


if __name__ == "__main__":
    # Example usage: Create a 24 microphone circle
    
    center = [0, 0, 5] # x, y, z
    radius = 1.0
    n_mics = 24

    mic_array = MicrophoneCircle(center=center, radius=radius, n_mics=n_mics)
    mic_array.plot_microphone_positions()
    print(mic_array)
    
    # Print the microphone positions
    print(mic_array.get_microphone_positions())
