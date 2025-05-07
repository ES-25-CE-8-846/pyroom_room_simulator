import numpy as np
from pathlib import Path
from tools import RoomSimulator

import logging

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, dataset_params: dict = None):
        """
        Initialize the DatasetGenerator.
        """
        self.parse_params(**dataset_params)
        self.save_params(dataset_params)
        self.random_gen = np.random.default_rng(seed=self.seed)

    def parse_params(self, **kwargs):
        self.name: str = kwargs.get("name", "unnamed_run")
        self.splits: dict = kwargs.get(
            "splits",
            {
                "train": {
                    "num_rooms": 1,
                    "num_phone_pos": 1,
                },
                "val": {
                    "num_rooms": 1,
                    "num_phone_pos": 1,
                },
                "test": {
                    "num_rooms": 1,
                    "num_phone_pos": 1,
                },
            },
        )
        # Ensure each split has num_rooms and num_phone_pos
        for split, params in self.splits.items():
            assert "num_rooms" in params, f"Split {split} must have num_rooms"
            assert "num_phone_pos" in params, f"Split {split} must have num_phone_pos"

        self.room_params: dict = kwargs.get(
            "room_params",
            {
                "fs": None,
                "n_mics": None,
                "mic_radius": None,
                "shape": "shoebox",
                "signal": None,
            },
        )
        # Ensure room_params has fs, n_mics, mic_radius, and signal
        assert "fs" in self.room_params, "room_params must have fs"
        assert "n_mics" in self.room_params, "room_params must have n_mics"
        assert "mic_radius" in self.room_params, "room_params must have mic_radius"
        assert "shape" in self.room_params, "room_params must have shape"
        assert "signal" in self.room_params, "room_params must have signal"

        self.shape: str = self.room_params["shape"]
        self.root: Path = Path(kwargs.get("root", "dataset")) / self.shape / self.name

        self.regularizer: str = kwargs.get("regularizer", "rt60")
        # Ensure regularizer is valid
        assert self.regularizer in [
            "rt60",
            "maxlen",
        ], "Regularizing must be either 'rt60' or 'maxlen'"

        self.dtype: type = kwargs.get("dtype", np.float32)
        self.seed: int = kwargs.get("seed", None)
        return True

    def save_params(self, params: dict):
        """
        Save the parameters to a yaml file.
        """
        # FIXME: Implement this, problems: variables should be stringified :)
        logger.info(f"Saving parameters to {self.root}/config.yaml")
        logger.critical("Saving parameters to yaml is not implemented yet")
        # import yaml
        # with open("dataset_params.yaml", "w") as file:
        #     yaml.dump(params, file)
        return True

    def simulate_room(
        self,
        room_params,
        num_phone_pos: int = 1,
        save_dir: Path = None,
        dtype=np.float32,
        plot=False,
    ):
        """
        Simulate the room using the RoomSimulator class.
        """
        simulator = RoomSimulator(seed=self.random_gen.integers(0, 10000))

        # Generate room
        simulator.compose_room(**room_params)
        if plot:
            simulator.plot_room()

        # Do for each phone position
        for i in range(num_phone_pos):

            # Generate phone position
            simulator.randomize_room()

            # Compute the RIRs and RT60 times then regularize them
            if self.regularizer == "rt60":
                rirs, rt60s = simulator.compute_rir(rt60=True)
                reg_rirs = simulator.regularize_rir(rirs, rt60s, dtype=dtype)

            elif self.regularizer == "maxlen":
                rirs = simulator.compute_rir(rt60=False)
                reg_rirs = simulator.regularize_rir(rirs, dtype=dtype)

            # Split the RIRs into bright and dark zones
            bz_rir, dz_rir = simulator.get_zones(rirs=reg_rirs)

            # Save the RIRs
            if save_dir is not None:
                save_path = save_dir / f"{i}".rjust(4, "0")
                logger.info(f"Saving RIRs to {save_path}")

                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path.absolute(), bz_rir=bz_rir, dz_rir=dz_rir)
                logger.info(f"RIRs saved to {save_path.absolute()}")

        return simulator

    def start(self):
        """
        Start the dataset generation process.
        """
        logger.info("Starting dataset generation...")

        # Create the root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=False)

        # Generate data for each split
        for split, params in self.splits.items():
            logger.info(f"Generating {split} split...")

            # Generate rooms dataset
            for i in range(params["num_rooms"]):
                logger.info(f"Generating room {i}/{params['num_rooms']}...")
                save_dir = self.root / split / f"room_{i}"
                self.simulate_room(
                    self.room_params,
                    num_phone_pos=params["num_phone_pos"],
                    save_dir=save_dir,
                    dtype=self.dtype,
                    plot=False,
                    seed=self.seed,
                )
                logger.warning(
                    f"Successfully generated room {i+1}/{params['num_rooms']}!"
                )

            logger.warning(f"Successfully generated {split} split!")

        logger.warning("Dataset generation completed!")


def main():

    # Set logging configuration
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    )

    # Specify signal source
    from scipy.io import wavfile

    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono

    # Example usage
    dataset_params = {
        "name": "run1",
        "root": Path("dataset"),
        "splits": {
            "train": {
                "num_rooms": 10,
                "num_phone_pos": 10,
            },
            "val": {
                "num_rooms": 2,
                "num_phone_pos": 10,
            },
            "test": {
                "num_rooms": 2,
                "num_phone_pos": 10,
            },
        },
        "room_params": {
            "fs": fs,
            "n_mics": 12,
            "mic_radius": 0.5,
            "shape": "shoebox",  # "shoebox", "l_room", "t_room"
            "signal": signal,
            "room_bounds": {
                "min_width": 3.0,
                "max_width": 10.0,
                "min_length": 3.0,
                "max_length": 10.0,
                "min_extrude": 2.0,
                "max_extrude": 5.0,
            },
            # "desired_rt60": 0.5,
            "material_properties_bounds": {  #  # If desired_rt68 is None, this will be used
                "energy_absorption": (0.6, 0.9),
                "scattering": (0.05, 0.1),
            },
            # "ray_tracing_params": {
            #     "receiver_radius": 0.1,
            #     "n_rays": 10000,
            #     "energy_thres": 0.01,
            # },
        },
        "regularizer": "rt60",
        "dtype": np.float32,
        "seed": 69,
    }

    dataset_generator = DatasetGenerator(dataset_params)
    dataset_generator.start()


if __name__ == "__main__":
    main()
