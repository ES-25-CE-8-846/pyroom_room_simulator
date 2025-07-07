# Dataset Generation

## Running the dataset generation

Running the script is as simple as:  
```shell
git clone https://github.com/ES-25-CE-8-846/pyroom_room_simulator.git
cd pyroom_room_simulator
python3 dataset_generator.py
```

## Configuration
To define the dataset generation process simply modify the ```dataset_params``` dict, found on [line 255](https://github.com/ES-25-CE-8-846/pyroom_room_simulator/blob/18cc721a9b21b839c13f967bfec4702b07933a60/dataset_generator.py#L255)

A default example of the dict is given below
- The root is a folder created relative to the location of the script, thus in the example, a folder named "dataset" will be created at the root of the repo.
- Within the root folder, a folder containing the generated dataset is created, named by the "name" parameter, in the example the specific dataset run is called "run_post_hand_in". 

<pre lang="md"> <code> dataset_params = { 
    "name": "run_post_hand_in", 
    "root": Path("dataset"), 
    "splits": { 
        "train": { 
            "num_rooms": 1000, 
            "num_phone_pos": 10, 
        }, 
        "val": { 
            "num_rooms": 200, 
            "num_phone_pos": 10, 
        }, 
        "test": { 
            "num_rooms": 100, 
            "num_phone_pos": 10, 
        }, 
    }, 
    "room_params": { 
        "fs": fs, 
        "n_mics": 12, 
        "mic_radius": 0.5, # m 
        "shape": "shoebox", # "shoebox", "l_room", "t_room" 
        "signal": signal, 
        "room_bounds": { 
            "min_width": 3.0, # m 
            "max_width": 10.0, # m 
            "min_length": 3.0, # m 
            "max_length": 10.0, # m 
            "min_extrude": 3.0, # m 
            "max_extrude": 6.0, # m 
        }, 
        # "desired_rt60": 0.5, 
        "material_properties_bounds": {
            "energy_absorption": (0.05, 0.4), 
            "scattering": (0.05, 0.3), 
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
} </code> </pre>

