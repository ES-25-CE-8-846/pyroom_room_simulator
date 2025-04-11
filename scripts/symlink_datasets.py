import numpy as np
import pathlib as path
import argparse


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Create symbolic links to datasets.")
    parser.add_argument("--data_root", type=str, help="Root directory for datasets.")
    parser.add_argument("--name", type=str, default='mixed', help="Name of the merged dataset.")
    parser.add_argument("remainder", nargs=argparse.REMAINDER, help="Datasets to merge i.e. \"shoebox/run1\" \"shoebox/run2\" \"l_room/run4\".")

    args = parser.parse_args()

    # Check if the data_root & name is provided
    if not args.data_root:
        raise ValueError("Please provide a dataset root directory using --data_root.")
    if not args.name:
        raise ValueError("Please provide a name for the merged dataset using --name.")
    
    # Create the root path
    root_path = path.Path(args.data_root)

    # Check if the root path exists
    if not root_path.exists():
        raise FileNotFoundError(f"The specified dataset root directory does not exist: {root_path}")
    
    # Create the merged dataset directory
    merged_dataset_path = root_path / args.name

    try:
        merged_dataset_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"The specified merged dataset directory already exists: {merged_dataset_path}. Please choose a different name.")
    

    # Compose new dataset name based on the included datasets
    dataset_name = ''
    for dataset in sorted(args.remainder):
        if 'shoebox' in dataset and 's' not in dataset_name:
            dataset_name += 's'
        if 'l_room' in dataset and 'l' not in dataset_name:
            dataset_name += 'l'
        if 't_room' in dataset and 't' not in dataset_name:
            dataset_name += 't'
        dataset_name += dataset.split('/')[-1][-1]  # Add the last character of the dataset name    
    
    merged_dataset_path = merged_dataset_path / dataset_name

    # Create symbolic links from the train, test and val folder in each of the datasets.
    for i, dataset in enumerate(sorted(args.remainder)):
        dataset_path = root_path / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"The specified dataset does not exist: {dataset_path}")
        
        # Make a unique identifier for the dataset
        unique_identifier = dataset[0] + dataset[-1] # e.g. s3 for shoebox/run3

        # Create a directory for the dataset configs.
        dataset_config_path = merged_dataset_path / 'configs'
        dataset_config_path.mkdir(parents=True, exist_ok=True)

        # Link the config.yml file 
        config_file_path = dataset_path / 'config.yml'
        if config_file_path.exists():
            symlink_target = dataset_config_path / f"{unique_identifier}_config.yml"
            try:
                symlink_target.symlink_to(config_file_path.resolve())
            except FileExistsError:
                print(f"Symlink already exists: {symlink_target}")

        for subdir in ['train', 'test', 'val']:
            source_dir = dataset_path / subdir
            target_dir = merged_dataset_path / subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            if not source_dir.exists():
                print(f"Warning: '{subdir}' directory does not exist in {dataset_path}. Skipping.")
                continue

            for folder_path in source_dir.iterdir():
                if folder_path.exists() and not folder_path.is_file():
                    # Construct unique symlink name to avoid collisions
                    symlink_name = f"{unique_identifier}_{folder_path.name}"
                    symlink_target = target_dir / symlink_name
                    try:
                        symlink_target.symlink_to(folder_path.resolve())
                    except FileExistsError:
                        print(f"Symlink already exists: {symlink_target}")

    # create a config file that contains information about the combined dataset
    config_file_path = merged_dataset_path / 'config.txt'
    with config_file_path.open('w') as f:
        f.write(f"Combined dataset: {dataset_name}\n")
        f.write(f"Number of datasets: {len(args.remainder)}\n")
        f.write("Datasets included:\n")
        for dataset in sorted(args.remainder):
            f.write(f"- {dataset}\n")
    print(f"Combined dataset created at: {merged_dataset_path}")

