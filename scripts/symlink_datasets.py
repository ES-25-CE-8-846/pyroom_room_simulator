import numpy as np
import pathlib as path
import argparse


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Create symbolic links to datasets.")
    parser.add_argument("--data_root", type=str, help="Root directory for datasets.")
    parser.add_argument("--name", type=str, help="Name of the merged dataset.")
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
    
    # Create symbolic links to the specified datasets
    for i, dataset in enumerate(args.remainder):
        dataset_path = root_path / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"The specified dataset does not exist: {dataset_path}")
        
        # Split the dataset path into parts
        dataset_parts = dataset_path.parts

        
        # Create a symbolic link in the merged dataset directory
        symlink_path = merged_dataset_path / f'{i}_{dataset_parts[-2]}_{dataset_parts[-1]}' # <--- Change this once we know the naming convention
        symlink_path.symlink_to(dataset_path, target_is_directory=True)
        print(f"Created symbolic link: {symlink_path} -> {dataset_path}")
