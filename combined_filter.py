import numpy as np
from pathlib import Path
from tqdm import tqdm

dataset_root = Path(__file__).parent / "dataset" / "shoebox" / "run2"
train_path = dataset_root / "train"


train_filter_paths = sorted(list(train_path.rglob("filters_4096*/*.npz")))

filter_methods = ["q_acc", "q_vast", "q_pm"]
combined_filters = np.zeros((len(filter_methods), 3, 4096))

for n, method in tqdm(enumerate(filter_methods), total=len(filter_methods)):
    combined_filter: np.ndarray = np.zeros_like(np.load(train_filter_paths[0])[method]) # (3,4096)

    for filter_path in train_filter_paths:
        combined_filter += np.load(filter_path)[method]
    
    combined_filter /= len(train_filter_paths)
    combined_filters[n] = combined_filter

save_path = train_path / "combined_filters.npz"
np.savez_compressed(save_path, q_acc=combined_filters[0], q_vast=combined_filters[1], q_pm=combined_filters[2])
print(f"Combined filters saved to: {save_path}")
    


    







