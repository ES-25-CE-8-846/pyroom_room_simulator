'''
A script for computing the average filters for the vast, acc, and pm algorithms based on the covariance matrices and cross-correlation vectors of the RIRs.
'''
import os
import gc
import scipy
import logging
import numpy as np
import scipy.linalg as spl
import scipy.signal as sps
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy


def jdiag(A: np.ndarray, B: np.ndarray, descend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint diagonalization of two matrices A and B.
    
    Parameters:
        A (ndarray): A matrix
        B (ndarray): B matrix
        descending (bool): If True, sort eigenvalues in descending order

    Returns:
        ndarray: joint diagonalization matrix (eigenvectors)
        ndarray: diagonal matrix of eigenvalues (eigenvalues)
    """
    eig_val, eig_vec = spl.eigh(A, B)  # joint diagonalization
    if descend:
        # sorting the eigenvalues in descending order
        idx = eig_val.argsort()[::-1]
        return eig_vec[:, idx].astype(np.float32), eig_val[idx].astype(np.float32)
    return eig_vec.astype(np.float32), eig_val.astype(np.float32)

def fit_vast(rank: int, mu: float, r_B: np.ndarray, eig_vec: np.ndarray, eig_val_vec: np.ndarray) -> np.ndarray:
        """Fit the VAST filter using the given rank."""
        weights = 1 / (mu + eig_val_vec[:rank])
        q_vast = eig_vec[:, :rank] @ (weights * (eig_vec[:, :rank].T @ r_B))
        return q_vast

def VAST_from_cov(R_B, r_B, R_D, L=3, fs=48_000, J=1024, mu=1.0, reg_param=1e-5, acc=True, vast=True, pm=True):
    """Generate VAST control filters for the given covariance matrices.

    Args:
        R_B (np.ndarray): Covariance matrix for the bright zone, shape (L*J, L*J).
        r_B (np.ndarray): Cross-correlation vector for the bright zone, shape (L*J,).
        R_D (np.ndarray): Covariance matrix for the dark zone, shape (L*J, L*J).
        L (int, optional): Number of sources (loudspeakers). Defaults to 3.
        fs (int, optional): The sampling frequency. Defaults to 48_000.
        J (int, optional): The filter size. Defaults to 1024.
        mu (float, optional): Importance on DZ power minimization. Defaults to 1.0.
        reg_param (float, optional): Regularization parameter. Defaults to 1e-5.

    Returns:
        dict: Returns a dict containing control filters calculated using the VAST method, this includes ACC, VAST and PM.
    """
    
    # Ensure the covariance matrices and cross-correlation vector are in the correct shape
    assert R_B.shape == (L*J, L*J), "R_B must be of shape (L*J, L*J)"
    assert r_B.shape == (L*J,), "r_B must be of shape (L*J,)"
    assert R_D.shape == (L*J, L*J), "R_D must be of shape (L*J, L*J)"

    ## VAST ##
    print("Calculating joint diagonalization...")
    scaled_reg_matrix = np.eye(L*J) * reg_param # Regularization matrix for the joint diagonalization
    eig_vec, eig_val = jdiag(R_B, R_D + scaled_reg_matrix, descend=True) # Joint diagonalization of R_B and R_D
    
    # Using the VAST algorithm to calculate control filters for ACC method (rank=1) and PM method (rank = L*J)
    print("Calculating VAST filters...")
    
    if acc:
        print("Calculating ACC filter...")
        q_acc = fit_vast(rank=1, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val) # ACC method solution from VAST using 1st eig val and vec
    else: q_acc = None
    
    if vast:
        print("Calculating VAST filter...")
        vast_rank = int(np.ceil(L*J/8)) # CHANGE THIS TO SELECT THE RANK OF VAST, 1 \leq vast_rank \leq L*J
        q_vast = fit_vast(rank=vast_rank, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val)
    else: q_vast = None
    
    if pm:
        print("Calculating PM filter...")
        q_pm = fit_vast(rank=L*J, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val) # PM method solution from VAST using L*J eig vals and vecs (full rank)
    else: q_pm = None
    
    print("VAST filters calculated successfully!")
    return { # Dictionary to store the filters, reshape them into filters for each source
        "q_acc": np.reshape(q_acc, (L, J)) if q_acc is not None else None,
        "q_vast": np.reshape(q_vast, (L, J)) if q_vast is not None else None,
        "q_pm": np.reshape(q_pm, (L, J)) if q_pm is not None else None,
        "R_B": R_B,
        "r_B": r_B,
        "R_D": R_D,
        "config": {
            "fs": fs,
            "J": J,
            "mu": mu,
            "reg_param": reg_param,
        }
    }

def main():

    # Define range to compute
    low = 0
    high = 500

    # Load covariance matrices and cross-correlation vectors
    rirs_root = Path(__file__).parent / "dataset" / "shoebox" / "run_post_hand_in" / "train"
    rirs_paths = sorted(list(rirs_root.rglob("room_*/0000.npz")))[low:high]


    # VAST Hyperparams
    J = 4096 # Filter length
    mu = 1.0 # Importance on DZ power minimization
    reg_param = 1e-5 # Regularization parameter
    fs = 44100 # Sampling frequency

    avg_R_B, avg_r_B, avg_R_D = None, None, None
    count = 0

    # Iterative averaging as i'm unsure whether the covariance matrices would fit into memory...
    for rirs_path in tqdm(rirs_paths):
        covariance_path = rirs_path.parent / f"covariance_{J}" / rirs_path.name
        if not covariance_path.exists():
            print(f"Skipping {covariance_path}, not found.")
            continue

        with np.load(covariance_path) as cov:
            R_B = cov["R_B"].astype(np.float128)
            r_B = cov["r_B"].astype(np.float128)
            R_D = cov["R_D"].astype(np.float128)

            if avg_R_B is None:
                avg_R_B = np.zeros_like(R_B)
                avg_r_B = np.zeros_like(r_B)
                avg_R_D = np.zeros_like(R_D)

            avg_R_B += R_B
            avg_r_B += r_B
            avg_R_D += R_D
            count += 1

    avg_R_B /= count
    avg_r_B /= count
    avg_R_D /= count

    # Optionally save the averaged matrices
    average_folder_path = rirs_root / f"average_filters"
    average_folder_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(average_folder_path / "averaged_covariance.npz", R_B=avg_R_B, R_D=avg_R_D)

    print(f"Averaged {count} matrices.")
    print(f"Saved in {average_folder_path}")

    # Compute the average filters using the averaged covariance matrices
    avg_filters = VAST_from_cov(avg_R_B, avg_r_B, avg_R_D, L=3, fs=fs, J=J, mu=mu, reg_param=reg_param)
    np.savez_compressed(average_folder_path / "averaged_filters.npz", q_acc=avg_filters["q_acc"], q_vast=avg_filters["q_vast"], q_pm=avg_filters["q_pm"])

if __name__ == "__main__":
    main()
