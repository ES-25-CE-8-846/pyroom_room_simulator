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
logger = logging.getLogger(__name__)

def ac_evaluation(filters, bz_rirs, dz_rirs, plot=False):
    """Function to compute the ac
    Args:
        filters (torch.Tensor): the filters as a torch tensor shape (B, L, K) (batch, loudspeakers, filter coefficients)
        bz_rirs (torch.Tensor): the bright zone room impulse responses shape (B, M, L, N) (batch, mics, loudspeakers, rir length)
            n speakers, n microphones, impulse responses length
        dz_rirs (torch.Tensor): the dark zone room impulse responses shape (B, M, L, N)
            n speakers, n microphones, impulse responses length
    Returns:
        ac (float): the acoustic contrast
    """

    filters = filters[:,np.newaxis, :, :]

    # convolve filters with bz rirs
    filter_bz = np.sum(
        sps.fftconvolve(filters, bz_rirs, axes=3), axis=2, keepdims=True
    )
    filter_dz = np.sum(
        sps.fftconvolve(filters, dz_rirs, axes=3), axis=2, keepdims=True
    )

    if plot:
        # print("ac shapes")
        # print(bz_rirs.shape)
        # print(filter_dz.shape)
        # print(filter_bz[0,:,0,:].shape)
        # print("-------")
        plot_filters(filters=filter_bz[0,:,0,:], name="fft(filter,rir) for bz")
        plot_filters(filters=filter_dz[0,:,0,:], name="fft(filter,rir) dz")

    # compute rfft for bz and dz
    h_b = scipy.fft.rfftn(filter_bz, axes=3)
    h_d = scipy.fft.rfftn(filter_dz, axes=3)

    m_b = h_b.shape[2]
    m_d = h_d.shape[2]

    # Compute total energy
    E_b = np.sum(np.abs(h_b) ** 2)
    E_d = np.sum(np.abs(h_d) ** 2)

    # Compute acoustic contrast
    ac = 10 * np.log10((m_d * E_b) / (m_b * E_d))

    # compute mean across batch
    ac = np.mean(ac)

    # print(f"acc {acc}, acc_shape {acc.shape}, m_b {m_b}, m_d {m_d}")
    return ac

def convmtx(h, n):
    h = np.asarray(h).flatten()
    col = np.concatenate([h, np.zeros(n - 1)])
    row = np.zeros(n)
    row[0] = h[0]
    #return toeplitz(col, row).T # Transpose to match MATLAB's convmtx output, might be wrong though
    return spl.toeplitz(col, row)

def get_zone_convolution_mtx(rirs, M, K, L, J):
    """
    Create the correlation/convolution matrix, G, of sound zone RIRs
    """
    G = np.zeros((M * (K+J-1), L*J))
    for m in range(M):
        for ll in range(L):
            G[m*(K+J-1):(m+1)*(K+J-1), ll*J:(ll+1)*J] = spl.convolution_matrix(rirs[:, m, ll], J)
    return G.astype(np.float32)

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

def get_cov_mtx(BZ_rirs, DZ_rirs, M_b, M_d, K, L, J):
    """Compute the covariance matrices and cross-correlation vector for the BZ and DZ RIRs.

    Args:
        BZ_rirs (np.ndarray): RIR for the bright zone, shape (K, M_b, L).
        DZ_rirs (np.ndarray): RIR for the dark zone, shape (K, M_d, L).
        M_b (int): Number of microphones in the bright zone.
        M_d (int): Number of microphones in the dark zone.
        K (int): Length of the RIRs.
        L (int): Number of sources(loudspeakers).
        J (int): Length of the control filter.

    Returns:
        array: Covariance matrix R_B
        array: cross-correlation vector r_B
        array: Covariance matrix R_D.
    """
    # Correlation matrix G_B of the Bright Zone (BZ) RIRs
    logger.info("Calculating correlation matrix for BZ...")
    G_B = get_zone_convolution_mtx(BZ_rirs, M=M_b, K=K, L=L, J=J)
    
    desired_source = 1 # indexed from 0, source 1 is the top middle loudspeaker on the phone
    p_T = G_B[:, desired_source*J] # selecting the set of desired RIRs from the G_B matrix
    delay_d_B_samples = int(np.ceil(J/2)) # Initial delay added to maintain causality, here delay = half of control filter length
    d_B = np.concatenate((np.zeros(delay_d_B_samples), p_T[: -delay_d_B_samples])) # adding the initial delay to the desired RIRs. This will be used as the desired RIR in the optimization problem
    
    R_B = G_B.T @ G_B; # autocorrelation matrix for the bright zone (BZ)
    r_B = G_B.T @ d_B; # cross correlation vector for the bright zone (BZ)
    del G_B, d_B
    gc.collect()
    
    # Correlation matrix G_D of the Dark Zone (DZ) RIRs
    logger.info("Calculating correlation matrix for DZ...")
    G_D = get_zone_convolution_mtx(DZ_rirs, M=M_d, K=K, L=L, J=J)
    
    R_D = G_D.T @ G_D; # autocorrelation matrix for the dark zone (DZ)
    del G_D
    gc.collect()
    
    return R_B, r_B, R_D

def fit_vast(rank: int, mu: float, r_B: np.ndarray, eig_vec: np.ndarray, eig_val_vec: np.ndarray) -> np.ndarray:
        """Fit the VAST filter using the given rank."""
        weights = 1 / (mu + eig_val_vec[:rank])
        q_vast = eig_vec[:, :rank] @ (weights * (eig_vec[:, :rank].T @ r_B))
        return q_vast

def VAST(BZ_rirs, DZ_rirs, fs=48_000, J=1024, mu=1.0, reg_param=1e-5, acc=True, vast=True, pm=True):
    """Generate VAST control filters for the given RIRs.

    Args:
        BZ_rirs (np.ndarray): Numpy array of shape (mics, sources, length) representing the RIRs in the bright zone.
        DZ_rirs (np.ndarray): Numpy array of shape (mics, sources, length) representing the RIRs in the dark zone.
        fs (int, optional): The sampling frequency. Defaults to 48_000.
        J (int, optional): The filter size. Defaults to 1024.
        mu (float, optional): Importance on DZ power minimization. Defaults to 1.0.
        reg_param (float, optional): Regularization parameter. Defaults to 1e-5.

    Returns:
        dict: Returns a dict containing control filters calculated using the VAST method, this includes ACC, VAST and PM.
    """
    
    M_b = BZ_rirs.shape[0] # number of microphones in bright zone
    M_d = DZ_rirs.shape[0] # number of microphones in dark zone
    
    assert BZ_rirs.shape[1] == DZ_rirs.shape[1], "Number of sources in bright and dark zones must be the same"
    L = BZ_rirs.shape[1] # number of sources
    
    K = max(BZ_rirs.shape[2], DZ_rirs.shape[2]) # length of RIRs
    
    # Change rirs matrices to be of shape K x M x L, currently they are M x L x K (Matching matlab code)
    BZ_rirs = np.transpose(BZ_rirs, (2, 0, 1)) # K x M_b x L
    DZ_rirs = np.transpose(DZ_rirs, (2, 0, 1)) # K x M_d x L
    
    # Get the covariance matrices and cross-correlation vector
    logger.info("Calculating covariance matrices and cross-correlation vectors...")
    R_B, r_B, R_D = get_cov_mtx(BZ_rirs, DZ_rirs, M_b, M_d, K, L, J)
    
    ## VAST ##
    logger.info("Calculating joint diagonalization...")
    scaled_reg_matrix = np.eye(L*J) * reg_param # Regularization matrix for the joint diagonalization
    eig_vec, eig_val = jdiag(R_B, R_D + scaled_reg_matrix, descend=True) # Joint diagonalization of R_B and R_D
    
    # Using the VAST algorithm to calculate control filters for ACC method (rank=1) and PM method (rank = L*J)
    logger.info("Calculating VAST filters...")
    
    if acc:
        logger.info("Calculating ACC filter...")
        q_acc = fit_vast(rank=1, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val) # ACC method solution from VAST using 1st eig val and vec
    else: q_acc = None
    
    if vast:
        logger.info("Calculating VAST filter...")
        vast_rank = int(np.ceil(L*J/8)) # CHANGE THIS TO SELECT THE RANK OF VAST, 1 \leq vast_rank \leq L*J
        #print("VAST rank:", vast_rank)
        q_vast = fit_vast(rank=vast_rank, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val)
    else: q_vast = None
    
    if pm:
        logger.info("Calculating PM filter...")
        q_pm = fit_vast(rank=L*J, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val) # PM method solution from VAST using L*J eig vals and vecs (full rank)
    else: q_pm = None
    
    logger.info("VAST filters calculated successfully!")
    return { # Dictionary to store the filters, reshape them into filters for each source
        "q_acc": np.reshape(q_acc, (L, J)) if q_acc is not None else None,
        "q_vast": np.reshape(q_vast, (L, J)) if q_vast is not None else None,
        "q_pm": np.reshape(q_pm, (L, J)) if q_pm is not None else None,
        "config": {
            "fs": fs,
            "J": J,
            "mu": mu,
            "reg_param": reg_param,
        }
    }


def plot_filters(filters: np.ndarray, name: str):
    """Plot the filters in a grid of subplots."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, filters.shape[0], figsize=(15, 5))
    for n, filter in enumerate(filters):
        #print(f"Filter {n+1} shape:", filter.shape)
        ax[n].plot(filter)
        ax[n].set_title(f"{name} filter {n+1}")
        ax[n].set_xlabel("Samples")
        ax[n].set_ylabel("Amplitude")
    fig.suptitle(f"{name} filters")
    plt.tight_layout()
    plt.show()



def main():

    # Load rirs
    rirs_root = Path(f"{__file__}").parent / "dataset" / "shoebox" / "run1_sparse" / "test"
    rirs_paths = sorted(list(rirs_root.rglob("room_*/*.npz")))
    
    # log which filters have been processed
    with open(rirs_root / "processed_filters.txt", "w") as file:
        for item in rirs_paths:
            file.write(f"{item}\n")

    # VAST Hyperparams
    J = 4096 # Filter length
    mu = 1.0 # Importance on DZ power minimization
    reg_param = 1e-5 # Regularization parameter
    fs = 44100 # Sampling frequency
    plot_filt = True

    for rirs_path in tqdm(rirs_paths):
        rirs_dict = np.load(rirs_path)
        bz_rir, dz_rir = rirs_dict["bz_rir"].astype(np.float32), rirs_dict["dz_rir"].astype(np.float32)
        print("rir shapes:", bz_rir.shape, dz_rir.shape)
        del rirs_dict
        gc.collect()
        
        # Generate VAST filters
        filter_path = rirs_path.parent / f"filters_{J}_{mu}_{reg_param}" / rirs_path.name
        if filter_path.exists():
            filters = np.load(filter_path)
            print(f"VAST filters loaded from: {filter_path}")
        else:
            print(f"Computing VAST filters and saving to: {filter_path}")
            filters = VAST(bz_rir, dz_rir, fs=fs, J=J, mu=mu, reg_param=reg_param)
            os.makedirs(filter_path.parent, exist_ok=True)
            np.savez_compressed(filter_path, q_acc=filters["q_acc"], q_vast=filters["q_vast"], q_pm=filters["q_pm"])

        # Calculate the acoustic contrast
        print(f"filters shape:", filters["q_acc"].shape)
        bz_rir_eval = bz_rir[np.newaxis, :, :, :] # Reshaping to match nn output which has batches
        dz_rir_eval = dz_rir[np.newaxis, :, :, :]
        for name in ["q_acc", "q_vast", "q_pm"]:
            if filters[name] is not None:
                filter_eval = filters[name][np.newaxis, :, :]
                ac = ac_evaluation(filter_eval, deepcopy(bz_rir_eval), deepcopy(dz_rir_eval))
                print(f"AC for {name} filter:", ac)
                if plot_filt: plot_filters(filters=filters[name], name=name)
        
        # Use dirac delta filter as comparison
        filter_eval = np.zeros_like(filter_eval)
        filter_eval[0, :, 0] = 1.0
        ac = ac_evaluation(filter_eval, deepcopy(bz_rir_eval), deepcopy(dz_rir_eval))
        print(f"AC for dirac delta filter:", ac)
        if plot_filt: plot_filters(filters=filter_eval[0,:], name="dirac")


if __name__ == "__main__":
    main()