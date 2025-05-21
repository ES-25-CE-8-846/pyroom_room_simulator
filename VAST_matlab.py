import os
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz, convolution_matrix, eigh
from tools import RoomSimulator
import logging
logger = logging.getLogger(__name__)


def convmtx(h, n):
    h = np.asarray(h).flatten()
    col = np.concatenate([h, np.zeros(n - 1)])
    row = np.zeros(n)
    row[0] = h[0]
    #return toeplitz(col, row).T # Transpose to match MATLAB's convmtx output, might be wrong though
    return toeplitz(col, row)

def get_zone_convolution_mtx(rirs, M, K, L, J):
    """
    Create the correlation/convolution matrix, G, of sound zone RIRs
    """
    G = np.zeros((M * (K+J-1), L*J))
    for m in range(M):
        for ll in range(L):
            G[m*(K+J-1):(m+1)*(K+J-1), ll*J:(ll+1)*J] = convolution_matrix(rirs[:, m, ll], J)
    return G

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
    eig_val, eig_vec = eigh(A, B)  # joint diagonalization
    if descend:
        # sorting the eigenvalues in descending order
        idx = eig_val.argsort()[::-1]
        return eig_vec[:, idx], eig_val[idx]
    return eig_vec, eig_val

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
    
    desired_source = 1 # indexed from 0
    p_T = G_B[:, desired_source*J] # selecting the set of desired RIRs from the G_B matrix
    delay_d_B_samples = int(np.ceil(J/2)) # Initial delay added to maintain causality, here delay = half of control filter length
    d_B = np.concatenate((np.zeros(delay_d_B_samples), p_T[: -delay_d_B_samples])) # adding the initial delay to the desired RIRs. This will be used as the desired RIR in the optimization problem
    
    R_B = G_B.T @ G_B; # autocorrelation matrix for the bright zone (BZ)
    r_B = G_B.T @ d_B; # cross correlation vector for the bright zone (BZ)
    
    # Correlation matrix G_D of the Dark Zone (DZ) RIRs
    logger.info("Calculating correlation matrix for DZ...")
    G_D = get_zone_convolution_mtx(DZ_rirs, M=M_d, K=K, L=L, J=J)
    
    R_D = G_D.T @ G_D; # autocorrelation matrix for the dark zone (DZ)
    
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
    print("Number of microphones in bright zone:", M_b)
    print("Number of microphones in dark zone:", M_d)
    
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
        print("VAST rank:", vast_rank)
        q_vast = fit_vast(rank=vast_rank, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val)
    else: q_vast = None
    
    if pm:
        logger.info("Calculating PM filter...")
        q_pm = fit_vast(rank=L*J, mu=mu, r_B=r_B, eig_vec=eig_vec, eig_val_vec=eig_val) # PM method solution from VAST using L*J eig vals and vecs (full rank)
    else: q_pm = None
    
    logger.info("VAST filters calculated successfully!")
    return { # Dictionary to store the filters, reshape them into filters for each source
        "q_acc": np.reshape(q_acc, (J, L)).T if q_acc is not None else None,
        "q_vast": np.reshape(q_vast, (J, L)).T if q_vast is not None else None,
        "q_pm": np.reshape(q_pm, (J, L)).T if q_pm is not None else None,
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
        print(f"Filter {n+1} shape:", filter.shape)
        ax[n].plot(filter)
        ax[n].set_title(f"{name} filter {n+1}")
        ax[n].set_xlabel("Samples")
        ax[n].set_ylabel("Amplitude")
    fig.suptitle(f"{name} filters")
    plt.tight_layout()
    plt.show()
    


def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    )
    
    # specify signal source
    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono
    #signal = signal / np.max(np.abs(signal) + 1e-8)  # Normalize
    
    target_gain = 0.5  # Target gain for the signal
    signal_norm_gain = target_gain / np.sqrt(np.mean(signal**2))  # Calculate the normalization gain
    signal = signal * signal_norm_gain  # Apply the normalization gain to the signal

    room_params = {
        "fs": fs,
        "n_mics": 12,
        "mic_radius": 0.5,
        "shape": "shoebox",
        "signal": signal,
        "room_bounds": {
            "min_width": 3.0, 
            "max_width": 10.0, 
            "min_length": 3.0, 
            "max_length": 10.0, 
            "min_extrude": 2.0, 
            "max_extrude": 5.0
        },
        "material_properties_bounds": {
                "energy_absorption": (0.6, 0.9),
                "scattering": (0.05, 0.1),
        },
        # "ray_tracing_params": {
        #     "receiver_radius": 0.05,
        #     "n_rays": 10000,
        #     "energy_thres": 1e-7,
        # },
    }

    ### Generate room shiz ###
    room_sim = RoomSimulator(seed=42)
    room_sim.compose_room(**room_params)
    # room_sim.plot_room()
    rirs, rt60s = room_sim.compute_rir(rt60=True, plot=False) # Compute the RIRs and RT60 times
    reg_rirs = room_sim.regularize_rir(rirs, rt60s) # Regularize the RIRs to the RT60 length
    (bz_rir, dz_rir), (bz_rt60s, dz_rt60s) = room_sim.get_zones(rirs=reg_rirs, rt60=rt60s) # Split the RIRs and RT60s into bright and dark zones
    print("BZ RIR shape:", bz_rir.shape)
    print("DZ RIR shape:", dz_rir.shape)
    ##########################
    
    # Generate VAST filters
    J = 4096 # Filter length
    mu = 0.1 # Importance on DZ power minimization
    reg_param = 1e-5 # Regularization parameter
    filter_path = Path(f"filters/vast_filters_{J}_{mu}_{reg_param}.npz")
    if filter_path.exists():
        filters = np.load(filter_path)
        print(f"VAST filters loaded from {filter_path}")
    else:
        filters = VAST(bz_rir, dz_rir, fs=fs, J=J, mu=mu, reg_param=reg_param)
        os.makedirs("filters", exist_ok=True)  # Create the filters directory if it doesn't exist
        np.savez_compressed(filter_path, q_acc=filters["q_acc"], q_vast=filters["q_vast"], q_pm=filters["q_pm"])  # Save the filters to a compressed npz file
        print(f"VAST filters saved to {filter_path}")
    
    # Calculate the acoustic contrast
    from evaluation import acc_evaluation
    from copy import deepcopy
    print(f"filter shape:", filters["q_acc"].shape)
    bz_rir_eval = bz_rir[np.newaxis, :, :, :]
    dz_rir_eval = dz_rir[np.newaxis, :, :, :]
    for name in ["q_acc", "q_vast", "q_pm"]:
        if filters[name] is not None:
            filter_eval = filters[name][np.newaxis, :, :]
            ac = acc_evaluation(filter_eval, deepcopy(bz_rir_eval), deepcopy(dz_rir_eval)) # Call the acc_evaluation function to calculate the acoustic contrast
            print(f"AC for {name} filter:", ac)
        
    # For dirac delta filter
    filter_eval = np.zeros_like(filter_eval)
    filter_eval[0, 0, 0] = 1.0
    ac = acc_evaluation(filter_eval, deepcopy(bz_rir_eval), deepcopy(dz_rir_eval))
    print(f"AC for dirac delta filter:", ac)
    
    # Plot the filters
    for name in ["q_acc", "q_vast", "q_pm"]:
        if filters[name] is not None:
            print(f"{name} filter shape:", filters[name].shape)
            plot_filters(filters[name], name)
        
    
    # I do something with the filters
    room_sim.room.image_source_model()
    print("Simulating room acoustics...")
    room_sim.room.simulate()
    print("Simulation complete")
    
    original_mic_signals = room_sim.room.mic_array.signals  # shape: [num_mics, signal_len]
    
    # Make a global scaling factor to avoid clipping when saving to WAV
    #global_scaling_factor = np.max(np.abs(original_mic_signals)) + 1e-8  # Avoid division by zero
    original_bright_mic_signal = original_mic_signals[room_params["n_mics"]]*signal_norm_gain#/global_scaling_factor  # First bright mic
    original_dark_mic_signal = original_mic_signals[0]*signal_norm_gain#/global_scaling_factor  # First dark mic
    
    # Save original mic signals to WAV files
    os.remove("bright_mic_original.wav") if os.path.exists("bright_mic_original.wav") else None
    os.remove("dark_mic_original.wav") if os.path.exists("dark_mic_original.wav") else None
    wavfile.write("bright_mic_original.wav", fs, original_bright_mic_signal)
    wavfile.write("dark_mic_original.wav", fs, original_dark_mic_signal)

    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(original_bright_mic_signal, label="Bright Zone Mic 1 Signal")
    plt.plot(original_dark_mic_signal, label="Dark Zone Mic 1 Signal")
    plt.title("Signal Comparison Between original Bright and Dark Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(r"results/original_bright_dark_zone_comparison.png")
    #plt.show()
    
    # Convolve the original mic signals with the RIRs to get the filtered signals
    filtered_signals = []
    for filter in filters["q_vast"]:
        filtered_signal = fftconvolve(signal, filter)[:len(signal)]
        filtered_signals.append(filtered_signal)
    
    # Inject filtered signals into room
    for i, src in enumerate(room_sim.room.sources):
        src.signal = filtered_signals[i]
        
    print("Simulating room with filtered signal...")
    room_sim.room.simulate()
    print("Simulation done")
    
    filtered_mic_signals = room_sim.room.mic_array.signals  # shape: [num_mics, signal_len]
    filtered_bright_mic_signal = filtered_mic_signals[room_params["n_mics"]]*signal_norm_gain# / global_scaling_factor  # First bright mic
    filtered_dark_mic_signal = filtered_mic_signals[0]*signal_norm_gain# / global_scaling_factor  # First dark mic
    
    # Compare the filtered signals from bright and dark zone
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_bright_mic_signal, label="Bright Zone Mic 1 Signal", alpha=0.9)
    plt.plot(filtered_dark_mic_signal, label="Dark Zone Mic 1 Signal", alpha=0.9)
    plt.title("Signal Comparison Between Bright and Dark Zone Mics after VAST")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(r"results/filtered_bright_dark_zone_comparison.png")

    #
    plt.figure(figsize=(12, 6))
    plt.plot(original_bright_mic_signal, label="Original Bright Zone Mic 1 Signal", alpha=0.9)
    plt.plot(filtered_bright_mic_signal, label="Filtered Bright Zone Mic 1 Signal", alpha=0.9)
    plt.title("Signal Comparison Between original Bright and Filtered Bright Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(r"results/bright_zone_comparison.png")

    plt.figure(figsize=(12, 6))
    plt.plot(original_dark_mic_signal, label="Original Dark Zone Mic 1 Signal", alpha=0.9)
    plt.plot(filtered_dark_mic_signal, label="Filtered Dark Zone Mic 1 Signal", alpha=0.9)
    plt.title("Signal Comparison Between original Dark and Filtered Dark Zone Mics")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(r"results/dark_zone_comparison.png")
    
    # Save to WAV
    wavfile.write("bright_mic_filtered.wav", fs, filtered_bright_mic_signal)
    wavfile.write("dark_mic_filtered.wav", fs, filtered_dark_mic_signal)

if __name__ == "__main__":
    main()


