import os
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz, cholesky, schur
from tools import RoomSimulator
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


def convmtx(h, n):
    h = np.asarray(h).flatten()
    col = np.concatenate([h, np.zeros(n - 1)])
    row = np.zeros(n)
    row[0] = h[0]
    #return toeplitz(col, row).T # Transpose to match MATLAB's convmtx output, might be wrong though
    return toeplitz(col, row)

def jdiag(A: np.ndarray, B: np.ndarray, eva_option='matrix'):
    """
    Joint diagonalization via generalized eigenvalue problem:
        Aq = dBq
    Computes Q and D such that:
        Q.T @ A @ Q = D
        Q.T @ B @ Q = I
        inv(B) @ A @ Q = Q @ D
    
    Parameters:
        A (ndarray): (semi-)positive definite matrix
        B (ndarray): positive definite matrix
        eva_option (str): 'vector' to return D as 1D array, 'matrix' for diagonal matrix

    Returns:
        Q (ndarray): joint diagonalizer
        D (ndarray): diagonal matrix or vector of generalized eigenvalues
    """
    if eva_option not in ['matrix', 'vector']:
        raise ValueError("eva_option must be 'matrix' or 'vector'")

    try:
        # B = L @ L.T (Cholesky decomposition)
        logger.info("Calculating Cholesky decomposition...")
        Bc = cholesky(B, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix B is NOT positive definite.")

    # Transform A to standard eigenvalue problem: C = inv(L) @ A @ inv(L.T)
    logger.info("Transforming A to standard eigenvalue problem...")
    # TODO: Check if this is correct
    C: np.ndarray = np.linalg.solve(Bc, A) # Equivalent to inv(Bc) @ A
    C = np.linalg.solve(Bc.T, C.T).T  # Equivalent to inv(Bc) @ A @ inv(Bc).T
    
    # Schur decomposition of the symmetric matrix C = U @ T @ U.T
    logger.info("Calculating Schur decomposition...")
    T, U = schur(C)

    # Back-transform to get generalized eigenvectors
    logger.info("Back-transforming to get generalized eigenvectors...")
    X = np.linalg.solve(Bc.T, U)

    # Sort eigenvalues (descending) and re-order eigenvectors
    logger.info("Sorting eigenvalues and re-ordering eigenvectors...")
    dd = np.diag(T)
    dind = np.argsort(dd)[::-1]
    D_vals = dd[dind]
    Q = X[:, dind]

    if eva_option == 'matrix':
        D = np.diag(D_vals)
    else:  # eva_option == 'vector'
        D = D_vals

    return Q, D



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
    
    # Correlation matrix G_B of the Bright Zone (BZ) RIRs
    logger.info("Calculating correlation matrix for BZ...")
    G_B = np.zeros((M_b * (K+J-1), L*J))
    for m in range(M_b):
        for ll in range(L):
            G_B[m*(K+J-1):(m+1)*(K+J-1), ll*J:(ll+1)*J] = convmtx(BZ_rirs[:, m, ll], J)
    
    desired_source = 1
    p_T = G_B[:, (desired_source-1)*J+1] # selecting the set of desired RIRs from the G_B matrix
    delay_d_B_samples = int(np.ceil(J/2)) # Initial delay added to maintain causality, here delay = half of control filter length
    d_B = np.concatenate([np.zeros((delay_d_B_samples, 1)), p_T[: -delay_d_B_samples].reshape(-1, 1)], axis=0) # adding the initial delay to the desired RIRs. This will be used as the desired RIR in the optimization problem
    
    R_B = G_B.conjugate().T @ G_B; # autocorrelation matrix for the bright zone (BZ)
    r_B = G_B.conjugate().T @ d_B; # cross correlation vector for the bright zone (BZ)
    
    # Correlation matrix G_D of the Dark Zone (DZ) RIRs
    logger.info("Calculating correlation matrix for DZ...")
    G_D = np.zeros((M_d * (K+J-1), L*J))
    for m in range(M_d):
        for ll in range(L):
            G_D[m*(K+J-1):(m+1)*(K+J-1), ll*J:(ll+1)*J] = convmtx(DZ_rirs[:, m, ll], J)
    
    R_D = G_D.conjugate().T @ G_D; # autocorrelation matrix for the bright zone (BZ)
    
    ## VAST ##
    logger.info("Calculating joint diagonalization...")
    reg_matrix = np.eye(L*J)
    eig_vec, eig_val = jdiag(R_B, R_D + reg_param*reg_matrix, eva_option='matrix')
    eig_val_vec = np.diag(eig_val)
    
    # Using the VAST algorithm to calculate control filters for ACC method (rank=1) and PM method (rank = L*J)
    logger.info("Calculating VAST filters...")
    
    if acc:
        logger.info("Calculating ACC filter...")
        # q_acc = (1/(mu+eig_val_vec[0]))*(eig_vec[:,0].T * r_B)*eig_vec[:,0] # acc method solution from VAST using 1st eig val and vec
        q_acc = (1 / (mu + eig_val_vec[0])) * (eig_vec[:, 0].conjugate().T @ r_B) * eig_vec[:, 0] # ACC method solution from VAST using 1st eig val and vec
    else: q_acc = None # ACC filter is not calculated
    
    if vast:
        logger.info("Calculating VAST filter...")
        vast_rank = int(np.ceil(L*J/8)) # CHANGE THIS TO SELECT THE RANK OF VAST, 1 \leq vast_rank \leq L*J
        q_vast = np.zeros((J*L,))
        print("VAST rank:", vast_rank)
        print("VAST filter shape:", q_vast.shape)
        for v in tqdm(range(vast_rank)):
            q_vast = q_vast + ((1 / (mu + eig_val_vec[v])) * (eig_vec[:, v].conjugate().T @ r_B) * eig_vec[:, v]) #VAST control filter
            logger.info(f"VAST filter {v+1}/{vast_rank} calculated")
            logger.info(f"VAST filter shape: {q_vast.shape}")
    else: q_vast = None # VAST filter is not calculated
    
    if pm:
        logger.info("Calculating PM filter...")
        q_pm = np.zeros((J*L,))
        for v in tqdm(range(J*L)):
            q_pm = q_pm + (1 / (mu + eig_val_vec[v])) * (eig_vec[:, v].conjugate().T @ r_B) * eig_vec[:, v] #Pressure Matching method solution, calculated using vast Considering full rank, i.e., L*J
    else: q_pm = None # PM filter is not calculated
    
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
    signal = signal / np.max(np.abs(signal) + 1e-8)  # Normalize

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
    filters = VAST(bz_rir, dz_rir, fs=fs, J=2048, mu=1.0, reg_param=1e-1, acc=True, vast=True, pm=True)
    
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
    global_scaling_factor = np.max(np.abs(original_mic_signals)) + 1e-8  # Avoid division by zero
    original_bright_mic_signal = original_mic_signals[room_params["n_mics"]]/global_scaling_factor  # First bright mic
    original_dark_mic_signal = original_mic_signals[0]/global_scaling_factor  # First dark mic
    
    # Save original mic signals to WAV files
    os.remove("bright_mic_original.wav") if os.path.exists("bright_mic_original.wav") else None
    os.remove("dark_mic_original.wav") if os.path.exists("dark_mic_original.wav") else None
    sf.write("bright_mic_original.wav", original_bright_mic_signal, fs)
    sf.write("dark_mic_original.wav", original_dark_mic_signal, fs)

    import matplotlib.pyplot as plt
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
    for filter in filters["q_pm"]:
        filtered_signal = fftconvolve(signal, filter)[:len(signal)]
        filtered_signals.append(filtered_signal)
        
    # Inject filtered signals into room
    for i, src in enumerate(room_sim.room.sources):
        src.signal = filtered_signals[i]
        
    print("Simulating room with filtered signal...")
    room_sim.room.simulate()
    print("Simulation done")
    
    filtered_mic_signals = room_sim.room.mic_array.signals  # shape: [num_mics, signal_len]
    filtered_bright_mic_signal = filtered_mic_signals[room_params["n_mics"]] / global_scaling_factor  # First bright mic
    filtered_dark_mic_signal = filtered_mic_signals[0] / global_scaling_factor  # First dark mic
    
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
    sf.write("bright_mic_filtered.wav", filtered_bright_mic_signal, fs)
    sf.write("dark_mic_filtered.wav", filtered_dark_mic_signal, fs)

if __name__ == "__main__":
    main()


