import torch
import torchaudio
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import torchaudio.functional as F

def plot(waveform, title, sample_rate=16000):
    wav_numpy = waveform.numpy()
    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / sample_rate
    figure, axes = plt.subplots(2, 1)
    axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
    axes[0].grid(True)
    axes[1].specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)

def evaluate_signals(Dry_sound, SAMPLE_BZ, SAMPLE_DZ, Original_BZ, Original_DZ, plot=False):
    """
    Prints the estimated metrics (STOI, PESQ, MOS) for the original and filtered signals using the SQUIM model.

    Input:
    Dry_sound: Path to the dry sound file
    SAMPLE_BZ: Path to the bright mic filtered sound file
    SAMPLE_DZ: Path to the dark mic filtered sound file
    Original_BZ: Path to the bright mic original sound file
    Original_DZ: Path to the dark mic original sound file
    plot: Boolean to plot the waveforms

    """
    WAVEFORM_DRY, SAMPLE_RATE_DRY = torchaudio.load(Dry_sound)
    WAVEFORM_BZ, SAMPLE_RATE_BZ = torchaudio.load(SAMPLE_BZ)
    WAVEFORM_DZ, SAMPLE_RATE_DZ = torchaudio.load(SAMPLE_DZ)
    WAVEFORM_BZ_ORIGINAL, SAMPLE_RATE_BZ_ORIGINAL = torchaudio.load(Original_BZ)
    WAVEFORM_DZ_ORIGINAL, SAMPLE_RATE_DZ_ORIGINAL = torchaudio.load(Original_DZ)
    # Convert to mono if needed
    WAVEFORM_DRY = WAVEFORM_DRY[0:1, :]
    WAVEFORM_DZ = WAVEFORM_DZ[0:1, :]
    WAVEFORM_BZ = WAVEFORM_BZ[0:1, :]  
    Original_DZ = WAVEFORM_DZ_ORIGINAL[0:1, :]  
    Original_BZ = WAVEFORM_BZ_ORIGINAL[0:1, :]  

    # Resample to 16kHz
    if SAMPLE_RATE_DRY != 16000:
        WAVEFORM_DRY = F.resample(WAVEFORM_DRY, SAMPLE_RATE_DRY, 16000)
    if SAMPLE_RATE_BZ != 16000:
        WAVEFORM_BZ = F.resample(WAVEFORM_BZ, SAMPLE_RATE_BZ, 16000)
    if SAMPLE_RATE_DZ != 16000:
        WAVEFORM_DZ = F.resample(WAVEFORM_DZ, SAMPLE_RATE_DZ, 16000)
    if SAMPLE_RATE_BZ_ORIGINAL != 16000:
        Original_BZ = F.resample(Original_BZ, SAMPLE_RATE_BZ_ORIGINAL, 16000)
    if SAMPLE_RATE_DZ_ORIGINAL != 16000:
        Original_DZ = F.resample(Original_DZ, SAMPLE_RATE_DZ_ORIGINAL, 16000)

    # Trim to shortest length
    min_len = min(WAVEFORM_DRY.shape[1], WAVEFORM_BZ.shape[1], WAVEFORM_DZ.shape[1], Original_BZ.shape[1], Original_DZ.shape[1])
    WAVEFORM_DRY = WAVEFORM_DRY[:, :min_len]
    WAVEFORM_BZ = WAVEFORM_BZ[:, :min_len]
    WAVEFORM_DZ = WAVEFORM_DZ[:, :min_len]
    Original_BZ = Original_BZ[:, :min_len]
    Original_DZ = Original_DZ[:, :min_len]

    # Plot waveforms
    if plot == True:
        plot(WAVEFORM_BZ, "BZ Mic Signal")
        plot(WAVEFORM_DZ, "DZ Mic Signal")
        plot(Original_BZ, "BZ Original Signal")
        plot(Original_DZ, "DZ Original Signal")
        plt.show()

    # ML-predicted metrics (subjective-ish)
    objective_model = SQUIM_OBJECTIVE.get_model() # TODO: CALCULATE THIS WITHOUT SQUIM
    subjective_model = SQUIM_SUBJECTIVE.get_model()

    # Original DZ
    stoi_dz, pesq_dz, si_sdr_dz = objective_model(Original_DZ[0:1, :])
    dz_mos = subjective_model(Original_DZ, WAVEFORM_DRY)
    print(f"SQUIM Estimated metrics for original DZ:")
    print(f"STOI: {stoi_dz[0]:.3f}")
    print(f"PESQ: {pesq_dz[0]:.3f}")
    print(f"MOS: {dz_mos[0]:.3f}")
    #print(f"SI-SDR: {si_sdr_dz[0]:.3f}")

    # Filtered DZ
    stoi_dz, pesq_dz, si_sdr_dz = objective_model(WAVEFORM_DZ[0:1, :])
    dz_mos = subjective_model(WAVEFORM_DZ, WAVEFORM_DRY)
    print(f"SQUIM Estimated metrics for filtered DZ:")
    print(f"STOI: {stoi_dz[0]:.3f}")
    print(f"PESQ: {pesq_dz[0]:.3f}")
    print(f"MOS: {dz_mos[0]:.3f}")
    #print(f"SI-SDR: {si_sdr_dz[0]:.3f}")

    # Original BZ
    stoi_bz, pesq_bz, si_sdr_bz = objective_model(Original_BZ[0:1, :])
    bz_mos = subjective_model(Original_BZ, WAVEFORM_DRY)
    print(f"SQUIM Estimated metrics for original BZ:")
    print(f"STOI: {stoi_bz[0]:.3f}")
    print(f"PESQ: {pesq_bz[0]:.3f}")
    print(f"MOS: {bz_mos[0]:.3f}")
    #print(f"SI-SDR: {si_sdr_bz[0]:.3f}")
    
    # Filtered BZ
    stoi_bz, pesq_bz, si_sdr_bz = objective_model(WAVEFORM_BZ[0:1, :]) # TODO: Ensure this works
    bz_mos = subjective_model(WAVEFORM_BZ, WAVEFORM_DRY)
    print(f"SQUIM Estimated metrics for filtered BZ:")
    print(f"STOI: {stoi_bz[0]:.3f}")
    print(f"PESQ: {pesq_bz[0]:.3f}")
    print(f"MOS: {bz_mos[0]:.3f}")
    #print(f"SI-SDR: {si_sdr_bz[0]:.3f}")



if __name__ == "__main__":
    # Load the audio files
    Dry_sound = r"Path\to\your\dry_sound.wav"  # Replace with the path to your dry sound file
    SAMPLE_BZ = r"Path\to\your\bright_mic_filtered.wav"  # Replace with the path to your bright mic filtered sound file
    SAMPLE_DZ = r"Path\to\your\dark_mic_filtered.wav"  # Replace with the path to your dark mic filtered sound file
    Original_BZ = r"Path\to\your\bright_mic_original.wav"  # Replace with the path to your bright mic original sound file
    Original_DZ = r"Path\to\your\dark_mic_original.wav"  # Replace with the path to your dark mic original sound file

    # Evaluate the signals
    evaluate_signals(Dry_sound, SAMPLE_BZ, SAMPLE_DZ, Original_BZ, Original_DZ)