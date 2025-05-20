from tools.speech_assesment import evaluate_signals

# Load the audio files
Dry_sound = r"wav_files\relaxing-guitar-loop-v5-245859.wav"  # Replace with the path to your dry sound file
SAMPLE_BZ = r"bright_mic_filtered.wav"  # Replace with the path to your bright mic filtered sound file
SAMPLE_DZ = r"dark_mic_filtered.wav"  # Replace with the path to your dark mic filtered sound file
Original_BZ = r"bright_mic_original.wav"  # Replace with the path to your bright mic original sound file
Original_DZ = r"dark_mic_original.wav"  # Replace with the path to your dark mic original sound file

# Evaluate the signals
evaluate_signals(Dry_sound, SAMPLE_BZ, SAMPLE_DZ, Original_BZ, Original_DZ)