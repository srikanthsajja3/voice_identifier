import librosa
import numpy as np

def load_and_preprocess(file_path, duration=4, sr=22050):
    """
    Loads a standard WAV file and converts it into a normalized Mel-Spectrogram (128x173).
    """
    # Direct load with librosa (works natively for standard WAV)
    audio, _ = librosa.load(file_path, duration=duration, sr=sr)
    
    # Ensure all files are the same length
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, sr * duration - len(audio)))
    elif len(audio) > sr * duration:
        audio = audio[:sr * duration]
        
    # Generate Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Min-Max Normalization (Scale to 0-1)
    norm_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))
    
    # Reshape for CNN input (Height, Width, Channel)
    return norm_spectrogram[..., np.newaxis]
