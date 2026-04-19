import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

def save_waveform(file_path, output_path):
    """Generates and saves the waveform (time-domain) image."""
    audio, sr = librosa.load(file_path, duration=4, sr=22050)
    plt.figure(figsize=(10, 2))
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color='#38bdf8')
    plt.title("Audio Waveform")
    plt.axis('off')
    plt.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_mfcc(file_path, output_path):
    """Generates and saves the MFCC (texture) image."""
    audio, sr = librosa.load(file_path, duration=4, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 2))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.axis('off')
    plt.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

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
