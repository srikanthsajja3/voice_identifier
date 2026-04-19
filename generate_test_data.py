import numpy as np
import scipy.io.wavfile as wav
import os

def generate_synthetic_noise():
    base_dir = "./data"
    classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", 
               "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    
    sr = 22050
    duration = 4
    
    print(f"--- Generating Synthetic Dataset in {base_dir} ---")
    
    for i, cls in enumerate(classes):
        cls_path = os.path.join(base_dir, cls)
        os.makedirs(cls_path, exist_ok=True)
        
        for j in range(10):  # Create 10 samples per class
            t = np.linspace(0, duration, int(sr * duration))
            
            # Synthesize different sounds for each class
            if i == 8:   # Siren: Frequency sweep
                data = np.sin(2 * np.pi * (400 + 200 * np.sin(2 * np.pi * 0.5 * t)) * t)
            elif i == 1: # Car Horn: Two-tone steady sound
                data = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 660 * t)
            elif i == 6: # Gunshot: Short burst of noise
                data = np.random.uniform(-1, 1, len(t))
                data[int(sr*0.5):] = 0 # Silent after 0.5s
            else:        # Others: Modulated noise
                data = np.random.uniform(-0.5, 0.5, len(t)) * np.sin((i+1) * t)
            
            # Normalize and convert to 16-bit PCM
            data = (data / np.max(np.abs(data)) * 32767).astype(np.int16)
            
            file_name = os.path.join(cls_path, f"test_sample_{j}.wav")
            wav.write(file_name, sr, data)
            
    print("Done! You now have a test dataset to run the project.")

if __name__ == "__main__":
    generate_synthetic_noise()
