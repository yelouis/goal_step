# Phase 1: The Acoustic Characterization Pass (Global Scan)

## Overview
The goal of this phase is to analyze the entire audio track of an Ego4D video to identify its baseline acoustic profile. By calculating the Global Noise Floor ($N_g$) and applying stationary noise masking, we can suppress persistent background sounds (e.g., HVAC units) and distinguish between "active" (noisy actions) and "passive" (quiet) regions.

## Data Storage Notes
- Raw audio tracks should be read from `/Volumes/Extreme SSD/ego4d_data/audio/`.
- Spectrograms and noise profiles generated here should be cached to `/Volumes/Extreme SSD/ego4d_data/cache/phase1/`.

## Theoretical Pipeline

1. **Audio Extraction:** Extract the `.wav` track from the Ego4D `.mp4`.
2. **Frequency Domain Transformation:** Apply Short-Time Fourier Transform (STFT).
3. **Global Noise Floor ($N_g$):** Compute the median magnitude of the energy envelope, and its Median Absolute Deviation (MAD).
4. **Stationary Noise Masking:** Average frames that fall below $N_g + \text{MAD}$ to build a "quiet" spectral profile. Subtract this profile from the entire track.
5. **Pattern Discovery:** Flag long contiguous regions where energy is heavily suppressed as "passive / baseline" segments.

## Pseudocode Implementation

```python
import librosa
import numpy as np
import os
import scipy.stats

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
AUDIO_RATE = 16000 # Standardizing on 16kHz
N_FFT = 2048
HOP_LENGTH = 512

def extract_audio(video_id: str) -> np.ndarray:
    video_path = os.path.join(SSD_BASE, f"videos/{video_id}.mp4")
    # Load audio - librosa automatically converts to mono and target sr
    y, sr = librosa.load(video_path, sr=AUDIO_RATE)
    return y

def calculate_global_noise_floor(energy_envelope: np.ndarray) -> tuple:
    """
    Calculate Ng (Median) and MAD.
    """
    Ng = np.median(energy_envelope)
    mad = scipy.stats.median_abs_deviation(energy_envelope)
    return Ng, mad

def generate_stationary_mask(S_mag: np.ndarray, Ng: float, mad: float) -> np.ndarray:
    """
    Identify persistent frequency bands to create a spectral mask.
    S_mag shape: (frequencies, frames)
    """
    # Sum energy across frequencies for each frame
    frame_energies = np.sum(S_mag, axis=0)
    
    # Identify quiet frames (below the MAD threshold)
    quiet_threshold = Ng + (2.0 * mad)
    quiet_frame_indices = np.where(frame_energies < quiet_threshold)[0]
    
    if len(quiet_frame_indices) == 0:
        # Fallback if no quiet frames exist (rare)
        return np.zeros_like(S_mag[:, 0])
        
    # Build mask by averaging the spectra of all quiet frames
    quiet_spectra = S_mag[:, quiet_frame_indices]
    spectral_mask = np.mean(quiet_spectra, axis=1)
    
    return spectral_mask

def apply_acoustic_characterization(video_id: str):
    # 1. Load data
    audio_signal = extract_audio(video_id)
    
    # 2. Compute STFT and Magnitude
    S = librosa.stft(audio_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    
    # 3. Compute Energy Envelope
    energy_envelope = librosa.feature.rms(S=S_mag, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # 4. Get Baselines
    Ng, mad = calculate_global_noise_floor(energy_envelope)
    
    # 5. Masking
    spectral_mask = generate_stationary_mask(S_mag, Ng, mad)
    
    # Subtract mask (broadcasting across all frames), floor to 0
    S_mag_clean = np.maximum(S_mag - spectral_mask[:, np.newaxis], 0.0)
    
    # Recalculate clean energy envelope
    clean_energy_envelope = librosa.feature.rms(S=S_mag_clean, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # Save clean envelope to SSD for Phase 2
    save_path = os.path.join(SSD_BASE, f"cache/phase1/{video_id}_clean_energy.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, clean_energy_envelope)
    
    print(f"Phase 1 complete for {video_id}. Ng={Ng:.4f}, MAD={mad:.4f}")

if __name__ == '__main__':
    apply_acoustic_characterization("sample_ego4d_vid_001")
```

## Verification Strategy
- **Visual STFT Check:** To test that the masking logic truly suppresses stationary HVAC noise without deleting tool sounds, temporarily use `matplotlib.pyplot` to render `librosa.display.specshow` of `S_mag` vs `S_mag_clean`.
- **Unit Assertion:** Assert that taking `np.mean(S_mag_clean)` over known silent temporal regions in a test video is effectively `0.0`.
