import numpy as np
import librosa
import scipy.stats

def get_bispectrum_features(audio_array, sr=16000, n_fft=512):
    """
    Computes a simplified 1D Bispectral vector from audio.
    """
    # 1. Compute STFT (Short-Time Fourier Transform)
    # Result shape: (freq_bins, time_frames)
    S = librosa.stft(audio_array, n_fft=n_fft)
    S_complex = np.abs(S) * np.exp(1j * np.angle(S)) # Keep complex info

    # 2. Bispectrum Computation (Simplified for Speed)
    # B(f1, f2) = X(f1) * X(f2) * X*(f1+f2)
    # We only take the diagonal to save time: B(f, f)
    # This detects phase coupling at specific harmonics.
    
    # Get indices
    freq_bins = S.shape[0]
    
    # We calculate interaction of f with itself (Diagonal slice of Bispectrum)
    # We only go up to freq_bins/2 because f1+f2 must exist in the spectrum
    limit = freq_bins // 2
    
    b_mag = []
    
    for f in range(limit):
        # The terms: X(f), X(f), and X*(2f)
        term1 = S_complex[f, :]
        term2 = S_complex[f, :]
        term3 = np.conjugate(S_complex[2*f, :])
        
        # Bispectrum slice
        B_slice = term1 * term2 * term3
        
        # Average over time to get a single value per frequency
        b_mag.append(np.mean(np.abs(B_slice)))

    # 3. Downsample/Interpolate to fixed size (e.g., 128 dimensions)
    # Neural Networks need fixed input sizes.
    b_vector = np.array(b_mag)
    
    # Resize to target dimension (128) using interpolation
    target_dim = 128
    if len(b_vector) != target_dim:
        x_old = np.linspace(0, 1, len(b_vector))
        x_new = np.linspace(0, 1, target_dim)
        b_vector = np.interp(x_new, x_old, b_vector)
        
    # 4. Normalization (Crucial for Neural Nets)
    b_vector = (b_vector - np.mean(b_vector)) / (np.std(b_vector) + 1e-6)
    
    return b_vector.astype(np.float32)