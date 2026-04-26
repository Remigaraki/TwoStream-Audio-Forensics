import numpy as np
import librosa

def get_bispectrum_features(audio_array, sr=16000, n_fft=512, target_dim=128):
    """
    Computes a simplified 1D Bispectral vector from audio.
    Statistical Higher-Order Phase Coupling math (Stream 2).
    """
    S = librosa.stft(audio_array, n_fft=n_fft)
    S_complex = np.abs(S) * np.exp(1j * np.angle(S))
    
    freq_bins = S.shape[0]
    limit = freq_bins // 2
    b_mag = []
    
    for f in range(limit):
        term1 = S_complex[f, :]
        term2 = S_complex[f, :]
        term3 = np.conjugate(S_complex[2*f, :])
        B_slice = term1 * term2 * term3
        b_mag.append(np.mean(np.abs(B_slice)))

    b_vector = np.array(b_mag)
    
    if len(b_vector) != target_dim:
        x_old = np.linspace(0, 1, len(b_vector))
        x_new = np.linspace(0, 1, target_dim)
        b_vector = np.interp(x_new, x_old, b_vector)
        
    b_vector = (b_vector - np.mean(b_vector)) / (np.std(b_vector) + 1e-6)
    return b_vector.astype(np.float32)
