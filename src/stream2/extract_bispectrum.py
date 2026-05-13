from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import librosa
import numpy as np
from scipy.fft import dct
from sklearn.decomposition import PCA


def _to_mono_audio(audio_array: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    if audio.ndim != 1:
        raise ValueError("audio_array must be a 1D waveform or a 2D channel-first/last array")
    return audio


def _resample_audio(audio_array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    audio = _to_mono_audio(audio_array)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


def _resize_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == target_dim:
        return vector
    if vector.size == 0:
        return np.zeros(target_dim, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, vector.size)
    x_new = np.linspace(0.0, 1.0, target_dim)
    resized = np.interp(x_new, x_old, vector)
    return resized.astype(np.float32)


def _linear_filterbank(sample_rate: int, n_fft: int, n_filters: int) -> np.ndarray:
    n_frequency_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_frequency_bins), dtype=np.float32)

    frequency_points = np.linspace(0.0, sample_rate / 2.0, n_filters + 2)
    fft_bins = np.floor((n_fft + 1) * frequency_points / sample_rate).astype(int)
    fft_bins = np.clip(fft_bins, 0, n_frequency_bins - 1)

    for index in range(1, n_filters + 1):
        left = fft_bins[index - 1]
        center = max(fft_bins[index], left + 1)
        right = max(fft_bins[index + 1], center + 1)
        right = min(right, n_frequency_bins)

        for bin_index in range(left, center):
            filterbank[index - 1, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, right):
            filterbank[index - 1, bin_index] = (right - bin_index) / max(right - center, 1)

    return filterbank


def extract_lfcc_features(
    audio_array: np.ndarray,
    sr: int = 16000,
    target_sr: int = 16000,
    n_lfcc: int = 120,
    n_fft: int = 512,
    hop_length: int = 160,
    n_filters: Optional[int] = None,
) -> np.ndarray:
    """
    Extract a fixed-length LFCC summary vector.
    """
    audio = _resample_audio(audio_array, sr, target_sr)
    n_filters = n_filters or n_lfcc

    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2
    filterbank = _linear_filterbank(target_sr, n_fft, n_filters)
    linear_energies = np.dot(filterbank, spectrogram)
    log_energies = np.log(linear_energies + 1e-8)

    lfcc_vector = dct(np.mean(log_energies, axis=1), type=2, norm="ortho")
    lfcc_vector = _resize_vector(lfcc_vector, n_lfcc)
    lfcc_vector = (lfcc_vector - np.mean(lfcc_vector)) / (np.std(lfcc_vector) + 1e-6)
    return lfcc_vector.astype(np.float32)


def extract_bispectrum_matrix(
    audio_array: np.ndarray,
    sr: int = 16000,
    target_sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    bispectrum_bins: int = 128,
) -> np.ndarray:
    """
    Build a bispectrum magnitude map that captures higher-order phase coupling.
    """
    audio = _resample_audio(audio_array, sr, target_sr)
    spectrum = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrum = spectrum[:bispectrum_bins]

    frequency_bins = spectrum.shape[0]
    bispectrum = np.zeros((bispectrum_bins, bispectrum_bins), dtype=np.float32)

    for first_index in range(frequency_bins):
        max_second_index = min(frequency_bins, bispectrum_bins - first_index)
        if max_second_index <= 0:
            continue

        first_term = spectrum[first_index]
        for second_index in range(max_second_index):
            third_index = first_index + second_index
            triple_product = first_term * spectrum[second_index] * np.conjugate(spectrum[third_index])
            magnitude = np.log1p(np.abs(np.mean(triple_product)))
            phase_term = np.cos(np.angle(np.mean(triple_product)))
            bispectrum[first_index, second_index] = magnitude * phase_term

    bispectrum = (bispectrum - np.mean(bispectrum)) / (np.std(bispectrum) + 1e-6)
    return bispectrum.astype(np.float32)


@dataclass
class Stream2ForensicFeaturePipeline:
    """
    Full Stream 2 pipeline: LFCC summary + bispectrum PCA projection.
    """

    lfcc_dim: int = 120
    bispectrum_bins: int = 128
    bispectrum_components: int = 128
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    random_state: int = 42
    pca: Optional[PCA] = field(default=None, init=False)

    def fit(self, audio_arrays: Sequence[np.ndarray], sample_rates: Optional[Sequence[int]] = None):
        if sample_rates is not None and len(sample_rates) != len(audio_arrays):
            raise ValueError("audio_arrays and sample_rates must have the same length")

        flattened_features = []
        for index, audio_array in enumerate(audio_arrays):
            sr = sample_rates[index] if sample_rates is not None else self.sample_rate
            bispectrum_matrix = extract_bispectrum_matrix(
                audio_array,
                sr=sr,
                target_sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                bispectrum_bins=self.bispectrum_bins,
            )
            flattened_features.append(bispectrum_matrix.reshape(-1))

        if len(flattened_features) < 2:
            raise ValueError("At least two audio examples are required to fit the bispectrum PCA")

        feature_matrix = np.stack(flattened_features)
        n_components = min(self.bispectrum_components, feature_matrix.shape[0], feature_matrix.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.random_state, svd_solver="randomized")
        self.pca.fit(feature_matrix)
        return self

    def _project_bispectrum(self, bispectrum_matrix: np.ndarray) -> np.ndarray:
        flattened = bispectrum_matrix.reshape(1, -1)
        if self.pca is None:
            return _resize_vector(flattened.squeeze(0), self.bispectrum_components)

        projected = self.pca.transform(flattened).squeeze(0)
        projected = _resize_vector(projected, self.bispectrum_components)
        return projected.astype(np.float32)

    def transform(self, audio_array: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        current_sr = self.sample_rate if sr is None else sr
        lfcc_vector = extract_lfcc_features(
            audio_array,
            sr=current_sr,
            target_sr=self.sample_rate,
            n_lfcc=self.lfcc_dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_filters=self.lfcc_dim,
        )
        bispectrum_matrix = extract_bispectrum_matrix(
            audio_array,
            sr=current_sr,
            target_sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            bispectrum_bins=self.bispectrum_bins,
        )
        bispectrum_vector = self._project_bispectrum(bispectrum_matrix)
        combined = np.concatenate((lfcc_vector, bispectrum_vector), axis=0)
        return combined.astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self.pca is not None

    def save(self, path) -> None:
        """Serialise the fitted pipeline (PCA basis + config) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Stream2ForensicFeaturePipeline":
        """Load a previously saved pipeline from disk."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def transform_full(self, audio_array: np.ndarray, sr: Optional[int] = None) -> dict:
        current_sr = self.sample_rate if sr is None else sr
        lfcc_vector = extract_lfcc_features(
            audio_array,
            sr=current_sr,
            target_sr=self.sample_rate,
            n_lfcc=self.lfcc_dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_filters=self.lfcc_dim,
        )
        bispectrum_matrix = extract_bispectrum_matrix(
            audio_array,
            sr=current_sr,
            target_sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            bispectrum_bins=self.bispectrum_bins,
        )
        bispectrum_vector = self._project_bispectrum(bispectrum_matrix)
        return {
            "lfcc": lfcc_vector,
            "bispectrum_matrix": bispectrum_matrix,
            "bispectrum_vector": bispectrum_vector,
            "forensic": np.concatenate((lfcc_vector, bispectrum_vector), axis=0).astype(np.float32),
        }


def get_bispectrum_features(audio_array: np.ndarray, sr: int = 16000, n_fft: int = 512, target_dim: int = 128):
    """
    Backwards-compatible helper that returns a compact bispectrum summary.
    """
    bispectrum_matrix = extract_bispectrum_matrix(
        audio_array,
        sr=sr,
        target_sr=sr,
        n_fft=n_fft,
        hop_length=max(1, n_fft // 4),
        bispectrum_bins=target_dim,
    )
    return _resize_vector(bispectrum_matrix.reshape(-1), target_dim)


def get_forensic_features(audio_array: np.ndarray, sr: int = 16000, pipeline: Optional[Stream2ForensicFeaturePipeline] = None):
    """
    Return the full Stream 2 forensic feature vector.
    """
    pipeline = pipeline or Stream2ForensicFeaturePipeline()
    return pipeline.transform(audio_array, sr=sr)
