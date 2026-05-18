"""BispectralPCA: fit / transform / save / load a PCA compressor for bispectra."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA


class BispectralPCA:
    """
    Compress (128, 128) bispectrum matrices down to a 128-dim vector via PCA.

    Usage
    -----
    pca = BispectralPCA()
    pca.fit(list_of_128x128_arrays)
    vec = pca.transform(one_128x128_array)  # shape (128,)
    pca.fit_and_save(train_bispectra, "path/to/pca.joblib")

    pca2 = BispectralPCA.load("path/to/pca.joblib")
    """

    N_COMPONENTS: int = 128

    def __init__(self, n_components: int = N_COMPONENTS, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._pca: PCA | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_bispectra: List[np.ndarray]) -> "BispectralPCA":
        """
        Fit PCA on a list of (128, 128) bispectrum arrays.

        Asserts that cumulative explained variance >= 0.95; warns otherwise.
        """
        flat = np.stack([b.reshape(-1) for b in train_bispectra], axis=0)  # [N, 16384]
        n_comp = min(self.n_components, flat.shape[0] - 1, flat.shape[1])

        self._pca = PCA(n_components=n_comp, random_state=self.random_state)
        self._pca.fit(flat)

        explained = float(self._pca.explained_variance_ratio_.sum())
        if explained < 0.95:
            warnings.warn(
                f"BispectralPCA: cumulative explained variance is {explained:.3f} < 0.95. "
                "Consider providing more training samples.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, bispectrum: np.ndarray) -> np.ndarray:
        """
        Project a single (128, 128) bispectrum to a (128,) vector.
        If PCA has not been fitted, returns the first n_components elements
        of the flattened bispectrum (fallback, should not happen in production).
        """
        flat = bispectrum.reshape(1, -1).astype(np.float32)

        if self._pca is None:
            raise RuntimeError("BispectralPCA has not been fitted yet. Call fit() first.")

        projected = self._pca.transform(flat).squeeze(0)

        # Pad to exactly n_components if PCA returned fewer components
        if projected.shape[0] < self.n_components:
            projected = np.pad(projected, (0, self.n_components - projected.shape[0]))

        return projected.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def fit_and_save(
        self, train_bispectra: List[np.ndarray], path: Union[str, Path]
    ) -> "BispectralPCA":
        """Fit and immediately serialise to *path*."""
        self.fit(train_bispectra)
        self.save(path)
        return self

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BispectralPCA":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BispectralPCA, got {type(obj).__name__}")
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._pca is not None
