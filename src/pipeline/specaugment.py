"""SpecAugment: time + frequency masking on an STFT view, reconstructed via ISTFT.

Operates on a single raw waveform tensor so it can sit in front of Stream 1
(RawNet2 consumes raw waveforms, not spectrograms) without changing the
[1, T] interface.
"""
from __future__ import annotations

import torch


def spec_augment(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    freq_mask_param: int = 15,
    time_mask_param: int = 25,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Args:
        waveform: [1, T] float tensor
    Returns:
        [1, T] float tensor — magnitude-masked, phase-preserved, reconstructed via ISTFT.
    """
    x = waveform.squeeze(0)  # [T]
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True,
    )  # [F, T_frames]
    mag = spec.abs()
    phase = spec.angle()
    n_freq, n_time = mag.shape

    for _ in range(n_freq_masks):
        f = int(torch.randint(0, freq_mask_param + 1, (1,), generator=generator))
        if f == 0:
            continue
        f0 = int(torch.randint(0, max(n_freq - f, 1), (1,), generator=generator))
        mag[f0:f0 + f, :] = 0.0

    for _ in range(n_time_masks):
        t = int(torch.randint(0, time_mask_param + 1, (1,), generator=generator))
        if t == 0:
            continue
        t0 = int(torch.randint(0, max(n_time - t, 1), (1,), generator=generator))
        mag[:, t0:t0 + t] = 0.0

    masked = torch.polar(mag, phase)
    out = torch.istft(
        masked, n_fft=n_fft, hop_length=hop_length, window=window,
        length=x.shape[-1],
    )
    return out.unsqueeze(0)  # [1, T]
