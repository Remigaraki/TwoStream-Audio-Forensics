import torch
import torchaudio.transforms as T

def process_waveform(waveform: torch.Tensor, sr: int, target_sr: int = 16000, fixed_length: int = 64000) -> torch.Tensor:
    """
    16kHz resampling, 4-second padding/trimming (64,000 samples).
    """
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Ensure shape is (Channels, Time) -> (1, fixed_length)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True) # convert stereo to mono
        
    current_len = waveform.shape[1]
    if current_len > fixed_length:
        start = torch.randint(0, current_len - fixed_length, (1,)).item()
        waveform = waveform[:, start : start + fixed_length]
    elif current_len < fixed_length:
        padding = fixed_length - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    return waveform
