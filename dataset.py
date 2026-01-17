import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    def __init__(self, base_dir, protocol_file, transform=None, fixed_length=64000):
        """
        Args:
            base_dir (str): Path to the folder containing .flac files
            protocol_file (str): Path to the .txt file with labels (Real/Spoof)
            fixed_length (int): Number of samples to force (default 4 seconds at 16k)
        """
        self.base_dir = base_dir
        self.fixed_length = fixed_length
        self.file_list = []
        self.labels = [] # 0 for Real (Bonafide), 1 for Fake (Spoof)

        # 1. Parse the Protocol File
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                file_name = parts[1]
                label_str = parts[4] # 'bonafide' or 'spoof'
                
                self.file_list.append(file_name)
                # Map 'bonafide' -> 0, 'spoof' -> 1
                self.labels.append(0 if label_str == 'bonafide' else 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Locate file
        file_name = self.file_list[idx]
        file_path = os.path.join(self.base_dir, file_name + ".flac")
        
        # --- NEW LOADING BLOCK START (Fixes Colab Crash) ---
        # 1. Load using soundfile (It returns a numpy array)
        waveform_np, sample_rate = sf.read(file_path)
        
        # 2. Convert to PyTorch Tensor
        waveform = torch.from_numpy(waveform_np).float()
        
        # 3. Ensure shape is (Channels, Time) -> (1, 64000)
        # soundfile usually returns (Time, Channels) or just (Time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # Add channel dimension
        else:
            waveform = waveform.t() # Transpose if stereo
        # --- NEW LOADING BLOCK END ---

        # 3. Fix Length (Cut or Pad)
        # We need exactly 64000 samples for the model to work
        current_len = waveform.shape[1]
        
        if current_len > self.fixed_length:
            # Too long? Crop a random section
            start = torch.randint(0, current_len - self.fixed_length, (1,)).item()
            waveform = waveform[:, start : start + self.fixed_length]
        
        elif current_len < self.fixed_length:
            # Too short? Pad with zeros
            padding = self.fixed_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # 4. Return Tensor and Label
        label = self.labels[idx]
        return waveform, torch.tensor(label, dtype=torch.long)