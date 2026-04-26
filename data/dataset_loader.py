import os
import torch
import soundfile as sf
from torch.utils.data import Dataset
from features.audio_utils import process_waveform

class ASVspoofDataset(Dataset):
    """
    PyTorch DataLoaders for ASVspoof 5 & WaveFake.
    """
    def __init__(self, base_dir, protocol_file):
        self.base_dir = base_dir
        self.file_list = []
        self.labels = []
        with open(protocol_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    self.file_list.append(parts[1])
                    self.labels.append(0 if parts[4] == 'bonafide' else 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.base_dir, file_name + ".flac")
        
        waveform_np, sample_rate = sf.read(file_path)
        waveform = torch.from_numpy(waveform_np).float()
        
        waveform = process_waveform(waveform, sample_rate)
        
        label = self.labels[idx]
        return waveform, torch.tensor(label, dtype=torch.long)
