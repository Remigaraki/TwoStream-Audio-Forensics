import torch
import torch.nn as nn

class RawNet2Encoder(nn.Module):
    """
    Neural Baseline (Stream 1). Extracts raw waveform embeddings.
    """
    def __init__(self):
        super(RawNet2Encoder, self).__init__()
        self.conv = nn.Conv1d(1, 128, kernel_size=3)
        self.fc = nn.Linear(128, 1024) 

    def forward(self, x):
        # x expected to be [batch, 1, 64000]
        x = self.conv(x)
        x = torch.mean(x, dim=2) # Global average pooling
        x = self.fc(x)
        return x # Output: [batch, 1024]
