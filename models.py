import torch
import torch.nn as nn

# --- MOCK COMPONENTS (Placeholders for the Real Deal) ---

class MockRawNet2(nn.Module):
    """
    Simulates RawNet2 behavior without needing the 100MB weights file.
    Output: 1024-dimensional embedding.
    """
    def __init__(self):
        super(MockRawNet2, self).__init__()
        # A simple conv layer to simulate processing
        self.conv = nn.Conv1d(1, 128, kernel_size=3)
        self.fc = nn.Linear(128 * 63998, 1024) # Flatten simulation

    def forward(self, x):
        # x shape: (Batch, 64000) -> needs channel dim for Conv1d
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# --- REAL FUSION ARCHITECTURE ---

class StatisticalStream(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256):
        super(StatisticalStream, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)

class TwoStreamFusionNet(nn.Module):
    def __init__(self, rawnet_model):
        super(TwoStreamFusionNet, self).__init__()
        self.rawnet_branch = rawnet_model
        self.stat_branch = StatisticalStream(input_dim=128, output_dim=256)
        
        # 1024 (RawNet) + 256 (Stats) = 1280
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2) # [Real, Fake]
        )

    def forward(self, audio, stats):
        ear_emb = self.rawnet_branch(audio)
        microscope_emb = self.stat_branch(stats)
        combined = torch.cat((ear_emb, microscope_emb), dim=1)
        return self.classifier(combined)