import torch
import torch.nn as nn
from src.stream1.rawnet2 import RawNet2Encoder
from src.fusion.attention_fusion import AttentionFusion

class StatisticalStream(nn.Module):
    def __init__(self, input_dim=248, hidden_dim=256, output_dim=64):
        super(StatisticalStream, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TwoStreamFusionNet(nn.Module):
    """
    The final combined PyTorch nn.Module.
    Combines RawNet2 and Bispectrum via Attention.
    """
    def __init__(self):
        super(TwoStreamFusionNet, self).__init__()
        self.rawnet = RawNet2Encoder(output_dim=256)
        self.stat_stream = StatisticalStream(input_dim=248, output_dim=64)
        
        self.fusion = AttentionFusion(stream1_dim=256, stream2_dim=64, embed_dim=256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2) # [Real, Fake]
        )

    def forward(self, audio: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        ear_emb = self.rawnet(audio)
        microscope_emb = self.stat_stream(stats)
        
        fused_emb = self.fusion(ear_emb, microscope_emb)
        out = self.classifier(fused_emb)
        return out
