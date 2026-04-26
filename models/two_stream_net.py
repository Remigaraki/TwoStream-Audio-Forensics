import torch
import torch.nn as nn
from models.rawnet2 import RawNet2Encoder
from models.attention_fusion import AttentionFusion

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
    """
    The final combined PyTorch nn.Module.
    Combines RawNet2 and Bispectrum via Attention.
    """
    def __init__(self):
        super(TwoStreamFusionNet, self).__init__()
        self.rawnet = RawNet2Encoder()
        self.stat_stream = StatisticalStream(input_dim=128, output_dim=256)
        
        self.fusion = AttentionFusion(stream1_dim=1024, stream2_dim=256, embed_dim=512)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2) # [Real, Fake]
        )

    def forward(self, audio, stats):
        ear_emb = self.rawnet(audio)
        microscope_emb = self.stat_stream(stats)
        
        fused_emb = self.fusion(ear_emb, microscope_emb)
        out = self.classifier(fused_emb)
        return out
