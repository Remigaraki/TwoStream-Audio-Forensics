import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """
    Cross-modal attention mechanism combining Stream 1 and Stream 2.
    """

    def __init__(self, stream1_dim: int = 256, stream2_dim: int = 64, embed_dim: int = 256):
        super(AttentionFusion, self).__init__()
        self.raw_projection = nn.Linear(stream1_dim, embed_dim)
        self.stat_projection = nn.Linear(stream2_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        raw_token = self.raw_projection(s1)
        stat_token = self.stat_projection(s2)

        tokens = torch.stack((raw_token, stat_token), dim=1)
        query = stat_token.unsqueeze(1)
        attended, _ = self.attention(query=query, key=tokens, value=tokens)

        fused = torch.cat((self.norm(attended.squeeze(1)), stat_token), dim=1)
        return self.output(fused)
