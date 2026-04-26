import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Attention mechanism combining Stream 1 and Stream 2.
    """
    def __init__(self, stream1_dim=1024, stream2_dim=256, embed_dim=512):
        super(AttentionFusion, self).__init__()
        self.proj1 = nn.Linear(stream1_dim, embed_dim)
        self.proj2 = nn.Linear(stream2_dim, embed_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 2), # 2 attention weights
            nn.Softmax(dim=1)
        )
        
    def forward(self, s1, s2):
        # Project both streams to same dimension
        h1 = self.proj1(s1)
        h2 = self.proj2(s2)
        
        # Calculate attention weights
        concat = torch.cat((h1, h2), dim=1)
        attn_weights = self.attention(concat)
        
        # Apply weights: w1 * h1 + w2 * h2
        weighted_h1 = h1 * attn_weights[:, 0].unsqueeze(1)
        weighted_h2 = h2 * attn_weights[:, 1].unsqueeze(1)
        
        fused = weighted_h1 + weighted_h2
        return fused
