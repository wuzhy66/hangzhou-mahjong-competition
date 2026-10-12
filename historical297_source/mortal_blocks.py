"""Pure PyTorch Mortal-v4 blocks for inference without an RLlib installation."""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // 16), nn.Mish(),
            nn.Linear(channels // 16, channels),
        )
        for layer in self.shared_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        weight = (self.shared_mlp(inputs.mean(-1)) +
                  self.shared_mlp(inputs.amax(-1))).sigmoid()
        return inputs * weight.unsqueeze(-1)


class MortalResidualBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(groups, channels, eps=1e-3), nn.Mish(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels, eps=1e-3), nn.Mish(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
        )
        self.attention = ChannelAttention(channels)

    def forward(self, inputs):
        return inputs + self.attention(self.body(inputs))
