import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import positional_encoding

__all__ = ["ImplicitDecoder"]

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# --------------------------------------------------------------------------- #
#  Implicit Decoder                                                           #
# --------------------------------------------------------------------------- #

class ImplicitDecoder(nn.Module):
    def __init__(self,
                 voxel_feature_dim=768,
                 hidden_dim=768,
                 output_dim=768,
                 L=0,
                 pe_type='none'):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.L = L
        self.pe_type = pe_type

        if self.pe_type == 'sinusoidal':
            self.pe_dim = 2 * self.L * 3
        elif self.pe_type == 'concat':
            self.pe_dim = 3
        elif self.pe_type == 'none':
            self.pe_dim = 0
        else:
            raise ValueError(f"Unknown pe_type {self.pe_type}")

        self.input_dim = self.voxel_feature_dim + self.pe_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim + self.pe_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4 = nn.LayerNorm(self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

        self.apply(init_weights_kaiming)

    def forward(self, voxel_features, coords_3d=None):
        if self.pe_type == 'sinusoidal':
            pe = positional_encoding(coords_3d, L=self.L)
        elif self.pe_type == 'concat':
            pe = coords_3d
        else:
            pe = torch.zeros((voxel_features.shape[0], 0), device=voxel_features.device)

        x = torch.cat([voxel_features, pe], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)), inplace=True)
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)
        x = torch.cat([x, pe], dim=-1)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)
        return self.fc5(x) 