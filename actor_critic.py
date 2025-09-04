from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Hybrid CNN + MLP Actor-Critic for Tetris.

    Inputs
    - spatial: Tensor of shape (N, C, 20, 10) or (C, 20, 10)
               Default C=7 channels: [board, current, holes, surface, wells, overhang, parity]
    - flat:    Tensor of shape (N, flat_dim) or (flat_dim,)
               Default flat_dim=52: next(5x7)=35, hold(7), can_hold(1), +9 scalars.

    Outputs
    - logits:  Tensor of shape (N, action_space_n) – unnormalized action scores
    - value:   Tensor of shape (N,) – state-value estimates
    """

    def __init__(self, action_space_n: int, in_channels: int = 7, flat_dim: int = 52):
        super().__init__()
        self.action_space_n = int(action_space_n)
        self.in_channels = int(in_channels)
        self.flat_dim = int(flat_dim)

        # CNN branch for spatial input (2, 20, 10)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # After convs: (N, 64, 20, 10) => flattened size
        self._cnn_flat_dim = 64 * 20 * 10

        # MLP branch for flat features (43)
        self.mlp_flat = nn.Sequential(
            nn.Linear(self.flat_dim, 64),
            nn.ReLU(inplace=True),
        )

        # Shared body
        combined_dim = self._cnn_flat_dim + 64
        self.body = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.policy = nn.Linear(256, self.action_space_n)  # logits (fallback/no-candidate)
        self.value = nn.Linear(256, 1)                     # scalar value

        self._init_weights()

        # Candidate scorer components
        # Shared tiny CNN for candidate rasters (1, 20, 10) -> 32-d embedding
        self.cand_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # (N*A, 32, 1, 1)
        )
        # Project state and candidate features to 32 dims
        self.state_proj = nn.Linear(256, 32)
        self.cand_feat_proj = nn.LazyLinear(32)  # infers input_dim=F on first use
        # Fuse [cand_embed(32), feat(32), state(32)] -> 64 -> 1
        self.cand_fuse = nn.Linear(96, 64)
        self.cand_out = nn.Linear(64, 1)

    def _init_weights(self):
        # Simple, stable init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        cand_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (rasters, feats)
        valid_mask: Optional[torch.Tensor] = None,  # (N, A) bool
    ):
        spatial, flat = inputs

        # Ensure batch dimension and float dtype
        if spatial.dim() == 3:
            spatial = spatial.unsqueeze(0)
        if flat.dim() == 1:
            flat = flat.unsqueeze(0)
        spatial = spatial.float()
        flat = flat.float()

        # CNN branch
        x_spatial = self.cnn(spatial)
        x_spatial = x_spatial.view(x_spatial.size(0), -1)

        # MLP branch
        x_flat = self.mlp_flat(flat)

        # Combine and shared trunk
        x = torch.cat([x_spatial, x_flat], dim=1)
        x = self.body(x)

        # Value head (always computed from shared trunk)
        value = self.value(x).squeeze(-1)

        if cand_inputs is None:
            # Fallback: simple policy head over all actions
            logits = self.policy(x)
            return logits, value

        # Candidate scoring path
        rasters, feats = cand_inputs  # rasters: (N, A, 1, 20, 10), feats: (N, A, F)
        N, A = int(rasters.shape[0]), int(rasters.shape[1])

        # Encode rasters via shared CNN (merge batch and candidate dims)
        r_in = rasters.view(N * A, 1, rasters.size(-2), rasters.size(-1))
        r_emb = self.cand_cnn(r_in).view(N * A, -1)  # (N*A, 32)

        # Project candidate features and shared state
        f_in = feats.view(N * A, -1)
        f_emb = self.cand_feat_proj(f_in)            # (N*A, 32)
        s_emb = self.state_proj(x)                   # (N, 32)
        s_emb = s_emb.unsqueeze(1).expand(N, A, 32).contiguous().view(N * A, 32)

        # Fuse
        z = torch.cat([r_emb, f_emb, s_emb], dim=1)  # (N*A, 96)
        z = F.relu(self.cand_fuse(z))
        cand_scores = self.cand_out(z).view(N, A)    # (N, A)

        # If a valid_mask is given, keep scores and mask later in training/sampling
        return cand_scores, value


def build_default_model(action_space_n: int, in_channels: int = 7, flat_dim: int = 52) -> ActorCritic:
    """Convenience builder for the default preprocessing dimensions.

    action_space_n should match your wrapper's fixed action space size
    (e.g., MAX_ACTIONS_PER_PIECE).
    """
    return ActorCritic(action_space_n=action_space_n, in_channels=in_channels, flat_dim=flat_dim)
