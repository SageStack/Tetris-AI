from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Hybrid CNN + MLP Actor-Critic for Tetris.

    Inputs
    - spatial: Tensor of shape (N, 2, 20, 10) or (2, 20, 10)
               Two planes: [board_binary, current_piece_binary]
    - flat:    Tensor of shape (N, flat_dim) or (flat_dim,)
               Default flat_dim=43 for: next(5x7 one-hot)=35, hold(7), can_hold(1)

    Outputs
    - logits:  Tensor of shape (N, action_space_n) – unnormalized action scores
    - value:   Tensor of shape (N,) – state-value estimates
    """

    def __init__(self, action_space_n: int, flat_dim: int = 43):
        super().__init__()
        self.action_space_n = int(action_space_n)
        self.flat_dim = int(flat_dim)

        # CNN branch for spatial input (2, 20, 10)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
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
        self.policy = nn.Linear(256, self.action_space_n)  # logits
        self.value = nn.Linear(256, 1)                     # scalar value

        self._init_weights()

    def _init_weights(self):
        # Simple, stable init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
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

        # Heads
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value


def build_default_model(action_space_n: int) -> ActorCritic:
    """Convenience builder for the default preprocessing dimensions.

    action_space_n should match your wrapper's fixed action space size
    (e.g., MAX_ACTIONS_PER_PIECE).
    """
    return ActorCritic(action_space_n=action_space_n, flat_dim=43)

