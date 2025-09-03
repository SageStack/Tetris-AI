"""
Training setup and initialization for Tetris PPO.

Defines hyperparameters and instantiates: device, vectorized env,
Actor-Critic model, optimizer, and TensorBoard SummaryWriter.
"""

from __future__ import annotations

import os
import torch
from torch.utils.tensorboard import SummaryWriter

from tetris import make_vec_env
from actor_critic import build_default_model


# -----------------------------
# Hyperparameters
# -----------------------------

# Environment
NUM_ENVS: int = 16
TOTAL_TIMESTEPS: int = 10_000_000

# PPO core
N_STEPS: int = 2048            # rollout length per env before update
PPO_EPOCHS: int = 10           # optimization epochs per update
BATCH_SIZE: int = 64           # minibatch size
LEARNING_RATE: float = 3e-4
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_COEF: float = 0.2

# Loss weighting
VF_COEF: float = 0.5           # value function loss weight
ENT_COEF: float = 0.01         # entropy bonus weight


# -----------------------------
# Device & Logging
# -----------------------------

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a TensorBoard writer (creates ./runs/ by default)
writer: SummaryWriter = SummaryWriter()


def _print_setup_summary(action_space_n: int) -> None:
    num_updates = TOTAL_TIMESTEPS // max(1, (NUM_ENVS * N_STEPS))
    print("=== Training Setup ===")
    print(f"Device:            {device}")
    print(f"Envs:              {NUM_ENVS} (subproc)")
    print(f"Action space (n):  {action_space_n}")
    print(f"Total timesteps:   {TOTAL_TIMESTEPS}")
    print(f"Rollout length:    {N_STEPS}")
    print(f"PPO epochs:        {PPO_EPOCHS}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Gamma:             {GAMMA}")
    print(f"GAE lambda:        {GAE_LAMBDA}")
    print(f"Clip coef:         {CLIP_COEF}")
    print(f"VF coef:           {VF_COEF}")
    print(f"Ent coef:          {ENT_COEF}")
    print(f"Num updates:       {num_updates}")


if __name__ == "__main__":
    # Vectorized Environment (multiprocessing). Keep under __main__ for 'spawn'.
    vec_env = make_vec_env(num_envs=NUM_ENVS, backend="subproc")

    # Model & optimizer
    model = build_default_model(vec_env.action_space_n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    _print_setup_summary(vec_env.action_space_n)

    # Example: write hyperparameters to TensorBoard
    try:
        writer.add_text(
            "setup/summary",
            "\n".join(
                [
                    f"device: {device}",
                    f"num_envs: {NUM_ENVS}",
                    f"total_timesteps: {TOTAL_TIMESTEPS}",
                    f"n_steps: {N_STEPS}",
                    f"ppo_epochs: {PPO_EPOCHS}",
                    f"batch_size: {BATCH_SIZE}",
                    f"learning_rate: {LEARNING_RATE}",
                    f"gamma: {GAMMA}",
                    f"gae_lambda: {GAE_LAMBDA}",
                    f"clip_coef: {CLIP_COEF}",
                    f"vf_coef: {VF_COEF}",
                    f"ent_coef: {ENT_COEF}",
                    f"action_space_n: {vec_env.action_space_n}",
                ]
            ),
        )
    except Exception as e:
        # Logging is best-effort; continue if TB not available
        print(f"Warning: failed to log setup to TensorBoard: {e}")

    # At this point, the training loop can proceed using vec_env, model, optimizer, and writer.
    # Ensure to close vec_env and writer when done.
    # For now, just clean up immediately (placeholder behavior).
    vec_env.close()
    writer.close()

