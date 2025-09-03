"""
Training setup and initialization for Tetris PPO.

Defines hyperparameters and instantiates: device, vectorized env,
Actor-Critic model, optimizer, and TensorBoard SummaryWriter.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

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
    print(f"Envs:              {NUM_ENVS} (sync)")
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
    vec_env = make_vec_env(num_envs=NUM_ENVS, backend="sync")

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

    # ---------------
    # Rollout Buffer
    # ---------------

    class RolloutBuffer:
        """Stores rollout data for PPO across parallel envs.

        Shapes use (T, N, ...) where T=N_STEPS and N=NUM_ENVS.
        """

        def __init__(
            self,
            n_steps: int,
            num_envs: int,
            spatial_obs_shape: Tuple[int, int, int],  # (2, 20, 10)
            flat_obs_dim: int,  # 43
            device: torch.device,
        ):
            self.n_steps = int(n_steps)
            self.num_envs = int(num_envs)
            self.device = device

            # Storage tensors (pre-allocated on device)
            self.spatial_obs = torch.zeros(
                (n_steps, num_envs, *spatial_obs_shape), device=device, dtype=torch.float32
            )
            self.flat_obs = torch.zeros(
                (n_steps, num_envs, flat_obs_dim), device=device, dtype=torch.float32
            )
            self.actions = torch.zeros((n_steps, num_envs), device=device, dtype=torch.long)
            self.log_probs = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)
            self.rewards = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)
            self.dones = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)
            self.values = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)

            self.step = 0

        def add(
            self,
            spatial_obs: Tensor,          # (N, 2, 20, 10)
            flat_obs: Tensor,             # (N, 43)
            actions: Tensor,              # (N,)
            log_probs: Tensor,            # (N,)
            rewards: Tensor,              # (N,)
            dones: Tensor,                # (N,)
            values: Tensor,               # (N,)
        ) -> None:
            t = self.step
            if t >= self.n_steps:
                raise RuntimeError("RolloutBuffer is full. Call compute/clear before adding more.")
            self.spatial_obs[t].copy_(spatial_obs)
            self.flat_obs[t].copy_(flat_obs)
            self.actions[t].copy_(actions)
            self.log_probs[t].copy_(log_probs)
            self.rewards[t].copy_(rewards)
            self.dones[t].copy_(dones)
            self.values[t].copy_(values)
            self.step += 1

        @torch.no_grad()
        def compute_returns_and_advantages(
            self,
            last_values: Tensor,  # (N,)
            last_dones: Tensor,   # (N,) float or bool
            gamma: float,
            gae_lambda: float,
        ) -> Tuple[Tensor, Tensor]:
            """Compute GAE advantages and returns.

            Returns (advantages, returns) of shape (T, N) on the same device.
            """
            T, N = self.n_steps, self.num_envs
            advantages = torch.zeros((T, N), device=self.device, dtype=torch.float32)
            returns = torch.zeros((T, N), device=self.device, dtype=torch.float32)

            next_adv = torch.zeros((N,), device=self.device, dtype=torch.float32)
            next_value = last_values
            next_not_done = 1.0 - last_dones.float()

            for t in reversed(range(T)):
                not_done = 1.0 - self.dones[t]
                delta = self.rewards[t] + gamma * next_value * next_not_done - self.values[t]
                next_adv = delta + gamma * gae_lambda * next_adv * next_not_done
                advantages[t] = next_adv
                next_value = self.values[t]
                next_not_done = not_done
            returns = advantages + self.values
            return advantages, returns

        def clear(self) -> None:
            self.step = 0

    # ---------------------------
    # Helpers: obs batching util
    # ---------------------------

    def batch_obs_to_tensors(
        obs_batch: Sequence[Tuple[object, object]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Convert a list of (spatial, flat) into batched torch tensors on device.

        - spatial: (2, 20, 10) -> (N, 2, 20, 10)
        - flat: (43,) -> (N, 43)
        """
        # Convert via torch.as_tensor to preserve dtype if np arrays, fallback to float32
        spatial_list, flat_list = zip(*obs_batch)
        spatial = torch.as_tensor(spatial_list, device=device, dtype=torch.float32)
        flat = torch.as_tensor(flat_list, device=device, dtype=torch.float32)
        return spatial, flat

    # -------------------------
    # Data Collection (Rollout)
    # -------------------------

    # Initial reset and state tensors
    obs_list = vec_env.reset()
    spatial_tensor, flat_tensor = batch_obs_to_tensors(obs_list, device)

    # Shapes for buffer
    spatial_shape = tuple(spatial_tensor.shape[1:])  # (2, 20, 10)
    flat_dim = int(flat_tensor.shape[1])             # 43
    buffer = RolloutBuffer(N_STEPS, NUM_ENVS, spatial_shape, flat_dim, device)

    global_step_counter = 0

    # Per-env episodic trackers for logging
    ep_returns = torch.zeros(NUM_ENVS, dtype=torch.float64)
    ep_lengths = torch.zeros(NUM_ENVS, dtype=torch.int64)

    while global_step_counter < TOTAL_TIMESTEPS:
        # Rollout N_STEPS
        model.eval()
        buffer.clear()
        with torch.no_grad():
            for t in range(N_STEPS):
                # Forward pass
                logits, values = model((spatial_tensor, flat_tensor))

                # Mask invalid actions
                mask_list = vec_env.valid_action_masks()  # List[List[int]] (N, A)
                mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
                logits = logits.masked_fill(~mask, -torch.inf)

                # Sample actions
                dist = Categorical(logits=logits)
                actions = dist.sample()                # (N,)
                log_probs = dist.log_prob(actions)     # (N,)

                # Step environments
                next_obs_list, rewards_np, dones_np, infos_list = vec_env.step(actions.detach().cpu().tolist())

                # Convert rewards/dones
                rewards = torch.tensor(rewards_np, device=device, dtype=torch.float32)
                dones = torch.tensor(dones_np, device=device, dtype=torch.float32)

                # Store step in buffer
                buffer.add(spatial_tensor, flat_tensor, actions, log_probs, rewards, dones, values)

                # Logging trackers
                ep_returns += rewards.double().cpu()
                ep_lengths += 1

                # Reset envs that are done; log episode info
                for i, done in enumerate(dones_np):
                    if done:
                        # Log episodic stats
                        try:
                            writer.add_scalar("charts/episodic_return", ep_returns[i].item(), global_step_counter)
                            writer.add_scalar("charts/episodic_length", ep_lengths[i].item(), global_step_counter)
                        except Exception:
                            pass
                        # Reset trackers for that env
                        ep_returns[i] = 0.0
                        ep_lengths[i] = 0

                # Prepare next state; reset any done envs to continue rollout
                for i, d in enumerate(dones_np):
                    if d:
                        # Reset just that env by calling underlying env reset
                        # Subproc/sync API expects list of seeds or none; we can perform individual resets
                        # by creating a fresh observation via vec_env.reset() is all-envs; but we only want one.
                        # Workaround: call step with noop until wrapper returns a fresh episode on next step.
                        # However, our wrapper doesn't auto-reset; use batched reset across all envs when any done.
                        # Simpler approach: call reset() for all envs and keep others' obs unchanged.
                        pass

                # Efficiently reset only done envs by directly querying wrapper states
                # via a follow-up reset call with seeds per-env. We need updated obs list next.
                # We'll rebuild obs_list applying resets where done.
                new_obs_list: List[Tuple[object, object]] = list(next_obs_list)
                if any(dones_np):
                    reset_indices = [i for i, d in enumerate(dones_np) if d]
                    # Prefer per-env reset when using SyncVecTetris
                    if hasattr(vec_env, "envs"):
                        for idx in reset_indices:
                            new_obs_list[idx] = vec_env.envs[idx].reset()
                    else:
                        # Fallback: if partial reset unavailable, reset all envs
                        # and continue with those fresh observations.
                        new_obs_list = vec_env.reset()

                # Update current state tensors
                spatial_tensor, flat_tensor = batch_obs_to_tensors(new_obs_list, device)

                # Increase global step count
                global_step_counter += NUM_ENVS

            # End for N_STEPS

            # Bootstrap value for last obs
            last_logits, last_values = model((spatial_tensor, flat_tensor))
            last_dones = torch.tensor([0.0] * NUM_ENVS, device=device)  # By construction, states here are non-terminal
            advantages, returns = buffer.compute_returns_and_advantages(
                last_values=last_values, last_dones=last_dones, gamma=GAMMA, gae_lambda=GAE_LAMBDA
            )

        # At this point: buffer holds transitions; advantages/returns are computed.
        # Learning phase (PPO update) would go here.

        # Learning step would consume `buffer`, `advantages`, and `returns` here.
        # After optimizing, loop continues collecting the next rollout.

    # Cleanup
    vec_env.close()
    writer.close()
