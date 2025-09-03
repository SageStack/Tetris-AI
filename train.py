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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

import argparse
import time
import uuid
import json
from tetris import make_vec_env, TetrisEnvWrapper, BOARD_W, BOARD_H
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

# Prefer CUDA (NVIDIA), then MPS (Apple Silicon), else CPU
if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# TensorBoard writer is created in __main__ so we can customize logdir per run


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
    # CLI args
    parser = argparse.ArgumentParser(description="Train PPO on Tetris")
    parser.add_argument("--render", action="store_true", help="Enable realtime rendering via observer env")
    parser.add_argument("--tile", type=int, default=30, help="Tile size for rendering (pixels)")
    parser.add_argument("--run-name", type=str, default=None, help="Base name for this run's logs under runs/")
    parser.add_argument(
        "--log-root",
        type=str,
        default="runs",
        help="Root directory for TensorBoard logs (default: runs)",
    )
    parser.add_argument("--notes", type=str, default=None, help="Optional notes string to attach to the run")
    args = parser.parse_args()

    # Build unique log directory for this run
    base_name = args.run_name or "tetris-ppo"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    run_dir = os.path.join(args.log_root, f"{base_name}-{timestamp}-{short_uid}")
    os.makedirs(run_dir, exist_ok=True)

    # Create a TensorBoard writer using the explicit run directory
    writer: SummaryWriter = SummaryWriter(log_dir=run_dir)

    # Vectorized Environment (multiprocessing). Keep under __main__ for 'spawn'.
    vec_env = make_vec_env(num_envs=NUM_ENVS, backend="sync")

    # Optional: observer environment for visualization in main process
    observer_env = None
    screen = None
    clock = None
    render_enabled = bool(args.render)
    tile_size = int(args.tile)
    if render_enabled:
        try:
            import pygame  # type: ignore
        except Exception as e:
            print(f"Render disabled: pygame import failed: {e}")
            render_enabled = False
        if render_enabled:
            observer_env = TetrisEnvWrapper()
            # Pygame init
            pygame.init()
            W = 20 + BOARD_W * tile_size + 20 + 150
            H = 20 + BOARD_H * tile_size + 20
            screen = pygame.display.set_mode((W, H))
            # Show run id in the window title
            try:
                run_id_title = os.path.basename(run_dir)
            except Exception:
                run_id_title = "run"
            pygame.display.set_caption(f"Tetris PPO - Observer - {run_id_title}")
            clock = pygame.time.Clock()
            # Pre-create a small font for lightweight HUD text
            try:
                hud_font = pygame.font.SysFont("consolas", 18)
            except Exception:
                hud_font = pygame.font.Font(None, 18)

    # Model & optimizer
    model = build_default_model(vec_env.action_space_n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    _print_setup_summary(vec_env.action_space_n)
    print(f"Run directory:    {run_dir}")

    # Example: write hyperparameters to TensorBoard
    # Log setup and persist config for reproducibility
    try:
        setup_lines = [
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
            f"run_dir: {run_dir}",
        ]
        if args.notes:
            setup_lines.append(f"notes: {args.notes}")
        writer.add_text("setup/summary", "\n".join(setup_lines))
        # Save config.json in the run directory
        config = {
            "device": str(device),
            "num_envs": NUM_ENVS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_steps": N_STEPS,
            "ppo_epochs": PPO_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_coef": CLIP_COEF,
            "vf_coef": VF_COEF,
            "ent_coef": ENT_COEF,
            "action_space_n": vec_env.action_space_n,
            "render": bool(args.render),
            "tile": tile_size,
            "run_name": base_name,
            "run_dir": run_dir,
            "notes": args.notes,
            "timestamp": timestamp,
            "uid": short_uid,
        }
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        # Logging is best-effort; continue if TB not available
        print(f"Warning: failed to log setup or save config: {e}")

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

    # Observer initial reset
    if render_enabled and observer_env is not None:
        observer_obs = observer_env.reset()

    # Shapes for buffer
    spatial_shape = tuple(spatial_tensor.shape[1:])  # (2, 20, 10)
    flat_dim = int(flat_tensor.shape[1])             # 43
    buffer = RolloutBuffer(N_STEPS, NUM_ENVS, spatial_shape, flat_dim, device)

    global_step_counter = 0
    # Indicates final rendering state after training completes
    training_is_complete = False

    # Per-env episodic trackers for logging (keep float32 for MPS compatibility)
    ep_returns = torch.zeros(NUM_ENVS, dtype=torch.float32)
    ep_lengths = torch.zeros(NUM_ENVS, dtype=torch.int64)

    while global_step_counter < TOTAL_TIMESTEPS:
        # Rollout N_STEPS
        model.eval()
        buffer.clear()
        with torch.no_grad():
            for t in range(N_STEPS):
                # Build combined batch including observer (if enabled)
                if render_enabled and observer_env is not None:
                    # Convert observer obs to tensors and append as batch=1
                    obs_spatial, obs_flat = observer_obs
                    obs_spatial_t = torch.as_tensor(obs_spatial, device=device, dtype=torch.float32).unsqueeze(0)
                    obs_flat_t = torch.as_tensor(obs_flat, device=device, dtype=torch.float32).unsqueeze(0)
                    spatial_in = torch.cat([spatial_tensor, obs_spatial_t], dim=0)
                    flat_in = torch.cat([flat_tensor, obs_flat_t], dim=0)
                else:
                    spatial_in, flat_in = spatial_tensor, flat_tensor

                # Forward pass on combined batch
                logits_all, values_all = model((spatial_in, flat_in))

                # Mask invalid actions (vec + optional observer)
                mask_vec = vec_env.valid_action_masks()  # List[List[int]] (N, A)
                if render_enabled and observer_env is not None:
                    mask_obs = observer_env.valid_action_mask()
                    mask_combined = mask_vec + [mask_obs]
                else:
                    mask_combined = mask_vec
                mask = torch.tensor(mask_combined, device=device, dtype=torch.bool)
                logits_all = logits_all.masked_fill(~mask, -torch.inf)

                # Sample actions over combined
                dist_all = Categorical(logits=logits_all)
                actions_all = dist_all.sample()            # (N [+ 1],)
                log_probs_all = dist_all.log_prob(actions_all)

                # Slice vec vs observer
                actions_vec = actions_all[:NUM_ENVS]
                log_probs_vec = log_probs_all[:NUM_ENVS]
                values_vec = values_all[:NUM_ENVS]

                # Step training environments
                next_obs_list, rewards_np, dones_np, infos_list = vec_env.step(actions_vec.detach().cpu().tolist())

                # Step observer environment and render (do not store/learn from it)
                if render_enabled and observer_env is not None:
                    import pygame  # type: ignore
                    action_obs = int(actions_all[-1].item())
                    observer_obs, observer_reward, observer_done, observer_info = observer_env.step(action_obs)
                    # Draw
                    if screen is not None:
                        observer_env.render(screen, tile=tile_size)
                        # Lightweight HUD: show current run id in the top-left
                        try:
                            run_id = os.path.basename(run_dir)
                        except Exception:
                            run_id = "run"
                        if 'hud_font' in locals() and hud_font is not None:
                            text_surface = hud_font.render(f"Run: {run_id}", True, (230, 230, 230))
                            screen.blit(text_surface, (10, 6))
                        # process events so the window stays responsive
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                render_enabled = False
                        pygame.display.flip()
                        if clock is not None:
                            clock.tick(30)
                    # Reset observer episode if done
                    if observer_done:
                        observer_obs = observer_env.reset()

                # Convert rewards/dones
                rewards = torch.tensor(rewards_np, device=device, dtype=torch.float32)
                dones = torch.tensor(dones_np, device=device, dtype=torch.float32)

                # Store step in buffer
                buffer.add(spatial_tensor, flat_tensor, actions_vec, log_probs_vec, rewards, dones, values_vec)

                # Logging trackers (avoid float64 on MPS; accumulate on CPU float32)
                ep_returns += rewards.detach().cpu().to(torch.float32)
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
        # -----------------------------
        # PPO Learning Phase
        # -----------------------------
        # Part 1: Flatten rollout into a single batch and normalize advantages
        T, N = buffer.n_steps, buffer.num_envs
        batch_total = T * N

        b_spatial_obs = buffer.spatial_obs.reshape(batch_total, *spatial_shape)
        b_flat_obs = buffer.flat_obs.reshape(batch_total, flat_dim)
        b_actions = buffer.actions.reshape(batch_total)
        b_log_probs = buffer.log_probs.reshape(batch_total)
        b_advantages = advantages.reshape(batch_total)
        b_returns = returns.reshape(batch_total)

        # Normalize advantages for stability
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std()
        b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

        # Part 2: PPO epochs and minibatch optimization
        model.train()
        inds = torch.arange(batch_total, device=device)

        # Track averages for logging
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        minibatch_updates = 0

        for epoch in range(PPO_EPOCHS):
            # Shuffle indices each epoch to break temporal correlation
            perm = inds[torch.randperm(batch_total, device=device)]
            for start in range(0, batch_total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, batch_total)
                mb_inds = perm[start:end]

                mb_spatial = b_spatial_obs[mb_inds]
                mb_flat = b_flat_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_old_logp = b_log_probs[mb_inds]
                mb_adv = b_advantages[mb_inds]
                mb_ret = b_returns[mb_inds]

                new_logits, new_values = model((mb_spatial, mb_flat))
                dist = Categorical(logits=new_logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Critic loss (value function)
                value_loss = F.mse_loss(new_values.squeeze(-1), mb_ret)

                # Actor loss (PPO clipped surrogate)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = mb_adv * ratio
                surr2 = mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Total loss
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                # Accumulate logs
                total_policy_loss += policy_loss.detach().item()
                total_value_loss += value_loss.detach().item()
                total_entropy += entropy.detach().item()
                total_loss += loss.detach().item()
                minibatch_updates += 1

        # Log averaged losses for this update
        if minibatch_updates > 0:
            try:
                writer.add_scalar("loss/policy", total_policy_loss / minibatch_updates, global_step_counter)
                writer.add_scalar("loss/value", total_value_loss / minibatch_updates, global_step_counter)
                writer.add_scalar("loss/entropy", total_entropy / minibatch_updates, global_step_counter)
                writer.add_scalar("loss/total", total_loss / minibatch_updates, global_step_counter)
            except Exception:
                pass

        # After optimizing, loop continues collecting the next rollout.

    # Mark training as complete
    training_is_complete = True

    # Close training resources before showing final screen
    vec_env.close()
    writer.close()

    # Post-training completion screen
    if render_enabled and observer_env is not None and screen is not None:
        try:
            import pygame  # type: ignore

            def draw_completion_screen(surface) -> None:
                # Render the final board state
                observer_env.render(surface, tile=tile_size)

                # Semi-transparent overlay
                overlay = pygame.Surface(surface.get_size(), flags=pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                surface.blit(overlay, (0, 0))

                # Centered text
                font = pygame.font.Font(None, 50)
                text_surface = font.render("Training Complete", True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))
                surface.blit(text_surface, text_rect)

                pygame.display.flip()

            running = True
            while running:
                draw_completion_screen(screen)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if clock is not None:
                    clock.tick(30)

            pygame.quit()
        except Exception as e:
            # Best-effort; ensure pygame quits if initialized
            try:
                import pygame  # type: ignore
                pygame.quit()
            except Exception:
                pass
            print(f"Post-training completion screen error: {e}")
