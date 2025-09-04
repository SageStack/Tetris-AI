"""
Run a trained Tetris PPO policy for interactive inference.

Features
- Choose a weights file via --weights or a file dialog popup
- Greedy action selection from the trained model
- Pygame viewer with simple controls

Controls
- ESC/Q: quit
- R: reset episode
- O: open another weights file (file dialog)
"""

from __future__ import annotations

import os
import time
import argparse
from typing import Optional

import torch

try:
    import pygame  # type: ignore
except Exception as e:
    pygame = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

from tetris import TetrisEnvWrapper, BOARD_W, BOARD_H, build_candidate_tensors_from_board_plane
from actor_critic import build_default_model


def _choose_weights_path_dialog(initial_dir: Optional[str] = None, default_name: str = "model.pt") -> Optional[str]:
    try:
        from tkinter import Tk, filedialog  # type: ignore
        root = Tk(); root.withdraw()
        if initial_dir is None:
            initial_dir = os.getcwd()
        path = filedialog.askopenfilename(
            title="Select model weights",
            initialdir=initial_dir,
            defaultextension=".pt",
            filetypes=[("PyTorch", "*.pt *.pth"), ("All Files", "*.*")],
        )
        try:
            root.destroy()
        except Exception:
            pass
        return path if path else None
    except Exception:
        return None


@torch.no_grad()
def _ensure_lazy_initialized(model: torch.nn.Module,
                             env: TetrisEnvWrapper,
                             device: torch.device) -> None:
    # Run a single dummy forward to initialize LazyLinear in the model before loading state dict.
    obs = env.reset()
    spatial, flat = obs
    spatial_t = torch.as_tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0)
    flat_t = torch.as_tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)
    action_dim = env.action_space_n
    board_plane = spatial[0]  # channel 0: board
    pid = env.current_piece_id()
    rasters_i, feats_i, _ = build_candidate_tensors_from_board_plane(board_plane, pid, pad_to=action_dim)
    rasters_t = torch.from_numpy(rasters_i).to(device) if np is not None else torch.as_tensor(rasters_i, dtype=torch.float32, device=device)
    feats_t = torch.from_numpy(feats_i).to(device) if np is not None else torch.as_tensor(feats_i, dtype=torch.float32, device=device)
    rasters_t = rasters_t.unsqueeze(0)  # (1, A, 1, H, W)
    feats_t = feats_t.unsqueeze(0)      # (1, A, F)
    model.eval()
    _ = model((spatial_t, flat_t), cand_inputs=(rasters_t, feats_t))


def _load_model(weights_path: str, device: torch.device, env: TetrisEnvWrapper) -> torch.nn.Module:
    ckpt = torch.load(weights_path, map_location=device)
    action_space_n = int(ckpt.get("action_space_n", env.action_space_n))
    in_channels = int(ckpt.get("in_channels", 7))
    flat_dim = int(ckpt.get("flat_dim", 52))
    model = build_default_model(action_space_n, in_channels=in_channels, flat_dim=flat_dim).to(device)
    # Initialize lazy modules before loading
    _ensure_lazy_initialized(model, env, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"Warning: load_state_dict non-strict. Missing: {missing}, Unexpected: {unexpected}")
    model.eval()
    return model


def _select_weights(args_weights: Optional[str]) -> Optional[str]:
    if args_weights and os.path.isfile(args_weights):
        return args_weights
    # Try default locations
    candidates = []
    if args_weights:
        candidates.append(args_weights)
    for d in (".", "runs"):
        try:
            for name in ("best_model.pt", "last_model.pt", "model.pt"):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    candidates.append(p)
        except Exception:
            pass
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Popup dialog as fallback
    return _choose_weights_path_dialog()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained Tetris policy in a viewer")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pt/.pth weights; if omitted, show file dialog")
    parser.add_argument("--tile", type=int, default=24, help="Tile size for rendering")
    parser.add_argument("--fps", type=int, default=30, help="Viewer frame rate")
    parser.add_argument("--step-interval", type=float, default=0.2, help="Seconds between actions")
    parser.add_argument("--seed", type=int, default=None, help="Env seed")
    args = parser.parse_args()

    if pygame is None:
        raise RuntimeError("pygame not available; pip install pygame")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Env and viewer
    env = TetrisEnvWrapper(seed=args.seed, preprocess=True)
    pygame.init()
    W = 20 + BOARD_W * args.tile + 20 + 150
    H = 20 + BOARD_H * args.tile + 20
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tetris Inference")
    clock = pygame.time.Clock()

    # Load model
    weights_path = _select_weights(args.weights)
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError("Weights file not selected or not found")
    model = _load_model(weights_path, device, env)

    step_timer = 0.0
    obs = env.reset()
    running = True
    while running:
        dt = clock.tick(max(1, int(args.fps))) / 1000.0
        step_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                elif event.key == pygame.K_o:
                    # reload weights via file dialog
                    new_path = _choose_weights_path_dialog(initial_dir=os.path.dirname(weights_path) or ".")
                    if new_path and os.path.isfile(new_path):
                        weights_path = new_path
                        model = _load_model(weights_path, device, env)

        # Action step on interval
        if step_timer >= max(0.01, float(args.step_interval)):
            step_timer = 0.0
            spatial, flat = obs
            spatial_t = torch.as_tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0)
            flat_t = torch.as_tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)
            action_dim = env.action_space_n
            # Candidates
            board_plane = spatial[0]
            pid = env.current_piece_id()
            rasters_i, feats_i, _ = build_candidate_tensors_from_board_plane(board_plane, pid, pad_to=action_dim)
            rasters_t = torch.from_numpy(rasters_i).to(device) if np is not None else torch.as_tensor(rasters_i, dtype=torch.float32, device=device)
            feats_t = torch.from_numpy(feats_i).to(device) if np is not None else torch.as_tensor(feats_i, dtype=torch.float32, device=device)
            rasters_t = rasters_t.unsqueeze(0)
            feats_t = feats_t.unsqueeze(0)
            logits, _value = model((spatial_t, flat_t), cand_inputs=(rasters_t, feats_t))
            mask = torch.tensor(env.valid_action_mask(), device=device, dtype=torch.bool).unsqueeze(0)
            logits = logits.masked_fill(~mask, -torch.inf)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _r, done, _info = env.step(action)
            if done:
                obs = env.reset()

        # Render
        env.render(screen, tile=args.tile)
        # HUD: show weights file name
        try:
            font = pygame.font.SysFont("consolas", 18)
        except Exception:
            font = pygame.font.Font(None, 18)
        base = os.path.basename(weights_path)
        text = font.render(f"Weights: {base}", True, (230, 230, 230))
        screen.blit(text, (10, 6))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
