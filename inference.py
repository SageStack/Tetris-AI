"""
Inference runner for the Tetris PPO model with a simple GUI to select
model weights and a speed dial to control how fast the AI makes moves.

Notes:
- GUI uses PyQt5 (not Tkinter). If PyQt5 is unavailable, falls back to CLI args.
- Game rendering uses pygame.
- The AI selects placements using the candidate-scoring path from the
  trained ActorCritic model.

Usage examples:
  python inference.py                       # opens PyQt5 dialog
  python inference.py --weights runs/.../best_model.pt --speed 4.0

Controls during game:
- [ and ] : decrease / increase moves per second
- R       : reset game
- ESC/Q   : quit
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    import numpy as np
except Exception:
    np = None

try:
    import pygame  # type: ignore
except Exception:
    pygame = None

from tetris import (
    BOARD_H,
    BOARD_W,
    TetrisEnvWrapper,
    build_candidate_tensors_from_board_plane,
    PLACEMENTS_BY_PIECE,
)
from actor_critic import build_default_model, ActorCritic


# -----------------------------
# Device selection (best available)
# -----------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


@dataclass
class InferenceConfig:
    weights_path: str
    moves_per_sec: float = 4.0
    tile: int = 28
    greedy: bool = True


def _load_model(weights_path: str, action_space_n: int) -> Tuple[ActorCritic, int, int]:
    """Load an ActorCritic model from a checkpoint.

    Returns (model, in_channels, flat_dim).
    - Accepts both "export" payloads and full training checkpoints.
    - If dims are missing, defaults to (7, 52).
    """
    payload = torch.load(weights_path, map_location="cpu")
    # State dict extraction
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        in_channels = int(payload.get("in_channels", 7))
        flat_dim = int(payload.get("flat_dim", 52))
    elif isinstance(payload, dict):
        # Assume raw state dict was saved
        state_dict = payload
        in_channels = 7
        flat_dim = 52
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(payload)}")

    # Build and load model
    model = build_default_model(action_space_n=action_space_n,
                                in_channels=in_channels,
                                flat_dim=flat_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model, in_channels, flat_dim


def _choose_action(model: ActorCritic,
                   spatial_t: torch.Tensor,
                   flat_t: torch.Tensor,
                   piece_id: int,
                   valid_mask: torch.Tensor) -> int:
    """Compute candidate scores and pick an action (greedy)."""
    # Build candidate tensors from board plane (channel 0)
    board_plane = spatial_t[0, 0].detach().cpu().numpy() if np is not None else spatial_t[0, 0].detach().cpu().tolist()
    rasters, feats, _ = build_candidate_tensors_from_board_plane(board_plane, piece_id)
    rasters_t = torch.from_numpy(rasters) if np is not None else torch.as_tensor(rasters)
    feats_t = torch.from_numpy(feats) if np is not None else torch.as_tensor(feats)
    rasters_t = rasters_t.unsqueeze(0).to(DEVICE)  # (1, A, 1, H, W)
    feats_t = feats_t.unsqueeze(0).to(DEVICE)      # (1, A, F)

    with torch.no_grad():
        logits, _ = model((spatial_t, flat_t), cand_inputs=(rasters_t, feats_t))
        # Mask invalid
        logits = logits.masked_fill(~valid_mask, -torch.inf)
        action = int(torch.argmax(logits, dim=1).item())
    return action


def run_inference(cfg: InferenceConfig) -> None:
    if pygame is None:
        raise RuntimeError("pygame not available. pip install pygame")

    pygame.init()
    W = 20 + BOARD_W * cfg.tile + 20 + 160
    H = 20 + BOARD_H * cfg.tile + 20
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tetris Inference")
    clock = pygame.time.Clock()

    env = TetrisEnvWrapper(preprocess=True)
    # Mock a reset to infer dims
    obs_spatial, obs_flat = env.reset()
    action_space_n = int(env.action_space_n)

    # Model
    model, in_c, flat_d = _load_model(cfg.weights_path, action_space_n)
    # Prepare initial tensors
    spatial_t = torch.as_tensor(obs_spatial, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    flat_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # Timing control
    moves_per_sec = max(0.25, float(cfg.moves_per_sec))
    move_interval = 1.0 / moves_per_sec
    move_timer = 0.0

    # Human-like execution state machine
    stage: str = "idle"           # one of: idle, rotate, horiz, drop
    target_rot: Optional[int] = None
    target_x: Optional[int] = None

    def draw_hud(surface) -> None:
        try:
            font = pygame.font.SysFont("consolas", 18)
        except Exception:
            font = pygame.font.Font(None, 18)
        info_lines = [
            f"Speed: {moves_per_sec:.2f} moves/sec ([ and ] to adjust)",
            f"Weights: {os.path.basename(cfg.weights_path)}",
            "R: reset, ESC/Q: quit",
        ]
        y = 10
        for line in info_lines:
            txt = font.render(line, True, (235, 235, 235))
            surface.blit(txt, (W - 155, y))
            y += 20

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        move_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs_spatial, obs_flat = env.reset()
                    spatial_t = torch.as_tensor(obs_spatial, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    flat_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    move_timer = 0.0
                elif event.key == pygame.K_LEFTBRACKET:  # '['
                    new_speed = max(0.25, moves_per_sec - 0.5)
                    moves_per_sec = new_speed
                    move_interval = 1.0 / moves_per_sec
                elif event.key == pygame.K_RIGHTBRACKET:  # ']'
                    moves_per_sec = min(30.0, moves_per_sec + 0.5)
                    move_interval = 1.0 / moves_per_sec

        # AI move step (low-level inputs at the configured rate)
        if move_timer >= move_interval:
            move_timer = 0.0
            # If we don't have a target, pick one using the model
            if stage == "idle":
                mask_list = env.valid_action_mask()
                valid_mask = torch.tensor([mask_list], device=DEVICE, dtype=torch.bool)
                pid = env.current_piece_id()
                action_idx = _choose_action(model, spatial_t, flat_t, pid, valid_mask)
                plist = PLACEMENTS_BY_PIECE[int(pid)]
                placement = plist[action_idx % len(plist)]
                target_rot = int(placement.rot)
                target_x = int(placement.x)
                stage = "rotate"

            # Execute one low-level input according to the current stage
            done = False
            info = {}
            if stage == "rotate" and target_rot is not None:
                cur_rot = int(env.env.cur_rot)
                diff = (target_rot - cur_rot) % 4
                if diff == 0:
                    stage = "horiz"
                else:
                    # prefer 1x ccw instead of 3x cw
                    low_action = 3 if diff == 3 else 2
                    _, _, done, info = env.env.step(low_action)
            elif stage == "horiz" and target_x is not None:
                cur_x = int(env.env.cur_x)
                if cur_x == target_x:
                    stage = "drop"
                else:
                    low_action = 0 if cur_x > target_x else 1
                    _, _, done, info = env.env.step(low_action)
            elif stage == "drop":
                # soft drop one cell; repeat until lock
                _, _, done, info = env.env.step(4)
                if info.get("locked", False):
                    # piece placed; choose a new target next time
                    stage = "idle"
                    target_rot, target_x = None, None

            # Refresh model inputs after each low-level step
            obs_spatial, obs_flat = env.observe()
            spatial_t = torch.as_tensor(obs_spatial, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            flat_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            if done:
                time.sleep(0.15)
                obs_spatial, obs_flat = env.reset()
                spatial_t = torch.as_tensor(obs_spatial, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                flat_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                stage = "idle"
                target_rot, target_x = None, None

        # Render
        env.render(screen, tile=cfg.tile)
        draw_hud(screen)
        pygame.display.flip()

    pygame.quit()


# -----------------------------
# PyQt5 lightweight launcher
# -----------------------------

def _launch_pyqt_dialog() -> Optional[InferenceConfig]:
    try:
        from PyQt5 import QtWidgets, QtCore
    except Exception as e:
        print(f"PyQt5 unavailable ({e}); falling back to CLI args.")
        return None

    class Launcher(QtWidgets.QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Tetris AI - Inference")
            self.setMinimumWidth(460)

            layout = QtWidgets.QVBoxLayout(self)

            # Weights picker
            weights_group = QtWidgets.QGroupBox("Model Weights (.pt / .pth)")
            wg_layout = QtWidgets.QHBoxLayout()
            self.weights_edit = QtWidgets.QLineEdit(self)
            self.browse_btn = QtWidgets.QPushButton("Browse…", self)
            self.browse_btn.clicked.connect(self._on_browse)
            wg_layout.addWidget(self.weights_edit)
            wg_layout.addWidget(self.browse_btn)
            weights_group.setLayout(wg_layout)

            # Speed dial
            speed_group = QtWidgets.QGroupBox("Speed (moves per second)")
            sg_layout = QtWidgets.QHBoxLayout()
            self.speed_dial = QtWidgets.QDial(self)
            self.speed_dial.setRange(1, 30)  # 1..30 mps
            self.speed_dial.setNotchesVisible(True)
            self.speed_dial.setValue(4)
            self.speed_label = QtWidgets.QLabel("4.0 mps", self)
            self.speed_dial.valueChanged.connect(self._on_speed_change)
            sg_layout.addWidget(self.speed_dial)
            sg_layout.addWidget(self.speed_label)
            speed_group.setLayout(sg_layout)

            # Tile size
            tile_group = QtWidgets.QGroupBox("Tile Size (pixels)")
            tg_layout = QtWidgets.QHBoxLayout()
            self.tile_spin = QtWidgets.QSpinBox(self)
            self.tile_spin.setRange(16, 40)
            self.tile_spin.setValue(28)
            tg_layout.addWidget(self.tile_spin)
            tile_group.setLayout(tg_layout)

            # Buttons
            btn_layout = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start", self)
            self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
            self.start_btn.clicked.connect(self.accept)
            self.cancel_btn.clicked.connect(self.reject)
            btn_layout.addStretch(1)
            btn_layout.addWidget(self.cancel_btn)
            btn_layout.addWidget(self.start_btn)

            layout.addWidget(weights_group)
            layout.addWidget(speed_group)
            layout.addWidget(tile_group)
            layout.addLayout(btn_layout)

        def _on_browse(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Model Weights",
                os.getcwd(),
                "PyTorch Weights (*.pt *.pth);;All Files (*)",
            )
            if path:
                self.weights_edit.setText(path)

        def _on_speed_change(self, val: int):
            self.speed_label.setText(f"{float(val):.1f} mps")

        def result_config(self) -> Optional[InferenceConfig]:
            path = self.weights_edit.text().strip()
            if not path:
                return None
            return InferenceConfig(
                weights_path=path,
                moves_per_sec=float(self.speed_dial.value()),
                tile=int(self.tile_spin.value()),
            )

    app = QtWidgets.QApplication(sys.argv)
    dlg = Launcher()
    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        cfg = dlg.result_config()
        return cfg
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Tetris AI inference")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt/.pth)")
    parser.add_argument("--speed", type=float, default=4.0, help="Moves per second (1..30)")
    parser.add_argument("--tile", type=int, default=28, help="Tile size in pixels (16..40)")
    args = parser.parse_args()

    if args.weights is None:
        cfg = _launch_pyqt_dialog()
        if cfg is None:
            parser.error("--weights is required when GUI is unavailable")
    else:
        cfg = InferenceConfig(weights_path=args.weights, moves_per_sec=float(args.speed), tile=int(args.tile))

    run_inference(cfg)


if __name__ == "__main__":
    main()
