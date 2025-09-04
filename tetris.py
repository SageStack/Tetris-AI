import sys
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

# Optional imports (numpy for observations; pygame for rendering)
try:
    import numpy as np
except Exception:
    np = None  # Observation can still be returned as lists

try:
    import pygame
except Exception:
    pygame = None

# -----------------------------
# Tetris Environment (Gym-like)
# -----------------------------

BOARD_W = 10
BOARD_H = 20
VISIBLE_ROWS = 20  # logical board height is BOARD_H

# Actions
ACTIONS = {
    0: "left",
    1: "right",
    2: "rotate_cw",
    3: "rotate_ccw",
    4: "soft_drop",
    5: "hard_drop",
    6: "hold",
    7: "noop",
}

# 4x4 spawn matrices (Super Rotation System style orientation)
# 1-based ids map to colors/piece types: 1=I,2=O,3=T,4=S,5=Z,6=J,7=L
PIECE_MATS: Dict[int, List[List[List[int]]]] = {
    1: [  # I
        [
            [0,0,0,0],
            [1,1,1,1],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
    2: [  # O
        [
            [0,1,1,0],
            [0,1,1,0],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
    3: [  # T
        [
            [0,1,0,0],
            [1,1,1,0],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
    4: [  # S
        [
            [0,1,1,0],
            [1,1,0,0],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
    5: [  # Z
        [
            [1,1,0,0],
            [0,1,1,0],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
    6: [  # J
        [
            [1,0,0,0],
            [1,1,1,0],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
    7: [  # L
        [
            [0,0,1,0],
            [1,1,1,0],
            [0,0,0,0],
            [0,0,0,0],
        ]
    ],
}

# Precompute all 4 rotations for each piece by rotating the 4x4 matrix

def rot90(mat: List[List[int]]) -> List[List[int]]:
    # Rotate 4x4 clockwise
    return [list(reversed(col)) for col in zip(*mat)]

PIECES: Dict[int, List[List[List[int]]]] = {}
for pid, mats in PIECE_MATS.items():
    base = mats[0]
    r1 = rot90(base)
    r2 = rot90(r1)
    r3 = rot90(r2)
    PIECES[pid] = [base, r1, r2, r3]

# Simple 7-bag generator
class SevenBag:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.bag: List[int] = []

    def next(self) -> int:
        if not self.bag:
            self.bag = [1,2,3,4,5,6,7]
            self.rng.shuffle(self.bag)
        return self.bag.pop()

# Recorder for JSONL transitions
class TransitionRecorder:
    def __init__(self):
        self.fp = None
        self.enabled = False
        self.t = 0

    def start(self, path: str):
        self.fp = open(path, 'w', encoding='utf-8')
        self.enabled = True
        self.t = 0

    def stop(self):
        if self.fp:
            self.fp.close()
        self.fp = None
        self.enabled = False

    def record(self, s: Dict[str, Any], a: int, r: float, s2: Dict[str, Any], done: bool, info: Dict[str, Any]):
        if not self.enabled or self.fp is None:
            return
        rec = {
            "t": self.t,
            "action": a,
            "reward": r,
            "done": done,
            "state": s,
            "next_state": s2,
            "info": info,
        }
        self.fp.write(json.dumps(rec) + "\n")
        self.t += 1

# Core environment
class TetrisEnv:
    def __init__(self, seed: Optional[int] = None, record_path: Optional[str] = None):
        self.rng = random.Random(seed)
        self.bag = SevenBag(self.rng)
        self.board: List[List[int]] = [[0]*BOARD_W for _ in range(BOARD_H)]
        self.cur_id: int = 0
        self.cur_rot: int = 0
        self.cur_x: int = 3  # top-left of 4x4 box
        self.cur_y: int = 0
        self.hold_id: Optional[int] = None
        self.can_hold: bool = True
        self.queue: List[int] = [self.bag.next() for _ in range(5)]
        self.score: int = 0
        self.lines: int = 0
        self.level: int = 1
        self.game_over: bool = False
        # Last-step derived info for richer observations
        self.last_lines_cleared: int = 0
        self.combo_count: int = 0
        self.prev_clear_was_tetris: bool = False
        self.b2b_active: bool = False
        self.last_covered_holes_delta: int = 0
        self.recorder = TransitionRecorder()
        if record_path:
            self.enable_recording(record_path)
        self._spawn_new()

    # ----------------- Recording -----------------
    def enable_recording(self, path: str):
        self.recorder.start(path)

    def disable_recording(self):
        self.recorder.stop()

    # ----------------- Utilities -----------------
    def _clone_board(self):
        return [row[:] for row in self.board]

    def _cur_matrix(self) -> List[List[int]]:
        return PIECES[self.cur_id][self.cur_rot]

    def _collides(self, x: int, y: int, rot: int) -> bool:
        mat = PIECES[self.cur_id][rot]
        for r in range(4):
            for c in range(4):
                if mat[r][c]:
                    bx = x + c
                    by = y + r
                    if bx < 0 or bx >= BOARD_W or by < 0 or by >= BOARD_H:
                        return True
                    if self.board[by][bx] != 0:
                        return True
        return False

    def _lock_piece(self):
        mat = self._cur_matrix()
        # Compute holes mask BEFORE placement to determine how many holes get covered
        pre_board = self._clone_board()
        pre_holes_mask = _compute_holes_mask(pre_board)
        for r in range(4):
            for c in range(4):
                if mat[r][c]:
                    bx = self.cur_x + c
                    by = self.cur_y + r
                    if 0 <= by < BOARD_H and 0 <= bx < BOARD_W:
                        self.board[by][bx] = self.cur_id
        # Compute covered holes delta: number of cells that were holes and are now filled by the locked piece
        covered = 0
        for r in range(4):
            for c in range(4):
                if mat[r][c]:
                    bx = self.cur_x + c
                    by = self.cur_y + r
                    if 0 <= by < BOARD_H and 0 <= bx < BOARD_W:
                        if pre_holes_mask[by][bx] == 1:
                            covered += 1
        self.last_covered_holes_delta = covered

        cleared = self._clear_lines()
        # Track last-step line clear info and combo/b2b
        self.last_lines_cleared = int(cleared)
        if cleared > 0:
            self.combo_count += 1
        else:
            self.combo_count = 0
        # Simplified B2B: only Tetrises (4 lines) count in this environment
        if cleared == 4 and self.prev_clear_was_tetris:
            self.b2b_active = True
        else:
            self.b2b_active = False
        # Update previous clear state (only True if this clear is a Tetris)
        self.prev_clear_was_tetris = (cleared == 4)
        self.lines += cleared
        # Simple reward mapping; you can change this for RL
        # Standard-like but small-scale rewards
        reward = {0:0, 1:1.0, 2:3.0, 3:5.0, 4:8.0}[cleared]
        self.score += int(reward * 10)
        self.can_hold = True
        self._spawn_new()
        return reward, cleared

    def _clear_lines(self) -> int:
        new_rows = [row for row in self.board if any(cell == 0 for cell in row)]
        cleared = BOARD_H - len(new_rows)
        while len(new_rows) < BOARD_H:
            new_rows.insert(0, [0]*BOARD_W)
        self.board = new_rows
        return cleared

    def _spawn_new(self):
        if not self.queue:
            self.queue.append(self.bag.next())
        self.cur_id = self.queue.pop(0)
        while len(self.queue) < 5:
            self.queue.append(self.bag.next())
        self.cur_rot = 0
        self.cur_x = 3
        self.cur_y = 0
        # If colliding at spawn, game over
        if self._collides(self.cur_x, self.cur_y, self.cur_rot):
            self.game_over = True

    def _hard_drop_distance(self) -> int:
        dy = 0
        y = self.cur_y
        while not self._collides(self.cur_x, y+1, self.cur_rot):
            y += 1
            dy += 1
        return dy

    def _rotate_with_kicks(self, dir: int):
        # dir: +1 cw, -1 ccw
        new_rot = (self.cur_rot + (1 if dir > 0 else 3)) % 4
        # try kicks
        kicks = [(0,0), (-1,0), (1,0), (-2,0), (2,0), (0,-1)]
        for dx, dy in kicks:
            nx, ny = self.cur_x + dx, self.cur_y + dy
            if not self._collides(nx, ny, new_rot):
                self.cur_x, self.cur_y, self.cur_rot = nx, ny, new_rot
                return True
        return False

    # ----------------- Public API -----------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.bag = SevenBag(self.rng)
        self.board = [[0]*BOARD_W for _ in range(BOARD_H)]
        self.cur_id = 0
        self.cur_rot = 0
        self.cur_x = 3
        self.cur_y = 0
        self.hold_id = None
        self.can_hold = True
        self.queue = [self.bag.next() for _ in range(5)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.last_lines_cleared = 0
        self.combo_count = 0
        self.prev_clear_was_tetris = False
        self.b2b_active = False
        self.last_covered_holes_delta = 0
        self._spawn_new()
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        # return numpy arrays if available, else lists
        board = self._clone_board()
        cur = {
            "id": self.cur_id,
            "rot": self.cur_rot,
            "x": self.cur_x,
            "y": self.cur_y,
            "matrix": PIECES[self.cur_id][self.cur_rot],
        }
        obs = {
            "board": np.array(board, dtype=np.int8) if np is not None else board,
            "current": cur,
            "next_queue": self.queue[:],
            "hold": self.hold_id if self.hold_id is not None else 0,
            "can_hold": self.can_hold,
            "score": self.score,
            "lines": self.lines,
            "level": self.level,
            "game_over": self.game_over,
            # Last-step info for richer observations
            "last_lines_cleared": self.last_lines_cleared,
            "combo": self.combo_count,
            "b2b": 1 if self.b2b_active else 0,
            "last_covered_holes_delta": self.last_covered_holes_delta,
        }
        return obs

    def step(self, action: int):
        if self.game_over:
            # Step in terminal state returns zero reward
            return self.get_observation(), 0.0, True, {"terminal": True}

        a = ACTIONS.get(action, "noop")
        pre_obs = self.get_observation()
        reward_from_lock = 0.0
        cleared = 0
        locked = False

        if a == "left":
            if not self._collides(self.cur_x - 1, self.cur_y, self.cur_rot):
                self.cur_x -= 1
        elif a == "right":
            if not self._collides(self.cur_x + 1, self.cur_y, self.cur_rot):
                self.cur_x += 1
        elif a == "rotate_cw":
            self._rotate_with_kicks(+1)
        elif a == "rotate_ccw":
            self._rotate_with_kicks(-1)
        elif a == "soft_drop":
            if not self._collides(self.cur_x, self.cur_y + 1, self.cur_rot):
                self.cur_y += 1
            else:
                reward_from_lock, cleared = self._lock_piece()
                locked = True
        elif a == "hard_drop":
            dist = self._hard_drop_distance()
            self.cur_y += dist
            reward_from_lock, cleared = self._lock_piece()
            # small shaped reward for hard drop distance
            reward_from_lock += 0.1 * dist
            locked = True
        elif a == "hold":
            if self.can_hold:
                if self.hold_id is None:
                    self.hold_id = self.cur_id
                    self._spawn_new()
                else:
                    self.hold_id, self.cur_id = self.cur_id, self.hold_id
                    self.cur_rot = 0
                    self.cur_x, self.cur_y = 3, 0
                    if self._collides(self.cur_x, self.cur_y, self.cur_rot):
                        self.game_over = True
                self.can_hold = False
        else:
            # noop
            pass

        # gravity tick (one step call = one frame; optional gravity can be applied externally in loop)
        # Here we do not apply automatic gravity inside step to keep it deterministic per action.

        done = self.game_over
        info = {
            "lines_cleared": cleared,
            "locked": locked,
            "combo": int(self.combo_count),
            "b2b": int(1 if self.b2b_active else 0),
            "covered_holes_delta": int(self.last_covered_holes_delta),
        }
        post_obs = self.get_observation()
        total_reward = reward_from_lock

        # Record transition
        self.recorder.record(
            _serialize_obs(pre_obs), action, float(total_reward), _serialize_obs(post_obs), done, info
        )

        return post_obs, float(total_reward), done, info

    def apply_gravity(self):
        if self.game_over:
            return 0.0, False, {"terminal": True}
        # soft drop by 1; if collides -> lock
        if not self._collides(self.cur_x, self.cur_y + 1, self.cur_rot):
            self.cur_y += 1
            return 0.0, False, {}
        else:
            r, cleared = self._lock_piece()
            return r, True, {"lines_cleared": cleared}

    # ----------------- Rendering (pygame) -----------------
    def render(self, surface, tile=30):
        if pygame is None:
            raise RuntimeError("pygame not available; pip install pygame")
        surface.fill((18,18,24))
        ox, oy = 20, 20
        # draw board grid
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                val = self.board[y][x]
                rect = pygame.Rect(ox + x*tile, oy + y*tile, tile, tile)
                pygame.draw.rect(surface, (40,40,50), rect, 1)
                if val:
                    pygame.draw.rect(surface, piece_color(val), rect.inflate(-2,-2))
        # draw current piece
        mat = self._cur_matrix()
        for r in range(4):
            for c in range(4):
                if mat[r][c]:
                    x = self.cur_x + c
                    y = self.cur_y + r
                    if y >= 0:
                        rect = pygame.Rect(ox + x*tile, oy + y*tile, tile, tile)
                        pygame.draw.rect(surface, piece_color(self.cur_id), rect.inflate(-2,-2))
        # ghost piece
        gy = self.cur_y + self._hard_drop_distance()
        for r in range(4):
            for c in range(4):
                if mat[r][c]:
                    x = self.cur_x + c
                    y = gy + r
                    if y >= 0:
                        rect = pygame.Rect(ox + x*tile, oy + y*tile, tile, tile)
                        pygame.draw.rect(surface, (200,200,200), rect, 1)
        # side panels
        draw_panel(surface, ox + BOARD_W*tile + 20, oy, self)


def piece_color(pid: int) -> Tuple[int,int,int]:
    return {
        1: (0, 240, 240),  # I cyan
        2: (240, 240, 0),  # O yellow
        3: (160, 0, 240),  # T purple
        4: (0, 240, 0),    # S green
        5: (240, 0, 0),    # Z red
        6: (0, 0, 240),    # J blue
        7: (240, 160, 0),  # L orange
    }[pid]


def draw_text(surface, text, pos, size=20, color=(230,230,230)):
    font = pygame.font.SysFont("consolas", size)
    surf = font.render(text, True, color)
    surface.blit(surf, pos)


def draw_panel(surface, px, py, env: TetrisEnv):
    # Info
    draw_text(surface, f"Score: {env.score}", (px, py))
    draw_text(surface, f"Lines: {env.lines}", (px, py+24))
    draw_text(surface, f"Level: {env.level}", (px, py+48))
    draw_text(surface, f"Hold:", (px, py+90))
    # hold piece
    draw_mini_piece(surface, px, py+120, env.hold_id)
    draw_text(surface, f"Next:", (px, py+180))
    ny = py + 210
    for i, pid in enumerate(env.queue[:5]):
        draw_mini_piece(surface, px, ny + i*60, pid)


def draw_mini_piece(surface, px, py, pid: Optional[int]):
    if pid is None or pid == 0:
        pygame.draw.rect(surface, (60,60,70), pygame.Rect(px, py, 100, 50), 1)
        return
    tile = 12
    mat = PIECES[pid][0]
    # center within 100x50
    offx, offy = px + 10, py + 10
    for r in range(4):
        for c in range(4):
            if mat[r][c]:
                rect = pygame.Rect(offx + c*tile, offy + r*tile, tile, tile)
                pygame.draw.rect(surface, piece_color(pid), rect.inflate(-1,-1))


# ---------- Helpers for JSON-safe obs ----------

def _serialize_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    def convert_board(b):
        if np is not None and isinstance(b, np.ndarray):
            return b.tolist()
        return b
    cur = obs["current"]
    return {
        "board": convert_board(obs["board"]),
        "current": {
            "id": int(cur["id"]),
            "rot": int(cur["rot"]),
            "x": int(cur["x"]),
            "y": int(cur["y"]),
            "matrix": cur["matrix"],
        },
        "next_queue": list(map(int, obs["next_queue"])),
        "hold": int(obs["hold"]) if obs["hold"] is not None else 0,
        "can_hold": bool(obs["can_hold"]),
        "score": int(obs["score"]),
        "lines": int(obs["lines"]),
        "level": int(obs["level"]),
        "game_over": bool(obs["game_over"]),
        "last_lines_cleared": int(obs.get("last_lines_cleared", 0)),
        "combo": int(obs.get("combo", 0)),
        "b2b": int(obs.get("b2b", 0)),
        "last_covered_holes_delta": int(obs.get("last_covered_holes_delta", 0)),
    }


# -----------------------------
# Interactive game with pygame
# -----------------------------

def run_game(record_path: Optional[str] = None):
    if pygame is None:
        raise RuntimeError("pygame not available. pip install pygame")
    pygame.init()
    W = 20 + BOARD_W*30 + 20 + 150
    H = 20 + BOARD_H*30 + 20
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tetris (AI-ready)")
    clock = pygame.time.Clock()
    env = TetrisEnv(seed=None, record_path=record_path)

    gravity_timer = 0.0
    gravity_interval = 0.6  # seconds per cell; tune for speed
    recording = record_path is not None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        gravity_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    env.step(0)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    env.step(1)
                elif event.key in (pygame.K_UP, pygame.K_x):
                    env.step(2)
                elif event.key in (pygame.K_z,):
                    env.step(3)
                elif event.key in (pygame.K_DOWN,):
                    env.step(4)
                elif event.key in (pygame.K_SPACE,):
                    env.step(5)
                elif event.key in (pygame.K_c,):
                    env.step(6)
                elif event.key in (pygame.K_p,):
                    # pause toggle (simple)
                    gravity_timer = -9999  # freeze
                elif event.key in (pygame.K_r,):
                    env.reset()
                    gravity_timer = 0.0
                elif event.key in (pygame.K_s,):
                    # toggle recording
                    if recording:
                        env.disable_recording()
                        recording = False
                    else:
                        env.enable_recording("tetris_dataset.jsonl")
                        recording = True

        # gravity
        if gravity_timer >= gravity_interval and gravity_timer >= 0:
            gravity_timer = 0.0
            env.apply_gravity()

        # render
        env.render(screen, tile=30)
        if recording:
            draw_text(screen, "REC", (W - 60, 10), size=24, color=(255,80,80))
        if env.game_over:
            draw_text(screen, "GAME OVER - Press R", (W//2 - 140, 10), size=24, color=(255,120,120))

        pygame.display.flip()

    pygame.quit()


# -----------------------------
# Minimal headless example
# -----------------------------

def random_rollout(steps=200, seed=0, record_path="tetris_dataset.jsonl"):
    env = TetrisEnv(seed=seed, record_path=record_path)
    total = 0.0
    for _ in range(steps):
        if env.game_over:
            env.reset()
        a = env.rng.randrange(0, 8)
        _, r, done, _ = env.step(a)
        total += r
        # apply gravity each step as well
        env.apply_gravity()
    env.disable_recording()
    return total


# -----------------------------
# High-Level Action Wrapper
# -----------------------------

# We define a fixed-size high-level action space that maps an index to
# a placement for the CURRENT piece: (rotation, target_x_of_4x4_top_left).
# For fixed-size compatibility, we use the maximum number of placements
# across any piece. For a 10x20 board this is typically ~34.

from typing import NamedTuple


class Placement(NamedTuple):
    rot: int
    x: int  # top-left of the 4x4 matrix on the board


def _allowed_x_positions_for_mat(mat: List[List[int]]) -> List[int]:
    # Identify occupied columns in the 4x4
    cols = [c for c in range(4) if any(mat[r][c] for r in range(4))]
    if not cols:
        return list(range(BOARD_W))
    min_c, max_c = min(cols), max(cols)
    # top-left x must satisfy: 0 <= x + min_c and x + max_c <= BOARD_W-1
    lo = -min_c
    hi = (BOARD_W - 1) - max_c
    return list(range(lo, hi + 1))


def _compute_piece_placements() -> Dict[int, List[Placement]]:
    by_piece: Dict[int, List[Placement]] = {}
    for pid in range(1, 8):
        placements: List[Placement] = []
        for rot in range(4):
            mat = PIECES[pid][rot]
            for x in _allowed_x_positions_for_mat(mat):
                placements.append(Placement(rot=rot, x=x))
        by_piece[pid] = placements
    return by_piece


PLACEMENTS_BY_PIECE: Dict[int, List[Placement]] = _compute_piece_placements()
MAX_ACTIONS_PER_PIECE: int = max(len(v) for v in PLACEMENTS_BY_PIECE.values())


def _compute_column_heights(board_list: List[List[int]]) -> List[int]:
    heights: List[int] = [0] * BOARD_W
    for x in range(BOARD_W):
        h = 0
        for y in range(BOARD_H):
            if board_list[y][x] != 0:
                h = BOARD_H - y
                break
        heights[x] = h
    return heights


def _compute_holes_mask(board_list: List[List[int]]) -> List[List[int]]:
    mask = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    heights = _compute_column_heights(board_list)
    for x in range(BOARD_W):
        h = heights[x]
        if h <= 0:
            continue
        top_y = BOARD_H - h
        for y in range(top_y, BOARD_H):
            if board_list[y][x] == 0:
                mask[y][x] = 1
    return mask


def _compute_surface_mask(board_list: List[List[int]]) -> List[List[int]]:
    mask = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    for x in range(BOARD_W):
        for y in range(BOARD_H):
            if board_list[y][x] != 0:
                mask[y][x] = 1
                break
    return mask


def _compute_well_mask(board_list: List[List[int]]) -> List[List[int]]:
    mask = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    heights = _compute_column_heights(board_list)
    for x in range(BOARD_W):
        h_x = heights[x]
        if x == 0:
            m = heights[1]
        elif x == BOARD_W - 1:
            m = heights[BOARD_W - 2]
        else:
            m = min(heights[x - 1], heights[x + 1])
        depth = m - h_x
        if depth > 0:
            y_start = BOARD_H - m
            y_end = BOARD_H - h_x  # exclusive
            for y in range(max(0, y_start), min(BOARD_H, y_end)):
                # Only mark if this cell is currently empty
                if board_list[y][x] == 0:
                    mask[y][x] = 1
    return mask


def _compute_overhang_mask(board_list: List[List[int]], holes_mask: List[List[int]]) -> List[List[int]]:
    mask = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    for y in range(1, BOARD_H):  # start at row 1 to allow y-1 indexing
        for x in range(BOARD_W):
            if holes_mask[y][x] == 1 and board_list[y - 1][x] != 0:
                mask[y - 1][x] = 1
    return mask


def _parity_plane_list() -> List[List[int]]:
    return [[(r + c) & 1 for c in range(BOARD_W)] for r in range(BOARD_H)]


def preprocess_observation(obs: Dict[str, Any]) -> Tuple[Any, Any]:
    # spatial planes: (7, 20, 10) -> [board_binary, current_piece, holes, surface, wells, overhang, parity]
    board = obs["board"]
    if np is None:
        board_list = board
        # Plane 1: board
        plane_board = [[1 if cell != 0 else 0 for cell in row] for row in board_list]
        # Plane 2: current piece
        cur = obs["current"]
        cur_x, cur_y = int(cur["x"]), int(cur["y"])
        cur_mat = cur["matrix"]
        plane_cur = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        for r in range(4):
            for c in range(4):
                if cur_mat[r][c]:
                    bx, by = cur_x + c, cur_y + r
                    if 0 <= bx < BOARD_W and 0 <= by < BOARD_H:
                        plane_cur[by][bx] = 1
        # Derived geometry
        holes_mask = _compute_holes_mask(board_list)
        surface_mask = _compute_surface_mask(board_list)
        well_mask = _compute_well_mask(board_list)
        overhang_mask = _compute_overhang_mask(board_list, holes_mask)
        parity_plane = _parity_plane_list()
        spatial = [
            plane_board,
            plane_cur,
            holes_mask,
            surface_mask,
            well_mask,
            overhang_mask,
            parity_plane,
        ]
    else:
        board_list = board.tolist() if isinstance(board, np.ndarray) else board
        arr = np.array(board_list, dtype=np.int8)
        plane_board = (arr != 0).astype(np.float32)
        # Plane 2: current piece
        cur = obs["current"]
        cur_x, cur_y = int(cur["x"]), int(cur["y"])
        cur_mat = cur["matrix"]
        plane_cur = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                if cur_mat[r][c]:
                    bx, by = cur_x + c, cur_y + r
                    if 0 <= bx < BOARD_W and 0 <= by < BOARD_H:
                        plane_cur[by, bx] = 1.0
        # Derived geometry (numpy)
        # Holes mask
        heights = np.zeros((BOARD_W,), dtype=np.int32)
        for x in range(BOARD_W):
            col = arr[:, x]
            nz = np.nonzero(col)[0]
            if nz.size > 0:
                top_y = int(nz[0])
                heights[x] = BOARD_H - top_y
            else:
                heights[x] = 0
        holes_mask = np.zeros_like(plane_board, dtype=np.float32)
        for x in range(BOARD_W):
            h = int(heights[x])
            if h <= 0:
                continue
            top_y = BOARD_H - h
            col = arr[top_y:, x]
            if col.size > 0:
                holes = (col == 0).astype(np.float32)
                holes_mask[top_y:, x] = holes
        # Surface mask
        surface_mask = np.zeros_like(plane_board, dtype=np.float32)
        for x in range(BOARD_W):
            col = arr[:, x]
            nz = np.nonzero(col)[0]
            if nz.size > 0:
                surface_mask[int(nz[0]), x] = 1.0
        # Well mask
        well_mask = np.zeros_like(plane_board, dtype=np.float32)
        for x in range(BOARD_W):
            h_x = int(heights[x])
            if x == 0:
                m = int(heights[1])
            elif x == BOARD_W - 1:
                m = int(heights[BOARD_W - 2])
            else:
                m = int(min(heights[x - 1], heights[x + 1]))
            depth = m - h_x
            if depth > 0:
                y_start = BOARD_H - m
                y_end = BOARD_H - h_x
                y_start = max(0, y_start)
                y_end = min(BOARD_H, y_end)
                # mark only empties as well cells
                col_slice = arr[y_start:y_end, x]
                if col_slice.size > 0:
                    mask_slice = (col_slice == 0).astype(np.float32)
                    well_mask[y_start:y_end, x] = mask_slice
        # Overhang mask (filled directly above a hole)
        overhang_mask = np.zeros_like(plane_board, dtype=np.float32)
        # For y from 1..H-1, if holes_mask[y,x]==1 and arr[y-1,x]!=0 => overhang_mask[y-1,x]=1
        if BOARD_H >= 2:
            over_positions = (holes_mask[1:, :] == 1.0) & (arr[:-1, :] != 0)
            overhang_mask[:-1, :] = over_positions.astype(np.float32)
        # Parity plane
        yy, xx = np.mgrid[0:BOARD_H, 0:BOARD_W]
        parity_plane = ((yy + xx) & 1).astype(np.float32)

        spatial = np.stack([
            plane_board.astype(np.float32),
            plane_cur.astype(np.float32),
            holes_mask.astype(np.float32),
            surface_mask.astype(np.float32),
            well_mask.astype(np.float32),
            overhang_mask.astype(np.float32),
            parity_plane.astype(np.float32),
        ], axis=0)

    # flat vector: next queue (5x7 one-hot), hold (7 one-hot), can_hold (1), plus 9 scalars
    next_q: List[int] = list(obs["next_queue"])[:5]
    hold_id: int = int(obs["hold"]) if obs["hold"] is not None else 0
    can_hold: bool = bool(obs["can_hold"])

    # Scalars
    last_lines = int(obs.get("last_lines_cleared", 0))
    combo = int(obs.get("combo", 0))
    is_b2b = int(obs.get("b2b", 0))

    # Compute board-derived scalars: heights, agg_height, max_height, bumpiness, hole_count, tetris_ready
    if np is None:
        board_list = board_list  # already ensured
        heights = _compute_column_heights(board_list)
        agg_height = int(sum(heights))
        max_height = int(max(heights) if heights else 0)
        bumpiness = int(sum(abs(heights[i] - heights[i + 1]) for i in range(BOARD_W - 1)))
        holes_mask_l = _compute_holes_mask(board_list)
        hole_count = int(sum(sum(row) for row in holes_mask_l))
        # tetris_ready: any well depth >= 4
        well_depths = []
        for x in range(BOARD_W):
            h_x = heights[x]
            if x == 0:
                m = heights[1]
            elif x == BOARD_W - 1:
                m = heights[BOARD_W - 2]
            else:
                m = min(heights[x - 1], heights[x + 1])
            well_depths.append(max(0, m - h_x))
        tetris_ready = 1 if any(d >= 4 for d in well_depths) else 0
        covered_holes_delta_last = int(obs.get("last_covered_holes_delta", 0))
    else:
        heights_list = heights.tolist() if 'heights' in locals() else _compute_column_heights(board_list)
        agg_height = int(sum(heights_list))
        max_height = int(max(heights_list) if heights_list else 0)
        bumpiness = int(sum(abs(heights_list[i] - heights_list[i + 1]) for i in range(BOARD_W - 1)))
        hole_count = int(holes_mask.sum()) if 'holes_mask' in locals() and isinstance(holes_mask, np.ndarray) else int(sum(sum(row) for row in _compute_holes_mask(board_list)))
        well_depths = []
        for x in range(BOARD_W):
            h_x = int(heights_list[x])
            if x == 0:
                m = int(heights_list[1])
            elif x == BOARD_W - 1:
                m = int(heights_list[BOARD_W - 2])
            else:
                m = int(min(heights_list[x - 1], heights_list[x + 1]))
            well_depths.append(max(0, m - h_x))
        tetris_ready = 1 if any(d >= 4 for d in well_depths) else 0
        covered_holes_delta_last = int(obs.get("last_covered_holes_delta", 0))

    if np is None:
        next_oh = []
        for pid in next_q:
            row = [0]*7
            if 1 <= pid <= 7:
                row[pid-1] = 1
            next_oh.extend(row)
        hold_oh = [0]*7
        if 1 <= hold_id <= 7:
            hold_oh[hold_id-1] = 1
        extra_scalars = [
            last_lines,
            combo,
            is_b2b,
            agg_height,
            max_height,
            bumpiness,
            hole_count,
            covered_holes_delta_last,
            int(tetris_ready),
        ]
        flat = next_oh + hold_oh + [1 if can_hold else 0] + extra_scalars
    else:
        next_oh = np.zeros((5, 7), dtype=np.float32)
        for i, pid in enumerate(next_q):
            if 1 <= pid <= 7:
                next_oh[i, pid-1] = 1.0
        hold_oh = np.zeros((7,), dtype=np.float32)
        if 1 <= hold_id <= 7:
            hold_oh[hold_id-1] = 1.0
        extra = np.array([
            float(last_lines),
            float(combo),
            float(is_b2b),
            float(agg_height),
            float(max_height),
            float(bumpiness),
            float(hole_count),
            float(covered_holes_delta_last),
            float(tetris_ready),
        ], dtype=np.float32)
        flat = np.concatenate([
            next_oh.reshape(-1),
            hold_oh,
            np.array([1.0 if can_hold else 0.0], dtype=np.float32),
            extra,
        ])

    return spatial, flat


# -----------------------------
# Candidate Simulation Helpers
# -----------------------------

def _board_collides(board_bin: List[List[int]], mat: List[List[int]], x: int, y: int) -> bool:
    for r in range(4):
        for c in range(4):
            if mat[r][c]:
                bx = x + c
                by = y + r
                if bx < 0 or bx >= BOARD_W or by < 0 or by >= BOARD_H:
                    return True
                if board_bin[by][bx] != 0:
                    return True
    return False


def _hard_drop_y_on_board(board_bin: List[List[int]], mat: List[List[int]], x: int) -> Optional[int]:
    # Drop from y=0 down until collision would occur on next step
    # If collides at y=0 immediately and cannot move down, still try to find a non-colliding y >= 0
    y = 0
    # If immediately colliding at y=0 and also at y=1 etc., there is no valid landing
    # We'll find first y where not colliding; if none, return None
    # First, find first non-colliding y
    while y < BOARD_H and _board_collides(board_bin, mat, x, y):
        y += 1
    if y >= BOARD_H:
        return None
    # Now descend until next step collides
    while y + 1 < BOARD_H and not _board_collides(board_bin, mat, x, y + 1):
        y += 1
    return y


def _apply_piece_on_board(board_bin: List[List[int]], mat: List[List[int]], x: int, y: int) -> List[List[int]]:
    newb = [row[:] for row in board_bin]
    for r in range(4):
        for c in range(4):
            if mat[r][c]:
                bx = x + c
                by = y + r
                if 0 <= bx < BOARD_W and 0 <= by < BOARD_H:
                    newb[by][bx] = 1
    return newb


def _clear_lines_on_board(board_bin: List[List[int]]) -> Tuple[List[List[int]], int]:
    new_rows = [row[:] for row in board_bin if any(cell == 0 for cell in row)]
    cleared = BOARD_H - len(new_rows)
    while len(new_rows) < BOARD_H:
        new_rows.insert(0, [0] * BOARD_W)
    return new_rows, cleared


def _well_depths_from_heights(heights: List[int]) -> List[int]:
    depths: List[int] = []
    for x in range(BOARD_W):
        h_x = heights[x]
        if x == 0:
            m = heights[1]
        elif x == BOARD_W - 1:
            m = heights[BOARD_W - 2]
        else:
            m = min(heights[x - 1], heights[x + 1])
        depths.append(max(0, m - h_x))
    return depths


def build_candidate_tensors_from_board_plane(board_plane: Any, piece_id: int, pad_to: int = MAX_ACTIONS_PER_PIECE) -> Tuple[Any, Any, int]:
    """Compute candidate rasters and features for the given board occupancy and current piece.

    Returns (rasters, feats, K) where
      - rasters: (pad_to, 1, H, W) float32
      - feats:   (pad_to, F) float32 with F=6
      - K: number of valid candidates for this piece (<= pad_to)
    """
    F_DIM = 6  # [lines_cleared, holes_after, agg_height_after, bumpiness_after, well_depth_max_after, covered_holes]

    # Normalize board to list[List[int]] binary
    if np is not None and hasattr(board_plane, "astype"):
        b = (board_plane.astype("uint8")).tolist()
    else:
        b = [[1 if cell else 0 for cell in row] for row in board_plane]

    placements = PLACEMENTS_BY_PIECE[int(piece_id)]
    K = len(placements)
    H, W = BOARD_H, BOARD_W

    if np is not None:
        rasters = np.zeros((pad_to, 1, H, W), dtype=np.float32)
        feats = np.zeros((pad_to, F_DIM), dtype=np.float32)
    else:
        rasters = [[[ [0.0 for _ in range(W)] for _ in range(H) ]] for _ in range(pad_to)]
        feats = [[0.0 for _ in range(F_DIM)] for _ in range(pad_to)]

    pre_holes_mask = _compute_holes_mask(b)

    for i, plc in enumerate(placements):
        mat = PIECES[piece_id][plc.rot]
        y = _hard_drop_y_on_board(b, mat, plc.x)
        if y is None:
            # Unreachable; leave zeros
            continue
        # piece raster at landing
        if np is not None:
            pr = np.zeros((H, W), dtype=np.float32)
            for r in range(4):
                for c in range(4):
                    if mat[r][c]:
                        bx, by = plc.x + c, y + r
                        if 0 <= bx < W and 0 <= by < H:
                            pr[by, bx] = 1.0
            rasters[i, 0] = pr
        else:
            pr = [[0.0 for _ in range(W)] for _ in range(H)]
            for r in range(4):
                for c in range(4):
                    if mat[r][c]:
                        bx, by = plc.x + c, y + r
                        if 0 <= bx < W and 0 <= by < H:
                            pr[by][bx] = 1.0
            rasters[i][0] = pr

        # apply piece, clear lines
        b_after = _apply_piece_on_board(b, mat, plc.x, y)
        b_after, lines_cleared = _clear_lines_on_board(b_after)

        # covered holes: how many previously hole cells are now filled by piece
        covered = 0
        for r in range(4):
            for c in range(4):
                if mat[r][c]:
                    bx, by = plc.x + c, y + r
                    if 0 <= bx < W and 0 <= by < H:
                        if pre_holes_mask[by][bx] == 1:
                            covered += 1

        heights = _compute_column_heights(b_after)
        agg_height = sum(heights)
        bumpiness = sum(abs(heights[j] - heights[j + 1]) for j in range(W - 1))
        well_depth_max = max(_well_depths_from_heights(heights)) if W >= 2 else 0
        holes_after = sum(sum(row) for row in _compute_holes_mask(b_after))

        if np is not None:
            feats[i] = np.array([
                float(lines_cleared),
                float(holes_after),
                float(agg_height),
                float(bumpiness),
                float(well_depth_max),
                float(covered),
            ], dtype=np.float32)
        else:
            feats[i] = [
                float(lines_cleared),
                float(holes_after),
                float(agg_height),
                float(bumpiness),
                float(well_depth_max),
                float(covered),
            ]

    return rasters, feats, K


class TetrisEnvWrapper:
    """High-level action wrapper over TetrisEnv.

    - Fixed action space size: MAX_ACTIONS_PER_PIECE
    - Each action index is interpreted as a placement for the CURRENT piece,
      by indexing into that piece's (rotation, x) list modulo its length.
    - Executes low-level actions deterministically: rotate, move, hard drop.
    """

    def __init__(self, seed: Optional[int] = None, preprocess: bool = True, record_path: Optional[str] = None):
        self.env = TetrisEnv(seed=seed, record_path=record_path)
        self.preprocess = preprocess
        self.action_space_n = MAX_ACTIONS_PER_PIECE

    def reset(self, seed: Optional[int] = None):
        obs = self.env.reset(seed=seed)
        return preprocess_observation(obs) if self.preprocess else obs

    # Rendering passthrough using the underlying env's renderer
    def render(self, surface, tile: int = 30):
        if pygame is None:
            raise RuntimeError("pygame not available; pip install pygame")
        self.env.render(surface, tile=tile)

    def _map_action_to_current_piece(self, action_id: int) -> Placement:
        # map a global index [0, MAX_ACTIONS_PER_PIECE) to current piece's placement via modulo
        cur_pid = self.env.cur_id
        plist = PLACEMENTS_BY_PIECE[cur_pid]
        idx = int(action_id) % len(plist)
        return plist[idx]

    def _do_rotations(self, target_rot: int):
        # rotate the minimal number of steps, preferring ccw if 3 steps cw
        cur_rot = self.env.cur_rot
        diff = (target_rot - cur_rot) % 4
        if diff == 0:
            return
        if diff == 3:
            self.env.step(3)  # rotate_ccw
        else:
            for _ in range(diff):
                self.env.step(2)  # rotate_cw

    def _move_horizontally(self, target_x: int):
        # Move left/right toward target_x while possible
        while self.env.cur_x != target_x:
            if self.env.cur_x > target_x:
                prev_x = self.env.cur_x
                self.env.step(0)  # left
                if self.env.cur_x == prev_x:
                    break  # blocked
            else:
                prev_x = self.env.cur_x
                self.env.step(1)  # right
                if self.env.cur_x == prev_x:
                    break  # blocked

    def step(self, action_id: int):
        if self.env.game_over:
            obs = self.env.get_observation()
            return (preprocess_observation(obs) if self.preprocess else obs), 0.0, True, {"terminal": True}

        # Map to placement for the current piece
        placement = self._map_action_to_current_piece(action_id)

        # Execute low-level sequence: rotate -> move -> hard drop
        pre_obs = self.env.get_observation()
        pre_board = pre_obs["board"]

        # Utilities to compute board metrics for shaping
        def _count_holes(board) -> int:
            holes = 0
            # iterate columns; count empty cells below the first filled cell
            for x in range(BOARD_W):
                seen_block = False
                for y in range(BOARD_H):
                    val = board[y][x]
                    if val:
                        seen_block = True
                    else:
                        if seen_block:
                            holes += 1
            return holes

        def _max_stack_height(board) -> int:
            max_h = 0
            for x in range(BOARD_W):
                col_h = 0
                for y in range(BOARD_H):
                    if board[y][x]:
                        # height measured from bottom in blocks
                        col_h = BOARD_H - y
                        break
                if col_h > max_h:
                    max_h = col_h
            return max_h

        pre_holes = _count_holes(pre_board)
        pre_max_h = _max_stack_height(pre_board)
        total_reward = 0.0
        locked = False

        self._do_rotations(placement.rot)
        self._move_horizontally(placement.x)
        _, r, done, info_last = self.env.step(5)  # hard_drop
        total_reward += float(r)
        locked = info_last.get("locked", False)

        post_obs = self.env.get_observation()
        post_board = post_obs["board"]

        # Reward shaping applied only when a piece locks
        if locked:
            post_holes = _count_holes(post_board)
            post_max_h = _max_stack_height(post_board)

            # Penalty if holes increased (flat -0.2)
            if post_holes > pre_holes:
                total_reward += -0.2

            # Penalty for increasing maximum stack height (-0.1 per block)
            if post_max_h > pre_max_h:
                total_reward += -0.1 * float(post_max_h - pre_max_h)
        info = {
            "locked": locked,
            "lines_cleared": info_last.get("lines_cleared", 0),
            "combo": info_last.get("combo", 0),
            "b2b": info_last.get("b2b", 0),
            "covered_holes_delta": info_last.get("covered_holes_delta", 0),
            "action_executed": {
                "rot": placement.rot,
                "x": placement.x,
            },
        }
        return (preprocess_observation(post_obs) if self.preprocess else post_obs), total_reward, done, info

    def valid_action_mask(self) -> List[int]:
        # 1 for indices < len(plist), else 0
        n = len(PLACEMENTS_BY_PIECE[self.env.cur_id])
        return [1 if i < n else 0 for i in range(self.action_space_n)]

    def observe(self):
        obs = self.env.get_observation()
        return preprocess_observation(obs) if self.preprocess else obs

    def current_piece_id(self) -> int:
        return int(self.env.cur_id)

    # expose some underlying env data
    @property
    def done(self) -> bool:
        return self.env.game_over

    def apply_gravity(self):
        return self.env.apply_gravity()


# -----------------------------
# Vectorized Environments
# -----------------------------

class SyncVecTetris:
    """Synchronous in-process vectorized environment.
    Useful fallback without multiprocessing overhead.
    """

    def __init__(self, num_envs: int, seeds: Optional[List[Optional[int]]] = None, preprocess: bool = True):
        self.num_envs = num_envs
        if seeds is None:
            seeds = [None] * num_envs
        self.envs = [TetrisEnvWrapper(seed=s, preprocess=preprocess) for s in seeds]
        self.action_space_n = self.envs[0].action_space_n

    def reset(self, seeds: Optional[List[Optional[int]]] = None):
        obs_batch = []
        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds is not None else None
            obs_batch.append(env.reset(seed=seed))
        return obs_batch

    def step(self, actions: List[int]):
        results = [env.step(int(a)) for env, a in zip(self.envs, actions)]
        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def valid_action_masks(self) -> List[List[int]]:
        return [env.valid_action_mask() for env in self.envs]

    def current_piece_ids(self) -> List[int]:
        return [env.current_piece_id() for env in self.envs]

    def close(self):
        pass


class SubprocVecTetris:
    """Multiprocess vectorized env using multiprocessing.Pipe.

    Note: On some platforms, 'spawn' start method is used, so ensure this
    code runs under a __main__ guard when creating instances.
    """

    def __init__(self, num_envs: int, seeds: Optional[List[Optional[int]]] = None, preprocess: bool = True):
        import multiprocessing as mp
        self.num_envs = num_envs
        self._ctx = mp.get_context("spawn")
        if seeds is None:
            seeds = [None] * num_envs
        self.remotes, self.work_remotes = zip(*[self._ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        self.action_space_n = MAX_ACTIONS_PER_PIECE
        for wr, seed in zip(self.work_remotes, seeds):
            p = self._ctx.Process(target=_worker, args=(wr, seed, preprocess))
            p.daemon = True
            p.start()
            self.processes.append(p)
        # close worker ends in parent
        for wr in self.work_remotes:
            wr.close()

    def reset(self, seeds: Optional[List[Optional[int]]] = None):
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", None if seeds is None else seeds[i]))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions: List[int]):
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", int(a)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    def valid_action_masks(self) -> List[List[int]]:
        for remote in self.remotes:
            remote.send(("mask", None))
        return [remote.recv() for remote in self.remotes]

    def current_piece_ids(self) -> List[int]:
        for remote in self.remotes:
            remote.send(("pid", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=1.0)


def _worker(remote, seed: Optional[int], preprocess: bool):
    env = TetrisEnvWrapper(seed=seed, preprocess=preprocess)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            ob = env.reset(seed=data)
            remote.send(ob)
        elif cmd == "step":
            ob, r, d, info = env.step(int(data))
            remote.send((ob, r, d, info))
        elif cmd == "mask":
            remote.send(env.valid_action_mask())
        elif cmd == "pid":
            remote.send(env.current_piece_id())
        elif cmd == "close":
            remote.close()
            break
        else:
            remote.send((None, None))


def make_vec_env(num_envs: int, seeds: Optional[List[Optional[int]]] = None, preprocess: bool = True, backend: str = "sync"):
    """Factory for vectorized Tetris envs.

    backend: 'sync' (in-process) or 'subproc' (multiprocessing)
    """
    if backend == "sync":
        return SyncVecTetris(num_envs=num_envs, seeds=seeds, preprocess=preprocess)
    elif backend == "subproc":
        return SubprocVecTetris(num_envs=num_envs, seeds=seeds, preprocess=preprocess)
    else:
        raise ValueError("Unsupported backend: {}".format(backend))


if __name__ == "__main__":
    # CLI:
    #   python tetris_ai_env.py            -> run game (no recording)
    #   python tetris_ai_env.py record     -> run game and record to tetris_dataset.jsonl
    #   python tetris_ai_env.py headless   -> run a random rollout to produce dataset
    mode = sys.argv[1] if len(sys.argv) > 1 else "game"
    if mode == "game":
        run_game(None)
    elif mode == "record":
        run_game("tetris_dataset.jsonl")
    elif mode == "headless":
        total = random_rollout(steps=500)
        print("Random rollout finished. Total reward:", total)
    elif mode == "vec_headless":
        # demo: vectorized random rollout with high-level actions
        vec = make_vec_env(num_envs=4, preprocess=True, backend="sync")
        obs_batch = vec.reset()
        total = [0.0] * 4
        steps = 100
        import random as _r
        for _ in range(steps):
            actions = [_r.randrange(0, vec.action_space_n) for _ in range(vec.num_envs)]
            obs_batch, rewards, dones, infos = vec.step(actions)
            total = [t + r for t, r in zip(total, rewards)]
            # optional gravity tick
            # not applied here; wrapper uses hard drop
            # reset on done for demo
            for i, d in enumerate(dones):
                if d:
                    obs_batch[i] = vec.envs[i].reset()
        print("Vec rollout totals:", total)
    else:
        print("Unknown mode. Use: game | record | headless")
