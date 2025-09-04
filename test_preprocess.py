"""
Lightweight unit tests for preprocess_observation geometry planes and scalars.

Run: python test_preprocess.py
"""

from typing import List

from tetris import (
    TetrisEnv,
    preprocess_observation,
    BOARD_W,
    BOARD_H,
)


def make_board_from_heights(heights: List[int]) -> List[List[int]]:
    assert len(heights) == BOARD_W
    board = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    for x, h in enumerate(heights):
        if h <= 0:
            continue
        top_y = BOARD_H - h
        for y in range(top_y, BOARD_H):
            board[y][x] = 1
    return board


def test_holes_mask():
    env = TetrisEnv(seed=0)
    obs = env.get_observation()
    # Create a hole: put a single filled cell at y=10 in column x=3
    board = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    board[10][3] = 1
    obs["board"] = board
    spatial, flat = preprocess_observation(obs)
    # holes plane index = 2
    holes = spatial[2] if not hasattr(spatial, "shape") else spatial[2]
    holes_sum = sum(sum(row) for row in holes) if not hasattr(holes, "sum") else float(holes.sum())
    assert holes_sum > 0.0, f"Expected holes_mask sum > 0, got {holes_sum}"


def test_well_mask_and_bumpiness():
    env = TetrisEnv(seed=0)
    obs = env.get_observation()
    # Flat board baseline with height 4 across
    flat_heights = [4] * BOARD_W
    board_flat = make_board_from_heights(flat_heights)
    obs["board"] = board_flat
    spatial_flat, flat_flat = preprocess_observation(obs)
    # bumpiness index: 35 (queue) + 7 (hold) + 1 (can_hold) + 3 (lines, combo, b2b) + 2 (agg, max) = 48
    bump_idx = 35 + 7 + 1 + 3 + 2
    bumpiness_flat = float(flat_flat[bump_idx]) if hasattr(flat_flat, "__getitem__") else float(flat_flat[bump_idx])

    # Board with center trench (column 5 low, neighbors high)
    trench_heights = [4, 4, 6, 6, 6, 0, 6, 6, 4, 4]
    board_trench = make_board_from_heights(trench_heights)
    obs["board"] = board_trench
    spatial_trench, flat_trench = preprocess_observation(obs)

    well_plane = spatial_trench[4] if not hasattr(spatial_trench, "shape") else spatial_trench[4]
    well_sum = sum(sum(row) for row in well_plane) if not hasattr(well_plane, "sum") else float(well_plane.sum())
    assert well_sum > 0.0, "Expected well_mask to mark trench cells"

    bumpiness_trench = float(flat_trench[bump_idx])
    assert (
        bumpiness_trench > bumpiness_flat
    ), f"Expected bumpiness to increase: {bumpiness_trench} <= {bumpiness_flat}"


def test_lines_cleared_last_scalar():
    env = TetrisEnv(seed=0)
    # Build board with bottom row having two holes at x=4 and x=5
    for x in range(BOARD_W):
        env.board[BOARD_H - 1][x] = 1
    env.board[BOARD_H - 1][4] = 0
    env.board[BOARD_H - 1][5] = 0

    # Set current piece to O and align to fill the two holes
    env.cur_id = 2  # O
    env.cur_rot = 0
    # For O piece, occupied columns are cur_x+1 and cur_x+2 in the 4x4 mat
    # To fill x=4 and x=5, set cur_x=3
    env.cur_x = 3
    env.cur_y = 0

    post_obs, r, done, info = env.step(5)  # hard_drop
    spatial, flat = preprocess_observation(post_obs)
    lines_last_idx = 35 + 7 + 1  # index of lines_cleared_last
    lines_last = int(flat[lines_last_idx])
    assert lines_last == 1, f"Expected last_lines_cleared=1, got {lines_last}"


if __name__ == "__main__":
    test_holes_mask()
    print("test_holes_mask: OK")
    test_well_mask_and_bumpiness()
    print("test_well_mask_and_bumpiness: OK")
    test_lines_cleared_last_scalar()
    print("test_lines_cleared_last_scalar: OK")
    print("All tests passed.")
