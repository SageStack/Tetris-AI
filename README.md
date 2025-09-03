Tetris PPO – Training and Hyperparameter Sweep

Quick start
- Train with defaults: `python train.py`
- Train with overrides and TensorBoard logs to a fixed dir:
  `python train.py --run-dir runs/exp1 --num-envs 8 --n-steps 1024 --ppo-epochs 2 --batch-size 256 --learning-rate 3e-4 --total-timesteps 500000`
- Run an Optuna sweep (TPE + ASHA). By default, every trial opens a viewer for stage 1:
  `python sweep.py --trials 20 --study-name ppo-opt --profile balanced`
  - Render all stages of each trial: add `--render-all-stages`
  - Disable rendering entirely: add `--no-render`

Notes
- Metrics: each training run writes `config.json` and `metrics.json` under its run directory, and streams scalars to TensorBoard.
- Device: prefers CUDA, then MPS (Apple Silicon), else CPU.
- Rendering: add `--render` to visualize an observer environment during training (slows training).
- Sweeps: `sweep.py` orchestrates training via subprocess and uses Optuna. Install with `pip install optuna` if missing.
  - When rendering during sweeps, the viewer auto-closes between stages/trials by default to avoid blocking. In standalone runs, this behavior is also the default; use `--no-auto-close-render` to keep the window open for a completion screen.
  - Resuming: sweeps use a persistent SQLite storage under `runs/sweeps/<study_name>/study.db` by default. If interrupted, re-run the same command and it will continue scheduling new trials up to the requested total. Completed stages are detected via `metrics.json` and won’t be rerun.
