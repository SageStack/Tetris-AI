Tetris PPO – Training and Hyperparameter Sweep

Quick start
- Train with defaults: `python train.py`
- Train with overrides and TensorBoard logs to a fixed dir:
  `python train.py --run-dir runs/exp1 --num-envs 8 --n-steps 1024 --ppo-epochs 2 --batch-size 256 --learning-rate 3e-4 --total-timesteps 500000`
- Run an Optuna sweep (TPE + ASHA), rendering the first trial:
  `python sweep.py --trials 20 --study-name ppo-opt --profile balanced`

Notes
- Metrics: each training run writes `config.json` and `metrics.json` under its run directory, and streams scalars to TensorBoard.
- Device: prefers CUDA, then MPS (Apple Silicon), else CPU.
- Rendering: add `--render` to visualize an observer environment during training (slows training).
- Sweeps: `sweep.py` orchestrates training via subprocess and uses Optuna. Install with `pip install optuna` if missing.
