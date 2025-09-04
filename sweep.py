"""
Optuna-based hyperparameter sweep for Tetris PPO using TPE + ASHA.

This script orchestrates sweeps by calling train.py as a subprocess.
It reports intermediate scores to Optuna so the ASHA pruner can stop
underperforming trials early. By default it renders the first trial's
first stage so you get a live viewer when the sweep starts.

Examples:
  python sweep.py --trials 20 --study-name ppo-tpe --profile balanced
  python sweep.py --trials 30 --profile thorough --render-every 5

Notes:
- Requires: pip install optuna
- Each stage writes metrics.json under the stage directory; the objective
  reads mean_return_last_100 as the stage score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # load variables from .env if present

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
    from optuna.storages import RDBStorage
except Exception as e:
    print("Error: Optuna is required for this sweep runner. Install with 'pip install optuna'.")
    raise

# Persistent storage configuration for scalable, parallel sweeps.
# Prefer DB_URL from environment/.env; fallback to OPTUNA_DB_URL for compatibility.
# Example PostgreSQL URL:
#   postgresql+psycopg2://USER:PASSWORD@HOST:5432/optuna
# For quick local testing, you can switch to SQLite with:
#   sqlite:///db.sqlite3
# Note: SQLite is fine for a single machine; use PostgreSQL for true parallel, multi-host sweeps.
DB_URL = os.environ.get("DB_URL") or os.environ.get("OPTUNA_DB_URL", "")


def _parse_render_selector(args) -> callable:
    """Build a predicate deciding whether to render a given trial index (1-based).

    Supports:
      --render-first
      --render-all
      --render-trials "1,5,9"
      --render-every K
    Precedence: all > trials list > every > first.
    """
    if not getattr(args, "render", True):
        return lambda i: False
    if getattr(args, "render_all", False):
        return lambda i: True
    if getattr(args, "render_trials", None):
        raw = str(args.render_trials)
        idxs = set()
        for tok in raw.split(','):
            tok = tok.strip()
            if tok.isdigit():
                idxs.add(int(tok))
        if idxs:
            return lambda i: i in idxs
    if getattr(args, "render_every", None):
        try:
            k = int(args.render_every)
            if k > 0:
                return lambda i: (i - 1) % k == 0
        except Exception:
            pass
    if getattr(args, "render_first", False):
        return lambda i: i == 1
    # Default: render every trial when rendering is enabled
    return lambda i: True


def _loguniform(lo: float, hi: float) -> float:
    """Sample log-uniform in [lo, hi]."""
    assert lo > 0 and hi > lo
    r = random.random()
    return math.exp(math.log(lo) * (1 - r) + math.log(hi) * r)


def suggest_hparams(trial: "optuna.Trial", profile: str) -> dict:
    """Suggest a hyperparameter set using Optuna for the given profile."""
    if profile == "quick":
        num_envs = trial.suggest_categorical("num_envs", [8, 12])
        n_steps = trial.suggest_categorical("n_steps", [512, 1024])
        ppo_epochs = trial.suggest_categorical("ppo_epochs", [2])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
        clip_coef = trial.suggest_float("clip_coef", 0.15, 0.25)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
        vf_coef = trial.suggest_float("vf_coef", 0.4, 0.8)
        gamma = trial.suggest_float("gamma", 0.99, 0.997)
        gae_lambda = trial.suggest_float("gae_lambda", 0.92, 0.96)
    elif profile == "thorough":
        num_envs = trial.suggest_categorical("num_envs", [8, 12, 16, 24, 32])
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        ppo_epochs = trial.suggest_categorical("ppo_epochs", [2, 3, 4])
        batch_size = trial.suggest_categorical(
            "batch_size", [64, 128, 256, 512])
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        clip_coef = trial.suggest_float("clip_coef", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
        vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
        gamma = trial.suggest_float("gamma", 0.985, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.97)
    else:  # balanced
        num_envs = trial.suggest_categorical("num_envs", [8, 12, 16, 24])
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        ppo_epochs = trial.suggest_categorical("ppo_epochs", [2, 3])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        clip_coef = trial.suggest_float("clip_coef", 0.15, 0.25)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.015)
        vf_coef = trial.suggest_float("vf_coef", 0.4, 0.9)
        gamma = trial.suggest_float("gamma", 0.99, 0.998)
        gae_lambda = trial.suggest_float("gae_lambda", 0.92, 0.96)

    # Ensure batch size is compatible with rollout
    batch_total = int(num_envs) * int(n_steps)
    if batch_size > batch_total:
        batch_size = max(64, batch_total // 4)

    return {
        "num_envs": int(num_envs),
        "n_steps": int(n_steps),
        "ppo_epochs": int(ppo_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "clip_coef": float(clip_coef),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
    }


def _run_stage(stage_dir: Path, cfg: dict, total_timesteps: int, backend: str, base_seed: int | None, trial_number: int, stage_index: int, render: bool, render_tile: int) -> float:
    """Run one training stage and return the stage score (mean_return_last_100)."""
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Fast-path resume: if a previous run finished this stage and wrote metrics,
    # reuse it instead of rerunning training.
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            score = float(metrics.get("mean_return_last_100", 0.0))
            print(
                f"[sweep] resume: trial {trial_number:03d} stage {stage_index} already complete; score={score:.3f}")
            return score
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--run-dir",
        str(stage_dir),
        "--total-timesteps",
        str(int(total_timesteps)),
        "--backend",
        backend,
        "--notes",
        f"optuna trial {trial_number} stage {stage_index}",
        # overrides
        "--num-envs",
        str(cfg["num_envs"]),
        "--n-steps",
        str(cfg["n_steps"]),
        "--ppo-epochs",
        str(cfg["ppo_epochs"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--clip-coef",
        str(cfg["clip_coef"]),
        "--ent-coef",
        str(cfg["ent_coef"]),
        "--vf-coef",
        str(cfg["vf_coef"]),
        "--gamma",
        str(cfg["gamma"]),
        "--gae-lambda",
        str(cfg["gae_lambda"]),
    ]
    if render:
        cmd += ["--render", "--tile",
                str(int(render_tile)), "--auto-close-render"]
    if base_seed is not None:
        cmd += ["--seed", str(int(base_seed) + trial_number)]

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    print(
        f"[sweep] trial {trial_number:03d} stage {stage_index} -> {stage_dir}")
    subprocess.run(cmd, env=env, check=True)

    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            return float(metrics.get("mean_return_last_100", 0.0))
        except Exception as e:
            print(
                f"[sweep] warning: failed to parse metrics in {metrics_path}: {e}")
            return 0.0
    print(f"[sweep] warning: metrics.json missing for stage {stage_dir}")
    return 0.0


def _parse_stages(args) -> list[int]:
    # Allow explicit stages or (base, rungs, growth)
    if args.stages:
        try:
            vals = [int(x.strip())
                    for x in str(args.stages).split(',') if x.strip()]
            if vals:
                return vals
        except Exception:
            pass
    stages = [int(args.base_steps)]
    for _ in range(1, int(args.rungs)):
        stages.append(int(stages[-1] * float(args.growth)))
    return stages


def main():
    ap = argparse.ArgumentParser(
        description="Optuna TPE + ASHA sweep for Tetris PPO")
    ap.add_argument("--trials", type=int, default=20,
                    help="Number of Optuna trials")
    ap.add_argument("--study-name", type=str, default="ppo-optuna",
                    help="Study name under runs/sweeps/")
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage URL (e.g., sqlite:///study.db)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for reproducibility")
    ap.add_argument("--backend", type=str,
                    choices=["sync", "subproc"], default="sync")
    ap.add_argument("--profile", type=str,
                    choices=["quick", "balanced", "thorough"], default="balanced")
    # Stages: either explicit or geometric schedule
    ap.add_argument("--stages", type=str, default=None,
                    help="Comma-separated timesteps per stage, e.g., '15000,60000,180000'")
    ap.add_argument("--base-steps", dest="base_steps", type=int, default=15000,
                    help="Base timesteps for stage 1 (if --stages not set)")
    ap.add_argument("--rungs", type=int, default=3,
                    help="Number of stages for ASHA (if --stages not set)")
    ap.add_argument("--growth", type=float, default=3.0,
                    help="Multiplier for each subsequent stage (if --stages not set)")
    # Rendering control (default: render first trial's first stage)
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="Disable all rendering during sweep")
    ap.set_defaults(render=True)
    ap.add_argument("--render-first", action="store_true",
                    default=False, help="Render the first trial only")
    ap.add_argument("--render-all", action="store_true",
                    default=False, help="Render every trial (default behavior)")
    ap.add_argument("--render-trials", type=str, default=None,
                    help="Comma-separated trial indices to render, e.g., '1,5,9'")
    ap.add_argument("--render-every", type=int, default=None,
                    help="Render every K trials (e.g., K=5)")
    ap.add_argument("--render-tile", type=int, default=22,
                    help="Tile size for rendered trials")
    ap.add_argument("--render-all-stages", action="store_true", default=False,
                    help="Render all stages of selected trials (not just stage 1)")
    args = ap.parse_args()

    random.seed(args.seed)

    study_dir = Path("runs") / "sweeps" / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    stages = _parse_stages(args)
    should_render = _parse_render_selector(args)

    # Optuna storage: prefer explicit DB via CLI or env, else local SQLite file for easy testing
    # Priority: --storage > DB_URL/OPTUNA_DB_URL > local sqlite under runs/sweeps/<study>/study.db
    resolved_db_url = (args.storage or DB_URL).strip()
    if not resolved_db_url:
        try:
            resolved_db_url = f"sqlite:///{(study_dir / 'study.db').absolute()}"
        except Exception:
            resolved_db_url = f"sqlite:///{study_dir}/study.db"
    storage = RDBStorage(url=resolved_db_url)

    # Use Bayesian optimization via TPE. Enable options that play well with parallel workers.
    # Prefer a parallel-friendly TPE; fall back gracefully for older Optuna versions
    try:
        sampler = TPESampler(
            seed=args.seed,
            multivariate=True,
            group=True,
            constant_liar=True,
        )
    except TypeError:
        try:
            sampler = TPESampler(seed=args.seed, constant_liar=True)
        except TypeError:
            sampler = TPESampler(seed=args.seed)
    pruner = SuccessiveHalvingPruner(
        min_resource=stages[0],
        reduction_factor=max(2, int(args.growth)),
        min_early_stopping_rate=0,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )

    csv_path = study_dir / "results.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "trial,stage,score,num_envs,n_steps,ppo_epochs,batch_size,learning_rate,clip_coef,ent_coef,vf_coef,gamma,gae_lambda,run_dir\n"
        )

    def objective(trial: "optuna.Trial") -> float:
        cfg = suggest_hparams(trial, args.profile)
        # Ensure at least one PPO update per stage
        min_steps = cfg["num_envs"] * cfg["n_steps"]

        trial_idx = trial.number + 1
        trial_root = study_dir / f"trial_{trial_idx:03d}"

        last_score = 0.0
        for si, requested_steps in enumerate(stages, start=1):
            eff_steps = max(int(requested_steps), int(min_steps))
            stage_dir = trial_root / f"stage_{si}"
            do_render = should_render(trial_idx) and (
                args.render_all_stages or si == 1)  # default: stage 1 only
            score = _run_stage(
                stage_dir=stage_dir,
                cfg=cfg,
                total_timesteps=eff_steps,
                backend=args.backend,
                base_seed=args.seed,
                trial_number=trial_idx,
                stage_index=si,
                render=do_render,
                render_tile=args.render_tile,
            )
            last_score = score

            # Log row if not already present (avoid duplicates on resume)
            existing = False
            try:
                if csv_path.exists():
                    needle = f"{trial_idx},{si},"
                    for line in csv_path.read_text().splitlines():
                        if line.startswith(needle):
                            existing = True
                            break
            except Exception:
                pass
            if not existing:
                with csv_path.open("a") as fp:
                    fp.write(
                        f"{trial_idx},{si},{score:.6f},{cfg['num_envs']},{cfg['n_steps']},{cfg['ppo_epochs']},{cfg['batch_size']},{cfg['learning_rate']:.6g},{cfg['clip_coef']:.4f},{cfg['ent_coef']:.5f},{cfg['vf_coef']:.4f},{cfg['gamma']:.5f},{cfg['gae_lambda']:.4f},{stage_dir}\n"
                    )

            # Report to Optuna for pruning
            trial.report(last_score, step=eff_steps)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return last_score

    # Optional: print progress and best so far after each trial
    def _callback(study: "optuna.Study", trial: "optuna.FrozenTrial"):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(
                f"[optuna] completed trial {trial.number} value={trial.value:.3f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"[optuna] pruned trial {trial.number}")
        best = study.best_trial if study.best_trial else None
        if best is not None:
            best_cfg = best.params
            best_out = {"score": best.value, **best_cfg}
            (study_dir / "best_config.json").write_text(json.dumps(best_out, indent=2))

    # Continue up to the requested total number of trials on resume
    try:
        from optuna.trial import TrialState  # type: ignore
        finished = [t for t in study.get_trials(deepcopy=False) if t.state in (
            TrialState.COMPLETE, TrialState.PRUNED)]
        already_done = len(finished)
    except Exception:
        already_done = 0
    remaining = max(0, int(args.trials) - already_done)
    if remaining == 0:
        print(
            f"[optuna] nothing to do: already have {already_done} finished trials (target={args.trials})")
    else:
        print(
            f"[optuna] resuming: {already_done} finished; running {remaining} more to reach {args.trials}")
        study.optimize(objective, n_trials=remaining, callbacks=[_callback])

    print("[sweep] best value:", study.best_value)
    print("[sweep] best params:", study.best_params)


if __name__ == "__main__":
    main()
