"""Train SAC on Hopper-v5, or continue a run that was stopped.

Usage:
    python train.py
    python train.py --set sac.learning_rate=1e-3 --tag lr1e-3
    python train.py --resume runs/<run_name>

Press Ctrl+C to stop: progress is saved and the run can be resumed later.
"""

import argparse
import sys
import time
from pathlib import Path

from hopper_sac.builders import make_env, make_eval_env, make_model
from hopper_sac.callbacks import RunCallback, trim_eval_log
from hopper_sac.checkpoint import (
    RunState,
    has_snapshot,
    load_snapshot,
    load_state,
    save_snapshot,
    save_state,
)
from hopper_sac.config import (
    DEFAULT_CONFIG,
    RUNS_DIR,
    RunPaths,
    load_config,
    new_run_paths,
    save_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on Hopper-v5.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue a stopped run")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value (can be repeated)",
    )
    parser.add_argument("--tag", help="short label appended to the run folder name")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where new runs are created")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resuming = args.resume is not None

    if resuming:
        paths = RunPaths(args.resume.resolve())
        if not paths.state_file.exists():
            sys.exit(f"Not a run folder (no state.json): {paths.root}")
        # The run's own config is used; --set can still adjust it,
        # e.g. run.total_timesteps to train for longer.
        cfg = load_config(paths.config_file, args.overrides)
        state = load_state(paths)
        if state.status == "completed" and state.num_timesteps >= cfg.run.total_timesteps:
            print(f"[run] already completed at {state.num_timesteps} steps: {paths.root}")
            return
        if not has_snapshot(paths):
            sys.exit(f"No snapshot to resume from in {paths.resume_dir}; start a new run instead.")
    else:
        cfg = load_config(args.config, args.overrides)
        paths = new_run_paths(cfg, args.tag, args.runs_dir)
        paths.create()
        state = RunState()
        save_state(paths, state)
    save_config(cfg, paths.config_file)
    print(f"[run] {paths.root}")

    env = make_env(cfg, monitor_file=paths.logs_dir / "train_episodes", append=resuming)
    eval_env = make_eval_env(cfg)

    if resuming:
        model = load_snapshot(cfg, env, paths)
        trim_eval_log(paths.evals_file, model.num_timesteps)
        print(f"[resume] continuing from step {model.num_timesteps}")
    else:
        model = make_model(cfg, env, paths)

    start_step = model.num_timesteps
    remaining = cfg.run.total_timesteps - start_step
    print(f"[run] device={model.device} steps {start_step} -> {cfg.run.total_timesteps}")

    start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=RunCallback(cfg, paths, eval_env, state),
            tb_log_name="sac",
            reset_num_timesteps=not resuming,
            progress_bar=True,
        )
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
    elapsed = time.perf_counter() - start

    save_snapshot(model, paths, state, status)
    done = model.num_timesteps - start_step
    print(f"[train] {done} steps in {elapsed:.1f}s ({done / elapsed:.0f} FPS)")

    if status == "completed":
        final_path = paths.models_dir / "final_model.zip"
        model.save(final_path)
        print(f"[save] {final_path}")
        print(f"[done] best eval mean reward: {state.best_mean_reward:.1f}")
    else:
        print(f"[stop] interrupted at step {model.num_timesteps}; progress saved.")
        print(f"[stop] continue with: python train.py --resume {paths.root}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
