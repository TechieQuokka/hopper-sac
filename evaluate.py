"""Score a saved model, and optionally record videos.

Usage:
    python evaluate.py runs/<run_name>                      # best model, 10 episodes
    python evaluate.py runs/<run_name> --model final
    python evaluate.py runs/<run_name> --model step_050000
    python evaluate.py runs/<run_name> --video              # also save videos/<model>.mp4
    python evaluate.py runs/<run_name> --video-stages       # one video per checkpoint + best
"""

import argparse
import os
import sys
from pathlib import Path

# Render off-screen on the GPU. Must be set before MuJoCo is imported.
# (The default windowed backend logs OpenGL warnings on this machine.)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402

from hopper_sac.builders import EVAL_SEED_OFFSET, make_env  # noqa: E402
from hopper_sac.config import Config, RunPaths, load_config  # noqa: E402
from hopper_sac.media import record_video  # noqa: E402


def model_file(paths: RunPaths, name: str) -> Path:
    """Resolve `best`, `final`, or a checkpoint name such as `step_050000`."""
    if name in ("best", "final"):
        return paths.models_dir / f"{name}_model.zip"
    return paths.checkpoints_dir / f"{name}.zip"


def score(cfg: Config, model: SAC, episodes: int) -> None:
    env = make_env(cfg)
    env.reset(seed=cfg.run.seed + EVAL_SEED_OFFSET)
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=episodes, deterministic=True, return_episode_rewards=True
    )
    env.close()
    print(
        f"[eval] {episodes} episodes: return {np.mean(rewards):.1f} +/- {np.std(rewards):.1f} "
        f"(min {np.min(rewards):.1f}, max {np.max(rewards):.1f}), mean length {np.mean(lengths):.0f}"
    )


def video(cfg: Config, model: SAC, path: Path, episodes: int) -> None:
    env = make_env(cfg, render_mode="rgb_array")
    returns = record_video(model, env, path, episodes=episodes, seed=cfg.run.seed + EVAL_SEED_OFFSET)
    env.close()
    shown = ", ".join(f"{r:.0f}" for r in returns)
    print(f"[video] {path} (episode returns: {shown})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a saved model, optionally record videos.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="best", help="best, final, or a checkpoint name like step_050000")
    parser.add_argument("--episodes", type=int, default=10, help="episodes used for scoring")
    parser.add_argument("--video", action="store_true", help="record the chosen model")
    parser.add_argument("--video-stages", action="store_true", help="record every checkpoint and the best model")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu is enough for running a trained policy")
    args = parser.parse_args()

    paths = RunPaths(args.run_dir.resolve())
    if not paths.config_file.exists():
        sys.exit(f"Not a run folder (no config.toml): {paths.root}")
    cfg = load_config(paths.config_file)

    target = model_file(paths, args.model)
    if not target.exists():
        sys.exit(f"Model not found: {target}")
    model = SAC.load(target, device=args.device)
    print(f"[load] {target}")
    score(cfg, model, args.episodes)

    if args.video:
        video(cfg, model, paths.videos_dir / f"{args.model}.mp4", args.video_episodes)

    if args.video_stages:
        stages = sorted(paths.checkpoints_dir.glob("step_*.zip")) + [model_file(paths, "best")]
        for stage in stages:
            if not stage.exists():
                continue
            name = "best" if stage.parent == paths.models_dir else stage.stem
            if args.video and name == args.model:
                continue  # already recorded above
            video(cfg, SAC.load(stage, device=args.device), paths.videos_dir / f"{name}.mp4", args.video_episodes)


if __name__ == "__main__":
    main()
