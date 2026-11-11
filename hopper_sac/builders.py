"""Build environments and the SAC model from a Config."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from hopper_sac.config import Config, RunPaths

EVAL_SEED_OFFSET = 10_000


def make_env(
    cfg: Config,
    monitor_file: Path | None = None,
    append: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    """Create the environment wrapped in a Monitor that tracks episode returns.

    If `monitor_file` is given, every finished episode is also written to
    `<monitor_file>.monitor.csv`. Use `append=True` when resuming a run so the
    existing file is kept.
    """
    kwargs = {}
    if render_mode and cfg.run.render_camera:
        kwargs["camera_name"] = cfg.run.render_camera
    env = gym.make(cfg.run.env_id, render_mode=render_mode, **kwargs)
    return Monitor(
        env,
        filename=str(monitor_file) if monitor_file else None,
        override_existing=not append,
    )


def make_eval_env(cfg: Config) -> gym.Env:
    """Create a separate environment for evaluation, with its own seed."""
    env = make_env(cfg)
    env.reset(seed=cfg.run.seed + EVAL_SEED_OFFSET)
    return env


def make_model(cfg: Config, env: gym.Env, paths: RunPaths) -> SAC:
    """Create a fresh SAC model with the hyperparameters from the config."""
    sac = cfg.sac
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=sac.learning_rate,
        buffer_size=sac.buffer_size,
        learning_starts=sac.learning_starts,
        batch_size=sac.batch_size,
        tau=sac.tau,
        gamma=sac.gamma,
        train_freq=sac.train_freq,
        gradient_steps=sac.gradient_steps,
        ent_coef=sac.ent_coef,
        policy_kwargs={"net_arch": list(sac.net_arch)},
        tensorboard_log=str(paths.tensorboard_dir),
        seed=cfg.run.seed,
        device=cfg.run.device,
        verbose=0,
    )
