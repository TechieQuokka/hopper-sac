"""Resumable snapshots: model + replay buffer + progress state.

A snapshot lives in `<run>/resume/` and is overwritten each time. Every file
is first written under a temporary name and then renamed, so a crash in the
middle of saving leaves the previous snapshot intact.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

from hopper_sac.config import Config, RunPaths

MODEL_FILE = "model.zip"
BUFFER_FILE = "replay_buffer.pkl"


@dataclass
class RunState:
    """Progress of a run, stored as `<run>/state.json`."""

    num_timesteps: int = 0
    best_mean_reward: float | None = None
    status: str = "running"  # running | interrupted | completed
    updated_at: str = ""


def _fsync(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _commit(tmp: Path, final: Path) -> None:
    """Flush a finished temporary file to disk, then move it into place."""
    _fsync(tmp)
    os.replace(tmp, final)


def save_state(paths: RunPaths, state: RunState) -> None:
    state.updated_at = datetime.now().isoformat(timespec="seconds")
    tmp = paths.state_file.with_name(paths.state_file.name + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2) + "\n")
    _commit(tmp, paths.state_file)


def load_state(paths: RunPaths) -> RunState:
    return RunState(**json.loads(paths.state_file.read_text()))


def has_snapshot(paths: RunPaths) -> bool:
    return (paths.resume_dir / MODEL_FILE).exists() and (paths.resume_dir / BUFFER_FILE).exists()


def save_snapshot(model: SAC, paths: RunPaths, state: RunState, status: str = "running") -> None:
    """Save everything needed to continue training later.

    Order matters: the replay buffer goes first, the model second, the state
    last. If saving is cut short, the model on disk is never newer than its
    buffer, and the model file (which stores its own step count) stays the
    source of truth for where to resume from.
    """
    resume = paths.resume_dir
    resume.mkdir(parents=True, exist_ok=True)

    tmp_buffer = resume / "replay_buffer.tmp.pkl"
    model.save_replay_buffer(tmp_buffer)
    _commit(tmp_buffer, resume / BUFFER_FILE)

    tmp_model = resume / "model.tmp.zip"
    model.save(tmp_model)
    _commit(tmp_model, resume / MODEL_FILE)

    state.num_timesteps = model.num_timesteps
    state.status = status
    save_state(paths, state)


def load_snapshot(cfg: Config, env: gym.Env, paths: RunPaths) -> SAC:
    """Load the model and replay buffer of the latest snapshot."""
    model = SAC.load(
        paths.resume_dir / MODEL_FILE,
        env=env,
        device=cfg.run.device,
        tensorboard_log=str(paths.tensorboard_dir),
    )
    model.load_replay_buffer(paths.resume_dir / BUFFER_FILE)
    # Loading re-applies the original seed. Re-seed with the step count so the
    # resumed part does not replay the same random sequence as the beginning.
    model.set_random_seed(cfg.run.seed + model.num_timesteps)
    return model
