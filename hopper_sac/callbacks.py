"""Periodic work during training: evaluation, checkpoints, resumable snapshots."""

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from hopper_sac.checkpoint import RunState, save_snapshot
from hopper_sac.config import Config, RunPaths

EVAL_FIELDS = ["timesteps", "mean_reward", "std_reward", "min_reward", "max_reward", "mean_length"]


def append_eval_row(path: Path, row: dict) -> None:
    """Append one evaluation result; the file keeps growing across resumes."""
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_eval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def trim_eval_log(path: Path, max_timesteps: int) -> None:
    """Drop evaluations recorded after the snapshot we are resuming from.

    After a hard kill the log can be ahead of the last snapshot; those steps
    will be trained again, so their rows would otherwise appear twice.
    """
    rows = read_eval_rows(path)
    kept = [row for row in rows if int(row["timesteps"]) <= max_timesteps]
    if len(kept) == len(rows):
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    tmp.replace(path)


class RunCallback(BaseCallback):
    """Evaluate, checkpoint, and snapshot at the intervals set in the config."""

    def __init__(self, cfg: Config, paths: RunPaths, eval_env: gym.Env, state: RunState):
        super().__init__()
        self.cfg = cfg
        self.paths = paths
        self.eval_env = eval_env
        self.state = state

        self._last_index: dict[str, int] = {}

    def _on_training_start(self) -> None:
        # Count intervals from the current step, so a resumed run does not
        # repeat work that was already done before it stopped.
        for name, freq in self._intervals().items():
            self._last_index[name] = self.num_timesteps // freq

    def _intervals(self) -> dict[str, int]:
        return {
            "eval": self.cfg.eval.freq,
            "checkpoint": self.cfg.save.checkpoint_freq,
            "snapshot": self.cfg.save.snapshot_freq,
        }

    def _is_due(self, name: str) -> bool:
        index = self.num_timesteps // self._intervals()[name]
        if index > self._last_index[name]:
            self._last_index[name] = index
            return True
        return False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Runs after the new transition is stored in the replay buffer.
        # (`_on_step` runs before that, so a snapshot taken there would miss
        # the latest transition.)
        step = self.num_timesteps
        # Evaluate before the snapshot so the saved state has the latest best score.
        if self._is_due("eval"):
            self._evaluate(step)
        if self._is_due("checkpoint"):
            self.model.save(self.paths.checkpoints_dir / f"step_{step:06d}.zip")
        if self._is_due("snapshot"):
            save_snapshot(self.model, self.paths, self.state)

    def _evaluate(self, step: int) -> None:
        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.cfg.eval.episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean_reward = float(np.mean(rewards))
        append_eval_row(
            self.paths.evals_file,
            {
                "timesteps": step,
                "mean_reward": round(mean_reward, 3),
                "std_reward": round(float(np.std(rewards)), 3),
                "min_reward": round(float(np.min(rewards)), 3),
                "max_reward": round(float(np.max(rewards)), 3),
                "mean_length": round(float(np.mean(lengths)), 1),
            },
        )
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", float(np.mean(lengths)))

        best = self.state.best_mean_reward
        if best is None or mean_reward > best:
            self.state.best_mean_reward = mean_reward
            self.model.save(self.paths.models_dir / "best_model.zip")
