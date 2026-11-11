"""Quick end-to-end check before the real project.

Runs a very short SAC training on Hopper-v5, then verifies that the model can
be saved, loaded, and used to record a video. The agent will NOT learn to hop
in this few steps; the goal is only to confirm that the pipeline works and to
measure training speed on the chosen device.

Usage:
    python tests/quick_check.py --device cuda
    python tests/quick_check.py --device cpu
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

ENV_ID = "Hopper-v5"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class UpdatePhaseTimer(BaseCallback):
    """Record when gradient updates start, to measure the real training speed.

    Before `learning_starts` the agent only collects random data (very fast),
    so the overall FPS would overestimate the speed of a long run.
    """

    def __init__(self, learning_starts: int):
        super().__init__()
        self.learning_starts = learning_starts
        self.update_start_time = None
        self.update_start_step = None

    def _on_step(self) -> bool:
        if self.update_start_time is None and self.num_timesteps >= self.learning_starts:
            self.update_start_time = time.perf_counter()
            self.update_start_step = self.num_timesteps
        return True


def train(device: str, timesteps: int, learning_starts: int, seed: int) -> Path:
    env = gym.make(ENV_ID)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=learning_starts,
        device=device,
        seed=seed,
        verbose=0,
    )

    timer = UpdatePhaseTimer(learning_starts)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=timer, progress_bar=True)
    end = time.perf_counter()

    total_fps = timesteps / (end - start)
    update_steps = timesteps - timer.update_start_step
    update_fps = update_steps / (end - timer.update_start_time)
    print(f"[train] device={model.device} total={end - start:.1f}s")
    print(f"[train] overall FPS: {total_fps:.0f}")
    print(f"[train] update-phase FPS: {update_fps:.0f} (over {update_steps} steps)")
    print(f"[train] estimated time for 300k steps: {300_000 / update_fps / 60:.0f} min")

    model_path = OUTPUT_DIR / f"quick_sac_{device}.zip"
    model.save(model_path)
    env.close()
    print(f"[save] {model_path}")
    return model_path


def record(model_path: Path, device: str, episodes: int, seed: int) -> Path:
    model = SAC.load(model_path, device=device)
    print(f"[load] {model_path}")

    env = gym.make(ENV_ID, render_mode="rgb_array")
    frames = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            episode_return += float(reward)
            steps += 1
            done = terminated or truncated
        print(f"[eval] episode {episode}: return={episode_return:.1f} steps={steps}")

    fps = env.metadata.get("render_fps", 30)
    env.close()

    video_path = OUTPUT_DIR / f"quick_sac_{device}.mp4"
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"[video] {video_path} ({len(frames)} frames, {frames[0].shape}, {fps} fps)")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = train(args.device, args.timesteps, args.learning_starts, args.seed)
    record(model_path, args.device, args.episodes, args.seed)
    print("[done] quick check passed")


if __name__ == "__main__":
    main()
