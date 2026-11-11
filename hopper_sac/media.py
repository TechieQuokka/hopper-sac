"""Outputs for people: mp4 videos of the agent and learning-curve figures."""

from pathlib import Path

import gymnasium as gym
import imageio
import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from hopper_sac.callbacks import read_eval_rows  # noqa: E402

# Chart colors: fixed categorical order, neutral inks for all text.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def record_video(
    model: SAC,
    env: gym.Env,
    path: Path,
    episodes: int = 1,
    seed: int = 0,
) -> list[float]:
    """Run the agent in an `rgb_array` environment and save the frames as mp4.

    Returns the return of each recorded episode.
    """
    frames, returns = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            episode_return += float(reward)
            done = terminated or truncated
        returns.append(episode_return)

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=env.metadata.get("render_fps", 30))
    return returns


def plot_learning_curves(runs: dict[str, Path], path: Path, title: str) -> None:
    """Draw eval mean return (with a +/- 1 std band) against env steps.

    `runs` maps a label to that run's `evals.csv`. Up to 8 runs are supported,
    one fixed color each.
    """
    if len(runs) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} runs can share one chart, got {len(runs)}")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for (label, evals_file), color in zip(runs.items(), SERIES_COLORS):
        rows = read_eval_rows(evals_file)
        steps = [int(row["timesteps"]) for row in rows]
        mean = [float(row["mean_reward"]) for row in rows]
        std = [float(row["std_reward"]) for row in rows]
        low = [m - s for m, s in zip(mean, std)]
        high = [m + s for m, s in zip(mean, std)]

        ax.fill_between(steps, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label)
        if len(runs) <= 4 and steps:
            # Direct label at the line end: the value in ink, not in the series color.
            ax.annotate(
                f"{mean[-1]:,.0f}",
                (steps[-1], mean[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK_SECONDARY,
            )

    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel("Environment steps", color=INK_SECONDARY)
    ax.set_ylabel("Eval return (mean of episodes)", color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:,.0f}k")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.margins(x=0.02)
    ax.set_xlim(left=0)
    if ax.get_ylim()[0] >= 0:
        ax.set_ylim(bottom=0)

    if len(runs) > 1:
        legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
