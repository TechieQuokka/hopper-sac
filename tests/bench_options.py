"""Compare training speed of optimization candidates.

Each candidate trains for a few thousand steps in its own process (thread
settings are process-wide) and reports the FPS of the update phase, i.e. the
part where every env step is followed by updates. `progress` turns on the
SB3 progress bar, to measure what the bar itself costs.

Usage:
    python tests/bench_options.py                 # all candidates, 2 repeats
    python tests/bench_options.py --only gpu gpu-threads1 --repeats 3
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TOTAL_STEPS = 6_000
LEARNING_STARTS = 1_000
# Timing starts a little after updates begin, so one-off warm-up costs
# (CUDA start-up, torch.compile) are not counted.
TIMER_START = 2_000

# name -> options. `batch` groups N env steps and then does N updates at once
# (same number of updates in total).
CANDIDATES: dict[str, dict] = {
    "gpu": {"device": "cuda"},
    "gpu-threads1": {"device": "cuda", "threads": 1},
    "gpu-threads2": {"device": "cuda", "threads": 2},
    "cpu-threads1": {"device": "cpu", "threads": 1},
    "cpu-threads2": {"device": "cpu", "threads": 2},
    "cpu-threads4": {"device": "cpu", "threads": 4},
    "gpu-batch8": {"device": "cuda", "batch": 8},
    "gpu-batch64": {"device": "cuda", "batch": 64},
    "gpu-compile": {"device": "cuda", "compile": True},
    "gpu-progress": {"device": "cuda", "progress": True},
}


def run_worker(options: dict) -> None:
    """Train once with the given options and print the result as JSON."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from hopper_sac.builders import make_env, make_model
    from hopper_sac.config import RunPaths, load_config

    if "threads" in options:
        torch.set_num_threads(options["threads"])

    batch = options.get("batch", 1)
    cfg = load_config(
        overrides=[
            f"run.device={options['device']}",
            f"run.total_timesteps={TOTAL_STEPS}",
            f"sac.learning_starts={LEARNING_STARTS}",
            f"sac.train_freq={batch}",
            f"sac.gradient_steps={batch}",
        ]
    )
    env = make_env(cfg)
    model = make_model(cfg, env, RunPaths(PROJECT_ROOT / "tests" / "output" / "bench_options"))
    model.tensorboard_log = None

    if options.get("compile"):
        # Only forward() of a compiled module is optimized, so compile the
        # inner networks that SAC calls through forward().
        actor, critic, critic_target = model.actor, model.critic, model.critic_target
        actor.latent_pi = torch.compile(actor.latent_pi)
        for net in (critic, critic_target):
            for i, q_net in enumerate(net.q_networks):
                net.q_networks[i] = torch.compile(q_net)

    class Timer(BaseCallback):
        start_time = None
        start_step = None

        def _on_step(self) -> bool:
            if self.start_time is None and self.num_timesteps >= TIMER_START:
                self.start_time = time.perf_counter()
                self.start_step = self.num_timesteps
            return True

    timer = Timer()
    model.learn(total_timesteps=TOTAL_STEPS, callback=timer, progress_bar=options.get("progress", False))
    if options["device"] == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - timer.start_time
    fps = (model.num_timesteps - timer.start_step) / elapsed
    print(json.dumps({"fps": fps, "threads": torch.get_num_threads()}))


def run_candidate(name: str) -> dict:
    result = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(CANDIDATES[name])],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return {"error": last_line}
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--only", nargs="+", choices=list(CANDIDATES), help="run only these candidates")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.worker:
        run_worker(json.loads(args.worker))
        return

    names = args.only or list(CANDIDATES)
    results: dict[str, list[dict]] = {name: [] for name in names}
    # Interleave repeats so slow drift of the machine affects all candidates equally.
    for repeat in range(args.repeats):
        for name in names:
            outcome = run_candidate(name)
            results[name].append(outcome)
            shown = outcome.get("error") or f"{outcome['fps']:.0f} FPS"
            print(f"[{repeat + 1}/{args.repeats}] {name:14s} {shown}", flush=True)

    baseline = None
    print(f"\n{'candidate':14s} {'threads':>7s} {'FPS (mean)':>11s} {'runs':>16s} {'vs first':>9s} {'300k steps':>11s}")
    for name in names:
        ok = [r for r in results[name] if "fps" in r]
        if not ok:
            print(f"{name:14s} failed: {results[name][0]['error']}")
            continue
        fps = statistics.mean(r["fps"] for r in ok)
        baseline = baseline or fps
        runs = ", ".join(f"{r['fps']:.0f}" for r in ok)
        print(
            f"{name:14s} {ok[0]['threads']:7d} {fps:11.0f} {runs:>16s} "
            f"{fps / baseline:8.2f}x {300_000 / fps / 60:8.0f} min"
        )


if __name__ == "__main__":
    main()
