"""Train several config variants and compare them.

A sweep file lists variants (a tag plus config overrides). Each variant becomes
one run under `runs/<sweep name>/`. Running the same command again skips
finished variants and resumes stopped ones, so a sweep can be interrupted
with Ctrl+C and continued later.

Usage:
    python sweep.py configs/sweeps/first_sweep.toml
    python sweep.py configs/sweeps/first_sweep.toml --jobs 3   # 3 runs at a time
"""

import argparse
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hopper_sac.callbacks import read_eval_rows
from hopper_sac.checkpoint import has_snapshot, load_state
from hopper_sac.config import PROJECT_ROOT, RUNS_DIR, RunPaths
from hopper_sac.media import plot_learning_curves


def find_run(sweep_dir: Path, tag: str) -> RunPaths | None:
    """Return the most recent run folder of this variant, if any."""
    matches = sorted(p for p in sweep_dir.glob(f"*_{tag}") if (p / "state.json").exists())
    return RunPaths(matches[-1]) if matches else None


def run_variant(sweep_dir: Path, base: Path, common: list[str], variant: dict) -> str:
    """Start, resume, or skip one variant. Returns what happened."""
    tag = variant["tag"]
    existing = find_run(sweep_dir, tag)

    if existing and load_state(existing).status == "completed":
        return "skipped (already completed)"

    if existing and has_snapshot(existing):
        command = [sys.executable, "train.py", "--resume", str(existing.root)]
        action = "resumed"
    else:
        command = [sys.executable, "train.py", "--config", str(base), "--runs-dir", str(sweep_dir), "--tag", tag]
        for item in [*common, *variant.get("set", [])]:
            command += ["--set", item]
        action = "trained"

    log_file = sweep_dir / f"{tag}.log"
    with open(log_file, "a") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return f"FAILED (see {log_file})"

    run = find_run(sweep_dir, tag)
    return action if run and load_state(run).status == "completed" else "stopped before finishing"


def summarize(sweep_dir: Path, variants: list[dict], name: str) -> None:
    curves = {}
    print(f"\n{'variant':14s} {'steps':>8s} {'best eval':>10s} {'last eval':>10s} {'mean last 3':>12s}")
    for variant in variants:
        run = find_run(sweep_dir, variant["tag"])
        rows = read_eval_rows(run.evals_file) if run else []
        if not rows:
            print(f"{variant['tag']:14s} (no results)")
            continue
        rewards = [float(row["mean_reward"]) for row in rows]
        last3 = sum(rewards[-3:]) / len(rewards[-3:])
        print(f"{variant['tag']:14s} {rows[-1]['timesteps']:>8s} {max(rewards):10.0f} {rewards[-1]:10.0f} {last3:12.0f}")
        curves[variant["tag"]] = run.evals_file

    if curves:
        out = sweep_dir / "compare.png"
        plot_learning_curves(curves, out, f"Sweep {name}: eval return during training")
        print(f"\n[plot] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train several config variants and compare them.")
    parser.add_argument("sweep_file", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="number of runs to train at the same time")
    parser.add_argument("--summary-only", action="store_true", help="only print the table and redraw the chart")
    args = parser.parse_args()

    with open(args.sweep_file, "rb") as f:
        spec = tomllib.load(f)
    sweep, variants = spec["sweep"], spec["variants"]
    tags = [v["tag"] for v in variants]
    if len(set(tags)) != len(tags):
        sys.exit("Variant tags must be unique")
    if any(a != b and a.endswith(f"_{b}") for a in tags for b in tags):
        sys.exit("A variant tag must not end with '_<another tag>'")

    sweep_dir = RUNS_DIR / sweep["name"]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    base = (PROJECT_ROOT / sweep["base"]).resolve()
    common = sweep.get("common", [])

    if not args.summary_only:
        print(f"[sweep] {sweep['name']}: {len(variants)} variants, {args.jobs} at a time -> {sweep_dir}")
        try:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {v["tag"]: pool.submit(run_variant, sweep_dir, base, common, v) for v in variants}
                for tag, future in futures.items():
                    print(f"[sweep] {tag}: {future.result()}", flush=True)
        except KeyboardInterrupt:
            # Ctrl+C also reaches the training processes, which save their progress.
            print("\n[sweep] interrupted; run the same command again to continue.")
            return

    summarize(sweep_dir, variants, sweep["name"])


if __name__ == "__main__":
    main()
