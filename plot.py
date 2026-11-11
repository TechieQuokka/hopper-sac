"""Draw the learning curve of one run, or compare several runs in one chart.

Usage:
    python plot.py runs/<run_name>                       # -> runs/<run_name>/plots/learning_curve.png
    python plot.py runs/<a> runs/<b> --out compare.png   # several runs overlaid
"""

import argparse
import sys
from pathlib import Path

from hopper_sac.config import RunPaths, load_config
from hopper_sac.media import plot_learning_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves from eval logs.")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="output PNG (required when comparing several runs)")
    parser.add_argument("--title")
    args = parser.parse_args()

    runs = {}
    for run_dir in args.run_dirs:
        paths = RunPaths(run_dir.resolve())
        if not paths.evals_file.exists():
            sys.exit(f"No eval log found: {paths.evals_file}")
        runs[paths.root.name] = paths.evals_file

    first = RunPaths(args.run_dirs[0].resolve())
    if len(runs) == 1:
        out = args.out or first.plots_dir / "learning_curve.png"
    elif args.out:
        out = args.out
    else:
        sys.exit("--out is required when comparing several runs")

    env_id = load_config(first.config_file).run.env_id
    title = args.title or f"SAC on {env_id}: eval return during training"
    plot_learning_curves(runs, out, title)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
