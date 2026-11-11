"""Check that a stopped training run can be resumed without gaps.

Two scenarios, each run through the real `train.py` command line:
    1. Ctrl+C (SIGINT): progress is saved at the moment of the interrupt.
    2. Hard kill (SIGKILL): the run restarts from the latest snapshot.

After resuming, the run must reach the target step count with a continuous
evaluation log and a full replay buffer. The script also reports the size and
save time of one snapshot.

Usage:
    python tests/check_resume.py
"""

import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC  # noqa: E402

from hopper_sac.callbacks import read_eval_rows  # noqa: E402
from hopper_sac.checkpoint import BUFFER_FILE, MODEL_FILE, RunState, save_snapshot  # noqa: E402
from hopper_sac.config import RunPaths  # noqa: E402

WORK_DIR = PROJECT_ROOT / "tests" / "output" / "resume_check"
TOTAL_STEPS = 6_000
EVAL_FREQ = 500
SNAPSHOT_FREQ = 1_000
CHECKPOINT_FREQ = 2_000
OVERRIDES = [
    f"run.total_timesteps={TOTAL_STEPS}",
    "sac.learning_starts=1000",
    f"eval.freq={EVAL_FREQ}",
    "eval.episodes=2",
    f"save.snapshot_freq={SNAPSHOT_FREQ}",
    f"save.checkpoint_freq={CHECKPOINT_FREQ}",
]


def train_command(extra: list[str]) -> list[str]:
    return [sys.executable, "train.py", *extra]


def start_new_run(tag: str, log_file: Path) -> tuple[subprocess.Popen, RunPaths]:
    runs_dir = WORK_DIR / tag
    args = ["--runs-dir", str(runs_dir), "--tag", tag]
    for item in OVERRIDES:
        args += ["--set", item]
    process = subprocess.Popen(
        train_command(args), cwd=PROJECT_ROOT, stdout=open(log_file, "w"), stderr=subprocess.STDOUT
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        found = list(runs_dir.glob("*/state.json"))
        if found:
            return process, RunPaths(found[0].parent)
        time.sleep(0.2)
    process.kill()
    raise RuntimeError("run folder was not created in time")


def read_state(paths: RunPaths) -> dict:
    return json.loads(paths.state_file.read_text())


def eval_steps(paths: RunPaths) -> list[int]:
    return [int(row["timesteps"]) for row in read_eval_rows(paths.evals_file)]


def wait_until(condition, process: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("training finished before it could be stopped")
        if condition():
            return
        time.sleep(0.05)
    process.kill()
    raise RuntimeError("timed out waiting for training progress")


def resume(paths: RunPaths, log_file: Path) -> None:
    subprocess.run(
        train_command(["--resume", str(paths.root)]),
        cwd=PROJECT_ROOT,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        check=True,
    )


def check(condition: bool, message: str) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        raise SystemExit(f"check failed: {message}")


def verify_completed(paths: RunPaths, exact_buffer: bool) -> None:
    state = read_state(paths)
    check(state["status"] == "completed", f"status is completed (got {state['status']})")
    check(state["num_timesteps"] == TOTAL_STEPS, f"reached {TOTAL_STEPS} steps (got {state['num_timesteps']})")

    expected = list(range(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ))
    check(eval_steps(paths) == expected, f"eval log is continuous, no gaps or duplicates ({len(expected)} rows)")

    checkpoints = sorted(p.name for p in paths.checkpoints_dir.glob("step_*.zip"))
    expected_ckpt = [f"step_{s:06d}.zip" for s in range(CHECKPOINT_FREQ, TOTAL_STEPS + 1, CHECKPOINT_FREQ)]
    check(checkpoints == expected_ckpt, f"checkpoints present: {checkpoints}")
    check((paths.models_dir / "best_model.zip").exists(), "best_model.zip exists")
    check((paths.models_dir / "final_model.zip").exists(), "final_model.zip exists")

    tb_runs = [p.name for p in paths.tensorboard_dir.iterdir()]
    check(tb_runs == ["sac_1"], f"TensorBoard log continues in one folder: {tb_runs}")

    model = SAC.load(paths.resume_dir / MODEL_FILE, device="cpu")
    model.load_replay_buffer(paths.resume_dir / BUFFER_FILE)
    check(model.num_timesteps == TOTAL_STEPS, "saved model step count matches")
    size = model.replay_buffer.size()
    if exact_buffer:
        check(size == TOTAL_STEPS, f"replay buffer holds all {TOTAL_STEPS} transitions (got {size})")
    else:
        # Ctrl+C can land between an env step and storing it, losing one transition.
        check(size in (TOTAL_STEPS - 1, TOTAL_STEPS), f"replay buffer holds {size} of {TOTAL_STEPS} transitions")


def scenario_sigint() -> RunPaths:
    print("\n=== Scenario 1: Ctrl+C (SIGINT), then resume ===")
    process, paths = start_new_run("sigint", WORK_DIR / "sigint_first.log")
    wait_until(lambda: read_state(paths)["num_timesteps"] >= 3_000, process)
    time.sleep(1.0)  # stop between snapshots, not exactly on one
    process.send_signal(signal.SIGINT)
    process.wait(timeout=120)

    state = read_state(paths)
    stopped_at = state["num_timesteps"]
    check(state["status"] == "interrupted", f"status is interrupted (got {state['status']})")
    check(3_000 < stopped_at < TOTAL_STEPS, f"stopped mid-run at step {stopped_at}")
    check(stopped_at % SNAPSHOT_FREQ != 0, "progress saved at the interrupt, not only at the last snapshot")

    resume(paths, WORK_DIR / "sigint_resume.log")
    verify_completed(paths, exact_buffer=False)
    return paths


def scenario_kill() -> None:
    print("\n=== Scenario 2: hard kill (SIGKILL), then resume ===")
    process, paths = start_new_run("kill", WORK_DIR / "kill_first.log")

    # Kill once the eval log is ahead of the last snapshot, so that resuming
    # has to drop the rows it is about to produce again.
    def log_ahead_of_snapshot() -> bool:
        steps = eval_steps(paths)
        snapshot_step = read_state(paths)["num_timesteps"]
        return snapshot_step >= 2_000 and bool(steps) and steps[-1] > snapshot_step

    wait_until(log_ahead_of_snapshot, process)
    process.kill()
    process.wait(timeout=60)

    state = read_state(paths)
    snapshot_step = state["num_timesteps"]
    check(state["status"] == "running", "state still says running (no chance to save)")
    check(eval_steps(paths)[-1] > snapshot_step, f"eval log is ahead of snapshot step {snapshot_step}")

    resume(paths, WORK_DIR / "kill_resume.log")
    verify_completed(paths, exact_buffer=True)


def measure_snapshot(paths: RunPaths) -> None:
    print("\n=== Snapshot cost ===")
    model = SAC.load(paths.resume_dir / MODEL_FILE, device="cuda")
    model.load_replay_buffer(paths.resume_dir / BUFFER_FILE)
    target = RunPaths(WORK_DIR / "snapshot_timing")
    target.create()

    times = []
    for _ in range(3):
        start = time.perf_counter()
        save_snapshot(model, target, RunState())
        times.append(time.perf_counter() - start)

    model_mb = (target.resume_dir / MODEL_FILE).stat().st_size / 1e6
    buffer_mb = (target.resume_dir / BUFFER_FILE).stat().st_size / 1e6
    print(f"  model.zip: {model_mb:.1f} MB | replay_buffer.pkl: {buffer_mb:.1f} MB (buffer_size={model.buffer_size})")
    print(f"  save time: {', '.join(f'{t:.2f}s' for t in times)}")


def main() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    paths = scenario_sigint()
    scenario_kill()
    measure_snapshot(paths)
    print("\n[done] resume check passed")


if __name__ == "__main__":
    main()
