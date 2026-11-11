"""Measure CPU vs GPU speed for the work SAC actually does on Hopper-v5.

Part 1 times each piece of one SAC training step on both devices:
    - environment step (MuJoCo, always CPU)
    - action selection (actor forward pass, batch of 1)
    - replay buffer sampling (includes copying the batch to the device)
    - gradient update (actor + twin critics, batch of 256)

Part 2 times a plain MLP training step while growing the network width and
the batch size, to show where the GPU starts to pay off.

Usage:
    python tests/bench_device.py
"""

import time

import gymnasium as gym
import torch
from stable_baselines3 import SAC

ENV_ID = "Hopper-v5"
BATCH_SIZE = 256


def sync(device: str) -> None:
    """Wait for queued GPU work so that wall-clock timing is accurate."""
    if device == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, iters: int, device: str, warmup: int = 50) -> float:
    """Return the average time of `fn` in milliseconds."""
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / iters * 1000


def bench_env_step(iters: int = 5_000) -> float:
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    env.action_space.seed(0)

    def step() -> None:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

    result = time_ms(step, iters, "cpu")
    env.close()
    return result


def bench_sac(device: str) -> dict[str, float]:
    env = gym.make(ENV_ID)
    # A huge learning_starts makes learn() only collect random transitions,
    # which fills the replay buffer without doing any gradient update.
    model = SAC("MlpPolicy", env, learning_starts=10**9, device=device, seed=0, verbose=0)
    model.learn(total_timesteps=3_000)
    obs, _ = env.reset(seed=0)

    results = {
        "action selection (batch 1)": time_ms(
            lambda: model.predict(obs, deterministic=False), 2_000, device
        ),
        "buffer sample (batch 256)": time_ms(
            lambda: model.replay_buffer.sample(BATCH_SIZE), 2_000, device
        ),
        # train() samples a batch itself, so this includes the sampling time.
        "gradient update (batch 256)": time_ms(
            lambda: model.train(gradient_steps=1, batch_size=BATCH_SIZE), 2_000, device
        ),
    }
    env.close()
    return results


def bench_mlp(device: str, width: int, batch: int, budget_s: float = 2.0) -> float:
    """Time one forward + backward + optimizer step of a 2-hidden-layer MLP."""
    net = torch.nn.Sequential(
        torch.nn.Linear(14, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    x = torch.randn(batch, 14, device=device)
    y = torch.randn(batch, 1, device=device)

    def step() -> None:
        loss = torch.nn.functional.mse_loss(net(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(5):
        step()
    sync(device)
    iters, start = 0, time.perf_counter()
    while iters < 10 or (time.perf_counter() - start < budget_s and iters < 2_000):
        step()
        iters += 1
    sync(device)
    return (time.perf_counter() - start) / iters * 1000


def main() -> None:
    print(f"torch {torch.__version__} | CPU threads: {torch.get_num_threads()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("=== Part 1: one SAC training step on Hopper-v5 (ms, lower is faster) ===")
    env_ms = bench_env_step()
    cpu, gpu = bench_sac("cpu"), bench_sac("cuda")
    print(f"{'task':32s} {'CPU':>8s} {'GPU':>8s} {'GPU speedup':>12s}")
    print(f"{'env step (MuJoCo, CPU only)':32s} {env_ms:8.3f} {env_ms:8.3f} {'-':>12s}")
    for name in cpu:
        print(f"{name:32s} {cpu[name]:8.3f} {gpu[name]:8.3f} {cpu[name] / gpu[name]:11.2f}x")

    update = "gradient update (batch 256)"
    action = "action selection (batch 1)"
    cpu_total = env_ms + cpu[action] + cpu[update]
    gpu_total = env_ms + gpu[action] + gpu[update]
    print(f"{'sum per step':32s} {cpu_total:8.3f} {gpu_total:8.3f} {cpu_total / gpu_total:11.2f}x")
    print(f"{'-> implied FPS':32s} {1000 / cpu_total:8.0f} {1000 / gpu_total:8.0f}")

    print("\n=== Part 2: plain MLP training step, growing size (ms) ===")
    print(f"{'width':>6s} {'batch':>6s} {'CPU':>10s} {'GPU':>10s} {'GPU speedup':>12s}")
    for width, batch in [(256, 256), (256, 4096), (1024, 256), (1024, 4096), (2048, 4096)]:
        c, g = bench_mlp("cpu", width, batch), bench_mlp("cuda", width, batch)
        print(f"{width:6d} {batch:6d} {c:10.3f} {g:10.3f} {c / g:11.2f}x")


if __name__ == "__main__":
    main()
