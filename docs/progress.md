# Progress and handoff

Last updated: 2026-11-11 11:40

This document records where the work stands so it can be continued later.
For usage and layout, see `README.md`.

## Status in one paragraph

The main goal is done: SAC on `Hopper-v5` trains in about 24 minutes, the robot hops for the
full 1000-step episode (eval return about 3226), and the videos and learning curve exist.
Training can be stopped and resumed. An optional tuning experiment (three full-length runs)
was paused partway, resumed, and finished on 2026-11-11 at 11:13: `gradient_steps=2` learns
in about half the steps but ends at about the same return (3203-3364, one seed).
The default config is unchanged. Nothing is running now.

## What is done

| Stage | Result |
|---|---|
| 1. Config, model, basic training | Done. `train.py`, `configs/sac_hopper.toml`, `hopper_sac/config.py`, `hopper_sac/builders.py` |
| 2. Snapshots and resume | Done. `hopper_sac/checkpoint.py`, `hopper_sac/callbacks.py`; verified by `tests/check_resume.py` (Ctrl+C and hard kill) |
| 3. Speed optimization | Done. No candidate gave a reliable gain; settings unchanged |
| 4. Baseline run, videos, curve | Done. 300k steps, eval return 3226 |
| 5a. Tuning: first sweep (100k steps x 6) | Done. `gradient_steps=2` (`utd2`) learned fastest |
| 5b. Tuning: full 300k runs of top 3 | Done. Similar final return; `gradient_steps=2` learns faster in steps |

## Results so far

All from a single seed (0). Eval returns swing by hundreds between evaluations,
so small differences between variants may be chance.

### Baseline (`runs/sac_hopper_seed0_20261111-073501_baseline`)

- 300k steps in 1435 s (209 FPS) on an RTX 3060.
- Best model, 10 eval episodes: return 3228 +/- 3, all episodes 1000 steps long.
- Below the Gymnasium reward threshold of 3800.
- Return stayed under 1000 until about 190k steps, then rose quickly.
  It collapsed once (452 at 280k) and recovered at the next evaluation.
- Outputs: `videos/best.mp4`, `videos/step_050000.mp4` ... `step_300000.mp4`, `plots/learning_curve.png`.

### First sweep (`runs/first_sweep`, 100k steps each)

| Variant | Best eval | Last eval | Mean of last 3 |
|---|---|---|---|
| baseline | 784 | 408 | 388 |
| lr1e-3 | 427 | 403 | 351 |
| starts5k | 750 | 546 | 582 |
| utd2 | 1215 | 1215 | 980 |
| gamma0.98 | 356 | 347 | 341 |
| net400-300 | 733 | 288 | 294 |

The sweep's baseline reproduced the first 100k steps of the 300k baseline run exactly
(same seed gives the same eval values).

### Full candidates (`runs/full_candidates`, 300k steps each)

| Variant | Change vs baseline | Best eval | Last eval | Mean of last 3 | First eval above 3000 |
|---|---|---|---|---|---|
| baseline | - | 3226 | 3226 | 2248 | 260k |
| `starts5k` | `sac.learning_starts=5000` | 3203 | 3203 | 3114 | 290k |
| `utd2` | `sac.gradient_steps=2` | 3317 | 3304 | 2570 | 150k |
| `utd2-starts5k` | both | 3364 | 3364 | 3330 | 210k |

- Final returns are within about 160 of each other; with one seed this does not show
  that any variant ends higher.
- Both `gradient_steps=2` runs passed 2500 by 110-130k steps; the baseline and `starts5k`
  needed 260k.
- Late collapses, each recovered by the next evaluation: baseline 452 at 280k,
  `utd2` 1115 at 290k, `utd2-starts5k` 2104 at 230k. `starts5k` had none after 260k.
- `utd2-starts5k` best model, 10 eval episodes: return 3360 +/- 7, all episodes 1000 steps long.
  Videos (`best.mp4` and one per 50k checkpoint) are in its `videos/` folder.
- Figures: `runs/full_candidates/compare_with_baseline.png` (with baseline),
  `runs/full_candidates/compare.png` (the three variants only).
- The runs were stopped at 217k / 130k / 126k steps and resumed together with
  `python sweep.py configs/sweeps/full_candidates.toml --jobs 3`. All three continued from
  their saved step, so resuming several interrupted runs through a sweep is now exercised.
- Wall-clock cost was not measured cleanly: the runs shared the GPU three at a time
  (`starts5k` 88-99 FPS, the two `utd2` runs 57-62 FPS, vs 209 FPS for the baseline alone).

### Speed

- 93-96% of one training step is the gradient update; the MuJoCo step is about 0.1 ms.
- GPU vs CPU for one SAC step: about 1.3x. A 300k run alone on GPU: about 24 min.
- CPU thread count, grouped updates, `torch.compile`: no reliable gain. Run-to-run noise
  on this machine is about +/-20% while the desktop is in use.
- Runs in parallel share the GPU: three at a time ran at about 55-100 FPS each instead of 209.

## Decisions made

- Use Stable-Baselines3 SAC rather than a from-scratch implementation; train on GPU.
- `tests/` holds quick trial scripts run before the real project (not pytest).
- Tuning by manual sweeps first. Optuna only later, and installing it needs approval.
- `gradient_steps = 2` is not the default in `configs/sac_hopper.toml` for now: it costs about
  1.5x the training time per step and its gain is seen on one seed only.
- `moviepy` was not installed because it would downgrade `pillow` (12.2.0 -> 11.3.0).
  Videos are written with `imageio` + `imageio-ffmpeg` instead.
- Videos use Hopper's `track` camera (`run.render_camera`). The default camera stays still,
  so the robot hops out of view.
- `evaluate.py` renders with `MUJOCO_GL=egl`. The default windowed backend logs
  `OpenGL error 0x502` and writes `MUJOCO_LOG.TXT` on this machine.

## Known limitations

- A resumed run is not bit-identical to an uninterrupted one (random state is not restored).
- A hard kill loses the progress since the last snapshot (at most 10k steps).
- Ctrl+C can lose a single transition if it lands right after an environment step.
- Runs created before the camera fix have no `render_camera` in their `config.toml`;
  the default (`track`) applies, so their videos are fine too.
- The videos were checked by sampling frames, not by watching them end to end.

## Possible next steps

1. Confirm `gradient_steps=2` with 2-3 seeds, e.g. `--set run.seed=1`, before adopting it.
   If it holds, fewer steps (about 200k) may be enough, which would offset its extra cost.
2. Time a `gradient_steps=2` run alone to get its real wall-clock cost.
3. Try longer training (e.g. 500k steps) to see whether the return passes 3800.
4. Later study topics from the original plan: compare with PPO/TD3, other environments
   (Walker2d, HalfCheetah), a from-scratch SAC, JAX-based training (Brax/MJX, SBX).

## Housekeeping

- Disk use: `runs/` is about 567 MB (each run keeps a 60 MB replay buffer in `resume/`).
  Finished runs no longer need `resume/`; it can be deleted to save space.
- `tests/output/` still holds four small files from the first quick check (6.3 MB).
- Installed during this work in the `rl` env: `tqdm`, `rich`, `imageio-ffmpeg`
  (plus `pygments`, `markdown-it-py`, `mdurl`). No existing package version changed.
- Published as a public GitHub repository on 2026-11-11. `runs/` and `tests/output/` are
  excluded by `.gitignore`; the figures shown in `README.md` are in `assets/`
  (frames and the GIF come from the `utd2-starts5k` videos).
