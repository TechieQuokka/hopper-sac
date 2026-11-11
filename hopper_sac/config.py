"""Experiment settings (loaded from TOML) and the folder layout of a run."""

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sac_hopper.toml"


@dataclass
class RunConfig:
    name: str = "sac_hopper"
    env_id: str = "Hopper-v5"
    seed: int = 0
    device: str = "cuda"
    total_timesteps: int = 300_000
    # Camera used for videos. Hopper's "track" camera follows the robot;
    # the default free camera stays still, so the robot hops out of view.
    # Use "" for the environment's default camera.
    render_camera: str = "track"


@dataclass
class SACConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 300_000
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto"
    net_arch: list[int] = field(default_factory=lambda: [256, 256])


@dataclass
class EvalConfig:
    freq: int = 10_000
    episodes: int = 5


@dataclass
class SaveConfig:
    snapshot_freq: int = 10_000
    checkpoint_freq: int = 50_000


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    save: SaveConfig = field(default_factory=SaveConfig)


SECTIONS = {f.name: f.default_factory for f in fields(Config)}


def _section_keys(section: str) -> set[str]:
    return {f.name for f in fields(SECTIONS[section])}


def _parse_value(raw: str) -> Any:
    """Parse an override value with TOML rules; fall back to a plain string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_override(data: dict[str, dict[str, Any]], item: str) -> None:
    """Apply one `section.key=value` (or unambiguous `key=value`) override."""
    if "=" not in item:
        raise ValueError(f"Override must look like key=value, got: {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()

    if "." in key:
        section, name = key.split(".", 1)
    else:
        matches = [s for s in SECTIONS if key in _section_keys(s)]
        if len(matches) != 1:
            raise ValueError(
                f"Override key {key!r} matches sections {matches}; use section.key=value"
            )
        section, name = matches[0], key

    if section not in SECTIONS or name not in _section_keys(section):
        raise ValueError(f"Unknown setting: {section}.{name}")
    data.setdefault(section, {})[name] = _parse_value(raw.strip())


def load_config(path: Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> Config:
    """Read a TOML file, apply command-line overrides, and validate the keys."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for item in overrides or []:
        _apply_override(data, item)

    unknown_sections = set(data) - set(SECTIONS)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    sections = {}
    for section, cls in SECTIONS.items():
        values = data.get(section, {})
        unknown = set(values) - _section_keys(section)
        if unknown:
            raise ValueError(f"Unknown settings in [{section}]: {sorted(unknown)}")
        sections[section] = cls(**values)
    return Config(**sections)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(value)


def save_config(cfg: Config, path: Path) -> None:
    """Write the resolved settings as TOML, so a run records what it used."""
    lines = []
    for section, values in asdict(cfg).items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
        lines.append("")
    path.write_text("\n".join(lines))


@dataclass(frozen=True)
class RunPaths:
    """Locations of everything one run produces."""

    root: Path

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def resume_dir(self) -> Path:
        return self.root / "resume"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def tensorboard_dir(self) -> Path:
        return self.logs_dir / "tensorboard"

    @property
    def evals_file(self) -> Path:
        return self.logs_dir / "evals.csv"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def create(self) -> None:
        for directory in (
            self.resume_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.videos_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def new_run_paths(cfg: Config, tag: str | None = None, runs_dir: Path = RUNS_DIR) -> RunPaths:
    """Build a unique run folder name: <name>_seed<seed>_<timestamp>[_<tag>]."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{cfg.run.name}_seed{cfg.run.seed}_{stamp}"
    if tag:
        name += f"_{tag}"
    return RunPaths(Path(runs_dir).resolve() / name)
