"""Training run records: provenance, manifests, attempts and optional W&B logging.

Local files remain authoritative. A run folder holds `manifest.json`, `model.zip`, the
periodic checkpoints and one log folder per attempt; failures are kept, never replaced.
"""

import datetime
import hashlib
import importlib.metadata
import json
import numbers
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from crashlearn import SIM_DIR
from crashlearn.reward import RewardConfig

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_KEYS = ("features_sha256", "history_sha256")
# What a resumed attempt must share with the run it continues.
FROZEN_ON_RESUME = (
    "history_depth",
    "train_seed",
    "sim_sha256",
    *PREPROCESSING_KEYS,
    "reward_sha256",
)
# Frozen training settings, with their value in manifests written before the key existed.
FROZEN_CONFIG = {
    "n_envs": None,
    "fixed_map_workers": False,
    "max_speed": 10.0,
    "random_start": False,
}


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _commit() -> str:
    # the training image has no git: the station passes the commit through the environment
    return os.environ.get("GIT_COMMIT") or _git("rev-parse", "HEAD")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance() -> dict:
    """The code and simulator a result depends on. Preprocessing hashes gate every reload."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    package = Path(__file__).parent
    return {
        "commit": _commit(),
        "dirty": None if status == "unknown" else bool(status),
        "sim_sha256": sha256(SIM_DIR / "env_simulation.py"),
        "features_sha256": sha256(package / "features.py"),
        "history_sha256": sha256(package / "history.py"),
        "reward_sha256": sha256(package / "reward.py"),
        "reward_version": RewardConfig().version,
        "created_utc": now(),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def require_empty(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: earlier results are kept, not replaced")


def load_completed_run(run_dir: Path) -> dict:
    """The manifest of a completed run whose model and preprocessing match this code.

    Raises ValueError otherwise: an old model fed with changed features drives blind.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"{run_dir}: run status is {manifest['status']!r}")
    model = run_dir / "model.zip"
    if sha256(model) != manifest["model_sha256"]:
        raise ValueError(f"{model} does not match its manifest")
    current = provenance()
    for key in PREPROCESSING_KEYS:
        if manifest[key] != current[key]:
            raise ValueError(f"{run_dir}: training has a different {key}; use its original sources")
    return manifest


def save_model(model, path: Path) -> str:
    """Save a Stable-Baselines3 model and return its SHA-256."""
    allow_sb3_saves()
    model.save(path)
    return sha256(path)


def allow_sb3_saves() -> None:
    """The simulator ships an empty `gym` namespace, whose version SB3 reads when saving."""
    import gym

    if not hasattr(gym, "__version__"):
        gym.__version__ = "vendored-namespace"


class Run:
    """One training run folder and its manifest.

    `Run.open` starts a run or resumes one; `with run.attempt(...)` records how each
    attempt ends (completed, failed, interrupted) and always writes the manifest.
    """

    def __init__(self, directory: Path, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        self.model = None
        self._started = 0.0

    @classmethod
    def open(
        cls, directory: Path, *, history_depth: int, seed: int, config: dict, resume: Path | None
    ) -> "Run":
        """Raises ValueError, before writing anything, if the folder already holds another
        run or if a resume would change what the run trains."""
        current = provenance()
        if resume is None:
            require_empty(directory)
            manifest = {
                **current,
                "run_id": uuid.uuid4().hex,
                "driver": f"ppo-h{history_depth}",
                "train_seed": seed,
                "history_depth": history_depth,
                "config": config,
                "attempts": [],
                "versions": {
                    name: _version(name)
                    for name in ("numpy", "torch", "stable-baselines3", "gymnasium", "wandb")
                },
                "python": platform.python_version(),
                "host": platform.node(),
                "image_id": os.environ.get("TRAIN_IMAGE_ID"),
            }
        else:
            if not resume.resolve().is_relative_to(directory.resolve()):
                raise ValueError("checkpoint must belong to the original run directory")
            path = directory / "manifest.json"
            if not path.is_file():
                raise ValueError("--resume needs the original run directory and its manifest")
            manifest = json.loads(path.read_text())
            if manifest.get("derivation"):
                raise ValueError(
                    "inference derivations cannot resume; initialize a new training run"
                )
            expected = current | {"history_depth": history_depth, "train_seed": seed}
            frozen = [(k, manifest[k], expected[k]) for k in FROZEN_ON_RESUME] + [
                (k, manifest["config"].get(k, default), config[k])
                for k, default in FROZEN_CONFIG.items()
            ]
            for key, recorded, requested in frozen:
                if recorded != requested:
                    raise ValueError(f"resume would change {key}; start a separate run")
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory, manifest)

    @property
    def number(self) -> int:
        """Number of the current attempt, from 1."""
        return len(self.manifest["attempts"])

    @property
    def log_dir(self) -> Path:
        return self.directory / "logs" / f"attempt-{self.number:03d}"

    def save(self) -> None:
        write_manifest(self.directory / "manifest.json", self.manifest)

    @contextmanager
    def attempt(self, *, budget_timesteps: int, resumed_from: Path | None, **details):
        """Record one attempt. The block registers its cleanups on the yielded ExitStack.

        An exception marks the attempt failed (interrupted for KeyboardInterrupt). A cleanup
        error also fails it, even after `complete`. The manifest is written in every case.
        """
        record = {
            "started_utc": now(),
            "status": "running",
            "commit": _commit(),
            "resumed_from": str(resumed_from) if resumed_from else None,
            "checkpoint_sha256": sha256(resumed_from) if resumed_from else None,
            "budget_timesteps": budget_timesteps,
            **details,
        }
        self.manifest["attempts"].append(record)
        self.manifest |= {
            "status": "running",
            "budget_timesteps": budget_timesteps,
            "resumed_from": record["resumed_from"],
        }
        record["log_dir"] = str(self.log_dir.relative_to(self.directory))
        self.save()
        self._started = time.perf_counter()
        cleanups = ExitStack()
        try:
            yield cleanups
        except (Exception, KeyboardInterrupt) as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record |= {"status": status, "error": traceback.format_exc()}
            record["wall_time_s"] = self._elapsed()
            self.manifest |= {"status": status, "error": record["error"]}
            if self.model is not None:
                record["last_step"] = self.model.num_timesteps
            raise
        finally:
            try:
                cleanups.close()
            except Exception:
                record["cleanup_error"] = traceback.format_exc()
                self.manifest["status"] = record["status"] = "failed"
                self.manifest.setdefault("error", record["cleanup_error"])
                raise
            finally:
                record.setdefault("wall_time_s", self._elapsed())
                record["finished_utc"] = now()
                self.manifest["total_wall_time_s"] = round(
                    sum(a.get("wall_time_s", 0) for a in self.manifest["attempts"]), 2
                )
                self.save()

    def attach(self, model) -> None:
        """Record the model this attempt trains, so that a failure reports its last step."""
        import torch

        self.model = model
        self.manifest["attempts"][-1] |= {
            "first_step": model.num_timesteps,
            "device": str(model.device),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(model.device)
            if model.device.type == "cuda"
            else None,
        }
        self.save()

    def complete(self) -> Path:
        """Save the final model and mark the attempt completed. Returns the model path."""
        path = self.directory / "model.zip"
        record = self.manifest["attempts"][-1]
        self.manifest.pop("error", None)
        self.manifest["model_sha256"] = save_model(self.model, path)
        elapsed = self._elapsed()
        steps = self.model.num_timesteps
        self.manifest |= {
            "status": "completed",
            "timesteps_done": steps,
            "wall_time_s": elapsed,
            "steps_per_s": round((steps - record["first_step"]) / elapsed, 1),
        }
        record |= {"status": "completed", "last_step": steps, "wall_time_s": elapsed}
        return path

    def _elapsed(self) -> float:
        return round(time.perf_counter() - self._started, 2)


def attach_wandb(logger, run: Run, *, project, entity, group, mode):
    """A network/logging failure disables only this writer, never the local CSV."""
    from stable_baselines3.common.logger import KVWriter

    manifest = run.manifest
    attempt = manifest["attempts"][-1]

    def disable(error, action="disabled"):
        attempt["wandb_error"] = str(error)
        print(f"W&B {action}: {error}", file=sys.stderr)

    try:
        import wandb

        remote = wandb.init(
            project=project,
            entity=entity,
            id=manifest["run_id"],
            name=run.directory.name,
            group=group,
            mode=mode,
            resume="allow" if mode == "online" else None,
            dir=str(run.log_dir),
            config={
                **manifest["config"],
                "seed": manifest["train_seed"],
                "history_depth": manifest["history_depth"],
                "commit": manifest["commit"],
                "versions": manifest["versions"],
            },
            settings=wandb.Settings(init_timeout=20),
        )
    except Exception as error:
        disable(error)
        return None

    class WandbWriter(KVWriter):
        enabled = True

        def write(self, key_values, key_excluded, step=0):
            if not self.enabled:
                return
            try:
                numeric = {k: v for k, v in key_values.items() if isinstance(v, numbers.Number)}
                remote.log(numeric, step=step)
            except Exception as error:
                self.enabled = False
                disable(error)

        def close(self):
            try:
                remote.finish()
            except Exception as error:
                disable(error, "finish failed")

        def checkpoint(self, path: Path):
            if not self.enabled:
                return
            try:
                artifact = wandb.Artifact(f"ppo-{manifest['run_id']}", type="checkpoint")
                artifact.add_file(str(path))
                remote.log_artifact(artifact)
            except Exception as error:
                self.enabled = False
                disable(error)

    writer = WandbWriter()
    logger.output_formats.append(writer)
    attempt["wandb"] = {
        "mode": mode,
        "id": remote.id,
        "url": remote.url if mode == "online" else None,
    }
    return writer
