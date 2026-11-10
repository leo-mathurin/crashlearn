"""Persistent run records and optional W&B logging. Local files remain authoritative."""

import datetime
import importlib.metadata
import json
import numbers
import os
import platform
import sys
import uuid
from pathlib import Path

from crashlearn.evaluation import provenance


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def prepare_run(args) -> tuple[dict, Path]:
    path = args.output_dir / "manifest.json"
    current = provenance()
    if args.resume:
        if not args.resume.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("checkpoint must belong to the original run directory")
        if not path.is_file():
            raise ValueError("--resume needs the original run directory and its manifest")
        manifest = json.loads(path.read_text())
        if manifest.get("derivation"):
            raise ValueError("inference derivations cannot resume; initialize a new training run")
        expected = {"history_depth": args.history_depth, "train_seed": args.seed}
        expected |= {
            k: current[k]
            for k in ("sim_sha256", "features_sha256", "history_sha256", "reward_sha256")
        }
        for key, value in expected.items():
            if manifest[key] != value:
                raise ValueError(f"resume would change {key}; start a separate run")
        if manifest["config"]["n_envs"] != args.n_envs:
            raise ValueError("resume would change n_envs; use the original value")
        if manifest["config"].get("fixed_map_workers", False) != args.fixed_map_workers:
            raise ValueError("resume would change fixed_map_workers; use the original value")
        if manifest["config"].get("max_speed", 10.0) != args.max_speed:
            raise ValueError("resume would change max_speed; start a new initialized run")
        if manifest["config"].get("random_start", False) != args.random_start:
            raise ValueError("resume would change random_start; start a new initialized run")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(
                f"{args.output_dir} already holds a run: failures are kept, not replaced"
            )
        manifest = {
            **current,
            "run_id": uuid.uuid4().hex,
            "driver": f"ppo-h{args.history_depth}",
            "train_seed": args.seed,
            "history_depth": args.history_depth,
            "attempts": [],
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "stable-baselines3", "gymnasium", "wandb")
            },
            "python": platform.python_version(),
            "host": platform.node(),
            "image_id": os.environ.get("TRAIN_IMAGE_ID"),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = {
        "started_utc": now(),
        "status": "running",
        "commit": current["commit"],
        "resumed_from": str(args.resume) if args.resume else None,
        "checkpoint_sha256": None,
        "budget_timesteps": args.timesteps,
        "device": args.device,
        "torch_threads": args.torch_threads,
    }
    manifest.setdefault("attempts", []).append(attempt)
    manifest |= {
        "status": "running",
        "budget_timesteps": args.timesteps,
        "resumed_from": attempt["resumed_from"],
    }
    return manifest, path


def attach_wandb(logger, args, manifest: dict):
    """A network/logging failure disables only this writer, never the local CSV."""
    from stable_baselines3.common.logger import KVWriter

    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=manifest["run_id"],
            name=args.output_dir.name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            resume="allow" if args.wandb_mode == "online" else None,
            dir=str(args.log_dir),
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
        manifest["attempts"][-1]["wandb_error"] = str(error)
        print(f"W&B disabled: {error}", file=sys.stderr)
        return None

    class WandbWriter(KVWriter):
        enabled = True

        def write(self, key_values, key_excluded, step=0):
            if not self.enabled:
                return
            try:
                numeric = {k: v for k, v in key_values.items() if isinstance(v, numbers.Number)}
                run.log(numeric, step=step)
            except Exception as error:
                self.enabled = False
                manifest["attempts"][-1]["wandb_error"] = str(error)
                print(f"W&B disabled: {error}", file=sys.stderr)

        def close(self):
            try:
                run.finish()
            except Exception as error:
                manifest["attempts"][-1]["wandb_error"] = str(error)
                print(f"W&B finish failed: {error}", file=sys.stderr)

        def checkpoint(self, path: Path):
            if not self.enabled:
                return
            try:
                artifact = wandb.Artifact(f"ppo-{manifest['run_id']}", type="checkpoint")
                artifact.add_file(str(path))
                run.log_artifact(artifact)
            except Exception as error:
                self.enabled = False
                manifest["attempts"][-1]["wandb_error"] = str(error)
                print(f"W&B disabled: {error}", file=sys.stderr)

    writer = WandbWriter()
    logger.output_formats.append(writer)
    manifest["attempts"][-1]["wandb"] = {
        "mode": args.wandb_mode,
        "id": run.id,
        "url": run.url if args.wandb_mode == "online" else None,
    }
    return writer
