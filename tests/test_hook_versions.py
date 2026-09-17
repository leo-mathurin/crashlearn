"""A remote pre-commit hook must not run a different version of a tool locked in uv.lock.

Hooks are local (`uv run`) today, so nothing is compared; this fails if a remote hook
repo such as ruff-pre-commit comes back with a rev that drifts from the lock.
"""

import re
import tomllib

from conftest import ROOT

# `- repo: <url>` then `rev: <tag>` or `rev: <sha> # frozen: <tag>`
REMOTE_REPO = re.compile(r"-\s*repo:\s*(\S+)\s+rev:\s*(\S+)(?:\s*#\s*frozen:\s*(\S+))?")


def remote_hook_versions(config: str) -> dict[str, str]:
    """Tool name -> version for each remote repo (`astral-sh/ruff-pre-commit` -> `ruff`)."""
    return {
        url.rstrip("/").split("/")[-1].removesuffix("-pre-commit"): (frozen or rev).lstrip("v")
        for url, rev, frozen in REMOTE_REPO.findall(config)
    }


def drift(config: str, lock: dict) -> list[str]:
    locked = {p["name"]: p["version"] for p in lock["package"]}
    return [
        f"{tool}: pre-commit rev {version} vs uv.lock {locked[tool]}"
        for tool, version in remote_hook_versions(config).items()
        if tool in locked and locked[tool] != version
    ]


def test_hooks_match_uv_lock():
    config = (ROOT / ".pre-commit-config.yaml").read_text()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert drift(config, lock) == []


def test_drift_is_reported_with_both_versions():
    config = (
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "    rev: e8f7d69b957a30acb2c875078d5fd012de24b64c # frozen: v0.16.6\n"
        "  - repo: https://github.com/opensource-nepal/commitlint\n"
        "    rev: v2.0.0\n"
    )
    lock = {
        "package": [
            {"name": "ruff", "version": "0.16.7"},
            {"name": "commitlint", "version": "2.0.0"},
        ]
    }
    assert drift(config, lock) == ["ruff: pre-commit rev 0.16.6 vs uv.lock 0.16.7"]
