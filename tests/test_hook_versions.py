"""Version pins must agree between the git hooks and CI: a hook that validates something
different from CI is worse than no hook at all.

Two pairs are compared: remote pre-commit hooks against `uv.lock` (nothing today, the Ruff
hooks are local), and the commitlint packages between `.pre-commit-config.yaml` and the
Commitlint workflow.
"""

import re
import tomllib

from conftest import ROOT

# `- repo: <url>` then `rev: <tag>` or `rev: <sha> # frozen: <tag>`
REMOTE_REPO = re.compile(r"-\s*repo:\s*(\S+)\s+rev:\s*(\S+)(?:\s*#\s*frozen:\s*(\S+))?")
COMMITLINT_DEP = re.compile(r"@commitlint/[\w-]+@([\d.]+)")
COMMITLINT_ENV = re.compile(r"COMMITLINT_VERSION:\s*([\d.]+)")
HOOK_CONFIG = ROOT / ".pre-commit-config.yaml"
WORKFLOW = ROOT / ".github" / "workflows" / "commitlint.yml"


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
    config = HOOK_CONFIG.read_text()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert drift(config, lock) == []


def test_drift_is_reported_with_both_versions():
    config = (
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "    rev: e8f7d69b957a30acb2c875078d5fd012de24b64c # frozen: v0.16.6\n"
        "  - repo: https://github.com/alessandrojcm/commitlint-pre-commit-hook\n"
        "    rev: v9.26.0\n"
    )
    lock = {"package": [{"name": "ruff", "version": "0.16.7"}]}
    assert drift(config, lock) == ["ruff: pre-commit rev 0.16.6 vs uv.lock 0.16.7"]


def test_commitlint_version_matches_between_hook_and_ci():
    hook_versions = set(COMMITLINT_DEP.findall(HOOK_CONFIG.read_text()))
    ci_versions = set(COMMITLINT_ENV.findall(WORKFLOW.read_text()))
    assert hook_versions, "the commit-msg hook must pin @commitlint/config-conventional"
    assert ci_versions, "the workflow must pin COMMITLINT_VERSION"
    assert hook_versions == ci_versions, f"hook {sorted(hook_versions)} vs CI {sorted(ci_versions)}"


def test_commitlint_hook_rev_is_pinned():
    rev = dict(
        (url, frozen or tag) for url, tag, frozen in REMOTE_REPO.findall(HOOK_CONFIG.read_text())
    )
    hook = "https://github.com/alessandrojcm/commitlint-pre-commit-hook"
    assert re.fullmatch(r"v\d+\.\d+\.\d+", rev.get(hook, "")), f"unpinned commitlint hook: {rev}"
