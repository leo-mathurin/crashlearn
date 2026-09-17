"""Commit message rules, checked through scripts/lint_commit_msg.py (same entry as hook and CI).

Examples mirror the README section on commit messages.
"""

import subprocess
import sys

import pytest
from conftest import ROOT

SCRIPT = ROOT / "scripts" / "lint_commit_msg.py"
LONG_NAME = "Some Very Long Contributor Name " * 3

VALID = {
    "simple": "feat: add track inventory",
    "scope_breaking": "fix(tracks)!: seal the test split",
    "header_72": "docs: " + "x" * 66,
    "body_and_trailers": (
        "fix: clamp steering\n\nExplain why in wrapped lines.\n\n"
        f"Refs: E-9\nCo-Authored-By: {LONG_NAME} <someone.with.a.long.address@example.com>"
    ),
    "long_url": "docs: link report\n\nSee https://example.com/" + "a" * 120,
    "git_revert": 'Revert "feat: add track inventory"\n\nThis reverts commit 5760147.',
    "comments_stripped": "feat: add x\n# Please enter the commit message\n# " + "c" * 120,
}
INVALID = {
    "not_conventional": "Update readme",
    "capitalized_type": "Feat: add x",
    "no_space": "feat:add x",
    "header_73": "docs: " + "x" * 67,
    "long_body_line": "feat: add x\n\n" + "b" * 101,
    "fixup": "fixup! feat: add x",
    "squash": "squash! feat: add x",
    "amend": "amend! feat: add x\n\nfeat: add x",
}


def lint(message: str, *flags: str) -> int:
    cmd = [sys.executable, str(SCRIPT), *flags, "--message", message]
    return subprocess.run(cmd, capture_output=True).returncode


@pytest.mark.parametrize("message", VALID.values(), ids=VALID.keys())
def test_valid_messages(message):
    assert lint(message) == 0


@pytest.mark.parametrize("message", INVALID.values(), ids=INVALID.keys())
def test_invalid_messages(message):
    assert lint(message) == 1


@pytest.mark.parametrize("prefix", ["fixup!", "squash!", "amend!"])
def test_autosquash_commits_allowed_locally(prefix):
    assert lint(f"{prefix} feat: add x", "--allow-autosquash") == 0
