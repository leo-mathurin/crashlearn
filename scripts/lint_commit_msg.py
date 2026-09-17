"""Lint a commit message: commitlint (Conventional Commits, header length) plus the rules
commitlint (PyPI) lacks: body line length and unsquashed fixup commits.

Exit code 0 if valid, 1 otherwise.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADER_MAX = 72
BODY_LINE_MAX = 100
AUTOSQUASH_PREFIXES = ("fixup! ", "squash! ", "amend! ")
TRAILER = re.compile(r"^[A-Za-z][\w-]*: ")  # Co-Authored-By: ..., Refs: E-9
SCISSORS = "# ------------------------ >8 ------------------------"
COMMITLINT = Path(sys.executable).with_name("commitlint")


def clean(message: str) -> str:
    """Drop git comment lines and everything below the `git commit -v` scissors line."""
    message = message.split(SCISSORS)[0]
    return "\n".join(line for line in message.splitlines() if not line.startswith("#")).strip()


def body_errors(message: str) -> list[str]:
    return [
        f"body line {n} is {len(line)} > {BODY_LINE_MAX} characters"
        for n, line in enumerate(message.splitlines()[1:], start=2)
        if len(line) > BODY_LINE_MAX and not TRAILER.match(line) and "://" not in line
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="commit message file (commit-msg hook)")
    source.add_argument("--hash", help="lint an existing commit")
    source.add_argument("--message", help="lint this text (e.g. a PR title)")
    parser.add_argument(
        "--allow-autosquash",
        action="store_true",
        help="accept fixup!/squash!/amend! commits (local work, not CI)",
    )
    args = parser.parse_args()

    if args.file:
        message = args.file.read_text()
    elif args.hash:
        cmd = ["git", "log", "-1", "--format=%B", args.hash]
        message = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    else:
        message = args.message
    message = clean(message)
    header = message.split("\n", 1)[0]
    print(f"commit message: {header}", flush=True)  # before commitlint's own output in CI logs

    if header.startswith(AUTOSQUASH_PREFIXES):
        if args.allow_autosquash:
            return 0
        print(
            "error: unsquashed fixup/squash/amend commit, run `git rebase -i --autosquash`",
            file=sys.stderr,
        )
        return 1

    errors = body_errors(message)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    cmd = [str(COMMITLINT), "--max-header-length", str(HEADER_MAX), "--hide-input", message]
    commitlint_ok = subprocess.run(cmd).returncode == 0
    return 0 if commitlint_ok and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
