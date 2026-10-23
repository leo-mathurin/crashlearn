#!/usr/bin/env python3
"""verify_submission.py — Validates a submission before the tournament.

Uses the same agent_loader the tournament uses so "passes verify" means
"loads and runs in the tournament".

Modes:
    Default (no flags): type-conformance check on a single agent
    --grandprix:       verify tournament delivery structure (4 driver folders)

Usage:
    python verify_submission.py --submission submission/
    python verify_submission.py --submission submission/ --grandprix
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

# Import agent_loader from same directory (shared with tournament)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_loader import validate_and_load_agent

ALLOWED_MODULES = {"onnxruntime", "numpy", "os"}
MAX_OPSET = 17


# ---------------------------------------------------------------------------
# Import check — AST based, no code execution
# ---------------------------------------------------------------------------

def _extract_imports(source_path: Path) -> list[str]:
    """Return top-level import names from a Python source file."""
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def check_imports(agent_path: Path) -> list[str]:
    """Check that agent.py only imports from the allowed set.

    Returns a list of disallowed imports (empty if all OK).
    """
    imports = _extract_imports(agent_path)
    disallowed = [m for m in imports if m not in ALLOWED_MODULES]
    return disallowed


# ---------------------------------------------------------------------------
# Opset check — must be ≤ MAX_OPSET for pinned onnxruntime
# ---------------------------------------------------------------------------

def check_opset(model_path: Path) -> str | None:
    """Verify ONNX model opset ≤ MAX_OPSET.

    Returns None if OK, otherwise an error message.
    """
    try:
        import onnx
        from onnx import TensorProto

        model = onnx.load(str(model_path))
    except Exception as e:
        return f"Failed to load ONNX model: {e}"

    max_found = 0
    for opset_import in model.opset_import:
        # opset_import has fields: domain (str, default ''), version (int)
        if hasattr(opset_import, "version"):
            max_found = max(max_found, opset_import.version)

    if max_found > MAX_OPSET:
        return (
            f"ONNX opset {max_found} exceeds maximum allowed {MAX_OPSET}. "
            f"Re-export with opset ≤ {MAX_OPSET}."
        )
    return None


# ---------------------------------------------------------------------------
# External data check — .onnx.data must exist when model references it
# ---------------------------------------------------------------------------

def check_external_data(model_path: Path) -> str | None:
    """Verify that referenced external data files exist.

    Returns None if OK, otherwise an error message.
    """
    try:
        import onnx
    except Exception as e:
        return f"Cannot import onnx: {e}"

    model = onnx.load(str(model_path))

    for init in model.graph.initializer:
        if not init.external_data:
            continue
        # external_data is a list of {key, value} pairs
        # The filename is under the key "filename"
        for entry in init.external_data:
            if entry.key == "filename":
                ref_file = Path(entry.value)
                if not ref_file.exists():
                    return (
                        f"External data file '{ref_file}' referenced by "
                        f"model.onnx does not exist."
                    )
    return None


# ---------------------------------------------------------------------------
# Default mode — single agent validation
# ---------------------------------------------------------------------------

def run_single_agent_check(submission_path: Path) -> bool:
    """Run full validation on a single agent submission.

    Returns True if all checks pass, False otherwise.
    """
    print(f"\n=== Single Agent Check ===")
    print(f"  Path: {submission_path.resolve()}")
    all_ok = True

    # 1. File presence
    agent_py = submission_path / "agent.py"
    model_onnx = submission_path / "model.onnx"

    if not agent_py.is_file():
        print(f"  [FAIL] Missing required file: agent.py")
        return False
    if not model_onnx.is_file():
        print(f"  [FAIL] Missing required file: model.onnx")
        return False

    # 2. Import check
    disallowed = check_imports(agent_py)
    if disallowed:
        print(f"  [FAIL] Forbidden imports: {', '.join(disallowed)}")
        print(f"        Only 'onnxruntime', 'numpy', 'os' are allowed.")
        all_ok = False
    else:
        print(f"  [PASS] Imports restricted to allowed modules.")

    # 3. Opset check
    opset_err = check_opset(model_onnx)
    if opset_err:
        print(f"  [FAIL] {opset_err}")
        all_ok = False
    else:
        print(f"  [PASS] ONNX opset ≤ {MAX_OPSET}.")

    # 4. External data check
    ext_err = check_external_data(model_onnx)
    if ext_err:
        print(f"  [FAIL] {ext_err}")
        all_ok = False
    else:
        print(f"  [PASS] External data files present (if any).")

    # 5. Load via agent_loader (instantiation, ONNX load, predict contract)
    try:
        team_name, agent = validate_and_load_agent(
            submission_path, team_name=submission_path.name, unique_id=0
        )
        print(f"  [PASS] Agent loaded: {team_name}")
        print(f"  [PASS] predict() returns (float, float) scalars.")
    except Exception as e:
        print(f"  [FAIL] agent_loader error: {e}")
        all_ok = False

    if all_ok:
        print(f"\n  ✓ Submission VALID — ready for tournament.")
    else:
        print(f"\n  ✗ Submission INVALID — fix errors above.")

    return all_ok


# ---------------------------------------------------------------------------
# --grandprix mode — multi-driver folder structure validation
# ---------------------------------------------------------------------------

def run_grandprix_check(submission_path: Path) -> bool:
    """Validate tournament delivery structure under a submission directory.

    Checks:
      - Contains subdirectories (one per driver)
      - Each subdirectory has agent.py + model.onnx (+ .onnx.data if needed)
      - All folder names are unique
      - Each agent loads correctly via agent_loader

    Returns True if all checks pass, False otherwise.
    """
    print(f"\n=== Grand Prix Structure Check ===")
    print(f"  Path: {submission_path.resolve()}")
    all_ok = True

    if not submission_path.is_dir():
        print(f"  [FAIL] Submission path does not exist: {submission_path}")
        return False

    subdirs = sorted([
        d for d in submission_path.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    ])

    if not subdirs:
        print(f"  [FAIL] No valid team subdirectories found in {submission_path}")
        return False

    print(f"  Found {len(subdirs)} driver folder(s): {[d.name for d in subdirs]}")

    # Check unique names
    names = [d.name for d in subdirs]
    if len(names) != len(set(names)):
        print(f"  [FAIL] Duplicate folder names detected!")
        all_ok = False

    # Check each folder
    folder_results: dict[str, bool] = {}
    for folder in subdirs:
        folder_ok = True
        print(f"\n  ── {folder.name} ──")

        agent_py = folder / "agent.py"
        model_onnx = folder / "model.onnx"

        # File presence
        if not agent_py.is_file():
            print(f"    [FAIL] Missing: agent.py")
            folder_ok = False
        if not model_onnx.is_file():
            print(f"    [FAIL] Missing: model.onnx")
            folder_ok = False

        if not folder_ok:
            print(f"    [SKIP] Skipping remaining checks for this folder.")
            all_ok = False
            folder_results[folder.name] = False
            continue

        # Import check
        disallowed = check_imports(agent_py)
        if disallowed:
            print(f"    [FAIL] Forbidden imports: {', '.join(disallowed)}")
            folder_ok = False
        else:
            print(f"    [PASS] Imports OK.")

        # Opset check
        opset_err = check_opset(model_onnx)
        if opset_err:
            print(f"    [FAIL] {opset_err}")
            folder_ok = False
        else:
            print(f"    [PASS] ONNX opset ≤ {MAX_OPSET}.")

        # External data check
        ext_err = check_external_data(model_onnx)
        if ext_err:
            print(f"    [FAIL] {ext_err}")
            folder_ok = False
        else:
            print(f"    [PASS] External data present (if needed).")

        # agent_loader validation
        try:
            team_name, agent = validate_and_load_agent(
                folder, team_name=folder.name, unique_id=subdirs.index(folder)
            )
            print(f"    [PASS] Agent loaded: {team_name}")
        except Exception as e:
            print(f"    [FAIL] agent_loader error: {e}")
            folder_ok = False

        folder_results[folder.name] = folder_ok
        all_ok = all_ok and folder_ok

    # Check unique names
    if len(names) != len(set(names)):
        all_ok = False

    # Summary
    print(f"\n  ── Summary ──")
    for name, ok in folder_results.items():
        status = "✓" if ok else "✗"
        print(f"    {status} {name}")

    if len(names) != len(set(names)):
        print(f"  ✗ Duplicate folder names!")
        all_ok = False

    # Naming reminder
    print(f"\n  ⚠ REMINDER: Please name your folders driver1, driver2, driver3, driver4")
    print(f"    as specified in the project subject (CrashAndLearn-project.md).")

    if all_ok and len(subdirs) >= 2:
        print(f"\n  ✓ Grand Prix structure VALID — ready for tournament.")
    elif not all_ok:
        print(f"\n  ✗ Grand Prix structure INVALID — fix errors above.")
    else:
        print(f"\n  ℹ Only {len(subdirs)} driver(s) found — expected 4 for a full Grand Prix.")

    return all_ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crash&Learn Grand Prix — Verify submission"
    )
    parser.add_argument(
        "--submission",
        type=str,
        default="submission/",
        help="Path to submission directory (default: submission/)",
    )
    parser.add_argument(
        "--grandprix",
        action="store_true",
        help="Check tournament delivery structure (multi-driver folders)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    submission_path = Path(args.submission).resolve()

    if not submission_path.exists():
        print(f"[ERROR] Path does not exist: {submission_path}")
        sys.exit(1)

    if args.grandprix:
        success = run_grandprix_check(submission_path)
    else:
        success = run_single_agent_check(submission_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
