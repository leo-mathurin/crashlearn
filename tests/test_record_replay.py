"""Replays of the dashboard come from the unchanged recorder (scripts/record_replay.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from crashlearn import tracks

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("record_replay", ROOT / "scripts/record_replay.py")
record_replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_replay)


def test_team_names_keep_the_grid_order_and_fit_the_recorder():
    drivers = ["driver1", "driver2", "driver1", record_replay.CRASH_TEST]
    names = record_replay.team_names(drivers)
    assert names == ["1-driver1", "2-driver2", "3-driver1b", "4-crash"]
    # The recorder sorts the team folders and keeps ten characters of each name.
    assert names == sorted(names)
    assert all(len(name) <= 10 for name in names)
    assert record_replay.team_names(["driver2"]) == ["driver2"]


def test_sealed_test_tracks_are_refused(tmp_path):
    sealed = tracks.sealed_test_tracks(unseal=True)[0]
    with pytest.raises(SystemExit, match="sealed test track"):
        record_replay.record(sealed, ["driver1"], 0, 1, tmp_path / "replay.npz")
    assert not (tmp_path / "replay.npz").exists()


def test_demo_replays_use_no_sealed_track():
    sealed = set(tracks.sealed_test_tracks(unseal=True))
    demo = {tracks.canonical_name(map_name) for map_name, _, _ in record_replay.DEMO.values()}
    assert not demo & sealed


@pytest.mark.skipif(
    not (ROOT / "submission/driver1/agent.py").is_file(), reason="submission/driver1 is absent"
)
def test_records_a_race_with_a_delivered_driver_and_a_car_that_ends_in_dnf(tmp_path):
    out = tmp_path / "replay.npz"
    record_replay.record("Spielberg", ["driver1", record_replay.CRASH_TEST], 0, 1, out)
    with np.load(out) as replay:
        assert str(replay["map_name"]) == "Spielberg"
        assert list(replay["team_names"]) == ["1-driver1", "2-crash"]
        # 0 DNF, 1 ACTIVE, 2 FINISHED: the delivered driver finishes, the test car does not.
        assert list(replay["status"][-1]) == [2, 0]
        assert replay["poses"].shape[1:] == (2, 3)
