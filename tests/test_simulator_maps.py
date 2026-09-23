"""`set_map` behaviour on the provided circuits (E-10).

The track list, the Mexico City duplicate and the per-track assets are covered by
tests/test_tracks.py through `crashlearn.tracks`, the single source of track names.
"""

import env_simulation as es
import pytest

from crashlearn import tracks

TRACKS = tracks.final_retraining_tracks()  # all 23 circuits; driven straight only, no policy


@pytest.fixture(autouse=True)
def _fresh_sim():
    """Restore the example map after each test: `set_map` state is a module global."""
    yield
    es.close()
    if es.get_current_map() not in (None, "example"):
        es.set_map("example")


def test_simulator_lists_24_names_for_23_distinct_circuits():
    names = es.get_available_maps()
    assert len(names) == 24
    assert set(names) == set(TRACKS) | {"Mexico City"}


def test_set_map_is_idempotent_and_reported():
    es.set_map("Spa")
    assert es.get_current_map() == "Spa"
    sim_before = es._get_sim()._sim
    es.set_map("Spa")
    assert es._get_sim()._sim is sim_before, "no-op when the map is already loaded"


def test_set_map_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown map"):
        es.set_map("Monaco")


@pytest.mark.slow
@pytest.mark.parametrize("name", TRACKS)
def test_four_cars_start_clean_on_each_map(name):
    """Measured: across all 23 circuits, 4 cars start with no wall contact (20 steps at 2 m/s)."""
    es.set_map(name)
    es.reset(4)
    sim = es._get_sim()
    assert sim._total_arc > 250.0, "centerline loaded (measured lengths range 260-554 m)"
    walls = 0
    for _ in range(20):
        for cid in range(4):
            es.apply_action(cid, 2.0, 0.0)
        es.simulation_step()
        info = es.get_step_info()
        walls += sum(info["collisions"]["wall"].values())
    assert walls == 0
    assert all(info["agent_status"][c] == 1 for c in range(4))
    for cid in range(4):
        scan = es.get_obs(cid)["lidar"]
        assert es._LIDAR_MIN <= scan.min() and scan.max() <= es._LIDAR_MAX
