"""Inventory of the provided circuits and a `set_map` sweep (E-10, feeds into E-9)."""

import filecmp
import os

import env_simulation as es
import pytest
from conftest import MAPS_DIR

CANONICAL_MAPS = [
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "MexicoCity",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "YasMarina",
    "Zandvoort",
]
MAPS_WITHOUT_RACELINE = {"Montreal", "Shanghai"}


@pytest.fixture(autouse=True)
def _fresh_sim():
    """Restore the example map after each test: `set_map` state is a module global."""
    yield
    es.close()
    if es.get_current_map() not in (None, "example"):
        es.set_map("example")


def test_inventory_has_24_names_for_23_distinct_circuits():
    names = es.get_available_maps()
    assert len(names) == 24
    assert set(names) == set(CANONICAL_MAPS) | {"Mexico City"}


def test_mexico_city_folder_is_a_misnamed_copy_of_mexicocity():
    """'Mexico City' files are prefixed MexicoCity_ and identical to the canonical folder."""
    dup = os.path.join(MAPS_DIR, "Mexico City")
    ref = os.path.join(MAPS_DIR, "MexicoCity")
    for fname in (
        "MexicoCity_centerline.csv",
        "MexicoCity_map.png",
        "MexicoCity_map.yaml",
        "MexicoCity_raceline.csv",
    ):
        assert filecmp.cmp(os.path.join(dup, fname), os.path.join(ref, fname), shallow=False)
    assert not os.path.exists(os.path.join(dup, "Mexico City_map.png"))


@pytest.mark.parametrize("name", CANONICAL_MAPS)
def test_each_canonical_map_has_expected_files(name):
    d = os.path.join(MAPS_DIR, name)
    for suffix in ("_centerline.csv", "_map.png", "_map.yaml"):
        assert os.path.isfile(os.path.join(d, name + suffix)), name + suffix
    has_raceline = os.path.isfile(os.path.join(d, f"{name}_raceline.csv"))
    assert has_raceline == (name not in MAPS_WITHOUT_RACELINE)
    # No map ships a _config.yaml: the starting pose is derived from the centerline.
    assert not os.path.exists(os.path.join(d, f"{name}_config.yaml"))


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
@pytest.mark.parametrize("name", CANONICAL_MAPS)
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
