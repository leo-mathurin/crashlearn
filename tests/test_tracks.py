"""Track inventory and fixed splits (E-9). Splits are copied here on purpose:
editing tracks.yaml to move a track must fail CI and require a team decision."""

import collections

import pytest

from crashlearn import tracks

TRAIN = {
    "Austin", "BrandsHatch", "Budapest", "Catalunya", "Hockenheim", "IMS", "Melbourne",
    "MexicoCity", "Monza", "Nuerburgring", "Sakhir", "SaoPaulo", "Sepang", "Silverstone",
    "Zandvoort",
}  # fmt: skip
VALIDATION = {"Montreal", "Sochi", "Spielberg", "YasMarina"}
TEST = {"MoscowRaceway", "Oschersleben", "Shanghai", "Spa"}
TRACKS = tracks._load()["tracks"]


def test_yaml_matches_vendored_directories():
    names = {t["name"] for t in TRACKS}
    aliases = {a for t in TRACKS for a in t.get("aliases", [])}
    assert not names & aliases
    assert (
        set(tracks.track_dirs()) == names | aliases == TRAIN | VALIDATION | TEST | {"Mexico City"}
    )


@pytest.mark.parametrize("track", TRACKS, ids=lambda t: t["name"])
def test_generated_fields_are_up_to_date(track):
    fresh = tracks.compute_generated_fields(tracks.MAPS_DIR / track["name"])
    assert {k: track.get(k) for k in fresh} == fresh, "run scripts/track_inventory.py --write"


def test_mexico_city_is_an_exact_duplicate():
    def hashes(name):
        return collections.Counter(
            a["sha256"] for a in tracks.inventory(tracks.MAPS_DIR / name).values()
        )

    assert hashes("Mexico City") == hashes("MexicoCity")
    assert tracks.canonical_name("Mexico City") == "MexicoCity"
    assert tracks.canonical_name("MexicoCity") == "MexicoCity"


def test_unknown_track_is_rejected():
    with pytest.raises(KeyError):
        tracks.canonical_name("Monaco")


def test_splits_are_the_framed_ones():
    assert set(tracks.train_tracks()) == TRAIN and len(tracks.train_tracks()) == 15
    assert set(tracks.validation_tracks()) == VALIDATION and len(tracks.validation_tracks()) == 4
    assert set(tracks.sealed_test_tracks(unseal=True)) == TEST
    assert set(tracks.final_retraining_tracks()) == TRAIN | VALIDATION | TEST
    assert len(tracks.final_retraining_tracks()) == 23
    assert "Mexico City" not in TRAIN | VALIDATION | TEST


def test_sealed_test_is_not_reachable_by_default():
    with pytest.raises(PermissionError):
        tracks.sealed_test_tracks()
    assert not TEST & set(tracks.train_tracks() + tracks.validation_tracks())


def test_montreal_and_shanghai_have_no_raceline():
    missing = {t["name"] for t in TRACKS if "raceline" not in t["assets"]}
    assert missing == {"Montreal", "Shanghai"}


def test_tight_turns_follow_the_threshold():
    xy = tracks.read_centerline(tracks.MAPS_DIR / "IMS" / "IMS_centerline.csv")
    assert tracks.describe_geometry(xy, 0.5)["geometry_tight_turns"] == 0
    assert tracks.describe_geometry(xy, 0.0)["geometry_tight_turns"] == 1  # whole loop is one run


@pytest.fixture
def restore_example_map():
    yield
    import env_simulation

    env_simulation.set_map("example")


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(TRAIN | VALIDATION | TEST))
def test_track_loads_in_simulator(name, restore_example_map):
    result = tracks.load_and_step(name, steps=20)
    assert result["time"] == pytest.approx(1.0)
    assert result["status"] == 1
