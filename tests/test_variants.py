"""Modified tracks (E-60): names, geometry of each transformation, drawing, loading."""

import numpy as np
import pytest
import yaml
from scipy import ndimage

from crashlearn import tracks, variants

BASES = tracks.train_tracks() + tracks.validation_tracks()


def wall_gap_m(a: np.ndarray, b: np.ndarray, resolution: float) -> float:
    """95th percentile distance from each wall pixel of `a` to the nearest wall of `b`."""
    walls_a, walls_b = a <= 128, b <= 128
    return float(np.percentile(ndimage.distance_transform_edt(~walls_b)[walls_a], 95)) * resolution


def clearance_m(track: variants.TrackMap) -> np.ndarray:
    """Distance from each centerline point to the nearest wall."""
    distance = ndimage.distance_transform_edt(track.image > 128) * track.resolution
    row, col = track.pixel(track.centerline[:, 0], track.centerline[:, 1])
    return distance[np.round(row).astype(int), np.round(col).astype(int)]


def test_names():
    assert variants.parse_name("Sochi@mirror+reverse") == ("Sochi", ("mirror", "reverse"))
    assert variants.parse_name("Mexico City@narrow") == ("MexicoCity", ("narrow",))
    for bad in ("Sochi", "Sochi@", "Sochi@twist"):
        with pytest.raises(ValueError):
            variants.parse_name(bad)
    with pytest.raises(KeyError):
        variants.parse_name("Monaco@mirror")


@pytest.mark.parametrize("base", sorted(tracks.sealed_test_tracks(unseal=True)))
def test_sealed_test_tracks_are_never_a_base(base):
    with pytest.raises(PermissionError):
        variants.build(f"{base}@mirror")


@pytest.mark.parametrize("base", ["Montreal", "Silverstone", "MexicoCity"])
def test_render_reproduces_the_official_walls(base):
    official = variants.load_official(base)
    drawn = variants.render(official)
    assert wall_gap_m(official.image, drawn, official.resolution) < 0.1
    assert wall_gap_m(drawn, official.image, official.resolution) < 0.1


@pytest.mark.slow
@pytest.mark.parametrize("base", BASES)
def test_render_reproduces_every_official_map(base):
    test_render_reproduces_the_official_walls(base)


def test_reverse_keeps_the_start_and_swaps_the_sides():
    official = variants.load_official("Sochi")
    back = variants.reverse(official)
    c, r = official.centerline, back.centerline
    order = np.r_[0, np.arange(len(c) - 1, 0, -1)]  # the start stays first
    assert np.array_equal(r[:, :2], c[order, :2])
    assert np.array_equal(r[:, 2:], c[order][:, [3, 2]])
    assert np.array_equal(back.image, official.image)
    assert np.allclose(variants.reverse(back).centerline, c)


def test_mirror_maps_each_point_to_the_same_clearance():
    official = variants.load_official("Sochi")
    flipped = variants.mirror(official)
    assert np.array_equal(flipped.centerline[:, 0], -official.centerline[:, 0])
    assert np.array_equal(flipped.centerline[:, 2:], official.centerline[:, 3:1:-1])
    assert np.allclose(clearance_m(flipped), clearance_m(official), atol=official.resolution)
    twice = variants.mirror(flipped)
    assert np.array_equal(twice.image, official.image)
    assert np.allclose(twice.origin, official.origin)


def test_narrow_shrinks_the_corridor():
    official = variants.load_official("Montreal")
    narrowed = variants.narrow(official)
    widths = official.centerline[:, 2:].sum(axis=1)
    assert np.allclose(narrowed.centerline[:, 2:].sum(axis=1), widths * variants.NARROW_FACTOR)
    assert np.all(clearance_m(narrowed) > 0.2)  # the car (0.31 m wide) still fits
    ratio = np.median(clearance_m(narrowed) / clearance_m(official))
    assert ratio == pytest.approx(variants.NARROW_FACTOR, abs=0.05)


def test_files_round_trip(tmp_path):
    name = "Montreal@mirror+narrow"
    target = variants.ensure(name, tmp_path)
    meta = yaml.safe_load((target / f"{name}_map.yaml").read_text())
    built = variants.build(name)
    assert meta["image"] == f"{name}_map.png"
    assert np.allclose(meta["origin"], built.origin)
    saved = np.loadtxt(target / f"{name}_centerline.csv", delimiter=",", comments="#")
    assert np.allclose(saved, built.centerline, atol=1e-6)
    assert variants.ensure(name, tmp_path) == target  # second call: nothing rebuilt


@pytest.mark.slow
def test_env_drives_a_variant():
    from crashlearn.env import CrashLearnEnv, sim

    name = "Montreal@mirror+reverse"
    env = CrashLearnEnv(maps=name, max_episode_steps=40)
    try:
        env.reset(seed=0)
        expected = variants.build(name).centerline[:, :2]
        assert np.allclose(sim._get_sim()._waypoints[: len(expected), :2], expected, atol=1e-6)
        for _ in range(20):
            *_, info = env.step(np.array([0.3, 0.0]))
        assert info["progress_delta"] >= 0
    finally:
        env.close()
