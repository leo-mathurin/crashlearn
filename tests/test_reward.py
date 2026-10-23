"""RaceReward (E-15): terms, one-off events and the exploits the formula must not pay."""

import math

import numpy as np
import pytest

from crashlearn.reward import (
    STATUS_ACTIVE,
    STATUS_DNF,
    STATUS_FINISHED,
    TERMS,
    RaceReward,
    RewardConfig,
)

TRACK_M = 400.0
STILL = np.zeros(2)


def step(delta=0.0, *, velocity=6.0, status=STATUS_ACTIVE, lap=False, wall=False, vehicle=False):
    """(raw obs, raw step info) of car 0 for one decision."""
    obs = {"velocity": velocity}
    info = {
        "agent_status": {0: status},
        "progress_delta": {0: delta},
        "lap_complete": {0: lap},
        "collisions": {"wall": {0: wall}, "vehicle": {0: vehicle}},
    }
    return obs, info


@pytest.fixture
def reward():
    r = RaceReward()
    r.reset(track_length_m=TRACK_M)
    return r


def test_terms_are_all_logged_and_sum_to_the_total(reward):
    total, terms = reward(*step(0.3 / TRACK_M, lap=True, wall=True), 0, STILL)
    assert tuple(terms) == TERMS
    assert total == pytest.approx(math.fsum(terms.values()))
    assert terms["progress"] == pytest.approx(0.3)
    assert terms["lap"] == 20


def test_progress_is_paid_per_metre_on_every_track():
    short, long_ = RaceReward(), RaceReward()
    short.reset(track_length_m=293.0)
    long_.reset(track_length_m=487.0)
    # 1 m of centerline is a larger lap fraction on the shorter track
    _, a = short(*step(1 / 293.0), 0, STILL)
    _, b = long_(*step(1 / 487.0), 0, STILL)
    assert a["progress"] == pytest.approx(b["progress"]) == pytest.approx(1.0)


def test_reversing_over_a_stretch_takes_back_what_driving_it_paid(reward):
    there = [reward(*step(0.3 / TRACK_M), 0, STILL)[1]["progress"] for _ in range(50)]
    back = [
        reward(*step(-0.3 / TRACK_M, velocity=-2.0), 0, STILL)[1]["progress"] for _ in range(50)
    ]
    assert math.fsum(there + back) == pytest.approx(0.0, abs=1e-9)


def test_standing_still_never_pays(reward):
    for _ in range(199):
        total, terms = reward(*step(0.0, velocity=0.0), 0, STILL)
        assert total < 0 and terms["slow"] < 0
        assert terms["reverse"] == 0  # an idle car is not reversing


def test_a_short_reverse_is_free_a_long_one_is_not(reward):
    grace = RewardConfig().reverse_grace_steps
    terms = [reward(*step(0.0, velocity=-1.0), 0, STILL)[1] for _ in range(grace + 1)]
    assert all(t["reverse"] == 0 for t in terms[:grace])
    assert terms[grace]["reverse"] < 0


def test_contacts_are_logged_but_not_weighted(reward):
    _, terms = reward(*step(0.3 / TRACK_M, wall=True, vehicle=True), 0, STILL)
    assert terms["wall_contact"] == terms["vehicle_contact"] == 0


def test_finish_and_dnf_are_one_off_terminal_terms(reward):
    _, fin = reward(*step(0.3 / TRACK_M, status=STATUS_FINISHED, lap=True), 0, STILL)
    assert (fin["finish"], fin["lap"], fin["dnf"]) == (50, 20, 0)
    reward.reset(track_length_m=TRACK_M)
    _, dnf = reward(*step(0.0, status=STATUS_DNF), 0, STILL)
    assert (dnf["dnf"], dnf["finish"], dnf["slow"]) == (-200, 0, 0)


def test_action_changes_cost_quadratically_and_reset_clears_the_history(reward):
    _, first = reward(*step(), 0, np.array([1.0, 1.0]))
    assert first["action_change"] == 0  # no previous action
    _, flip = reward(*step(), 0, np.array([-1.0, -1.0]))
    assert flip["action_change"] == pytest.approx(-0.02 * 8)
    reward.reset(track_length_m=TRACK_M)
    _, after_reset = reward(*step(), 0, np.array([1.0, 1.0]))
    assert after_reset["action_change"] == 0


def test_a_lethal_shortcut_does_not_pay():
    """Sochi (E-10): +126 m through a wall, then DNF. A shortcut that kills must lose."""
    r = RaceReward()
    r.reset(track_length_m=TRACK_M)
    _, shortcut = r(*step(126 / TRACK_M), 0, STILL)
    _, death = r(*step(0.0, status=STATUS_DNF), 0, STILL)
    assert shortcut["progress"] + death["dnf"] < 0


def test_track_length_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        RaceReward().reset(track_length_m=0.0)
