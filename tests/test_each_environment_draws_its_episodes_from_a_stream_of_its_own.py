"""Each environment draws its episodes from a stream of its own.

Environment i draws each episode, its start and then its slots, from the stream named
"episodes i" of the seed, so its k-th episode is the same whatever num_envs is and
comes again from the seed alone, a seed left out being drawn and recorded;
reset(seed=) starts the streams again; and the costs each environment draws once,
four draws of the stream named "costs" for each, are its own whatever num_envs is
and stay as drawn for its life (Specification, Episodes).
"""

import numpy as np
import pytest

import probatum
from probatum._streams import stream
from test_an_episode_runs_again_as_a_backtest_from_its_start import VENUE, universe


def episodes(env, count=6):
    """The starts and slots of each environment's first `count` episodes, one a step,
    as the info gives them."""
    _, info = env.reset()
    starts, slots = [info["start"]], [info.get("instruments")]

    for _ in range(count - 1):
        _, _, _, _, info = env.step(env.action_space.sample())
        starts.append(info["start"])
        slots.append(info.get("instruments"))

    return np.array(starts), slots


def built(num_envs, seed, data=None, **given):
    return probatum.env(universe() if data is None else data, VENUE, window=8,
                        num_envs=num_envs, seed=seed, episode_length=1,
                        autoreset="same_step", **given)


def test_the_first_episode_starts_at_the_first_draw_of_its_stream():
    _, info = built(3, 4).reset()

    # every bar from the window's last to the one before the last can start one
    starts = np.arange(7, len(universe()) - 1)
    assert info["start"].tolist() == [
        starts[stream(4, f"episodes {i}").integers(len(starts))] for i in range(3)]


def test_an_environment_on_a_slice_opens_no_episode_before_a_whole_window_of_it():
    data = universe()[100:]

    starts, _ = episodes(built(64, 8, data), 30)

    assert starts.min() >= 7 and starts.max() <= len(data) - 2


@pytest.mark.parametrize("sample", [None, 1])
def test_an_environment_s_episodes_are_the_same_whatever_the_number_of_them(sample):
    few, picked  = episodes(built(2, 4, sample=sample))
    many, chosen = episodes(built(5, 4, sample=sample))

    assert np.array_equal(few, many[:, :2])
    if sample:
        assert all(np.array_equal(a, b[:2]) for a, b in zip(picked, chosen))

    # and every environment does not draw the same
    assert len(set(many[:, 2:].ravel().tolist())) > 1


def test_a_reset_with_a_seed_starts_every_stream_again():
    env = built(3, 1)
    env.reset()
    env.step(env.action_space.sample())

    _, info = env.reset(seed=9)

    assert np.array_equal(info["start"], episodes(built(3, 9), 1)[0][0])
    # numbered from the stream's first episode again
    assert info["episode"].tolist() == [0, 0, 0]


def test_a_seed_left_out_is_drawn_and_recorded_so_its_episodes_come_again():
    env  = built(3, None)
    seed = env.provenance["seed"]

    assert isinstance(seed, int)
    assert np.array_equal(episodes(env)[0], episodes(built(3, seed))[0])


def rates(num_envs, reset=None):
    """Each environment's taker rate, read off its first step's fees over what it
    traded, a weight of one on one instrument from no position."""
    paid = []

    def reward(state):
        paid.append(state.fees / state.turnover)
        return state.log_return

    env = probatum.env(universe()["BTCUSDT"], VENUE, window=8, num_envs=num_envs,
                       seed=6, costs={"taker": (0.001, 0.004)}, reward=reward)
    env.reset(seed=reset)
    env.step(np.ones((num_envs, 1), dtype=np.float32))

    return paid[0]


def test_each_environment_draws_its_costs_once_whatever_the_number_of_them():
    few, many = rates(2), rates(5)

    assert few == pytest.approx(many[:2], rel=1e-9)

    # environment i's rates are the i-th four draws of the stream named costs, the
    # taker's the second of them
    taken = 0.001 + 0.003 * stream(6, "costs").random((5, 4))[:, 1]
    assert many == pytest.approx(taken, rel=1e-9)
    # a reset's seed draws episodes again, and leaves the costs as they were drawn
    assert rates(2, reset=99) == pytest.approx(few, rel=1e-9)


def test_a_drawn_fee_is_every_instrument_s_in_place_of_its_own():
    # BTCUSDT charges a taker rate of 1% of its own, which the drawn rate replaces
    venue = VENUE.replace(instruments={"BTCUSDT": {"fees": {"taker": 0.01}}})
    paid  = []

    def reward(state):
        paid.append(state.fees / state.turnover)
        return state.log_return

    env = probatum.env(universe(), venue, window=8, num_envs=3, seed=6,
                       costs={"taker": (0.001, 0.004)}, reward=reward)
    env.reset()
    env.step(np.array([[1.0, 0.0]] * 3, dtype=np.float32))

    assert np.all((paid[0] >= 0.001) & (paid[0] <= 0.004))
