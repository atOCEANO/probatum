"""An environment is built and registered as Gymnasium expects.

probatum.env returns a Gymnasium vector environment whose spaces follow its
observation and its actions, probatum.rl registers it as probatum/Trading-v0, so
gymnasium.make builds the single environment, which passes Gymnasium's own checker,
and gymnasium.make_vec the batch, and import probatum loads no Gymnasium. An
environment says what its runs assume as a backtest's assumptions, each cost it draws
shown as its range, records what it was built from, and refuses what it cannot take
from probatum's tree (Specification, The environment).
"""

import subprocess
import sys
import warnings

import gymnasium
import gymnasium.utils.env_checker
import numpy as np
import pytest

import probatum
import probatum.rl
from probatum import (DataError, ProbatumKeyError, ProbatumNotImplementedError,
                      ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError,
                      VenueError)
from test_an_episode_runs_again_as_a_backtest_from_its_start import (HOUR, VENUE,
                                                                     universe, walked)

PLAIN = probatum.Venue(fees={"maker": 0.0, "taker": 0.0})
MIXED = probatum.venues.binance.replace(balance={"spot": 5_000.0, "margin": 5_000.0},
                                        funding="none")


def mixed(rows=120):
    """A spot pair and a perpetual on one clock."""
    rng   = np.random.default_rng(8)
    times = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * HOUR
    btc   = probatum.Data(
        walked(30_000.0, rows, rng),
        symbol = "BTCUSDT",
        market = "spot",
        time   = times,
    )
    eth   = probatum.Data(
        walked(2_000.0, rows, rng),
        symbol = "ETHUSDT",
        market = "linear",
        time   = times,
    )

    return btc.join(eth)


def test_import_probatum_and_a_backtest_load_no_gymnasium():
    code = ("import sys, numpy, probatum\n"
            "from probatum import *\n"
            "bars = numpy.tile([100.0, 101.0, 99.0, 100.0, 1e6], (40, 1))\n"
            "data = probatum.Data(bars, symbol='BTCUSDT', market='linear',\n"
            "                     time=numpy.arange(40) * 3_600_000_000_000)\n"
            "venue = probatum.Venue(fees={'maker': 0.0, 'taker': 0.0})\n"
            "probatum.backtest(numpy.zeros((40, 1)), data, venue)\n"
            "assert 'gymnasium' not in sys.modules\n")

    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_spaces_follow_the_observation_and_the_actions():
    one = probatum.env(universe()["BTCUSDT"], VENUE, window=64)

    # the Specification's worked example: 64 bars of five, four for the slot and four
    # for the account
    assert one.single_observation_space == gymnasium.spaces.Box(-np.inf, np.inf, (328,),
                                                                np.float32)
    assert one.observation_space.shape == (1, 328)
    assert one.single_action_space     == gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float32)

    spot = probatum.env(mixed(), MIXED, window=4, num_envs=3)

    assert spot.single_action_space.low.tolist() == [0.0, -1.0]
    assert spot.action_space.shape == (3, 2)
    assert (probatum.env(mixed(), MIXED, window=4, actions="discrete").single_action_space
            == gymnasium.spaces.MultiDiscrete([2, 3]))
    assert (probatum.env(universe()["BTCUSDT"], VENUE, window=4,
                         actions="discrete").single_action_space
            == gymnasium.spaces.Discrete(3))
    assert probatum.env(mixed(), MIXED, window=4,
                        actions="simplex").single_action_space.shape == (3,)

    orders = probatum.env(mixed(), MIXED, window=4, actions="orders")

    assert orders.single_action_space.shape        == (2, 2)
    assert orders.single_action_space.low.tolist() == [[-1.0, 0.0], [-1.0, 0.0]]
    # each slot adds its resting orders and the nearest one's distance
    assert orders.single_observation_space.shape == (4 * 2 * 5 + 6 * 2 + 4,)

    split = probatum.env(mixed(), MIXED, window=4, layout="dict",
                         features=lambda data: np.ones((len(data), 2, 3)))

    assert split.single_observation_space["market"].shape  == (4, 2, 3)
    assert split.single_observation_space["account"].shape == (4 * 2 + 4,)

    obs, _ = split.reset(seed=0)

    assert obs["market"].shape == (1, 4, 2, 3) and obs["account"].shape == (1, 12)


def test_gymnasium_makes_the_single_environment_and_the_batch():
    data   = universe()
    single = gymnasium.make("probatum/Trading-v0", data=data, venue=VENUE, window=16)

    assert isinstance(single.unwrapped, probatum.rl.Env)

    with warnings.catch_warnings():
        # a Box unbounded either way draws Gymnasium's warning, and an observation of
        # ratios and returns has no bound
        warnings.filterwarnings("ignore", message=".*infinity")
        gymnasium.utils.env_checker.check_env(single.unwrapped)

    obs, info = single.reset(seed=1)

    assert obs.shape == (16 * 2 * 5 + 4 * 2 + 4,) and info["step"] == 0

    batch = gymnasium.make_vec("probatum/Trading-v0", num_envs=4, data=data,
                               venue=VENUE, window=16)

    assert isinstance(batch, probatum.rl.VectorEnv) and batch.num_envs == 4

    with pytest.raises(ProbatumTypeError):
        probatum.rl.Env(data, VENUE, num_envs=2)


def test_an_environment_says_what_it_assumes_with_each_drawn_cost_s_range():
    env  = probatum.env(universe(), VENUE, window=8, num_envs=2, seed=3,
                        costs={"taker": (0.0004, 0.0008), "slippage": (1.0, 5.0),
                               "impact": (0.05, 0.2)})
    rows = {row["name"]: row for row in env.describe()}

    assert rows["fees"]["value"] == ("maker 0.02%, taker 0.04% to 0.08%, drawn for each "
                                     "environment")
    assert rows["fees"]["source"].endswith("; taker set by the user")
    assert rows["fees"]["grade"]      == "simplified"
    assert rows["slippage"]["value"]  == "1 bp to 5 bps, drawn for each environment"
    assert rows["slippage"]["source"] == "set by the user"
    assert rows["impact"]["value"]    == "sqrt, k = 0.05 to 0.2, drawn for each environment"
    assert rows["impact"]["grade"]    == "simplified"

    plain = probatum.env(universe(), VENUE, window=8).describe()

    assert {row["name"]: row["value"] for row in plain}["fees"] == ("maker 0.02%, taker "
                                                                   "0.05%")

    # a maker rate drawn where the venue charges no taker is still a charge
    free  = probatum.env(universe(), PLAIN, window=8,
                         costs={"maker": (0.0001, 0.0003)}).describe()
    fees  = {row["name"]: row for row in free}["fees"]

    assert fees["value"] == "maker 0.01% to 0.03%, drawn for each environment, taker 0%"
    assert fees["grade"] == "simplified"

    # and a taker drawn there is what a perpetual with no maintenance is closed at
    taken = probatum.env(universe(), PLAIN, window=8,
                         costs={"taker": (0.0004, 0.0008)}).describe()
    closed = {row["name"]: row for row in taken}["liquidation fee"]

    assert (closed["value"], closed["grade"]) == ("the taker rate", "simplified")
    assert closed["source"] == "set by the user"


def test_an_environment_records_what_it_was_built_from():
    env = probatum.env(universe(), VENUE, window=8, num_envs=2, seed=3, sample=1,
                       costs={"taker": (0.0004, 0.0008)})
    held = env.provenance

    assert held["seed"] == 3 and held["venue"] == repr(VENUE)
    assert held["data"] == universe().provenance
    assert held["settings"]["window"] == 8 and held["settings"]["sample"] == 1
    assert held["settings"]["costs"] == {"taker": [0.0004, 0.0008]}
    assert held["streams"] == {"episodes {i}": "PCG64", "costs": "PCG64"}

    random = probatum.env(universe(), VENUE.replace(intrabar="random"), window=8, seed=3)

    assert random.provenance["streams"] == {
        "episodes {i}": "PCG64",
        "intrabar":     "ChaCha20, each environment's",
    }


def shorting(actions, state):
    return np.full(state.weights.shape, -0.5)


REFUSED = [
    (dict(num_envs=0), ProbatumValueError,
     "num_envs: 0; give a whole number of environments from 1"),
    (dict(threads=0), ProbatumValueError,
     "threads: 0; give a whole number of threads from 1"),
    (dict(threads=2.5), ProbatumTypeError,
     "threads: 2.5; give a whole number of threads"),
    (dict(window=2.5), ProbatumTypeError, "window: 2.5; give a whole number of bars"),
    (dict(layout="grid"), ProbatumValueError,
     "layout: 'grid'; give \"flat\" or \"dict\""),
    (dict(actions="continuous"), ProbatumValueError,
     "actions: 'continuous'; give \"box\", \"simplex\", \"discrete\" or \"orders\""),
    (dict(max_leverage=0.0), ProbatumValueError,
     "max_leverage: 0.0; give a multiple of equity above 0"),
    (dict(max_offset=-1), ProbatumValueError,
     "max_offset: -1; give a number of ticks from 0"),
    (dict(decode=3), ProbatumTypeError, "decode: an object of type int; give a function"),
    (dict(decode=shorting, actions="orders"), ProbatumValueError,
     "decode: a function, and actions=\"orders\" are limits"),
    (dict(reward="profit"), ProbatumValueError,
     "reward: 'profit'; give \"log\", \"sharpe\" or a function of the state"),
    (dict(reward="sharpe", window=1), ProbatumValueError,
     "reward: \"sharpe\" over a window of 1 bar, whose moments keep the last return "
     "alone"),
    (dict(episode_length=0), ProbatumValueError,
     "episode_length: 0; give a whole number of steps from 1"),
    (dict(sample=3), ProbatumValueError, "sample: 3 of 2 instruments; give at most 2"),
    (dict(normalise=1), ProbatumTypeError, "normalise: 1; give True or False"),
    (dict(record="transitions"), ProbatumNotImplementedError,
     "record: 'transitions', which an environment does not record yet"),
    (dict(record="full"), ProbatumValueError, "record: 'full'; leave it \"none\""),
    (dict(costs={"fee": (0.0, 0.1)}), ProbatumKeyError,
     "costs: 'fee', which is no cost an environment draws"),
    (dict(costs={"taker": 0.001}), ProbatumTypeError,
     "costs: taker 0.001; give a (low, high) pair of numbers"),
    (dict(costs={"taker": (0.002, 0.001)}), VenueError,
     "costs: taker (0.002, 0.001); give finite bounds, the low one first"),
    (dict(costs={"maker": (-1.0, 0.0)}), VenueError,
     "costs: maker from -1.0, a rebate past the notional"),
    (dict(costs={"slippage": (-1.0, 2.0)}), VenueError,
     "costs: slippage from -1.0, which would pay the account to trade"),
    (dict(tolerance=-0.1), ProbatumValueError,
     "tolerance: -0.1; give a fraction of equity from 0"),
    (dict(autoreset="sometimes"), ProbatumValueError,
     "autoreset: 'sometimes'; give \"next_step\", \"same_step\" or \"disabled\""),
    (dict(fill=0.0), ProbatumValueError,
     "fill: 0.0, and no features are given for it to fill"),
    (dict(window=400), DataError,
     "window: 400, and no bar before the last has an instrument shown"),
    (dict(features=lambda data: np.full((len(data), 2, 1), 1e39)), DataError,
     "features: 1e+39 at bar 0"),
    (dict(venue=PLAIN, actions="orders"), VenueError,
     "actions: \"orders\" rests each limit whole ticks from the close, and the venue "
     "gives BTCUSDT no tick"),
    (dict(data=mixed(), venue=MIXED, sample=1), ProbatumValueError,
     "sample: 1 of a universe of spot and perpetual instruments"),
]


@pytest.mark.parametrize("given, kind, said", REFUSED)
def test_an_environment_refuses_what_it_cannot_take_with_the_fix(given, kind, said):
    given = {**dict(data=universe(), venue=VENUE, window=8), **given}

    with pytest.raises(kind) as caught:
        probatum.env(**given)

    assert type(caught.value) is kind and str(caught.value).startswith(said)


def nan_reward(state):
    return np.full(len(state.equity), np.nan)


STEPPED = [
    (dict(), np.zeros(3), ProbatumValueError,
     "actions: shape (3,) for 2 environments of 2 slots; give (2, 2)"),
    (dict(), np.full((2, 2), np.nan), ProbatumValueError,
     "actions: nan for environment 0; give finite actions"),
    (dict(), [[1.0, 0.0], [1.0]], ProbatumValueError,
     "actions: an object of type list, which numpy reads as no one array; give (2, 2)"),
    (dict(actions="discrete"), np.array([[1, 3], [1, 1]]), ProbatumValueError,
     "actions: 3 for environment 0's slot 1; give a whole choice from 0 to 2"),
    (dict(decode=lambda actions, state: np.zeros(2)), np.zeros((2, 2)),
     ProbatumValueError,
     "decode: returned shape (2,) for 2 environments of 2 slots; return (2, 2)"),
    (dict(data=mixed(), venue=MIXED, decode=shorting), np.zeros((2, 2)),
     ProbatumValueError,
     "decode: -0.5 for environment 0's slot 0, BTCUSDT, which trades spot and cannot "
     "be short"),
    (dict(reward=nan_reward), np.zeros((2, 2)), ProbatumValueError,
     "reward: nan for environment 0; return finite rewards"),
]


@pytest.mark.parametrize("given, actions, kind, said", STEPPED)
def test_a_step_refuses_actions_and_returns_it_cannot_take(given, actions, kind, said):
    env = probatum.env(**{**dict(data=universe(), venue=VENUE, window=8, num_envs=2),
                          **given})
    env.reset(seed=0)

    with pytest.raises(kind) as caught:
        env.step(actions)

    assert type(caught.value) is kind and str(caught.value).startswith(said)


def test_one_environment_refuses_an_action_its_batch_refuses():
    single = probatum.rl.Env(universe(), VENUE, window=8)
    single.reset(seed=0)

    with pytest.raises(ProbatumValueError, match="actions: an object of type list"):
        single.step([1.0, [1.0]])


def test_a_step_before_a_reset_and_a_mask_naming_nothing_are_refused():
    env = probatum.env(universe(), VENUE, window=8, num_envs=2, autoreset="disabled")

    with pytest.raises(ProbatumRuntimeError, match="step: before any reset"):
        env.step(np.zeros((2, 2)))
    with pytest.raises(ProbatumRuntimeError, match="options: a reset_mask before every"):
        env.reset(options={"reset_mask": np.array([True, False])})

    env.reset()

    with pytest.raises(ProbatumValueError, match="options: a reset_mask naming no"):
        env.reset(options={"reset_mask": np.zeros(2, dtype=bool)})
    with pytest.raises(ProbatumValueError, match="options: a reset_mask of int64"):
        env.reset(options={"reset_mask": np.array([1, 0])})
    with pytest.raises(ProbatumValueError, match="options: a reset_mask numpy reads"):
        env.reset(options={"reset_mask": [True, [False]]})


def test_each_observation_is_its_own_array_unless_copy_is_off():
    for copy in (True, False):
        env = probatum.env(universe(), VENUE, window=8, num_envs=2, copy=copy)
        first, _ = env.reset(seed=0)
        held = first.copy()
        second, *_ = env.step(np.ones((2, 2), dtype=np.float32))

        # copy=False writes each step's observation into the one array it hands out
        assert (second is first) is not copy
        assert np.array_equal(first, held) is copy
