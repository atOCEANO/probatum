"""An agent carries the spec it trained with.

An agent is judged fairly only if a backtest shows it what it saw in training and
reads its outputs as they were read. So an environment records, as one JSON document,
its inputs, its observation, its outputs, its venue and the span it trained on, a
custom function by the hash of its output on probe bars built the same way on every
machine, and with normalise=True the mean and deviation each market feature is written
over (Specification, The agent spec; ADR 0039).
"""

import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import probatum
from probatum import ProbatumWarning, SpecError, _agent
from reference import environment
from reference.simulator import Bar
from test_an_episode_runs_again_as_a_backtest_from_its_start import HOUR, VENUE, universe


def momentum(data):
    """Each instrument's return over its last bar and its volume in tens of thousands."""
    closes = data.closes
    out    = np.full(closes.shape + (2,), np.nan)

    out[1:, :, 0] = closes[1:] / closes[:-1] - 1.0
    out[:, :, 1]  = data.volumes / 1e4
    return out


def doubled(data):
    return 2.0 * momentum(data)


def rewritten(data):
    """momentum, written another way: the same output from other source."""
    closes = data.closes
    moved  = np.vstack([np.full((1, closes.shape[1]), np.nan), closes[1:] / closes[:-1]])
    return np.stack([moved - 1.0, data.volumes / 1e4], axis=2)


def hashed(function):
    text = inspect.getsource(function).encode("utf-8")
    return "b2:" + hashlib.blake2b(text, digest_size=8).hexdigest()


def test_the_spec_records_the_inputs_the_observation_and_the_outputs():
    data = universe()
    spec = probatum.env(data, VENUE, window=8, max_leverage=2.0, tolerance=0.01,
                        num_envs=2, seed=1).agent_spec

    assert (spec["schema"], spec["window"]) == (1, 8)
    assert (spec["instruments"], spec["slots"], spec["sample"]) == (
        ["BTCUSDT", "ETHUSDT"], 2, None)
    assert (spec["features"], spec["fill"], spec["normalisation"]) == (
        {"builtin": "candles"}, None, None)
    assert spec["observation"] == {
        "size":    8 * 2 * 5 + 2 * 4 + 4,
        "layout":  "flat",
        "shape":   [8, 2, 5],
        "account": ["weight", "upnl", "entry_distance", "shown"] * 2
                   + ["log_return", "leverage", "margin_ratio", "cash_weight"],
    }
    assert spec["actions"] == {
        "space":        "box",
        "low":          [-1.0, -1.0],
        "high":         1.0,
        "max_leverage": 2.0,
    }
    assert spec["decode"]    == {"builtin": "box"}
    assert spec["reward"]    == {"builtin": "log", "floor": 0.01}
    assert spec["tolerance"] == 0.01
    assert spec["training"] == {
        "fingerprint": data.fingerprint,
        "interval":    "1h",
        "bars":        300,
        "start":       "2023-11-14T22:13:20Z",
        "end":         "2023-11-27T09:13:20Z",
    }
    assert spec["version"] == probatum.__version__


def test_the_spec_is_json_and_handed_out_as_a_copy():
    env  = probatum.env(universe(), VENUE, window=8, features=momentum, normalise=True,
                        seed=1)
    spec = env.agent_spec

    assert json.loads(json.dumps(spec, allow_nan=False)) == spec

    spec["window"] = 64
    spec["normalisation"]["mean"][0] = 9.0

    assert env.agent_spec["window"] == 8
    assert env.agent_spec["normalisation"]["mean"][0] != 9.0


@pytest.mark.parametrize("venue", [
    # a fee side a kind's table drops, which a merge side by side would put back
    probatum.venues.binance.replace(kinds={"linear": {"fees": {"taker": None}}}),
    # a table of an instrument whose tiers end at an infinite bound
    probatum.venues.hyperliquid_perp.replace(instruments={"AAVEUSDC": {"lot": 0.1}}),
])
def test_a_venue_s_changes_build_it_again_through_its_repr_and_its_spec(venue):
    assert eval(repr(venue), {"probatum": probatum}) == venue

    spec = {"venue": {"name": venue._source, "taken": venue.taken,
                      "changed": json.loads(json.dumps(
                          probatum._storage._plain(probatum._venue._changes(venue)),
                          allow_nan=False))}}

    assert _agent._trained_venue(spec) == venue


def test_the_venue_is_its_preset_the_day_it_was_taken_and_each_setting_changed():
    random = VENUE.replace(intrabar="random", leverage=5.0)
    spec   = probatum.env(universe(), random, window=8, seed=1).agent_spec

    assert spec["venue"] == {"name": "binance_perp", "taken": "2026-10-16",
                             "changed": {"leverage": 5.0, "intrabar": "random",
                                         "funding": "none"}}

    plain = probatum.Venue(fees={"maker": 0.0, "taker": 0.0}, balance=50_000.0)
    one   = probatum.Data(
        universe()["BTCUSDT"].bars[:, 0],
        symbol = "BTCUSDT",
        market = "linear",
        time   = universe().time,
    )
    held  = probatum.env(one, plain, window=8, seed=1).agent_spec["venue"]

    assert held == {
        "name":    None,
        "taken":   None,
        "changed": {"balance": 50_000.0, "fees": {"maker": 0.0, "taker": 0.0}},
    }

    # a variant renamed is still its preset's, and a venue built by hand under a
    # preset's name is not that preset
    renamed = VENUE.replace(name="desk", leverage=5.0)
    held    = probatum.env(universe(), renamed, window=8, seed=1).agent_spec["venue"]

    assert held == {
        "name":    "binance_perp",
        "taken":   "2026-10-16",
        "changed": {"leverage": 5.0, "funding": "none", "name": "desk"},
    }

    named = probatum.Venue(name="binance_perp", fees={"maker": 0.0, "taker": 0.0})
    held  = probatum.env(one, named, window=8, seed=1).agent_spec["venue"]

    assert held == {"name": None, "taken": None,
                    "changed": {"fees": {"maker": 0.0, "taker": 0.0},
                                "name": "binance_perp"}}

    # and the spec builds each of them again
    for venue, data in ((random, universe()), (renamed, universe()), (plain, one),
                        (named, one)):
        held    = probatum.env(data, venue, window=8, seed=1).agent_spec["venue"]
        rebuilt = (getattr(probatum.venues, held["name"]).replace(**held["changed"])
                   if held["name"] else probatum.Venue(**held["changed"]))
        assert rebuilt == venue


@pytest.mark.parametrize("given, actions, decode, account", [
    (dict(actions="simplex"),
     {"space": "simplex", "low": 0.0, "high": 1.0, "size": 3}, {"builtin": "simplex"},
     4),
    (dict(actions="discrete", max_leverage=3.0),
     {"space": "discrete", "choices": [3, 3], "max_leverage": 3.0},
     {"builtin": "discrete"}, 4),
    (dict(actions="orders", max_offset=5),
     {"space": "orders", "low": [-1.0, 0.0], "high": [1.0, 1.0], "max_leverage": 1.0,
      "max_offset": 5.0}, {"builtin": "orders"}, 6),
])
def test_each_kind_of_action_is_recorded_with_its_space(given, actions, decode, account):
    spec = probatum.env(universe(), VENUE, window=8, seed=1, **given).agent_spec

    assert (spec["actions"], spec["decode"])   == (actions, decode)
    assert len(spec["observation"]["account"]) == 2 * account + 4
    if given["actions"] == "orders":
        assert spec["observation"]["account"][4:6] == ["resting", "nearest"]


def test_a_sample_records_the_universe_it_draws_its_slots_from():
    spec = probatum.env(universe(), VENUE, window=8, sample=1, seed=1).agent_spec

    assert (spec["instruments"], spec["slots"], spec["sample"]) == (
        ["BTCUSDT", "ETHUSDT"], 1, 1)
    assert spec["observation"]["shape"] == [8, 1, 5]


def test_a_custom_feature_function_is_held_by_its_output_on_the_probe():
    data = universe()
    spec = probatum.env(data, VENUE, window=8, features=momentum, seed=1).agent_spec
    held = spec["features"]

    assert held["custom"] == "momentum" and held["width"] == 2
    assert held["source_hash"] == hashed(momentum)

    probe = _agent._probe(data.names, ["linear", "linear"], data.interval)
    assert held["reference"] == _agent._digest(momentum(probe))

    # a width is the features', whatever the number of instruments
    wider = probatum.env(data, VENUE, window=8, seed=1,
                         features=lambda data: np.dstack([momentum(data)] * 2)[:, :, :3])
    assert wider.agent_spec["features"]["width"] == 3

    # the same output from other source passes, another output is refused by both
    # references
    _agent._held(spec, features=rewritten)
    with pytest.raises(SpecError) as caught:
        _agent._held(spec, features=doubled)

    said = str(caught.value)
    assert said.startswith("features: doubled gives b2:") and held["reference"] in said

    # an array is recorded by its width alone, which the backtest's features must have
    arrayed = probatum.env(data, VENUE, window=8, features=momentum(data), seed=1)
    assert arrayed.agent_spec["features"] == {"array": 2}


def dated(data):
    """momentum, for bars before 2024 alone."""
    assert data.time[0] < 1_704_067_200_000_000_000, "trained on 2023's bars alone"
    return momentum(data)


def fixed(data):
    """momentum, cut to 300 bars wherever 512 are given."""
    return momentum(data)[:300] if len(data) == 512 else momentum(data)


def test_a_function_that_cannot_run_on_its_probe_leaves_no_reference_and_warns():
    with pytest.warns(ProbatumWarning, match="features: dated raised AssertionError on "
                                             "the agent spec's probe"):
        held = probatum.env(universe(), VENUE, window=8, features=dated, seed=1)

    assert held.agent_spec["features"]["reference"] is None

    with pytest.warns(ProbatumWarning, match=r"features: fixed gives shape \(300, 2, 2\) "
                                             r"on the agent spec's probe"):
        shaped = probatum.env(universe(), VENUE, window=8, features=fixed, seed=1)

    assert shaped.agent_spec["features"]["reference"] is None

    # a function passed again has nothing to be held to, which says so
    with pytest.warns(ProbatumWarning, match="features: the agent spec holds no "
                                             "reference for dated"):
        _agent._held(held.agent_spec, features=dated)


def test_the_probe_carries_each_instrument_s_market_as_the_venue_gives_it():
    seen = []

    def marketed(data):
        seen.append([each.market for each in data.instruments])
        return momentum(data)

    data = universe()
    bare = probatum.Data(data.bars, symbol=list(data.names), time=data.time)

    for given, trained in ((data, ["linear", "linear"]), (bare, [None, None])):
        seen.clear()
        spec = probatum.env(given, VENUE, window=8, features=marketed, seed=1).agent_spec
        assert spec["markets"] == ["linear", "linear"]
        # the training bars, as the Data holds them, then the probe, on the markets the
        # venue gave them
        assert seen == [trained, ["linear", "linear"]]


def halved(actions, state):
    return 0.5 * actions


@pytest.mark.parametrize("trained, features, decode, said", [
    (dict(), momentum, None,
     "features: momentum, and the agent trained on the candles; leave features out"),
    (dict(features=momentum), None, None,
     "features: none, and the agent trained on the function momentum; pass it as "
     "features="),
    (dict(features=momentum), np.zeros((300, 2, 2)), None,
     "features: an array, and the agent trained on the function momentum; pass it as "
     "features="),
    (dict(features=np.zeros((300, 2, 2))), momentum, None,
     "features: momentum, and the agent trained on an array of 2 features; pass the "
     "bars' own array as features="),
    (dict(features=np.zeros((300, 2, 2))), None, None,
     "features: none, and the agent trained on an array of 2 features; pass the bars' "
     "own array as features="),
    (dict(features=np.zeros((300, 2, 2))), np.zeros((300, 2, 3)), None,
     "features: an array of 3 features, and the agent trained on 2; pass the bars' own "
     "array of 2"),
    (dict(), None, halved,
     "decode: halved, and the agent's actions are decoded as box; leave decode out"),
    (dict(decode=halved), None, None,
     "decode: none, and the agent trained with halved; pass it as decode="),
])
def test_a_function_of_another_kind_or_none_is_refused_against_the_spec(trained, features,
                                                                       decode, said):
    spec = probatum.env(universe(), VENUE, window=8, seed=1, **trained).agent_spec

    with pytest.raises(SpecError) as caught:
        _agent._held(spec, features=features, decode=decode)

    assert str(caught.value).startswith(said)


def test_holding_a_decode_to_its_spec_never_loads_gymnasium():
    code = textwrap.dedent("""
        import sys
        import numpy as np
        from probatum import _agent

        def halved(actions, state):
            return 0.5 * actions

        probe = _agent._probe_actions((1,), "box", np.array([-1.0]), None)
        spec  = {"slots": 1, "features": {"builtin": "candles"},
                 "actions": {"space": "box", "low": [-1.0], "high": 1.0},
                 "decode": {"custom": "halved", "reference": _agent._digest(0.5 * probe)}}
        _agent._held(spec, decode=halved)
        print("gymnasium" in sys.modules)
    """)

    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "False"


def test_a_custom_decode_is_held_by_its_output_on_the_probe_actions():
    def halved(actions, state):
        return 0.5 * actions

    def quartered(actions, state):
        return 0.25 * actions

    spec = probatum.env(universe(), VENUE, window=8, decode=halved, seed=1).agent_spec
    held = spec["decode"]

    assert held["custom"] == ("test_a_custom_decode_is_held_by_its_output_on_the_probe_"
                              "actions.<locals>.halved")

    probe = _agent._probe_actions((2,), "box", np.array([-1.0, -1.0]), None)
    assert held["reference"] == _agent._digest(0.5 * probe)

    _agent._held(spec, decode=lambda actions, state: actions / 2.0)
    with pytest.raises(SpecError) as caught:
        _agent._held(spec, decode=quartered)

    assert str(caught.value).startswith("decode: test_a_custom_decode_is_held_by_its_"
                                        "output_on_the_probe_actions.<locals>.quartered "
                                        "gives b2:")


def test_a_decode_meets_its_probe_once_and_only_when_the_spec_is_first_read():
    calls = []

    def counted(actions, state):
        calls.append(len(actions))
        return np.zeros((len(actions), 2))

    env = probatum.env(universe(), VENUE, window=8, num_envs=3, decode=counted, seed=1)

    env.reset()
    env.step(np.zeros((3, 2), dtype=np.float32))

    assert calls == [3]

    first, again = env.agent_spec, env.agent_spec
    assert calls == [3, 8] and first == again


def test_a_custom_reward_is_recorded_by_its_name_and_source():
    def paid(state):
        return state.log_return

    spec = probatum.env(universe(), VENUE, window=8, reward=paid, seed=1).agent_spec

    assert spec["reward"] == {"custom": paid.__qualname__, "source_hash": hashed(paid)}

    sharpe = probatum.env(universe(), VENUE, window=8, reward="sharpe", seed=1)
    assert sharpe.agent_spec["reward"] == {
        "builtin": "sharpe",
        "floor":   0.01,
        "eta":     2.0 / 9.0,
    }


def test_the_probe_bars_are_the_reference_s_to_the_bit():
    names = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    probe = _agent._probe(names, ["linear", "spot", "inverse"], "15m")

    assert probe.names == tuple(names) and probe.interval == "15m"
    assert [each.market for each in probe.instruments] == ["linear", "spot", "inverse"]
    assert probe.time[0] == 1_704_067_200_000_000_000
    assert (np.diff(probe.time) == 900_000_000_000).all()

    held = np.array(environment.probe(3)).transpose(1, 0, 2)
    assert probe.bars.shape == (512, 3, 5)
    assert np.array_equal(probe.bars, held)


@pytest.mark.parametrize("space, kind, lows, choices", [
    ((2,), "box", [-1.0, 0.0], None),
    ((3,), "simplex", [-1.0, -1.0], None),
    ((), "discrete", [-1.0], [3]),
    ((2,), "discrete", [-1.0, 0.0], [3, 2]),
])
def test_the_probe_actions_are_the_reference_s_to_the_bit(space, kind, lows, choices):
    held = _agent._probe_actions(space, kind, np.array(lows),
                                 None if choices is None else np.array(choices))

    if kind == "box":
        components = [("box", low, 1.0) for low in lows]
    elif kind == "simplex":
        components = [("simplex",)] * (len(lows) + 1)
    else:
        components = [("discrete", n) for n in choices]

    assert held.shape            == (8,) + space
    assert held.ravel().tolist() == np.ravel(environment.probe_actions(components)).tolist()
    assert held.dtype            == (np.int64 if kind == "discrete" else np.float64)


def test_a_reference_is_the_reference_s_hash_of_the_values_rounded_to_float32():
    values = np.array([[1.0, -0.0, np.nan], [1e-50, -1e-50, 3.0e38 * 10.0]])
    held   = _agent._digest(values)

    assert held == environment.digest(values.shape, values.ravel())
    # a negative zero and a value float32 rounds to one are +0.0, as is a NaN's payload
    assert held == _agent._digest(np.array([[1.0, 0.0, -np.nan], [0.0, 0.0, np.inf]]))
    assert held != _agent._digest(values.T)


def shown_by_the_rule(onsets, window, printed, delisted):
    rows = len(printed)
    out  = np.zeros(printed.shape, dtype=bool)

    for i, e in enumerate(onsets):
        if e is None:
            continue

        for t in range(rows):
            out[t, i] = t + 1 >= e + window and printed[t, i] and not delisted[t, i]

    return out


@pytest.mark.parametrize("seed", [3, 8, 13])
def test_the_normalisation_statistics_are_the_reference_s(seed):
    # a universe of three instruments, each with bars it never printed, one listed late
    # and one delisted, and features that warm up
    rng      = np.random.default_rng(seed)
    rows, n  = 60, 3

    close   = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (rows, n)), axis=0))
    candles = np.stack([close * (1.0 + rng.normal(0.0, 0.002, (rows, n))),
                        close * 1.01, close * 0.99, close,
                        rng.uniform(0.0, 1e3, (rows, n))], axis=2)

    printed = rng.random((rows, n)) > 0.15
    printed[:10, 1]      = False
    candles[~printed]    = np.nan
    candles[20:30, 2, 4] = 0.0

    delisted = np.zeros((rows, n), dtype=bool)
    delisted[45:, 0] = True

    # the second feature parked at one value, whose mean its sum does not hold exactly
    values = rng.normal(0.0, 1.0, (rows, n, 2))
    values[:, :, 1]  = 0.1
    values[:12, 2]   = np.nan
    values[~printed] = np.nan

    window  = 5
    columns = [[Bar(*candles[t, i]) if printed[t, i] else None for t in range(rows)]
               for i in range(n)]
    rowsof  = [[list(values[t, i]) for i in range(n)] for t in range(rows)]

    for features, held in ((None, None), (rowsof, values)):
        onsets = environment.onsets(columns, features)
        shown  = shown_by_the_rule(onsets, window, printed, delisted)
        want   = environment.moments(columns, features, window,
                                     lambda i, t, shown=shown: bool(shown[t, i]))
        got    = _agent._moments(held, candles, printed, shown, window)

        assert np.allclose(got, want, rtol=1e-10, atol=1e-13)
        if features is not None:
            assert got[1][1] == want[1][1] == 1.0


def test_a_feature_that_never_varies_or_holds_no_value_is_scaled_by_one():
    def steady(data):
        out = momentum(data)
        out[:, :, 1] = 3.0
        return out

    env  = probatum.env(universe(), VENUE, window=8, features=steady, normalise=True,
                        seed=1)
    held = env.agent_spec["normalisation"]

    assert (held["mean"][1], held["deviation"][1]) == (3.0, 1.0)

    obs, _ = env.reset()
    assert (obs[0, 1:8 * 2 * 2:2] == 0.0).all()

    hidden  = np.zeros((4, 1), dtype=bool)
    nothing = _agent._moments(np.ones((4, 1, 2)), None, ~hidden, hidden, 2)
    assert nothing == [(0.0, 1.0), (0.0, 1.0)]


def test_a_feature_parked_at_one_value_is_written_as_zero_over_a_deviation_of_one():
    def parked(data):
        out = momentum(data)
        out[:, :, 1] = 0.0001
        return out

    env  = probatum.env(universe(), VENUE, window=8, features=parked, normalise=True,
                        num_envs=4, seed=1)
    held = env.agent_spec["normalisation"]

    assert held["deviation"][1] == 1.0
    assert held["mean"][1]      == pytest.approx(0.0001, rel=1e-12)

    obs, _ = env.reset()
    assert np.abs(obs[:, 1:8 * 2 * 2:2]).max() < 1e-12


def level(data):
    """Each bar's close in thousands and its volume in tens of thousands, NaN on a bar
    its instrument did not print."""
    return np.stack([data.closes / 1e3, data.volumes / 1e4], axis=2)


@pytest.mark.parametrize("features", [None, level])
@pytest.mark.parametrize("chosen", [["BTCUSDT", "ETHUSDT"], ["BTCUSDT"]])
def test_a_slice_s_statistics_are_the_reference_s_over_the_rows_its_run_steps(features,
                                                                             chosen):
    # bars each instrument left unprinted, a selection whose run never steps the rows
    # only the other printed, and a slice whose bars before it no observation reads
    rng   = np.random.default_rng(21)
    rows  = 120
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (rows, 2)), axis=0))
    bars  = np.stack([close, close * 1.01, close * 0.99, close,
                      rng.uniform(1e3, 1e4, (rows, 2))], axis=2)
    bars[rng.random((rows, 2)) < 0.1] = np.nan
    times = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * HOUR
    whole = probatum.Data(
        bars,
        symbol = ["BTCUSDT", "ETHUSDT"],
        market = "linear",
        time   = times,
    )

    data = whole.select(chosen)
    got  = probatum.env(data[30:], VENUE, window=6, features=features, normalise=True,
                        seed=1).agent_spec["normalisation"]

    n       = len(chosen)
    stepped = np.flatnonzero(data.available.any(axis=1))
    first   = int((stepped < 30).sum())
    seen    = data.available[stepped]
    columns = [[Bar(*data.bars[r, i]) if seen[k, i] else None
                for k, r in enumerate(stepped)] for i in range(n)]
    held    = None if features is None else [[list(level(data)[r, i]) for i in range(n)]
                                             for r in stepped]

    # an instrument's onset is its first printed bar of the slice
    onsets = [first + int(np.argmax(seen[first:, i])) for i in range(n)]
    shown  = shown_by_the_rule(onsets, 6, seen, np.zeros(seen.shape, dtype=bool))
    want   = environment.moments(columns, held, 6, lambda i, t: bool(shown[t, i]))

    assert np.allclose(list(zip(got["mean"], got["deviation"])), want, rtol=1e-10,
                       atol=1e-13)


def test_a_normalised_observation_writes_each_market_value_over_its_spec_s_scale():
    data  = universe()
    plain = probatum.env(data, VENUE, window=8, features=momentum, num_envs=4, seed=7)
    held  = probatum.env(data, VENUE, window=8, features=momentum, normalise=True,
                         num_envs=4, seed=7)
    scale = held.agent_spec["normalisation"]

    raw, _  = plain.reset()
    seen, _ = held.reset()

    market  = 8 * 2 * 2
    mean    = np.tile(scale["mean"], market // 2)
    spread  = np.tile(scale["deviation"], market // 2)
    written = raw[:, :market] != 0.0
    want    = (raw[:, :market].astype(np.float64) - mean) / spread

    assert np.allclose(seen[:, :market][written], want[written], rtol=1e-5, atol=1e-6)
    assert (seen[:, :market][~written] == 0.0).all()
    # the account block is never scaled
    assert np.array_equal(seen[:, market:], raw[:, market:])

    # and the statistics are each feature's over every value the observations hold
    onsets = [1, 1]
    shown  = shown_by_the_rule(onsets, 8, data.available, np.zeros((300, 2), bool))
    values = momentum(data)
    want   = _agent._moments(values, data.bars, data.available, shown, 8)

    assert scale == {"mean": [m for m, _ in want], "deviation": [d for _, d in want]}


def test_a_feature_inside_a_billionth_of_one_value_takes_its_smallest_as_its_mean():
    # values a hair apart, whose sum's mean is not the smallest, and values far apart
    # whose squares underflow to a deviation of zero
    values  = np.array([[[1.0, 0.0]], [[1.0 + 1e-12, 1e-200]], [[1.0, 0.0]]])
    printed = np.ones((3, 1), dtype=bool)
    held    = _agent._moments(values, None, printed, printed, 1)

    assert held[0]    == (1.0, 1.0)
    assert held[1][1] == 1.0

    rows = [[list(values[t, 0])] for t in range(3)]
    want = environment.moments([[Bar(1.0, 1.0, 1.0, 1.0, 1.0)] * 3], rows, 1,
                               lambda i, t: True)
    assert want[0] == (1.0, 1.0) and want[1][1] == 1.0


def test_a_decode_that_cannot_run_on_its_probe_leaves_no_reference_and_warns_once():
    def paired(actions, state):
        assert len(actions) == 2, "decodes two environments at once"
        return 0.5 * actions

    env = probatum.env(universe(), VENUE, window=8, num_envs=2, decode=paired, seed=1)

    with pytest.warns(ProbatumWarning, match="decode: .*paired raised AssertionError on "
                                             "the agent spec's probe actions"):
        spec = env.agent_spec

    assert spec["decode"]["reference"] is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert env.agent_spec == spec

    with pytest.warns(ProbatumWarning, match="decode: the agent spec holds no reference "
                                             "for .*paired"):
        _agent._held(spec, decode=paired)


def test_a_function_passed_again_that_cannot_run_on_its_probe_is_refused():
    spec = probatum.env(universe(), VENUE, window=8, features=momentum, seed=1).agent_spec

    with pytest.raises(SpecError) as caught:
        _agent._held(spec, features=dated)

    said = str(caught.value)
    assert said.startswith("features: dated raised AssertionError on the agent spec's "
                           "probe") and spec["features"]["reference"] in said

    halving = probatum.env(universe(), VENUE, window=8, decode=halved, seed=1).agent_spec

    def broken(actions, state):
        raise ValueError("no state here")

    with pytest.raises(SpecError) as caught:
        _agent._held(halving, decode=broken)

    assert str(caught.value).startswith("decode: test_a_function_passed_again_that_"
                                        "cannot_run_on_its_probe_is_refused.<locals>."
                                        "broken raised ValueError on the agent spec's "
                                        "probe actions")

    with pytest.raises(SpecError) as caught:
        _agent._held(spec, features=momentum, decode="halved")

    assert str(caught.value).startswith("decode: an object of type str, and the agent's "
                                        "actions are decoded as box")


def widening(data):
    """momentum, twice over wherever 512 bars are given."""
    return np.dstack([momentum(data)] * (2 if len(data) == 512 else 1))


def test_a_width_is_the_training_bars_whatever_the_probe_gives():
    with pytest.warns(ProbatumWarning):
        held = probatum.env(universe(), VENUE, window=8, features=dated, seed=1)

    assert held.agent_spec["features"]["width"] == 2

    with pytest.warns(ProbatumWarning, match=r"features: widening gives shape "
                                             r"\(512, 2, 4\) on the agent spec's probe, "
                                             r"where the training bars gave 2 features"):
        wide = probatum.env(universe(), VENUE, window=8, features=widening, seed=1)

    assert wide.agent_spec["features"] == {
        "custom":      "widening",
        "width":       2,
        "source_hash": hashed(widening),
        "reference":   None,
    }


def test_a_custom_decode_records_no_leverage_it_does_not_scale():
    spec = probatum.env(universe(), VENUE, window=8, decode=halved, max_leverage=3.0,
                        seed=1).agent_spec

    assert spec["actions"] == {"space": "box", "low": [-1.0, -1.0], "high": 1.0}
