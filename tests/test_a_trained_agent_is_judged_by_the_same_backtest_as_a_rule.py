"""A trained agent is judged by the same backtest as a rule.

backtest runs an agent as one environment of the spec it trained with, over the Data
from its first bar, its window warmed on the bars before a slice, asking it at each
close for its action on the observation training showed it. What it decided, played
again as weights, gives its equity to the bit, and its frictionless baseline is the
agent run again without costs. A function it trained with is held to its probe when it
is passed again, a Data of other instruments is refused, and bars that overlap its
training span, another venue or another interval warn (Specification, An agent in a
backtest; ADR 0039).
"""

import copy
import json
import math
import re
import warnings

import gymnasium
import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import (DataError, ProbatumAttributeError, ProbatumKeyError,
                      ProbatumTypeError, ProbatumValueError, ProbatumWarning, SpecError,
                      VenueError)
from reference import environment
from test_an_agent_carries_the_spec_it_trained_with import doubled, momentum
from test_an_episode_runs_again_as_a_backtest_from_its_start import (HOUR, VENUE,
                                                                     universe, walked)


class Agent:
    """An agent that acts at random from a stream of its own, keeping each observation
    it is shown and each action it took."""


    def __init__(self, space, seed=5):
        self.space = copy.deepcopy(space)
        self.space.seed(seed)
        self.seen  = []
        self.acted = []


    def __call__(self, observation):
        self.seen.append(observation)
        self.acted.append(np.asarray(self.space.sample())[None])
        return self.acted[-1]


class Steady:
    """An agent that asks for the same weights at every close, beside a value as a
    policy that returns one does, counting how often it is asked."""


    def __init__(self):
        self.weights = np.array([[0.5, -0.25]], dtype=np.float32)
        self.asked   = 0


    def __call__(self, observation):
        self.asked += 1
        return self.weights, None


def tilted(observation):
    """An action read from the market block alone, so what it decides is the same
    whatever the account holds."""
    return np.tanh(50.0 * observation[:, :2])


def tanh(actions, state):
    return np.tanh(2.0 * actions)


def gapped(data):
    """momentum, with ETHUSDT's first feature lost two bars before bar 150."""
    out = momentum(data)
    if len(out) > 148:
        out[148, 1, 0] = np.nan
    return out


def unprobed(data):
    """momentum, which cannot run on the agent spec's probe of 512 bars."""
    if len(data) == 512:
        raise ValueError("no probe")
    return momentum(data)


def widened(data):
    """unprobed with a third feature."""
    out = unprobed(data)
    return np.concatenate([out, out[:, :, :1]], axis=2)


def relative(data):
    """Each instrument's close over the mean close of every instrument at the bar, a
    feature only the whole universe gives."""
    closes = data.closes
    return (closes / np.nanmean(closes, axis=1, keepdims=True))[:, :, None]


def paired(step=HOUR, rows=300, seed=3, market="linear"):
    """universe's two instruments on a clock of ``step``, on ``market``."""
    rng   = np.random.default_rng(seed)
    times = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * step
    pair  = [probatum.Data(walked(price, rows, rng), symbol=name, market=market,
                           time=times)
             for name, price in (("BTCUSDT", 30_000.0), ("ETHUSDT", 2_000.0))]
    return pair[0].join(pair[1])


def coined(rows=300):
    """An inverse contract of 100 dollars on universe's clock, as the router gives one."""
    close  = 20_000.0 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0, 0.01, rows)))
    opened = np.r_[close[0], close[:-1]]
    times  = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * HOUR
    frame  = pd.DataFrame({
        "open":   opened,
        "high":   np.maximum(opened, close) * 1.01,
        "low":    np.minimum(opened, close) * 0.99,
        "close":  close,
        "volume": 1e6,
    }, index=pd.to_datetime(times, utc=True))
    frame.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", warnings=[], volume_unit="contract",
                       contract_size=100.0)

    return probatum.Data(frame)


def halving(rows=16):
    """Bars that each open where the last closed and halve inside themselves."""
    opened = 100.0 * np.power(0.5, np.arange(rows, dtype=np.float64))
    bars   = np.stack([opened, opened, opened / 2, opened / 2, np.full(rows, 1e9)], axis=1)
    return probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(rows, dtype=np.int64) * HOUR,
    )


@pytest.mark.parametrize("settings", [
    dict(),
    dict(actions="simplex", normalise=True, tolerance=0.05),
    dict(actions="discrete", features=momentum, max_leverage=1.5),
    dict(features=momentum, normalise=True, layout="dict", max_leverage=3.0),
    dict(decode=tanh),
], ids=["box", "simplex", "discrete", "dict", "decode"])
def test_what_an_agent_decided_played_again_as_weights_gives_its_equity_to_the_bit(
        settings):
    data   = universe()
    test   = data[150:]
    env    = probatum.env(data[:150], VENUE, window=8, seed=1, **settings)
    agent  = Agent(env.single_action_space)
    result = probatum.backtest(agent, test, VENUE, spec=env,
                               features=settings.get("features"),
                               decode=settings.get("decode"))

    # asked at each close but the last, and again by the run without costs
    assert len(agent.seen) == 2 * (len(test) - 1)
    assert result.policy.shape == (len(test), 2) and np.isnan(result.policy[-1]).all()

    again = probatum.backtest(result.policy, test, VENUE,
                              tolerance=settings.get("tolerance", 0.0))

    assert np.array_equal(again.equity, result.equity)
    assert np.array_equal(again.trades["pnl"], result.trades["pnl"])
    assert len(result.fills) > 100

    if "decode" in settings:
        decoded = np.tanh(2.0 * np.concatenate(agent.acted[:149])).astype(np.float64)
        assert np.array_equal(result.policy[:-1], decoded)


@pytest.mark.parametrize("settings", [
    dict(normalise=True),
    dict(features=momentum, normalise=True),
    dict(features=momentum, layout="dict", actions="orders"),
], ids=["candles", "function", "dict"])
def test_an_agent_is_shown_the_observation_training_showed_it_at_the_same_bar(settings):
    data = universe()
    env  = probatum.env(data[:150], VENUE, window=8, seed=4, **settings)
    first, info = env.reset()
    start = int(info["start"][0])
    agent = Agent(env.single_action_space)

    with pytest.warns(ProbatumWarning, match="overlap the agent's training span"):
        probatum.backtest(agent, data[start:], VENUE, spec=env,
                          features=settings.get("features"))

    seen = agent.seen[0]
    if isinstance(seen, dict):
        assert all(np.array_equal(seen[key], first[key]) for key in first)
    else:
        assert np.array_equal(seen, first)


@pytest.mark.parametrize("features", [None, momentum])
def test_an_agent_decides_at_a_slice_s_first_close_on_the_bars_before_it(features):
    data  = universe()
    env   = probatum.env(data[:100], VENUE, window=8, features=features, seed=1)
    whole = Agent(env.single_action_space)

    with pytest.warns(ProbatumWarning, match="overlap the agent's training span"):
        probatum.backtest(whole, data, VENUE, spec=env, features=features)

    part   = Agent(env.single_action_space)
    result = probatum.backtest(part, data[150:], VENUE, spec=env, features=features)
    market = 8 * 2 * (5 if features is None else 2)

    assert np.array_equal(part.seen[0][0, :market], whole.seen[150][0, :market])
    assert part.seen[0][0, market + 3] == part.seen[0][0, market + 7] == 1.0
    assert not np.isnan(result.policy[0]).any()


def test_a_feature_lost_before_a_slice_keeps_its_instrument_out_until_the_window_passes():
    data   = universe()
    env    = probatum.env(data[:100], VENUE, window=8, features=gapped, seed=1)
    result = probatum.backtest(Agent(env.single_action_space), data[150:], VENUE, spec=env,
                               features=gapped)

    # lost at bar 148, ETHUSDT's window holds no lost value from bar 156 on
    assert np.isnan(result.policy[:6, 1]).all() and not np.isnan(result.policy[6:-1]).any()
    assert not np.isnan(result.policy[:6, 0]).any()

    with (pytest.warns(ProbatumWarning, match="overlap the agent's training span"),
          pytest.raises(DataError, match=r"features: NaN at bar 148 \(2023-11-21 ")):
        probatum.backtest(Agent(env.single_action_space), data, VENUE, spec=env,
                          features=gapped)

    # trained with a fill, the agent's run fills the value as its training did
    filled = probatum.env(data[:150], VENUE, window=8, features=gapped, fill="previous",
                          seed=1)
    with pytest.warns(ProbatumWarning, match="overlap the agent's training span"):
        result = probatum.backtest(Agent(filled.single_action_space), data, VENUE,
                                   spec=filled, features=gapped)

    assert not np.isnan(result.policy[8:-1]).any()


def test_a_changed_feature_function_is_refused():
    data  = universe()
    env   = probatum.env(data[:150], VENUE, window=8, features=momentum, seed=1)
    agent = Agent(env.single_action_space)
    held  = env.agent_spec["features"]["reference"]

    with pytest.raises(SpecError, match=rf"features: doubled gives b2:[0-9a-f]{{16}} on "
                                        rf"its probe, and the agent trained on momentum, "
                                        rf"which gave {held}; pass the function it "
                                        rf"trained with"):
        probatum.backtest(agent, data[150:], VENUE, spec=env, features=doubled)

    with pytest.raises(SpecError, match="features: none, and the agent trained on the "
                                        "function momentum; pass it as features="):
        probatum.backtest(agent, data[150:], VENUE, spec=env)

    assert not agent.seen

    probatum.backtest(agent, data[150:], VENUE, spec=env, features=momentum)

    assert len(agent.seen) == 2 * 149


def test_an_agent_s_spec_is_read_from_itself_its_file_its_directory_or_its_environment(
        tmp_path):
    data   = universe()
    env    = probatum.env(data[:150], VENUE, window=8, seed=1)
    single = probatum.rl.Env(data[:150], VENUE, window=8, seed=1)
    made   = gymnasium.make("probatum/Trading-v0", data=data[:150], venue=VENUE, window=8,
                            seed=1)
    spec   = env.agent_spec
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    runs = [probatum.backtest(Agent(env.single_action_space), data[150:], VENUE,
                              spec=given)
            for given in (spec, tmp_path / "spec.json", str(tmp_path), env, single, made)]

    for result in runs:
        assert np.array_equal(result.equity, runs[0].equity)
        assert result.provenance["spec"] == spec
    assert runs[0].provenance["policy"] == "the callable Agent"


REFUSED = [
    (dict(spec=None), SpecError,
     "spec: this agent carries no spec; pass spec= with the env it trained on or its "
     "JSON"),
    (dict(spec={"schema": 2}), SpecError,
     "spec: schema 2, and this probatum reads schema 1; pass a spec env.agent_spec "
     "recorded"),
    (dict(spec=7), ProbatumTypeError,
     "spec: an object of type int; pass the agent spec, a path to its JSON or the "
     "environment the agent trained in"),
    (dict(stop=0.04), ProbatumValueError,
     "stop: 0.04 beside an agent, which places its own trades; leave it out"),
    (dict(take_profit=0.04), ProbatumValueError,
     "take_profit: 0.04 beside an agent, which places its own trades; leave it out"),
    (dict(sizing="quantity"), ProbatumValueError,
     "sizing: 'quantity' beside an agent, whose spec says what its actions trade; leave "
     "it out"),
    (dict(tolerance=0.01), ProbatumValueError,
     "tolerance: 0.01 beside an agent, which trades with the tolerance its spec holds; "
     "leave it out"),
    (dict(risk_free=math.inf), ProbatumValueError,
     "risk_free: inf; give the annual rate as a finite number, such as 0.04"),
    (dict(instruments=["BTCUSDT"]), SpecError,
     "instruments: 1 instrument, BTCUSDT, and the agent trades 2 slots; name 2 of them "
     "with instruments="),
    (dict(instruments=["ETHUSDT", "BTCUSDT"]), SpecError,
     "instruments: ETHUSDT, BTCUSDT, and the agent trained on BTCUSDT, ETHUSDT, in that "
     "order; name them with instruments="),
    (dict(instruments=["XRPUSDT", "BTCUSDT"]), ProbatumKeyError,
     "instruments: no instrument 'XRPUSDT'; use some of BTCUSDT, ETHUSDT"),
    (dict(features=momentum), SpecError,
     "features: momentum, and the agent trained on the candles; leave features out"),
    (dict(decode=tanh), SpecError,
     "decode: tanh, and the agent's actions are decoded as box; leave decode out"),
    (dict(venue=VENUE.replace(intrabar="random")), VenueError,
     "venue: intrabar is \"random\", which backtest refuses so that a backtest repeats "
     "without a seed; replace(intrabar=\"pessimistic\") or \"optimistic\""),
]


@pytest.mark.parametrize("given, error, says", REFUSED)
def test_what_an_agent_s_backtest_cannot_run_is_refused_naming_the_fix(given, error, says):
    data     = universe()
    env      = probatum.env(data[:150], VENUE, window=8, seed=1)
    agent    = Agent(env.single_action_space)
    settings = {"spec": env, **given}
    venue    = settings.pop("venue", VENUE)

    with pytest.raises(error, match=f"^{re.escape(says)}$"):
        probatum.backtest(agent, data[150:], venue, **settings)

    assert not agent.seen


def test_a_spec_file_that_cannot_be_read_is_refused_naming_it(tmp_path):
    data  = universe()
    agent = Agent(probatum.env(data[:150], VENUE, window=8, seed=1).single_action_space)

    with pytest.raises(SpecError, match=r"spec: '.*none\.json' cannot be read \(No such "
                                        r"file or directory\); pass the spec.json "
                                        r"probatum.save wrote, or its directory"):
        probatum.backtest(agent, data[150:], VENUE, spec=tmp_path / "none.json")

    (tmp_path / "spec.json").write_text("{", encoding="utf-8")
    with pytest.raises(SpecError, match=r"spec: '.*spec\.json' holds no JSON \(.*\); pass "
                                        r"the spec.json probatum.save wrote"):
        probatum.backtest(agent, data[150:], VENUE, spec=tmp_path)


class Hold(probatum.Strategy):

    def next(self, bar, account):
        pass


def test_an_agent_s_settings_beside_a_policy_that_is_no_agent_are_refused():
    data    = universe()
    env     = probatum.env(data[:150], VENUE, window=8, seed=1)
    weights = np.full((len(data), 2), np.nan)

    with pytest.raises(ProbatumValueError, match="^spec: an object of type VectorEnv "
                                                 "beside an array, which is no agent; "
                                                 "leave it out$"):
        probatum.backtest(weights, data, VENUE, spec=env)

    with pytest.raises(ProbatumValueError, match="^features: a function beside a "
                                                 "Strategy, which is no agent; leave it "
                                                 "out$"):
        probatum.backtest(Hold(), data, VENUE, features=momentum)

    with pytest.raises(ProbatumValueError, match=r"^instruments: \['BTCUSDT'\] beside an "
                                                 r"array, which is no agent; leave it "
                                                 r"out$"):
        probatum.backtest(weights, data, VENUE, instruments=["BTCUSDT"])


def test_an_agent_s_instruments_are_taken_from_the_data_in_the_order_it_trained_on():
    data   = universe()
    env    = probatum.env(data[:150], VENUE, window=8, features=momentum, seed=1)
    turned = data.select(["ETHUSDT", "BTCUSDT"])

    with pytest.raises(SpecError, match="^data: ETHUSDT, BTCUSDT, and the agent trained "
                                        "on BTCUSDT, ETHUSDT, in that order; name them "
                                        "with instruments=$"):
        probatum.backtest(Agent(env.single_action_space), turned[150:], VENUE, spec=env,
                          features=momentum)

    own, held = Agent(env.single_action_space), Agent(env.single_action_space)
    probatum.backtest(own, data[150:], VENUE, spec=env, features=momentum)
    probatum.backtest(held, turned[150:], VENUE, spec=env, features=momentum,
                      instruments=["BTCUSDT", "ETHUSDT"])

    assert np.array_equal(np.concatenate(held.seen), np.concatenate(own.seen))

    # an array of features is over the Data given, its slots taken in the agent's order
    array = probatum.env(data[:150], VENUE, window=8, features=momentum(data)[:150],
                         seed=1)
    plain, taken = Agent(env.single_action_space), Agent(env.single_action_space)
    probatum.backtest(plain, data[150:], VENUE, spec=array, features=momentum(data)[150:])
    probatum.backtest(taken, turned[150:], VENUE, spec=array,
                      features=momentum(turned)[150:], instruments=["BTCUSDT", "ETHUSDT"])

    assert np.array_equal(np.concatenate(taken.seen), np.concatenate(plain.seen))

    with pytest.raises(DataError, match=r"^features: shape \(150, 1, 2\) for a Data of 150 "
                                        r"bars of 2 instruments; give \(150, 2, F\) over "
                                        r"it, whose slots instruments= takes$"):
        probatum.backtest(Agent(env.single_action_space), turned[150:], VENUE, spec=array,
                          features=momentum(turned)[150:, :1],
                          instruments=["BTCUSDT", "ETHUSDT"])

    with pytest.raises(DataError, match=r"^features: shape \(151, 2, 2\) for a Data of 150 "
                                        r"bars of 2 instruments; give \(150, 2, F\) over "
                                        r"it, whose slots instruments= takes$"):
        probatum.backtest(Agent(env.single_action_space), turned[150:], VENUE, spec=array,
                          features=momentum(turned)[149:],
                          instruments=["BTCUSDT", "ETHUSDT"])


def test_a_sampled_agent_trades_the_instruments_named_as_its_slots():
    data  = universe()
    env   = probatum.env(data[:150], VENUE, window=8, sample=1, seed=1)
    agent = Agent(env.single_action_space)

    with pytest.raises(SpecError, match="^data: 2 instruments, BTCUSDT, ETHUSDT, and the "
                                        "agent trades 1 slot; name 1 of them with "
                                        "instruments=$"):
        probatum.backtest(agent, data[150:], VENUE, spec=env)

    result = probatum.backtest(agent, data[150:], VENUE, spec=env, instruments=["ETHUSDT"])

    assert result.policy.shape == (150, 1)

    again = probatum.backtest(result.policy, data[150:].select(["ETHUSDT"]), VENUE)

    assert np.array_equal(again.equity, result.equity)


def test_a_sampled_agent_s_features_are_worked_over_the_universe_as_in_training():
    # its function saw every instrument in training, of which its slot is one
    data  = universe()
    env   = probatum.env(data[:150], VENUE, window=8, sample=1, features=relative, seed=1)
    agent = Agent(env.single_action_space)

    probatum.backtest(agent, data[150:], VENUE, spec=env, features=relative,
                      instruments=["ETHUSDT"])

    # the first close's window, the eight bars to the slice's first, of ETHUSDT's column
    window = relative(data)[143:151, 1, 0].astype(np.float32)
    assert agent.seen[0][0, :8].tolist() == window.tolist() != [1.0] * 8


def test_an_agent_trades_each_slot_on_the_market_it_trained_on():
    venue = probatum.venues.binance.replace(funding="none")
    perps = paired()
    env   = probatum.env(perps[:150], venue, window=8, seed=1)
    spots = paired(market="spot")

    with pytest.raises(SpecError, match="^data: BTCUSDT is spot at binance, and the agent "
                                        "trained on it as linear; give it the market it "
                                        "trained on$"):
        probatum.backtest(Agent(env.single_action_space), spots[150:], venue, spec=env)

    # a Data that holds a symbol on two markets names each by its market
    both = spots.join(perps)
    own, named = Agent(env.single_action_space), Agent(env.single_action_space)
    probatum.backtest(own, perps[150:], venue, spec=env)
    probatum.backtest(named, both[150:], venue, spec=env,
                      instruments=["linear:BTCUSDT", "linear:ETHUSDT"])

    assert np.array_equal(np.concatenate(named.seen), np.concatenate(own.seen))

    # the same instruments in another order are named again, not read on their markets
    wallets = venue.replace(balance={"spot": 5_000.0, "margin": 5_000.0})
    mixed   = probatum.env(both[:150], wallets, window=8, seed=1)

    with pytest.raises(SpecError, match="^data: linear:BTCUSDT, linear:ETHUSDT, "
                                        "spot:BTCUSDT, spot:ETHUSDT, and the agent trained "
                                        "on spot:BTCUSDT, spot:ETHUSDT, linear:BTCUSDT, "
                                        "linear:ETHUSDT, in that order; name them with "
                                        "instruments=$"):
        probatum.backtest(Agent(mixed.single_action_space), perps.join(spots)[150:],
                          wallets, spec=mixed)


def test_a_sweep_of_an_agent_opens_and_measures_the_account_of_its_slots():
    venue = probatum.venues.binance.replace(funding="none")
    perps = paired()
    env   = probatum.env(perps[:150], venue, window=8, seed=1)

    # an inverse contract first, whose notional a sweep of the slots never counts
    both   = coined().join(paired(market="spot")).join(perps)
    slots  = ["linear:BTCUSDT", "linear:ETHUSDT"]
    agent  = dict(spec=env, instruments=slots)
    steady = Steady()

    # the slots trade perpetuals alone, whose account one balance opens
    sized = probatum.capacity(steady, both[150:], venue, [1_000.0, 50_000.0], **agent)

    # each level runs the agent once, with no run without costs beside it
    assert steady.asked == 2 * 149

    alone = probatum.capacity(Steady(), perps[150:], venue, [1_000.0, 50_000.0], spec=env)

    assert (sized.total_return_pct, sized.cost_bps) == (alone.total_return_pct,
                                                        alone.cost_bps)

    swept = probatum.cost_curve(Steady(), both[150:], venue, scales=[0.0, 1.0], **agent)

    assert swept.cost_bps == probatum.cost_curve(Steady(), perps[150:], venue,
                                                 scales=[0.0, 1.0], spec=env).cost_bps

    with pytest.warns(ProbatumWarning, match="still makes money there"):
        assert (probatum.breakeven_bps(Steady(), both[150:], venue, **agent)
                == probatum.breakeven_bps(Steady(), perps[150:], venue, spec=env))


def test_a_feature_function_of_another_width_is_refused_where_no_probe_holds_it():
    data = universe()
    with pytest.warns(ProbatumWarning, match="so the agent spec holds no reference"):
        env = probatum.env(data[:150], VENUE, window=8, features=unprobed, seed=1)

    with (pytest.warns(ProbatumWarning, match="holds no reference for unprobed"),
          pytest.raises(SpecError, match="^features: widened gives 3 features a bar, and "
                                         "the agent trained on 2; pass the function it "
                                         "trained with$")):
        probatum.backtest(Agent(env.single_action_space), data[150:], VENUE, spec=env,
                          features=widened)


def test_capacity_and_the_cost_sweeps_run_an_agent_given_its_spec_and_functions():
    data   = universe()
    env    = probatum.env(data[:150], VENUE, window=8, features=momentum, seed=1)
    agent  = dict(spec=env, features=momentum)
    result = probatum.backtest(tilted, data[150:], VENUE, **agent)

    # what tilted decides is the same whatever the account holds, so each level runs
    # as its decisions played again do
    sized = probatum.capacity(tilted, data[150:], VENUE, [1_000.0, 50_000.0], **agent)
    again = probatum.capacity(result.policy, data[150:], VENUE, [1_000.0, 50_000.0])

    assert sized.total_return_pct == again.total_return_pct

    swept  = probatum.cost_curve(tilted, data[150:], VENUE, scales=[0.0, 2.0], **agent)
    played = probatum.cost_curve(result.policy, data[150:], VENUE, scales=[0.0, 2.0])

    assert swept.total_return_pct == played.total_return_pct
    assert swept.total_return_pct[0] != swept.total_return_pct[1]
    assert (probatum.breakeven_bps(tilted, data[150:], VENUE, **agent)
            == probatum.breakeven_bps(result.policy, data[150:], VENUE))


def test_bars_that_overlap_the_training_span_another_venue_or_interval_warn():
    data = universe()
    env  = probatum.env(data[:150], VENUE, window=8, seed=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean = probatum.backtest(Agent(env.single_action_space), data[150:], VENUE,
                                  spec=env)

    assert (clean.provenance["overlap"], clean.provenance["trained_venue"]) == (None, True)

    with pytest.warns(ProbatumWarning, match="^data: bars from 2023-11-14 22:13:20 to "
                                             "2023-11-27 09:13:20, which overlap the "
                                             "agent's training span from 2023-11-14 "
                                             "22:13:20 to 2023-11-21 03:13:20; give bars "
                                             "after 2023-11-21 03:13:20 for a run out of "
                                             "sample$"):
        inside = probatum.backtest(Agent(env.single_action_space), data, VENUE, spec=env)

    assert inside.provenance["overlap"] == ["2023-11-14T22:13:20Z", "2023-11-21T03:13:20Z"]

    other = VENUE.replace(intrabar="optimistic")
    with pytest.warns(ProbatumWarning, match="^venue: binance_perp taken 2026-10-16 with "
                                             "intrabar, funding changed, and the agent "
                                             "trained at binance_perp taken 2026-10-16 "
                                             "with funding changed; the run judges it at "
                                             "a venue it did not learn$"):
        moved = probatum.backtest(Agent(env.single_action_space), data[150:], other,
                                  spec=env)

    assert moved.provenance["trained_venue"] is False

    with pytest.warns(ProbatumWarning, match="^data: 2h bars, and the agent trained on 1h "
                                             "bars, so its window of 8 spans another "
                                             "length of time; give 1h bars$"):
        probatum.backtest(Agent(env.single_action_space), paired(2 * HOUR)[200:], VENUE,
                          spec=env)


def test_an_agent_is_no_longer_asked_once_its_account_holds_nothing():
    plain = probatum.Venue(fees={"maker": 0.0, "taker": 0.0})
    data  = halving()
    env   = probatum.env(data[:8], plain, window=2, max_leverage=5.0, seed=0)
    asked = []

    def long(observation):
        asked.append(observation)
        return np.ones((1, 1), dtype=np.float32)

    result = probatum.backtest(long, data[8:], plain, spec=env)

    # asked at the first close of its run and of its run without costs, and no more
    assert result.bust and len(asked) == 2
    assert result.policy[0, 0] == 5.0 and np.isnan(result.policy[1:]).all()
    assert result.equity[1] <= 0.0

    # with orders, none is placed
    ticked = plain.replace(tick=0.0001)
    env    = probatum.env(data[:8], ticked, window=2, actions="orders", max_leverage=5.0,
                          seed=0)
    result = probatum.backtest(lambda observation: np.array([[[1.0, 0.0]]]), data[8:],
                               ticked, spec=env)

    assert result.bust and np.isnan(result.policy[1:]).all()
    assert (result.orders["bar"] <= 1).all()


def test_an_agent_s_frictionless_baseline_is_the_agent_run_again_without_costs_at_once():
    data   = universe()
    env    = probatum.env(data[:150], VENUE, window=8, seed=1)
    steady = Steady()
    result = probatum.backtest(steady, data[150:], VENUE, spec=env)

    assert steady.asked == 2 * 149

    # the agent changed after its backtest leaves the baseline the run at the call gave
    steady.weights = np.array([[-1.0, 1.0]], dtype=np.float32)
    replayed       = probatum.backtest(result.policy, data[150:], VENUE)

    assert np.array_equal(result.baselines["frictionless"],
                          replayed.baselines["frictionless"])
    assert not np.array_equal(result.baselines["frictionless"], result.equity)
    assert steady.asked == 2 * 149


def test_an_agent_run_at_the_account_level_keeps_no_decisions_and_no_baselines():
    data = universe()
    env  = probatum.env(data[:150], VENUE, window=8, seed=1)
    lean = probatum.backtest(Agent(env.single_action_space), data[150:], VENUE,
                             spec=env, record="account")
    full = probatum.backtest(Agent(env.single_action_space), data[150:], VENUE,
                             spec=env)

    assert lean.policy is None and np.array_equal(lean.equity, full.equity)

    with pytest.raises(ProbatumAttributeError, match="result.baselines"):
        lean.baselines
    with pytest.raises(ProbatumAttributeError, match="result.fills"):
        lean.fills


@pytest.mark.parametrize("layout", ["flat", "dict"])
def test_an_agent_sees_its_observation_through_the_sb3_statistics_its_spec_holds(layout):
    data = universe()
    env  = probatum.env(data[:150], VENUE, window=8, layout=layout, seed=1)
    spec = env.agent_spec
    raw  = Agent(env.single_action_space)

    probatum.backtest(raw, data[150:], VENUE, spec=spec)

    rng  = np.random.default_rng(3)
    size = spec["observation"]["size"]

    mean, var = rng.normal(0.0, 1.0, size), rng.uniform(0.5, 2.0, size)
    market    = 8 * 2 * 5

    if layout == "flat":
        spec["sb3"] = {
            "mean":    mean.tolist(),
            "var":     var.tolist(),
            "clip":    1.5,
            "epsilon": 1e-8,
        }
    else:
        # the account block left as it is, as VecNormalize leaves a key it is not given
        spec["sb3"] = {"market": {"mean": mean[:market].reshape(8, 2, 5).tolist(),
                                  "var":  var[:market].reshape(8, 2, 5).tolist()},
                       "clip": 1.5, "epsilon": 1e-8}

    seen = Agent(env.single_action_space)
    probatum.backtest(seen, data[150:], VENUE, spec=spec)

    for before, after in zip(raw.seen, seen.seen):
        if layout == "dict":
            assert np.array_equal(after["account"], before["account"])
            before, after = before["market"].reshape(1, -1), after["market"].reshape(1, -1)
        expected = [environment.sb3_normalised(float(x), m, v, 1e-8, 1.5)
                    for x, m, v in zip(before[0], mean, var)]
        assert after.dtype == np.float32 and after[0].tolist() == expected


def test_an_agent_trading_orders_records_each_slot_s_size_and_offset(tmp_path):
    data = universe()
    env  = probatum.env(data[:150], VENUE, window=8, actions="orders", max_offset=10,
                        max_leverage=0.1, seed=1)

    # sizes and offsets past their bounds, which each is placed at
    pairs  = np.array([[[1.7, 0.25], [-0.2, 1.0]], [[-1.7, -0.5], [0.2, 1.5]]])
    placed = np.array([[[1.0, 0.25], [-0.2, 1.0]], [[-1.0, 0.0], [0.2, 1.0]]])
    asked  = []

    def limits(observation):
        asked.append(observation)
        return pairs[len(asked) % 2][None]

    result = probatum.backtest(limits, data[150:], VENUE, spec=env)

    assert result.policy.shape == (150, 2, 2) and np.isnan(result.policy[-1]).all()
    assert np.array_equal(result.policy[:-1], placed[np.arange(1, 150) % 2])
    assert not result.bust and len(result.fills) > 50

    result.save(tmp_path / "orders")
    loaded = probatum.Result.load(tmp_path / "orders")

    assert np.array_equal(loaded.policy, result.policy, equal_nan=True)

    # the first close's BTCUSDT sell rests round(0.0 * 10) ticks above it, and ETHUSDT's
    # buy round(1.0 * 10) = 10 ticks of 0.01 below its close, each put on the venue's grid
    log    = result.orders
    first  = log["price"][(log["kind"] == "placed") & (log["bar"] == 0)]
    closes = data[150:].closes[0]

    assert abs(first[0] - closes[0]) < 0.1 and abs(first[1] - (closes[1] - 0.1)) < 0.01
