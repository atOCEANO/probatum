"""Transitions are every step an agent took.

An environment built with record="transitions" keeps a row for each step each of its
environments takes in an episode: the observation it acted on, its action as given,
the step's reward and flags and the observation the step returned, worked here again
from what each step returned under every autoreset mode, and a step that resets an
environment, leaves an ended one as it was or raises in its reward takes none, each
episode numbered in the order its environment began it. An agent's backtest at the
transitions level keeps a row for each close it was asked at until its episode ended,
rewarded as its spec says. export writes the rows as one Arrow IPC file that pyarrow
and polars open and the package reads back, and a Result keeps them through save, load
and pickle, refusing another run's on load (Specification, Transitions; ADR 0040).
"""

import copy
import json
import math
import pickle
import re
import shutil
import warnings

import numpy as np
import pytest

import probatum
from probatum import (ProbatumAttributeError, ProbatumKeyError, ProbatumValueError,
                      SpecError)
from probatum import _ipc
from probatum._transitions import Transitions
from test_an_episode_runs_again_as_a_backtest_from_its_start import (HOUR, VENUE,
                                                                     universe, walked)


def built(**settings):
    """Three environments on `universe`'s bars recording transitions, episodes of six
    steps."""
    given = {**dict(window=8, num_envs=3, seed=4, episode_length=6,
                    record="transitions"), **settings}
    return probatum.env(universe(), VENUE, **given)


def flat(obs):
    """``obs``, every environment's or one's, as the flat float32 rows the transitions
    keep, the market block and then the account block."""
    if not isinstance(obs, dict):
        return np.asarray(obs)

    market, account = obs["market"], obs["account"]
    return np.concatenate([market.reshape(account.shape[:-1] + (-1,)), account], axis=-1)


def worked(env, steps):
    """The rows ``steps`` random steps of ``env`` should leave, worked from what each
    step returned: each environment stepped for real a row of the observation, episode
    and step it last returned, its action, the reward and flags, and the observation
    the step returned, or under same_step the last of an episode it ended; the disabled
    mode's ended environments reset after the step that ended them."""
    space = env.action_space
    space.seed(5)
    obs, info = env.reset()

    mode, n = env.autoreset_mode.name, env.num_envs
    last, ended, rows = flat(obs).copy(), np.zeros(n, dtype=bool), []
    dtype = np.int64 if env._kind == "discrete" else np.float32

    for _ in range(steps):
        action = space.sample()
        before = info
        obs, reward, terminated, truncated, info = env.step(action)
        after = flat(obs).copy()
        given = after.copy()
        for i in np.flatnonzero(info.get("_final_obs", np.zeros(n, dtype=bool))):
            given[i] = flat(info["final_obs"][i])

        acted = np.asarray(action).reshape(n, -1).astype(dtype)
        for i in np.flatnonzero(~ended):
            rows.append((i, before["episode"][i], before["step"][i], last[i], acted[i],
                         reward[i], terminated[i], truncated[i], given[i]))

        last  = after
        ended = (terminated | truncated) & (mode != "SAME_STEP")
        if mode == "DISABLED" and ended.any():
            obs, info = env.reset(options={"reset_mask": ended})
            last[ended] = flat(obs)[ended]
            ended[:]    = False

    return rows


@pytest.mark.parametrize("settings", [
    dict(),
    dict(autoreset="same_step", layout="dict"),
    dict(autoreset="disabled", actions="discrete"),
    dict(autoreset="same_step", actions="orders"),
    dict(layout="dict", actions="discrete", sample=1),
], ids=["next-step", "same-step-dict", "disabled-discrete", "same-step-orders",
        "sampled-discrete"])
def test_every_row_is_the_step_its_environment_took(settings):
    env  = built(**settings)
    rows = worked(env, 30)
    held = env.transitions

    assert len(held) == len(rows) and held.columns == [
        "env", "episode", "step", "observation", "action", "reward", "terminated",
        "truncated", "next_observation"]
    for k, name in enumerate(held.columns):
        values = np.array([row[k] for row in rows], dtype=held[name].dtype)
        assert held[name].tobytes() == values.tobytes(), name
    assert not held["observation"].flags.writeable


def test_a_step_that_resets_an_environment_takes_no_row():
    # the Specification's worked example: an episode ends at its third step, the step
    # after resets it and gives no row, and the next episode's rows count from 0
    env = built(episode_length=3)
    env.action_space.seed(1)
    env.reset()
    for _ in range(8):
        env.step(env.action_space.sample())

    held = env.transitions
    mine = held["env"] == 0

    assert held["step"][mine].tolist()      == [0, 1, 2, 0, 1, 2]
    assert held["episode"][mine].tolist()   == [0, 0, 0, 1, 1, 1]
    assert held["truncated"][mine].tolist() == [False, False, True] * 2


def test_a_seeded_reset_numbers_the_next_episode_on():
    # the info numbers an episode in its stream, which the seed starts again, and the
    # rows every episode the environment began, so env and episode name one
    env = built()
    env.action_space.seed(1)
    env.reset(seed=1)
    for _ in range(2):
        env.step(env.action_space.sample())

    _, info = env.reset(seed=1)
    env.step(env.action_space.sample())
    held = env.transitions

    assert info["episode"].tolist() == [0, 0, 0]
    assert held["episode"].tolist() == [0] * 6 + [1] * 3
    assert held["step"].tolist()    == [0] * 3 + [1] * 3 + [0] * 3


def test_a_step_whose_reward_raised_takes_no_row_and_the_next_opens_where_it_left():
    calls = []

    def reward(state):
        calls.append(state)
        if len(calls) == 3:
            raise ValueError("the third step's reward")
        return state.log_return

    env, plain = built(reward=reward), built(reward=lambda state: state.log_return)
    env.action_space.seed(1)
    env.reset()
    plain.reset()

    for k in range(5):
        action = env.action_space.sample()
        if k == 2:
            with pytest.raises(ValueError, match="the third step's reward"):
                env.step(action)
        else:
            env.step(action)
        plain.step(action)

    held, kept = env.transitions, plain.transitions

    # every row but the third step's, the fourth opening on the bar the third reached
    taken = kept["step"] != 2
    assert all(np.array_equal(held[name], kept[name][taken]) for name in held.columns)


def test_a_decode_that_writes_over_its_actions_leaves_them_kept_as_given():
    def decode(actions, state):
        actions *= 0.0
        return np.full(state.weights.shape, 0.1)

    env = built(decode=decode)
    env.action_space.seed(1)
    env.reset()
    action = env.action_space.sample()
    given  = action.copy()
    env.step(action)

    assert not action.any()
    assert env.transitions["action"].tobytes() == given.tobytes()

    # actions a row cannot keep are refused before the engines step
    for shape in ((3, 3), (2, 3)):
        with pytest.raises(ProbatumValueError, match=re.escape(
                f"actions: shape {shape} for 3 environments, which a transition keeps as "
                f"one action of the space each; give (3, 2)")):
            env.step(np.zeros(shape))
    assert env.step(given)[4]["step"].tolist() == [2, 2, 2]

    # an agent's backtest keeps what the agent returned, before the decode wrote over it
    trained     = probatum.env(universe(), VENUE, window=8, decode=decode)
    agent, seen = asked()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", probatum.ProbatumWarning)
        result = probatum.backtest(agent, universe(), VENUE, spec=trained, decode=decode,
                                   record="transitions")

    actions = np.concatenate([np.tanh(obs[:, :2] * 10.0) for obs in seen])
    assert result.transitions["action"].tobytes() == actions.astype(np.float32).tobytes()


def test_transitions_read_again_carry_the_spec_as_it_stands():
    env = built()
    env.reset()
    env.step(env.action_space.sample())
    held = env.transitions

    assert env.transitions is held

    # as probatum.sb3.SpecCallback copies SB3's statistics in after a rollout's last step
    env._spec["sb3"] = {"clip": 10.0, "epsilon": 1e-8, "mean": [0.0], "var": [1.0]}

    assert env.transitions.metadata["spec"] == env.agent_spec != held.metadata["spec"]


def asked():
    """An agent of two slots that keeps every observation it is handed, and what it
    was handed."""
    seen = []

    def agent(obs):
        seen.append(np.array(obs))
        return np.tanh(np.asarray(obs)[:, :2] * 10.0)

    return agent, seen


def backtested(record="transitions", **settings):
    """An agent's backtest on `universe`, of the spec of an environment built with
    ``settings``, at ``record``, and what the agent was handed."""
    env         = probatum.env(universe(), VENUE, window=8, num_envs=2, seed=4,
                               **settings)
    agent, seen = asked()
    with warnings.catch_warnings():
        # the bars are the training span's, which a run out of sample would not be
        warnings.simplefilter("ignore", probatum.ProbatumWarning)
        result = probatum.backtest(agent, universe(), VENUE, spec=env, record=record)

    return result, seen, env


def test_an_agent_s_backtest_keeps_a_row_for_each_close_it_was_asked_at():
    result, seen, env = backtested()
    held = result.transitions

    assert len(held) == len(seen) == len(result.equity) - 1
    assert held["observation"].tobytes() == np.concatenate(seen).tobytes()
    assert np.array_equal(held["next_observation"][:-1], held["observation"][1:])
    assert (held["env"] == 0).all() and (held["episode"] == 0).all()
    assert held["step"].tolist() == list(range(len(held)))

    actions = np.concatenate([np.tanh(obs[:, :2] * 10.0) for obs in seen])
    assert held["action"].tobytes() == actions.astype(np.float32).tobytes()

    # rewarded as the spec's log reward, on the run's own equity, floored at 1% of it
    equity  = result.equity
    floor   = 0.01 * equity[0]
    rewards = [probatum._probatum.ln(max(after, floor) / before)
               for before, after in zip(equity[:-1], equity[1:])]

    assert held["reward"].tolist() == rewards
    assert not held["terminated"].any() and held["truncated"].tolist()[-2:] == [False, True]
    assert held.metadata["spec"]   == env.agent_spec
    assert held.metadata["reward"] == {"builtin": "log", "floor": 0.01}
    # the account level's records beside them, and none of the full level's
    assert result.provenance["record"] == "transitions" and len(result.ledger) == len(equity)
    with pytest.raises(ProbatumAttributeError, match="^result.fills: a run recorded at "
                                                     "'transitions' keeps no fill log"):
        result.fills


def test_an_agent_s_rows_end_with_its_episode_where_its_backtest_goes_on(tmp_path):
    # a spot holding bought at 100 that the bar of row 20 takes to 0.5, below its floor
    # of a hundredth of the balance, which no liquidation reached, so the run asks the
    # agent at every close to the last while its episode ended at the crash
    bars      = np.tile([100.0, 100.0, 100.0, 100.0, 1e6], (30, 1))
    bars[20]  = [100.0, 100.0, 0.5, 0.5, 1e6]
    bars[21:] = [0.5, 0.5, 0.5, 0.5, 1e6]

    data  = probatum.Data(
        bars,
        symbol = "BTCUSDT",
        market = "spot",
        time   = np.arange(30, dtype=np.int64) * HOUR,
    )
    plain = probatum.Venue(fees={"maker": 0.0, "taker": 0.0})
    env   = probatum.env(data, plain, window=4)
    seen  = []

    def agent(obs):
        seen.append(obs)
        return np.ones((1, 1), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", probatum.ProbatumWarning)
        result = probatum.backtest(agent, data, plain, spec=env, record="transitions")

    held = result.transitions

    assert len(seen) == len(result.equity) - 1 == 29
    assert held["terminated"].tolist() == [False] * 19 + [True]

    # a load works the crash's reward out on the floor, as the run did
    result.save(tmp_path / "run")
    loaded = probatum.Result.load(tmp_path / "run")

    assert loaded.transitions["reward"][-1] == probatum._probatum.ln(0.01)


def test_an_agent_trained_on_the_sharpe_reward_is_rewarded_with_it(tmp_path):
    result, _, _ = backtested(reward="sharpe")
    result.save(tmp_path / "run")

    assert probatum.Result.load(tmp_path / "run").transitions.metadata == (
        result.transitions.metadata)

    # the differential Sharpe of the run's own log returns, from moments at zero
    equity, eta = result.equity, 2.0 / (8 + 1)
    floor, a, b, rewards = 0.01 * equity[0], 0.0, 0.0, []
    for before, after in zip(equity[:-1], equity[1:]):
        r      = probatum._probatum.ln(max(after, floor) / before)
        da, db = r - a, r * r - b
        spread = b - a * a
        rewards.append((b * da - a * db / 2.0) / (spread * math.sqrt(spread))
                       if spread > 0.0 else 0.0)
        a, b = a + eta * da, b + eta * db

    assert result.transitions["reward"].tolist() == rewards
    assert result.transitions.metadata["reward"]["builtin"] == "sharpe"


def test_an_agent_whose_spec_records_a_reward_of_its_own_is_refused():
    trained = probatum.env(universe(), VENUE, window=8, reward=lambda state: state.equity)
    agent, _ = asked()

    with pytest.raises(SpecError, match=re.escape(
            "record: 'transitions' for an agent whose spec records a reward of its own, ")):
        probatum.backtest(agent, universe(), VENUE, spec=trained, record="transitions")


def test_export_writes_one_file_any_arrow_reader_opens_and_the_package_reads_back(
        tmp_path):
    pa     = pytest.importorskip("pyarrow")
    ipc    = pytest.importorskip("pyarrow.ipc")
    polars = pytest.importorskip("polars")
    env    = built(actions="discrete", sample=1)
    env.action_space.seed(1)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())

    held = env.transitions
    path = tmp_path / "steps.arrow"
    env.export("arrow", path)
    table = ipc.open_file(path).read_all()
    size  = held["observation"].shape[1]

    assert table.schema.names == held.columns and table.num_rows == len(held)
    assert table.schema.field("observation").type == pa.list_(
        pa.field("item", pa.float32(), nullable=False), size)
    assert table.schema.field("action").type == pa.list_(
        pa.field("item", pa.int64(), nullable=False), 1)
    for name in ("observation", "next_observation", "action"):
        column = np.stack(table.column(name).to_numpy(zero_copy_only=False))
        assert column.tobytes() == held[name].tobytes()
    assert table.column("reward").to_pylist() == held["reward"].tolist()

    told = {key.decode(): json.loads(value) for key, value in table.schema.metadata.items()}

    assert told == {f"probatum.{key}": value for key, value in held.metadata.items()}
    assert told["probatum.spec"]    == env.agent_spec
    assert told["probatum.version"] == probatum.__version__
    assert told["probatum.spaces"] == {
        "observation": env.agent_spec["observation"],
        "action":      env.agent_spec["actions"],
    }
    assert polars.read_ipc(path).height == len(held)

    back = Transitions._read(path)

    assert all(back[name].tobytes() == held[name].tobytes() for name in held.columns)
    assert back.metadata == held.metadata

    # a list another writer left an item of missing is not one the package reads
    nulls = pa.table({"x": pa.array([[1.0, None, 2.0]], type=pa.list_(pa.float32(), 3))})
    with ipc.new_file(tmp_path / "nulls.arrow", nulls.schema) as writer:
        writer.write_table(nulls)

    with pytest.raises(_ipc.Refused, match=re.escape(
            "holds lists in 'x' whose items are missing or not 3 a row")):
        _ipc.read(tmp_path / "nulls.arrow")


def test_a_result_keeps_its_transitions_through_save_load_and_pickle(tmp_path):
    result, _, env = backtested()
    held = result.transitions

    # a deep copy's metadata is its own, beside the columns no copy changes
    deep = copy.deepcopy(result)
    deep.transitions.metadata["version"] = "0.0.0"

    assert held.metadata["version"] == probatum.__version__
    assert deep.transitions["observation"] is held["observation"]

    result.save(tmp_path / "run")

    for again in (probatum.Result.load(tmp_path / "run"),
                  pickle.loads(pickle.dumps(result))):
        assert all(again.transitions[name].tobytes() == held[name].tobytes()
                   for name in held.columns)
        assert again.transitions.metadata == held.metadata

    plain = result.to_dict()["transitions"]

    assert len(plain["reward"])         == len(held)
    assert plain["metadata"]["version"] == probatum.__version__

    result.export("arrow", tmp_path / "steps.arrow")

    assert Transitions._read(tmp_path / "steps.arrow").metadata == held.metadata

    # a run saved again below the level leaves no transitions behind it
    full, _, _ = backtested("account")
    full.save(tmp_path / "run")

    assert not (tmp_path / "run" / "transitions.arrow").exists()

    result.save(tmp_path / "run")

    # its rows under another spec, and the same agent's steps on other bars, are not
    # this run's
    agent, _ = asked()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", probatum.ProbatumWarning)
        elsewhere = probatum.backtest(agent, universe(seed=9), VENUE, spec=env,
                                      record="transitions").transitions

    respec = Transitions({name: held[name] for name in held.columns},
                         dict(held.metadata, spec=dict(held.metadata["spec"], window=6)))

    assert len(elsewhere) == len(held) and elsewhere.metadata == held.metadata

    for other in (respec, elsewhere):
        other.export("arrow", tmp_path / "run" / "transitions.arrow")
        with pytest.raises(ProbatumValueError, match=re.escape(
                "transitions.arrow does not hold the steps of the agent its result.json "
                "describes")):
            probatum.Result.load(tmp_path / "run")

    shutil.copy(tmp_path / "run" / "ledger.arrow", tmp_path / "run" / "transitions.arrow")

    with pytest.raises(ProbatumValueError, match=re.escape(
            "transitions.arrow holds no transitions export wrote; give a directory a "
            "Result was saved into")):
        probatum.Result.load(tmp_path / "run")

    (tmp_path / "run" / "transitions.arrow").unlink()

    with pytest.raises(ProbatumValueError, match=re.escape("holds no transitions.arrow; "
                                                           "give a directory a Result "
                                                           "was saved into")):
        probatum.Result.load(tmp_path / "run")


def test_transitions_are_refused_where_none_were_kept(tmp_path):
    plain = probatum.env(universe(), VENUE, window=8)

    with pytest.raises(ProbatumAttributeError, match=re.escape(
            "env.transitions: an environment recorded at 'none' keeps no transitions; "
            "build it with record=\"transitions\"")):
        plain.transitions

    run = probatum.backtest(np.zeros((len(universe()), 2)), universe(), VENUE)

    for said, call in (("result.transitions", lambda: run.transitions),
                       ("result.export", lambda: run.export("arrow", tmp_path / "x"))):
        with pytest.raises(ProbatumAttributeError, match=re.escape(
                f"{said}: a run recorded at 'full' keeps no transitions; backtest an "
                f"agent with record=\"transitions\"")):
            call()

    env = built()

    with pytest.raises(ProbatumValueError, match=re.escape(
            "format: 'csv'; give \"arrow\", the one format export writes")):
        env.export("csv", tmp_path / "x.csv")
    with pytest.raises(ProbatumKeyError, match=re.escape(
            "transitions: no column 'obs'; use one of env, episode, step, observation")):
        env.transitions["obs"]


def test_a_row_holds_the_bytes_the_specification_counts():
    # the worked example: one instrument's candles over a 64-bar window
    rng  = np.random.default_rng(2)
    data = probatum.Data(
        walked(30_000.0, 120, rng),
        symbol = "BTCUSDT",
        market = "linear",
        time   = np.arange(120, dtype=np.int64) * HOUR,
    )
    env  = probatum.env(data, VENUE, window=64, record="transitions")

    env.reset()
    env.step(env.action_space.sample())
    held = env.transitions
    row  = held["observation"][0].nbytes + held["next_observation"][0].nbytes

    assert row == 2 * 328 * 4 == 2_624 and held["action"][0].nbytes == 4
