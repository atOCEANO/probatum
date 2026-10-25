"""A run recorded at the account level keeps the ledger, the round trips, the equity and
the statistics of the same run at the full level, to the bit, and no holdings, fill log
or order log, through a reset as before it; and a level the engine does not keep, or
one asked of an engine that records nothing, is refused. Its Result, an array's or a
Strategy's, holds the full level's numbers, prices the assumptions its ledger and round
trips measure, refuses by name what it does not keep, and crosses a pickle, a copy and
a save as it is, leaving nothing of a full Result saved where it is saved.
"""

import copy
import pickle

import numpy as np
import pytest

import probatum
from probatum import ProbatumAttributeError, ProbatumTypeError, ProbatumValueError
from probatum._venue import _run
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, bars, one, pair, same, weights)
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross

# the assumptions rows a fill log prices, which a level below the full one prices only
# where its ledger and round trips measure them
LOGGED = ("fees", "liquidation fee", "margin bound", "liquidation")


def levered():
    """Five times levered into a fall of a tenth with no cap, so a takeover closes both
    instruments' round trips."""
    rows = np.full((40, 2), np.nan)
    rows[3] = 5.0
    return rows, VENUE.replace(leverage=0.0)


def played(record, rows, venue=VENUE):
    """The binding's engine at ``record`` after an array of ``rows`` run twice on the
    pair, a reset between, with what it recorded each time."""
    data   = pair()
    run    = _run(venue, data)
    engine = probatum._probatum.Engine(
        run.candles,
        report = True,
        record = record,
        **run.settings,
    )

    seen = []
    for _ in range(2):
        probatum._probatum.run_array(engine, np.ascontiguousarray(rows), start=run.start)
        seen.append((engine.tables(run.start, list(data.names)),
                     engine.stats(8_760.0, 0.0), engine.equity_curve(), engine.trades()))

    return engine, seen


@pytest.mark.parametrize("policy", [lambda: (weights(), VENUE), levered])
def test_the_account_level_keeps_the_full_levels_ledger_and_round_trips_alone(policy):
    rows, venue = policy()

    full, wholes = played("full", rows, venue)
    account, parts = played("account", rows, venue)

    for (tables, stats, curve, trades), (whole, *kept) in zip(parts, wholes):
        assert list(tables)     == ["ledger", "trades"]
        assert tables["ledger"] == whole["ledger"]
        assert tables["trades"] == whole["trades"]
        assert same([stats, curve, trades], kept)

    assert len(wholes[0][0]["fills"]) and len(wholes[0][0]["holdings"])
    assert (account.fills(), account.order_log(), account.order_log_since(0),
            account.account(), account.num_events()) == (None, None, None, None, 0)
    assert full.num_events() and full.account() is not None

    ledger = account.ledger()

    assert "position" not in ledger and "position" in full.ledger()
    assert same(ledger, {name: column for name, column in full.ledger().items()
                         if name in ledger})


def test_the_levered_run_is_taken_over_on_both_instruments():
    rows, venue = levered()

    _, wholes = played("full", rows, venue)
    trades = wholes[0][0]["trades"]

    assert list(trades.column("reason")) == ["liquidation", "liquidation"]


@pytest.mark.parametrize("record, report, said", [
    ("account", False,
     "record: \"account\" with report=False, which records nothing; pass report=True"),
    ("transitions", True, "record: \"transitions\"; use \"full\" or \"account\""),
])
def test_a_level_the_engine_does_not_keep_is_refused(record, report, said):
    with pytest.raises(ProbatumValueError) as caught:
        probatum._probatum.Engine(np.ones((3, 5)), report=report, record=record)

    assert str(caught.value) == said


def spot():
    """Weights from 0 to 1 on one spot instrument, which has no perpetual's rows."""
    data = probatum.Data(bars(), symbol="BTCUSDT", market="spot")
    return np.abs(weights(width=1)), data, VENUE


RUNS = {
    "weights":  lambda: (weights(), pair(), VENUE),
    "levered":  lambda: (levered()[0], pair(), levered()[1]),
    "spot":     spot,
    "strategy": lambda: (Cross(), one(), VENUE),
}


def both(name):
    """The Result of the run ``name`` names at the full level and at the account level."""
    policy, data, venue = RUNS[name]()
    return (probatum.backtest(policy, data, venue),
            probatum.backtest(policy, data, venue, record="account"))


def costs(result):
    return {row["name"]: row["cost"] for row in result.assumptions}


@pytest.mark.parametrize("name", RUNS)
def test_a_result_at_the_account_level_holds_the_full_levels_numbers(name):
    full, account = both(name)

    for array in ("time", "equity", "returns"):
        np.testing.assert_array_equal(getattr(account, array), getattr(full, array))
    assert account.ledger == full.ledger and account.trades == full.trades
    assert (account.stats, account.costs, account.bust, account.liquidations) == (
        full.stats, full.costs, full.bust, full.liquidations)
    assert account.policy is None and full.policy is not None
    assert (full.provenance["record"], account.provenance["record"]) == ("full",
                                                                         "account")
    # both instruments are closed out on one bar
    assert account.liquidations == (1 if name == "levered" else 0)

    whole, kept = costs(full), costs(account)

    assert whole.keys() == kept.keys()

    perpetual = name != "spot"
    unpriced  = (LOGGED if name == "levered" else LOGGED[2:]) if perpetual else ()

    for row in whole:
        assert kept[row] == (None if row in unpriced else whole[row]), row

    rows = [{key: value for key, value in row.items() if key != "cost"}
            for row in full.assumptions]

    assert rows == [{key: value for key, value in row.items() if key != "cost"}
                    for row in account.assumptions]


def test_a_result_at_the_account_level_refuses_by_name_what_it_does_not_keep():
    full, account = both("strategy")
    fix = "; run it with record=\"full\""

    for name, what in [("holdings", "no holdings"), ("fills", "no fill log"),
                       ("orders", "no order log"),
                       ("baselines", "no copy of its policy to play them"),
                       ("position", "no holdings"), ("weight", "no holdings"),
                       ("pnl", "no holdings")]:
        with pytest.raises(ProbatumAttributeError) as caught:
            getattr(account, name)
        assert str(caught.value) == (f"result.{name}: a run recorded at 'account' "
                                     f"keeps {what}{fix}")
        assert not hasattr(account, name) and hasattr(full, name)

    for call, said in [
        (lambda: account["BTCUSDT"],
         "result['BTCUSDT']: a run recorded at 'account' keeps no holdings, which an "
         "instrument's record reads"),
        (lambda: account.explain(full.fills[0]),
         "explain: a run recorded at 'account' keeps no fill log or order log to tell"),
        (lambda: probatum.combine({"a": full, "b": account}),
         "results: a run recorded at 'account' keeps no holdings, fill log or order "
         "log to join"),
    ]:
        with pytest.raises(ProbatumValueError) as caught:
            call()
        assert type(caught.value) is ProbatumValueError
        assert str(caught.value) == said + fix

    assert repr(account) == repr(full).replace(
        f"to {full.equity[-1]:,.2f}", f"to {full.equity[-1]:,.2f}, recorded at 'account'")


def test_a_result_at_the_account_level_crosses_a_pickle_a_copy_and_a_save(tmp_path):
    full, account = both("levered")
    told = account.to_dict()

    assert [told[name] for name in ("holdings", "fills", "orders", "baselines",
                                    "policy")] == [None] * 5

    whole = full.to_dict()

    for name in ("time", "equity", "returns", "ledger", "trades", "stats", "costs",
                 "bust", "liquidations"):
        assert same(told[name], whole[name]), name

    full.save(tmp_path)

    assert (tmp_path / "fills.arrow").is_file() and (tmp_path / "policy.npy").is_file()

    account.save(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "assumptions.arrow", "equity.npy", "ledger.arrow", "result.json",
        "returns.npy", "stats.arrow", "time.npy", "trades.arrow"]

    for again in (pickle.loads(pickle.dumps(account)), copy.deepcopy(account),
                  probatum.Result.load(tmp_path)):
        assert again.ledger == account.ledger and again.trades == account.trades
        assert again.assumptions == account.assumptions
        assert (again.stats, again.liquidations, again.policy) == (
            account.stats, account.liquidations, None)
        assert again.provenance["record"] == "account" and not hasattr(again, "fills")

    full.save(tmp_path)

    assert probatum.Result.load(tmp_path).fills == full.fills


@pytest.mark.parametrize("record, kind, said", [
    ("none", ProbatumValueError,
     "record: 'none', which keeps nothing a Result holds; use \"full\" or \"account\""),
    ("Account", ProbatumValueError, "record: 'Account'; use \"full\" or \"account\""),
    (None, ProbatumTypeError, "record: None; use \"full\" or \"account\""),
])
def test_backtest_refuses_a_level_it_does_not_keep(record, kind, said):
    for policy in (weights(width=1), Cross()):
        with pytest.raises(kind) as caught:
            probatum.backtest(policy, one(), VENUE, record=record)
        assert type(caught.value) is kind and str(caught.value) == said


def test_the_transitions_level_is_an_agent_s_alone():
    for policy, what in ((weights(width=1), "an array"), (Cross(), "a Strategy")):
        with pytest.raises(ProbatumValueError) as caught:
            probatum.backtest(policy, one(), VENUE, record="transitions")
        assert str(caught.value) == (f"record: 'transitions' beside {what}, which takes "
                                     f"no steps of an environment; use \"full\" or "
                                     f"\"account\", or backtest an agent")
