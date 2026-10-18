"""A run recorded at the account level keeps the ledger, the round trips, the equity and
the statistics of the same run at the full level, to the bit, and no holdings, fill log
or order log, through a reset as before it; and a level the engine does not keep, or
one asked of an engine that records nothing, is refused.
"""

import numpy as np
import pytest

import probatum
from probatum import ProbatumValueError
from probatum._venue import _run
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (
    VENUE, pair, same, weights)


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
