"""``backtest`` reads a policy in one fixed order. A table is matched to the Data's clock
and instruments by label. An array of weights, with its barriers, makes the run the
Engine makes stepped from Python with the same targets and the same brackets, and one
of quantities reaches the positions it wants. A class, a model that fits and an agent
are each refused or not run yet, with what to do instead, and so is anything the run
cannot read before it starts. The Result holds what the run recorded, its bars counted
from its first row, the copy of the policy the run read and what it was given,
annualized over the whole Data whatever slice of it ran.
"""

import math

import numpy as np
import pandas as pd
import pytest

import probatum
from probatum import Venue
from probatum._venue import _run
from test_every_depth_gives_one_ledger import placed

VENUE = Venue(fees={"maker": 0.0, "taker": 0.0})


def bars(n=40, level=100.0, freq="h", zone="UTC", start="2024-01-01"):
    """Bars that swing about `level` by a tenth of it."""
    t      = np.arange(n, dtype=np.float64)
    close  = level * (1.0 + 0.1 * np.sin(t / 3.0))
    opened = np.concatenate([[close[0]], close[:-1]])
    index  = pd.date_range(start, periods=n, freq=freq, tz=zone)

    return pd.DataFrame({
        "open":   opened,
        "high":   np.maximum(opened, close) * 1.01,
        "low":    np.minimum(opened, close) * 0.99,
        "close":  close,
        "volume": 1e6,
    }, index=index)


def pair():
    return probatum.Data({"BTCUSDT": bars(), "ETHUSDT": bars(level=10.0)},
                         market="linear")


def one():
    return probatum.Data(bars(), symbol="BTCUSDT", market="linear")


def weights(n=40, width=2, seed=3):
    """Weights from -1 to 1 with NaN holding on a third of the rows."""
    rng = np.random.default_rng(seed)
    out = rng.uniform(-1.0, 1.0, (n, width))
    out[rng.random((n, width)) < 0.35] = np.nan
    return out


def indexed(data, rows, names=("BTCUSDT", "ETHUSDT")):
    return pd.DataFrame(rows, columns=list(names),
                        index=pd.DatetimeIndex(data.time.astype("datetime64[ns]"),
                                               tz="UTC"))


def stepped(rows, data, venue=VENUE, tolerance=0.0):
    """The Engine over the same settings stepped from Python, a target at each close
    naming every instrument its row does not hold."""
    run    = _run(venue, data)
    engine = probatum._probatum.Engine(run.candles, report=True, **run.settings)
    engine.reset()

    for row in rows[:-1]:
        named = {k: float(value) for k, value in enumerate(row) if not math.isnan(value)}
        if named:
            engine.target(named, tolerance)
        engine.step()

    return engine


def same(mine, theirs):
    """Whether two records are equal, NaN where both hold NaN."""
    if isinstance(mine, dict):
        return mine.keys() == theirs.keys() and all(
            same(mine[key], theirs[key]) for key in mine)

    if isinstance(mine, list):
        return len(mine) == len(theirs) and all(map(same, mine, theirs))

    if isinstance(mine, np.ndarray):
        return np.array_equal(mine, theirs, equal_nan=mine.dtype.kind == "f")

    if isinstance(mine, float) and isinstance(theirs, float):
        return mine == theirs or (math.isnan(mine) and math.isnan(theirs))

    return mine == theirs


def same_run(result, engine):
    """Whether a Result holds the tables an engine run from the first bar records."""
    width = len(result.holdings) // len(result.time)
    held  = engine.tables(0, list(result.holdings["instrument"][:width]))

    for name, table in held.items():
        assert getattr(result, name)._held == table, name


def test_an_array_runs_as_the_engine_stepped_with_the_same_targets():
    data, rows = pair(), weights()

    result = probatum.backtest(rows, data, VENUE, tolerance=0.3)

    assert result.fills
    same_run(result, stepped(rows, data, tolerance=0.3))
    assert (result.provenance["tolerance"], result.provenance["sizing"]) == (0.3,
                                                                              "weight")


def test_an_instrument_alone_takes_a_one_dimensional_array():
    data, rows = one(), weights(width=1)

    flat = probatum.backtest(rows[:, 0], data, VENUE)

    assert same(flat.fills, probatum.backtest(rows, data, VENUE).fills)
    same_run(flat, stepped(rows, data))


def test_an_array_s_barriers_are_the_brackets_the_engine_stepped_from_python_places():
    data        = one()
    rows        = weights(width=1)[:, 0]
    take_profit = np.full(40, 0.04)
    take_profit[::3] = np.nan

    result = probatum.backtest(rows, data, VENUE, stop=0.02, take_profit=take_profit)
    run    = _run(VENUE, data)
    engine, counts = placed(probatum._probatum.Engine(run.candles, report=True, **run.settings),
                            list(rows), 0.02, [[value] for value in take_profit])

    assert counts["filled"]
    same_run(result, engine)


@pytest.mark.parametrize("library", ["pandas", "pyarrow", "polars"])
def test_a_table_is_matched_to_the_clock_and_the_instruments_by_label(library):
    data, rows = pair(), weights()
    frame = indexed(data, rows[:, ::-1], names=("ETHUSDT", "BTCUSDT")).iloc[::-1]
    frame.index.name = "time"

    if library == "pyarrow":
        frame = pytest.importorskip("pyarrow").Table.from_pandas(frame.reset_index())
    elif library == "polars":
        frame = pytest.importorskip("polars").from_pandas(frame.reset_index())

    result = probatum.backtest(frame, data, VENUE)

    same_run(result, stepped(rows, data))
    np.testing.assert_array_equal(result.policy, rows)


def test_a_pandas_series_is_a_table_of_one_column():
    data, rows = one(), weights(width=1)
    series = pd.Series(rows[:, 0], index=pd.DatetimeIndex(
        data.time.astype("datetime64[ns]"), tz="UTC")).iloc[::-1]

    same_run(probatum.backtest(series, data, VENUE), stepped(rows, data))

    with pytest.raises(ValueError, match="an unnamed pandas series for a universe of 2 "
                                         "instruments"):
        probatum.backtest(series, pair(), VENUE)


def test_a_daily_table_meets_its_bars_on_their_dates():
    # daily bars stamped at midnight in New York are dated as the Data dates them
    frame = bars(freq="D", zone="America/New_York")
    data  = probatum.Data(frame, symbol="AAPL", market="spot")
    rows  = np.abs(weights(width=1))
    table = pd.DataFrame({"AAPL": rows[:, 0]}, index=frame.index)

    same_run(probatum.backtest(table, data, VENUE), stepped(rows, data))


def test_a_table_may_leave_out_a_row_no_instrument_printed():
    eth       = bars(level=10.0).drop(pd.Timestamp("2024-01-01 20:00", tz="UTC"))
    selection = probatum.Data({"BTCUSDT": bars(), "ETHUSDT": eth},
                              market="linear")["ETHUSDT"]

    table = pd.DataFrame({"ETHUSDT": np.full(len(eth), 0.5)}, index=eth.index)

    assert probatum.backtest(table, selection, VENUE).fills


def test_a_table_that_does_not_match_the_data_is_refused():
    data, rows = pair(), weights()
    table = indexed(data, rows)

    with pytest.raises(ValueError, match="no column for ETHUSDT; give one for each of "
                                         "BTCUSDT, ETHUSDT"):
        probatum.backtest(table[["BTCUSDT"]], data, VENUE)
    with pytest.raises(ValueError, match="column SOLUSDT, which the data does not hold"):
        probatum.backtest(table.assign(SOLUSDT=0.0), data, VENUE)
    with pytest.raises(ValueError, match="column ETHUSDT holds values of (object|str); "
                                         "give numbers"):
        probatum.backtest(table.assign(ETHUSDT="buy"), data, VENUE)
    with pytest.raises(ValueError, match=r"no row at 2024-01-01 05:00.*NaN to hold"):
        probatum.backtest(table.drop(table.index[5]), data, VENUE)

    shifted = table.set_axis(table.index + pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match="a table row at 2024-01-01 00:30, which is not "
                                         "on the data's clock"):
        probatum.backtest(shifted, data, VENUE)

    with pytest.raises(ValueError, match="two rows at 2024-01-01 03:00"):
        probatum.backtest(pd.concat([table, table.iloc[3:4]]), data, VENUE)
    with pytest.raises(ValueError, match="policy: a table with no times"):
        probatum.backtest(table.reset_index(drop=True), data, VENUE)

    polars = pytest.importorskip("polars")
    with pytest.raises(ValueError, match="policy: a table with no times"):
        probatum.backtest(polars.Series("BTCUSDT", rows[:, 0]), one(), VENUE)


class OnlyAnArray:
    """A producer that exports the Arrow array interface and nothing else."""


    def __init__(self, array):
        self.array = array


    def __arrow_c_array__(self, requested_schema=None):
        return self.array.__arrow_c_array__(requested_schema)


def test_an_array_is_read_through_any_interface_the_host_can_read():
    data, rows = one(), weights(width=1)
    expected = stepped(rows, data)
    pyarrow  = pytest.importorskip("pyarrow")

    same_run(probatum.backtest(OnlyAnArray(pyarrow.array(rows[:, 0])), data, VENUE),
             expected)
    # numpy hands no byte-swapped array over DLPack, which then reads it as any other
    same_run(probatum.backtest(rows.astype(">f8"), data, VENUE), expected)
    same_run(probatum.backtest(memoryview(np.ascontiguousarray(rows)), data, VENUE),
             expected)


class Cross(probatum.Strategy):

    def next(self, state, engine):
        pass


class Fitted:

    def fit(self, x, y):
        return self


def test_what_backtest_does_not_run_is_answered_with_what_to_do_instead():
    data = pair()

    with pytest.raises(TypeError, match=r"got the class Cross, not an instance; pass "
                                        r"Cross\(\)"):
        probatum.backtest(Cross, data, VENUE)
    with pytest.raises(TypeError, match="an object of type Fitted, a model that fits, "
                                        "which backtest does not fit; pass its "
                                        "predictions as an array"):
        probatum.backtest(Fitted(), data, VENUE)
    with pytest.raises(NotImplementedError, match="a function, which backtest does not "
                                                  "run as an agent yet"):
        probatum.backtest(lambda observation: 0.0, data, VENUE)
    with pytest.raises(TypeError, match=r"an object of type str; pass a \(T, N\) array"):
        probatum.backtest("weights", data, VENUE)
    with pytest.raises(TypeError, match="an object of type list; pass"):
        probatum.backtest([[1.0, 0.0]] * 40, data, VENUE)


def test_an_array_that_does_not_fit_the_data_is_refused_naming_where():
    data = pair()

    with pytest.raises(ValueError, match=r"a \(T,\) array for a universe of 2 "
                                         r"instruments; pass a \(T, 2\) array"):
        probatum.backtest(np.ones(40), data, VENUE)
    with pytest.raises(ValueError, match=r"shape \(39, 2\) for a Data of 40 rows and 2 "
                                         r"instruments; pass one of \(40, 2\)"):
        probatum.backtest(np.ones((39, 2)), data, VENUE)
    with pytest.raises(ValueError, match=r"shape \(39,\) for a Data of 40 rows and 1 "
                                         r"instrument; pass one of \(40, 1\)"):
        probatum.backtest(np.ones(39), one(), VENUE)

    rows = weights()
    rows[3, 1] = np.inf
    with pytest.raises(ValueError, match=r"inf at row 3 \(2024-01-01 03:00.*\) for "
                                         r"ETHUSDT; give a finite value, or NaN to hold"):
        probatum.backtest(rows, data, VENUE)

    with pytest.raises(TypeError, match="policy: an array of booleans; give weights, 1.0 "
                                        "to enter, 0.0 to leave and NaN to hold"):
        probatum.backtest(np.zeros((40, 2), dtype=bool), data, VENUE)


@pytest.mark.parametrize("given, says", [
    (dict(sizing="weights"), "sizing: 'weights'"),
    (dict(tolerance=-0.1), "tolerance: -0.1; give a fraction of equity from 0"),
    (dict(risk_free=math.nan), "risk_free: nan; give the annual rate as a finite number"),
])
def test_a_setting_the_run_cannot_take_is_refused_before_it_runs(given, says):
    with pytest.raises(ValueError, match=says):
        probatum.backtest(weights(), pair(), VENUE, **given)


def test_a_short_on_spot_is_refused_before_the_run():
    data = probatum.Data(bars(), symbol="BTCUSDT", market="spot")
    rows = np.full(40, np.nan)
    rows[5] = -0.5

    with pytest.raises(ValueError, match=r"-0.5 at row 5 \(2024-01-01 05:00.*\) for "
                                         r"BTCUSDT, which is spot and cannot be short"):
        probatum.backtest(rows, data, VENUE)


def test_a_venue_that_decides_at_random_and_an_empty_data_are_refused():
    with pytest.raises(ValueError, match="venue: intrabar is \"random\""):
        probatum.backtest(weights(), pair(), VENUE.replace(intrabar="random"))
    with pytest.raises(ValueError, match="data: no bars to run"):
        probatum.backtest(weights(n=0), pair()["2030-01-01":], VENUE)


@pytest.mark.parametrize("given", [0.0, -0.04, math.inf, math.nan, True, "0.04",
                                   [0.04] * 40, np.array(True)])
def test_a_barrier_that_is_no_return_above_zero_is_refused(given):
    with pytest.raises((TypeError, ValueError), match="stop: "):
        probatum.backtest(weights(), pair(), VENUE, stop=given)


def test_a_barrier_of_one_value_any_way_written_is_a_number():
    for given in (0.04, np.float32(0.04), np.array(0.04)):
        result = probatum.backtest(weights(), pair(), VENUE, stop=given)
        assert result.provenance["stop"] == float(given)


def test_a_barrier_of_rows_is_checked_cell_by_cell_and_takes_nan_for_none():
    data    = pair()
    returns = np.full((40, 2), 0.03)
    returns[::4] = np.nan

    probatum.backtest(weights(), data, VENUE, take_profit=returns)

    # a table of barriers is read by label as a table of weights is
    frame    = indexed(data, returns[:, ::-1], names=("ETHUSDT", "BTCUSDT"))
    by_label = probatum.backtest(weights(), data, VENUE, take_profit=frame)

    assert same(by_label.orders,
                probatum.backtest(weights(), data, VENUE, take_profit=returns).orders)

    returns[7, 0] = 0.0
    with pytest.raises(ValueError, match=r"take_profit: 0.0 at row 7 .* for BTCUSDT; "
                                         r"give a return above 0, or NaN for none"):
        probatum.backtest(weights(), data, VENUE, take_profit=returns)

    with pytest.raises(ValueError, match=r"stop: an array of shape \(40, 3\)"):
        probatum.backtest(weights(), data, VENUE, stop=np.full((40, 3), 0.03))


def test_a_barrier_closes_a_round_trip_and_the_same_side_is_not_bought_back():
    result  = probatum.backtest(np.ones(40), one(), VENUE, stop=0.02)
    reasons = [fill["reason"] for fill in result.fills]

    assert reasons[:2] == ["rebalance", "stop"]
    assert "rebalance" not in reasons[2:]

    again = np.ones(40)
    again[20] = 0.0
    reasons = [fill["reason"]
               for fill in probatum.backtest(again, one(), VENUE, stop=0.02).fills]

    assert reasons.count("rebalance") > 1


def test_quantities_reach_the_position_each_row_wants():
    rows = np.full(40, np.nan)
    rows[[2, 10, 30]] = [3.0, -1.5, 0.0]

    result   = probatum.backtest(rows, one(), VENUE, sizing="quantity")
    position = result.holdings["position"]

    assert (position[3], position[11], position[31]) == (3.0, -1.5, 0.0)
    assert result.provenance["policy"] == "a (40, 1) array of quantities"


def test_a_selection_runs_as_a_data_of_its_instrument_alone():
    # ETH lists six hours after BTC and misses hours 20 and 21, rows its selection's
    # account never sees: hour 19's close takes the last value of hours 19 to 21,
    # hour 21's NaN holding, and the round trip it opens from flat takes its
    # take-profit from the row that value came from; the rows before hour 6 are not
    # read
    hours     = [pd.Timestamp(f"2024-01-01 {hour}:00", tz="UTC") for hour in (20, 21)]
    eth       = bars(level=10.0).iloc[6:].drop(hours)
    selection = probatum.Data({"BTCUSDT": bars(), "ETHUSDT": eth},
                              market="linear")["ETHUSDT"]

    alone = probatum.Data(eth, symbol="ETHUSDT", market="linear")
    rows  = np.full(40, np.nan)
    rows[[0, 8, 15, 19, 20, 25]] = [1.0, 1.0, 0.0, 0.25, 0.5, -0.5]
    take_profit = np.full(40, 0.3)
    take_profit[19], take_profit[20] = 0.2, 0.005

    own = np.full(len(alone), np.nan)
    at  = np.searchsorted(alone.time, selection.time[[8, 15, 19, 25]])
    own[at] = [1.0, 0.0, 0.5, -0.5]
    own_take_profit = np.full(len(alone), 0.3)
    own_take_profit[at[2]] = 0.005

    picked   = probatum.backtest(rows, selection, VENUE, take_profit=take_profit)
    expected = probatum.backtest(own, alone, VENUE, take_profit=own_take_profit)

    assert same([picked.ledger, picked.fills, picked.orders, picked.trades],
                [expected.ledger, expected.fills, expected.orders, expected.trades])
    np.testing.assert_array_equal(picked.time, alone.time)
    assert picked.provenance["periods_per_year"] == expected.provenance[
        "periods_per_year"]
    assert picked.stats == expected.stats


def test_a_slice_counts_its_bars_from_its_first_row_and_annualizes_as_the_whole():
    data   = pair()
    part   = data["2024-01-01 10:00":]
    result = probatum.backtest(weights(n=30), part, VENUE)

    np.testing.assert_array_equal(result.time, part.time)
    assert result.ledger["bar"].tolist() == list(range(30))
    assert all(0 < fill["bar"] < 30 for fill in result.fills)
    assert result.equity[0] == 10_000.0

    whole = probatum.backtest(weights(), data, VENUE)

    assert result.provenance["periods_per_year"] == whole.provenance["periods_per_year"]


def test_an_inverse_contract_s_holdings_hold_a_row_a_bar_for_its_instrument():
    frame = bars(level=50_000.0)
    frame.attrs.update(exchange="binance", market_type="inverse", symbol="BTCUSD",
                       interval="1h", quote="USD", contract_size=100.0, warnings=[],
                       volume_unit="contract")

    result = probatum.backtest(np.full(40, 0.5), probatum.Data(frame), Venue(balance=1.0))

    assert len(result.holdings) == 40
    assert set(result.holdings["instrument"]) == {"BTCUSD"}
    assert result.holdings["position"][2] > 0.0


def test_the_result_holds_what_the_run_recorded_and_what_it_was_given():
    data, rows = pair(), weights()

    given  = rows.copy()
    result = probatum.backtest(given, data, VENUE, stop=0.05, risk_free=0.01)
    given[:] = 0.0

    np.testing.assert_array_equal(result.policy, rows)
    assert not result.policy.flags.writeable
    np.testing.assert_array_equal(result.time, data.time)
    assert result.equity[0] == 10_000.0
    np.testing.assert_allclose(result.returns[1:], result.equity[1:] / result.equity[:-1]
                               - 1.0, rtol=0.0, atol=0.0)
    assert result.returns[0] == 0.0
    assert set(result.stats) >= {"sharpe", "max_drawdown_pct", "total_return_pct"}
    assert (result.bust, result.liquidations) == (False, 0)

    given = result.provenance
    assert given["data"]["fingerprint"] == data.fingerprint
    assert (given["venue"], given["policy"], given["stop"], given["take_profit"]) == (
        repr(VENUE), "a (40, 2) array of weights", 0.05, None)
    assert (given["risk_free"], given["periods_per_year"]) == (0.01, 8_760.0)
    assert repr(result).startswith("Result(a (40, 2) array of weights on probatum.Venue(")


def test_an_account_a_takeover_empties_counts_its_liquidation_and_earns_nothing_after():
    # ten times levered into a fall of a tenth on both perpetuals at once, one bar of
    # forced closes on two instruments
    rows = np.full((40, 2), np.nan)
    rows[3] = 5.0

    result = probatum.backtest(rows, pair(), VENUE.replace(leverage=0.0))
    closes = [fill for fill in result.fills if fill["reason"] == "liquidation"]

    assert result.bust and result.liquidations == 1 and len(closes) == 2

    gone = int(np.flatnonzero(result.equity <= 0.0)[0])
    assert result.returns[gone] == -1.0 and not result.returns[gone + 1:].any()
