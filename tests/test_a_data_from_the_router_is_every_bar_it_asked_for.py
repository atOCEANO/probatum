"""`Data.from_router` fetches every bar from its start to its end that has closed,
through an exchange-router object, chaining calls back from the newest, since one call
walks back only so far. A call that brings nothing earlier is asked once more, and
where it brings nothing again the venue's history begins there, which the provenance
records and a warning names. A symbol the router fails on raises the router's own
error rather than being left out, and a funding series that misses part of its
instrument's span, at its head, at its tail or between, is refused, so a run never
charges nothing in silence. Each instrument takes its base asset and order limits from
the router's symbol information, and the pull logs each instrument it built and each
call it made to the probatum logger.
"""

import asyncio
import logging
import re
import warnings

import numpy as np
import pandas as pd
import polars
import pyarrow as pa
import pytest

import probatum
import probatum._data
from probatum import DataError, ProbatumTypeError, ProbatumWarning

HOUR    = 3_600_000_000_000
START   = 1_704_067_200_000_000_000    # 2024-01-01 00:00 UTC
CANDLES = ["open", "high", "low", "close", "volume", "volume_usd"]
FUNDING = ["rate", "funding_per_hour", "cycle_ms"]


def bars(n, start=START, step=HOUR, close=100.0):
    times  = start + step * np.arange(n, dtype=np.int64)
    closes = close + np.arange(n, dtype=np.float64)
    return times, np.column_stack([closes, closes + 1.0, closes - 1.0, closes,
                                   np.full(n, 1_000.0)])


def settlements(first=START + 8 * HOUR, last=START + 48 * HOUR, every=8 * HOUR,
                rate=0.0001):
    times = np.arange(first, last + 1, every, dtype=np.int64)
    return times, np.full(len(times), rate)


class Batch:
    """What candles_many returns: the frames it got, those that came with warnings
    filed as degraded, and the symbols it failed on."""


    def __init__(self, frames, failed):
        self.ok       = {s: f for s, f in frames.items() if not f.attrs["warnings"]}
        self.degraded = {s: (f, f.attrs["warnings"]) for s, f in frames.items()
                         if f.attrs["warnings"]}
        self.failed = failed


    def data(self):
        return {**self.ok, **{s: f for s, (f, _) in self.degraded.items()}}


class Answered(Exception):
    """An exchange-router error, which carries the status the service answered."""


    def __init__(self, detail, status):
        super().__init__(detail)
        self.detail, self.status = detail, status


class Router:
    """A router serving frames shaped as exchange-router 3.0.0 builds them, walking
    back from each call's start at most `cap` rows. Its funding walks back the same
    way, or with `window`, the cycle it believes, answers as Binance does: with the
    oldest `limit` events of the `limit` cycles before the start. It lists every
    symbol it holds but those `unlisted`, as for one delisted since its bars."""


    def __init__(self, held, funding=None, cap=1_000, market="linear", silent=0,
                 failing=(), window=None, unlisted=()):
        self.held, self.funding = held, funding or {}
        self.cap, self.market, self.window = cap, market, window
        self.silent, self.failing, self.calls = silent, failing, []
        self.unlisted = unlisted


    def frame(self, symbol, times, values, columns, attrs):
        meta = dict(exchange="binance", market_type=self.market, symbol=symbol,
                    schema_version=3, warnings=[])

        if len(times):
            index = pd.DatetimeIndex(pd.to_datetime(times, utc=True), name="datetime")
            out   = pd.DataFrame(values, index=index, columns=columns)
            meta.update(attrs)
        else:
            # the router's own empty frame, with no index of times and no column types
            out = pd.DataFrame(columns=columns)
            meta["warnings"] = [f"0 rows returned for binance/{self.market}/{symbol}"]

        out.attrs.update(meta)
        return out


    def rows(self, times, limit, start):
        upper = start * 1_000_000 if start is not None else times[-1]
        return np.flatnonzero(times <= upper)[-min(limit, self.cap):]


    def get_candles(self, exchange, market_type, symbol, interval="1h", limit=100,
                    start=None, verbose=None):
        self.calls.append((symbol, start, limit, verbose))

        times, values = self.held[symbol]
        rows = self.rows(times, limit, start)
        if self.silent:
            self.silent -= 1
            rows = rows[:0]

        values = np.column_stack([values[rows], values[rows][:, 3] * 1_000.0])
        attrs  = dict(quote="USDT", volume_unit="base")
        if self.market == "inverse":
            attrs = dict(quote="USD", volume_unit="contract", contract_size=100.0)
        if symbol in self.unlisted:
            # the router writes an empty quote for a symbol it holds no information on
            attrs["quote"] = ""

        return self.frame(symbol, times[rows], values, CANDLES,
                          dict(attrs, interval=interval, warnings=[f"served {symbol}"]))


    def get_symbol_info(self, exchange, market_type, symbol):
        if symbol in self.unlisted:
            raise Answered(f"Symbol {symbol} not found on Binance {market_type}", 400)

        # its own name for the symbol however it was typed, an inverse contract's
        # minimum in dollars and its sizes in contracts
        inverse = market_type == "inverse"
        return dict(symbol=symbol.upper(), base_asset=symbol[:3].upper(),
                    quote_asset="USD" if inverse else "USDT",
                    min_notional=100.0 if inverse else 5.0, max_qty=1_000.0,
                    qty_unit="contract" if inverse else "base",
                    contract_size=100.0 if inverse else None)


    def candles_many(self, exchange, market_type, symbols, interval="1h", limit=100,
                     start=None, verbose=None, max_concurrent=8):
        frames = {s: self.get_candles(exchange, market_type, s, interval, limit, start,
                                      verbose)
                  for s in symbols if s not in self.failing}
        return Batch(frames, {s: RuntimeError("upstream 503") for s in self.failing})


    def get_funding_rate(self, exchange, market_type, symbol, limit=100, start=None,
                         verbose=None):
        times, rates = self.funding[symbol]
        if self.window is None:
            rows = self.rows(times, limit, start)
        else:
            count, upper = min(limit, 1_000), start * 1_000_000
            inside = (times >= upper - count * self.window) & (times <= upper)
            rows   = np.flatnonzero(inside)[:count]

        values = np.column_stack([rates[rows], rates[rows] / 8.0,
                                  np.full(len(rows), 8 * 3_600_000.0)])
        return self.frame(symbol, times[rows], values, FUNDING,
                          dict(warnings=[f"funded {symbol}"]))


def fetch(router, symbols=("BTCUSDT",), **asked):
    return probatum.Data.from_router(router, "binance", "linear", "1h", list(symbols),
                                     **{**dict(start="2024-01-01",
                                               end="2024-01-02 23:00"), **asked})


def test_calls_are_chained_back_to_the_start():
    router = Router({"BTCUSDT": bars(100)}, cap=30)
    data   = fetch(router, end="2024-01-05 03:00")

    assert len(data) == 100 and data.time[0] == START
    assert len(router.calls) == 4
    assert {call[3] for call in router.calls} == {False}
    assert data.provenance["source"]          == "router"
    assert data.provenance["warnings"]        == ["served BTCUSDT"]
    assert "history" not in data.provenance


def test_a_text_end_takes_in_the_whole_period_it_names():
    data = fetch(Router({"BTCUSDT": bars(100)}), start="2024-01-02", end="2024-01-02")

    assert len(data) == 24 and data.time[0] == START + 24 * HOUR


def test_a_call_that_brings_nothing_is_asked_once_more():
    data = fetch(Router({"BTCUSDT": bars(100)}, cap=30, silent=1),
                 end="2024-01-05 03:00")

    assert len(data) == 100


def test_a_history_that_begins_after_the_start_is_recorded_and_warned_of():
    router = Router({"BTCUSDT": bars(50, start=START + 50 * HOUR)}, cap=30)

    with pytest.warns(ProbatumWarning, match=re.escape(
            "data: BTCUSDT's bars begin at 2024-01-03 02:00, after the start; the "
            "venue serves none earlier, see data.provenance[\"history\"]")) as caught:
        data = fetch(router, end="2024-01-05 03:00")

    # named at the line that asked for the bars, fetch's call
    warned = caught.pop(ProbatumWarning)
    assert warned.filename == __file__
    assert warned.lineno   == fetch.__code__.co_firstlineno + 1
    assert len(data)       == 50
    assert data.provenance["history"] == {"BTCUSDT": "2024-01-03 02:00"}


def test_bars_that_end_before_the_end_are_warned_of():
    held = {
        "BTCUSDT": bars(48),
        "ETHUSDT": bars(40, close=10.0),
        "SOLUSDT": bars(30, close=50.0),
    }

    with pytest.warns(ProbatumWarning, match=re.escape(
            "data: ETHUSDT's bars end at 2024-01-02 15:00, before the end, and those "
            "of 1 more instrument end early too; the venue served none later")):
        fetch(Router(held), held)


def test_several_symbols_come_from_one_batch_and_one_it_failed_on_raises_its_error():
    held = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    data = fetch(Router(held), held)

    assert data.names == ("BTCUSDT", "ETHUSDT") and data.closes.shape == (48, 2)

    with pytest.raises(RuntimeError, match="upstream 503") as caught:
        fetch(Router(held, failing=("ETHUSDT",)), held)

    assert caught.value.__notes__ == [
        "symbols: the router failed on ETHUSDT; leave it out, or fetch it again once "
        "the router can serve it"]

    with pytest.raises(RuntimeError, match="upstream 503") as caught:
        fetch(Router(held, failing=("BTCUSDT", "ETHUSDT")), held)

    assert caught.value.__notes__ == [
        "symbols: the router failed on all 2; fetch them again once it can serve them"]


def test_a_pull_logs_each_instrument_it_built_and_each_call_to_the_probatum_logger(
        caplog):
    held    = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    funding = {"BTCUSDT": settlements(), "ETHUSDT": settlements()}

    # a batch of thirty bars each, so each symbol walks back once for the rest, and
    # each funding series walks back past its first event and is asked once more
    with caplog.at_level(logging.DEBUG, logger="probatum"):
        fetch(Router(held, funding=funding, cap=30), held, funding=True)

    said = [(record.levelname, record.getMessage()) for record in caplog.records
            if record.name == "probatum"]
    pulled = [line for symbol in held for line in (
        ("DEBUG", f"from_router: binance linear {symbol}, 1h bars back from "
                  f"2024-01-01 17:59:59"),
        ("DEBUG", f"from_router: binance linear {symbol}'s symbol information"),
        ("INFO", f"from_router: {symbol}, 48 1h bars from 2024-01-01 00:00 to "
                 f"2024-01-02 23:00"))]
    funded = [line for symbol in held for line in (
        ("DEBUG", f"from_router: binance linear {symbol}'s funding back from "
                  f"2024-01-03 00:00:01"),
        *[("DEBUG", f"from_router: binance linear {symbol}'s funding back from "
                    f"2024-01-01 07:59:59")] * 2,
        ("INFO", f"from_router: {symbol}'s funding, 6 events from 2024-01-01 08:00 "
                 f"to 2024-01-03 00:00"))]

    assert said == [("DEBUG", "from_router: binance linear, 2 symbols' 1h bars in one "
                              "batch"), *pulled, *funded]


def test_each_instrument_takes_its_base_asset_and_order_limits_from_the_router():
    held = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    btc, eth = fetch(Router(held), held).instruments

    assert (btc.base, btc.quote, btc.settlement)     == ("BTC", "USDT", "USDT")
    assert (btc.min_notional, btc.max_qty, eth.base) == (5.0, 1_000.0, "ETH")

    [coin] = probatum.Data.from_router(
        Router({"BTCUSD": bars(48)}, market="inverse"), "binance", "inverse", "1h",
        ["BTCUSD"], start="2024-01-01", end="2024-01-02 23:00").instruments

    assert (coin.base, coin.quote, coin.settlement)         == ("BTC", "USD", "BTC")
    assert (coin.contract, coin.min_notional, coin.max_qty) == (100.0, None, 1_000.0)


def test_a_symbol_the_router_no_longer_lists_keeps_its_bars_without_its_limits():
    held = {"BTCUSDT": bars(48), "LUNAUSDT": bars(48, close=10.0)}

    with pytest.warns(ProbatumWarning, match=re.escape(
            "data: the router does not list LUNAUSDT now, saying 'Symbol LUNAUSDT not "
            "found on Binance linear', as for a symbol delisted or paused, so its base "
            "asset and order limits are unknown; see data.instruments")):
        data = fetch(Router(held, unlisted=("LUNAUSDT",)), held)

    luna = data.instruments[1]
    assert len(data) == 48 and data.instruments[0].base == "BTC"
    assert (luna.base, luna.min_notional, luna.max_qty, luna.quote) == (
        None, None, None, None)


def test_an_instrument_takes_the_router_s_name_for_its_symbol_however_typed():
    data = fetch(Router({"btcusdt": bars(48)}), ["btcusdt"])

    assert data.names == ("BTCUSDT",) and data.instruments[0].symbol == "BTCUSDT"

    held = {"btcusdt": bars(48), "BTCUSDT": bars(48)}
    with pytest.raises(DataError, match=re.escape(
            "symbols: 'btcusdt' and 'BTCUSDT' are one instrument, BTCUSDT; name it "
            "once")):
        fetch(Router(held), held)


@pytest.mark.parametrize("status", [503, 404])
def test_a_router_that_cannot_answer_for_a_symbol_raises_its_own_error(status):
    class Down(Router):

        def get_symbol_info(self, exchange, market_type, symbol):
            raise Answered(f"upstream {status}", status)

    with pytest.raises(Answered, match=f"upstream {status}"):
        fetch(Down({"BTCUSDT": bars(48)}))


def test_the_bar_still_open_is_left_out_whatever_the_end(monkeypatch):
    monkeypatch.setattr(probatum._data, "_now", lambda: START + 47 * HOUR + HOUR // 2)

    for end in (None, "2024-01-03"):
        data = fetch(Router({"BTCUSDT": bars(48)}), end=end)
        assert len(data) == 47 and data.provenance["fetched"] == "2024-01-02 23:30"


def test_funding_events_are_kept_on_the_instrument_they_charge():
    held    = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    funding = {"BTCUSDT": settlements(), "ETHUSDT": settlements(rate=-0.0002)}
    data    = fetch(Router(held, funding), held, funding=True)
    events  = data.funding

    assert len(events) == 12
    assert events["instrument"][events["rate"] < 0].tolist() == [1] * 6
    assert np.isnan(data.funding_rate[6, 0]) and data.funding_rate[7, 0] == 0.0001
    assert len(data["2024-01-02":].funding) == 6
    assert "funded BTCUSDT" in data.provenance["warnings"]


def test_a_view_and_a_join_keep_each_event_on_its_instrument():
    held   = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    router = Router(held, {
        "BTCUSDT": settlements(),
        "ETHUSDT": settlements(rate=-0.0002),
    })

    both    = fetch(router, held, funding=True)
    flipped = both.select(["ETHUSDT", "BTCUSDT"]).funding

    assert flipped["instrument"][flipped["rate"] < 0].tolist() == [0] * 6

    joined = fetch(router, ["BTCUSDT"], funding=True).join(
        fetch(router, ["ETHUSDT"], funding=True))

    assert joined.funding["instrument"][joined.funding["rate"] < 0].tolist() == [1] * 6
    np.testing.assert_array_equal(joined.funding_rate, both.funding_rate)


def test_a_funding_event_is_part_of_the_fingerprint_and_the_time_of_the_fetch_is_not(
        monkeypatch):
    times, rates = settlements()
    moved = rates.copy()
    moved[2] = 0.0002

    first = fetch(Router({"BTCUSDT": bars(48)}, {"BTCUSDT": (times, rates)}),
                  funding=True)

    monkeypatch.setattr(probatum._data, "_now",
                        lambda: probatum._data.time.time_ns() + HOUR)
    again = fetch(Router({"BTCUSDT": bars(48)}, {"BTCUSDT": (times, rates)}),
                  funding=True)
    changed = fetch(Router({"BTCUSDT": bars(48)}, {"BTCUSDT": (times, moved)}),
                    funding=True)

    assert first.provenance["fetched"] != again.provenance["fetched"]
    assert first.fingerprint == again.fingerprint != changed.fingerprint


def test_events_at_one_time_give_one_fingerprint_whatever_the_instruments_order():
    held    = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    funding = {"BTCUSDT": settlements(), "ETHUSDT": settlements(rate=-0.0002)}
    both    = fetch(Router(held, funding), held, funding=True)
    swapped = fetch(Router(held, funding), ["ETHUSDT", "BTCUSDT"], funding=True)

    assert both.select(["ETHUSDT", "BTCUSDT"]).fingerprint == swapped.fingerprint


def test_coarser_bars_keep_each_funding_event_at_its_stamp():
    held    = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    funding = {"BTCUSDT": settlements(), "ETHUSDT": settlements(rate=-0.0002)}
    data    = fetch(Router(held, funding), held, funding=True)
    four    = data.resample("4h")

    np.testing.assert_array_equal(four.funding, data.funding)
    assert np.isnan(four.funding_rate[0, 0]) and four.funding_rate[1, 0] == 0.0001

    flipped = data.select(["ETHUSDT", "BTCUSDT"]).resample("4h").funding

    assert flipped["instrument"][flipped["rate"] < 0].tolist() == [0] * 6

    # the event at 08:00 belongs to the bar from 04:00, which the slice keeps behind it
    late = data["2024-01-01 06:00":].resample("4h").funding

    assert sorted(set(late["time"].tolist())) == [START + h * HOUR
                                                  for h in (16, 24, 32, 40, 48)]

    # the bar from 44:00 is not covered to its end, and its event goes with it
    short = data[:"2024-01-02 21:00"].resample("4h").funding

    assert short["time"].max() == START + 40 * HOUR


def test_funding_events_are_saved_and_loaded_with_their_bars(tmp_path):
    held    = {"BTCUSDT": bars(48), "ETHUSDT": bars(48, close=10.0)}
    funding = {"BTCUSDT": settlements(), "ETHUSDT": settlements(rate=-0.0002)}
    data    = fetch(Router(held, funding), held, funding=True)

    data.save(tmp_path / "pair")
    loaded = probatum.Data.load(tmp_path / "pair")

    assert loaded.fingerprint == data.fingerprint
    np.testing.assert_array_equal(loaded.funding, data.funding)
    assert loaded.provenance  == data.provenance
    assert loaded.instruments == data.instruments
    assert loaded.instruments[1].base == "ETH" and loaded.instruments[1].max_qty == 1e3

    events = pa.ipc.open_file(tmp_path / "pair" / "funding.arrow").read_all()

    assert events.num_rows == 12 and events["instrument"].to_pylist()[:2] == [0, 1]
    assert polars.read_ipc(tmp_path / "pair" / "funding.arrow")["rate"].min() < 0

    # a slice keeps the rate known before it, which the bars before it carry
    late = data["2024-01-02":]
    late.save(tmp_path / "late")
    again = probatum.Data.load(tmp_path / "late")

    assert again.fingerprint == late.fingerprint and len(again) == 24
    np.testing.assert_array_equal(again.funding_rate, late.funding_rate)
    assert not np.isnan(again.funding_rate[0]).any()


def test_a_settlement_stamped_within_a_second_of_a_bar_s_end_is_charged_at_it():
    times, rates = settlements()
    times[0] += 200_000_000

    data = fetch(Router({"BTCUSDT": bars(48)}, {"BTCUSDT": (times, rates)}),
                 funding=True)

    assert np.isnan(data.funding_rate[6, 0]) and data.funding_rate[7, 0] == 0.0001


@pytest.mark.parametrize("every", [HOUR, 4 * HOUR, 8 * HOUR])
def test_funding_from_a_venue_that_answers_a_window_of_cycles_is_whole(every):
    # Binance reads the count as that many cycles before the start and answers with
    # the oldest events in them, so a count longer than the span's events would lose
    # the newest
    times, rates = settlements(START - 240 * HOUR, every=every)
    router = Router({"BTCUSDT": bars(48)}, {"BTCUSDT": (times, rates)}, window=every)
    owned  = times[(times > START) & (times <= START + 48 * HOUR)]

    assert fetch(router, funding=True).funding["time"].tolist() == owned.tolist()


def test_a_funding_cycle_that_changes_is_no_hole():
    eight, four = settlements(last=START + 24 * HOUR), settlements(
        START + 28 * HOUR, every=4 * HOUR)
    events = (np.concatenate([eight[0], four[0]]), np.concatenate([eight[1], four[1]]))
    data   = fetch(Router({"BTCUSDT": bars(48)}, {"BTCUSDT": events}), funding=True)

    assert len(data.funding) == 9


def holed():
    times, rates = settlements()
    return np.delete(times, 2), np.delete(rates, 2)


def unrated():
    times, rates = settlements()
    rates[1] = np.nan
    return times, rates


@pytest.mark.parametrize("settings, funding, says", [
    ({}, settlements(START + 24 * HOUR),
     "funding: BTCUSDT's funding history begins at 2024-01-02 00:00, more than a "
     "cycle after its first bar at 2024-01-01 00:00; start the bars at 2024-01-02 "
     "00:00 or load them without funding"),
    ({}, settlements(last=START + 24 * HOUR),
     "funding: BTCUSDT's funding history ends at 2024-01-02 00:00, more than a cycle "
     "before its last bar ends at 2024-01-03 00:00; end the bars by 2024-01-02 00:00 "
     "or load them without funding"),
    ({}, holed(),
     "funding: BTCUSDT's funding history misses the events between 2024-01-01 16:00 "
     "and 2024-01-02 08:00; load the bars before or after them, or without funding"),
    ({}, unrated(),
     "funding: BTCUSDT's event at 2024-01-01 16:00 has no rate; load it without "
     "funding, or after that time"),
    ({}, (np.zeros(0, dtype=np.int64), np.zeros(0)),
     "funding: BTCUSDT has no funding events over its bars, from 2024-01-01 00:00 to "
     "2024-01-03 00:00; load it without funding"),
    # a venue that answers a window of the cycle it believes, eight hours, for
    # events four hours apart returns only the older half of the span
    (dict(window=8 * HOUR), settlements(START - 240 * HOUR, every=4 * HOUR),
     "funding: BTCUSDT's funding history ends at 2024-01-01 20:00, more than a cycle "
     "before its last bar ends at 2024-01-03 00:00"),
])
def test_a_funding_series_that_misses_part_of_the_span_is_refused(settings, funding,
                                                                   says):
    router = Router({"BTCUSDT": bars(48)}, {"BTCUSDT": funding}, **settings)

    with pytest.raises(DataError, match=re.escape(says)):
        fetch(router, funding=True)


def test_gaps_in_the_bars_are_warned_of():
    times, values = bars(48)
    held = {"BTCUSDT": (np.delete(times, [10, 11, 30]),
                        np.delete(values, [10, 11, 30], axis=0))}

    with pytest.warns(ProbatumWarning, match=re.escape(
            "data: BTCUSDT misses 2 bars after 2024-01-01 09:00, one of 2 gaps; see "
            "data.gaps()")):
        fetch(Router(held))


class Async:

    async def get_candles(self, *args, **kwargs):
        await asyncio.sleep(0)


class Candles:

    def get_candles(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("router, settings, error, says", [
    (Async(), {}, ProbatumTypeError,
     "router: an AsyncRouter, whose calls return coroutines"),
    (object(), {}, ProbatumTypeError, "router: object, which has no get_candles"),
    (Candles(), {}, ProbatumTypeError, "router: Candles, which has no get_symbol_info"),
    (Router({}), dict(symbols="BTCUSDT"), ProbatumTypeError,
     "symbols: 'BTCUSDT'; give a list"),
    (Router({}), dict(symbols=5), ProbatumTypeError,
     "symbols: 5; give a list of symbols"),
    (Router({}), dict(symbols=[]), DataError, "symbols: none; name at least one"),
    (Router({}), dict(symbols=["BTCUSDT", "BTCUSDT"]), DataError,
     "symbols: ['BTCUSDT', 'BTCUSDT'] names BTCUSDT twice; name each once"),
    (Router({}), dict(market="spot", funding=True), DataError,
     "funding: True for the spot market, which charges none"),
    (Router({}), dict(funding="yes"), ProbatumTypeError,
     "funding: 'yes'; give True or False"),
    (Router({}), dict(interval="7m"), DataError, "interval: '7m'; use one of"),
    (Router({}), dict(start=None), ProbatumTypeError,
     "start: None; give a time or the text of one"),
    (Router({}), dict(start=1_704_067_200_000), ProbatumTypeError,
     "start: 1704067200000; give a time or the text of one"),
    (Router({}), dict(start="2200-01-01", end=None), DataError,
     "start: '2200-01-01', where no 1h bar has closed yet; give an earlier start"),
    (Router({}), dict(end="2023-12-31"), DataError,
     "end: '2023-12-31', before the start, '2024-01-01'; give an end after"),
    (Router({"BTCUSDT": bars(0)}), {}, DataError,
     "symbols: BTCUSDT has no 1h bars on binance linear over the span asked"),
])
def test_a_fetch_it_cannot_make_is_refused(router, settings, error, says):
    asked = dict(market="linear", interval="1h", symbols=["BTCUSDT"],
                 start="2024-01-01", end="2024-01-02 23:00")
    asked.update(settings)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(error, match=re.escape(says)):
            probatum.Data.from_router(router, "binance", asked.pop("market"),
                                      asked.pop("interval"), **asked)
