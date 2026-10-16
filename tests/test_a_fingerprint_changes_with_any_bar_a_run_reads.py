"""A Data's fingerprint is 128 bits of blake2b over everything a run on its bars can
read: the clock and its interval, the bars, which of them printed, the funding events
and the instruments, and for a slice the bars before it (ADR 0005). The same bars share
it however they arrived, and any change to what a run reads changes it; where the bars
came from does not.
"""

import re

import numpy as np
import pandas as pd

import probatum


def frame(n=24, close=100.0, start="2024-01-01"):
    closes = close + np.arange(n, dtype=np.float64)
    index  = pd.date_range(start, periods=n, freq="h", tz="UTC", name="datetime")

    return pd.DataFrame({
        "open":   closes,
        "high":   closes + 1.0,
        "low":    closes - 1.0,
        "close":  closes,
        "volume": 1_000.0,
    }, index=index)


def routed(warnings):
    out = frame()
    out.attrs.update(exchange="binance", market_type="linear", symbol="BTCUSDT",
                     interval="1h", quote="USDT", warnings=warnings)
    return out


def test_the_same_bars_give_the_same_fingerprint_however_they_arrive():
    source = frame()
    data   = probatum.Data(source, symbol="BTCUSDT")
    bars   = source[["open", "high", "low", "close", "volume"]].to_numpy()
    arrays = probatum.Data(bars, time=source.index, symbol="BTCUSDT")

    assert re.fullmatch(r"b2:[0-9a-f]{32}", data.fingerprint)
    assert data.fingerprint == arrays.fingerprint == data.provenance["fingerprint"]
    assert (probatum.Data(routed(["one warning"])).fingerprint
            == probatum.Data(routed(["another"])).fingerprint)


def test_any_change_to_what_a_run_reads_changes_it():
    nudged = frame()
    nudged.iloc[5, 3] += 1e-9

    data = [probatum.Data(frame(), symbol="BTCUSDT"),
            probatum.Data(nudged, symbol="BTCUSDT"),
            probatum.Data(frame(start="2024-01-02"), symbol="BTCUSDT"),
            probatum.Data(frame().drop(frame().index[5]), symbol="BTCUSDT"),
            probatum.Data(frame(), symbol="BTCUSDT", interval="30m"),
            probatum.Data(frame(), symbol="ETHUSDT"),
            probatum.Data(frame(), symbol="BTCUSDT", market="linear")]

    assert len({one.fingerprint for one in data}) == len(data)


def test_a_slice_s_fingerprint_covers_the_bars_before_it():
    # a run on a slice reads the bars before it for its features' warm-up
    whole = probatum.Data(frame(), symbol="BTCUSDT")

    assert whole[:10].fingerprint == probatum.Data(frame().iloc[:10],
                                                   symbol="BTCUSDT").fingerprint
    assert whole[5:10].fingerprint != probatum.Data(frame().iloc[5:10],
                                                    symbol="BTCUSDT").fingerprint
    assert whole[5:10].fingerprint != whole[6:10].fingerprint

    earlier = frame()
    earlier.iloc[:5, :4] += 1.0

    assert (probatum.Data(earlier, symbol="BTCUSDT")[5:10].fingerprint
            != whole[5:10].fingerprint)

    pair = probatum.Data({"BTCUSDT": frame(), "ETHUSDT": frame(close=10.0)})

    assert pair["ETHUSDT"].fingerprint == probatum.Data(frame(close=10.0),
                                                        symbol="ETHUSDT").fingerprint


def test_a_volume_of_minus_zero_reads_as_zero():
    assert (probatum.Data(frame().assign(volume=-0.0), symbol="X").fingerprint
            == probatum.Data(frame().assign(volume=0.0), symbol="X").fingerprint)
