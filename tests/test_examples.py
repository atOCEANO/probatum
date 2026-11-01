"""Run the Charts page's examples instead of only reading them.

``test_docs.py`` parses every fenced block and checks that each keyword names a real
parameter, and runs none of them, because most stand on bars and a strategy the page
describes in prose. Parsing cannot see an example that composes two calls which could
never work together, or reads an attribute no Result has.

So the page gets a prelude, and every block that stands on it is executed. The names
come from what the page itself binds where it binds them, so a block is run against
the values a reader would have in hand at that point. ``show()`` needs a notebook and
``save()`` a path, so both are swapped for ``spec()``, which builds the whole document
and raises everything a real call would; nothing here is asserted about pixels.
"""

import pathlib
import re

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import probatum
from probatum import chart
from probatum.chart.marks import (
    Background, Band, Histogram, Level, Line, Marker, Markers, Panel, Recorder,
    at_bar, at_next, ramp,
)

# the gate stages copy the doc set to /docs, and a source checkout, which is what
# CI runs from, has it beside tests
_CANDIDATES = (
    pathlib.Path("/docs/Charts.md"),
    pathlib.Path(__file__).resolve().parent.parent / ".Documentation" / "Charts.md",
)
PAGE = next((path for path in _CANDIDATES if path.is_file()), None)

# the browser stage copies only tests/, so this skips there rather than failing
# on a path that was never meant to exist
if PAGE is None:
    pytest.skip("the documentation is not here", allow_module_level=True)

# A block whose point is prose rather than a call, keyed by a phrase it contains
# so the list survives the page being edited around it. each one says why
NOT_EXECUTABLE = {
    "probatum.Data(frame,":             "the page's own setup, which prelude() binds from frame()",
    "class SmaCross":                   "declares the strategy the page stands on",
    "from probatum.chart.marks import": "the import the prelude already made",
}


def blocks():
    text = PAGE.read_text(encoding="utf-8")

    found, buffer, inside = [], [], False
    for line in text.splitlines():
        if line.startswith("```python"):
            inside, buffer = True, []
        elif line.startswith("```") and inside:
            found.append("\n".join(buffer))
            inside = False
        elif inside:
            buffer.append(line)

    return found


def frame(n=300, level=20_000.0, phase=0.0):
    # the swing has to be big enough against the drift for a 20/50 cross to fire,
    # or half the page stands on a run with no trades in it
    step  = np.sin(np.arange(n, dtype=np.float64) / 9.0 + phase) * level * 0.13
    close = level + np.arange(n, dtype=np.float64) * level * 6e-4 + step
    return pd.DataFrame(
        {
            "open":   close - 3.0,
            "high":   close + 25.0,
            "low":    close - 25.0,
            "close":  close,
            "volume": 900.0 + np.arange(n, dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
    )


class SmaCross(probatum.Strategy):
    """Long while the fast average is over the slow one."""


    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow


    def init(self, data):
        self.fast_line = probatum.ta.sma(data.close, self.fast)
        self.slow_line = probatum.ta.sma(data.close, self.slow)
        self.warmup    = self.slow
        self.log       = Recorder(data)


    def marks(self):
        return [
            Line(self.fast_line, f"SMA {self.fast}"),
            Line(self.slow_line, f"SMA {self.slow}", style="dashed"),
        ]


    def next(self, bar, account):
        self.log.at_bar(bar, leverage=account.leverage)

        if self.fast_line[bar.i] > self.slow_line[bar.i] and account.position <= 0.0:
            account.buy(0.1)
        elif self.fast_line[bar.i] < self.slow_line[bar.i] and account.position > 0.0:
            account.close()


def vam(close, n):
    logret = np.log(close / probatum.ta.shift(close))
    return np.log(close / probatum.ta.shift(close, n)) / (
        probatum.ta.stdev(logret, n) * np.sqrt(n))


def midpoint(data, n):
    return (probatum.ta.highest(data.high, n) + probatum.ta.lowest(data.low, n)) / 2.0


def prelude():
    """Every name the page uses, bound the way the page binds it."""
    data     = probatum.Data(frame(), symbol="BTCUSDT", market="spot")
    close    = data.close
    result   = probatum.backtest(SmaCross(fast=20, slow=50), data, "binance_spot")
    strategy = result.policy

    fast, slow = probatum.ta.sma(close, 20), probatum.ta.sma(close, 50)
    bands      = probatum.ta.bbands(close, 20, 2.0)
    atr        = probatum.ta.atr(data.high, data.low, close, 14)
    vam72      = vam(close, 72)
    lead       = np.full(26, np.nan)
    universe   = probatum.Data({"BTCUSDT": frame(), "ETHUSDT": frame(level=2_000.0,
                                                                    phase=1.0)},
                               market="spot")

    runs = {"20/50": result, "20/60": result, "24/50": result}

    return {
        "numpy": np, "pandas": pd, "probatum": probatum,
        "Background": Background, "Band": Band, "Histogram": Histogram,
        "Level": Level, "Line": Line, "Marker": Marker, "Markers": Markers,
        "Panel": Panel, "Recorder": Recorder, "ramp": ramp,
        "at_bar": at_bar, "at_next": at_next, "SmaCross": SmaCross, "vam": vam,
        "data": data, "close": close, "universe": universe,
        "strategy": strategy, "result": result, "win": result, "breakout": result,
        "neighbours": runs, "runs": runs,
        "fast": fast, "slow": slow,
        "rsi": probatum.ta.rsi(close, 14),
        "upper": bands.upper, "lower": bands.lower,
        "crossed": probatum.ta.crossover(fast, slow),
        "risk_off": atr > probatum.ta.sma(atr, 100),
        "trending": close > probatum.ta.sma(close, 50),
        "squeeze": probatum.ta.stdev(close, 20) / close,
        "fast_z": vam(close, 24), "slow_z": vam(close, 168),
        "vam24": vam(close, 24), "vam72": vam72, "vam168": vam(close, 168),
        "shade": ramp(vam72, colors=["#ff5470", "#8b97a5", "#2fe0a8"],
                      domain=(-3.0, 3.0)),
        "regime": np.where(vam72 > 2.0, "hi", np.where(vam72 < -2.0, "lo", "flat")),
        "forward": np.concatenate([close[4:] / close[:-4] - 1.0, np.full(4, np.nan)]),
        "recorded": np.linspace(0.0, 1.0, len(data) - 1),
        "GREEN": "#2fe0a8", "RED": "#ff5470",
        "senkou_a": np.concatenate([lead, (midpoint(data, 9) + midpoint(data, 26)) / 2.0]),
        "senkou_b": np.concatenate([lead, midpoint(data, 52)]),
    }


def test_the_page_is_not_quietly_empty():
    assert PAGE.exists()
    assert len(blocks()) > 35


def test_every_example_that_stands_on_the_prelude_actually_runs():
    # show() wants a notebook and save() wants a path, so both become spec(),
    # which builds the entire document and raises everything a real call raises.
    # A block that sets the session's look sets it for every later chart, so the
    # look the page found is put back once the page has run
    ran, skipped, failures = 0, [], []
    found = chart.defaults()

    try:
        for number, source in enumerate(blocks()):
            why = next((r for k, r in NOT_EXECUTABLE.items() if k in source), None)
            if why is not None:
                skipped.append((number, why))
                continue

            code      = re.sub(r"\.show\(\s*[^)]*\)", ".spec()", source)
            code      = re.sub(r"\.save\(\s*[^)]*\)", ".spec()", code)
            namespace = prelude()

            try:
                exec(compile(code, f"Charts.md block {number}", "exec"), namespace)
            except Exception as exc:
                failures.append(
                    f"\n--- block {number}: {type(exc).__name__}: {exc}\n{source}"
                )
            else:
                ran += 1
    finally:
        chart._DEFAULTS.update(found)

    assert not failures, "\n".join(failures)
    assert ran >= 35, f"only {ran} blocks ran, {len(skipped)} skipped"
