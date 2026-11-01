"""Build every chart the documentation shows, from data that cannot move.

The pictures on the pages are the output of the snippets printed beside them, so
this is where those snippets run. No simulation logic lives here and nothing is
drawn by hand: it loads the frozen parquet files of the sample dataset as Data, works
the features out the way a reader would, runs the strategies through
``probatum.backtest`` and calls ``Result.chart`` and ``Data.chart``. Every chart reads
the same published five-minute bars, so a rerun draws the same picture on any machine.

A call a page prints is made here call for call, but for the ``height`` that sizes
its shot. Run it through the charts stage, which also takes the shots, in the dark
and the light scheme:

  docker build --target charts -t probatum-charts .
  docker run --rm -v "<sample-market-data>/data:/data:ro" \
      -v "${PWD}/.Documentation:/out" probatum-charts
"""

import json
import pathlib

import numpy
import pandas

import probatum
from probatum.chart.marks import (
    Background, Band, Histogram, Level, Line, Markers, Panel, ramp,
)

DATA = pathlib.Path("/data")
OUT  = pathlib.Path("/work/charts")

GREEN = "#2fe0a8"
RED   = "#ff5470"
GREY  = "#8b97a5"

AHEAD = 26

WIDE   = 1440
NARROW = 1180

SHOTS = {}


def vam(close, n):
    logret = numpy.log(close / probatum.ta.shift(close))
    return numpy.log(close / probatum.ta.shift(close, n)) / (
        probatum.ta.stdev(logret, n) * numpy.sqrt(n))


def midpoint(data, n):
    return (probatum.ta.highest(data.high, n) + probatum.ta.lowest(data.low, n)) / 2.0


def held_by(result, data):
    # the bars the run was in the market, a round trip still open at the end running
    # to the last, its bars the run's placed on the Data's rows
    rows = numpy.searchsorted(data.time, result.time)
    held = numpy.zeros(len(data), dtype=bool)
    for trade in result.trades:
        last = rows[-1] if trade["open"] else rows[trade["close_bar"]]
        held[rows[trade["open_bar"]]:last + 1] = True

    return held


def keep(name, chart, width, caption, below=0, open_table=False):
    # the height is recorded here and read by the shooter, since a chart sizes itself
    # to its window and a shot taken at another height pads or squeezes the plot
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(str(OUT / f"{name}.html"))

    SHOTS[name] = {
        "height":  chart.spec()["height"],
        "width":   width,
        "caption": caption,
    }

    # the table panel opens under the plot and is not part of the chart's own
    # height, so a shot that has to include it asks for the room
    if below:
        SHOTS[name]["below"] = below
    if open_table:
        SHOTS[name]["open"] = True


def spot(symbol):
    return pandas.read_parquet(DATA / f"binance_spot_{symbol}_5m.parquet")


class Channel(probatum.Strategy):
    """Hold above the channel basis while MACD agrees."""


    def __init__(self, length=96, fast=12, slow=26, signal=9, trend=480):
        self.length, self.fast, self.slow = length, fast, slow
        self.signal, self.trend = signal, trend


    def init(self, data):
        close        = data.close
        bands        = probatum.ta.bbands(close, self.length, 2.0)
        self.basis   = bands.middle
        self.upper   = bands.upper
        self.lower   = bands.lower
        self.uptrend = close > probatum.ta.sma(close, self.trend)
        self.hist    = probatum.ta.macd(close, self.fast, self.slow, self.signal).histogram
        self.blocked = numpy.zeros(len(data), dtype=bool)
        self.warmup  = self.trend


    def next(self, bar, account):
        i = bar.i

        # in on a break of the upper edge, out on a loss of the basis. The two
        # thresholds are apart on purpose: entering and leaving on one line whipsaws
        # on every bar that straddles it
        if account.position == 0.0 and self.uptrend[i] and bar.close > self.upper[i]:
            if self.hist[i] > 0.0:
                account.target(1.0)
            else:
                self.blocked[i] = True
        elif account.position > 0.0 and bar.close < self.basis[i]:
            account.close()


    def marks(self):
        return [
            Background(self.uptrend, fill="#2fe0a814"),
            Band(self.upper, self.lower, f"channel {self.length}", fill="#8b97a514"),
            Line(self.basis, f"basis {self.length}", color=GREY, width=1,
                 style="dashed"),
            Markers(self.blocked, shape="arrow_down", offset=22),
            Histogram(self.hist, f"MACD {self.fast}/{self.slow}/{self.signal}",
                      panel="momentum",
                      color=numpy.where(self.hist >= 0.0, GREEN, RED)),
            Level(0.0, panel="momentum", style="dotted"),
        ]


class SmaCross(probatum.Strategy):
    """The plain cross the guide's examples are drawn against."""


    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow


    def init(self, data):
        self.fast_line = probatum.ta.sma(data.close, self.fast)
        self.slow_line = probatum.ta.sma(data.close, self.slow)
        self.warmup    = self.slow


    def next(self, bar, account):
        up = self.fast_line[bar.i] > self.slow_line[bar.i]
        if account.position == 0.0 and up:
            account.target(1.0)
        elif account.position > 0.0 and not up:
            account.close()


    def marks(self):
        return [
            Line(self.fast_line, f"SMA {self.fast}"),
            Line(self.slow_line, f"SMA {self.slow}", style="dashed"),
        ]


def gallery(raw):
    data     = raw.resample("1h")["2025":"2025"]
    breakout = probatum.backtest(Channel(), data, "binance_spot")

    # forty days of the year, the stretch that holds the most round trips
    keep("run", breakout.chart(
        breakout.policy.marks(),
        focus=(5_568, 6_528),
        title="A channel breakout with a MACD filter, BTCUSDT 1h, binance spot",
        height=760,
    ), WIDE, "a run framed on forty days")
    keep("table", breakout.chart(
        breakout.policy.marks(),
        focus=(5_568, 6_528),
        title="A channel breakout with a MACD filter, BTCUSDT 1h, binance spot",
        height=620,
    ), WIDE, "the same run with its round trips' table open", below=300,
        open_table=True)

    data = raw.resample("1h")["2025-10":"2025-10"]
    keep("price-hollow", data.chart(price="hollow", focus=160, height=420), NARROW,
         'price="hollow"')
    keep("price-bars", data.chart(price="bars", focus=160, height=420), NARROW,
         'price="bars"')
    keep("price-line", data.chart(price="line", focus=160, height=420), NARROW,
         'price="line"')

    data  = raw.resample("1h")["2025-09":"2025-11"]
    close = data.close
    bands = probatum.ta.bbands(close, 48, 2.0)
    rsi   = probatum.ta.rsi(close, 14)
    hist  = probatum.ta.macd(close, 12, 26, 9).histogram
    keep("indicators", data.chart(
        Band(bands.upper, bands.lower, "bands 48", fill="#4d9fff14"),
        Line(probatum.ta.sma(close, 200), "SMA 200", color="#8b97a5", width=1),
        Line(rsi, "RSI 14", panel="RSI"),
        Level(70.0, panel="RSI", style="dotted"),
        Level(30.0, panel="RSI", style="dotted"),
        Histogram(hist, "MACD 12/26/9", panel="MACD",
                  color=numpy.where(hist >= 0.0, "#2fe0a8", "#ff5470")),
        focus=400,
        height=720,
    ), WIDE, "arrays of your own on the price and on panels of their own")

    frames   = {symbol: spot(symbol) for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}
    universe = probatum.Data(frames, market="spot").resample("1h")["2025":"2025"]

    weights = numpy.where(universe.closes > probatum.ta.sma(universe.closes, 200),
                          1.0 / universe.closes.shape[1], 0.0)
    book    = probatum.backtest(weights, universe, "binance_spot")
    keep("universe", book.chart(
        instrument="ETHUSDT", focus=24 * 60,
        title="Four pairs above their 200-hour average, ETHUSDT of the four",
        height=660,
    ), WIDE, "one instrument of a four-pair account")


def projection(raw):
    data = raw.resample("4h")["2024":"2025"]

    # senkou a and b belong AHEAD bars later than they were computed, so the
    # shift is done here and the array is longer than the bars
    lead     = numpy.full(AHEAD, numpy.nan)
    senkou_a = numpy.concatenate([lead, (midpoint(data, 9) + midpoint(data, 26)) / 2.0])
    senkou_b = numpy.concatenate([lead, midpoint(data, 52)])

    keep("chart-projection", data.chart(
        Band(senkou_a, senkou_b, "cloud", fill="#2fe0a833"),
        future=26,
        focus=180,
        height=620,
    ), WIDE, "a cloud drawn twenty-six bars past the last candle")


def documented_examples(raw):
    data  = raw.resample("1h")["2025-01":"2025-06"]
    close = data.close

    rsi          = probatum.ta.rsi(close, 14)
    bands        = probatum.ta.bbands(close, 20, 2.0)
    upper, lower = bands.upper, bands.lower

    atr      = probatum.ta.atr(data.high, data.low, close, 14)
    risk_off = atr > probatum.ta.sma(atr, 100)

    fast, slow = probatum.ta.sma(close, 20), probatum.ta.sma(close, 50)
    crossed    = probatum.ta.crossover(fast, slow)
    trending   = close > probatum.ta.sma(close, 50)

    fast_z = vam(close, 24)
    slow_z = vam(close, 168)

    vam72 = vam(close, 72)
    shade = ramp(vam72, colors=[RED, GREY, GREEN], domain=(-3.0, 3.0))

    # two sigma either way, so both halves of the label map fire on this window
    regime  = numpy.where(vam72 > 2.0, "hi", numpy.where(vam72 < -2.0, "lo", "flat"))
    forward = numpy.concatenate([close[4:] / close[:-4] - 1.0, numpy.full(4, numpy.nan)])

    result   = probatum.backtest(SmaCross(fast=20, slow=50), data, "binance_spot")
    strategy = result.policy
    held     = held_by(result, data)
    worst    = min(result.trades, key=lambda trade: trade["pnl"])

    keep("ex-line", data.chart(
        Line(probatum.ta.sma(data.close, 50), "SMA 50"),
        height=520,
    ), NARROW, "a moving average over the candles")
    keep("ex-oscillator", data.chart(
        Line(rsi, "RSI 14", panel="rsi"),
        Level(70.0, panel="rsi", style="dotted"),
        Level(30.0, panel="rsi", style="dotted"),
        height=620,
    ), NARROW, "an oscillator on its own panel")
    keep("ex-band", data.chart(
        Band(upper, lower, "bollinger", fill="#8b97a51a"),
        height=520,
    ), NARROW, "a band between two arrays")
    keep("ex-background", data.chart(
        Background(risk_off, "risk off", fill="#ff547026"),
        height=520,
    ), NARROW, "a shaded regime behind the candles, named in the legend")
    keep("ex-markers", data.chart(
        Markers(crossed, shape="arrow_up", offset=-16),
        height=520,
    ), NARROW, "one marker per bar a mask is true on")
    keep("ex-candle-color", data.chart(
        candle_color=numpy.where(trending, "#4d9fff", None),
        height=520,
    ), NARROW, "the candles tinted by a condition")
    keep("ex-panels", data.chart(
        Line(fast_z, "z 24", panel="z24"),
        Line(slow_z, "z 168", panel="z168"),
        panels=[Panel("z24", range=(-5.0, 5.0)), Panel("z168", range=(-5.0, 5.0))],
        height=760,
    ), NARROW, "two panels pinned to the same range")
    keep("ex-feature", data.chart(
        Line(vam(data.close, 72), "vam 72", panel="vam"),
        Level(0.0, panel="vam", style="dotted"),
        height=620,
    ), NARROW, "your own feature beside the price")
    keep("ex-verdict", data.chart(
        Background(regime, fill={"hi": "#2fe0a81f", "lo": "#ff54701f"}),
        Line(vam72, "vam 72", panel="vam", color=shade),
        Band(vam72, 2.0, only="above", panel="vam", fill=("#2fe0a800", "#2fe0a859")),
        Line(forward, "fwd 4h", panel="fwd"),
        Level(0.0, panel="fwd", style="dotted"),
        candle_color=numpy.where(numpy.abs(vam72) > 2.0, "#4d9fff", None),
        title="vam 72 extremes against the next four hours",
        height=820,
    ), NARROW, "a feature judged against what came next")
    keep("ex-zones", result.chart(
        Line(rsi, "RSI 14", panel="momentum"),
        Level(70.0, panel="momentum", style="dotted"),
        Band(rsi, 70.0, only="above", panel="momentum", fill=("#ff547000", "#ff547059")),
        Band(rsi, 30.0, only="below", panel="momentum", fill=("#2fe0a800", "#2fe0a859")),
        height=820,
    ), NARROW, "conditional fills where a line passes its level")
    keep("ex-held", result.chart(
        strategy.marks(),
        candle_color=numpy.where(held, "#4d9fff", None),
        height=720,
    ), NARROW, "the bars the strategy was holding, tinted")
    keep("ex-focus", result.chart(
        strategy.marks(),
        focus=worst,
        height=660,
    ), NARROW, "the worst trade framed")
    keep("ex-notes", result.chart(
        title="each stretch traded on parameters fitted only on the bars before it",
        notes=[
            ["window", "fitted on", "traded", "sharpe"],
            [1, "2025-01-01 to 2025-06-30", "2025-07-01 to 2025-09-30", 1.94],
            [2, "2025-04-01 to 2025-09-30", "2025-10-01 to 2025-12-31", 0.71],
        ],
        height=440,
    ), NARROW, "a table of the caller's under the plot", below=300, open_table=True)


def main():
    raw  = probatum.Data(spot("BTCUSDT"), symbol="BTCUSDT", market="spot")
    seen = raw.provenance
    print(f"{len(raw):,} five-minute bars, {seen['first']} to {seen['last']}\n")

    gallery(raw)
    projection(raw)
    documented_examples(raw)

    (OUT / "manifest.json").write_text(
        json.dumps(SHOTS, indent=2, sort_keys=True), encoding="utf-8"
    )

    for name in sorted(SHOTS):
        size = (OUT / f"{name}.html").stat().st_size
        print(f"{name:<18} {SHOTS[name]['width']:>5}x{SHOTS[name]['height']:<4} "
              f"{size / 1_000_000:5.1f} MB")

    print(f"\n{len(SHOTS)} charts written to {OUT}")


if __name__ == "__main__":
    main()
