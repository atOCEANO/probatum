"""A venue preset is a dated TOML file, read and checked as it loads: its shape and the
type of each setting by the loader, and every value by the checks the Engine runs, so
a preset passes exactly what a venue built by hand does (ADR 0028).
"""

import datetime
import re

import numpy as np
import pytest

import probatum
from probatum import venues

HEAD = """
[venue]
name = "made"
taken = "2026-10-15"
kind = "linear"
"""


def tiers(*rows):
    cells = ", ".join(f"{{ up_to = {up_to}, leverage = {leverage}, rate = {rate}, "
                      f"amount = {amount} }}" for up_to, leverage, rate, amount in rows)
    return f"[instruments.BTCUSDT]\ntiers = [{cells}]\n"


def test_every_preset_loads_named_and_dated():
    for name in venues._names():
        preset = venues._preset(name)
        assert preset["name"] == name
        assert datetime.date.fromisoformat(preset["taken"])


def test_the_frictionless_preset_charges_a_round_trip_nothing():
    settings = venues._settings(venues._preset("frictionless"), ["linear:BTCUSDT"])
    candles  = np.array([(100.0, 101.0, 99.0, 100.0, 1e6)] * 3)
    e        = probatum._probatum.Engine(candles, report=True, **settings)

    e.reset()
    e.market_buy(1.0)
    e.step()
    e.market_sell(1.0)
    last = e.step()

    assert [fill["fee"] for fill in e.fills()] == [0.0, 0.0]
    assert last.equity == 10_000.0


@pytest.mark.parametrize("text, settings", [
    ("""
[defaults]
fees = { maker = 0.0002, taker = 0.0005 }
liquidation = "takeover"
funding = { every = 8, rate = 0.0001 }

[defaults.fills]
limit = "through"
window = 30
liquidity = { budget = 0.1 }
spread = "estimated"
slippage = { fixed = 1.0 }
impact = { sqrt = 0.1 }
sizes = "on"

[instruments.BTCUSDT]
reduce_only_exempt = true
fees = { taker = 0.0004 }
tiers = [{ up_to = 300000.0, leverage = 150.0, rate = 0.004 },
         { up_to = 800000.0, leverage = 100.0, rate = 0.005, amount = 300.0 }]
""", dict(fee_maker=0.0002, fee_taker=0.0004, liquidation="takeover", funding_hours=8,
          funding_rate=0.0001, limit="through", window=30, budget=0.1,
          spread="estimated", slippage_bps=1.0, impact=0.1, impact_model="sqrt",
          sizes=True, market="perp", reduce_only_exempt=True,
          tiers=[(300_000.0, 150.0, 0.004, 0.0), (800_000.0, 100.0, 0.005, 300.0)])),
    ("""
[defaults]
leverage = 20.0
max_open_orders = 50
liquidation = "book_then_backstop"
backstop = 0.5
funding = { every_bars = 8, rate = 0.0001 }

[defaults.fills]
spread = { fixed = 2.0 }
impact = { linear = 0.5 }

[instruments.BTCUSDT]
tiers_by = "quantity"
tiers = [{ up_to = 10.0, leverage = 50.0, rate = 0.005 },
         { up_to = 100.0, leverage = 20.0, rate = 0.01 }]
tick = 0.5
lot = 0.001
min_notional = 5.0
max_qty = 100.0
""", dict(leverage=20.0, max_open_orders=50, liquidation="book_then_backstop",
          backstop=0.5, funding_interval=8, funding_rate=0.0001, spread=2.0,
          slippage_bps=0.0, impact=0.5, impact_model="linear", sizes=False,
          market="perp", tiers_by="quantity", tick=0.5, lot=0.001, min_notional=5.0,
          max_qty=100.0, tiers=[(10.0, 50.0, 0.005, 0.0), (100.0, 20.0, 0.01, 0.0)])),
])
def test_a_preset_maps_every_setting_to_the_engines_names(text, settings):
    preset = venues._read(HEAD + text, "made")
    venues._check(preset)

    assert venues._settings(preset, ["BTCUSDT"]) == settings


def test_a_mixed_preset_charges_each_instrument_its_kinds_rates_but_a_side_of_its_own():
    text = """
[venue]
name = "made"
taken = "2026-10-15"

[defaults]
fees = { maker = 0.0002, taker = 0.0005 }

[kinds.spot]
fees = { maker = 0.001, taker = 0.001 }

[instruments."spot:BTCUSDT"]
fees = { taker = 0.0008 }
[instruments."linear:BTCUSDT"]
"""

    preset = venues._read(text, "made")
    venues._check(preset)
    settings = venues._settings(preset, ["spot:BTCUSDT", "linear:BTCUSDT"])

    assert settings["instruments"][0]["fee_maker"] == 0.001

    candles = np.full((2, 2, 5), 60_000.0)
    played  = probatum._probatum.run(
        candles, dict(settings, quote={"spot": 40_000.0, "margin": 40_000.0}),
        {0: [("buy", {"instrument": 0, "size": 0.5}),
             ("sell", {"instrument": 1, "size": 0.5})]})

    assert [fill["fee"] for fill in played["fills"]] == pytest.approx([24.0, 15.0])


def test_spot_markets_alone_take_no_liquidation_model():
    text = """
[venue]
name = "made"
taken = "2026-10-15"

[defaults]
liquidation = "book_then_backstop"
backstop = 0.66

[instruments."spot:BTCUSDT"]
[instruments."linear:BTCUSDT"]
tiers = [{ up_to = 100.0, leverage = 10.0, rate = 0.01 }]
"""

    preset = venues._read(text, "made")
    venues._check(preset)
    settings = venues._settings(preset, ["spot:BTCUSDT"])

    assert "liquidation" not in settings and "backstop" not in settings

    probatum._probatum.Engine(np.ones((1, 5)), **settings)


def test_two_inverse_contracts_are_refused_together_since_each_runs_alone():
    preset = venues._read(HEAD.replace('"linear"', '"inverse"')
                          + "[instruments.BTCUSD]\n[instruments.ETHUSD]\n", "made")

    with pytest.raises(ValueError, match=re.escape(
            "made: an inverse contract among 2 instruments; run it alone")):
        venues._settings(preset, ["BTCUSD", "ETHUSD"])


@pytest.mark.parametrize("text, says", [
    ("[venue", "made: expected ']' at the end of a table declaration"),
    (HEAD + "[extra]\n", "made: the file: 'extra' is not a setting"),
    ("defaults = 3\n" + HEAD, "made: defaults: 3 is not a table; give a table"),
    (HEAD.replace('name = "made"\n', ""), "made: venue: name None; give the preset's"),
    (HEAD.replace('"2026-10-15"', '"tomorrow"'),
     "made: venue: taken 'tomorrow' is not a day; give the day the tables were taken"),
    (HEAD.replace('"2026-10-15"', '"20261015"'),
     "made: venue: taken '20261015' is not a day; give the day the tables were taken"),
    (HEAD.replace('"linear"', '"futures"'),
     "made: venue: kind 'futures'; use spot, linear, inverse"),
    (HEAD + "[defaults]\nfee = 0.001\n", "made: defaults: 'fee' is not a setting"),
    (HEAD + "[kinds.spot]\nfees = { taker = -5.0 }\n",
     "made: kinds: spot, in a preset of one kind, linear; give every instrument's "
     "settings in defaults"),
    (HEAD + "[defaults.fills]\nbudget = 0.1\n",
     "made: defaults.fills: 'budget' is not a setting"),
    (HEAD + '[defaults.fills]\nspread = "wide"\n',
     "made: defaults.fills.spread: 'wide'; use 'none' or 'estimated' or { fixed = n }"),
    (HEAD + "[defaults.fills]\nimpact = { cubic = 1.0 }\n",
     "use 'none' or { linear = n } or { sqrt = n }"),
    (HEAD + "[defaults.fills]\nsizes = true\n",
     "made: defaults.fills.sizes: True; use 'on' or 'off'"),
    (HEAD + "[defaults.fills]\nwindow = 0\n", "made: window: 0; give the number of bars"),
    (HEAD + "[defaults.fills]\nliquidity = { budget = 0.0 }\n",
     "made: budget: 0; give the share of the trailing volume"),
    (HEAD + '[defaults.fills]\nlimit = "cross"\n',
     'made: limit: "cross" is not a fill rule; use touch or through'),
    (HEAD + '[defaults]\nfunding = "daily"\n',
     "made: defaults.funding: 'daily'; use 'none', 'data'"),
    (HEAD + "[defaults]\nfunding = { every = 8, every_bars = 3 }\n",
     "give every, in hours, or every_bars, and the rate"),
    (HEAD + "[defaults]\nfunding = { every = 8 }\n",
     "give every, in hours, or every_bars, and the rate"),
    (HEAD + "[defaults]\nfunding = { every_bars = -3, rate = 0.0 }\n",
     "made: funding_interval: -3; pass the bars between events, or 0 for none"),
    (HEAD + "[defaults]\nmax_open_orders = -1\n",
     "made: max_open_orders must be between 1 and 4096, got -1"),
    (HEAD + "[defaults]\nliquidation = 3\n",
     "made: defaults.liquidation: 3 is not a word; give its name"),
    (HEAD + "[defaults]\nfees = { maker = -1.5, taker = 0.0005 }\n",
     "made: fee_maker must be greater than -1.0"),
    (HEAD + '[defaults]\nfees = { maker = "low", taker = 0.0005 }\n',
     "made: defaults.fees.maker: 'low' is not a number; give a number"),
    (HEAD + "[defaults]\nleverage = true\n",
     "made: defaults.leverage: True is not a number"),
    (HEAD + "[instruments.BTCUSDT]\nfees = { taker = -2.0 }\n",
     "made: BTCUSDT: fee_taker must be greater than -1.0"),
    (HEAD + "[instruments.BTCUSDT]\ntierz = []\n",
     "made: BTCUSDT: 'tierz' is not a setting"),
    (HEAD + '[instruments.BTCUSDT]\nreduce_only_exempt = "yes"\n',
     "made: BTCUSDT.reduce_only_exempt: 'yes'; give true or false"),
    (HEAD + "[instruments.BTCUSDT]\nsigfigs = 5.0\n",
     "made: BTCUSDT.sigfigs: 5.0 is not a whole number"),
    (HEAD + "[instruments.BTCUSDT]\ntiers = 3\n",
     "made: BTCUSDT.tiers: 3 is not a list; give a list of tables of up_to, leverage, "
     "rate, amount"),
    (HEAD + "[instruments.BTCUSDT]\ntiers = [{ up_to = 100.0, leverage = 10.0 }]\n",
     "made: BTCUSDT.tiers[0]: no rate; give up_to, leverage, rate, amount"),
    (HEAD + tiers((100.0, 10.0, 0.01, 0.0), (50.0, 5.0, 0.02, 0.0)),
     "made: BTCUSDT: tiers[1]: up_to 50 is not above the bound before it, 100"),
    (HEAD + tiers((100.0, 10.0, 0.01, 0.0), (200.0, 20.0, 0.01, 0.0)),
     "made: BTCUSDT: tiers[1]: leverage 20 is above the tier before it, 10"),
    (HEAD + tiers((100.0, 10.0, 0.02, 0.0), (200.0, 5.0, 0.01, 0.0)),
     "made: BTCUSDT: tiers[1]: a maintenance rate of 0.01 is below the tier before it"),
    (HEAD + tiers((100.0, 0.0, 1.0, 0.0)),
     "made: BTCUSDT: tiers[0]: a maintenance rate of 1 is not below one"),
    (HEAD + tiers((100.0, 10.0, 0.1, 0.0)),
     "made: BTCUSDT: tiers[0]: a maintenance rate of 0.1 reaches the initial rate"),
    (HEAD + tiers((100.0, 10.0, 0.01, 0.0), (200.0, 5.0, 0.02, 5.0)),
     "made: BTCUSDT: tiers[1]: an amount of 5 breaks continuity"),
    (HEAD + tiers((100.0, 10.0, 0.01, 1.0)),
     "made: BTCUSDT: tiers[0]: an amount of 1 in the first tier"),
    (HEAD + '[defaults]\nliquidation = "book_then_backstop"\nbackstop = 0.66\n'
     + tiers((100.0, 10.0, 0.0, 0.0)),
     "made: BTCUSDT: tiers: no maintenance, where the venue books then backstops"),
    (HEAD + '[defaults]\nliquidation = "book_then_backstop"\nbackstop = 0.66\n'
     + tiers((100.0, 10.0, 0.01, 0.0)) + "[instruments.ETHUSDT]\n",
     "made: ETHUSDT: tiers: no maintenance, where the venue books then backstops"),
    (HEAD + "[instruments.BTCUSDT]\ntick = 0.5\nsigfigs = 5\ndecimals = 1\n",
     "made: BTCUSDT: sigfigs: 5 beside a tick of 0.5; give one price rule"),
    (HEAD + "[instruments.BTCUSDT]\n[instruments.ETHUSDT]\ntick = 0.5\nsigfigs = 5\n"
     "decimals = 1\n", "made: ETHUSDT: sigfigs: 5 beside a tick of 0.5"),
    (HEAD + "[instruments.BTCUSDT]\nlot = 1.0\nmax_qty = 0.5\n",
     "made: BTCUSDT: max_qty: 0.5; give a maximum of at least the lot of 1"),
    (HEAD.replace('"linear"', '"inverse"') + "[instruments.BTCUSD]\nsigfigs = 5\n"
     "decimals = 1\n", "made: BTCUSD: sigfigs: 5 for an inverse contract"),
    (HEAD.replace('"linear"', '"inverse"') + "[instruments.BTCUSD]\n"
     '[instruments.ETHUSD]\n[defaults.fills]\nlimit = "cross"\n',
     'made: limit: "cross" is not a fill rule; use touch or through'),
    (HEAD.replace('kind = "linear"\n', "") + "[instruments.BTCUSDT]\n",
     "made: 'BTCUSDT' names no kind; name an instrument of a preset that mixes kinds"),
])
def test_a_preset_the_loader_cannot_trust_is_refused(text, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        venues._check(venues._read(text, "made"))


def test_a_preset_it_does_not_hold_is_refused_naming_those_it_does():
    with pytest.raises(ValueError, match=re.escape(
            "venue: 'binance_futures' is not a preset; use one of")):
        venues._preset("binance_futures")


def test_a_combined_preset_holds_its_parts_each_at_its_own_rates():
    binance = venues._preset("binance")

    assert binance["combines"] == ["binance_spot", "binance_perp"]

    parts = {kind: venues._preset(f"binance_{name}")["instruments"]
             for kind, name in (("spot", "spot"), ("linear", "perp"))}

    assert binance["instruments"] == {f"{kind}:{key}": table
                                      for kind, tables in parts.items()
                                      for key, table in tables.items()}

    settings = venues._settings(binance, ["spot:BTCUSDT", "linear:BTCUSDT"])
    spot, perp = settings["instruments"]

    assert (spot["market"], spot["fee_maker"], spot["fee_taker"]) == (
        "spot", 0.001, 0.001)
    assert spot["tick"] > 0.0 and spot["lot"] > 0.0
    assert (perp["market"], perp["fee_maker"], perp["fee_taker"]) == (
        "perp", 0.0002, 0.0005)
    assert perp["reduce_only_exempt"] and perp["tiers"]


@pytest.mark.parametrize("parts, says", [
    ([("spot", ""), (None, "")],
     "made: combines 'part1'; combine presets of one kind each"),
    ([("spot", ""), ("spot", "")],
     "made: combines 'part1'; combine presets of one kind each"),
    ([("spot", '[defaults]\nliquidation = "takeover"\n'),
      ("linear", '[defaults]\nliquidation = "book_then_backstop"\n')],
     "made: combines 'part1', whose liquidation 'book_then_backstop' is not the "
     "'takeover' another gives; give the parts one liquidation"),
])
def test_a_combination_is_refused_unless_its_parts_are_of_one_kind_each_and_agree(
        parts, says):
    def part(k, kind, text):
        head = HEAD.replace('name = "made"', f'name = "part{k}"')
        return venues._read(head.replace('kind = "linear"\n', "" if kind is None
                                         else f'kind = "{kind}"\n') + text, f"part{k}")

    combining = venues._read(HEAD.replace('kind = "linear"\n',
                                          'combines = ["part0", "part1"]\n'), "made")

    with pytest.raises(ValueError, match=re.escape(says)):
        venues._combine(combining, [part(k, *each) for k, each in enumerate(parts)])


@pytest.mark.parametrize("text, says", [
    (HEAD.replace('kind = "linear"\n', 'combines = ["a", "b"]\n') + "[defaults]\n",
     "made: venue: combines ['a', 'b'] beside settings of its own; give a combining "
     "preset its venue alone"),
    (HEAD + 'combines = ["a", "b"]\n',
     "made: venue: combines ['a', 'b'] beside settings of its own"),
    (HEAD.replace('kind = "linear"\n', 'combines = "a"\n'),
     "made: venue: combines 'a'; give the list of the presets it combines"),
])
def test_a_combining_preset_is_refused_unless_it_names_its_parts_alone(text, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        venues._read(text, "made")
