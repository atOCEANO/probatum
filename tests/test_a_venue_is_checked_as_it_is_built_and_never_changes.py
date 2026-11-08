"""A Venue holds a venue's rules as one immutable, hashable object: a hand-built one
starts from the plain option in each row, the presets are Venues, and a variant made
with `replace` merges its tables into the venue's own and passes the checks every
preset passes, the Engine's among them, refused in the terms it was given in (ADRs
0002, 0007, 0028).
"""

import copy
import datetime
import pickle
import re

import numpy as np
import pytest

import probatum
from probatum import ProbatumAttributeError, ProbatumTypeError, Venue, VenueError


def test_a_venue_built_by_hand_starts_from_the_plain_option_in_each_row():
    plain = Venue().to_dict()

    assert plain["balance"] == 10_000.0 and plain["intrabar"] == "pessimistic"
    assert plain["fees"] == {"maker": 0.0002, "taker": 0.0006}
    assert (plain["leverage"], plain["liquidation"], plain["max_open_orders"],
            plain["funding"]) == (10.0, "takeover", 8, "none")
    assert plain["fills"] == {
        "limit":     "touch",
        "window":    24,
        "liquidity": {"budget": 1.0},
        "spread":    "none",
        "slippage":  "none",
        "impact":    "none",
        "sizes":     "off",
    }
    assert plain["backstop"] is plain["liquidation_fee"] is plain["tick"] is None
    assert (plain["kinds"], plain["instruments"], plain["name"]) == ({}, {}, None)

    venue = Venue(fees={"taker": 0.0005}, fills={"sizes": "on"})

    assert venue.to_dict()["fees"] == {"maker": 0.0002, "taker": 0.0005}
    assert venue.to_dict()["fills"]["limit"] == "touch"
    assert Venue(leverage=None, fees={"maker": None}) == Venue()

    tables = Venue(kind="linear", instruments={"BTCUSDT": {"tick": 0.1, "lot": None}})

    assert tables.to_dict()["instruments"] == {"BTCUSDT": {"tick": 0.1}}


def test_the_presets_are_venues_named_and_dated():
    perp = probatum.venues.bybit_perp

    assert isinstance(perp, Venue) and perp is probatum.venues.bybit_perp
    assert (perp.name, perp.kind) == ("bybit_perp", "linear")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", perp.taken)
    assert perp.to_dict()["fees"] == {"maker": 0.0002, "taker": 0.00055}
    assert perp.to_dict()["instruments"]["BTCUSDT"]["tiers"]

    both = probatum.venues.bybit

    assert both.kind is None and "spot:BTCUSDT" in both.to_dict()["instruments"]
    assert {"bybit_perp", "frictionless", "okx"} <= set(dir(probatum.venues))

    with pytest.raises(ProbatumAttributeError, match="has no attribute "
                                                     "'binance_futures'"):
        probatum.venues.binance_futures


def test_a_venue_never_changes_and_is_equal_to_one_with_the_same_settings():
    tables = {"BTCUSDT": {"tick": 0.1, "lot": 0.001}}
    venue  = Venue(kind="linear", instruments=tables, leverage=5)

    tables["BTCUSDT"]["tick"] = 7.0
    venue.to_dict()["instruments"]["BTCUSDT"]["lot"] = 9.0

    assert venue.to_dict()["instruments"] == {"BTCUSDT": {"tick": 0.1, "lot": 0.001}}

    with pytest.raises(ProbatumAttributeError, match=re.escape(
            "leverage: a Venue does not change; make a variant with "
            "replace(leverage=...)")):
        venue.leverage = 3.0
    with pytest.raises(ProbatumAttributeError, match=re.escape(
            "foo: not a setting of a venue, and a venue does not change")):
        venue.foo = 1

    same = Venue(
        kind        = "linear",
        instruments = {"BTCUSDT": {"lot": 0.001, "tick": 0.1}},
        leverage    = 5.0,
    )

    assert same == venue and hash(same) == hash(venue) and len({same, venue}) == 1
    assert venue != Venue(kind="linear", instruments=tables, leverage=5.0)
    assert Venue(**venue.to_dict()) == venue and venue.replace() is venue
    assert copy.deepcopy(venue) is venue
    assert pickle.loads(pickle.dumps(venue)) == venue


def test_settings_equal_in_value_make_one_venue_however_they_were_written():
    row   = {"up_to": 1e6, "leverage": 20.0, "rate": 0.005}
    one   = Venue(kind="linear", instruments={"BTCUSDT": {"tiers": [row]}})
    other = Venue(
        kind        = "linear",
        instruments = {"BTCUSDT": {"tiers": [dict(row, amount=0.0)]}},
    )

    assert one == other and one.to_dict()["instruments"]["BTCUSDT"]["tiers"] == [
        dict(row, amount=0.0)]
    assert Venue(leverage=-0.0)       == Venue(leverage=0.0)
    assert hash(Venue(leverage=-0.0)) == hash(Venue(leverage=0.0))
    assert (Venue(max_open_orders=np.int64(16), balance=np.float32(5.0))
            == Venue(max_open_orders=16, balance=5.0))
    assert Venue(taken=datetime.date(2026, 10, 16)).taken == "2026-10-16"


@pytest.mark.parametrize("name", ["bybit", "hyperliquid_perp", "okx_inverse"])
def test_a_preset_s_settings_build_the_same_venue(name):
    preset = getattr(probatum.venues, name)

    assert Venue(**preset.to_dict()) == preset
    assert pickle.loads(pickle.dumps(preset)) == preset


def test_a_variant_merges_its_tables_and_replaces_every_other_setting():
    perp    = probatum.venues.bybit_perp
    tables  = {"BTCUSDT": {"tick": 0.5, "fees": {"maker": 0.0}}, "ETHUSDT": None}
    variant = perp.replace(fees={"taker": 0.0004}, fills={"sizes": "off"},
                           max_open_orders=16, intrabar="random", instruments=tables)

    settings = variant.to_dict()

    assert settings["fees"]  == {"maker": 0.0002, "taker": 0.0004}
    assert settings["fills"] == dict(perp.to_dict()["fills"], sizes="off")
    assert (settings["max_open_orders"], settings["intrabar"]) == (16, "random")

    btc = settings["instruments"]["BTCUSDT"]

    assert btc["tick"] == 0.5 and btc["fees"] == {"maker": 0.0}
    assert btc["tiers"] == perp.to_dict()["instruments"]["BTCUSDT"]["tiers"]
    assert "ETHUSDT" not in settings["instruments"]
    assert variant.name == "bybit_perp" and variant != perp
    assert perp.to_dict()["max_open_orders"] == 8

    dropped = variant.replace(instruments={"BTCUSDT": {"tick": None}}, fees=None)

    assert "tick" not in dropped.to_dict()["instruments"]["BTCUSDT"]
    assert dropped.to_dict()["fees"] == {"maker": 0.0002, "taker": 0.0006}

    taken_over = probatum.venues.hyperliquid_perp.replace(liquidation="takeover")

    assert taken_over.to_dict()["backstop"] is None
    assert probatum.venues.bybit.replace(max_open_orders=600).to_dict()[
        "max_open_orders"] == 600

    with pytest.raises(ProbatumTypeError, match=re.escape(
            "fee_taker: not a setting of a venue; use one of balance, fees")):
        perp.replace(fee_taker=0.0004)


def test_a_variant_that_drops_a_table_the_venue_does_not_hold_is_refused():
    with pytest.raises(VenueError, match=re.escape(
            "instruments: 'BTCUSDTT': None for a table the venue does not hold; name "
            "one of its instruments")):
        probatum.venues.bybit_perp.replace(instruments={"BTCUSDTT": None})


def test_a_change_no_instrument_of_a_combined_preset_takes_is_refused():
    both = probatum.venues.bybit

    with pytest.raises(VenueError, match=re.escape(
            "fees: {'taker': 0.0}, which no instrument of bybit takes, as each of "
            "its kinds gives its own; give it in a kind's table, replace(kinds="
            "{'spot': {'fees': ...}})")):
        both.replace(fees={"taker": 0.0})

    cheaper = both.replace(kinds={"linear": {"fees": {"taker": 0.0004}}})

    assert cheaper.to_dict()["kinds"]["linear"]["fees"] == {
        "maker": 0.0002,
        "taker": 0.0004,
    }


def test_a_venue_shows_itself_as_the_code_that_builds_it():
    perp = probatum.venues.bybit_perp

    assert repr(perp) == "probatum.venues.bybit_perp"
    assert (repr(perp.replace(max_open_orders=16))
            == "probatum.venues.bybit_perp.replace(max_open_orders=16)")

    shown = repr(perp.replace(instruments={"BTCUSDT": {"tick": 0.5}}))

    assert shown.startswith("probatum.venues.bybit_perp.replace(instruments="
                            "{'BTCUSDT': ")

    for variant in (perp.replace(instruments={"BTCUSDT": {"tick": 0.5}}),
                    perp.replace(instruments={"BTCUSDT": {"tiers": None}})):
        assert eval(repr(variant), {"probatum": probatum}) == variant

    assert repr(Venue()) == "probatum.Venue()"
    assert (repr(Venue(fees={"maker": 0.0, "taker": 0.0005}, kind="spot"))
            == "probatum.Venue(fees={'maker': 0.0, 'taker': 0.0005}, kind='spot')")
    assert repr(Venue(name="bybit_perp")) == "probatum.Venue(name='bybit_perp')"


@pytest.mark.parametrize("settings, says", [
    (dict(fees={"maker": -2.0}),
     "fees.maker: -2; give a rate above -1, since a rebate cannot exceed the "
     "notional"),
    (dict(leverage=-1.0), "leverage: -1; give 0 or more"),
    (dict(liquidation_fee=1.0),
     "liquidation_fee: 1; give the rate on its notional a close at the close point "
     "pays"),
    (dict(liquidation="book_then_backstop", backstop=0.5),
     "tiers: no maintenance, where the venue books then backstops"),
    (dict(liquidation="TAKEOVER"),
     "liquidation: 'TAKEOVER'; write it in lowercase, 'takeover'"),
    (dict(fills={"spread": "wide"}),
     "fills.spread: 'wide'; use 'none' or 'estimated' or {\"fixed\": n}"),
    (dict(funding="hourly"),
     "funding: 'hourly'; use 'none', 'data', {\"every\": hours, \"rate\": r} or "
     "{\"every_bars\": n, \"rate\": r}"),
    (dict(max_open_orders=0), "max_open_orders: 0; give a whole number from 1 to 4096"),
    (dict(instruments={"BTCUSDT": {"tick": 0.1}}),
     "instruments: 'BTCUSDT' names no kind; key it as kind:SYMBOL, with kind one of "
     "spot, linear, inverse, or give the venue's kind with kind="),
    (dict(kind="linear", instruments={"BTCUSDT": {"tick": -0.1}}),
     "instruments.BTCUSDT: tick"),
    (dict(kinds={"linear": {"tick": -0.1}}), "kinds.linear: tick"),
    (dict(kind="linear", instruments={"ETHUSDT": {"delist": "soon"}}),
     "instruments.ETHUSDT.delist: 'soon' is not a time; give one such as"),
    (dict(kinds={"linear": {"delist": "2024-01-02"}}),
     "kinds.linear: 'delist' is not a setting; give fees, tiers"),
    (dict(kind="spot", liquidation_fee=7.0), "liquidation_fee: 7; give the rate"),
    (dict(kind="perp"), "kind: 'perp'; use spot, linear, inverse, or None"),
    (dict(intrabar="worst"), "intrabar: 'worst' is not a setting; use pessimistic"),
    (dict(balance=-1.0), "balance: -1.0; give the opening cash as a number from 0"),
    (dict(balance={"spot": 1.0}), "balance: {'spot': 1.0}; give the opening cash"),
    (dict(taken="2026-13-01"), "taken: '2026-13-01' is not a day; give the day"),
    (dict(name=""), "name: ''; give the venue a name, or leave it out"),
])
def test_a_setting_the_checks_refuse_is_refused_in_the_terms_it_was_given_in(settings,
                                                                             says):
    with pytest.raises(VenueError, match=re.escape(says)):
        Venue(**settings)

    if "instruments" not in settings and "kind" not in settings:
        with pytest.raises(VenueError, match=re.escape(says)):
            probatum.venues.frictionless.replace(**settings)


# in a preset's file the same faults are a VenueError, as the loader's tests hold
@pytest.mark.parametrize("settings, says", [
    (dict(fees={"taker": "low"}), "fees.taker: 'low' is not a number; give a number"),
    (dict(reduce_only_exempt=0), "reduce_only_exempt: 0; give True or False"),
    (dict(tiers=[(5_000.0, 50.0, 0.015, 0.0)]),
     "tiers[0]: [5000.0, 50.0, 0.015, 0.0] is not a dict; give a dict of up_to, "
     "leverage, rate, amount"),
    (dict(kind="linear", instruments={1: {}}),
     "instruments: a key of 1; key each instrument by its symbol as text"),
    (dict(kind="linear", instruments={"ETHUSDT": {"delist": 1_704_175_200}}),
     "instruments.ETHUSDT.delist: 1704175200; give a time or the text of one"),
    (dict(kind=1), "kind: 1; use spot, linear, inverse, or None"),
    (dict(intrabar=0), "intrabar: 0 is not a setting; use pessimistic"),
    (dict(balance="1000"), "balance: '1000'; give the opening cash as a number from 0"),
    (dict(taken=20_260_101), "taken: 20260101 is not a day; give the day"),
    (dict(fills={"sizes": True}), "fills.sizes: True; use 'on' or 'off'"),
    (dict(fills={"sizes": {"on": 1}}), "fills.sizes: {'on': 1}; use 'on' or 'off'"),
    (dict(funding=8), "funding: 8; use 'none', 'data'"),
])
def test_a_setting_of_the_wrong_type_is_a_type_error(settings, says):
    with pytest.raises(ProbatumTypeError, match=re.escape(says)):
        Venue(**settings)

    if "instruments" not in settings and "kind" not in settings:
        with pytest.raises(ProbatumTypeError, match=re.escape(says)):
            probatum.venues.frictionless.replace(**settings)


def test_a_spot_venue_s_refusal_carries_no_engine_error_behind_it():
    # the engine's own refusal raised inside its handler would be printed above the
    # venue's, with "During handling of the above exception"
    with pytest.raises(VenueError, match="backstop: none") as caught:
        Venue(kind="spot", liquidation="book_then_backstop")

    assert caught.value.__context__ is None


def test_a_preset_s_liquidation_fee_reaches_the_engine():
    preset = probatum.venues._read(
        '[venue]\nname = "made"\ntaken = 2026-10-16\nkind = "linear"\n\n'
        '[defaults]\nliquidation_fee = 0.01\n', "made")

    probatum.venues._check(preset)

    assert probatum.venues._settings(preset, ["BTCUSDT"])["liquidation_fee"] == 0.01

    spot = probatum.venues._settings(dict(preset, kind="spot"), ["BTCUSDT"])

    assert "liquidation_fee" not in spot


def test_an_instrument_s_delisting_time_is_held_as_its_moment_in_utc():
    # a day names its first moment, a time with an offset is read in UTC, and a time
    # with seconds, or a part of one, keeps them
    def held(when):
        venue = Venue(kind="linear", instruments={"ETHUSDT": {"delist": when}})
        return venue.to_dict()["instruments"]["ETHUSDT"]["delist"]

    plus_one = datetime.timezone(datetime.timedelta(hours=1))
    given    = ("2024-01-02", "2024-01-02 06:00",
                datetime.datetime(2024, 1, 2, 7, tzinfo=plus_one),
                np.datetime64("2024-01-02T06:00:30"),
                np.datetime64("2024-01-02T06:00:30.5"))

    assert [held(when) for when in given] == [
        "2024-01-02 00:00", "2024-01-02 06:00", "2024-01-02 06:00", "2024-01-02 06:00:30",
        "2024-01-02 06:00:30.500000000"]

    venue = Venue(kind="linear", instruments={"ETHUSDT": {"delist": "2024-01-02 06:00"}})

    assert venue == Venue(
        kind        = "linear",
        instruments = {
            "ETHUSDT": {"delist": datetime.datetime(2024, 1, 2, 6)}},
    )
    assert Venue(**venue.to_dict()) == venue
    # the engine's settings for the instrument leave it to the run, which reads it
    # beside the bars
    assert "delist" not in probatum.venues._instrument_settings(venue._preset, "ETHUSDT")

    preset = probatum.venues._read(
        '[venue]\nname = "made"\ntaken = 2026-10-16\nkind = "linear"\n\n'
        '[instruments.ETHUSDT]\ndelist = 2024-01-02T06:00:00Z\n', "made")

    assert preset["instruments"]["ETHUSDT"]["delist"] == "2024-01-02 06:00"
