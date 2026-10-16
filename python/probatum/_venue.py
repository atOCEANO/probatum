"""The venue a run is on: its rules as one immutable object, a preset's or one built by
hand, so a setting has one place to be given (ADR 0002)."""

import collections
import copy
import dataclasses
import datetime
import functools
import json
import math
import numbers

import numpy as np

from . import venues
from ._data import Data

# the plain option in each row, which a venue's setting left out or given as None takes
PLAIN = {"balance": 10_000.0, "fees": {"maker": 0.0002, "taker": 0.0006},
         "leverage": 10.0, "liquidation": "takeover", "max_open_orders": 8,
         "fills": {"limit": "touch", "window": 24, "liquidity": {"budget": 1.0},
                   "spread": "none", "slippage": "none", "impact": "none",
                   "sizes": "off"},
         "intrabar": "pessimistic", "funding": "none"}
INTRABAR = ("pessimistic", "optimistic", "random")
# what the venue sets for every instrument whose kind's table and own leave it out
SHARED = tuple(name for name in venues.INSTRUMENT if name != "fees")
# the settings in the order the constructor takes them
SETTINGS = ("balance", "fees", "leverage", "liquidation", "backstop", "liquidation_fee",
            "max_open_orders", "fills", "intrabar", "funding") + SHARED + (
            "kinds", "instruments", "kind", "name", "taken")

Run         = collections.namedtuple("Run", "candles settings start")
Run.__doc__ = """What the Engine is built from to run a Data on a venue: the candles of
every row up to the Data's last, the bars before its first among them for its window
to read, the Engine's settings, and the row its first bar is on."""


class Venue:
    """A venue's rules for a run: the opening ``balance``, a number for an account of
    one wallet or ``{"spot": s, "margin": m}`` for one of both; its ``fees``, maker and
    taker; the ``leverage`` it caps a perpetual at; how it closes an account out, its
    ``liquidation`` model with the ``backstop`` share book then backstop takes over at,
    and the ``liquidation_fee`` that places the close point where no maintenance is
    kept, the taker rate where it is None; the ``max_open_orders`` each instrument may
    rest; its ``fills`` model; how ``intrabar`` decides what a bar cannot show; and its
    ``funding``. The size and price rules and the tier table, given beside these, are
    every instrument's, unless its kind's table in ``kinds`` or its own in
    ``instruments`` gives another, keyed by its symbol, or as kind:SYMBOL where
    ``kind``, the kind of every instrument the venue lists, is None. Each setting left
    out, or given as None, takes the plain option, and None inside a table leaves that
    key out.

    A Venue does not change, and every setting passes the checks a preset's pass as it
    is built, so a variant made with `replace` is checked as well. The presets are
    Venues, ``probatum.venues.binance_perp`` among them."""

    __slots__ = ("_preset", "_balance", "_intrabar", "_source", "_key")


    def __init__(self, *, balance=None, fees=None, leverage=None, liquidation=None,
                 backstop=None, liquidation_fee=None, max_open_orders=None, fills=None,
                 intrabar=None, funding=None, tiers=None, tiers_by=None, tick=None,
                 lot=None, sigfigs=None, decimals=None, min_notional=None,
                 max_qty=None, reduce_only_exempt=None, kinds=None, instruments=None,
                 kind=None, name=None, taken=None):
        given = {setting: _python(value) for setting, value in dict(
            balance=balance, fees=fees, leverage=leverage, liquidation=liquidation,
            backstop=backstop, liquidation_fee=liquidation_fee,
            max_open_orders=max_open_orders, fills=fills, intrabar=intrabar,
            funding=funding, tiers=tiers, tiers_by=tiers_by, tick=tick, lot=lot,
            sigfigs=sigfigs, decimals=decimals, min_notional=min_notional,
            max_qty=max_qty, reduce_only_exempt=reduce_only_exempt).items()}

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name: {name!r}; give the venue's name as text")
        if name == "":
            raise ValueError("name: ''; give the venue a name, or leave it out")

        if type(taken) is datetime.date:
            taken = taken.isoformat()
        if taken is not None and not _day(taken):
            raise ValueError(f"taken: {taken!r} is not a day; give the day the venue's "
                             f"tables were taken, as YYYY-MM-DD")

        if kind is not None and kind not in venues.KINDS:
            raise ValueError(f"kind: {kind!r}; use {', '.join(venues.KINDS)}, or None "
                             f"for a venue of several kinds")

        intrabar = _given_or_plain(given, "intrabar")
        if intrabar not in INTRABAR:
            raise ValueError(f"intrabar: {intrabar!r} is not a setting; use "
                             f"{', '.join(INTRABAR)}")

        defaults = {setting: _given_or_plain(given, setting)
                    for setting in ("fees", "leverage", "liquidation",
                                    "max_open_orders", "fills", "funding")}
        defaults.update({setting: given[setting]
                         for setting in ("backstop", "liquidation_fee") + SHARED
                         if given[setting] is not None})

        tables = {} if instruments is None else _python(instruments)
        data   = copy.deepcopy(dict(defaults=defaults, instruments=tables,
                                    kinds={} if kinds is None else _python(kinds)))

        preset = dict(name=name, taken=taken, kind=kind, combines=[], python=True,
                      **venues._tables(None, data, kind))
        venues._check(preset)

        self._hold(preset, _balance(_given_or_plain(given, "balance")), intrabar, None)


    @classmethod
    def _of(cls, preset, balance=PLAIN["balance"], intrabar=PLAIN["intrabar"],
            source=None):
        """A Venue over ``preset``, checked already as the loader gives it, each venue
        setting it leaves out given its plain option, which shows itself as a variant
        of the preset ``source``."""
        preset   = copy.deepcopy(preset)
        defaults = preset["defaults"]

        for setting in ("fees", "fills"):
            defaults[setting] = _over(PLAIN[setting], defaults.get(setting))
        for setting in ("leverage", "liquidation", "max_open_orders", "funding"):
            defaults.setdefault(setting, PLAIN[setting])

        venue = object.__new__(cls)
        venue._hold(preset, balance, intrabar, source)

        return venue


    def _hold(self, preset, balance, intrabar, source):
        object.__setattr__(self, "_preset", preset)
        object.__setattr__(self, "_balance", balance)
        object.__setattr__(self, "_intrabar", intrabar)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_key", json.dumps(_canonical(self._view()),
                                                    sort_keys=True))


    def _view(self):
        """Every setting, as `to_dict` gives it, of the tables the venue holds."""
        preset = self._preset
        held   = dict(balance=self._balance, intrabar=self._intrabar,
                      kinds=preset["kinds"], instruments=preset["instruments"],
                      kind=preset["kind"], name=preset["name"], taken=preset["taken"])

        return {setting: held[setting] if setting in held
                else preset["defaults"].get(setting) for setting in SETTINGS}


    @property
    def name(self):
        """The preset's name, or None for a venue built by hand."""
        return self._preset["name"]


    @property
    def taken(self):
        """The day a preset's tables were taken, as YYYY-MM-DD, or None."""
        return self._preset["taken"]


    @property
    def kind(self):
        """The kind of every instrument the venue lists, or None where it lists
        several kinds, each instrument keyed as kind:SYMBOL."""
        return self._preset["kind"]


    def to_dict(self):
        """Every setting, as the constructor takes it, so that
        ``Venue(**venue.to_dict())`` is this venue."""
        return copy.deepcopy(self._view())


    def replace(self, **changes):
        """This venue with ``changes``, checked as a new one is: ``fees`` and ``fills``
        merge into the venue's own, a table in ``kinds`` or ``instruments`` merges into
        the one it names key by key, its fees side by side, and is dropped where it is
        given as None, ``liquidation="takeover"`` drops the backstop a venue that
        booked then backstopped held, and every other setting is replaced whole. A
        change every kind's table overrides, which would reach no instrument, is
        refused."""
        if not changes:
            return self

        settings = self.to_dict()

        for setting, value in changes.items():
            if setting not in settings:
                raise TypeError(f"{setting}: not a setting of a venue; use one of "
                                f"{', '.join(SETTINGS)}")

            held = settings[setting]
            if setting in ("fees", "fills") and isinstance(value, dict):
                value = {**held, **value}
            elif setting in ("kinds", "instruments") and isinstance(value, dict):
                value = _merged(setting, held, value)
            settings[setting] = value

        if changes.get("liquidation") == "takeover" and "backstop" not in changes:
            settings["backstop"] = None

        kinds = settings["kinds"]
        for setting in changes:
            if (setting in ("fees",) + SHARED and settings["kind"] is None and kinds
                    and all(_gives(table, setting) for table in kinds.values())):
                first, who = next(iter(kinds)), self.name or "the venue"
                raise ValueError(f"{setting}: {changes[setting]!r}, which no "
                                 f"instrument of {who} takes, as each of its kinds "
                                 f"gives its own; give it in a kind's table, "
                                 f"replace(kinds={{{first!r}: {{{setting!r}: ...}}}})")

        venue = Venue(**settings)
        object.__setattr__(venue, "_source", self._source)

        return venue


    def __eq__(self, other):
        return isinstance(other, Venue) and self._key == other._key


    def __hash__(self):
        return hash(self._key)


    def __repr__(self):
        # as the code that builds it: a variant of the preset it came from, or a venue
        # built by hand, each setting that differs from where it started
        source = self._source
        start  = (venues._venue(source) if source is not None else _plain())._view()
        shown  = []
        for setting, value in self._view().items():
            before = start[setting]
            if value == before:
                continue
            if setting in ("kinds", "instruments") and source is not None:
                value = {key: value.get(key) for key in {**before, **value}
                         if value.get(key) != before.get(key)}
            shown.append(f"{setting}={value!r}")

        if source is None:
            return f"probatum.Venue({', '.join(shown)})"

        head = f"probatum.venues.{source}"
        return f"{head}.replace({', '.join(shown)})" if shown else head


    def __setattr__(self, name, value):
        if name not in SETTINGS:
            raise AttributeError(f"{name}: not a setting of a venue, and a venue does "
                                 f"not change")

        raise AttributeError(f"{name}: a Venue does not change; make a variant with "
                             f"replace({name}=...)")


    def __delattr__(self, name):
        raise AttributeError(f"{name}: a Venue does not change; make a variant with "
                             f"replace")


    def __reduce__(self):
        return Venue._of, (self._preset, self._balance, self._intrabar, self._source)


    def __copy__(self):
        return self


    def __deepcopy__(self, memo):
        return self


def _run(venue, data, seed=None):
    """What the Engine is built from to run ``data`` on ``venue``, a Venue or a preset's
    name, as a `Run`, with ``seed`` the stream a venue deciding what a bar cannot show
    at random draws from. Each instrument takes its tables from the venue, keyed by its
    symbol, or as kind:SYMBOL on a venue of several kinds, and each order limit the
    router gave it where no table of the venue gives one; the rows none of the
    instruments printed are left out. Refused are an empty Data; an account at two
    venues or whose cash would be two assets; an instrument of no market or a kind the
    venue does not list; an inverse contract of no size the bars give; one balance for
    an account that uses both wallets, and the two of them for an inverse contract's,
    which has one; funding from the data where it holds none for a perpetual; an
    instrument without the lot and price rule the venue's sizes hold every order to,
    or with significant figures and no decimals; and a perpetual with no maintenance
    where the venue books then backstops."""
    venue  = _venue_of(venue)
    data   = data if isinstance(data, Data) else Data(data)
    origin = venue.name or "venue"
    if not len(data) or not data.available.any():
        raise ValueError("data: no bars to run; give a Data that has some")

    instruments, names = data.instruments, data.names
    markets = [_market(instrument, name, venue, origin)
               for instrument, name in zip(instruments, names)]
    held = sorted({instrument.venue for instrument in instruments if instrument.venue})
    if len(held) > 1:
        raise ValueError(f"data: instruments at {' and '.join(held)}, and an account "
                         f"is at one venue; run each venue's instruments on their own")

    _one_cash(instruments, names, markets)

    keys = [(instrument.symbol or name) if venue.kind else
            f"{market}:{instrument.symbol or name}"
            for instrument, name, market in zip(instruments, names, markets)]
    inverse = markets == ["inverse"]
    if inverse and instruments[0].contract is None:
        raise ValueError(f"data: {names[0]} is an inverse contract whose bars do not "
                         f"give the dollars a contract is worth; build it from frames "
                         f"that do, as the router's do for a symbol it lists")

    settings = venues._settings(venue._preset, keys,
                                instruments[0].contract if inverse else None)
    booked = venue._preset["defaults"]["liquidation"] == "book_then_backstop"

    for spec, instrument, name in zip(settings.get("instruments", [settings]),
                                      instruments, names):
        # the router's limits are the venue's where no table of the venue gives one
        for limit in ("min_notional", "max_qty"):
            given = getattr(instrument, limit)
            if given is not None and limit not in spec:
                spec[limit] = given
        _held_to_rules(spec, name, origin, settings["sizes"])
        if booked and spec["market"] != "spot" and not any(
                rate > 0.0 for _, _, rate, _ in spec.get("tiers", [])):
            raise ValueError(f"venue: {origin} books then backstops, and gives {name} "
                             f"no tier table that keeps maintenance; give {name} its "
                             f"tiers in instruments")

    perps = [k for k, market in enumerate(markets) if market != "spot"]
    if venue._preset["defaults"]["funding"] == "data" and perps:
        owned = set(data._root.funding["instrument"].tolist())
        bare  = [names[k] for k in perps if data._cols[k] not in owned]
        if bare:
            raise ValueError(f"funding: {origin} charges the funding the data holds, "
                             f"and the data holds none for {bare[0]}; fetch the bars "
                             f"with funding=True, or replace(funding=\"none\")")

        events = data.funding
        settings.update(funding_times=np.ascontiguousarray(events["time"]),
                        funding_rates=np.ascontiguousarray(events["rate"]),
                        funding_instruments=events["instrument"].tolist())

    balance = venue._balance
    if not isinstance(balance, dict) and "spot" in markets and perps:
        raise ValueError(f"balance: {balance}, one wallet's, and the data's spot and "
                         f"perpetual instruments use both; replace(balance="
                         f"{{\"spot\": s, \"margin\": m}})")

    if isinstance(balance, dict) and inverse:
        raise ValueError(f"balance: {balance}, the cash of two wallets, and an inverse "
                         f"contract's account holds its coin in one; give the coin as "
                         f"a number")

    settings.update(quote=balance, intrabar=venue._intrabar)
    if seed is not None:
        settings["seed"] = seed

    start, stop = data._rows
    whole = Data._of(data._root, (0, stop), data._cols)

    # a row none of these instruments printed is a row their account never sees, as
    # a selection of a wider universe has, and the Engine takes none
    printed = whole.available.any(axis=1)
    settings["times"] = np.ascontiguousarray(whole.time[printed])
    candles = np.ascontiguousarray(whole.bars[printed])
    if inverse:
        candles = candles[:, 0]

    return Run(candles, settings, int(printed[:start].sum()))


def _held_to_rules(spec, name, origin, sizes):
    """Refuse instrument ``name``'s settings ``spec`` where they hold it to no whole
    price rule, significant figures needing their decimals, and with the venue's
    ``sizes`` on, to no lot or no price rule."""
    if spec.get("sigfigs") is not None and spec.get("decimals") is None:
        raise ValueError(f"venue: {origin} gives {name} significant figures and no "
                         f"decimals, the cap its price rule needs; give {name} its "
                         f"decimals in instruments")

    rule    = spec.get("tick") or spec.get("sigfigs") is not None
    missing = [part for part, there in (("lot", spec.get("lot")), ("tick", rule))
               if not there]

    if sizes and missing:
        raise ValueError(f"venue: {origin} gives {name} no {' and no '.join(missing)}, "
                         f"and its sizes hold every order to them; give {name} its own "
                         f"in instruments, or replace(fills={{\"sizes\": \"off\"}})")


def _venue_of(venue):
    """``venue`` as a Venue: itself, or the preset of that name."""
    if isinstance(venue, Venue):
        return venue

    if isinstance(venue, str):
        venues._preset(venue)
        return venues._venue(venue)

    raise TypeError(f"venue: {venue!r}; pass a Venue, or a preset's name such as "
                    f"\"binance_perp\"")


def _market(instrument, name, venue, origin):
    """The kind of ``instrument``, its own or else ``venue``'s, refused where neither
    says, or where it is not a kind the venue lists: its one kind, or, on a venue of
    several, the kinds its tables name, a venue that names none taking every kind."""
    market = instrument.market or venue.kind
    if market is None:
        raise ValueError(f"data: {name} names no market, and {origin} is not of one "
                         f"kind; build the Data with market=")

    kinds = [venue.kind] if venue.kind else list(venue._preset["kinds"])
    if kinds and market not in kinds:
        raise ValueError(f"venue: {origin} lists {' and '.join(kinds)} instruments, "
                         f"and {name} is {market}; run it on a venue of its kind")

    return market


def _one_cash(instruments, names, markets):
    """Refuse an account whose cash would be two assets, where the instruments say: a
    perpetual settles in its quote, an inverse contract in its coin, and a spot pair
    is bought with its quote."""
    first = None
    held  = sorted(zip(instruments, names, markets), key=lambda item: item[2] == "spot")

    for instrument, name, market in held:
        asset = dataclasses.replace(instrument, market=market).settlement
        said  = f"{name} {'is quoted' if market == 'spot' else 'settles'} in {asset}"
        if asset is None:
            continue

        if first is None:
            first = (asset, said)
        elif asset != first[0]:
            raise ValueError(f"data: {first[1]} and {said}, and an account's cash is "
                             f"one asset; run them on their own")


@functools.cache
def _plain():
    """The venue built by hand with every setting left out."""
    return Venue()


def _day(text):
    """Whether ``text`` is a day written as YYYY-MM-DD."""
    try:
        return datetime.date.fromisoformat(text).isoformat() == text
    except (TypeError, ValueError):
        return False


def _over(plain, given):
    """A table ``given`` over the ``plain`` one, key by key, or ``plain`` where it is
    None; anything else is left for the checks to refuse."""
    if given is None:
        return copy.deepcopy(plain)
    return {**copy.deepcopy(plain), **given} if isinstance(given, dict) else given


def _given_or_plain(given, setting):
    """Setting ``setting`` of ``given``, its plain option where it is None, and for
    ``fees`` and ``fills`` a table given in part over the plain one."""
    if setting in ("fees", "fills"):
        return _over(PLAIN[setting], given[setting])
    return copy.deepcopy(PLAIN[setting]) if given[setting] is None else given[setting]


def _python(value):
    """``value`` with its numbers Python's, numpy's among them, and every key a table
    gives as None left out."""
    if isinstance(value, dict):
        return {key: _python(item) for key, item in value.items() if item is not None}

    if isinstance(value, (list, tuple)):
        return [_python(item) for item in value]

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, numbers.Integral):
        return int(value)

    if isinstance(value, numbers.Real):
        return float(value)

    return value


def _gives(table, setting):
    """Whether a kind's ``table`` gives ``setting`` of its own, fees with both rates."""
    if setting == "fees":
        return isinstance(table.get("fees"), dict) and {"maker", "taker"} <= set(
            table["fees"])
    return setting in table


def _balance(balance):
    """The opening ``balance`` as a Venue holds it: a number, or each wallet's."""
    wallets = (isinstance(balance, dict) and set(balance) == {"spot", "margin"})
    cash    = [balance[wallet] for wallet in ("spot", "margin")] if wallets else [balance]

    for value in cash:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0.0):
            raise ValueError(f"balance: {balance!r}; give the opening cash as a number "
                             f"from 0, or {{\"spot\": s, \"margin\": m}} for an "
                             f"account that holds both wallets")

    if wallets:
        return {wallet: float(balance[wallet]) for wallet in ("spot", "margin")}
    return float(balance)


def _merged(setting, held, given):
    """The tables ``held`` of ``setting``, keyed by kind or instrument, with ``given``
    merged in: a table key by key and its fees side by side, a table given None
    dropped, which the venue must hold."""
    out = dict(held)

    for key, table in given.items():
        if table is None:
            if key not in out:
                raise ValueError(f"{setting}: {key!r}: None for a table the venue "
                                 f"does not hold; name one of its {setting}")
            del out[key]
        elif isinstance(table, dict) and isinstance(out.get(key), dict):
            fees = out[key].get("fees")
            out[key] = {**out[key], **table}
            if isinstance(fees, dict) and isinstance(table.get("fees"), dict):
                out[key]["fees"] = {**fees, **table["fees"]}
        else:
            out[key] = table

    return out


def _canonical(value):
    """``value`` with every number a float and no zero signed, so settings equal in
    value give one key."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) + 0.0

    return value
