"""The venue presets: each a TOML file in this package, dated, read and checked when it
loads (ADR 0028), and reached as a `Venue`, ``probatum.venues.binance_perp``. A preset
holds the venue's defaults, each instrument's own tables, and each kind's settings
where it mixes spot and perpetual markets, or names the presets it combines.
"""

import datetime
import functools
import importlib.resources
import re
import tomllib

import numpy as np

from .._errors import DataError, ProbatumAttributeError, VenueError
from .._probatum import Engine

KINDS = ("spot", "linear", "inverse")
# what a venue sets for the whole account, and what an instrument may set apart
VENUE = ("fees", "liquidation", "backstop", "liquidation_fee", "funding", "leverage",
         "max_open_orders", "fills")
INSTRUMENT = ("fees", "tiers", "tiers_by", "tick", "lot", "sigfigs", "decimals",
              "min_notional", "max_qty", "reduce_only_exempt")
FILLS = ("limit", "window", "liquidity", "spread", "slippage", "impact", "sizes")
TIER  = ("up_to", "leverage", "rate", "amount")
# each setting's type in a file, the values themselves the Engine's to check
NUMBERS = ("backstop", "liquidation_fee", "leverage", "tick", "lot", "min_notional",
           "max_qty")
WHOLE = ("max_open_orders", "sigfigs", "decimals")
WORDS = ("liquidation", "tiers_by", "limit")
# the Engine's keywords as a Venue names them, for the refusals of one
SPOKEN = {
    "fee_maker":        "fees.maker",
    "fee_taker":        "fees.taker",
    "slippage_bps":     "fills.slippage",
    "budget":           "fills.liquidity.budget",
    "window":           "fills.window",
    "spread":           "fills.spread",
    "impact":           "fills.impact",
    "impact_model":     "fills.impact",
    "limit":            "fills.limit",
    "funding_rate":     "funding.rate",
    "funding_interval": "funding.every_bars",
    "funding_hours":    "funding.every",
    "quote":            "balance",
}

# the presets are reached by name, so the module exports nothing for a star import
__all__ = []


@functools.cache
def _names():
    """The packaged presets' names."""
    files = importlib.resources.files(__name__).iterdir()
    return tuple(sorted(path.name[:-5] for path in files
                        if path.name.endswith(".toml")))


def _said(origin, where):
    """What a refusal of the setting at ``where`` opens with: the preset and the
    setting's path in its file, or, for a Venue built in Python, which ``origin``
    None marks, the setting's path in what was passed."""
    if origin is None:
        return where.removeprefix("defaults").removeprefix(".")
    return f"{origin}: {where}"


def _origin(preset):
    """Who a refusal of ``preset`` names: None for a Venue built in Python, whose
    refusals speak the caller's terms, and the preset's name otherwise."""
    return None if preset.get("python") else preset["name"]


def _noun(origin):
    """What a table is called where ``origin`` gives it: a dict in Python."""
    return "dict" if origin is None else "table"


@functools.cache
def _file(name):
    """The packaged preset ``name`` as its file reads, its shape checked once."""
    path = importlib.resources.files(__name__) / f"{name}.toml"
    return _read(path.read_text(encoding="utf-8"), name)


@functools.cache
def _preset(name):
    """The packaged preset ``name``, read and checked once."""
    if name not in _names():
        raise VenueError(f"venue: {name!r} is not a preset; use one of "
                         f"{', '.join(_names())}")

    preset = _file(name)

    if preset["combines"]:
        for part in preset["combines"]:
            if part not in _names():
                raise VenueError(f"{name}: venue: combines {part!r}, which is not a "
                                 f"preset; use one of {', '.join(_names())}")
        preset = _combine(preset, [_file(part) for part in preset["combines"]])

    _check(preset)

    return preset


def _table(origin, where, value):
    """``value`` where a table belongs, at ``where`` in ``origin``."""
    if not isinstance(value, dict):
        noun = _noun(origin)
        raise VenueError(f"{_said(origin, where)}: {value!r} is not a {noun}; give a "
                         f"{noun}")

    return value


def _known(origin, where, table, allowed):
    """``table`` at ``where``, every key of it one of ``allowed``."""
    for key in table:
        if key not in allowed:
            raise VenueError(f"{_said(origin, where)}: {key!r} is not a setting; give "
                             f"{', '.join(allowed)}")

    return table


def _number(origin, where, value, whole=False):
    """``value`` where a number belongs, a whole one where ``whole``."""
    kinds = (int,) if whole else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds):
        need = "a whole number" if whole else "a number"
        raise VenueError(f"{_said(origin, where)}: {value!r} is not {need}; give "
                         f"{need}")

    return value


def _typed(origin, where, name, value):
    """Setting ``name`` at ``where``, of the type a file gives it."""
    if name in NUMBERS:
        return _number(origin, f"{where}.{name}", value)

    if name in WHOLE:
        return _number(origin, f"{where}.{name}", value, whole=True)

    if name in WORDS and not isinstance(value, str):
        raise VenueError(f"{_said(origin, f'{where}.{name}')}: {value!r} is not a "
                         f"word; give its name")

    if name in WORDS and value != value.lower():
        # the loader and the Venue compare the words as written
        raise VenueError(f"{_said(origin, f'{where}.{name}')}: {value!r}; write it "
                         f"in lowercase, {value.lower()!r}")

    if name == "reduce_only_exempt" and not isinstance(value, bool):
        said = "True or False" if origin is None else "true or false"
        raise VenueError(f"{_said(origin, f'{where}.{name}')}: {value!r}; give {said}")

    return value


def _read(text, origin):
    """The preset in the TOML ``text``, its shape checked and ``origin`` naming it in
    what is refused: its name, the day its tables were taken, the kind of every
    instrument where it holds one kind, or the presets it combines, its defaults, each
    kind's settings and each instrument's, each of its type. The values are checked
    by `_check`, where the Engine checks them."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        said = str(error)
        raise VenueError(f"{origin}: {said[:1].lower()}{said[1:]}; give TOML") from None

    _known(origin, "the file", data, ("venue", "defaults", "kinds", "instruments"))
    venue = _known(origin, "venue", _table(origin, "venue", data.get("venue")),
                   ("name", "taken", "kind", "combines"))

    name = venue.get("name")
    if not isinstance(name, str) or not name:
        raise VenueError(f"{origin}: venue: name {name!r}; give the preset's name")

    taken = venue.get("taken")
    if isinstance(taken, datetime.date):
        taken = taken.isoformat()
    try:
        day = datetime.date.fromisoformat(taken).isoformat()
    except (TypeError, ValueError):
        day = None

    if day != taken:
        raise VenueError(f"{origin}: venue: taken {taken!r} is not a day; give the day "
                         f"the tables were taken, as YYYY-MM-DD")

    kind = venue.get("kind")
    if kind is not None and kind not in KINDS:
        raise VenueError(f"{origin}: venue: kind {kind!r}; use {', '.join(KINDS)}")

    combines = venue.get("combines", [])
    if not isinstance(combines, list) or not all(isinstance(part, str)
                                                 for part in combines):
        raise VenueError(f"{origin}: venue: combines {combines!r}; give the list of "
                         f"the presets it combines")

    if combines and (kind is not None or set(data) != {"venue"}):
        raise VenueError(f"{origin}: venue: combines {combines!r} beside settings of "
                         f"its own; give a combining preset its venue alone")

    return dict(name=name, taken=taken, kind=kind, combines=combines,
                **_tables(origin, data, kind))


def _tables(origin, data, kind):
    """The defaults, kinds and instruments of ``data``, a preset whose instruments are
    all of ``kind``, or of several kinds where it is None, as a file gives them or a
    Venue is built: the shape of each table and the type of every setting checked."""
    defaults = _known(origin, "defaults",
                      _table(origin, "defaults", data.get("defaults", {})),
                      tuple(dict.fromkeys(VENUE + INSTRUMENT)))
    fills = _known(origin, "defaults.fills",
                   _table(origin, "defaults.fills", defaults.get("fills", {})), FILLS)
    kinds = _known(origin, "kinds", _table(origin, "kinds", data.get("kinds", {})),
                   KINDS)

    if kind is not None and kinds:
        if origin is None:
            raise VenueError(f"kinds: {', '.join(kinds)}, beside kind={kind!r}; give "
                             f"every instrument's settings with the venue's, or "
                             f"leave kind out")
        raise VenueError(f"{origin}: kinds: {', '.join(kinds)}, in a preset of one "
                         f"kind, {kind}; give every instrument's settings in defaults")

    tables = {"defaults": defaults}
    for each, table in kinds.items():
        tables[f"kinds.{each}"] = _known(origin, f"kinds.{each}",
                                         _table(origin, f"kinds.{each}", table),
                                         INSTRUMENT)

    instruments = _table(origin, "instruments", data.get("instruments", {}))
    for key, table in instruments.items():
        if not isinstance(key, str):
            raise VenueError(f"{_said(origin, 'instruments')}: a key of {key!r}; key "
                             f"each instrument by its symbol as text")

        _kind_of(origin, key, kind)
        where = _where(origin, key)
        tables[where] = _known(origin, where, _table(origin, where, table), INSTRUMENT)

    for where, table in tables.items():
        for setting, value in table.items():
            _typed(origin, where, setting, value)
        if "fees" in table:
            _rates(origin, f"{where}.fees", table["fees"])
        if "tiers" in table:
            _rows(origin, f"{where}.tiers", table["tiers"])
            # an amount left out is 0, as the Engine takes it, and is held so
            for row in table["tiers"]:
                row.setdefault("amount", 0.0)

    return dict(defaults=dict(defaults, fills=fills), kinds=kinds,
                instruments=instruments)


def _where(origin, key):
    """Where instrument ``key``'s own table sits: under its key in a file, which a
    preset's refusals name it by, and in a Venue's ``instruments``."""
    return f"instruments.{key}" if origin is None else key


def _combine(preset, parts):
    """``preset``, which combines ``parts`` in one account: each part's instruments
    under its kind, as kind:SYMBOL, each part's own settings and rates its kind's,
    and the venue's settings, the same in every part that gives one, but for
    funding, where a part's none yields to another's."""
    origin = preset["name"]

    defaults, kinds, instruments = {}, {}, {}
    for part in parts:
        kind = part["kind"]
        if kind is None or kind in kinds:
            raise VenueError(f"{origin}: combines {part['name']!r}; combine presets of "
                             f"one kind each, no two of a kind")

        kinds[kind] = {name: value for name, value in part["defaults"].items()
                       if name in INSTRUMENT}
        for symbol, table in part["instruments"].items():
            instruments[f"{kind}:{symbol}"] = table

        for name, value in part["defaults"].items():
            if name in INSTRUMENT or (name == "funding" and value == "none"):
                continue
            if defaults.get(name, value) != value:
                raise VenueError(f"{origin}: combines {part['name']!r}, whose {name} "
                                 f"{value!r} is not the {defaults[name]!r} another "
                                 f"gives; give the parts one {name}")
            defaults[name] = value

    defaults.setdefault("fills", {})
    return dict(preset, kind=None, defaults=defaults, kinds=kinds,
                instruments=instruments)


def _kind_of(origin, key, kind):
    """The kind and the symbol of instrument ``key`` in a preset of ``kind``: one kind
    for every instrument, or none, where each key names its own as ``kind:SYMBOL``."""
    if kind is not None:
        return kind, key

    found = re.fullmatch(r"(spot|linear|inverse):(.+)", key)
    if found is None:
        if origin is None:
            raise VenueError(f"instruments: {key!r} names no kind; key it as "
                             f"kind:SYMBOL, with kind one of {', '.join(KINDS)}, or "
                             f"give the venue's kind with kind=")
        raise VenueError(f"{origin}: {key!r} names no kind; name an instrument of a "
                         f"preset that mixes kinds as kind:SYMBOL, with kind one of "
                         f"{', '.join(KINDS)}")

    return found.group(1), found.group(2)


def _rates(origin, where, fees):
    """The maker and taker rates of a ``fees`` table, each where given, as the Engine
    names them."""
    _known(origin, where, _table(origin, where, fees), ("maker", "taker"))
    return {f"fee_{side}": _number(origin, f"{where}.{side}", rate)
            for side, rate in fees.items()}


def _rows(origin, where, tiers):
    """A tier table's rows as the Engine takes them: each a table of up_to, leverage,
    rate and amount, an amount left out being 0."""
    noun = _noun(origin)
    if not isinstance(tiers, list):
        raise VenueError(f"{_said(origin, where)}: {tiers!r} is not a list; give a "
                         f"list of {noun}s of {', '.join(TIER)}")

    rows = []
    for k, row in enumerate(tiers):
        at = f"{where}[{k}]"
        if not isinstance(row, dict):
            raise VenueError(f"{_said(origin, at)}: {row!r} is not a {noun}; give a "
                             f"{noun} of {', '.join(TIER)}")
        _known(origin, at, row, TIER)
        for name in ("up_to", "leverage", "rate"):
            if name not in row:
                raise VenueError(f"{_said(origin, at)}: no {name}; give "
                                 f"{', '.join(TIER)}")

        rows.append(tuple(_number(origin, f"{at}.{name}", row.get(name, 0.0))
                          for name in TIER))

    return rows


def _choice(origin, where, value, plain, given):
    """A setting that is one word of ``plain``, or a table of one key of ``given``
    with its number: the word, or the key and its number."""
    if isinstance(value, str) and value in plain:
        return value, None

    if isinstance(value, dict) and len(value) == 1 and next(iter(value)) in given:
        [(key, number)] = value.items()
        return key, _number(origin, f"{where}.{key}", number)

    shape  = "{{\"{}\": n}}" if origin is None else "{{ {} = n }}"
    shapes = [repr(word) for word in plain] + [shape.format(key) for key in given]
    raise VenueError(f"{_said(origin, where)}: {value!r}; use {' or '.join(shapes)}")


def _venue_settings(preset):
    """The Engine's settings for the whole account of ``preset``. Funding from the
    data is the caller's to give, with the data."""
    origin, defaults = _origin(preset), preset["defaults"]

    out = {}
    if "fees" in defaults:
        out.update(_rates(origin, "defaults.fees", defaults["fees"]))

    for name in ("liquidation", "backstop", "liquidation_fee", "leverage",
                 "max_open_orders"):
        if name in defaults:
            out[name] = defaults[name]

    funding = defaults.get("funding", "none")
    said    = _said(origin, "defaults.funding")
    if isinstance(funding, dict):
        _known(origin, "defaults.funding", funding, ("every", "every_bars", "rate"))
        if ("every" in funding) == ("every_bars" in funding) or "rate" not in funding:
            raise VenueError(f"{said}: {funding!r}; give every, in hours, or "
                             f"every_bars, and the rate")

        if "every" in funding:
            out["funding_hours"] = _number(origin, "defaults.funding.every",
                                           funding["every"])
        else:
            out["funding_interval"] = _number(origin, "defaults.funding.every_bars",
                                              funding["every_bars"], whole=True)
        out["funding_rate"] = _number(origin, "defaults.funding.rate", funding["rate"])
    elif funding not in ("none", "data"):
        shapes = (('{"every": hours, "rate": r}', '{"every_bars": n, "rate": r}')
                  if origin is None else
                  ("{ every = hours, rate = r }", "{ every_bars = n, rate = r }"))
        raise VenueError(f"{said}: {funding!r}; use 'none', 'data', {shapes[0]} or "
                         f"{shapes[1]}")

    fills = defaults["fills"]
    if "limit" in fills:
        out["limit"] = _typed(origin, "defaults.fills", "limit", fills["limit"])
    if "window" in fills:
        out["window"] = _number(origin, "defaults.fills.window", fills["window"],
                                whole=True)
    if "liquidity" in fills:
        liquidity = _known(origin, "defaults.fills.liquidity",
                           _table(origin, "defaults.fills.liquidity",
                                  fills["liquidity"]), ("budget",))
        out["budget"] = _number(origin, "defaults.fills.liquidity.budget",
                                liquidity.get("budget", 1.0))

    spread, bps = _choice(origin, "defaults.fills.spread", fills.get("spread", "none"),
                          ("none", "estimated"), ("fixed",))
    out["spread"] = {"none": None, "estimated": "estimated"}.get(spread, bps)

    _, slip = _choice(origin, "defaults.fills.slippage", fills.get("slippage", "none"),
                      ("none",), ("fixed",))
    out["slippage_bps"] = slip or 0.0

    model, k = _choice(origin, "defaults.fills.impact", fills.get("impact", "none"),
                       ("none",), ("linear", "sqrt"))
    out["impact"]       = k or 0.0
    out["impact_model"] = "linear" if model == "none" else model

    sizes, _ = _choice(origin, "defaults.fills.sizes", fills.get("sizes", "off"),
                       ("on", "off"), ())
    out["sizes"] = sizes == "on"

    return out


def _instrument_settings(preset, key):
    """Instrument ``key``'s settings as the Engine takes them for an instrument of a
    universe: the defaults, then its kind's, then its own, and its rates side by
    side, a side it gives no rate for keeping its kind's, or the venue's."""
    origin = _origin(preset)
    kind, _ = _kind_of(origin, key, preset["kind"])

    given = {name: value for name, value in preset["defaults"].items()
             if name in INSTRUMENT and name != "fees"}
    fees = {}
    for table in (preset["kinds"].get(kind, {}), preset["instruments"].get(key, {})):
        given.update(table)
        fees.update(table.get("fees", {}))

    given.pop("fees", None)

    out   = {"market": {"spot": "spot", "linear": "perp", "inverse": "inverse"}[kind]}
    where = _where(origin, key)
    out.update(_rates(origin, f"{where}.fees", fees))
    for name, value in given.items():
        if name == "tiers":
            out["tiers"] = _rows(origin, f"{where}.tiers", value)
        else:
            out[name] = value

    return out


def _settings(preset, keys, contract=None):
    """The Engine's settings for the instruments ``keys`` of ``preset``, in that order:
    each instrument's in the dict a universe takes, or for one instrument at the top
    level, where an inverse contract worth ``contract`` dollars stands alone. Spot
    markets alone take no liquidation model, since nothing closes a spot account out."""
    out   = _venue_settings(preset)
    specs = [_instrument_settings(preset, key) for key in keys]
    if all(spec["market"] == "spot" for spec in specs):
        for name in ("liquidation", "backstop", "liquidation_fee"):
            out.pop(name, None)

    if len(specs) == 1:
        [spec] = specs
        if spec["market"] == "inverse":
            spec["contract"] = contract
        out.update(spec)
        return out

    if any(spec["market"] == "inverse" for spec in specs):
        origin = _origin(preset) or "data"
        raise DataError(f"{origin}: an inverse contract among {len(keys)} "
                        f"instruments; run it alone, as its account holds it")

    out["market"]      = "perp"
    out["instruments"] = specs

    return out


def _check(preset):
    """Run every setting of ``preset`` through the checks the Engine runs, on one row of
    placeholder bars, so a preset passes exactly what a venue built by hand does: its
    spot and linear instruments as one universe, and each inverse contract alone, as
    its account holds it. A kind it holds but lists no instrument of is checked on a
    placeholder of that kind, and a preset of no kind that lists none on one of each;
    a refusal of a placeholder names its kind's table where the preset gives one. The
    universe holds each instrument with one order slot, and the venue's own count is
    checked on one of them, so a preset's size never bounds it, and a venue of spot
    markets alone has its liquidation settings checked on a perpetual of their own,
    which nothing else would read. Where the venue books then backstops, every
    perpetual keeps maintenance, so that any of its instruments runs without the
    others."""
    origin = _origin(preset)
    listed = list(preset["instruments"])

    if preset["kind"] is not None:
        missing = [] if listed else ["placeholder"]
    else:
        held    = {_kind_of(origin, key, None)[0] for key in listed}
        kinds   = list(preset["kinds"]) or ([] if listed else list(KINDS))
        missing = [f"{kind}:placeholder" for kind in kinds if kind not in held]

    keys    = listed + missing
    checked = dict(preset, instruments={key: preset["instruments"].get(key, {})
                                        for key in keys})
    kinds = [_kind_of(origin, key, preset["kind"])[0] for key in keys]
    shown = [_where(origin, key) for key in listed] + [
        f"kinds.{kind}" if kind in preset["kinds"] else None
        for kind in kinds[len(listed):]]

    def refused(chosen, variant):
        """What the Engine says, if anything, to the instruments at ``chosen`` of the
        preset ``variant``."""
        inverse  = kinds[chosen[0]] == "inverse"
        settings = _settings(variant, [keys[k] for k in chosen],
                             1.0 if inverse else None)

        if inverse:
            settings["quote"] = 1.0
        elif {"spot", "linear"} <= {kinds[k] for k in chosen}:
            settings["quote"] = {"spot": 1.0, "margin": 1.0}
        if "funding_hours" in settings:
            # funding by the hour stamps its events on the clock
            settings["times"] = np.zeros(1, dtype=np.int64)
        if len(chosen) > 1:
            settings["max_open_orders"] = 1

        try:
            Engine(np.ones((1, 5) if len(chosen) == 1 else (1, len(chosen), 5)),
                   **settings)
        except (ValueError, TypeError) as error:
            return str(error)

        return None

    def raised(name, said):
        """Refuse with what the Engine said, of the table ``name`` names or of the
        venue, in a Venue's own terms where it was built in Python."""
        if origin is None:
            found = re.match(r"([a-z_]+)(\W.*)", said, re.DOTALL)
            if found is not None and found.group(1) in SPOKEN:
                said = SPOKEN[found.group(1)] + found.group(2)
            raise VenueError(said if name is None else f"{name}: {said}")

        raise VenueError(f"{origin if name is None else f'{origin}: {name}'}: {said}")

    def run(chosen):
        said = refused(chosen, checked)
        if said is None:
            return

        name  = None
        found = re.match(r"instruments\[(\d+)\]: (.*)", said, re.DOTALL)
        if found is not None:
            name, said = shown[chosen[int(found.group(1))]], found.group(2)
        elif len(chosen) == 1 and shown[chosen[0]] is not None:
            # one instrument's refusal is the venue's where the venue's settings meet
            # the same one without the instrument's own table, or its kind's
            k    = chosen[0]
            bare = (dict(checked, instruments={**checked["instruments"], keys[k]: {}})
                    if k < len(listed) else
                    dict(checked, kinds={**checked["kinds"], kinds[k]: {}}))
            if refused(chosen, bare) != said:
                name = shown[k]

        raised(name, said)

    if preset["defaults"].get("liquidation") == "book_then_backstop":
        for k, kind in enumerate(kinds):
            tiers = _instrument_settings(checked, keys[k]).get("tiers", [])
            if kind != "spot" and not any(rate > 0.0 for _, _, rate, _ in tiers):
                raised(shown[k], "tiers: no maintenance, where the venue books then "
                                 "backstops; give a maintenance rate")

    shared = [k for k, kind in enumerate(kinds) if kind != "inverse"]
    if shared:
        run(shared)
    if len(shared) > 1:
        run(shared[:1])

    for k, kind in enumerate(kinds):
        if kind == "inverse":
            run([k])

    if all(kind == "spot" for kind in kinds):
        settings = dict(_venue_settings(checked), market="perp",
                        tiers=[(float("inf"), 10.0, 0.01, 0.0)])
        if "funding_hours" in settings:
            settings["times"] = np.zeros(1, dtype=np.int64)

        said = None
        try:
            Engine(np.ones((1, 5)), **settings)
        except (ValueError, TypeError) as error:
            said = str(error)

        # raised outside the handler, as run raises, so the traceback is the venue's
        # refusal alone and not the engine's with the venue's after it
        if said is not None:
            raised(None, said)


@functools.cache
def _venue(name):
    """The packaged preset ``name`` as a Venue, built once."""
    from .._venue import Venue
    return Venue._of(_preset(name), source=name)


def __getattr__(name):
    if name in _names():
        return _venue(name)
    raise ProbatumAttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_names()))
