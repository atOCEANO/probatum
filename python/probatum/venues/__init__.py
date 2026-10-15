"""The venue presets: each a TOML file in this package, dated, read and checked when it
loads (ADR 0028). A preset holds the venue's defaults, each kind's where it mixes spot
and perpetual markets, and each instrument's own tables.
"""

import datetime
import functools
import importlib.resources
import re
import tomllib

import numpy as np

from .._probatum import Engine

KINDS = ("spot", "linear", "inverse")
# what a venue sets for the whole account, and what an instrument may set apart
VENUE = ("fees", "liquidation", "backstop", "funding", "leverage", "max_open_orders",
         "fills")
INSTRUMENT = ("fees", "tiers", "tiers_by", "tick", "lot", "sigfigs", "decimals",
              "min_notional", "max_qty", "reduce_only_exempt")
FILLS = ("limit", "window", "liquidity", "spread", "slippage", "impact", "sizes")
TIER  = ("up_to", "leverage", "rate", "amount")
# each setting's type in a file, the values themselves the Engine's to check
NUMBERS = ("backstop", "leverage", "tick", "lot", "min_notional", "max_qty")
WHOLE   = ("max_open_orders", "sigfigs", "decimals")
WORDS   = ("liquidation", "tiers_by", "limit")


def _names():
    """The packaged presets' names."""
    files = importlib.resources.files(__name__).iterdir()
    return sorted(path.name[:-5] for path in files if path.name.endswith(".toml"))


@functools.cache
def _preset(name):
    """The packaged preset ``name``, read and checked once."""
    if name not in _names():
        raise ValueError(f"venue: {name!r} is not a preset; use one of "
                         f"{', '.join(_names())}")

    path   = importlib.resources.files(__name__) / f"{name}.toml"
    preset = _read(path.read_text(encoding="utf-8"), name)
    _check(preset)

    return preset


def _table(origin, where, value):
    """``value`` where a TOML table belongs, at ``where`` in ``origin``."""
    if not isinstance(value, dict):
        raise ValueError(f"{origin}: {where}: {value!r} is not a table; give a table")

    return value


def _known(origin, where, table, allowed):
    """``table`` at ``where``, every key of it one of ``allowed``."""
    for key in table:
        if key not in allowed:
            raise ValueError(f"{origin}: {where}: {key!r} is not a setting; give "
                             f"{', '.join(allowed)}")

    return table


def _number(origin, where, value, whole=False):
    """``value`` where a number belongs, a whole one where ``whole``."""
    kinds = (int,) if whole else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds):
        need = "a whole number" if whole else "a number"
        raise ValueError(f"{origin}: {where}: {value!r} is not {need}; give {need}")

    return value


def _typed(origin, where, name, value):
    """Setting ``name`` at ``where``, of the type a file gives it."""
    if name in NUMBERS:
        return _number(origin, f"{where}.{name}", value)

    if name in WHOLE:
        return _number(origin, f"{where}.{name}", value, whole=True)

    if name in WORDS and not isinstance(value, str):
        raise ValueError(f"{origin}: {where}.{name}: {value!r} is not a word; give "
                         f"its name")

    if name == "reduce_only_exempt" and not isinstance(value, bool):
        raise ValueError(f"{origin}: {where}.{name}: {value!r}; give true or false")

    return value


def _read(text, origin):
    """The preset in the TOML ``text``, its shape checked and ``origin`` naming it in
    what is refused: its name, the day its tables were taken, the kind of every
    instrument where it holds one kind, its defaults, each kind's settings and each
    instrument's, each of its type. The values are checked by `_check`, where the
    Engine checks them."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        said = str(error)
        raise ValueError(f"{origin}: {said[:1].lower()}{said[1:]}; give TOML") from None

    _known(origin, "the file", data, ("venue", "defaults", "kinds", "instruments"))
    venue = _known(origin, "venue", _table(origin, "venue", data.get("venue")),
                   ("name", "taken", "kind"))

    name = venue.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{origin}: venue: name {name!r}; give the preset's name")

    taken = venue.get("taken")
    if isinstance(taken, datetime.date):
        taken = taken.isoformat()
    try:
        day = datetime.date.fromisoformat(taken).isoformat()
    except (TypeError, ValueError):
        day = None

    if day != taken:
        raise ValueError(f"{origin}: venue: taken {taken!r} is not a day; give the day "
                         f"the tables were taken, as YYYY-MM-DD")

    kind = venue.get("kind")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"{origin}: venue: kind {kind!r}; use {', '.join(KINDS)}")

    defaults = _known(origin, "defaults",
                      _table(origin, "defaults", data.get("defaults", {})),
                      tuple(dict.fromkeys(VENUE + INSTRUMENT)))
    fills = _known(origin, "defaults.fills",
                   _table(origin, "defaults.fills", defaults.get("fills", {})), FILLS)
    kinds = _known(origin, "kinds", _table(origin, "kinds", data.get("kinds", {})),
                   KINDS)
    if kind is not None and kinds:
        raise ValueError(f"{origin}: kinds: {', '.join(kinds)}, in a preset of one "
                         f"kind, {kind}; give every instrument's settings in defaults")

    tables = {"defaults": defaults}
    for each, table in kinds.items():
        tables[f"kinds.{each}"] = _known(origin, f"kinds.{each}",
                                         _table(origin, f"kinds.{each}", table),
                                         INSTRUMENT)

    instruments = _table(origin, "instruments", data.get("instruments", {}))
    for key, table in instruments.items():
        _kind_of(origin, key, kind)
        tables[key] = _known(origin, key, _table(origin, key, table), INSTRUMENT)

    for where, table in tables.items():
        for setting, value in table.items():
            _typed(origin, where, setting, value)
        if "fees" in table:
            _rates(origin, f"{where}.fees", table["fees"])
        if "tiers" in table:
            _rows(origin, f"{where}.tiers", table["tiers"])

    return dict(name=name, taken=taken, kind=kind,
                defaults=dict(defaults, fills=fills), kinds=kinds,
                instruments=instruments)


def _kind_of(origin, key, kind):
    """The kind and the symbol of instrument ``key`` in a preset of ``kind``: one kind
    for every instrument, or none, where each key names its own as ``kind:SYMBOL``."""
    if kind is not None:
        return kind, key

    found = re.fullmatch(r"(spot|linear|inverse):(.+)", key)
    if found is None:
        raise ValueError(f"{origin}: {key!r} names no kind; name an instrument of a "
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
    if not isinstance(tiers, list):
        raise ValueError(f"{origin}: {where}: {tiers!r} is not a list; give a list of "
                         f"tables of {', '.join(TIER)}")

    rows = []
    for k, row in enumerate(tiers):
        at = f"{where}[{k}]"
        _known(origin, at, _table(origin, at, row), TIER)
        for name in ("up_to", "leverage", "rate"):
            if name not in row:
                raise ValueError(f"{origin}: {at}: no {name}; give {', '.join(TIER)}")

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

    shapes = [repr(word) for word in plain] + [f"{{ {key} = n }}" for key in given]
    raise ValueError(f"{origin}: {where}: {value!r}; use {' or '.join(shapes)}")


def _venue_settings(preset):
    """The Engine's settings for the whole account of ``preset``. Funding from the
    data is the caller's to give, with the data."""
    origin, defaults = preset["name"], preset["defaults"]
    out = {}
    if "fees" in defaults:
        out.update(_rates(origin, "defaults.fees", defaults["fees"]))

    for name in ("liquidation", "backstop", "leverage", "max_open_orders"):
        if name in defaults:
            out[name] = defaults[name]

    funding = defaults.get("funding", "none")
    if isinstance(funding, dict):
        _known(origin, "defaults.funding", funding, ("every", "every_bars", "rate"))
        if ("every" in funding) == ("every_bars" in funding) or "rate" not in funding:
            raise ValueError(f"{origin}: defaults.funding: {funding!r}; give every, in "
                             f"hours, or every_bars, and the rate")

        if "every" in funding:
            out["funding_hours"] = _number(origin, "defaults.funding.every",
                                           funding["every"])
        else:
            out["funding_interval"] = _number(origin, "defaults.funding.every_bars",
                                              funding["every_bars"], whole=True)
        out["funding_rate"] = _number(origin, "defaults.funding.rate", funding["rate"])
    elif funding not in ("none", "data"):
        raise ValueError(f"{origin}: defaults.funding: {funding!r}; use 'none', "
                         f"'data', {{ every = hours, rate = r }} or "
                         f"{{ every_bars = n, rate = r }}")

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
    origin = preset["name"]
    kind, _ = _kind_of(origin, key, preset["kind"])

    given = {name: value for name, value in preset["defaults"].items()
             if name in INSTRUMENT and name != "fees"}
    fees = {}
    for table in (preset["kinds"].get(kind, {}), preset["instruments"].get(key, {})):
        given.update(table)
        fees.update(table.get("fees", {}))

    given.pop("fees", None)

    out = {"market": {"spot": "spot", "linear": "perp", "inverse": "inverse"}[kind]}
    out.update(_rates(origin, f"{key}.fees", fees))
    for name, value in given.items():
        if name == "tiers":
            out["tiers"] = _rows(origin, f"{key}.tiers", value)
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
        out.pop("liquidation", None)
        out.pop("backstop", None)

    if len(specs) == 1:
        [spec] = specs
        if spec["market"] == "inverse":
            spec["contract"] = contract
        out.update(spec)
        return out

    if any(spec["market"] == "inverse" for spec in specs):
        raise ValueError(f"{preset['name']}: an inverse contract among {len(keys)} "
                         f"instruments; run it alone, as its account holds it")

    out["market"]      = "perp"
    out["instruments"] = specs

    return out


def _check(preset):
    """Run every setting of ``preset`` through the checks the Engine runs, on one row of
    placeholder bars, so a preset passes exactly what a venue built by hand does: its
    spot and linear instruments as one universe, and each inverse contract alone, as
    its account holds it. A preset that lists no instruments is checked on one
    instrument of its kind, or one of each where it holds any. Where the venue books
    then backstops, every perpetual keeps maintenance, so that any of its instruments
    runs without the others."""
    origin = preset["name"]
    listed = list(preset["instruments"])
    if listed:
        keys, shown = listed, listed
    elif preset["kind"] is not None:
        keys, shown = ["placeholder"], [None]
    else:
        keys  = [f"{kind}:placeholder" for kind in KINDS]
        shown = [None] * len(keys)

    checked = dict(preset, instruments={key: preset["instruments"].get(key, {})
                                        for key in keys})
    kinds = [_kind_of(origin, key, preset["kind"])[0] for key in keys]

    def refused(chosen, tables):
        """What the Engine says, if anything, to the instruments at ``chosen`` given
        the tables ``tables``."""
        inverse  = kinds[chosen[0]] == "inverse"
        settings = _settings(dict(checked, instruments=tables),
                             [keys[k] for k in chosen], 1.0 if inverse else None)
        if inverse:
            settings["quote"] = 1.0
        elif {"spot", "linear"} <= {kinds[k] for k in chosen}:
            settings["quote"] = {"spot": 1.0, "margin": 1.0}
        if "funding_hours" in settings:
            # funding by the hour stamps its events on the clock
            settings["times"] = np.zeros(1, dtype=np.int64)

        try:
            Engine(np.ones((1, 5) if len(chosen) == 1 else (1, len(chosen), 5)),
                   **settings)
        except (ValueError, TypeError) as error:
            return str(error)

        return None

    def run(chosen):
        said = refused(chosen, checked["instruments"])
        if said is None:
            return

        name  = None
        found = re.match(r"instruments\[(\d+)\]: (.*)", said, re.DOTALL)
        if found is not None:
            name, said = shown[chosen[int(found.group(1))]], found.group(2)
        elif len(chosen) == 1 and shown[chosen[0]] is not None:
            # one instrument's refusal is the venue's where the venue's settings meet
            # the same one without the instrument's own table
            bare = {**checked["instruments"], keys[chosen[0]]: {}}
            if refused(chosen, bare) != said:
                name = shown[chosen[0]]

        where = origin if name is None else f"{origin}: {name}"
        raise ValueError(f"{where}: {said}")

    if preset["defaults"].get("liquidation") == "book_then_backstop":
        for k, kind in enumerate(kinds):
            tiers = _instrument_settings(checked, keys[k]).get("tiers", [])
            if kind != "spot" and not any(rate > 0.0 for _, _, rate, _ in tiers):
                where = origin if shown[k] is None else f"{origin}: {shown[k]}"
                raise ValueError(f"{where}: tiers: no maintenance, where the venue "
                                 f"books then backstops; give a maintenance rate")

    shared = [k for k, kind in enumerate(kinds) if kind != "inverse"]
    if shared:
        run(shared)

    for k, kind in enumerate(kinds):
        if kind == "inverse":
            run([k])
