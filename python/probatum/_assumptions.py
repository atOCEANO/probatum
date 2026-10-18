"""What a run assumed (Specification, Assumptions): each modelling choice it made as a
row of its name, its value, its source, its grade, the direction of its error where it
is simplified and what it cost where the run's records measure it, read from the
settings each instrument ran, the engine's own rules and the Data (ADR 0027)."""

import fractions
import math

import numpy as np

from . import venues
from ._data import DAY, _when
from ._probatum import funding_inside
from ._venue import PLAIN, _settlement

# the grades from the most favourable to the least
GRADES  = ("venue", "simplified", "optimistic")
HOUR    = 3_600_000_000_000
MINUTE  = 60_000_000_000
USER    = "set by the user"
ENGINE  = "the engine"
KINDS   = {"spot": "spot", "perp": "linear", "inverse": "inverse"}
SOURCES = {
    "router": "the router",
    "frame":  "a frame",
    "array":  "an array",
    "join":   "joined Datas",
}


def ran(record, result):
    """The rows of a run's assumptions: the venue's, read for the instruments the run
    held as its ``record`` keeps them, and the engine's, each priced from ``result``'s
    ledger and fill log where they measure it, then the Data's."""
    reading = _Reading(record.venue, record)
    priced  = _priced(result, record.settings)
    asset   = _settlement(record.data.instruments,
                          [kind for _, kind, _, _ in reading.held])

    return (_venue_rows(reading, priced, asset)
            + _engine_rows(reading, priced, _inside(record, len(result.time)))
            + _data_rows(record.data, record.venue))


def described(venue):
    """The rows of ``venue``'s assumptions read from the venue alone: its own and the
    engine's, for every instrument it lists and one of each kind it takes but lists
    none of."""
    reading = _Reading(venue)
    return _venue_rows(reading) + _engine_rows(reading)


def bars(data):
    """The rows of ``data``'s assumptions, read from the Data alone."""
    return _data_rows(data)


class _Reading:
    """What one venue's rows read: the ``venue``, the preset it is a variant of and its
    settings for the whole account, and each instrument as its name, its kind, its key
    in the venue's tables and the Engine's settings for it, the run's as ``record``
    keeps them, or, where it is None, every instrument the venue lists and one of each
    kind it takes but lists none of, named by its kind."""


    def __init__(self, venue, record=None):
        preset = venue._preset
        self.venue, self.record = venue, record
        self.base  = None if venue._source is None else venues._preset(venue._source)
        self.level = venues._venue_settings(preset)

        if record is None:
            self.held = [(name, kind, key, venues._instrument_settings(preset, key))
                         for name, kind, key in _listed(preset)]
        else:
            data, settings = record.data, record.settings
            self.held = []
            for instrument, name, spec in zip(data.instruments, data.names,
                                              settings.get("instruments", [settings])):
                kind, symbol = KINDS[spec["market"]], instrument.symbol or name
                key = symbol if venue.kind else f"{kind}:{symbol}"
                self.held.append((name, kind, key, spec))

        self.perps = [entry for entry in self.held if entry[1] != "spot"]
        fetched    = (None if record is None
                      else record.data._root.provenance.get("fetched"))
        self.router = ("the router" if fetched is None
                       else f"the router, fetched {fetched}")


    def source(self, path, entry=None, calibrated=False):
        """Where the value of setting ``path`` came from: the venue's own, or where
        ``entry`` is an instrument, its own table's over its kind's over the venue's,
        read against the preset the venue is a variant of and the plain venue, and
        where ``calibrated``, a preset's value apart from the plain one read as a
        starting calibration."""
        key, kind = (None, None) if entry is None else (entry[2], entry[1])
        value = _given(self.venue._preset, path, key, kind)
        plain = _at(PLAIN, path)
        base  = self.base

        if base is not None:
            given = _given(base, path, key, kind)
            if given is not None:
                if value != given:
                    return USER
                if calibrated and value != plain:
                    return f"{base['name']}, a starting calibration"
                return f"{base['name']}, taken {base['taken']}"

        return "plain" if value == plain else USER


    def cell(self, name, entries, form, sources, by_kind=False, cost=None):
        """Row ``name`` read for each instrument of ``entries``: ``form`` gives its
        text, grade and direction from its Engine settings, and ``sources`` the label
        and the source of each setting that decided it; ``by_kind`` gives each kind's
        value apart on instruments of several."""
        counted = self.record is None
        read    = [(entry[0], entry[1]) + form(entry[3]) for entry in entries]
        texts   = [text for _, _, text, _, _ in read]
        lead    = max(dict.fromkeys(texts), key=texts.count)

        value = _valued([(said, kind, text) for said, kind, text, _, _ in read],
                        counted, by_kind)
        grade, direction = _graded([(grade, way) for _, _, _, grade, way in read])
        cells = [(entry[0], sources(entry)) for entry in entries]
        led   = {said for said, _, text, _, _ in read if text == lead}

        return name, value, _sourced(cells, led, counted), grade, direction, cost


def _venue_rows(reading, priced=None, asset=None):
    """The venue's rows for ``reading``'s instruments, each priced from ``priced``
    where a run measured it, the balance in ``asset`` where it is known."""
    venue, level = reading.venue, reading.level
    held, perps = reading.held, reading.perps
    defaults = venue._preset["defaults"]
    source, cell = reading.source, reading.cell

    priced = priced or {}
    rows   = []

    def plain(name, value, path, grade, direction=None, calibrated=False, cost=None):
        rows.append((name, value, source(path, calibrated=calibrated), grade, direction,
                     cost))

    balance = venue._balance
    if isinstance(balance, dict):
        value = (f"{_amount(balance['spot'], asset)} spot and "
                 f"{_amount(balance['margin'], asset)} margin")
    else:
        value = _amount(balance, asset)

    rows.append(("balance", value, "plain" if balance == PLAIN["balance"] else USER,
                 "venue", None, None))

    def taker(spec):
        return spec.get("fee_taker", level["fee_taker"])

    def fees(spec):
        if taker(spec) <= 0.0:
            return "not charged", "optimistic", None
        maker = spec.get("fee_maker", level["fee_maker"])
        return f"maker {_rate(maker)}, taker {_rate(taker(spec))}", "venue", None

    rows.append(cell("fees", held, fees,
                     lambda entry: [(side, source(("fees", side), entry))
                                    for side in ("maker", "taker")],
                     by_kind=True, cost=priced.get("fees")))

    def tiers(spec):
        return spec.get("tiers") or []

    def keeps(spec):
        return any(rate > 0.0 for _, _, rate, _ in tiers(spec))

    def perpetual(name, form, sources, off):
        if perps:
            rows.append(cell(name, perps, form, sources, cost=priced.get(name)))
        else:
            rows.append((name, "no perpetual", source(off), "venue", None, None))

    def leverage(spec):
        if tiers(spec):
            if any(lever == 0.0 for _, lever, _, _ in tiers(spec)):
                return "not capped", "optimistic", None
            return "each instrument's tier table", "venue", None

        if level["leverage"] == 0.0:
            return "not capped", "optimistic", None
        return f"capped at {level['leverage']:g}x", "simplified", "either"

    perpetual("leverage", leverage,
              lambda entry: [("tiers", source(("tiers",), entry)) if tiers(entry[3])
                             else ("leverage", source(("leverage",)))], ("leverage",))

    def kept(spec):
        if keeps(spec):
            return "each instrument's tier table", "venue", None
        return "not kept", "optimistic", None

    perpetual("maintenance", kept, lambda entry: [("tiers", source(("tiers",), entry))],
              ("tiers",))

    model = defaults["liquidation"]
    if not perps:
        plain("liquidation", "no perpetual", ("liquidation",), "venue")
    else:
        value = ("by takeover" if model == "takeover" else
                 f"on the book, then by backstop at {_share(defaults['backstop'])}")
        inputs = [("liquidation", source(("liquidation",)))]
        if model != "takeover":
            inputs.append(("backstop", source(("backstop",))))
        rows.append(("liquidation", value, _sourced([(None, inputs)], {None}, False),
                     "venue", None, priced.get("liquidation")))

    fee = defaults.get("liquidation_fee")

    def closing(spec):
        if keeps(spec):
            return "unused, as maintenance is kept", "venue", None
        if (taker(spec) if fee is None else fee) <= 0.0:
            return "not charged", "optimistic", None
        return "the taker rate" if fee is None else _rate(fee), "simplified", "either"

    def closed(entry):
        if keeps(entry[3]):
            return [("tiers", source(("tiers",), entry))]
        if fee is None:
            return [("taker", source(("fees", "taker"), entry))]
        return [("liquidation_fee", source(("liquidation_fee",)))]

    perpetual("liquidation fee", closing, closed, ("liquidation_fee",))

    most = level["max_open_orders"]
    plain("max open orders", f"{most} an instrument", ("max_open_orders",),
          "simplified", "pessimistic" if most <= PLAIN["max_open_orders"] else "either")

    limit = level["limit"]

    def ruled(spec):
        return spec.get("tick") or spec.get("sigfigs") is not None

    def through(spec):
        if limit == "through" and ruled(spec):
            return "through, half a step past the limit", "simplified", "either"
        return "on touch", "optimistic", None

    def limited(entry):
        # under through, an instrument with no price rule fills on touch, which the
        # rule it lacks decided as much as the setting
        sources = [("limit", source(("fills", "limit")))]
        if limit == "through" and not ruled(entry[3]):
            sources.append(("tick", source(("tick",), entry)))
        return sources

    rows.append(cell("limit fills", held, through, limited))

    plain("window", _counted(level["window"], "bar"), ("fills", "window"), "simplified",
          "either")
    plain("liquidity", f"{_rate(level['budget'])} of the trailing volume a bar",
          ("fills", "liquidity"), "simplified", "either", calibrated=True)

    # a row reads the value, never the option's word, so a fixed spread of 0 bp is
    # one not modelled
    spread, paid = level["spread"], priced.get("spread")
    if spread is None or spread == 0.0:
        plain("spread", "not modelled", ("fills", "spread"), "optimistic", cost=paid)
    elif spread == "estimated":
        plain("spread", "estimated, which errs high on the most liquid perpetuals",
              ("fills", "spread"), "simplified", "either", cost=paid)
    else:
        plain("spread", f"fixed at {_bps(spread)}", ("fills", "spread"), "simplified",
              "either", cost=paid)

    slippage, paid = level["slippage_bps"], priced.get("slippage")
    if slippage == 0.0:
        plain("slippage", "not modelled", ("fills", "slippage"), "optimistic",
              calibrated=True, cost=paid)
    else:
        plain("slippage", _bps(slippage), ("fills", "slippage"), "simplified", "either",
              calibrated=True, cost=paid)

    paid = priced.get("impact")
    if level["impact"] == 0.0:
        plain("impact", "not modelled", ("fills", "impact"), "optimistic",
              calibrated=True, cost=paid)
    else:
        plain("impact", f"{level['impact_model']}, k = {level['impact']:.3g}",
              ("fills", "impact"), "simplified", "either", calibrated=True, cost=paid)

    sizes = level["sizes"]
    if sizes:
        plain("sizes", "on the lot and the price rule", ("fills", "sizes"), "venue",
              cost=priced.get("sizes"))
    else:
        plain("sizes", "not rounded or bounded", ("fills", "sizes"), "optimistic",
              cost=priced.get("sizes"))

    limits = ("min_notional", "max_qty")

    def bounded(spec):
        given = tuple(spec.get(name) is not None for name in limits)
        if reading.record is None and not all(given):
            return "from the Data's instruments", "venue", None
        if all(given):
            return "a minimum notional and a maximum size", "venue", None

        said = {
            (False, True):  "a maximum size, no minimum notional",
            (True, False):  "a minimum notional, no maximum size",
            (False, False): "none given",
        }[given]
        return said, "simplified", "optimistic"

    def bound(entry, name):
        # a limit no table of the venue gives is the router's, which a run's Data gave
        # the instrument, or in the venue's own rows the Data's to give
        if _given(venue._preset, (name,), entry[2], entry[1]) is not None:
            return source((name,), entry)
        if entry[3].get(name) is not None:
            return reading.router
        return "the Data" if reading.record is None else source((name,), entry)

    if sizes:
        rows.append(cell("order limits", held, bounded,
                         lambda entry: [(name, bound(entry, name)) for name in limits]))
    else:
        plain("order limits", "sizes are off", ("fills", "sizes"), "venue")

    intrabar = venue._intrabar
    grade    = {
        "pessimistic": ("simplified", "pessimistic"),
        "optimistic":  ("optimistic", None),
        "random":      ("simplified", "either"),
    }
    rows.append(("intrabar", intrabar,
                 "plain" if intrabar == PLAIN["intrabar"] else USER,
                 *grade[intrabar], None))

    funding, paid = defaults["funding"], priced.get("funding")
    if not perps:
        plain("funding", "no perpetual", ("funding",), "venue")
    elif funding == "data":
        plain("funding", "from the data", ("funding",), "venue", cost=paid)
    elif funding == "none" or funding["rate"] == 0.0:
        plain("funding", "not charged", ("funding",), "optimistic", cost=paid)
    else:
        if "every" in funding:
            hours = funding["every"]
            every = "hour" if hours == 1 else f"{hours:g} hours"
        else:
            bars  = funding["every_bars"]
            every = "bar" if bars == 1 else f"{bars:,} bars"
        plain("funding", f"every {every} at {_rate(funding['rate'])}", ("funding",),
              "simplified", "either", cost=paid)

    return rows


def _engine_rows(reading, priced=None, inside=None):
    """The engine's own rules, each read in its off form where it cannot act on
    ``reading``'s instruments, the margin bound priced from ``priced`` where a run
    measured it, and with ``inside`` the funding events the run had inside a bar, or
    where it is None, as the venue alone says them."""
    venue, perp = reading.venue, bool(reading.perps)
    spot    = any(kind == "spot" for _, kind, _, _ in reading.held)
    funding = venue._preset["defaults"]["funding"]
    charged = funding == "data" or isinstance(funding, dict) and funding["rate"] != 0.0
    setting = venue._intrabar
    rows    = []

    def rule(name, value, grade, direction=None, off=None, cost=None):
        if off is None:
            rows.append((name, value, ENGINE, grade, direction, cost))
        else:
            rows.append((name, off, ENGINE, "venue", None, None))

    held     = None if perp else "no perpetual"
    unfunded = held or (None if charged else "no funding charged")

    rule("margin", "cross, every perpetual on one wallet", "venue", off=held)
    rule("spot borrowing", "not modelled, a spot buy held to the cash", "venue",
         off=None if spot else "no spot instrument")
    rule("liquidation trigger", "the bar's trade prices, not a mark", "simplified",
         "pessimistic", off=held)
    rule("funding price", "the close, not the mark", "simplified", "either",
         off=unfunded)
    rule("fills in the bar", "a taker's price held inside the bar's range",
         "simplified", "optimistic")
    rule("margin bound", "a reducing fill held where the margin runs out, what the bar "
         "cannot hold paid in", "venue", off=held,
         cost=None if priced is None else priced["margin bound"])
    # what an instrument with no bar hides is a liquidation, which a run of no
    # perpetual cannot have
    rule("an instrument with no bar", "held at its last mark",
         *(("simplified", "optimistic") if perp else ("venue", None)))
    rule("liquidation in slices", "not modelled, which errs against large positions",
         "simplified", "pessimistic", off=held)
    rule("auto-deleveraging", "not modelled, which flatters winners", "simplified",
         "optimistic", off=held)
    rule("a flip held back", "held whole, the side it opens with it", "simplified",
         "optimistic", off=held or ("none, under the optimistic setting"
                                    if setting == "optimistic" else None))

    if inside is None:
        counted = ""
        quiet   = ("no event falls inside a bar"
                   if isinstance(funding, dict) and "every_bars" in funding else None)
    else:
        counted = f", {_counted(inside, 'event')} inside a bar"
        quiet   = "no event fell inside a bar" if inside == 0 else None

    within = unfunded or quiet
    rule("funding paid inside a bar", f"paid on what the bar's fills left{counted}",
         "simplified", "either", off=within)
    rule("funding counted inside a bar", f"counted on the size the path holds{counted}",
         "simplified", {"pessimistic": "optimistic",
                        "optimistic": "pessimistic"}.get(setting, "either"), off=within)
    rule("a leg opened on the favourable side",
         "runs to its adverse extreme, under the optimistic setting", "simplified",
         "pessimistic", off=held or ("no exception under the pessimistic setting"
                                     if setting == "pessimistic" else None))

    return rows


def _data_rows(data, venue=None):
    """``data``'s rows, its bars graded as another venue's where they are the router's
    and ``venue``, the run's, is a preset of a different exchange."""
    root       = data._root
    provenance = root.provenance
    instruments, names = data.instruments, data.names
    exchanges = sorted({instrument.venue for instrument in instruments
                        if instrument.venue})

    # bars are the router's where every instrument names its venue, whatever the
    # provenance says of how the Data was put together
    router = all(instrument.venue for instrument in instruments)

    said = " and ".join(SOURCES.get(word, word)
                        for word in provenance["source"].split(", "))
    if "fetched" in provenance:
        said = said.replace("the router",
                            f"the router, fetched {provenance['fetched']}")
    if "resampled_from" in provenance:
        said += f", resampled from {provenance['resampled_from']}"

    own = (venue is None or venue._source is None
           or venue._source.partition("_")[0] in exchanges)
    graded = ("venue", None) if router and own else ("simplified", "either")
    span   = (f"{data.interval}, {_when(data.time[0])} to {_when(data.time[-1])}"
              if len(data) else f"{data.interval}, no bars")
    rows = [("bars", f"{span}, from {' and '.join(exchanges)}" if router else span,
             said, *graded, None)]

    gaps    = data.gaps()
    longest = max((gap.missing for gap in gaps), default=0) * root.step
    rows.append(("gaps", f"{len(gaps):,}, the longest {_lasting(longest)}" if gaps
                 else "none", said, "venue", None, None))

    events = len(data.funding)
    rows.append(("funding events", f"{events:,}" if events else "none", said, "venue",
                 None, None))

    # every view of a fetch shares its provenance, so a selection reads the history
    # of its own instruments alone
    symbols = {instrument.symbol for instrument in instruments}
    history = {symbol: when for symbol, when in provenance.get("history", {}).items()
               if symbol in symbols}
    value = "from each instrument's first bar"
    if history:
        value += "; the router's history of " + ", of ".join(
            f"{symbol} begins at {when}" for symbol, when in history.items())
    rows.append(("history", value, said, "venue", None, None))

    available, ended = data.available, []
    if len(data):
        last  = len(data) - 1 - np.argmax(available[::-1], axis=0)
        ended = [f"{name}'s bars end {_when(data.time[at])}"
                 for name, at, printed in zip(names, last, available.any(axis=0))
                 if printed and at < len(data) - 1]

    # bars that stop before the last row, the router's among them, leave a position
    # held at its last mark, which is no delisting
    if ended:
        more = f" and {len(ended) - 3:,} more" if len(ended) > 3 else ""
        rows.append(("delisting", f"no delisting time; {', '.join(ended[:3])}{more}",
                     said, "simplified", "either", None))
    elif router:
        rows.append(("delisting", "no delisting time; the router serves none", said,
                     "venue", None, None))
    else:
        rows.append(("delisting", "no delisting time; every instrument's bars run to "
                     "the last", said, "venue", None, None))

    rows.append(("universe", f"the user's list of {_counted(len(names), 'instrument')}",
                 said, "simplified", "optimistic", None))

    if router:
        rows.append(("prices", "as the venue printed them", said, "venue", None, None))
    else:
        rows.append(("prices", "as given; splits, dividends and borrow costs only as "
                     "the user adjusted them", said, "simplified", "either", None))

    return rows


def _priced(result, settings):
    """What each row with a cost cost in ``result``, in the asset its account settles
    in, read from its costs and its fill log, nothing run again: the fees less those a
    takeover paid, which are the liquidation fee's, the margin bound minus what it gave,
    the price it lifted times each fill's size and what it paid in, and the liquidation
    charge less that."""
    costs, fills = result.costs, result.fills
    taken    = (fills["reason"] == "liquidation") & ~fills["taker"]
    takeover = math.fsum(fills["fee"][taken])
    size, lifted, price = fills["size"], fills["lifted"], fills["price"]
    if settings["market"] == "inverse":
        # a contract is worth the contract's dollars in the coin at each price
        moved = np.abs(1.0 / (price - lifted) - 1.0 / price) * settings["contract"]
    else:
        moved = np.abs(lifted)

    bound = 0.0 - math.fsum(size * moved + fills["paid_in"])

    return {
        "fees":            costs["fees"] - takeover,
        "liquidation fee": takeover,
        "funding":         costs["funding"],
        "spread":          costs["spread"],
        "slippage":        costs["slippage"],
        "impact":          costs["impact"],
        "margin bound":    bound,
        "liquidation":     costs["liquidation"] - bound,
        "sizes":           costs["dust"],
    }


def _inside(record, rows):
    """How many funding events the run's Engine placed inside a bar of its first
    ``rows`` rows, as ``record`` keeps its settings."""
    settings = record.settings
    times    = settings["times"]

    if "funding_hours" in settings:
        counts = funding_inside(times, funding_hours=settings["funding_hours"])
    elif "funding_times" in settings:
        counts = funding_inside(times, funding_times=settings["funding_times"])
    else:
        return 0

    return int(counts[record.start:record.start + rows].sum())


def _listed(preset):
    """Each instrument ``preset`` lists, as its key, its kind and its key, and one of
    each kind it takes but lists none of, named by its kind, as its checks read them: a
    venue of one kind takes that kind, and one of several the kinds its tables name,
    every kind where it lists no instrument either."""
    listed, kind = list(preset["instruments"]), preset["kind"]
    if kind is not None:
        return [(key, kind, key) for key in listed] or [(kind, kind, "placeholder")]

    held  = [(key, key.partition(":")[0], key) for key in listed]
    kinds = list(preset["kinds"]) or ([] if listed else list(venues.KINDS))
    taken = {kind for _, kind, _ in held}

    return held + [(each, each, f"{each}:placeholder") for each in kinds
                   if each not in taken]


def _given(preset, path, key=None, kind=None):
    """The value setting ``path`` takes in ``preset``: the venue's own, or where ``key``
    names an instrument of ``kind``, its own table's over its kind's over the venue's,
    None where none gives it."""
    tables = [preset["defaults"]]
    if key is not None:
        tables = [preset["instruments"].get(key, {}),
                  preset["kinds"].get(kind, {})] + tables

    for table in tables:
        value = _at(table, path)
        if value is not None:
            return value

    return None


def _at(table, path):
    """What ``table`` holds at ``path``, None where it holds nothing there."""
    for part in path:
        if not isinstance(table, dict) or part not in table:
            return None
        table = table[part]

    return table


def _valued(forms, counted, by_kind):
    """The value of a row whose instruments read ``forms``, each its name, its kind and
    its text: the text alone where they agree, else with ``by_kind`` each kind's apart
    where they are of several, else the text most of them read, then each other one
    after the instruments that read it."""
    texts = [text for _, _, text in forms]
    if len(set(texts)) == 1:
        return texts[0]

    kinds = list(dict.fromkeys(kind for _, kind, _ in forms))
    if by_kind and len(kinds) > 1:
        apart = [[form for form in forms if form[1] == kind] for kind in kinds]
        return "; ".join(f"{kind} {_valued(own, counted, False)}"
                         for kind, own in zip(kinds, apart))

    lead = max(dict.fromkeys(texts), key=texts.count)
    return "; ".join([lead] + [
        f"{_named([name for name, _, said in forms if said == text], counted)} {text}"
        for text in dict.fromkeys(texts) if text != lead])


def _graded(forms):
    """The least favourable grade of ``forms``, each a grade and a direction, with the
    direction of its simplified ones where that is the grade, either where they
    differ."""
    grade = max((grade for grade, _ in forms), key=GRADES.index)
    if grade != "simplified":
        return grade, None

    ways = {way for given, way in forms if given == "simplified"}
    return grade, ways.pop() if len(ways) == 1 else "either"


def _sourced(cells, led, counted):
    """The source of a row whose instruments' settings came from ``cells``, each an
    instrument's name with each setting's label and source, ``led`` naming those of
    the form its value leads with: the source most of their settings came from, then
    each other source and what it decided, the same settings of every instrument named
    by the settings alone, and otherwise each set of instruments it decided the same
    settings of, with those settings where they are not all of theirs."""
    sources = [source for name, given in cells if name in led for _, source in given]
    first   = max(dict.fromkeys(sources), key=sources.count)
    labels  = {label for _, given in cells for label, _ in given}
    out     = [first]

    for other in dict.fromkeys(source for _, given in cells for _, source in given):
        if other == first:
            continue

        decided = {}
        for name, given in cells:
            keys = tuple(label for label, source in given if source == other)
            if keys:
                decided.setdefault(keys, []).append(name)

        if len(decided) == 1 and len(next(iter(decided.values()))) == len(cells):
            what = _listing(list(next(iter(decided))))
        else:
            what = _listing([_named(names, counted) if set(keys) == labels
                             else f"{_named(names, counted)} {_listing(list(keys))}"
                             for keys, names in decided.items()])
        out.append(f"{what} {other}")

    return "; ".join(out)


def _named(names, counted):
    """The instruments ``names``, named, or in a venue's own rows counted, where a kind
    stands for the instrument of it the venue does not list."""
    if not counted:
        return _listing(names)

    kinds  = [name for name in names if name in venues.KINDS]
    listed = len(names) - len(kinds)
    return _listing(kinds + ([_counted(listed, "instrument")] if listed else []))


def _listing(words):
    """``words`` as a sentence lists them: A, A and B, A, B and C."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _counted(n, noun):
    """``n`` of ``noun``, such as 1 bar or 4,383 events."""
    return f"{n:,} {noun}{'s' * (n != 1)}"


def _amount(value, asset):
    """An amount of money, in ``asset`` where it is known."""
    said = f"{value:,.12g}"
    return f"{said} {asset}" if asset else said


def _rate(rate):
    """A rate as a percentage, to three figures."""
    return f"{rate * 100.0:.3g}%"


def _bps(bps):
    """A number of basis points, to three figures."""
    return f"{bps:.3g} {'bp' if bps == 1.0 else 'bps'}"


def _share(share):
    """A share as a half, a third or a quarter where it is one, else a percentage."""
    fraction = fractions.Fraction(share).limit_denominator(4)
    if math.isclose(float(fraction), share, rel_tol=1e-12):
        return f"{fraction.numerator}/{fraction.denominator}"
    return _rate(share)


def _lasting(span):
    """A span of nanoseconds in the largest unit it is a whole number of."""
    for unit, length in (("day", DAY), ("hour", HOUR), ("minute", MINUTE)):
        if span % length == 0:
            return _counted(span // length, unit)

    return _counted(span // 1_000_000_000, "second")
