"""Run the engine and the reference over the same random cases and diff them.

The reference decides each case's actions on its own account as it goes, and they
are written down as a script the engine then plays in one call through the
binding's `run`, which names an order by the placement that made it. The reference
is written from the Specification and never from the engine, so a case the two part
company on is a question about a rule, and somebody has to decide which reading is
wrong. The engine still carries emsl's rules where the Specification has
moved on, so a case is compared up to the bar on which the reference first applied
a rule the engine has yet to take up. The reference records the bar each rule first
applies on, which is what decides it, and a stage that brings a rule into the engine
takes it off `NOT_IN_THE_ENGINE`, so the part of each case compared grows with the
engine.

  python -m reference.differential [cases] [seed]     (from tests/)
"""

import collections
import math
import os
import random
import re
import sys

import numpy as np

import probatum
from .simulator import Bar, Instrument, Simulator, Venue

HOUR = 3_600 * 10**9
TOL  = 1e-6

# the rules the reference applies that the engine does not have yet
NOT_IN_THE_ENGINE = {
    "a waiting market order cancelled",
}


def walk(rng, n):
    """Consistent bars from a random walk, a third of them violent enough to liquidate
    at 10x and the rest calm enough for a levered account to live until a gap kills
    it, many opening with one. A bar trades nothing or far more than any order here
    asks, so the liquidity budget binds only where a case draws a small one."""
    bars, price = [], 100.0
    for t in range(n):
        if t > 0 and rng.random() < 0.4:
            price = max(price * (1.0 + rng.uniform(-0.2, 0.2)), 0.01)

        move, wick = rng.choice([(0.02, 0.01), (0.02, 0.01), (0.25, 0.2)])
        close  = max(price * (1.0 + rng.uniform(-move, move)), 0.01)
        high   = max(price, close) * (1.0 + rng.uniform(0.0, wick))
        low    = max(min(price, close) * (1.0 - rng.uniform(0.0, wick)), 0.001)
        dead   = t > 0 and bars[-1].volume > 0.0 and rng.random() < 0.1
        volume = 0.0 if dead else rng.uniform(1e6, 1e7)
        bars.append(Bar(price, high, low, close, volume))
        price = close

    return bars


def clock(rng, n):
    """Open times an hour apart, but for a closure now and then of two to six hours,
    as a market that closes or a feed with a hole in it has."""
    times, at = [], 0
    for _ in range(n):
        times.append(at)
        at += HOUR * (rng.choice([2, 3, 6]) if rng.random() < 0.1 else 1)

    return times


def stamps(rng, times):
    """Funding events for a venue that charges from its own history: at a row's end,
    inside a row, across a closure, and a fraction of a second off a boundary, where
    it is taken as the boundary, each at its own rate."""
    events = []
    for _ in range(rng.randint(1, len(times))):
        row   = rng.randrange(len(times))
        end   = times[row] + HOUR
        after = times[row + 1] if row + 1 < len(times) else end
        stamp = rng.choice([
            end,
            times[row] + rng.randint(1, HOUR - 1),
            rng.randint(end, max(after, end)),
            end + rng.choice([-1, 1]) * rng.randint(1, 10**9 - 1),
        ])
        events.append((stamp, rng.choice([0.0001, 0.02, -0.01, 0.1])))

    return events


def config(rng, times, inverse=False):
    """A venue for one instrument, spot or a perpetual, or with `inverse` an inverse
    perpetual on contracts of 10 or 100 dollars, its balance in the coin and no
    other wallet."""
    market  = "inverse" if inverse else rng.choice(["spot", "perp"])
    perp    = market != "spot"
    funding = ("none",)
    if perp:
        rate    = rng.choice([0.0001, 0.02, -0.01, 0.1])
        funding = rng.choice([
            ("none",),
            ("every_bars", 3, rate),
            ("every", rng.choice([0.5, 1.0, 3.0, 8.0]), rate),
            ("data", stamps(rng, times)),
        ])

    cfg = dict(
        market=market,
        quote=rng.choice([1_000.0, 10_000.0]),
        # the other wallet's opening cash, which only a transfer puts to work
        idle=rng.choice([0.0, 0.0, 500.0, 5_000.0]),
        fee_taker=rng.choice([0.0, 0.0006, 0.002]),
        fee_maker=rng.choice([0.0, 0.0002]),
        slippage_bps=rng.choice([0.0, 5.0, 50.0]),
        # at 1 every second placement is refused, which is the only way the bound on
        # the book and on the pending queue gets reached
        max_open_orders=rng.choice([1, 2, 8]),
        leverage=rng.choice([0.0, 3.0, 10.0]) if perp else 10.0,
        funding=funding,
    )

    if inverse:
        # a balance in the coin worth at the walk's opening price of 100 what a
        # linear case holds in dollars, so its sizes in contracts sit as far below
        # the walk's volumes as a linear case's do
        cfg.update(quote=cfg["quote"] / 100.0, idle=0.0,
                   contract=rng.choice([10.0, 100.0]))

    cfg["tiers"], cfg["tiers_by"] = tier_table(rng, cfg["quote"], perp,
                                              contract=cfg.get("contract"))
    return cfg


def tier_table(rng, quote, perp, price=100.0, contract=None):
    """No table for most cases, and for some perpetuals three tiers of falling
    leverage, their bounds a few times the balance in notional, or in size at
    `price`, the walk's opening price, so the cap meets them, in contracts of
    `contract` dollars on an inverse perpetual. Half keep maintenance at rising
    rates, a notional table's amounts continuous across its bounds."""
    if not perp or rng.random() < 0.6:
        return None, "notional"

    basis = rng.choice(["notional", "quantity"])
    scale = quote if basis == "notional" else quote / price
    if basis == "quantity" and contract is not None:
        scale = quote * price / contract

    bounds   = [scale * b for b in sorted(rng.sample([1.0, 2.0, 4.0, 8.0, 16.0], 3))]
    leverage = sorted(rng.sample([20.0, 10.0, 5.0, 3.0, 2.0], 3), reverse=True)
    rates    = [0.0] * 3
    if rng.random() < 0.5:
        rates = sorted(rng.sample([0.002, 0.005, 0.01, 0.02], 3))

    amounts = [0.0] * 3
    if basis == "notional":
        for k in (1, 2):
            amounts[k] = amounts[k - 1] + (rates[k] - rates[k - 1]) * bounds[k - 1]

    return list(zip(bounds, leverage, rates, amounts)), basis


def balance(cfg):
    """The opening cash both take: a number for the wallet the market settles in, or
    both wallets by name when the other holds cash too."""
    if not cfg["idle"]:
        return cfg["quote"]

    if cfg["market"] == "spot":
        return {"spot": cfg["quote"], "margin": cfg["idle"]}
    return {"spot": cfg["idle"], "margin": cfg["quote"]}


def money(cfg):
    return cfg["quote"] + cfg["idle"]


# every order surface both simulators have, so the gate exercises all of them.
# Narrow it with KINDS=limit when a disagreement needs isolating
KINDS = os.environ.get(
    "KINDS",
    "market,limit,stop,fok,post,cancel,replace,reduce,close,ioc_limit,fok_limit,"
    "cancel_all,reduce_limit,stop_entry,exits,transfer,target,retarget,bracket",
).split(",")

# the kinds that only mean something with a live position or order to act on; a run
# in which one never met anything agreed about nothing
TRACKED = [kind for kind in ("close", "reduce", "cancel", "replace", "cancel_all",
                             "ioc_limit", "fok_limit", "reduce_limit", "stop_entry",
                             "exits", "bracket")
           if kind in KINDS]


def plan(rng, bars, cfg):
    """One action per bar but the last: a kind, a side, a size and a draw `u` that
    sets the action's price off the close it is decided at. A size runs from a
    twentieth of the balance's worth at that close to twelve times it, so the cap
    binds, the spot clamps bind and levered accounts die."""
    out = []

    # an inverse account has one wallet, so nothing to transfer to, and its sizes
    # are contracts
    kinds = [kind for kind in KINDS if cfg["market"] != "inverse" or kind != "transfer"]

    for t in range(len(bars) - 1):
        if rng.random() < 0.45:
            out.append(None)
            continue

        worth = cfg["quote"] / bars[t].close
        if cfg["market"] == "inverse":
            worth = cfg["quote"] * bars[t].close / cfg["contract"]
        size = worth * math.exp(rng.uniform(math.log(0.05), math.log(12.0)))
        out.append((rng.choice(kinds), rng.choice(["buy", "sell"]), size,
                    rng.random()))

    return out


def limit_price(side, close, u):
    """Mostly passive, and within a hundredth of the close on the other side, where
    the next bar usually trades through it and fills it as a taker."""
    if side == "buy":
        return close * (0.8 + 0.21 * u)
    return close * (0.99 + 0.21 * u)


def stop_price(close, u):
    return close * (0.8 + 0.4 * u)


def bracket_prices(held, side, close, u):
    """A bracket's stop and take-profit about `close`, below and above it for the
    long held, or for the long a buy is opening, and mirrored for a short, now and
    then with only one of the two."""
    low, high = close * (0.8 + 0.19 * u), close * (1.2 - 0.19 * u)
    long = held > 0.0 if held != 0.0 else side == "buy"
    stop, take_profit = (low, high) if long else (high, low)

    if u < 0.15:
        return stop, None
    if u > 0.85:
        return None, take_profit
    return stop, take_profit


def bracket(sim, i, t, side, size, close, u, theirs, script, **on):
    """A bracket on instrument `i`, on the position it holds, or placed while flat
    after the entry it waits for, a market order half the time, otherwise a limit or
    a breakout stop, so it arms when that entry fills, inside its bar where the bar
    reaches a resting one; a spot entry buys, since spot cannot be short. A resting
    entry's legs sit close about its own price, where a bar can reach them before
    the fill and not after it. Each order that rests is written into `theirs` with
    the placement that made it. Returns whether a leg rests."""
    held = sim.position[i]
    way  = (13.0 * u) % 1.0

    if held == 0.0:
        if sim.instruments[i].kind == "spot":
            side = "buy"

        if way < 0.5:
            script.take(t, side, size=size, tif="ioc", **on)
            sim.order(i, side, size)
        else:
            sign = 1.0 if side == "buy" else -1.0
            if way < 0.8:
                price = limit_price(side, close, u)
                entry = dict(limit=price)
            else:
                price = close * (1.0 + sign * 0.03 * ((5.0 * u) % 1.0))
                entry = dict(stop=price)

            placed = script.take(t, side, size=size, **entry, **on)
            order  = sim.order(i, side, size, **entry)
            if resting(sim, order):
                theirs.append((order, placed))

    stop, take_profit = bracket_prices(held, side, close, u)
    if held == 0.0 and way >= 0.5:
        gap         = 0.002 + 0.04 * ((11.0 * u) % 1.0)
        stop        = None if stop is None else price * (1.0 - sign * gap)
        take_profit = None if take_profit is None else price * (1.0 + sign * gap)

    placed = dict(zip(("stop", "take_profit"),
                      script.take(t, "bracket", stop=stop, take_profit=take_profit,
                                  **on)))
    legs = sim.bracket(i, stop=stop, take_profit=take_profit)
    theirs.extend((order, placed[sim.orders[order]["leg"]]) for order in legs)
    return bool(legs)


def prune(sim, theirs):
    """Drop from the front of `theirs` the orders that filled, expired or were
    cancelled, so a cancel or a replace acts on one still resting."""
    while theirs and not resting(sim, theirs[0][0]):
        theirs.pop(0)


def weight(market, u, k=0, among=1):
    """A target's weight for instrument `k` from the draw `u`, long only on spot and
    either way on a perpetual, shared among the `among` instruments it names so the
    account's exposure stays near one instrument's."""
    v = (u + 0.37 * k) % 1.0
    return (1.2 * v if market == "spot" else 6.0 * v - 3.0) / among


def settings(side, u):
    """A target's tolerance and transfer from the action's side and draw: a sell
    skips a leg below a twentieth of equity, and a high draw moves a spot shortfall
    across."""
    return dict(tolerance=0.05 if side == "sell" else 0.0, transfer=u > 0.5)


def engine_liquidation(model):
    """The engine's arguments for a liquidation model."""
    if model == "takeover":
        return dict(liquidation=model)
    return dict(liquidation=model[0], backstop=model[1])


def engine_fills(cfg):
    """The engine's arguments for a case's fill model."""
    out    = dict(budget=cfg.get("budget", 1.0), window=cfg.get("window", 24))
    spread = cfg.get("spread")
    if spread is not None:
        out["spread"] = spread if spread == "estimated" else spread[1]

    impact = cfg.get("impact")
    if impact is not None:
        out.update(impact_model=impact[0], impact=impact[1])

    return out


def engine_funding(source):
    """The engine's arguments for a funding source."""
    if source[0] == "every_bars":
        return dict(funding_interval=source[1], funding_rate=source[2])

    if source[0] == "every":
        return dict(funding_hours=source[1], funding_rate=source[2])

    if source[0] == "data":
        return dict(
            funding_times=np.array([stamp for stamp, _ in source[1]], dtype=np.int64),
            funding_rates=np.array([rate for _, rate in source[1]], dtype=np.float64),
        )

    return {}


def venue(cfg):
    """The engine's settings for a case, as the binding takes them."""
    return dict(
        market=cfg["market"], quote=balance(cfg), fee_taker=cfg["fee_taker"],
        fee_maker=cfg["fee_maker"], slippage_bps=cfg["slippage_bps"],
        max_open_orders=cfg["max_open_orders"], leverage=cfg["leverage"],
        tiers=cfg["tiers"], tiers_by=cfg["tiers_by"], intrabar=cfg["intrabar"],
        contract=cfg.get("contract"), **engine_funding(cfg["funding"]),
        **engine_liquidation(cfg.get("liquidation", "takeover")), **engine_fills(cfg),
        **engine_sizes(cfg), liquidation_fee=cfg.get("liquidation_fee"),
    )


def play(bars, times, cfg, script):
    """The engine through a whole script in one call, and what it recorded."""
    array = np.array([list(bar) for bar in bars], dtype=np.float64)
    return probatum._probatum.run(array, venue(cfg), script.bars,
                                  times=np.array(times, dtype=np.int64))


def make_reference(bars, cfg, times, interval=None, events=None, start=0):
    """The reference over one instrument on the clock `times`, its account opened at
    row `start`. `events`, when given, are the venue's own funding events in place of
    the configured source, which a batch env started part way in needs for a
    schedule counted in bars."""
    source = cfg["funding"]
    if events is None and source[0] == "data":
        events = source[1]

    if events is not None:
        funding = "data"
    elif source[0] in ("every_bars", "every"):
        funding = source
    else:
        funding = "none"

    kind  = {"perp": "linear"}.get(cfg["market"], cfg["market"])
    venue = Venue(
        balance         = balance(cfg),
        maker           = cfg["fee_maker"],
        taker           = cfg["fee_taker"],
        leverage        = cfg["leverage"],
        max_open_orders = cfg["max_open_orders"],
        slippage        = cfg["slippage_bps"],
        funding         = funding,
        intrabar        = cfg["intrabar"],
        liquidation     = cfg.get("liquidation", "takeover"),
        budget          = cfg.get("budget", 1.0),
        window          = cfg.get("window", 24),
        spread          = cfg.get("spread"),
        impact          = cfg.get("impact"),
        limit           = cfg.get("limit", "touch"),
        sizes           = cfg.get("sizes", False),
        liquidation_fee = cfg.get("liquidation_fee"),
    )

    if events is not None:
        events = [(stamp, None, rate) for stamp, rate in events]

    instrument = Instrument(
        "X",
        kind     = kind,
        contract = cfg.get("contract"),
        tiers    = cfg.get("tiers"),
        tiers_by = cfg.get("tiers_by", "notional"),
        **reference_rules(cfg, 0),
    )

    return Simulator(
        times,
        [[bar] for bar in bars],
        [instrument],
        venue,
        interval = interval,
        funding  = events,
        start    = start,
    )


def resting(sim, order_id):
    order = sim.orders.get(order_id)
    return order is not None and order["status"] == "resting"


class Script:
    """The engine's script, written as the reference decides: each bar's actions,
    and how many placements came before, which is how a cancel or a replace names
    its order."""


    def __init__(self):
        self.bars       = {}
        self.placements = 0


    def take(self, t, verb, **arguments):
        """Write one action at bar `t`, returning the placement it makes, if any: a
        bracket makes two, its stop's and then its take-profit's."""
        self.bars.setdefault(t, []).append((verb, arguments))

        if verb == "bracket":
            self.placements += 2
            return self.placements - 2, self.placements - 1

        if verb not in ("buy", "sell", "close"):
            return None

        self.placements += 1
        return self.placements - 1


def place(sim, action, t, close, theirs, script, reached=None):
    """One action, decided by the reference on its own account and taken there, and
    written into `script` for the engine. `theirs` holds the reference's resting
    orders, each its id and the placement that made it, oldest first, so a cancel
    or a replace names the same order on both sides. `reached` counts each action
    that met something live under the bar it first acts on."""
    if action is None:
        return

    kind, side, size, u = action
    prune(sim, theirs)

    if reached is not None:
        # what the action had to act on when it was placed: one that never meets a
        # live position or order agrees with anything, and after the fact there is
        # no telling that from a real agreement
        if kind in ("close", "reduce", "exits") and sim.position[0] != 0.0:
            reached[t + 1][kind] += 1
        elif kind in ("cancel", "replace") and theirs and resting(sim, theirs[0][0]):
            reached[t][kind] += 1
        elif kind == "cancel_all" and sim.resting():
            reached[t][kind] += 1

    got = None
    if kind in ("market", "fok", "reduce"):
        tif         = "fok" if kind == "fok" else "ioc"
        reduce_only = kind == "reduce"
        script.take(t, side, size=size, tif=tif, reduce_only=reduce_only)
        sim.order(0, side, size, tif=tif, reduce_only=reduce_only)
    elif kind in ("limit", "post", "ioc_limit", "fok_limit", "reduce_limit"):
        price = limit_price(side, close, u)
        tif   = {"ioc_limit": "ioc", "fok_limit": "fok"}.get(kind, "gtc")
        post_only, reduce_only = kind == "post", kind == "reduce_limit"
        placed = script.take(t, side, size=size, limit=price, tif=tif,
                             post_only=post_only, reduce_only=reduce_only)
        got = (sim.order(0, side, size, limit=price, tif=tif, post_only=post_only,
                         reduce_only=reduce_only), placed)
    elif kind in ("stop", "stop_entry"):
        # a protective stop only shrinks the position; a breakout stop opens one
        trigger, reduce_only = stop_price(close, u), kind == "stop"
        placed = script.take(t, side, size=size, stop=trigger, reduce_only=reduce_only)
        got    = (sim.order(0, side, size, stop=trigger, reduce_only=reduce_only), placed)
    elif kind == "close":
        script.take(t, "close")
        sim.close(0)
    elif kind == "exits":
        # a protective stop and a take-profit on the position held, the take-profit
        # in the earlier slot, which one wide bar can reach both of, so the worse is
        # moved ahead of it. The two tighten together, so a bar of any width can
        # reach both
        held = sim.position[0]
        if held == 0.0:
            return

        side = "sell" if held > 0.0 else "buy"
        low, high = close * (0.8 + 0.19 * u), close * (1.2 - 0.19 * u)
        trigger, price = (low, high) if held > 0.0 else (high, low)

        for leg in (dict(limit=price), dict(stop=trigger)):
            placed = script.take(t, side, size=size, reduce_only=True, **leg)
            order  = sim.order(0, side, size, reduce_only=True, **leg)
            if resting(sim, order):
                theirs.append((order, placed))
    elif kind == "bracket":
        rests = bracket(sim, 0, t, side, size, close, u, theirs, script)
        if rests and reached is not None:
            reached[t + 1][kind] += 1
    elif kind == "transfer":
        # into the margin wallet on a buy and out of it on a sell, from a twentieth
        # of the balance to twelve times it, so both clamps bind
        to = "margin" if side == "buy" else "spot"
        script.take(t, "transfer", amount=size * close, to=to)
        sim.transfer(size * close, to=to)
    elif kind in ("target", "retarget"):
        # a retarget is a first target, moving any spot shortfall across, that the
        # second replaces in the same decision
        weights = weight(sim.instruments[0].kind, u)
        if kind == "retarget":
            first = dict(settings(side, u), transfer=True)
            script.take(t, "target", weights=weights / 2.0, **first)
            sim.target(weights / 2.0, **first)

        script.take(t, "target", weights=weights, **settings(side, u))
        sim.target(weights, **settings(side, u))
    elif kind == "cancel":
        if theirs:
            order, placed = theirs.pop(0)
            script.take(t, "cancel", order=placed)
            sim.cancel(order)
    elif kind == "cancel_all":
        script.take(t, "cancel_all")
        sim.cancel_all()
        theirs.clear()
    elif kind == "replace" and theirs:
        # the side and kind stay the order's own: a replacement moves an order, it
        # never turns one into another, and it keeps its id on both sides
        order, placed = theirs[0]
        # a bracket leg given a size closes that much, and one moved without a size
        # still closes the position it finds
        sizeless = sim.orders[order]["leg"] and (7.0 * u) % 1.0 > 0.5
        size     = None if sizeless else size / 2.0

        if sim.orders[order]["kind"] == "stop":
            trigger = stop_price(close, u)
            script.take(t, "replace", order=placed, size=size, trigger=trigger)
            sim.replace(order, trigger=trigger, size=size)
        else:
            # now and then only the size moves, which is no new placement for
            # post_only, so the close is not read against the order again
            price = None if u < 0.25 else limit_price(sim.orders[order]["side"],
                                                      close, u)
            script.take(t, "replace", order=placed, size=size, price=price)
            sim.replace(order, price=price, size=size)

    if got is not None and resting(sim, got[0]):
        theirs.append(got)
        if reached is not None and live(kind, sim.position[0], side):
            reached[t + 1][kind] += 1


def live(kind, held, side):
    """Whether an order of `kind` left resting on a position of `held` acts on
    anything: a reduce_only limit only beside a position it reduces."""
    if kind == "reduce_limit":
        return (held > 0.0 and side == "sell") or (held < 0.0 and side == "buy")
    return kind in ("ioc_limit", "fok_limit", "stop_entry")


def run(bars, times, cfg, actions, reached=None):
    """The reference through the case, deciding as it goes, then the engine through
    the same decisions in one call: a row of each per bar, what the engine recorded,
    and the reference."""
    sim = make_reference(bars, cfg, times)
    script, theirs, rows = Script(), [], []
    funded = 0.0

    for t, action in enumerate(actions):
        place(sim, action, t, bars[t].close, theirs, script, reached)
        sim.step()
        funded += sim.ledger[-1]["funding"]
        rows.append((sim.position[0], sim.equity(), sim.spot_cash, sim.margin_cash,
                     funded))

    records = play(bars, times, cfg, script)
    ledger  = records["ledger"]
    paid    = np.cumsum(ledger["funding"])
    mine    = [(ledger["position"][t], ledger["equity"][t], ledger["spot_cash"][t],
                ledger["margin_cash"][t], paid[t]) for t in range(1, len(rows) + 1)]

    return list(zip(mine, rows)), records, sim


def set_aside(sim):
    """The first rule the case reached that the engine does not have yet, or None."""
    for rule in sorted(NOT_IN_THE_ENGINE):
        if sim.reached[rule]:
            return rule

    return None


def cut(sim, rules=NOT_IN_THE_ENGINE):
    """The first of `rules` the reference applied and the bar it first applied on,
    which the case is compared up to, or None to compare it whole."""
    found = [(sim.first[rule], rule) for rule in rules if rule in sim.first]
    if not found:
        return None

    bar, rule = min(found)
    return rule, bar


def tally(reached, rules, counted, sim, end):
    """Count what the part of a case compared reached: the actions that first act on
    a bar before `end`, and once a case each rule first applied before it."""
    for bar, kinds in counted.items():
        if end is None or bar < end:
            reached.update(kinds)

    rules.update(rule for rule, bar in sim.first.items() if end is None or bar < end)


def before(log, end):
    """The records of `log` written before bar `end`, or all of them with no `end`."""
    return log if end is None else [record for record in log if record["bar"] < end]


def closed_before(trips, end):
    """The round trips closed before bar `end`, or all of them with no `end`; one
    still open there has yet to take its exit, its best and its worst."""
    if end is None:
        return trips
    return [trip for trip in trips if not trip["open"] and trip["close_bar"] < end]


NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def worded(log):
    """The order log with a clamp's reason left out, which quotes sizes each simulator
    reaches by its own arithmetic, and every other reason's numbers read as floats to
    twelve figures: one prints a whole number with a point where the other leaves it
    off, and an inverse contract's sizes come back from the engine's terms a few
    floats from the contracts the reference counts in."""
    def floats(reason):
        return NUMBER.sub(lambda number: f"{float(number.group()):.12g}", reason)

    return [dict(event, reason=None) if event["kind"] == "clamped"
            else dict(event, reason=floats(event["reason"])) for event in log]


FIELDS = ("position", "equity", "spot_cash", "margin_cash", "funding_paid")


def close_enough(a, b, scale=1.0):
    if not math.isfinite(a) or not math.isfinite(b):
        return math.isfinite(a) == math.isfinite(b)
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b), scale)


def first_difference(rows, cfg):
    """The first bar the two disagree on and what about, or None."""
    scales = (1.0,) + (money(cfg),) * 4

    for i, (mine, theirs) in enumerate(rows):
        for name, a, b, scale in zip(FIELDS, mine, theirs, scales):
            if not close_enough(a, b, scale):
                return i, f"bar {i + 1}: {name} {a:.6f} against {b:.6f}"

    return None


FILL_FIELDS = ("bar", "order", "side", "size", "price", "reference", "spread",
               "slippage", "impact", "fee", "adjustment", "taker", "reason")
EVENT_FIELDS = ("bar", "kind", "order", "size", "price", "reason")
TRIP_FIELDS  = ("side", "size", "entry", "exit", "open_bar", "close_bar", "fees",
                "funding", "pnl", "ret", "best", "worst", "reason", "open")


LEDGER_FIELDS = ("equity", "equity_in_quote", "spot_cash", "margin_cash", "fees",
                 "funding", "spread", "slippage", "impact", "liquidation", "dust",
                 "position", "entry", "mark", "notional", "unrealized", "weight",
                 "drift", "margin_used", "maintenance", "gross_exposure",
                 "net_exposure", "leverage")


def ledger_difference(mine, theirs, end=None):
    """The first row the two ledgers part company on before bar `end`, or on any row
    with no `end`, as its bar and what about, or None."""
    for t, row in enumerate(theirs[:end]):
        held = row["instruments"][0]
        want = dict(row, position=held["position"], entry=held["entry"],
                    mark=held["mark"], notional=held["notional"],
                    unrealized=held["upnl"], weight=held["weight"], drift=held["drift"])
        for name in LEDGER_FIELDS:
            x = float(mine[name][t])
            if not close_enough(x, want[name]):
                return t, f"ledger row {t}: {name} {x!r} against {want[name]!r}"

    if end is None and len(mine["equity"]) != len(theirs):
        rows = min(len(mine["equity"]), len(theirs))
        return rows, f"{len(mine['equity'])} ledger rows against {len(theirs)}"

    return None


def log_difference(what, fields, mine, theirs):
    """The first row two logs part company on, as its bar and what about, or None. A
    float is held to the tolerance and anything else, a missing size included, must
    be equal."""
    for k, (a, b) in enumerate(zip(mine, theirs)):
        for name in fields:
            x, y = a[name], b[name]
            both = isinstance(x, float) and isinstance(y, float)
            if not (close_enough(x, y) if both else x == y):
                where = b.get("bar", b.get("open_bar"))
                return where, (f"{what} {k + 1}, on bar {where}: {name} {x!r} "
                               f"against {y!r}")

    if len(mine) != len(theirs):
        extra = (mine if len(mine) > len(theirs) else theirs)[min(len(mine),
                                                                  len(theirs))]
        return (extra.get("bar", extra.get("open_bar")),
                f"{len(mine)} {what}s against {len(theirs)}")

    return None


def disagreement(rows, records, sim, cfg, end=None):
    """Where the two first part company on any record written before bar `end`, or
    on any record with no `end`, as the bar and what about, or None: the account bar
    by bar, then the fill log, the order log, the round trips and the ledger."""
    found = first_difference(rows if end is None else rows[:max(end - 1, 0)], cfg)
    if found is not None:
        return found[0] + 1, found[1]

    return (log_difference("fill", FILL_FIELDS, before(records["fills"], end),
                           before(sim.fills, end))
            or log_difference("event", EVENT_FIELDS,
                              worded(before(records["order_log"], end)),
                              worded(before(sim.events, end)))
            or log_difference("round trip", TRIP_FIELDS,
                              closed_before(records["trades"], end),
                              closed_before(sim.round_trips(), end))
            or ledger_difference(records["ledger"], sim.ledger, end))


Outcome = collections.namedtuple(
    "Outcome", "compared agreed disagreed set_aside rules reached")


def setting(seed, case):
    """How case `case` of `seed` decides what its bars cannot show, drawn apart from
    the stream the cases come from, which it leaves as it was."""
    draw = random.Random(f"intrabar {seed} {case}")
    return draw.choice(["pessimistic", "optimistic"])


def fills(seed, case):
    """The fill model case `case` of `seed` trades under, drawn apart from the stream
    the cases come from: a budget that binds now and then against the walk's
    volumes, a window, and a spread and an impact or none."""
    draw = random.Random(f"fills {seed} {case}")
    return dict(
        budget=draw.choice([1.0, 1.0, 1e-4, 2e-5]),
        window=draw.choice([24, 24, 3]),
        spread=draw.choice([None, None, ("fixed", 10.0), "estimated"]),
        impact=draw.choice([None, None, ("linear", 50.0), ("sqrt", 1.0)]),
    )


def sizes(seed, case, quote, opens, contract=None):
    """The size and price rules case `case` of `seed` trades under, drawn apart from
    the stream the cases come from: whether a limit fills on touch or a step
    through, whether sizes hold, and for each instrument, at `opens` its first opens,
    a tick or, off an inverse contract, Hyperliquid's figures, and a lot, a minimum
    notional and a maximum order quantity scaled to what `quote` buys there, in
    contracts of `contract` dollars on an inverse perpetual."""
    draw = random.Random(f"sizes {seed} {case}")
    out  = dict(limit=draw.choice(["touch", "touch", "through"]),
                sizes=draw.random() < 0.5, rules=[])

    for price in opens:
        worth   = quote * price / contract if contract else quote / price
        kinds   = ["free", "fine", "coarse"] + ([] if contract else ["figures"])
        rule    = draw.choice(kinds)
        figures = rule == "figures"
        lot     = draw.choice([0.0, worth / 1_000.0, worth / 20.0])

        out["rules"].append(dict(
            tick={"fine": price / 10_000.0, "coarse": price / 200.0}.get(rule, 0.0),
            # two and three figures keep only whole numbers at the walks' prices
            sigfigs=draw.choice([2, 3, 5]) if figures else None,
            decimals=draw.choice([1, 2, 4]) if figures else None,
            lot=lot,
            min_notional=draw.choice([0.0, 0.0, quote / 10.0]),
            # a maximum off the lot as well as on it
            max_qty=draw.choice([None, None, 3.0 * worth, 3.0 * worth + lot / 2.0]),
            reduce_only_exempt=draw.random() < 0.5,
        ))

    return out


def opened(seed, case, tables):
    """Case `case` of `seed`'s tier `tables`, a third of them with the last bound left
    open, as a venue that sets no largest position leaves it, drawn apart from the
    stream the cases come from."""
    draw = random.Random(f"open {seed} {case}")
    out  = []

    for rows in tables:
        if rows is not None and draw.random() < 1.0 / 3.0:
            rows = rows[:-1] + [(math.inf, *rows[-1][1:])]
        out.append(rows)

    return out


def fees(seed, case, kinds):
    """The rates case `case` of `seed` charges each of its `kinds` instruments apart
    from the venue's, a maker rate, a taker rate, both or neither, drawn apart from
    the stream the cases come from."""
    draw = random.Random(f"fees {seed} {case}")
    out  = []

    for _ in kinds:
        own = {}
        if draw.random() < 0.3:
            own["fee_maker"] = draw.choice([0.0, 0.001])
        if draw.random() < 0.3:
            own["fee_taker"] = draw.choice([0.0005, 0.001, 0.003])
        out.append(own)

    return dict(fees=out)


def engine_sizes(cfg):
    """The engine's arguments for a case's limit rule, whether its sizes hold, and
    its one instrument's size and price rules."""
    rules = cfg.get("rules") or [{}]
    return dict(limit=cfg.get("limit", "touch"), sizes=cfg.get("sizes", False),
                **rules[0])


def reference_rules(cfg, k):
    """Instrument `k`'s size and price rules as the reference takes them, no maximum
    order quantity an infinite one."""
    rules = dict((cfg.get("rules") or [{}] * (k + 1))[k])
    if rules.get("max_qty", 0.0) is None:
        rules["max_qty"] = math.inf

    return rules


def liquidation(seed, case, tables):
    """How case `case` of `seed` closes its account out: half the venues whose tier
    `tables` keep maintenance book then backstop, at a half or two thirds of the
    requirement, drawn apart from the stream the cases come from."""
    if not any(rows and any(tier[2] > 0.0 for tier in rows) for rows in tables):
        return "takeover"

    draw = random.Random(f"liquidation {seed} {case}")
    if draw.random() < 0.5:
        return "takeover"
    return ("book_then_backstop", draw.choice([0.5, 2.0 / 3.0]))


def liquidation_fee(seed, case):
    """The fee case `case` of `seed` closes an account with no maintenance out at, or
    none for each instrument's taker rate, drawn apart from the stream the cases come
    from."""
    draw = random.Random(f"liquidation fee {seed} {case}")
    return draw.choice([None, None, 0.0, 0.01, 0.03])


def compare(cases, seed):
    """`cases` random cases from `seed` through both: how many were compared whole
    and agreed, the disagreements, the rules each case was cut at, and the rules and
    actions the parts compared reached."""
    rng = random.Random(seed)
    # the inverse cases come after the rest, from a stream of their own, so adding
    # them left every other case as it was
    inverse = random.Random(f"inverse {seed}")

    reached = collections.Counter()
    rules   = collections.Counter()
    aside   = collections.Counter()
    whole, agreed, disagreed = 0, 0, []

    for case in range(drawn(cases)):
        draw  = rng if case < cases else inverse
        bars  = walk(draw, draw.randint(8, 40))
        times = clock(draw, len(bars))
        cfg   = config(draw, times, inverse=case >= cases)

        cfg["intrabar"]        = setting(seed, case)
        cfg["liquidation"]     = liquidation(seed, case, [cfg["tiers"]])
        cfg["liquidation_fee"] = liquidation_fee(seed, case)
        [cfg["tiers"]] = opened(seed, case, [cfg["tiers"]])
        cfg.update(fills(seed, case))
        cfg.update(sizes(seed, case, cfg["quote"], [bars[0].open], cfg.get("contract")))

        actions = plan(draw, bars, cfg)
        counted = collections.defaultdict(collections.Counter)
        try:
            rows, records, sim = run(bars, times, cfg, actions, counted)
        except Exception as exc:
            disagreed.append((case, cfg, bars, actions,
                              f"raised: {type(exc).__name__}: {exc}"))
            continue

        end   = None
        found = cut(sim)
        if found is not None:
            rule, end = found
            aside[rule] += 1

        tally(reached, rules, counted, sim, end)
        whole += end is None

        found = disagreement(rows, records, sim, cfg, end)
        if found is not None:
            disagreed.append((case, cfg, bars, actions, found[1]))
        elif end is None:
            agreed += 1

    return Outcome(whole, agreed, disagreed, aside, rules, reached)


def drawn(cases):
    """The cases a comparison of `cases` draws: those, and an inverse perpetual for
    every fourth of them."""
    return cases + cases // 4


def render(out, cases):
    lines = [f"\n{out.compared} of {drawn(cases)} compared whole, {out.agreed} agreed, "
             f"{len(out.disagreed)} disagreed",
             "compared up to a rule the engine does not have yet: "
             + (", ".join(f"{why} {n}" for why, n in
                          sorted(out.set_aside.items())) or "none"),
             "actions placed with something to act on: "
             + ", ".join(f"{k} {v}" for k, v in sorted(out.reached.items())) + "\n"]

    for case, cfg, bars, actions, why in out.disagreed[:8]:
        lines.append(f"--- case {case} ({cfg['market']}) {why}")
        lines.append(f"    cfg  {cfg}")
        lines.append(f"    bars {[tuple(round(x, 4) for x in b) for b in bars[:6]]}")
        lines.append(f"    acts {actions[:6]}\n")

    if len(out.disagreed) > 8:
        lines.append(f"... and {len(out.disagreed) - 8} more")

    return "\n".join(lines)


def main(argv=None):
    argv  = sys.argv[1:] if argv is None else argv
    cases = int(argv[0]) if len(argv) > 0 else 400
    seed  = int(argv[1]) if len(argv) > 1 else 20261005
    out   = compare(cases, seed)

    print(render(out, cases))
    return 1 if out.disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
