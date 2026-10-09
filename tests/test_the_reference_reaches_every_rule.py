"""Random accounts through the reference, at one instrument and at several.

Every run keeps the reference's invariants: the ledger identity holds on every bar,
no margin wallet ends a bar owing money, no fill is priced outside what its market
traded from the last close through its bar, and a run that ends flat with no
liquidation charge and no dust has round trips that sum to its change in equity. The
reference counts each rule as it applies it, and every rule is reached at one
instrument and at several, wherever it can arise. A short run goes with every stage
and a long one with the slow stage.
"""

import collections
import random

import pytest

from reference.simulator import Bar, Instrument, Venue, run

HOUR = 3_600 * 10**9
SHORT, LONG = 600, 6_000

RULES = [
    "a bar clipped",
    "a bracket armed",
    "a bracket cancelled with its position",
    "a bracket leg filled",
    "a check closing the account at the open",
    "a corner or the closes reaching the trigger",
    "a delisting",
    "a favourable exit held back on a liquidating bar",
    "a fill held at a tier's bound",
    "a fill held at the last tier's bound",
    "a fill held to the margin cap",
    "a flip",
    "a flip's old side on the check",
    "a fok order filling nothing",
    "a forced close of a leg that joined during the bar",
    "a growing fill refused at no equity",
    "a leg joining the check after the open",
    "a limit filled as a maker",
    "a limit filled as a taker",
    "a limit held to the prices traded since the last close",
    "a limit on the book its bar opened through",
    "a liquidation at the closes",
    "a liquidation at the maintenance trigger",
    "a liquidation before the bar opens",
    "a liquidation by the check",
    "a market order",
    "a market order filling nothing",
    "a path inside the bars that never reaches the trigger",
    "a position held through a row without its bar",
    "a position's exits reordered",
    "a post_only limit its first bar opened through",
    "a post_only order refused at the last close",
    "a reduce_only order held to the position",
    "a reduce_only order with nothing to reduce",
    "a reducing fill bounded at a fee of its own",
    "a replace",
    "a round trip closed",
    "a size rounded down to the lot",
    "a spot buy held to its cash",
    "a spot sale held to the holding",
    "a stop triggered",
    "a takeover",
    "a taker's price held inside the bar",
    "a target",
    "a target's spot shortfall moved across",
    "a tier above the first",
    "a transfer",
    "a transfer clamped",
    "a value order",
    "a wallet that fees left below zero set to zero",
    "an OCO partner cancelled",
    "an account over the cap kept from growing",
    "an inverse contract filled",
    "an ioc or fok limit expiring after one bar",
    "an order below the minimum notional refused",
    "an order held to the maximum quantity",
    "an order trading against the account's own refused",
    "an order waiting for its instrument's next bar",
    "book then backstop falling back to takeover",
    "book then backstop selling on the book",
    "dust swept at the close",
    "funding at the close of its row",
    "funding between rows",
    "funding inside a bar counted toward the check",
    "impact paid",
    "slippage paid",
    "spread paid",
    "the cap with the other legs at the fill's fraction",
    "the liquidity budget",
    "the margin bound lifting a fill inside its bar",
    "the margin bound paying what the bar cannot hold",
    "tiers by quantity",
]
# a delisting, a hole in an instrument's bars, and the cap or the corners with other
# legs need a second instrument, and an inverse contract settles alone
SEVERAL_ONLY = {
    "a delisting",
    "a position held through a row without its bar",
    "an order waiting for its instrument's next bar",
    "the cap with the other legs at the fill's fraction",
    "a path inside the bars that never reaches the trigger",
    "a corner or the closes reaching the trigger",
}
ONE_ONLY = {"an inverse contract filled"}
# with sizes on every fill is a whole number of lots, so no position the reference
# books is left between the dust threshold and half a lot
UNREACHABLE = {"dust swept at the close"}


def series(rng, rows, start, holes, flat):
    """A random walk of bars: some open with a gap, some large, a share `flat` of
    them trade at one price throughout, and with `holes` some rows have no bar."""
    price, out = start, []

    for t in range(rows):
        gap = rng.choice([0.0, 0.0, 0.03, 0.10])
        o   = price * (1.0 + rng.gauss(0.0, gap))
        if rng.random() < flat:
            h = lo = c = o
        else:
            move, wick = rng.choice([(0.01, 0.003), (0.04, 0.02)])
            c  = o * (1.0 + rng.gauss(0.0, move))
            h  = max(o, c) * (1.0 + abs(rng.gauss(0.0, wick)))
            lo = min(o, c) * (1.0 - abs(rng.gauss(0.0, wick)))

        if t > 0 and holes and rng.random() < 0.15:
            out.append(None)
            continue

        out.append(Bar(o, h, lo, c, rng.choice([0.0, 50.0, 1_000.0, 1_000.0])))
        price = c

    return out


def tiers_for(rng, scale):
    """A tier table, or none: one tier, two by notional joined by their amount, or
    two by quantity; `scale` is a typical notional."""
    pick = rng.random()
    if pick < 0.3:
        return None, "notional"

    if pick < 0.6:
        leverage = rng.choice([5.0, 10.0, 25.0, 50.0])
        rates    = [0.0, 0.005, 0.01] + ([0.02] if leverage < 50.0 else [])
        return [(1e12, leverage, rng.choice(rates), 0.0)], "notional"

    first = scale * rng.choice([0.5, 2.0])
    last  = rng.choice([1e12, 6.0 * first])
    if pick < 0.85:
        return [(first, 25.0, 0.01, 0.0), (last, 10.0, 0.02, 0.01 * first)], "notional"

    return [(rng.choice([2.0, 20.0]), 25.0, 0.01, 0.0),
            (1e12, 10.0, 0.02, 0.0)], "quantity"


def account(seed, several):
    """One random account: its clock, bars, instruments, venue, decisions and
    funding events. A rough one has a small balance, no cap and high fees on bars
    that often trade at one price, which is where the margin bound and a wallet that
    fees leave below zero are found."""
    rng   = random.Random(seed)
    rough = rng.random() < 0.2
    rows  = rng.randint(4, 14)
    times = [0]

    for _ in range(rows - 1):
        times.append(times[-1] + HOUR * rng.choice([1, 1, 1, 1, 3]))

    sizes = rng.random() < 0.3
    kinds = ["linear", "linear"] if several else [rng.choice(["linear", "linear",
                                                              "spot", "inverse"])]
    if several and rng.random() < 0.4:
        kinds.append("spot")

    instruments, columns = [], []
    for k, kind in enumerate(kinds):
        start = [100.0, 20.0, 50.0][k]
        columns.append(series(rng, rows, start, holes=k > 0,
                              flat=0.4 if rough else 0.1))

        lot, least, most = (1.0, 0.5, 200.0) if kind == "inverse" else (0.1, 20.0, 20.0)
        options = dict(tick=0.01, lot=lot if sizes else 0.0,
                       min_notional=rng.choice([0.0, least]),
                       max_qty=rng.choice([1e12, most]))
        if kind == "spot":
            instruments.append(Instrument(f"S{k}", kind="spot", **options))
            continue

        scale = 5.0 if kind == "inverse" else 500.0
        tiers, by = (None, "notional") if rough else tiers_for(rng, scale)

        if kind == "inverse":
            instruments.append(Instrument(
                f"I{k}",
                kind     = "inverse",
                contract = 10.0,
                tiers    = tiers,
                tiers_by = by,
                **options,
            ))
        else:
            delist = None
            if k == 1 and rng.random() < 0.2:
                delist = times[rng.randrange(2, rows)]
                for t, time in enumerate(times):
                    if time >= delist:
                        columns[k][t] = None

            instruments.append(Instrument(
                f"P{k}",
                tiers    = tiers,
                tiers_by = by,
                delist   = delist,
                **options,
            ))

    bars       = [[column[t] for column in columns] for t in range(rows)]
    maintained = any(any(tier[2] > 0.0 for tier in inst.tiers)
                     for inst in instruments if inst.kind != "spot")

    liquidation = "takeover"
    if maintained and rng.random() < 0.5:
        liquidation = ("book_then_backstop", rng.choice([0.5, 2.0 / 3.0]))

    inverse = kinds == ["inverse"]
    total   = (rng.choice([0.05, 0.3, 1.0, 3.0]) if inverse
               else rng.choice([5.0, 100.0, 300.0, 1_000.0]))
    if rough:
        total = 0.05 if inverse else 5.0

    wallets = "spot" in kinds and (kinds != ["spot"] or rng.random() < 0.5)
    balance = dict(spot=total / 2.0, margin=total / 2.0) if wallets else total

    funding = rng.choice(["none", "data", ("every", 1, rng.uniform(-0.01, 0.04)),
                          ("every_bars", 2, rng.uniform(-0.01, 0.04))])
    events = None
    if funding == "data":
        events = [(int(rng.uniform(0, times[-1] + HOUR)), rng.randrange(len(kinds)),
                   rng.uniform(-0.01, 0.06)) for _ in range(rng.randint(1, 5))]

    venue = Venue(balance=balance, maker=rng.choice([0.0, 0.0002]),
                  taker=0.01 if rough else rng.choice([0.0, 0.0005, 0.001, 0.01]),
                  liquidation_fee=rng.choice([None, None, 0.01, 0.03]),
                  leverage=0.0 if rough else rng.choice([0.0, 5.0, 20.0, 50.0]),
                  limit=rng.choice(["touch", "through"]), window=rng.choice([3, 24]),
                  budget=rng.choice([1.0, 1.0, 0.05]),
                  spread=rng.choice([None, ("fixed", 10.0), "estimated"]),
                  # a sale on the book needs costs below the backstop's margin
                  slippage=rng.choice([0.0, 5.0] if liquidation != "takeover"
                                      else [0.0, 5.0, 300.0]),
                  impact=rng.choice([None, ("sqrt", 0.5), ("linear", 1.0)]),
                  sizes=sizes,
                  intrabar="pessimistic" if rough else rng.choice(["pessimistic",
                                                                   "optimistic"]),
                  funding=funding, liquidation=liquidation)

    script = {t: decisions(rng, bars[t], instruments, wallets, rough)
              for t in range(rows - 1)}

    return times, bars, instruments, venue, script, events


def decisions(rng, row, instruments, wallets, rough):
    """The verbs decided at one close."""
    out = []
    for _ in range(rng.randint(2 if rough else 0, 5 if rough else 4)):
        i = rng.randrange(len(instruments))
        if row[i] is None:
            continue

        inst, mark = instruments[i], row[i].close
        side = rng.choice(["buy", "sell"])
        size = rng.choice([0.05, 0.5, 1.0, 3.0, 10.0, 30.0])
        if inst.kind == "inverse":
            size *= 10.0

        verb = rng.random()
        if verb < 0.25:
            out.append((side, dict(instrument=i, size=size,
                                   tif=rng.choice(["ioc", "ioc", "fok"]),
                                   reduce_only=rng.random() < 0.15)))
        elif verb < 0.32:
            amount = rng.choice([50.0, 500.0, 3_000.0])
            out.append((f"{side}_value", dict(instrument=i, amount=amount)))
        elif verb < 0.5:
            out.append((side, dict(instrument=i, size=size,
                                   limit=mark * (1.0 + rng.uniform(-0.05, 0.05)),
                                   tif=rng.choice(["gtc", "gtc", "ioc", "fok"]),
                                   post_only=rng.random() < 0.2,
                                   reduce_only=rng.random() < 0.15)))
        elif verb < 0.62:
            out.append((side, dict(instrument=i, size=size,
                                   stop=mark * (1.0 + rng.uniform(-0.06, 0.06)),
                                   reduce_only=rng.random() < 0.3)))
        elif verb < 0.72 and inst.kind != "spot":
            out.append(("bracket", dict(instrument=i,
                                        stop=mark * rng.uniform(0.9, 0.999),
                                        take_profit=mark * rng.uniform(1.001, 1.1))))
        elif verb < 0.77:
            out.append(("replace", dict(order_id=rng.randint(1, 40),
                                        price=mark * rng.uniform(0.95, 1.05))))
        elif verb < 0.8:
            out.append(("cancel", dict(order_id=rng.randint(1, 40))))
        elif verb < 0.86:
            out.append(("close", dict(instrument=i)))
        elif verb < 0.95:
            weights = {k: (rng.uniform(0.0, 1.0) if instruments[k].kind == "spot"
                           else rng.uniform(-10.0, 10.0))
                       for k in range(len(instruments)) if row[k] is not None}
            out.append(("target", dict(weights=weights,
                                       transfer=wallets and rng.random() < 0.5)))
        elif wallets:
            out.append(("transfer", dict(amount=rng.choice([20.0, 200.0, 5_000.0]),
                                         to=rng.choice(["spot", "margin"]))))

    if rough:
        # a take-profit close to the market is the exit a liquidating bar holds back
        for i, inst in enumerate(instruments):
            if inst.kind != "spot" and row[i] is not None and rng.random() < 0.5:
                mark = row[i].close
                out.append(("bracket", dict(instrument=i,
                                            stop=mark * rng.uniform(0.95, 0.99),
                                            take_profit=mark * rng.uniform(1.001,
                                                                           1.02))))

    return out


def broken(sim):
    """Which invariant the run breaks, or None."""
    for t in range(1, len(sim.ledger)):
        before, after = sim.ledger[t - 1], sim.ledger[t]
        pnl = 0.0
        for i, inst in enumerate(sim.instruments):
            q, m0, m1 = (before["instruments"][i]["position"],
                         before["instruments"][i]["mark"],
                         after["instruments"][i]["mark"])
            pnl += inst.profit(q, m0, m1)
            for fill in sim.fills:
                if fill["bar"] == t and fill["instrument"] == i:
                    signed = fill["size"] if fill["side"] == "buy" else -fill["size"]
                    pnl += inst.profit(signed, fill["reference"], m1)

        costs = sum(after[name] for name in ("fees", "spread", "slippage", "impact",
                                             "funding", "liquidation", "dust"))
        scale = 1.0 + abs(before["equity"]) + abs(pnl) + abs(costs)
        if abs(after["equity"] - before["equity"] - (pnl - costs)) > 1e-9 * scale:
            return f"the ledger identity on bar {t}"

        spot = after["spot_cash"] + sum(
            row["notional"] for row, inst in zip(after["instruments"], sim.instruments)
            if inst.kind == "spot")
        if after["equity"] - spot < -1e-9 * scale:
            return f"the margin wallet owing money after bar {t}"

    for fill in sim.fills:
        t, i = fill["bar"], fill["instrument"]
        bar = sim.bars[t][i]
        if bar is None:
            continue

        close = sim.ledger[t - 1]["instruments"][i]["mark"]
        low, high = min(close, bar.low), max(close, bar.high)
        if not low - 1e-9 * high <= fill["price"] <= high * (1.0 + 1e-9):
            return f"a fill outside what its market traded, on bar {t}"

    last, first = sim.ledger[-1], sim.ledger[0]
    flat    = all(row["position"] == 0.0 for row in last["instruments"])
    charged = any(row["liquidation"] != 0.0 or row["dust"] != 0.0 for row in sim.ledger)
    if flat and not charged:
        total  = sum(trip["pnl"] for trip in sim.round_trips())
        change = last["equity"] - first["equity"]
        if abs(total - change) > 1e-9 * (1.0 + abs(first["equity"])):
            return "the round trips against the change in equity"

    return None


def accounts(count, several):
    """`count` seeded accounts through the reference: the rules they reached and the
    runs that broke an invariant."""
    reached, failures = collections.Counter(), []
    for seed in range(count):
        times, bars, instruments, venue, script, events = account(seed, several)
        sim = run(times, bars, instruments, venue, script=script, funding=events)
        reached.update(sim.reached)
        why = broken(sim)
        if why is not None:
            failures.append((seed, why))

    return reached, failures


@pytest.fixture(scope="module")
def short():
    return {several: accounts(SHORT, several) for several in (False, True)}


def test_random_accounts_keep_the_reference_invariants(short):
    assert [failures for _, failures in short.values()] == [[], []]

    named = set().union(*(reached for reached, _ in short.values()))

    assert named <= set(RULES), sorted(named - set(RULES))


@pytest.mark.parametrize("rule", [rule for rule in RULES if rule not in UNREACHABLE])
def test_random_accounts_reach_every_rule(rule, short):
    counts = {several: short[several][0][rule] for several in (False, True)}

    if rule not in SEVERAL_ONLY:
        assert counts[False] > 0, f"{rule}: {counts}"
    if rule not in ONE_ONLY:
        assert counts[True] > 0, f"{rule}: {counts}"


@pytest.mark.slow
@pytest.mark.parametrize("several", [False, True])
def test_many_random_accounts_keep_the_reference_invariants(several):
    _, failures = accounts(LONG, several)

    assert failures == []
