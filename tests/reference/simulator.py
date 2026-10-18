"""probatum's reference simulator, written from the Specification page.

It exists to disagree with the engine, so it is written from the rules as the
Specification states them and never from the Rust. Each step names the section it
implements, so a disagreement can be argued from the page rather than guessed at,
and every price, cash movement and log row is computed plainly, in the order the
page lists them. It is slow on purpose, and where the engine solves the
liquidation path in closed form, this finds its crossings by bisection.

It takes what the low-level binding takes, the bars on a clock, the instruments,
the venue and the decisions made at each close, and it returns what the binding
returns: the ledger, the fill log, the order log and the round trips. It also counts
each rule as it applies it, so a test of random cases can show what they reached.

Scope: the clock, the step's order of events, market, value, limit and stop
orders with their time in force, `post_only`, `reduce_only`, brackets, `replace`
and cancels, `target`, the slot table, maker after the first bar, self-trade
refusal, the worse exit first, the fill model (the trailing window, the liquidity
budget, the spread, slippage, impact, sizes and dust), fees, funding, the spot and
margin wallets with transfers, `target` across them, the margin cap with leverage
tiers, maintenance tiers, the liquidation path of several instruments in step with
its trigger, close and clip points, a flip's two sides on the check, the optimistic
corners, the liquidation rule, takeover and book then backstop, the margin bound,
the cap and the bound with every other leg at the fill's fraction, exits held on a
liquidating bar, inverse contracts settled in their coin, listing and delisting,
round trips and the ledger, and an array's rows in either unit with their barriers.
"""

import collections
import copy
import math
import statistics

DUST     = 1e-9      # Sizes and dust: with sizes off, a fill this small moves nothing
MAX_COST = 0.99      # Fills, A taker's price: s / 2 + slip + imp at most
SNAP     = 10**9     # Funding: a stamp within a second of a row boundary is on it

Bar  = collections.namedtuple("Bar", "open high low close volume")
Tier = collections.namedtuple("Tier", "up_to leverage rate amount")


def lots(size, lot):
    """Sizes and dust: the whole lots in `size`, a size floating point left a hair
    under a whole number of lots kept at that number."""
    return math.floor(size / lot + 1e-9)


def power(k):
    """Sizes and dust: `10^k`, exact for `k >= 0` and `1 / 10^-k` below zero."""
    return float(10 ** k) if k >= 0 else 1.0 / float(10 ** -k)
# a leg of the liquidation path: its price runs from start to end as s runs 0 to 1
Leg = collections.namedtuple("Leg", "instrument q entry start end")
# a leg of the check: it joins the path at frac, at start, and runs on to end; one a
# flip closed leaves it at leave, and is held to the end where leave is None
CheckLeg = collections.namedtuple("CheckLeg", "instrument q entry frac start end leave")
# what a position held during a bar is made of, for the check: start is None for a
# leg carried in from the open, and leave is set on the old side of a flip
Held = collections.namedtuple("Held", "size entry start join leave long")


def first_zero(f, lo=0.0, hi=1.0):
    """The first point in [lo, hi] where f falls to zero or below, by bisection, for
    an f above zero at lo and at or below it at hi that crosses zero once there."""
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid

    return hi


def allows(size, asked):
    """The order log's account of a clamp: `size` of the `asked`, or none of it."""
    if size > 0.0:
        return f"allows {size} of the {asked} asked"
    return f"allows none of the {asked} asked"


def booked(q, signed, size, held):
    """`size`, less the ulps by which booking it on a position of `q` would round
    the position past `held`, into a tier that asks more than the cap allowed;
    `signed` carries the fill's side."""
    while abs(q + math.copysign(size, signed)) > held:
        size = math.nextafter(size, 0.0)

    return size


def last_within(f, lo, hi):
    """The last point in [lo, hi] where f stays at or above zero, by bisection, for
    an f at or above zero at lo and below it at hi that changes sign once there,
    a jump included."""
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) >= 0.0:
            lo = mid
        else:
            hi = mid

    return lo


class Instrument:

    def __init__(self, name, kind="linear", contract=None, lot=0.0, tick=0.0,
                 sigfigs=None, decimals=None, min_notional=0.0, max_qty=math.inf,
                 reduce_only_exempt=False, tiers=None, tiers_by="notional",
                 delist=None, maker=None, taker=None):
        if kind not in ("spot", "linear", "inverse"):
            raise ValueError(f"kind: {kind!r} is not modelled; use spot, linear or "
                             f"inverse")

        if (kind == "inverse") != (contract is not None):
            raise ValueError(f"contract: {contract!r}; an inverse instrument takes "
                             f"the dollars a contract is worth, and no other does")

        if kind == "inverse" and not float(contract) > 0.0:
            raise ValueError(f"contract: {contract!r} is not a positive value")

        # Sizes and dust: the significant-figure rule is Hyperliquid's, whose
        # perpetuals are linear, and it comes with its cap on decimals
        if lot and max_qty < lot:
            raise ValueError(f"max_qty: {max_qty}; give a maximum of at least the lot "
                             f"of {lot}")

        if sigfigs is not None and (kind == "inverse" or tick or decimals is None):
            raise ValueError(f"sigfigs: {sigfigs!r} beside kind {kind!r}, tick {tick!r} "
                             f"and decimals {decimals!r}; give a linear or spot "
                             f"instrument its figures and their cap, and no tick")

        if tiers_by not in ("notional", "quantity"):
            raise ValueError(f"tiers_by: {tiers_by!r}; use notional or quantity")

        # Accounting, Fees: an instrument's own rates, the venue's where it has none
        for side, rate in (("maker", maker), ("taker", taker)):
            if rate is not None and not (math.isfinite(rate) and rate > -1.0):
                raise ValueError(f"{side}: {rate!r}; give a rate above -1")

        self.maker              = None if maker is None else float(maker)
        self.taker              = None if taker is None else float(taker)
        self.name               = name
        self.kind               = kind
        self.contract           = None if contract is None else float(contract)
        self.lot                = float(lot)
        self.tick               = float(tick)
        self.sigfigs            = sigfigs   # Hyperliquid's rule: significant figures,
        self.decimals           = decimals  # within a cap on decimal places
        self.min_notional       = float(min_notional)
        self.max_qty            = float(max_qty)
        self.reduce_only_exempt = reduce_only_exempt
        self.tiers              = [Tier(*map(float, tier)) for tier in (tiers or [])]
        self.tiers_by           = tiers_by
        self.delist             = delist    # nanoseconds, or None
        self.check_tiers()


    def check_tiers(self):
        """Margin, Maintenance: the loader refuses a notional table whose amounts break
        continuity or whose first amount is not zero, any rate not below one, a
        maintenance rate that reaches the tier's initial rate, and a table whose bounds
        do not rise, whose leverage rises or whose maintenance rate falls from one tier
        to the next, a leverage of 0.0 standing for no cap, and any value that is not
        finite, but a last bound left open, or a leverage or rate below zero. A notional
        table keeps each amount at the value continuity sets, once the one given is
        within rounding of it."""
        def cap(leverage):
            return leverage if leverage > 0.0 else math.inf

        previous, kept = None, []
        for k, tier in enumerate(self.tiers):
            # the last bound may be open, where the venue sets no largest position
            unbounded = k == len(self.tiers) - 1 and tier.up_to == math.inf
            if not all(math.isfinite(value) for value in tier[1:]) or not (
                    math.isfinite(tier.up_to) or unbounded):
                raise ValueError(f"tiers: {tier} holds a value that is not finite")
            if tier.leverage < 0.0 or tier.rate < 0.0:
                raise ValueError(f"tiers: {tier} holds a leverage or rate below zero")

            below = 0.0 if previous is None else previous.up_to
            if tier.up_to <= below:
                raise ValueError(f"tiers: a bound of {tier.up_to} not above {below}")
            if previous is not None and cap(tier.leverage) > cap(previous.leverage):
                raise ValueError(f"tiers: a leverage of {tier.leverage} above the "
                                 f"{previous.leverage} before it")
            if previous is not None and tier.rate < previous.rate:
                raise ValueError(f"tiers: a maintenance rate of {tier.rate} below the "
                                 f"{previous.rate} before it")
            if tier.rate >= 1.0:
                raise ValueError(f"tiers: a rate of {tier.rate} is not below one")
            if tier.leverage > 0.0 and tier.rate >= 1.0 / tier.leverage:
                raise ValueError(f"tiers: a maintenance rate of {tier.rate} reaches "
                                 f"the initial rate of {1.0 / tier.leverage}")

            if self.tiers_by == "notional":
                wanted = 0.0 if previous is None else (
                    previous.amount + (tier.rate - previous.rate) * previous.up_to)
                if abs(tier.amount - wanted) > 1e-9 * max(1.0, abs(wanted)):
                    raise ValueError(f"tiers: an amount of {tier.amount} where "
                                     f"continuity needs {wanted}")
                tier = tier._replace(amount=wanted)

            kept.append(tier)
            previous = tier

        self.tiers = kept


    # Inverse contracts: every quantity the account counts, for either kind, in the
    # currency the account is kept in

    def notional(self, q, p):
        """A position's notional, signed: `q * p`, or `q * V / p` for an inverse
        contract."""
        return q * self.contract / p if self.kind == "inverse" else q * p


    def profit(self, q, e, p):
        """The profit from `e` to `p`: `q * (p - e)`, or `u * (1 / e - 1 / p)`."""
        if q == 0.0:
            return 0.0

        if self.kind == "inverse":
            return q * self.contract * (1.0 / e - 1.0 / p)
        return q * (p - e)


    def average(self, size, entry, added, p):
        """The average entry once `added` joins `size` at `entry`: weighted by size, or
        the harmonic mean weighted by contracts."""
        if size == 0.0:
            return p

        if self.kind == "inverse":
            return (size + added) / (size / entry + added / p)
        return (entry * size + p * added) / (size + added)


    def units(self, amount, p):
        """The size a value order of `amount` fills at `p`."""
        return amount / self.contract if self.kind == "inverse" else amount / p


    # Sizes and dust: the price rule

    def figures_at(self, p):
        """The decimals the significant-figure rule keeps at `p`, a whole number
        always allowed."""
        e = 0

        while e < 22 and abs(p) >= power(e + 1):
            e += 1
        while e > -22 and abs(p) < power(e):
            e -= 1

        return min(max(self.sigfigs - 1 - e, 0), self.decimals)


    def on_grid(self, p):
        """`p` on the price rule's grid: the nearest tick, or the significant figures
        kept within the cap, ties to even."""
        if self.sigfigs is not None:
            d = self.figures_at(p)
            return round(p * power(d)) / power(d)
        if self.tick > 0.0:
            return round(p / self.tick) * self.tick
        return p


    def step(self, p):
        """The price rule's step at `p`, its tick or `10^-d`, and none without a
        rule."""
        if self.sigfigs is not None:
            return power(-self.figures_at(p))
        return self.tick


class Venue:

    def __init__(self, balance=10_000.0, maker=0.0002, taker=0.0006, leverage=10.0,
                 max_open_orders=8, limit="touch", window=24, budget=1.0, spread=None,
                 slippage=0.0, impact=None, sizes=False, intrabar="pessimistic",
                 funding="none", liquidation="takeover", liquidation_fee=None):
        if limit not in ("touch", "through"):
            raise ValueError(f"limit: {limit!r}; use touch or through")

        if intrabar not in ("pessimistic", "optimistic"):
            raise ValueError(f"intrabar: {intrabar!r}; use pessimistic or optimistic")

        if int(window) < 1:
            raise ValueError(f"window: {window!r}; give the number of bars before a "
                             "fill its window reads, from 1")

        if not float(budget) > 0.0:
            raise ValueError(f"budget: {budget!r}; give the share of the trailing "
                             "volume a bar's fills may take, above 0")

        self.balance         = balance
        self.maker           = float(maker)
        self.taker           = float(taker)
        self.leverage        = float(leverage)
        self.max_open_orders = int(max_open_orders)
        self.limit           = limit
        self.window          = int(window)
        self.budget          = float(budget)
        self.spread          = spread   # None, ("fixed", bps) or "estimated"
        self.slippage        = float(slippage)
        self.impact          = impact   # None, ("linear", k) or ("sqrt", k)
        self.sizes           = sizes
        self.intrabar        = intrabar
        self.funding         = funding  # "none", "data", ("every", hours, rate) or

                                        # ("every_bars", n, rate)
        if liquidation != "takeover" and not (
                isinstance(liquidation, tuple) and len(liquidation) == 2
                and liquidation[0] == "book_then_backstop"
                and 0.0 < liquidation[1] < 1.0):
            raise ValueError(f"liquidation: {liquidation!r}; use takeover or "
                             "('book_then_backstop', share) with a share above 0 and "
                             "below 1")

        self.liquidation = liquidation    # "takeover" or ("book_then_backstop", share)
        # None charges each instrument its own taker rate
        self.liquidation_fee = (None if liquidation_fee is None
                                else float(liquidation_fee))


def periods_per_year(times):
    """Time, Annualisation: the clock's rows less one per 365 days of its span."""
    days = (times[-1] - times[0]) / 86_400e9
    return (len(times) - 1) * 365.0 / days


def total(values):
    """`values` summed in order, one floating-point addition at a time, where the
    built-in sum compensates its additions."""
    out = 0.0

    for value in values:
        out += value

    return out


def mean(values):
    return total(values) / len(values)


def deviation(values):
    """The sample standard deviation of `values`, its mean and its squares summed in
    order."""
    centre = mean(values)
    return math.sqrt(total((v - centre) * (v - centre) for v in values)
                     / (len(values) - 1))


def edge(bars):
    """Fills, The spread estimate: Ardia, Guidotti and Kroencke's estimator over the
    bars' consecutive pairs, floored at zero, and zero where it is undefined."""
    if len(bars) < 3:
        return 0.0

    o     = [math.log(bar.open) for bar in bars]
    h     = [math.log(bar.high) for bar in bars]
    lo    = [math.log(bar.low) for bar in bars]
    c     = [math.log(bar.close) for bar in bars]
    m     = [(a + b) / 2.0 for a, b in zip(h, lo)]
    pairs = range(1, len(bars))
    r1    = [m[t] - o[t] for t in pairs]
    r2    = [o[t] - m[t - 1] for t in pairs]
    r3    = [m[t] - c[t - 1] for t in pairs]
    r4    = [c[t - 1] - m[t - 1] for t in pairs]
    r5    = [o[t] - c[t - 1] for t in pairs]
    tau   = [1.0 if h[t] != lo[t] or lo[t] != c[t - 1] else 0.0 for t in pairs]
    po    = (mean([tau[k] * (o[t] != h[t]) for k, t in enumerate(pairs)])
             + mean([tau[k] * (o[t] != lo[t]) for k, t in enumerate(pairs)]))

    pc = (mean([tau[k] * (c[t - 1] != h[t - 1]) for k, t in enumerate(pairs)])
          + mean([tau[k] * (c[t - 1] != lo[t - 1]) for k, t in enumerate(pairs)]))
    pt = mean(tau)
    if total(tau) < 2 or po == 0.0 or pc == 0.0:
        return 0.0

    d1 = [r - mean(r1) / pt * w for r, w in zip(r1, tau)]
    d3 = [r - mean(r3) / pt * w for r, w in zip(r3, tau)]
    d5 = [r - mean(r5) / pt * w for r, w in zip(r5, tau)]
    x1 = [-4.0 / po * a * b - 4.0 / pc * d * e for a, b, d, e in zip(d1, r2, d3, r4)]
    x2 = [-4.0 / po * a * b - 4.0 / pc * d * e for a, b, d, e in zip(d1, r5, d5, r4)]
    e1, e2 = mean(x1), mean(x2)
    v1 = mean([x * x for x in x1]) - e1 * e1
    v2 = mean([x * x for x in x2]) - e2 * e2
    s2 = (v2 * e1 + v1 * e2) / (v1 + v2) if v1 + v2 > 0.0 else (e1 + e2) / 2.0
    return math.sqrt(max(s2, 0.0))


class Trip:
    """One round trip, from the fill that takes a position off zero to the fill that
    returns it to zero or flips it (Accounting, Round trips)."""


    def __init__(self, instrument, inst, side, bar, time):
        self.instrument = instrument
        self.inst       = inst
        self.side       = side
        self.open_bar   = bar
        self.open_time  = time

        self.close_bar  = None
        self.close_time = None
        self.size       = 0.0
        self.opened     = 0.0
        self.entry      = 0.0
        self.closed     = 0.0
        self.exit       = 0.0
        self.realized   = 0.0
        self.fees       = 0.0
        self.funding    = 0.0
        self.reason     = None
        self.high = self.low = None


    def see(self, *prices):
        """Round trips: a price among those the round trip lived through."""
        for p in prices:
            self.high = p if self.high is None else max(self.high, p)
            self.low  = p if self.low is None else min(self.low, p)


    def open(self, size, price):
        self.entry = self.inst.average(self.opened, self.entry, size, price)
        self.opened += size
        self.see(price)


    def close(self, size, price):
        self.exit = self.inst.average(self.closed, self.exit, size, price)
        self.closed += size
        self.see(price)


    def report(self, mark=None, entry=None, held=0.0):
        """The round trip's row. An open one is marked at the last close."""
        unrealized = 0.0
        if self.close_bar is None and mark is not None:
            unrealized = self.inst.profit(held, entry, mark)

        pnl = self.realized + unrealized - self.fees - self.funding

        # its best and worst excursions, over the part of each bar it was in, as a
        # fraction of its entry
        up, down = self.high / self.entry - 1.0, self.low / self.entry - 1.0
        best, worst = (up, down) if self.side == "long" else (-down, -up)

        return dict(
            instrument=self.instrument, side=self.side, size=self.size,
            entry=self.entry, exit=self.exit if self.closed > 0.0 else math.nan,
            open_bar=self.open_bar, close_bar=self.close_bar,
            open_time=self.open_time, close_time=self.close_time,
            pnl=pnl, ret=pnl / abs(self.inst.notional(self.size, self.entry)),
            fees=self.fees, funding=self.funding, reason=self.reason,
            best=best, worst=worst, open=self.close_bar is None,
        )


# the account's state a bar can be resolved twice from (The rule, 4)
MUTABLE = ("position", "entry", "spot_cash", "margin_cash", "trip", "trips",
           "pending", "slots", "orders", "fills", "events", "costs", "budget_left",
           "legs", "next_id", "target_legs", "liquidations", "reached", "first",
           "start", "inside", "entered", "reach", "clipped", "charged", "made",
           "closer")


class Simulator:

    def __init__(self, times, bars, instruments, venue=None, interval=None,
                 funding=None, start=0):
        self.times       = [int(time) for time in times]
        self.bars        = bars                   # bars[t][i]: a Bar, or None
        self.instruments = list(instruments)
        self.venue       = venue or Venue()
        gaps             = [b - a for a, b in zip(self.times, self.times[1:])]
        self.interval    = int(interval or (statistics.median(gaps) if gaps else 0))
        n                = len(self.instruments)
        self.schedule_funding(funding)

        if self.venue.liquidation != "takeover" and not any(
                self.has_maintenance(i) for i in self.perps()):
            raise ValueError("liquidation: book_then_backstop with no maintenance in any "
                             "perpetual's tier table, where its backstop level is the "
                             "trigger itself; give one a maintenance rate")

        # Wallets: a number funds the one wallet a universe of one kind uses, and
        # a universe holding both kinds names each
        kinds = {inst.kind for inst in self.instruments}
        if "inverse" in kinds and n > 1:
            raise ValueError("instruments: an inverse account holds its one perpetual "
                             "alone, since it settles in that perpetual's coin")

        balance = self.venue.balance
        if isinstance(balance, dict):
            self.spot_cash   = float(balance.get("spot", 0.0))
            self.margin_cash = float(balance.get("margin", 0.0))
        elif kinds == {"spot"}:
            self.spot_cash, self.margin_cash = float(balance), 0.0
        elif "spot" not in kinds:
            self.spot_cash, self.margin_cash = 0.0, float(balance)
        else:
            raise ValueError("balance: one number for a universe with both wallets; "
                             "pass {'spot': ..., 'margin': ...}")

        self.t        = 0
        self.position = [0.0] * n
        self.entry    = [0.0] * n

        # where each position started on this bar's path (Liquidation, The rule, 4),
        # and whether a fill inside the bar last opened, flipped or grew it (What a bar
        # cannot show); and the fill that opened or flipped it, which the round trip it
        # began counts its entry bar from (Round trips)
        self.start       = [None] * n
        self.inside      = [False] * n
        self.entered     = [(None, False)] * n
        self.mark        = [None] * n
        self.history     = [[] for _ in range(n)]  # each instrument's own bars
        self.listed      = [False] * n
        self.delisted    = [False] * n
        self.trip        = [None] * n
        self.trips       = []
        self.pending     = []                      # market orders, filling at each one's next bar
        self.slots       = []                      # Orders, Maker, taker and the book: one table
        self.transfers   = []
        self.orders      = {}
        self.next_id     = 1
        self.fills       = []
        self.events      = []
        self.decided     = -1                      # the row whose close the last `target` came from
        self.target_legs = []

        # Orders, Target: the weight the latest target asked of each instrument, or
        # the position an array's quantity wanted, and what they were before the
        # decision's first target, which a second replaces
        self.asked           = [None] * n
        self.wanted          = [None] * n
        self.asked_before    = list(self.asked)
        self.wanted_before   = list(self.wanted)
        self.target_transfer = None    # where this decision's shortfall was queued
        self.paid_in         = 0.0     # what the last bound paid into the wallet
        self.reach           = 0.0     # what the bar's perpetual fills put at stake
        self.budget_left     = {}
        self.costs           = self.no_costs()

        # What a run records: each instrument's share of the bar's costs, what it has
        # made since the first row, and the instrument whose fill or sweep last left
        # no perpetual held, which a deficit set to zero is given to
        self.charged      = [0.0] * n
        self.made         = [0.0] * n
        self.closer       = None
        self.fills_before = 0
        self.clipped      = {}
        self.legs         = {}
        self.liquidations = 0
        self.reached      = collections.Counter()  # how often each rule applied
        self.first        = {}                     # and the bar it first applied on

        # an account opened at row `start` of the series, as an env reset there is,
        # knowing every row up to it
        self.t = start
        for t in range(start + 1):
            self.see_row(t)

        self.ledger = [self.row(start)]


    @staticmethod
    def no_costs():
        return dict(fees=0.0, spread=0.0, slippage=0.0, impact=0.0, funding=0.0,
                    liquidation=0.0, dust=0.0)


    def hit(self, rule, bar=None):
        """Count `rule`, first applied on `bar`, the current one unless a decision
        names the next, where what it decided acts."""
        self.reached[rule] += 1
        self.first.setdefault(rule, self.t if bar is None else bar)


    def see_row(self, t):
        for i, bar in enumerate(self.bars[t]):
            if bar is not None:
                self.listed[i] = True
                self.mark[i]   = bar.close
                self.history[i].append(bar)


    def perps(self):
        return [i for i, inst in enumerate(self.instruments) if inst.kind != "spot"]


    def wallet_equity(self):
        spot   = self.spot_cash
        margin = self.margin_cash

        for i, inst in enumerate(self.instruments):
            if self.position[i] == 0.0:
                continue
            if inst.kind == "spot":
                spot += inst.notional(self.position[i], self.mark[i])
            else:
                margin += inst.profit(self.position[i], self.entry[i], self.mark[i])

        return spot, margin


    def equity(self):
        spot, margin = self.wallet_equity()
        return spot + margin


    def taker(self, i):
        """Accounting, Fees: instrument `i`'s taker rate, its own or the venue's."""
        own = self.instruments[i].taker
        return self.venue.taker if own is None else own


    def maker(self, i):
        """Accounting, Fees: instrument `i`'s maker rate, its own or the venue's."""
        own = self.instruments[i].maker
        return self.venue.maker if own is None else own


    def closing_fee(self, i, maintained):
        """Liquidation, The trigger, close and clip points: the fee a close of
        instrument `i` pays, its taker rate with maintenance and its liquidation fee
        without, the taker rate unless the venue sets one."""
        fee = self.venue.liquidation_fee
        return self.taker(i) if maintained or fee is None else fee


    def has_maintenance(self, i):
        return any(tier.rate > 0.0 for tier in self.instruments[i].tiers)


    def tier_of(self, i, size, notional):
        """Margin, Maintenance: the tier a position's notional, or its size in
        contracts on a quantity table, falls in."""
        inst = self.instruments[i]
        if not inst.tiers:
            return None

        measure = notional if inst.tiers_by == "notional" else size
        for tier in inst.tiers:
            if measure <= tier.up_to:
                return tier
        return inst.tiers[-1]


    def leverage_of(self, i, size, notional):
        tier = self.tier_of(i, size, notional)
        return self.venue.leverage if tier is None else tier.leverage


    def maintenance_of(self, i, size, notional):
        """Margin, Maintenance: `N * m - c` on a notional table, `N * m` by quantity."""
        tier = self.tier_of(i, size, notional)
        if tier is None or tier.rate <= 0.0 or notional <= 0.0:
            return 0.0

        if self.instruments[i].tiers_by == "notional":
            return notional * tier.rate - tier.amount
        return notional * tier.rate


    def cap_requirement(self, prices=None, sizes=None):
        """Margin, The cap: each perpetual's notional over its tier's leverage."""
        total = 0.0

        for i in self.perps():
            size = abs(self.position[i]) if sizes is None else sizes.get(i, 0.0)
            if size == 0.0:
                continue

            price    = self.mark[i] if prices is None else prices.get(i, self.mark[i])
            notional = abs(self.instruments[i].notional(size, price))
            leverage = self.leverage_of(i, size, notional)
            if leverage > 0.0:
                total += notional / leverage

        return total


    def margin_unrealized(self, prices=None):
        total = 0.0

        for i in self.perps():
            if self.position[i] != 0.0:
                price = self.mark[i] if prices is None else prices.get(i, self.mark[i])
                total += self.instruments[i].profit(self.position[i], self.entry[i],
                                                    price)

        return total


    def margin_equity(self, prices=None):
        return self.margin_cash + self.margin_unrealized(prices)


    def account_maintained(self):
        return any(self.has_maintenance(i) for i in self.perps()
                   if self.position[i] != 0.0)


    def maintained_after(self, i, size):
        """Liquidation, The margin bound: whether the account keeps maintenance once a
        reducing fill of `size` on perpetual `i` lands, `i` gone where the fill closes
        all of it."""
        left = size != abs(self.position[i])
        return any(self.has_maintenance(j) for j in self.perps()
                   if self.position[j] != 0.0 and (j != i or left))


    def resting(self, instrument=None):
        return [o for o in self.slots if o is not None
                and (instrument is None or o["instrument"] == instrument)]


    def bar_for(self, t, i):
        """The rule, 3: a held perpetual's bar, clipped at the clip point."""
        return self.clipped.get(i, self.bars[t][i])


    def fraction(self, i, long, price, t):
        """The path: where `price` lies on instrument `i`'s adverse segment of bar
        `t`, from its open to its low for a long and to its high for a short; a price
        on the favourable side of the open, or a bar with no adverse move, is at 0,
        and so is a price past the adverse extreme, which traded on the gap before
        the bar opened."""
        bar = self.bars[t][i]
        if bar is None:
            return 0.0

        adverse = bar.low if long else bar.high
        if self.instruments[i].kind == "inverse":
            move = 1.0 / adverse - 1.0 / bar.open
            s    = (1.0 / price - 1.0 / bar.open) / move if move != 0.0 else 0.0
        else:
            move = adverse - bar.open
            s    = (price - bar.open) / move if move != 0.0 else 0.0

        return 0.0 if s > 1.0 else max(s, 0.0)


    def along(self, i, a, b, s):
        """The path: instrument `i`'s price a fraction `s` of the way from `a` to `b`,
        the way running in the reciprocal of the price on an inverse contract, which
        keeps its equity linear in `s` as a linear contract's is (Inverse
        contracts)."""
        if self.instruments[i].kind == "inverse":
            return 1.0 / (1.0 / a + s * (1.0 / b - 1.0 / a))
        return a + s * (b - a)


    def others_at(self, s, t, skip):
        """Margin, The cap, and The margin bound: every other perpetual held, at the
        same fraction of its own bar's move, or at its favourable extreme under the
        optimistic setting; one with no bar keeps its last mark."""
        prices = {}
        for j in self.perps():
            q = self.position[j]
            if j == skip or q == 0.0:
                continue

            bar = self.bars[t][j]
            if bar is None:
                prices[j] = self.mark[j]
            elif self.venue.intrabar == "optimistic":
                prices[j] = bar.high if q > 0.0 else bar.low
            else:
                adverse = bar.low if q > 0.0 else bar.high
                prices[j] = self.along(j, bar.open, adverse, s)

        return prices


    def window(self, i):
        """Fills, The window: the instrument's last W bars before the fill's bar, their
        mean volume, and the sample deviation of their log returns close to close."""
        bars = self.history[i][-self.venue.window:]
        if len(self.history[i]) > self.venue.window:
            self.hit("a window cut to its last bars")

        closes  = [bar.close for bar in self.history[i][-(self.venue.window + 1):]]
        returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
        volume  = mean([bar.volume for bar in bars]) if bars else 0.0
        sigma   = deviation(returns) if len(returns) >= 2 else 0.0
        return bars, volume, sigma


    def row(self, t, costs=None):
        costs = costs or self.no_costs()

        spot, margin = self.wallet_equity()
        equity = spot + margin

        # Inverse contracts: an inverse account's equity is in its coin, and valued in
        # dollars at the close beside it
        inverse = self.instruments[0].kind == "inverse"
        out     = dict(time=self.times[t], equity=equity, spot_cash=self.spot_cash,
                       margin_cash=self.margin_cash, terminal=equity <= 0.0,
                       equity_in_quote=equity * self.mark[0] if inverse else equity,
                       liquidations=self.liquidations, margin_used=0.0, maintenance=0.0,
                       gross_exposure=0.0, net_exposure=0.0, instruments=[], **costs)

        for i, inst in enumerate(self.instruments):
            q, mark = self.position[i], self.mark[i]
            held     = q != 0.0
            exposure = inst.notional(q, mark) if held else 0.0

            # What a run records: the account's exposure at the closes, unsigned and
            # signed, over every position
            out["gross_exposure"] += abs(exposure)
            out["net_exposure"] += exposure

            if held and inst.tiers:
                tier = self.tier_of(i, abs(q), abs(exposure))
                if tier is not inst.tiers[0]:
                    self.hit("a tier above the first")
                if inst.tiers_by == "quantity":
                    self.hit("tiers by quantity")

            # What a run records: the margin the cap asks and the maintenance are the
            # account's, summed over its perpetuals
            if held and inst.kind != "spot":
                leverage = self.leverage_of(i, abs(q), abs(exposure))
                if leverage > 0.0:
                    out["margin_used"] += abs(exposure) / leverage
                out["maintenance"] += self.maintenance_of(i, abs(q), abs(exposure))

            weight = exposure / equity if held and equity > 0.0 else 0.0
            # Orders, Target: the weight the latest target asked less the one held;
            # An array: a wanted position asks its notional at this close over equity
            asked, wanted = self.asked[i], self.wanted[i]
            if wanted is not None:
                asked = (math.nan if mark is None else
                         inst.notional(wanted, mark) / equity if equity > 0.0 else 0.0)

            out["instruments"].append(dict(
                position=q, entry=self.entry[i], mark=mark, notional=abs(exposure),
                upnl=inst.profit(q, self.entry[i], mark) if held else 0.0,
                weight=weight, drift=math.nan if asked is None else asked - weight,
                pnl=self.made[i],
            ))

        # and its leverage, gross exposure over equity, nothing once equity is gone
        out["leverage"] = out["gross_exposure"] / equity if equity > 0.0 else 0.0
        return out


    def log(self, kind, order, reason, size=None, price=None, side=None):
        self.events.append(dict(
            time=self.times[self.t], bar=self.t, kind=kind,
            order=None if order is None else order["id"],
            instrument=None if order is None else order["instrument"],
            side=side, size=size, price=price, reason=reason,
        ))


    def log_order(self, kind, order, reason):
        """The fill log and the order log: a placed or armed event records the order as
        it is queued or rests, its side, its whole size and a limit's price or a
        stop's trigger."""
        price = order["price"] if order["kind"] == "limit" else order["trigger"]
        self.log(kind, order, reason, size=order["size"], price=price,
                 side=order["side"])


    def lot_down(self, i, size):
        """Fills, Sizes and dust: with sizes on, down to a whole number of lots."""
        lot = self.instruments[i].lot
        if not self.venue.sizes or lot <= 0.0:
            return size

        rounded = lots(size, lot) * lot
        if rounded < size - DUST:
            self.hit("a size rounded down to the lot")
        return rounded


    def round_price(self, i, price):
        """Fills, Sizes and dust: to the nearest tick, or to Hyperliquid's rule."""
        if not self.venue.sizes or price is None or not math.isfinite(price):
            return price

        inst    = self.instruments[i]
        rounded = inst.on_grid(price)
        if rounded != price:
            self.hit("a price put on significant figures" if inst.sigfigs is not None
                     else "a price put on a tick")
        return rounded


    def largest(self, i):
        """Fills, Sizes and dust: the maximum order quantity rounded down to the lot."""
        inst = self.instruments[i]
        if inst.lot <= 0.0 or math.isinf(inst.max_qty):
            return inst.max_qty
        return lots(inst.max_qty, inst.lot) * inst.lot


    def held(self, i, wanted):
        """A value order's size, no more than the largest order where sizes hold."""
        if self.venue.sizes and wanted > self.largest(i):
            self.hit("an order held to the maximum quantity")
            return self.largest(i)
        return wanted


    def remainder_done(self, i, remaining):
        """An order retires when what remains of it rounds down to no lot."""
        lot = self.instruments[i].lot
        if self.venue.sizes and lot > 0.0:
            return lots(remaining, lot) <= 0
        return remaining <= self.dust(i)


    def dust(self, i):
        """Fills, Sizes and dust: with sizes off, a size this small moves nothing, in
        base units, or in dollars of contracts on an inverse contract."""
        inst = self.instruments[i]
        return DUST / inst.contract if inst.kind == "inverse" else DUST


    def new_order(self, instrument, side, kind, size=None, value=None, price=None,
                  trigger=None, tif="gtc", post_only=False, reduce_only=False,
                  reason="market"):
        order = dict(id=self.next_id, instrument=instrument, side=side, kind=kind,
                     size=None if size is None else float(size),
                     value=None if value is None else float(value),
                     price=None if price is None else float(price),
                     trigger=None if trigger is None else float(trigger),
                     given=None if size is None else float(size),
                     filled=0.0, placed=self.t, seen=0, status="pending", tif=tif,
                     post_only=post_only, reduce_only=reduce_only, reason=reason,
                     pair=None, leg=None, armed=True, armed_at=None, armed_by=None)
        self.next_id += 1
        self.orders[order["id"]] = order

        return order


    def refuse(self, order, why):
        """The order log: a refused event records the side and the size the call gave,
        and a leg asked on an instrument yet to print has no size."""
        order["status"] = "refused"
        given = order["given"]
        self.log("refused", order, why, side=order["side"],
                 size=None if given is None or math.isnan(given) else given)

        return order["id"]


    def size_refusal(self, order):
        """Orders and Sizes: why an order's size cannot be placed, or None. With sizes
        on it is rounded down to the lot, held to the maximum order quantity rounded
        down to the lot and checked against the minimum notional at the last close,
        and an order given in the quote currency is judged by its amount."""
        i    = order["instrument"]
        inst = self.instruments[i]
        if not self.listed[i]:
            return "the instrument has not printed its first bar"

        if self.delisted[i]:
            return "the instrument is delisted"

        if order["size"] is None:
            amount = order["value"]
            if amount is None:
                return None
            if not math.isfinite(amount) or amount <= 0.0:
                return f"amount: {amount} cannot fill"
            if self.venue.sizes and amount < inst.min_notional:
                if not (order["reduce_only"] and inst.reduce_only_exempt):
                    self.hit("an order below the minimum notional refused")
                    return (f"amount: {amount} is below the minimum notional of "
                            f"{inst.min_notional}")
                self.hit("a reduce-only order spared the minimum notional")
            return None

        if not math.isfinite(order["size"]) or order["size"] <= 0.0:
            return f"size: {order['size']} cannot fill"

        if not self.venue.sizes:
            return None

        rounded = self.lot_down(i, order["size"])
        if rounded <= 0.0:
            self.hit("a size rounded to no lot refused")
            return f"size: {order['size']} rounds to nothing at a lot of {inst.lot}"

        most = self.largest(i)
        if rounded > most:
            self.hit("an order held to the maximum quantity")
            order["asked"] = rounded
            rounded = most

        order["size"] = rounded

        if abs(inst.notional(rounded, self.mark[i])) < inst.min_notional:
            if not (order["reduce_only"] and inst.reduce_only_exempt):
                self.hit("an order below the minimum notional refused")
                return (f"size: {rounded} is below the minimum notional of "
                        f"{inst.min_notional}")
            self.hit("a reduce-only order spared the minimum notional")

        # a replace held to the maximum can leave less than a lot to fill
        left = rounded - order["filled"]
        if self.remainder_done(i, left):
            return f"size: {left} cannot fill"
        return None


    def place_market(self, order):
        """Orders, Order types: a market order fills at its instrument's next open,
        and a value order's size is its amount over its fill price (Units)."""
        why = self.size_refusal(order)
        if why is None and len([o for o in self.pending if o["instrument"]
                                == order["instrument"]]) >= self.venue.max_open_orders:
            why = f"max_open_orders: {self.venue.max_open_orders} already waiting"

        if why is not None:
            return self.refuse(order, why)

        # Orders, Units: a value order on spot or a linear contract is placed for its
        # amount
        self.log_order("placed", order, order["reason"] if order["value"] is None
                       else f"value of {order['value']}")
        self.log_hold(order)
        self.pending.append(order)

        return order["id"]


    def log_hold(self, order):
        """Sizes and dust: an order held to the largest order logs the hold once its
        placement stands, and a refused one never does."""
        asked = order.pop("asked", None)
        if asked is not None:
            self.log("clamped", order, "the maximum order quantity "
                     + allows(order["size"], asked), size=order["size"])


    def crossing(self, order, others):
        """Maker, taker and the book: the first of `others`, in slot order, that a limit
        at `order`'s price would trade against, the account's own resting limit of
        the other side that it reaches, or None. A leg yet to arm has no side, so it
        is neither checked nor met."""
        side = order["side"]
        if order["kind"] != "limit" or side is None:
            return None

        for own in others:
            if own["kind"] != "limit" or own["side"] in (None, side):
                continue
            if (side == "buy" and order["price"] >= own["price"]) or \
                    (side == "sell" and order["price"] <= own["price"]):
                return own
        return None


    def resting_refusal(self, order, replacing=None):
        """Why a limit or a stop cannot rest, or None."""
        i, side = order["instrument"], order["side"]
        why = self.size_refusal(order)
        if why is not None:
            return why

        # Sizes and dust: an inverse price rounded to zero has no reciprocal
        def tradable(price):
            inverse = self.instruments[i].kind == "inverse"
            return math.isfinite(price) and (not inverse or price > 0.0)

        if order["kind"] == "limit" and not tradable(order["price"]):
            return f"limit: {order['price']} is not a price"

        if order["kind"] == "stop" and not tradable(order["trigger"]):
            return f"stop: {order['trigger']} is not a price"

        others = [o for o in self.resting(i) if o is not replacing]
        if len(others) >= self.venue.max_open_orders:
            return f"max_open_orders: {self.venue.max_open_orders} already resting"

        if order["kind"] == "limit" and side is not None:
            own = self.crossing(order, others)
            if own is not None:
                self.hit("an order trading against the account's own refused")
                return f"it would trade against the account's own order {own['id']}"

            # Order types: post_only is checked against the last close, and a
            # replace is a new placement for it only when it moves the price
            moved = replacing is None or order["price"] != replacing["price"]
            if order["post_only"] and moved:
                close = self.mark[i]
                if (side == "buy" and order["price"] >= close) or \
                        (side == "sell" and order["price"] <= close):
                    self.hit("a post_only order refused at the last close")
                    return "post_only: it would take liquidity at the last close"

        return None


    def rest(self, order, slot=None):
        """Maker, taker and the book: the first free slot of the one table."""
        if slot is None:
            slot = next((k for k, o in enumerate(self.slots) if o is None), None)

        if slot is None:
            self.slots.append(order)
        else:
            self.slots[slot] = order

        order["status"] = "resting"


    def place_resting(self, order):
        order["price"]   = self.round_price(order["instrument"], order["price"])
        order["trigger"] = self.round_price(order["instrument"], order["trigger"])

        why = self.resting_refusal(order)
        if why is not None:
            return self.refuse(order, why)

        self.rest(order)
        self.log_order("placed", order, order["reason"])
        self.log_hold(order)

        return order["id"]


    def order(self, instrument, side, size=None, limit=None, stop=None, tif="gtc",
              post_only=False, reduce_only=False, value=None, reason="market"):
        if limit is not None:
            order = self.new_order(instrument, side, "limit", size=size, price=limit,
                                   tif=tif, post_only=post_only,
                                   reduce_only=reduce_only, reason="limit")
            return self.place_resting(order)

        if stop is not None:
            order = self.new_order(instrument, side, "stop", size=size, trigger=stop,
                                   reduce_only=reduce_only, reason="stop")
            return self.place_resting(order)

        tif   = "fok" if tif == "fok" else "ioc"
        order = self.new_order(instrument, side, "market", size=size, value=value,
                               tif=tif, reduce_only=reduce_only, reason=reason)
        return self.place_market(order)


    def buy(self, instrument, size, limit=None, stop=None, tif="gtc", post_only=False,
            reduce_only=False):
        return self.order(instrument, "buy", size, limit, stop, tif, post_only,
                          reduce_only)


    def sell(self, instrument, size, limit=None, stop=None, tif="gtc", post_only=False,
             reduce_only=False):
        return self.order(instrument, "sell", size, limit, stop, tif, post_only,
                          reduce_only)


    def buy_value(self, instrument, amount, tif="ioc", reduce_only=False):
        return self.value(instrument, "buy", amount, tif, reduce_only)


    def sell_value(self, instrument, amount, tif="ioc", reduce_only=False):
        return self.value(instrument, "sell", amount, tif, reduce_only)


    def value(self, instrument, side, amount, tif, reduce_only):
        """Orders, Units: a market order for an amount of the quote currency, its size
        the amount over its fill price, and on an inverse contract the contracts the
        amount in dollars is worth, a size it has as it is placed, and one that is
        no amount refused as one."""
        inst = self.instruments[instrument]
        if inst.kind == "inverse" and math.isfinite(amount) and amount > 0.0:
            return self.order(instrument, side, amount / inst.contract, tif=tif,
                              reduce_only=reduce_only)
        return self.order(instrument, side, value=amount, tif=tif, reduce_only=reduce_only)


    def close(self, instrument):
        """Orders: a market order for the whole position, sized when it is placed."""
        q = self.position[instrument]
        if q == 0.0:
            return None
        return self.order(instrument, "sell" if q > 0.0 else "buy", abs(q),
                          reduce_only=True)


    def bracket(self, instrument, stop=None, take_profit=None):
        """Order types: two sizeless reduce-only legs, one OCO pair, armed at once
        beside a position held, and so then held to the self-trade rule (Maker, taker
        and the book), and otherwise on the first fill that opens one."""
        q    = self.position[instrument]
        side = None if q == 0.0 else ("sell" if q > 0.0 else "buy")
        legs = []

        for leg, kind, price in (("stop", "stop", stop),
                                 ("take_profit", "limit", take_profit)):
            if price is None:
                continue

            order = self.new_order(instrument, side, kind,
                                   price=price if kind == "limit" else None,
                                   trigger=price if kind == "stop" else None,
                                   reduce_only=True, reason=leg)
            order["leg"]   = leg
            order["armed"] = side is not None
            self.place_resting(order)
            if order["status"] == "resting":
                legs.append(order)

        if len(legs) == 2:
            legs[0]["pair"], legs[1]["pair"] = legs[1]["id"], legs[0]["id"]

        for order in legs:
            if order["armed"]:
                self.hit("a bracket armed")
                if q < 0.0:
                    self.hit("a bracket armed on a short")
                self.log_order("armed", order, order["leg"])

        return [order["id"] for order in legs]


    def arm(self, i, filling):
        """Order types: the legs waiting on instrument `i` arm on the position the fill
        of `filling` has just opened there and keep its side; Maker, taker and the
        book: arming is a placement, and a take-profit that would then trade against
        the account's own is cancelled."""
        q = self.position[i]

        for order in self.resting(i):
            if not order["leg"] or order["armed"]:
                continue

            self.hit("a bracket armed")
            if q < 0.0:
                self.hit("a bracket armed on a short")
            if filling is not None and filling["kind"] != "market":
                self.hit("a bracket armed by a resting fill")

            order["armed"], order["armed_at"], order["seen"] = True, self.t, 0
            order["armed_by"] = None if filling is None else filling["kind"]
            order["side"]     = "sell" if q > 0.0 else "buy"
            self.log_order("armed", order, order["leg"])

            others = [o for o in self.resting(i) if o is not order]
            own    = self.crossing(order, others)
            if own is not None:
                self.hit("a take-profit cancelled as it armed against the "
                         "account's own")
                self.cancel(order["id"], "it would trade against the account's own "
                                         f"order {own['id']}")


    def disarm(self, i, filling):
        """A bracket goes with the position it armed on. The leg that closed it goes
        with its fill, and that leg's partner is left to the OCO rule."""
        spared = set() if filling is None else {filling["id"], filling["pair"]}

        for order in self.resting(i):
            if order["leg"] and order["armed"] and order["id"] not in spared:
                self.hit("a bracket cancelled with its position")
                if filling is not None and filling["leg"]:
                    self.hit("a bracket cancelled by another's leg")
                self.cancel(order["id"], "its position closed")


    def cancel(self, order_id, reason="cancelled by the account"):
        order = self.orders.get(order_id)
        if order is None or order["status"] not in ("pending", "resting"):
            return False

        if order["status"] == "pending":
            if reason == "cancelled by the account":
                self.hit("a waiting market order cancelled")
            self.pending.remove(order)
        else:
            self.slots[self.slots.index(order)] = None

        order["status"] = "cancelled"
        self.log("cancelled", order, reason)
        return True


    def cancel_all(self, instrument=None):
        live = [o for o in self.pending + self.resting()
                if instrument is None or o["instrument"] == instrument]
        for order in live:
            self.cancel(order["id"])

        return len(live)


    def replace(self, order_id, price=None, trigger=None, size=None):
        """Order types: the same order moved, keeping its id and its OCO pair. A
        successful replace takes the first free slot; a refused one changes nothing."""
        old = self.orders.get(order_id)
        if old is None or old["status"] != "resting":
            return None

        moved = dict(old)
        # the order log: a refused replace records the order's side and, where the
        # call gave a size, what the order filled with it
        given = dict(side=old["side"],
                     size=None if size is None else float(size) + old["filled"])

        if price is not None:
            moved["price"] = self.round_price(old["instrument"], float(price))
        if trigger is not None:
            moved["trigger"] = self.round_price(old["instrument"], float(trigger))

        if size is not None:
            moved["size"] = float(size) + old["filled"]
            # the size asked is what is left to fill, which has to be something
            remaining = moved["size"] - old["filled"]
            if not math.isfinite(remaining) or remaining <= 0.0:
                self.log("refused", old, f"replace: size: {remaining} cannot fill",
                         **given)
                return None

        if moved["kind"] == "limit" and moved["price"] != old["price"]:
            # a moved limit is a new placement for maker or taker and for post_only
            moved["placed"], moved["seen"] = self.t, 0

        why = self.resting_refusal(moved, replacing=old)
        if why is not None:
            self.log("refused", old, f"replace: {why}", **given)
            return None

        self.hit("a replace", self.t + 1)
        self.slots[self.slots.index(old)] = None
        old.update(moved)
        self.rest(old)
        self.log_order("placed", old, "replace")
        self.log_hold(old)

        return old["id"]


    def target(self, weights, tolerance=0.0, transfer=False, quantities=False):
        """Orders, Target: the legs that bring each named instrument to its weight,
        with `transfer` moving the spot legs' shortfall across from the margin wallet
        at the start of the next bar; An array: with `quantities`, to the position
        each value wants, in base units or contracts."""
        if not isinstance(weights, dict):
            if isinstance(weights, (int, float)):
                weights = [weights]
            weights = dict(enumerate(weights))

        for i, weight in sorted(weights.items()):
            inst = self.instruments[i]
            if inst.kind == "spot" and weight < 0.0:
                raise ValueError(f"weights: {weight} for {inst.name}, which cannot be "
                                 f"short; trade a perpetual instead")

        # a second target in one decision replaces the first: its legs, the
        # shortfall it moved across and the weights it asked for
        if self.decided == self.t:
            self.hit("a target replaced in its decision", self.t + 1)
            for order in self.target_legs:
                if order["status"] == "pending":
                    self.cancel(order["id"], "replaced by a later target")
            if self.target_transfer is not None:
                self.hit("a target's shortfall taken back", self.t + 1)
                del self.transfers[self.target_transfer]
            self.asked  = list(self.asked_before)
            self.wanted = list(self.wanted_before)
        else:
            self.asked_before  = list(self.asked)
            self.wanted_before = list(self.wanted)

        self.target_transfer = None

        # a leg an earlier decision's target left waiting for its instrument's next
        # bar would fill beside this one's, sized from the same position
        for order in list(self.pending):
            if (order["reason"] == "rebalance" and order["placed"] < self.t
                    and order["instrument"] in weights):
                self.hit("a waiting target leg replaced")
                self.cancel(order["id"], "replaced by a later target")

        self.hit("a target", self.t + 1)
        if quantities:
            self.hit("a target in quantities", self.t + 1)

        self.decided, self.target_legs = self.t, []
        for i, weight in weights.items():
            if quantities:
                self.asked[i], self.wanted[i] = None, weight
            else:
                self.asked[i], self.wanted[i] = weight, None

        equity = self.equity()
        reducing, growing = [], []

        for i, weight in sorted(weights.items()):
            inst = self.instruments[i]

            # the wanted position is the one whose notional at the close is the
            # weight's share of equity: base units, or contracts on an inverse. An
            # instrument yet to print its first bar has no close to size a leg on: a
            # weight of zero there asks for nothing, and any other places a leg
            # refused as any order there is
            close = self.mark[i]
            if close is None:
                if weight != 0.0:
                    self.hit("a target leg on an instrument yet to print")
                    # the order log gives it the side of its weight
                    growing.append((i, math.copysign(math.nan, weight)))
                continue

            unit   = inst.notional(1.0, close)
            wanted = weight if quantities else weight * equity / unit
            leg    = wanted - self.position[i]
            if self.venue.sizes and inst.lot > 0.0:
                leg = math.copysign(lots(abs(leg), inst.lot) * inst.lot, leg)

            if abs(leg) <= self.dust(i):
                continue

            # the tolerance reads the whole leg, before a flip splits it
            if abs(leg) * unit < tolerance * equity:
                self.hit("a target leg skipped below the tolerance")
                continue

            q = self.position[i]
            if q != 0.0 and (q + leg) * q < 0.0:
                self.hit("a flip split by a target", self.t + 1)
                reduce, grow = -q, q + leg
            elif q != 0.0 and abs(q + leg) < abs(q):
                reduce, grow = leg, 0.0
            else:
                reduce, grow = 0.0, leg

            def small(size, reduce_only):
                """A part below the minimum notional, which a reducing part escapes
                where the venue exempts reduce-only orders."""
                exempt = reduce_only and inst.reduce_only_exempt
                return (self.venue.sizes and abs(size) * unit < inst.min_notional
                        and not exempt)

            if reduce != 0.0:
                if small(reduce, True):
                    self.hit("a target part skipped below the minimum notional")
                else:
                    reducing.append((i, reduce))

            if abs(grow) > self.dust(i):
                if small(grow, False):
                    self.hit("a target part skipped below the minimum notional")
                else:
                    growing.append((i, grow))

        placed = []
        for legs, reduce_only in ((reducing, True), (growing, False)):
            for i, leg in legs:
                side     = "buy" if math.copysign(1.0, leg) > 0.0 else "sell"
                order_id = self.order(i, side, abs(leg), reduce_only=reduce_only,
                                      reason="rebalance")

                order = self.orders[order_id]
                self.target_legs.append(order)
                if order["status"] != "refused":
                    placed.append((i, math.copysign(order["size"], leg)))

        if transfer:
            # what the spot legs it queued cost at the close, fees in, less what
            # their sales raise and the cash the spot wallet holds
            needed = -self.spot_cash
            for i, leg in placed:
                if self.instruments[i].kind == "spot":
                    cost = leg * self.mark[i]
                    needed += cost + abs(cost) * self.taker(i)

            if needed > 0.0:
                self.hit("a target's spot shortfall moved across", self.t + 1)
                self.target_transfer = len(self.transfers)
                self.transfer(needed, to="spot")

        return [order["id"] for order in self.target_legs
                if order["status"] != "refused"]


    def transfer(self, amount, to="margin"):
        """Wallets: applied at the start of the next bar, before any fill."""
        if to not in ("spot", "margin"):
            raise ValueError(f"to: {to!r} is not a wallet; use spot or margin")

        if self.instruments[0].kind == "inverse":
            raise ValueError("transfer: an inverse account has one wallet, in its "
                             "coin, and nothing to transfer to")

        self.transfers.append(dict(amount=float(amount), to=to))


    def step(self):
        """The step, The order of events: from this row to the next."""
        if self.t + 1 >= len(self.times):
            return False

        self.t += 1
        t                 = self.t
        self.costs        = self.no_costs()
        self.charged      = [0.0] * len(self.instruments)
        self.fills_before = len(self.fills)
        self.clipped      = {}

        # Fills, The liquidity budget: what each instrument's fills may take this bar
        self.budget_left = {}
        for i, bar in enumerate(self.bars[t]):
            if bar is not None:
                _, volume, _ = self.window(i)
                self.budget_left[i] = self.venue.budget * volume

        self.apply_transfers()                                      # 1
        self.charge_funding(self.between[t], t, at_close=False)     # 2
        self.solve_carried(t)                                       # 3

        self.legs = {i: [Held(abs(self.position[i]), self.entry[i], None, 0.0, None,
                              self.position[i] > 0.0)]
                     for i in self.perps() if self.position[i] != 0.0}
        self.start = [bar.open if bar is not None and self.position[i] != 0.0 else None
                      for i, bar in enumerate(self.bars[t])]
        self.inside = [False] * len(self.bars[t])
        self.reach  = 0.0

        self.delist(t)
        self.fill_market_orders(t)                                  # 4
        if self.resting() and any(leg.start is not None
                                  for legs in self.legs.values() for leg in legs):
            self.clip_joined(t)

        self.resolve_resting(t)                                     # 5, 6 and 7
        self.charge_funding(self.owned[t], t, at_close=True)        # 8
        self.check_closes(t)
        self.live_through(t)
        self.see_row(t)                                             # 9

        self.sweep_dust()
        self.fold_pnl(t)
        self.ledger.append(self.row(t, self.costs))
        return True


    def charge(self, name, i, amount):
        """What a run records: `amount` of the bar's cost `name`, instrument `i`'s."""
        self.costs[name] += amount
        self.charged[i] += amount


    def fold_pnl(self, t):
        """What a run records: each instrument's pnl, the ledger identity split by
        instrument, its price PnL from its references and marks less its share of
        the bar's costs."""
        before = self.ledger[-1]["instruments"]
        fills  = self.fills[self.fills_before:]

        for i, inst in enumerate(self.instruments):
            mark  = self.mark[i]
            moved = inst.profit(before[i]["position"], before[i]["mark"], mark)
            for fill in fills:
                if fill["instrument"] == i:
                    signed = fill["size"] if fill["side"] == "buy" else -fill["size"]
                    moved += inst.profit(signed, fill["reference"], mark)
            self.made[i] += moved - self.charged[i]


    def snapshot(self):
        return copy.deepcopy({name: getattr(self, name) for name in MUTABLE})


    def restore(self, saved):
        for name, value in saved.items():
            setattr(self, name, value)


    def resolve_resting(self, t):
        """The step, 5 to 7: the resting orders, then the check, then the end of the
        bar for the orders it left standing. Under the pessimistic setting each
        perpetual fill that, when its turn comes, would reduce the position it meets
        with its reference on the favourable side of where that position started on
        the bar is first held back, and if the check then liquidates, it never
        happened (Liquidation, The rule, 4)."""
        if self.venue.intrabar == "pessimistic" and self.may_hold(t):
            saved = self.snapshot()
            held  = self.fill_resting_orders(t, hold_favourable=True)
            if self.check(t):
                if held:
                    self.hit("a favourable exit held back on a liquidating bar")
                self.end_resting_bar(t)
                return

            self.restore(saved)

        self.fill_resting_orders(t)
        self.check(t)
        self.end_resting_bar(t)


    def may_hold(self, t):
        """Whether the bar reaches a resting perpetual order that could be held back:
        one favourable to the position the resting orders find, or one priced beyond
        an order on the other side of its instrument, which could open the position
        it would then reduce. Where a perpetual keeps maintenance, an order beside
        one on the other side of its instrument is enough, priced anywhere: that
        order's fill can clip the bars, and the clip can lift the price the first
        meets past where its position started (Liquidation, The rule, 3)."""
        offers = []
        for order in self.resting():
            i   = order["instrument"]
            bar = self.bar_for(t, i)
            if order["leg"] and not order["armed"] and bar is not None \
                    and bar.volume > 0.0 and self.instruments[i].kind != "spot":
                # a perpetual leg yet to arm could arm on a position a fill before it
                # opens, on either side, so the dry run decides whether it is held
                return True

            found = None if bar is None else self.offer(order, bar)
            side  = self.side_of(order)
            if found is None or side is None or self.instruments[i].kind == "spot":
                continue
            offers.append((order, i, side, found[0]))

        clips = any(self.has_maintenance(i) for i in self.perps())

        def beyond(side, price, at):
            return clips or (price > at if side == "sell" else price < at)

        return any(self.favourable(order, price, t)
                   or any(j == i and other != side and beyond(side, price, at)
                          for _, j, other, at in offers)
                   for order, i, side, price in offers)


    def favourable(self, order, price, t):
        """A perpetual fill that reduces the position it meets, priced on the side of
        where that position started on the bar that it gains on: the bar reached
        such a price only on a move back toward it."""
        i    = order["instrument"]
        q    = self.position[i]
        side = self.side_of(order)

        if self.instruments[i].kind == "spot" or side is None:
            return False

        reduces = (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")
        if not reduces or self.start[i] is None:
            return False
        return price > self.start[i] if q > 0.0 else price < self.start[i]


    def schedule_funding(self, events):
        """Funding, When an event is charged: a row owns the events stamped after its
        open and up to its end, and one stamped between rows is charged at the start
        of the next. A stamp within a second of a row boundary is on it."""
        self.owned   = collections.defaultdict(list)
        self.between = collections.defaultdict(list)
        source       = self.venue.funding
        if source == "none":
            return

        stamps = []
        if source == "data":
            if events is None:
                raise ValueError("data: no funding series, and the venue charges from "
                                 "data; pass the events or a venue with funding none")
            stamps = [(int(time), instrument, float(rate))
                      for time, instrument, rate in events]
        elif source[0] == "every":
            every = int(source[1] * 3_600 * 10**9)
            stamp = (self.times[0] // every + 1) * every
            while stamp <= self.times[-1] + self.interval:
                stamps.append((stamp, None, float(source[2])))
                stamp += every
        elif source[0] == "every_bars":
            for r in range(source[1], len(self.times), source[1]):
                stamps.append((self.times[r] + self.interval, None, float(source[2])))
        else:
            raise ValueError(f"funding: {source!r}; use none, data, every or "
                             f"every_bars")

        boundaries = sorted(set(self.times) | {t + self.interval for t in self.times})

        for stamp, instrument, rate in stamps:
            nearest = min(boundaries, key=lambda b: abs(b - stamp))
            if abs(nearest - stamp) <= SNAP:
                stamp = nearest
            for r, start in enumerate(self.times):
                end = start + self.interval
                if start < stamp <= end:
                    self.owned[r].append((stamp, instrument, rate, stamp < end))
                    break
                if r + 1 < len(self.times) and end < stamp <= self.times[r + 1]:
                    self.between[r + 1].append((stamp, instrument, rate, False))
                    break


    def funding_due(self, events, t, at_close):
        """Funding, The charge: the signed notional times the rate for each perpetual
        held, `q * p * r` or `u / p * r`, at the close of the row that owns the event,
        or at the last mark where there is none."""
        due = []
        for _, instrument, rate, inside in events:
            held = self.perps() if instrument is None else [instrument]
            for i in held:
                q = self.position[i]
                if q == 0.0 or self.instruments[i].kind == "spot":
                    continue

                bar   = self.bars[t][i]
                price = bar.close if at_close and bar is not None else self.mark[i]
                due.append((i, self.instruments[i].notional(q, price) * rate, inside))

        return due


    def pay_funding(self, i, charge):
        self.margin_cash -= charge
        self.charge("funding", i, charge)
        if self.trip[i] is not None:
            self.trip[i].funding += charge


    def charge_funding(self, events, t, at_close):
        for i, charge, _ in self.funding_due(events, t, at_close):
            self.hit("funding at the close of its row" if at_close
                     else "funding between rows")
            self.pay_funding(i, charge)


    def counted_inside(self, t):
        """Funding, When an event is charged: the charges stamped inside bar `t` that
        `intrabar` counts toward its path, on the positions as they stand, a debit
        under the pessimistic setting and a credit under the optimistic one."""
        return [(i, charge) for i, charge, inside
                in self.funding_due(self.owned[t], t, at_close=True) if inside
                and (charge > 0.0 if self.venue.intrabar == "pessimistic"
                     else charge < 0.0)]


    def state_at(self, piece, wallet, s):
        """Margin equity, the maintenance requirement and each instrument's notional on
        a piece of the path at fraction s, with every leg at the same fraction of its
        move."""
        equity    = wallet
        sizes     = collections.defaultdict(float)
        notionals = collections.defaultdict(float)

        for leg in piece:
            inst  = self.instruments[leg.instrument]
            price = self.along(leg.instrument, leg.start, leg.end, s)
            equity += inst.profit(leg.q, leg.entry, price)
            sizes[leg.instrument] += abs(leg.q)
            notionals[leg.instrument] += abs(inst.notional(leg.q, price))

        requirement = sum(self.maintenance_of(i, sizes[i], notionals[i]) for i in sizes)
        return equity, requirement, notionals


    def solve(self, path, wallets, share=1.0):
        """Liquidation, The trigger, close and clip points: the first point on the
        path where margin equity falls to `share` of the requirement, and the point
        the account leaves the market, the margin wallet holding `wallets[k]` on piece
        `k`. Both are found by bisection: on a piece, equity less the requirement is
        concave for linear legs, and both it and the close level are monotone for an
        inverse account's one leg, so each crosses once."""
        maintained = any(self.has_maintenance(leg.instrument)
                         for piece in path for leg in piece)

        def g(k, s):
            equity, requirement, _ = self.state_at(path[k], wallets[k], s)
            return equity - share * requirement

        def h(k, s):
            equity, _, notionals = self.state_at(path[k], wallets[k], s)
            return equity - sum(self.closing_fee(i, maintained) * notional
                                for i, notional in notionals.items())

        trigger = None
        for k in range(len(path)):
            if g(k, 0.0) <= 0.0:
                trigger = (k, 0.0)
                break
            if g(k, 1.0) <= 0.0:
                trigger = (k, first_zero(lambda s, k=k: g(k, s)))
                break

        if trigger is None:
            return None

        if maintained:
            return trigger, trigger

        # with no maintenance the account leaves where it stays at or past the close
        # level all the way to the trigger; a leg joining the check's path adds to
        # the closing fee and funding counted from the open moves the wallet, so the
        # level can step where one piece meets the next
        k, s = trigger
        while True:
            if h(k, 0.0) > 0.0:
                return trigger, (k, first_zero(lambda u, k=k: h(k, u), 0.0, s))
            if k == 0 or h(k - 1, 1.0) > 0.0:
                return trigger, (k, 0.0)
            k, s = k - 1, 1.0


    def carried_path(self, t):
        """Liquidation, The path: every perpetual held into the bar, from its last mark
        to its open and from its open to its adverse extreme, all in step."""
        gap, bar_piece = [], []
        for i in self.perps():
            q = self.position[i]
            if q == 0.0:
                continue

            bar = self.bars[t][i]
            if bar is None:
                # an instrument unavailable on the bar keeps its last mark
                self.hit("a position held through a row without its bar")
                still = Leg(i, q, self.entry[i], self.mark[i], self.mark[i])
                gap.append(still)
                bar_piece.append(still)
                continue

            adverse = bar.low if q > 0.0 else bar.high
            gap.append(Leg(i, q, self.entry[i], self.mark[i], bar.open))
            bar_piece.append(Leg(i, q, self.entry[i], bar.open, adverse))

        return [gap, bar_piece]


    def solve_carried(self, t):
        """Liquidation, The rule, 1 to 3: the path for the positions carried in, the
        funding stamped inside the bar counted on its bar segment as the check counts
        it, since it comes after the open. A clip on the gap segment liquidates before
        the bar opens; one on the bar segment clips every held perpetual's bar there."""
        path = self.carried_path(t)
        if not path[0]:
            return

        counted = self.counted_inside(t)
        if counted:
            self.hit("funding inside a bar counted toward the path carried in")

        inside  = self.margin_cash - sum(charge for _, charge in counted)
        wallets = [self.margin_cash, inside]
        result  = self.solve(path, wallets)
        if result is None:
            return

        trigger, (k, s) = result
        # the gap is one move for every path, and every path starts the bar at the
        # opens; inside the bars, the optimistic setting waits for every path to reach
        # the trigger
        if trigger[0] == 1 and trigger[1] > 0.0 and not self.reached_by_every_path(
                path[1], inside, t):
            return

        if k == 0:
            self.hit("a liquidation before the bar opens")
            closes = [(leg.instrument, leg.q,
                       self.along(leg.instrument, leg.start, leg.end, s))
                      for leg in path[0]]
            # an account closed out before the bar opened never pays the funding
            # stamped inside it, so the backstop reads the bar without it
            self.liquidate(closes, t, path, [self.margin_cash] * 2, trigger,
                           moment="before")
            return

        self.hit("a bar clipped")
        if len({leg.instrument for leg in path[1]}) > 1:
            self.hit("a bar clipped across several perpetuals")

        for leg in path[1]:
            bar = self.bars[t][leg.instrument]
            if bar is None:
                continue

            clip = self.along(leg.instrument, leg.start, leg.end, s)
            hold = max if leg.q > 0.0 else min
            self.clipped[leg.instrument] = Bar(
                hold(bar.open, clip), hold(bar.high, clip), hold(bar.low, clip),
                hold(bar.close, clip), bar.volume)


    def clip_joined(self, t):
        """Liquidation, The rule, 3, for a leg that joined during the bar: once a fill
        has opened, grown or flipped a perpetual with orders still to resolve, the
        check's path is solved for the account as it then stands, and where it reaches
        the trigger every held perpetual's bar is clipped there. Without maintenance
        the trigger is the close point, past which the margin bound already holds every
        reducing fill, so only an account with maintenance is clipped. A clip only
        tightens. Returns whether it clipped."""
        if not self.account_maintained():
            return False

        legs = self.check_legs(t)
        held = [leg for leg in legs if leg.leave is None]
        if not held:
            return False

        wallet = self.margin_cash - sum(charge for _, charge in self.counted_inside(t))
        path, spans = self.pieces(legs)
        result = self.solve(path, [wallet] * len(path))
        if result is None:
            return False

        trigger, (k, u) = result
        if trigger != (0, 0.0) and not self.reached_by_every_path(held, wallet, t):
            return False

        a, b = spans[k]
        for leg in held:
            bar = self.bar_for(t, leg.instrument)
            if bar is None:
                continue

            start, end = self.ends(leg, a, b)
            clip = self.along(leg.instrument, start, end, u)
            hold = max if leg.q > 0.0 else min
            self.clipped[leg.instrument] = Bar(
                hold(bar.open, clip), hold(bar.high, clip), hold(bar.low, clip),
                hold(bar.close, clip), bar.volume)

        self.hit("a bar clipped for a leg that joined it")
        return True


    def reached_by_every_path(self, legs, wallet, t):
        """The path: under the optimistic setting an account holding several
        perpetuals is liquidated on a bar only if every path inside the bars reaches
        the trigger, which is so, past the start every path shares and the callers
        ask first, when a corner does, one perpetual at its adverse extreme and every
        other at its favourable extreme, or the closes do."""
        held = sorted({leg.instrument for leg in legs})
        if self.venue.intrabar == "pessimistic" or len(held) < 2:
            return True

        ends = {}
        for j in held:
            bar  = self.bars[t][j]
            long = next(leg.q for leg in legs if leg.instrument == j) > 0.0
            if bar is None:
                ends[j] = (self.mark[j],) * 3
            elif long:
                ends[j] = (bar.low, bar.high, bar.close)
            else:
                ends[j] = (bar.high, bar.low, bar.close)

        states = [{j: ends[j][0 if j == k else 1] for j in held} for k in held]
        states.append({j: ends[j][2] for j in held})

        for prices in states:
            piece = [Leg(leg.instrument, leg.q, leg.entry, prices[leg.instrument],
                         prices[leg.instrument]) for leg in legs]
            equity, requirement, _ = self.state_at(piece, wallet, 0.0)
            if equity - requirement <= 0.0:
                self.hit("a corner or the closes reaching the trigger")
                return True

        self.hit("a path inside the bars that never reaches the trigger")
        return False


    def track(self, i, before, after, price, opened):
        """The check's legs of a perpetual through the bar's fills. A reducing fill
        shrinks every held leg in proportion, as average-entry accounting does, so the
        legs' entries always sum to the position's, and an opening fill adds a leg
        where its price lies on its side's path. A flip keeps the old side until the
        flip's fraction, measured from the flip's price since the wallet holds what it
        realized, and the new side joins there."""
        legs    = self.legs.get(i, [])
        held    = [leg for leg in legs if leg.leave is None]
        left    = [leg for leg in legs if leg.leave is not None]
        floor   = max((leg.leave for leg in left), default=0.0)
        flipped = before * after < 0.0

        if flipped:
            bar = self.bars[self.t][i]
            if bar is not None and (price < bar.low if before > 0.0
                                    else price > bar.high):
                self.hit("a flip filled on the gap")

            # the old side leaves no earlier than it joined, so a second flip on the
            # bar never sets its new side beside the first flip's old side
            joined = max([leg.join for leg in held] + [floor])
            floor  = max(self.fraction(i, before > 0.0, price, self.t), joined)
            left += [leg._replace(entry=price, leave=floor) for leg in held]
            held = []
        elif after == 0.0:
            held = []
        elif abs(after) < abs(before):
            # in proportion, the last held leg taking what the others leave, so the
            # legs sum to the position and a lone leg holds its size exactly
            held = [leg._replace(size=leg.size * abs(after) / abs(before))
                    for leg in held]
            rest = abs(after)
            for leg in held:
                rest -= leg.size
            if held:
                held[-1] = held[-1]._replace(size=held[-1].size + rest)

        if opened > self.dust(i) and after != 0.0:
            join = floor if flipped else max(
                self.fraction(i, after > 0.0, price, self.t), floor)
            held.append(Held(opened, price, price, join, None, after > 0.0))

        self.legs[i] = held + left


    def check_legs(self, t):
        """The rule, 5: every carried leg from its open and every leg opened or grown
        during the bar from its fill price, each to its adverse extreme, with the old
        side of a flip up to the flip's fraction."""
        legs = []
        for i, records in self.legs.items():
            bar = self.bars[t][i]
            for leg in records:
                q = leg.size if leg.long else -leg.size
                if bar is None:
                    legs.append(CheckLeg(i, q, leg.entry, 0.0, self.mark[i],
                                         self.mark[i], leg.leave))
                    continue

                adverse = bar.low if leg.long else bar.high
                start   = bar.open if leg.start is None else leg.start
                legs.append(CheckLeg(i, q, leg.entry, leg.join, start, adverse,
                                     leg.leave))

        return legs


    def price_at(self, leg, s):
        """A check leg's price at fraction s; before it joins, its fill price."""
        if s <= leg.frac or leg.frac >= 1.0:
            return leg.start
        return self.along(leg.instrument, leg.start, leg.end,
                          (s - leg.frac) / (1.0 - leg.frac))


    def ends(self, leg, a, b):
        """A check leg's prices at the two ends of the piece from a to b. On the last
        piece, which has no length, a leg that joined at the path's end runs its whole
        adverse move, so the new side of a flip filled at the old side's extreme still
        goes to its own adverse extreme (The path)."""
        if a == b == 1.0 and leg.frac >= 1.0:
            return leg.start, leg.end
        return self.price_at(leg, a), self.price_at(leg, b)


    def pieces(self, legs):
        """The check's path, cut where a leg joins or leaves: on each piece the legs
        held through it move in step, and a leg that joined at the path's end, filled
        at its own adverse extreme or flipped at the old side's, is tested on a last
        piece of no length, without the old side that left there."""
        cuts = {0.0, 1.0} | {leg.frac for leg in legs}
        cuts |= {leg.leave for leg in legs if leg.leave is not None}
        cuts  = sorted(cuts)
        spans = list(zip(cuts, cuts[1:]))
        if any(leg.frac >= 1.0 for leg in legs):
            spans.append((1.0, 1.0))
        path = [[Leg(leg.instrument, leg.q, leg.entry, *self.ends(leg, a, b))
                 for leg in legs
                 if leg.frac <= a
                 and (leg.leave is None or (leg.leave >= b and leg.leave > a))]
                for a, b in spans]
        return path, spans


    def check(self, t):
        """Liquidation, The rule, 5: the path solved again against the raw bars for the
        account after the bar's fills, any funding stamped inside the bar counted as
        `intrabar` decides. Returns whether it liquidated."""
        legs = self.check_legs(t)
        held = [leg for leg in legs if leg.leave is None]
        if not held:
            return False

        counted = self.counted_inside(t)
        wallet  = self.margin_cash - sum(charge for _, charge in counted)

        if counted:
            self.hit("funding inside a bar counted toward the check")
        if any(leg.leave is not None for leg in legs):
            self.hit("a flip's old side on the check")
        if any(leg.frac > 0.0 for leg in legs):
            self.hit("a leg joining the check after the open")

        path, spans = self.pieces(legs)
        wallets = [wallet] * len(path)
        result  = self.solve(path, wallets)
        if result is None:
            return False

        trigger, (k, u) = result
        # every path starts where the check's does, so an account past the trigger
        # there is liquidated under either setting
        if trigger != (0, 0.0) and not self.reached_by_every_path(held, wallet, t):
            return False

        a, b = spans[k]
        self.hit("a liquidation by the check")
        if (k, u) == (0, 0.0):
            self.hit("a check closing the account at the open")

        for i, charge in counted:
            self.pay_funding(i, charge)          # an event counted is an event paid

        closes = []
        # in instrument order, a fixed order, each instrument's legs in their own
        for leg in sorted(held, key=lambda leg: leg.instrument):
            start, end = self.ends(leg, a, b)
            price = self.along(leg.instrument, start, end, u)
            if leg.frac > 0.0 and price == leg.start:
                # closed where it joined, at its fill price
                self.hit("a forced close of a leg that joined during the bar")
            closes.append((leg.instrument, leg.q, price))

        self.liquidate(closes, t, path, wallets, trigger)
        return True


    def check_closes(self, t):
        """The rule, 6: once the bar's own funding is charged, the account is checked
        at the closes, and closed there if a debit has taken it to its trigger. A
        wallet that fees have left below zero with nothing held is set to zero."""
        held = [i for i in self.perps() if self.position[i] != 0.0]
        if not held:
            if self.margin_cash < 0.0:
                self.hit("a wallet that fees left below zero set to zero")
                self.charge("liquidation", self.closer, self.margin_cash)
                self.margin_cash = 0.0
            return

        prices = {i: (self.bars[t][i].close if self.bars[t][i] is not None
                      else self.mark[i]) for i in held}
        piece = [Leg(i, self.position[i], self.entry[i], prices[i], prices[i])
                 for i in held]
        equity, requirement, _ = self.state_at(piece, self.margin_cash, 0.0)

        if equity - requirement <= 0.0:
            self.hit("a liquidation at the closes")
            # book then backstop reads the closes as its whole path
            self.liquidate([(i, self.position[i], prices[i]) for i in held], t,
                           [piece], [self.margin_cash], (0, 0.0), moment="close")


    def live_through(self, t):
        """Round trips: one still open at the close surely lived through its entry bar
        from the fill to the close, the fill, the extreme beyond it and the close
        after a fill inside the bar and the whole bar after one at the open, and
        through any later bar whole."""
        for i, trip in enumerate(self.trip):
            bar = self.bars[t][i]
            if trip is None or bar is None:
                continue

            if trip.open_bar != t:
                trip.see(bar.high, bar.low)
            else:
                trip.see(*self.entry_part(i, bar))


    def entry_part(self, i, bar):
        """Round trips: what a round trip's entry bar surely traded from its fill to
        the close, the whole bar after a fill at the open, and after a fill inside the
        bar the fill, the extreme beyond it and the close."""
        start, inside = self.entered[i]
        if not inside:
            return bar.high, bar.low
        if start < bar.open:
            return bar.low, max(start, bar.close)
        return min(start, bar.close), bar.high


    def liquidate(self, closes, t, path, wallets, trigger, moment="bar"):
        """Liquidation, The forced close: takeover closes every leg at its price at the
        clip, and book then backstop sells each from its trigger as a taker unless the
        path reaches the backstop level or the sale's costs would cross it."""
        if len({i for i, _, _ in closes}) < len(closes):
            self.hit("a forced close of a leg that joined during the bar")
        if len({i for i, _, _ in closes}) > 1:
            self.hit("a liquidation of several perpetuals")

        # the account keeps maintenance where its path does, as the clip was solved,
        # a leg that left during the bar among them
        maintained = any(self.has_maintenance(leg.instrument)
                         for piece in path for leg in piece)
        if maintained and not self.account_maintained():
            self.hit("a forced close after a perp with maintenance left the path")

        # counted where an account with maintenance is closed out, not where a solve
        # finds its trigger, which the optimistic setting or a stop can still spare
        if maintained:
            self.hit("a liquidation at the maintenance trigger")

        model = self.venue.liquidation
        if model != "takeover" and maintained:
            share = model[1]

            # each instrument's impact is sized on all it sells, and a sale before the
            # bar opened, priced off its trigger on the gap, is held inside what
            # traded from the last close through the bar, which holds that trigger
            sizes = collections.defaultdict(float)
            for i, q, _ in closes:
                sizes[i] += abs(q)

            sales = []
            for i, q, price in closes:
                side = "sell" if q > 0.0 else "buy"
                bar  = self.bars[t][i]
                if moment == "before" and bar is not None:
                    bar = bar._replace(low=min(bar.low, self.mark[i]),
                                       high=max(bar.high, self.mark[i]))
                sold, costs = self.taker_price(i, side, price, sizes[i], bar)
                sales.append((i, q, sold, costs))

            k, s = trigger
            kept = wallets[k]
            for i, q, sold, _ in sales:
                inst = self.instruments[i]
                kept += inst.profit(q, self.entry[i], sold)
                kept -= abs(inst.notional(q, sold)) * self.taker(i)

            _, requirement, _ = self.state_at(path[k], wallets[k], s)
            holds = kept >= share * requirement
            if holds and self.solve(path, wallets, share=share) is not None:
                # the bar cannot show that the sale finished before the path reached
                # the backstop level, which the setting decides
                self.hit("a sale on the book and its backstop level both reached")
                holds = self.venue.intrabar == "optimistic"

            if holds:
                self.hit("book then backstop selling on the book")
                for k, (i, q, sold, costs) in enumerate(sales):
                    # an instrument's last sale takes what its position still holds,
                    # which the sales of its other legs can leave a hair off their own
                    # size
                    if all(j != i for j, *_ in sales[k + 1:]):
                        q = self.position[i]
                    if q == 0.0:
                        continue

                    side = "sell" if q > 0.0 else "buy"
                    self.book(i, abs(q), sold, costs)
                    self.apply(i, side, abs(q), sold, self.taker(i), True, None,
                               "liquidation", reference=costs["reference"],
                               costs=costs, moment=moment)

                self.after_liquidation()
                return

            self.hit("book then backstop falling back to takeover")

        self.hit("a takeover")
        closed = {}
        for k, (i, q, price) in enumerate(closes):
            # an instrument's last close takes what its position still holds, which
            # the closes of its other legs can leave a hair off their own size
            if all(j != i for j, _, _ in closes[k + 1:]):
                q = self.position[i]
            if q == 0.0:
                continue

            rate = 0.0 if maintained else self.closing_fee(i, False)
            self.apply(i, "sell" if q > 0.0 else "buy", abs(q), price, rate, False,
                       None, "liquidation", reference=price, moment=moment)
            size, notional = closed.get(i, (0.0, 0.0))
            closed[i] = (size + abs(q),
                         notional + abs(self.instruments[i].notional(abs(q), price)))

        # whatever the margin wallet still holds is the liquidation charge, and the
        # wallet ends at exactly zero; What a run records: the charge is given to the
        # instruments closed, in proportion to each one's maintenance at the prices
        # its legs closed at, or to their notional there where none keeps any
        shares = {i: self.maintenance_of(i, size, notional)
                  for i, (size, notional) in closed.items()}
        if sum(shares.values()) <= 0.0:
            self.hit("a takeover charge split by notional")
            shares = {i: notional for i, (_, notional) in closed.items()}

        total = sum(shares.values())
        for i, share in shares.items():
            self.charge("liquidation", i, self.margin_cash * share / total)

        self.margin_cash = 0.0
        self.after_liquidation()


    def after_liquidation(self):
        for order in [o for o in self.pending + self.resting()
                      if self.instruments[o["instrument"]].kind != "spot"]:
            if order["leg"]:
                self.hit("a bracket cancelled by a liquidation")
            self.cancel(order["id"], "liquidation")

        self.liquidations += 1


    def apply_transfers(self):
        for transfer in self.transfers:
            amount = transfer["amount"]
            if transfer["to"] == "margin":
                cash, keep, to_zero = self.spot_cash, 0.0, False
            else:
                # the margin wallet keeps what covers the cap's requirement once its
                # unrealized PnL is counted, floored at zero so a profit cannot move;
                # with nothing required, a wallet holding a loss keeps that loss alone
                cash     = self.margin_cash
                required = self.cap_requirement()
                keep     = max(0.0, required - self.margin_unrealized())
                to_zero  = required == 0.0 and keep > 0.0

            allowed = max(0.0, cash - keep)
            moved   = min(amount, allowed)
            self.hit("a transfer")
            if moved < amount:
                self.hit("a transfer clamped")
                wallet = ("the spot wallet's cash" if transfer["to"] == "margin"
                          else "the margin wallet's free cash")
                self.log("clamped", None, f"transfer: {wallet} {allows(moved, amount)}",
                         size=moved)

            if transfer["to"] == "margin":
                self.spot_cash -= moved
                self.margin_cash += moved
            else:
                self.margin_cash -= moved
                self.spot_cash += moved
                if moved > 0.0 and moved == allowed:
                    # all it may give leaves exactly what it keeps, so with the cap off
                    # a wallet holding a loss is left at zero equity, its trigger
                    self.margin_cash = keep
                    # the path adds each perpetual's loss to the wallet in turn, which
                    # can leave what their sum kept a hair above zero: the wallet is
                    # lowered by the least that reaches the trigger as the path reads it
                    while to_zero and self.level_at_marks() > 0.0:
                        self.margin_cash = math.nextafter(self.margin_cash, -math.inf)

        self.transfers = []


    def level_at_marks(self):
        """Margin equity less the maintenance requirement with every perpetual held at
        its last mark, summed as the path sums it."""
        piece = [Leg(i, self.position[i], self.entry[i], self.mark[i], self.mark[i])
                 for i in self.perps() if self.position[i] != 0.0]
        equity, requirement, _ = self.state_at(piece, self.margin_cash, 0.0)
        return equity - requirement


    def delist(self, t):
        """Time, The clock: at the delisting time the waiting orders are cancelled and
        the position closes at its last close with no fee."""
        for i, inst in enumerate(self.instruments):
            if self.delisted[i] or inst.delist is None or self.times[t] < inst.delist:
                continue

            for order in [o for o in self.pending + self.resting()
                          if o["instrument"] == i]:
                self.cancel(order["id"], "delist")

            self.hit("a delisting")
            self.delisted[i] = True

            q = self.position[i]
            if q != 0.0:
                cash = self.margin_cash
                self.apply(i, "sell" if q > 0.0 else "buy", abs(q), self.mark[i],
                           rate=0.0, taker=False, order=None, reason="delist",
                           reference=self.mark[i], moment="before")
                self.zero_rounding(i, cash, 0.0, abs(q), self.mark[i])


    def taker_price(self, i, side, reference, size, bar):
        """Fills, A taker's price: the reference moved against the side by half the
        spread, the slippage and the impact, then held inside the bar."""
        bars, volume, sigma = self.window(i)
        spread = self.venue.spread

        if spread is None:
            s = 0.0
        elif spread == "estimated":
            s = edge(bars)
        else:
            s = spread[1] / 10_000.0

        slip = max(self.venue.slippage / 10_000.0, 0.0)
        imp  = 0.0
        if self.venue.impact is not None and volume > 0.0:
            model, k = self.venue.impact
            share = size / volume
            imp   = k * sigma * (math.sqrt(share) if model == "sqrt" else share)
            imp   = max(imp, 0.0)

        total = min(s / 2.0 + slip + imp, MAX_COST)
        scale = total / (s / 2.0 + slip + imp) if s / 2.0 + slip + imp > 0.0 else 1.0
        sign  = 1.0 if side == "buy" else -1.0

        # Inverse contracts: the rates move the reciprocal of the price, so each costs
        # its rate of the coin notional, and its part of the price, its rate of the
        # raw price, is its share of the move
        if self.instruments[i].kind == "inverse":
            raw  = reference / (1.0 - sign * total)
            unit = raw
        else:
            raw, unit = reference * (1.0 + sign * total), reference

        if bar is None:
            price = raw         # a liquidation sale where the instrument has no bar
        else:
            price = min(raw, bar.high) if side == "buy" else max(raw, bar.low)

        costs = dict(spread=unit * s / 2.0 * scale, slippage=unit * slip * scale,
                     impact=unit * imp * scale, adjustment=price - raw,
                     reference=reference)
        return price, costs


    def book(self, i, size, price, costs):
        """What a run records: a fill's costs are what it paid beyond its reference, in
        the account's currency, and its spread, slippage and impact share that in
        proportion to their parts of the price, so a price the bar's range held back
        pays each of them in proportion too."""
        computed = costs["spread"] + costs["slippage"] + costs["impact"]
        if computed <= 0.0:
            return

        if costs["adjustment"] != 0.0:
            self.hit("a taker's price held inside the bar")

        paid = abs(self.instruments[i].profit(size, costs["reference"], price))
        for name in ("spread", "slippage", "impact"):
            if costs[name] > 0.0:
                self.hit(self.paid(name))
            self.charge(name, i, paid * costs[name] / computed)


    def paid(self, name):
        """The rule a fill paying the cost `name` reaches, the spread and the impact
        by the kind the venue charges."""
        if name == "spread":
            return ("an estimated spread paid" if self.venue.spread == "estimated"
                    else "a fixed spread paid")
        if name == "impact":
            return f"{self.venue.impact[0]} impact paid"
        return "slippage paid"


    def fill_market_orders(self, t):
        waiting = []
        for order in self.pending:
            i   = order["instrument"]
            bar = self.bar_for(t, i)
            if bar is None:
                self.hit("an order waiting for its instrument's next bar")
                waiting.append(order)       # Time, The clock: it waits for its bar
                continue

            if bar.volume <= 0.0:
                # a bar that trades nothing fills no order, whatever its tif, so
                # nothing is weighed against it
                self.hit("a market order filling nothing")
                order["status"] = "expired"
                self.log("not_filled", order,
                         "nothing could fill: the bar printed no volume")
                continue

            side   = order["side"]
            inst   = self.instruments[i]
            budget = self.budget_left.get(i, 0.0)
            if order["value"] is not None:
                wanted = self.held(i, inst.units(order["value"], bar.open))
            else:
                wanted = order["size"] - order["filled"]

            allowed = min(wanted, budget)
            price, costs = self.taker_price(i, side, bar.open, allowed, bar)

            # Orders, Units: on a spent budget no fill prices a value order, and it
            # asks its amount over the open
            if order["value"] is not None and budget > 0.0:
                bought  = inst.units(order["value"], price)
                wanted  = self.held(i, bought)
                allowed = min(wanted, budget)
                # Sizes and dust: the hold is logged on the bar that prices the fill,
                # where the lots the amount buys are more than the largest order
                rounded, most = self.lot_down(i, bought), self.largest(i)
                if self.venue.sizes and rounded > most:
                    self.log("clamped", order, "the maximum order quantity "
                             + allows(most, rounded), size=most)

            if budget < wanted:
                self.hit("the liquidity budget")
            if order["value"] is not None:
                # Orders, Units: the size a value order fills is rounded down to the
                # lot before the wallet's clamps
                allowed = self.lot_down(i, allowed)

            # the order log: no clamp acts on a fill the bar offers nothing of
            size, by = (self.clamp(i, side, allowed, price, order["reduce_only"],
                                   self.taker(i)) if allowed > 0.0 else (0.0, None))
            size = self.lot_down(i, size)
            # Sizes and dust: a size at or below the dust moves nothing, so it allows
            # none of what was asked
            if size <= self.dust(i):
                size = 0.0

            # a value order asks the lots its amount buys
            asked = self.lot_down(i, wanted)
            if by is None and allowed < asked - self.dust(i):
                by = "the liquidity budget"
            whole         = size >= asked - self.dust(i)
            refused_whole = order["tif"] == "fok" and not whole

            if order["value"] is not None:
                if order["tif"] == "fok":
                    self.hit("a value order all or nothing")
                if order["reduce_only"]:
                    self.hit("a reduce-only value order")

            if size <= self.dust(i) or refused_whole:
                self.hit("a fok order filling nothing" if refused_whole
                         else "a market order filling nothing")
                order["status"] = "expired"
                if refused_whole:
                    why = f"fok: {by or 'the lot'} {allows(size, asked)}"
                elif by is None:
                    why = "nothing could fill: the size is below what a fill can book"
                else:
                    why = f"nothing could fill: {by} allows none"
                self.log("not_filled", order, why)
                continue

            self.hit("a value order" if order["value"] is not None
                     else "a market order")
            if not whole:
                self.log("clamped", order, f"{by or 'the lot'} {allows(size, asked)}",
                         size=size)

            # what is left can round to a hair below zero, and a bar never owes volume
            self.budget_left[i] = max(budget - size, 0.0)
            self.book(i, size, price, costs)
            cash = self.margin_cash
            price, costs = self.bound(i, side, size, price, bar, self.taker(i), costs)
            flat = self.position[i] == 0.0
            self.apply(i, side, size, price, self.taker(i), True, order,
                       order["reason"], reference=bar.open, costs=costs)
            self.zero_rounding(i, cash, self.paid_in, size, price)
            order["filled"] += size

            # a market order lives one bar: what it did not fill is gone
            if not whole:
                order["status"] = "expired"
                self.log("expired", order, "a market order takes one bar")
            else:
                order["status"] = "filled"

            if flat and self.position[i] != 0.0:
                self.arm(i, order)

        self.pending = waiting


    def side_of(self, order):
        """A bracket leg's side is taken from the position it finds (Order types)."""
        if order["side"] is not None:
            return order["side"]

        q = self.position[order["instrument"]]
        if q == 0.0:
            return None
        return "sell" if q > 0.0 else "buy"


    def inside_fill(self, order, t):
        """What a bar cannot show: whether `order` filled inside bar `t`, a resting
        order at a price the open had not reached, a limit beyond the open or a stop
        whose trigger lies beyond it."""
        if order is None or order["kind"] == "market":
            return False

        bar   = self.bars[t][order["instrument"]]
        level = order["price"] if order["kind"] == "limit" else order["trigger"]
        below = (order["side"] == "buy") == (order["kind"] == "limit")
        return level < bar.open if below else level > bar.open


    def reduces_only(self, order):
        """Whether `order` can only reduce the position it meets: a reduce-only order,
        or a sale on spot, which is held to the holding."""
        spot = self.instruments[order["instrument"]].kind == "spot"
        return order["reduce_only"] or (spot and self.side_of(order) == "sell")


    def after_fill(self, i, bar):
        """What a bar cannot show: the part of `bar` after the fill inside it that
        opened or flipped the position on instrument `i`. It opens at that fill's
        reference, or at the clip where one after the fill tightened past it, and
        closes at the close, reaches the extreme beyond the fill, and reaches the one
        on the open's side where it came after the fill: always where it works against
        the position, as the check runs the position from its fill to it, and where it
        favours the position only under the optimistic setting. Otherwise it goes no
        further that way than the fill and the close."""
        long    = self.position[i] > 0.0
        start   = max(self.start[i], bar.low) if long else min(self.start[i], bar.high)
        down    = start < bar.open
        favours = down == long
        after   = not favours or self.venue.intrabar == "optimistic"

        if down:
            high = bar.high if after else max(start, bar.close)
            return Bar(start, high, bar.low, bar.close, bar.volume)

        low = bar.low if after else min(start, bar.close)
        return Bar(start, bar.high, low, bar.close, bar.volume)


    def against(self, order):
        """Whether `order` would reduce the position on its instrument as it stands."""
        q, side = self.position[order["instrument"]], self.side_of(order)
        return (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")


    def met(self, order, bar):
        """The bar as `order` meets it, and the last close a limit is held to: the part
        after the fill inside it that opened the position on its instrument, with
        that fill's reference as its last close, for an order that can only reduce
        that position and would, and the bar itself otherwise."""
        i = order["instrument"]
        if self.inside[i] and self.reduces_only(order) and self.against(order):
            return self.after_fill(i, bar), self.start[i]
        return bar, self.mark[i]


    def offer_whole(self, order, t):
        """What a bar cannot show: the whole bar's offer to an order that meets only the
        part of it after its position's fill, to tell where that part decides
        otherwise, and None for one that meets the whole bar anyway."""
        i = order["instrument"]
        if not (order["armed"] and self.inside[i] and self.reduces_only(order)
                and self.against(order)):
            return None
        return self.offer(order, self.bar_for(t, i))


    def offer(self, order, bar, close=None):
        """Whether the bar reaches a resting order, at what reference, and whether it
        fills as a taker (Order types, Maker, taker and the book, Fills, Limits), a
        limit held to the trades since `close`, the last close unless given."""
        side = self.side_of(order)
        if side is None or not order["armed"]:
            return None

        inst = self.instruments[order["instrument"]]
        if close is None:
            close = self.mark[order["instrument"]]

        if order["kind"] == "limit":
            price = order["price"]
            # through needs the market a step past the limit; prices sit on the grid,
            # so half a step past is a whole one without a float landing short of it
            half    = inst.step(price) / 2.0 if self.venue.limit == "through" else 0.0
            touched = bar.low <= price if side == "buy" else bar.high >= price
            reached = (bar.low <= price - half) if side == "buy" else (
                bar.high >= price + half)
            if touched and not reached:
                self.hit("a limit a step short of through")
            opened_through = bar.open <= price if side == "buy" else bar.open >= price
            if not reached:
                return None

            # never past every price traded from the last close through the bar
            if side == "buy":
                price = min(price, max(close, bar.high))
            else:
                price = max(price, min(close, bar.low))
            return price, order["seen"] == 0 and opened_through

        trigger = order["trigger"]
        if side == "buy":
            if bar.high < trigger:
                return None
            return max(trigger, bar.open), True

        if bar.low > trigger:
            return None
        return min(trigger, bar.open), True


    def offered(self, order, bar):
        """`offer` on the bar as `order` meets it, counting a limit the prices traded
        since the last close hold back where it is offered, since its price orders the
        bar's exits even when a cancel or a clamp then leaves it unfilled."""
        found = self.offer(order, *self.met(order, bar))
        limit = found is not None and order["kind"] == "limit"
        if limit and found[0] != order["price"]:
            self.hit("a limit held to the prices traded since the last close")

        return found


    def fill_resting_orders(self, t, hold_favourable=False):
        """The step, 5: resting orders in slot order, the exits of the position each
        meets reordered by the PnL per unit each would realize, worst first unless
        optimistic. With `hold_favourable`, a fill that would reduce the position it
        meets with its reference on the favourable side of where that position
        started on the bar is held back when its turn comes. Returns how many were."""
        offers = []
        held   = 0
        before = list(self.position)

        for slot, order in enumerate(self.slots):
            if order is None:
                continue
            bar = self.bar_for(t, order["instrument"])
            if bar is None:
                continue                     # it waits for its instrument's next bar

            if order["kind"] == "limit" and order["post_only"] and order["seen"] == 0:
                side = order["side"]
                if (side == "buy" and bar.open <= order["price"]) or \
                        (side == "sell" and bar.open >= order["price"]):
                    self.hit("a post_only limit its first bar opened through")
                    self.cancel(order["id"],
                                "post_only, and the first bar opened through the limit")
                    continue
            if bar.volume <= 0.0:
                continue                     # a bar with no volume fills nothing

            found = self.offered(order, bar)
            if found is not None:
                offers.append((slot, order, found[0], found[1]))
            elif order["leg"] and not order["armed"]:
                # Order types: a leg yet to arm keeps its turn, which a fill before it
                # can arm it for
                offers.append((slot, order, None, None))

        for k in range(len(offers)):
            for j in range(k, len(offers)):
                slot, order, reference, _ = offers[j]
                if order["status"] != "resting":
                    continue
                # a leg a fill before it armed, and an order that can only reduce,
                # which meets a position a fill inside the bar opened from that fill
                # on, are offered the bar as it now stands for them
                if (reference is None and order["armed"]) or self.reduces_only(order):
                    found = self.offered(order, self.bar_for(t, order["instrument"]))
                    offers[j] = (slot, order) + (found or (None, None))

            self.order_exits(offers, k, before)
            slot, order, reference, taker = offers[k]
            if order["status"] != "resting":
                continue                     # its OCO partner filled first

            if reference is None:
                # unarmed, or the part of the bar it meets does not reach it, counted
                # where the whole bar reaches it at this turn
                full = self.offer_whole(order, t)
                if full is not None and not (hold_favourable
                                             and self.favourable(order, full[0], t)):
                    self.hit("an exit the bar reached only before its position's fill")
                continue

            i   = order["instrument"]
            bar = self.bar_for(t, i)
            part, _ = self.met(order, bar)
            side = self.side_of(order)
            if side is None:
                continue

            if hold_favourable and self.favourable(order, reference, t):
                held += 1
                reduced = ((before[i] > 0.0 and side == "sell")
                           or (before[i] < 0.0 and side == "buy"))
                if not reduced:
                    self.hit("an exit held back from a position the bar's orders "
                             "opened")
                if order["leg"] and order["armed_at"] == t \
                        and order["armed_by"] != "market":
                    self.hit("a leg a resting fill armed held back on a liquidating "
                             "bar")
                continue

            if order["kind"] == "limit" and order["post_only"] and taker:
                # Order types: a post-only limit the part of its first bar after a
                # fill opens at or through is cancelled there, as at the open
                self.hit("a post_only limit the part of its first bar after a fill "
                         "opened through")
                self.cancel(order["id"], "post_only, and the part of its first bar after "
                                         "a fill opened through the limit")
                continue

            remaining = (abs(self.position[i]) if order["size"] is None
                         else order["size"] - order["filled"])
            budget  = self.budget_left.get(i, 0.0)
            allowed = min(remaining, budget)
            if budget < remaining:
                self.hit("the liquidity budget")
            if order["kind"] == "stop":
                price, costs = self.taker_price(i, side, reference, allowed, part)
            else:
                price, costs = reference, dict(spread=0.0, slippage=0.0, impact=0.0,
                                               adjustment=0.0, reference=reference)

            rate = self.taker(i) if taker else self.maker(i)
            size, by = self.clamp(i, side, allowed, price, order["reduce_only"], rate)
            size  = self.lot_down(i, size)
            whole = size >= self.lot_down(i, remaining) - self.dust(i)
            if size <= self.dust(i) or (order["tif"] == "fok" and not whole):
                continue

            full = self.offer_whole(order, t)
            if full is not None and full != (reference, taker):
                self.hit("an exit priced from its position's fill")
            if not whole:
                if by is None and allowed < remaining - self.dust(i):
                    by = "the liquidity budget"
                self.log("clamped", order, f"{by or 'the lot'} {allows(size, remaining)}",
                         size=size)

            self.budget_left[i] = max(budget - size, 0.0)
            if order["kind"] == "stop":
                self.hit("a stop triggered")
            else:
                self.hit("a limit filled as a taker" if taker
                         else "a limit filled as a maker")
                if not taker and (part.open <= order["price"] if side == "buy"
                                  else part.open >= order["price"]):
                    self.hit("a limit on the book its bar opened through")

            if order["leg"]:
                self.hit("a bracket leg filled")
                if order["armed_at"] == t:
                    self.hit("a bracket leg filled on the bar it armed")
                    if order["armed_by"] != "market":
                        self.hit("a bracket leg filled on the bar a resting fill "
                                 "armed it")
                if order["size"] is None and before[i] != 0.0 \
                        and abs(self.position[i]) != abs(before[i]):
                    self.hit("a sizeless leg sized at a position the bar's fills moved")

            self.book(i, size, price, costs)
            cash = self.margin_cash
            price, costs = self.bound(i, side, size, price, part, rate, costs)
            reason = order["leg"] or order["reason"]
            was    = self.position[i]
            flat   = was == 0.0
            self.apply(i, side, size, price, rate, taker, order, reason,
                       reference=reference, costs=costs)
            self.zero_rounding(i, cash, self.paid_in, size, price)
            order["filled"] += size

            if order["leg"] and self.position[i] != 0.0 and was * self.position[i] > 0.0:
                self.hit("a bracket leg filled in part")
            if order["pair"] is not None:
                if self.cancel(order["pair"], "its OCO partner filled"):
                    self.hit("an OCO partner cancelled")

            # Order types: a bracket goes with the position it armed on, the leg whose
            # fill closed it included, whatever that leg had left to fill, which a leg
            # sized its own way logs as cancelled with the position
            if order["size"] is None or (order["leg"] and self.position[i] == 0.0):
                if order["size"] is not None and not self.remainder_done(
                        i, order["size"] - order["filled"]):
                    self.hit("a sized leg gone with the position its fill closed")
                    self.cancel(order["id"], "its position closed")
                done = self.position[i] == 0.0 and order["status"] == "resting"
            else:
                left = order["size"] - order["filled"]
                done = self.remainder_done(i, left)
                if done and left > self.dust(i):
                    self.hit("an order retired below a lot")
            if done:
                order["status"] = "filled"
                self.slots[self.slots.index(order)] = None

            # Order types: a bracket arms once the fill that opens its position is
            # booked, so an entry that fill completed is no longer the account's own
            # resting order (Maker, taker and the book)
            if flat and self.position[i] != 0.0:
                self.arm(i, order)

            # Liquidation, The rule, 3: a perpetual the fill opened, grew or flipped
            # has its path solved, and where it reaches the trigger the orders left
            # meet their bars clipped there
            now  = self.position[i]
            grew = was == 0.0 or was * now < 0.0 or abs(now) > abs(was)
            if self.instruments[i].kind != "spot" and grew and k + 1 < len(offers) \
                    and self.clip_joined(t):
                for j in range(k + 1, len(offers)):
                    slot_j, order_j = offers[j][0], offers[j][1]
                    found = self.offered(order_j, self.bar_for(t, order_j["instrument"]))
                    offers[j] = (slot_j, order_j) + (found or (None, None))

        return held


    def end_resting_bar(self, t):
        """The step, 7: the resting orders the check left standing have rested through
        the bar, and an ioc or fok limit among them has had its one bar, which a
        liquidation inside the bar would have cancelled first."""
        for order in self.resting():
            if self.bars[t][order["instrument"]] is None:
                if order["leg"]:
                    self.hit("a bracket waiting through a row without its bar")
                continue

            order["seen"] += 1
            # ioc and fok take one bar, filled or not, where gtc waits for the next
            if order["tif"] in ("ioc", "fok"):
                self.hit("an ioc or fok limit expiring after one bar")
                order["status"] = "expired"
                self.slots[self.slots.index(order)] = None
                self.log("expired", order, f"{order['tif']}: one bar")


    def order_exits(self, offers, k, before):
        """The step, 5: the exits of the position the offer at `k` meets, as the fills
        before it left that position, reorder among the slots they hold from `k` on,
        by the PnL per unit each would realize at its fill price, a stop's with the
        costs it pays for the size it asks, worst first unless optimistic. `before`
        holds the positions the bar's resting orders found."""
        def realize(item):
            _, order, price, _ = item
            # an OCO partner or a bracket a fill before it cancelled holds no slot, and
            # a leg yet to arm is not an exit
            if order["status"] != "resting" or price is None:
                return None

            i       = order["instrument"]
            q       = self.position[i]
            side    = self.side_of(order)
            reduces = (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")
            if not reduces:
                return None

            if order["kind"] == "stop":
                size = (abs(q) if order["size"] is None
                        else order["size"] - order["filled"])
                bar, _ = self.met(order, self.bar_for(self.t, i))
                price = self.taker_price(i, side, price, size, bar)[0]

            return self.instruments[i].profit(1.0 if q > 0.0 else -1.0, self.entry[i],
                                              price)

        if realize(offers[k]) is None:
            return

        i      = offers[k][1]["instrument"]
        places = [j for j in range(k, len(offers))
                  if offers[j][1]["instrument"] == i and realize(offers[j]) is not None]
        ordered = sorted(places, key=lambda j: realize(offers[j]),
                         reverse=self.venue.intrabar == "optimistic")

        if ordered != places:
            self.hit("a position's exits reordered")
            if before[i] * self.position[i] <= 0.0:
                self.hit("the exits of a position the bar's orders opened reordered")
            moved = [offers[j] for j in ordered]
            for place, item in zip(places, moved):
                offers[place] = item


    def clamp(self, i, side, size, price, reduce_only, rate):
        """Wallets, Orders and Margin: each clamp a fill passes, in the page's order,
        and the last that reduced it, as the order log names it, or None."""
        q  = self.position[i]
        by = None

        if reduce_only:
            reduces = (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")
            if not reduces:
                self.hit("a reduce_only order with nothing to reduce")
                return 0.0, "reduce_only"
            if size > abs(q):
                self.hit("a reduce_only order held to the position")
                by = "reduce_only"
            size = min(size, abs(q))

        if self.instruments[i].kind == "spot":
            if side == "sell":
                if size > max(q, 0.0):
                    self.hit("a spot sale held to the holding")
                    by = "the spot wallet's holding"
                size = min(size, max(q, 0.0))
            else:
                affordable = self.spot_cash / (price * (1.0 + rate))
                if size > affordable:
                    self.hit("a spot buy held to its cash")
                    by = "the spot wallet's cash"
                size = min(size, max(affordable, 0.0))
            return max(size, 0.0), by

        size, held = self.cap(i, side, max(size, 0.0), price, rate)
        return size, held or by


    def cap(self, i, side, size, price, rate):
        """Margin, The cap: summed notional over leverage may not exceed margin equity,
        measured before the fill's own fee but, for a flip, after the fee at `rate` on
        the side it closes; a growing fill is clamped to what it allows, an account
        over the cap may not grow, and one at or below zero equity may only shrink. A
        flip's new side fits the cap from zero. Returns the size, and the rule that
        held it or None: no equity, the last tier's bound or the margin cap, tested
        in that order."""
        q        = self.position[i]
        signed   = size if side == "buy" else -size
        reducing = min(size, abs(q)) if q * signed < 0.0 else 0.0
        growing  = size - reducing
        if growing <= self.dust(i):
            return size, None

        # the other legs sit where the fill sits on the path of the position it
        # flips or adds to, or from flat of the side it opens
        s      = self.fraction(i, q > 0.0 if q != 0.0 else side == "buy", price, self.t)
        others = self.others_at(s, self.t, skip=i)
        prices = {**others, i: price}

        def held_back():
            # the other legs count where the cap holds the fill back with them
            if others and s > 0.0:
                self.hit("the cap with the other legs at the fill's fraction")
            if any(self.bars[self.t][j] is None for j in others):
                self.hit("a held perp read at its last mark")

        equity = self.margin_equity(prices)
        if reducing > 0.0:
            equity -= abs(self.instruments[i].notional(reducing, price)) * rate

        if equity <= 0.0:
            self.hit("a growing fill refused at no equity")
            held_back()
            return reducing, "the margin wallet's equity"

        base  = 0.0 if q * signed <= 0.0 else abs(q)
        sizes = {j: abs(self.position[j]) for j in self.perps()}

        def need(extra):
            sizes[i] = base + extra
            return self.cap_requirement(prices, sizes)

        limit = growing
        inst  = self.instruments[i]
        if inst.tiers:
            last = inst.tiers[-1].up_to
            if inst.tiers_by == "notional":
                last /= abs(inst.notional(1.0, price))
            limit = min(limit, max(last - base, 0.0))

        # Margin, The cap: only a growing fill the cap holds is clamped, so one that fits
        # keeps the size it came with rather than reducing + limit, which can round a
        # float off it
        if need(0.0) >= equity:
            # no fill may raise the sum, which a fill on an instrument with no cap
            # never does, up to the last tier's bound
            if need(limit) <= need(0.0):
                if limit == growing:
                    return size, None
                self.hit("a fill held at the last tier's bound")
                return booked(q, signed, reducing + limit, base + limit), \
                    "the largest position"
            self.hit("an account over the cap kept from growing")
            held_back()
            extra = last_within(lambda x: need(0.0) - need(x), 0.0, limit)
            return booked(q, signed, reducing + extra, base + extra), "the margin cap"

        if limit < growing:
            self.hit("a fill held at the last tier's bound")

        if need(limit) <= equity:
            if limit == growing:
                return size, None
            return booked(q, signed, reducing + limit, base + limit), \
                "the largest position"

        self.hit("a fill held to the margin cap")
        held_back()
        # the requirement only rises with the fill's size, through every tier, and
        # jumps at a tier's bound, where the last size that fits is the bound itself
        extra = last_within(lambda x: equity - need(x), 0.0, limit)
        per   = abs(inst.notional(1.0, price)) if inst.tiers_by == "notional" else 1.0
        if any(abs(base + extra - tier.up_to / per) <= 1e-9 * tier.up_to / per
               for tier in inst.tiers[:-1]):
            self.hit("a fill held at a tier's bound")

        return booked(q, signed, reducing + extra, base + extra), "the margin cap"


    def bound(self, i, side, size, price, bar, rate, costs):
        """The margin bound: a reducing perpetual fill whose close-level equity at its
        price is below zero, with its own fee on the size it closes, is lifted toward
        the price where that equity is zero, never outside the bar the fill met nor
        past the position's average entry.
        The lift the bar holds is in the price and the part it cannot hold, net of the
        fee it moves, is paid into the wallet; both are a negative liquidation charge,
        and the fill log records the lift."""
        self.paid_in = 0.0
        q            = self.position[i]
        inst         = self.instruments[i]
        reduces      = (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")
        if inst.kind == "spot" or not reduces:
            return price, costs

        closed     = min(size, abs(q))
        part       = math.copysign(closed, q)    # what the fill closes, signed
        maintained = self.maintained_after(i, size)
        fee        = self.closing_fee(i, maintained)
        others     = self.others_at(self.fraction(i, q > 0.0, price, self.t), self.t,
                                    skip=i)

        def own_fee(p):
            return abs(inst.notional(closed, p)) * rate

        def kept(p):
            """What closing that part at `p` realizes, net of the fill's own fee."""
            return inst.profit(part, self.entry[i], p) - own_fee(p)

        def level(p, paid=rate):
            rest = sum(self.closing_fee(j, maintained)
                       * abs(self.instruments[j].notional(self.position[j], others[j]))
                       for j in others)
            rest += fee * abs(inst.notional(abs(q) - closed, p))
            return (self.margin_equity({**others, i: p})
                    - abs(inst.notional(closed, p)) * paid - rest)

        def lift(paid):
            """Where the fill is lifted to and what the bar cannot hold of the lift,
            the size it closes paying `paid`."""
            if level(price, paid) >= 0.0:
                return price, 0.0

            up   = 1.0 if q > 0.0 else -1.0
            span = price             # a purchase is lowered toward zero, never past it
            while up > 0.0 and level(price + span, paid) < 0.0 and span < 1e12:
                span *= 2.0
            star = price + up * first_zero(lambda d: -level(price + up * d, paid), 0.0,
                                           span)

            if q > 0.0:
                goal   = min(star, self.entry[i])
                lifted = max(price, min(goal, bar.high))
            else:
                goal   = max(star, self.entry[i])
                lifted = min(price, max(goal, bar.low))

            return lifted, max(kept(goal) - kept(lifted), 0.0)

        lifted, credit = lift(rate)
        # counted only where the closing fee would have booked something else
        if rate != fee and lift(fee) != (lifted, credit):
            self.hit("a reducing fill bounded at a fee of its own")

        if level(price) >= 0.0:
            return price, costs

        if (self.venue.liquidation_fee is not None and not maintained
                and self.account_maintained()):
            self.hit("a fill leaving no maintenance bounded at the liquidation fee")
        if any(self.bars[self.t][j] is None for j in others):
            self.hit("a held perp read at its last mark")
        if lifted != price:
            self.hit("the margin bound lifting a fill inside its bar")
        if credit > 0.0:
            self.hit("the margin bound paying what the bar cannot hold")
        if others and (lifted != price or credit > 0.0):
            self.hit("the margin bound beside another perp")

        self.paid_in = credit
        self.margin_cash += credit
        self.charge("liquidation", i, -(abs(inst.profit(size, price, lifted)) + credit))

        # The fill log and the order log: the fill log keeps the lift and what was paid
        # in apart from the bar's hold
        return lifted, dict(costs, adjustment=costs["adjustment"] + lifted - price,
                            lifted=lifted - price, paid_in=credit)


    def zero_rounding(self, i, cash, credit, size, price):
        """Numbers that are the same everywhere: a perpetual fill puts at stake the
        margin wallet's `cash` before it, the bound's `credit` and its notional, and
        one that leaves no perpetual held and the wallet within the rounding of what
        the bar's perpetual fills have put at stake sets the wallet to zero, as a
        takeover does, the hair the liquidation charge."""
        inst = self.instruments[i]
        if inst.kind == "spot":
            return

        self.reach += abs(cash) + credit + abs(inst.notional(size, price))
        if any(self.position[j] != 0.0 for j in self.perps()):
            return

        if abs(self.margin_cash) <= 1e-12 * self.reach:
            if self.margin_cash != 0.0:
                self.hit("a wallet a fill left within rounding of zero set to zero")
            self.charge("liquidation", i, self.margin_cash)
            self.margin_cash = 0.0


    def sweep_dust(self):
        """Fills, Sizes and dust: a position floating point leaves below half a lot is
        zeroed at the close, and its value there is the dust adjustment."""
        if not self.venue.sizes:
            return

        for i, inst in enumerate(self.instruments):
            q = self.position[i]
            if q == 0.0 or inst.lot <= 0.0 or abs(q) >= inst.lot / 2.0:
                continue

            self.hit("dust swept at the close")
            if inst.kind == "spot":
                value = inst.notional(q, self.mark[i])
            else:
                value = inst.profit(q, self.entry[i], self.mark[i])
            self.charge("dust", i, value)

            self.position[i], self.entry[i] = 0.0, 0.0
            if inst.kind != "spot" and all(self.position[j] == 0.0 for j in self.perps()):
                self.closer = i
            # Order types: a bracket goes with the position it armed on
            self.disarm(i, None)

            if self.trip[i] is not None:
                trip = self.trip[i]
                trip.close_bar, trip.close_time = self.t, self.times[self.t]
                trip.reason = "dust"
                self.trips.append(trip)
                self.trip[i] = None


    def apply(self, i, side, size, price, rate, taker, order, reason, reference,
              costs=None, moment="bar"):
        """Accounting: a fill's fee, its cash, its position and its round trip. The
        moment says where in its bar the fill sat: "before" the bar opened, inside
        the "bar", or at its "close"."""
        t     = self.t
        inst  = self.instruments[i]
        costs = costs or dict(spread=0.0, slippage=0.0, impact=0.0, adjustment=0.0)
        fee   = abs(inst.notional(size, price)) * rate
        # a takeover's closes are booked as no taker's, and pay the taker side
        own = inst.taker if taker or reason == "liquidation" else inst.maker
        if fee != 0.0 and rate == own:
            self.hit("a fee at its instrument's own rate")
        self.charge("fees", i, fee)

        signed   = size if side == "buy" else -size
        before   = self.position[i]
        closed   = min(size, abs(before)) if before * signed < 0.0 else 0.0
        opened   = size - closed
        realized = inst.profit(math.copysign(closed, before), self.entry[i], price)

        # Accounting, Positions: on spot the notional itself changes hands, on a
        # perpetual only the realized PnL and the fee move the margin wallet. The fee
        # goes first, as an addition of its own: the order decides the last bit of a
        # wallet a clamp empties, which a target's shortfall reads
        if inst.kind == "spot":
            self.spot_cash -= fee
            self.spot_cash -= inst.notional(signed, price)
            # Wallets: a hair below zero that floating point leaves is set to zero
            if -1e-8 * abs(inst.notional(size, price)) <= self.spot_cash < 0.0:
                self.hit("a spot wallet a fill left a hair below zero set to zero")
                self.charge("dust", i, self.spot_cash)
                self.spot_cash = 0.0
        else:
            self.margin_cash -= fee
            self.margin_cash += realized

        after = before + signed
        if abs(after) <= self.dust(i):
            after = 0.0

        if inst.kind == "inverse":
            self.hit("an inverse contract filled")
        if before * after < 0.0:
            self.hit("a flip")

        if closed > 0.0:
            trip = self.trip[i]
            # Round trips: one carried into this bar lived through it up to the fill
            # from its open, all of it when the fill came at the close, and none of
            # it when the fill came before the bar opened; one the bar opened and its
            # close closed, what the bar surely traded after the fill
            bar = self.bars[t][i]
            if trip.open_bar != t and bar is not None:
                if moment == "bar":
                    trip.see(bar.open)
                elif moment == "close":
                    trip.see(bar.high, bar.low)
            elif moment == "close" and bar is not None:
                trip.see(*self.entry_part(i, bar))

            trip.close(closed, price)
            trip.realized += realized
            trip.fees += fee * closed / size
            if after == 0.0 or after * before < 0.0:
                self.hit("a round trip closed")
                trip.close_bar, trip.close_time, trip.reason = t, self.times[t], reason
                self.trips.append(trip)
                self.trip[i] = None

        if after == 0.0:
            self.entry[i] = 0.0
        elif before == 0.0 or after * before < 0.0:
            self.entry[i] = price
        elif opened > 0.0:
            self.entry[i] = inst.average(abs(before), self.entry[i], opened, price)

        if opened > self.dust(i) and after != 0.0:
            if self.trip[i] is None:
                self.trip[i] = Trip(i, inst, "long" if after > 0.0 else "short", t,
                                    self.times[t])
            trip = self.trip[i]
            trip.open(opened, price)
            trip.fees += fee * opened / size

        if self.trip[i] is not None:
            self.trip[i].size = max(self.trip[i].size, abs(after))
        if self.instruments[i].kind != "spot":
            self.track(i, before, after, price, opened)

        # What a bar cannot show: what a fill inside the bar added traded only after
        # it, where a fill at the open that grows a position changes nothing
        began = before == 0.0 or before * after < 0.0
        if after == 0.0:
            self.start[i], self.inside[i] = None, False
        elif began or abs(after) > abs(before):
            inside = self.inside_fill(order, t)
            if began or inside:
                self.start[i], self.inside[i] = reference, inside
            if began:
                self.entered[i] = (reference, inside)

        self.position[i] = after
        if inst.kind != "spot" and all(self.position[j] == 0.0 for j in self.perps()):
            self.closer = i

        self.fills.append(dict(
            time=self.times[t], bar=t, instrument=i,
            order=None if order is None else order["id"], side=side, size=size,
            price=price, reference=reference, spread=costs["spread"],
            slippage=costs["slippage"], impact=costs["impact"], fee=fee,
            adjustment=costs["adjustment"], lifted=costs.get("lifted", 0.0),
            paid_in=costs.get("paid_in", 0.0), taker=taker, reason=reason,
        ))
        if order is not None:
            self.log("filled", order, reason, size=size, price=price)

        # a liquidation cancels every perpetual order itself, with its own reason
        if before != 0.0 and (after == 0.0 or before * after < 0.0) \
                and reason != "liquidation":
            self.disarm(i, order)

        return fee


    def round_trips(self):
        """Every round trip, the closed ones and then each still open at the end."""
        out = [trip.report() for trip in self.trips]
        for i, trip in enumerate(self.trip):
            if trip is not None:
                out.append(trip.report(self.mark[i], self.entry[i], self.position[i]))

        return out


def run(times, bars, instruments, venue=None, script=None, interval=None,
        funding=None):
    """Steps a simulator through every row, making the decisions `script` holds.

    `script` maps a row to the calls made at its close, each a verb and its keyword
    arguments, so a decision made at row t is filled on a later row, never on t.
    `funding` holds the venue's events, as (time, instrument, rate), when it charges
    from data.
    """
    sim    = Simulator(times, bars, instruments, venue, interval, funding)
    script = script or {}

    while True:
        for verb, kwargs in script.get(sim.t, []):
            getattr(sim, verb)(**kwargs)
        if not sim.step():
            return sim


def returns(barrier, k, i):
    """Barriers on an array: the return `barrier` sets for instrument `i` on row `k`,
    a number for every row or rows of their own, None where it sets none."""
    value = barrier if barrier is None or isinstance(barrier, (int, float)) else (
        barrier[k][i])
    return None if value is None or math.isnan(value) else float(value)


def levels(sim, entry, long, stop, take_profit):
    """Barriers on an array: a round trip's stop and take-profit at those returns
    from its average entry, each left out where it sets none or at or below zero."""
    out = {}
    if stop is not None:
        out["stop"] = entry * (1.0 - stop) if long else entry * (1.0 + stop)
    if take_profit is not None:
        out["take_profit"] = (entry * (1.0 + take_profit) if long
                              else entry * (1.0 - take_profit))

    for name, price in list(out.items()):
        if price <= 0.0:
            sim.hit("an array's barrier at or below zero left out")
            del out[name]

    return out


def play_array(sim, rows, sizing="weight", tolerance=0.0, stop=None, take_profit=None):
    """Orders, An array and Barriers on an array: `rows` through `sim` from the row its
    account opened at, row k of them the values it asks of its instruments at its kth
    close in the unit `sizing` names, a NaN holding and the last row never read.
    `stop` and `take_profit` are each None, a number or rows like `rows`, returns from
    the average entry that give every round trip its bracket at the close of the bar
    it opened on, and a barrier's fill holds its instrument flat until a row asks for
    zero or for the other side. Returns `sim`."""
    n       = len(sim.instruments)
    barred  = stop is not None or take_profit is not None
    held    = list(sim.position)      # each position at the last close
    decided = [None] * n              # the row whose decision last named each instrument
    opened  = [None] * n              # the row whose decision opened its round trip
    legs    = [{} for _ in range(n)]  # its bracket's legs, by name
    priced  = [None] * n              # the average entry they were priced at
    latched = [0.0] * n               # the side a barrier filled on, 0.0 for none

    for k, row in enumerate(rows[:-1]):
        t = sim.t

        # the brackets, instrument by instrument, before the row's target
        for i in range(n if barred else 0):
            q   = sim.position[i]
            ids = set(legs[i].values())
            if ids and any(fill["bar"] == t and fill["order"] in ids
                           for fill in sim.fills):
                latched[i] = math.copysign(1.0, held[i])
            if q == 0.0:
                legs[i] = {}
            elif held[i] == 0.0 or held[i] * q < 0.0:
                opened[i] = decided[i]
                priced[i] = sim.entry[i]
                at = levels(sim, priced[i], q > 0.0, returns(stop, opened[i], i),
                            returns(take_profit, opened[i], i))
                legs[i] = {}
                if at:
                    sim.hit("an array's bracket placed at its entry")
                    placed = sim.bracket(i, stop=at.get("stop"),
                                         take_profit=at.get("take_profit"))
                    legs[i] = {sim.orders[order]["leg"]: order for order in placed}
            elif legs[i] and sim.entry[i] != priced[i]:
                priced[i] = sim.entry[i]
                at = levels(sim, priced[i], q > 0.0, returns(stop, opened[i], i),
                            returns(take_profit, opened[i], i))
                sim.hit("an array's bracket moved with its entry")
                if "stop" in legs[i]:
                    sim.replace(legs[i]["stop"], trigger=at["stop"])
                if "take_profit" in legs[i]:
                    sim.replace(legs[i]["take_profit"], price=at["take_profit"])
            held[i] = q

        named = {}
        for i, value in enumerate(row):
            if math.isnan(value):
                continue
            if latched[i]:
                if value != 0.0 and math.copysign(1.0, value) == latched[i]:
                    sim.hit("an array's instrument held flat after its barrier")
                    continue
                sim.hit("an array's instrument named again after its barrier")
                latched[i] = 0.0
            named[i]   = float(value)
            decided[i] = k

        if named:
            sim.target(named, tolerance, quantities=sizing == "quantity")
        sim.step()

    return sim
