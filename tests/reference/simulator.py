"""probatum's reference simulator, written from the Specification page.

It exists to disagree with the engine, so it is written from the rules as the
Specification states them and never from the Rust. Each step names the section it
implements, so a disagreement can be argued from the page rather than guessed at,
and every price, cash movement and log row is computed plainly, in the order the
page lists them. It is slow on purpose.

It takes what the low-level binding takes, the bars on a clock, the instruments,
the venue and the decisions made at each close, and it returns what the binding
returns: the ledger, the fill log, the order log and the round trips.

Scope: the clock, the step's order of events, market, value, limit and stop
orders with their time in force, `post_only`, `reduce_only`, brackets, `replace`
and cancels, `target`, the slot table, maker after the first bar, self-trade
refusal, the worse exit first, the fill model (the trailing window, the liquidity
budget, the spread, slippage, impact, sizes and dust), fees, funding, the spot
and margin wallets with transfers, listing and delisting, round trips and the
ledger. The margin cap and liquidation, and so the check a funding event inside a
bar counts toward, inverse contracts and several instruments on one path are not
here.
"""

import collections
import math
import statistics

DUST     = 1e-9      # Sizes and dust: with sizes off, a fill this small moves nothing
MAX_COST = 0.99      # Fills, A taker's price: s / 2 + slip + imp at most
SNAP     = 10**9     # Funding: a stamp within a second of a row boundary is on it

Bar = collections.namedtuple("Bar", "open high low close volume")


class Instrument:

    def __init__(self, name, kind="linear", lot=0.0, tick=0.0, sigfigs=None,
                 decimals=None, min_notional=0.0, max_qty=math.inf,
                 reduce_only_exempt=False, delist=None):
        if kind not in ("spot", "linear"):
            raise ValueError(f"kind: {kind!r} is not modelled; use spot or linear")

        self.name               = name
        self.kind               = kind
        self.lot                = float(lot)
        self.tick               = float(tick)
        self.sigfigs            = sigfigs   # Hyperliquid's rule: significant figures,
        self.decimals           = decimals  # within a cap on decimal places
        self.min_notional       = float(min_notional)
        self.max_qty            = float(max_qty)
        self.reduce_only_exempt = reduce_only_exempt
        self.delist             = delist    # nanoseconds, or None


class Venue:

    def __init__(self, balance=10_000.0, maker=0.0002, taker=0.0006, leverage=10.0,
                 max_open_orders=8, limit="touch", window=24, budget=1.0, spread=None,
                 slippage=0.0, impact=None, sizes=False, intrabar="pessimistic",
                 funding="none"):
        if limit not in ("touch", "through"):
            raise ValueError(f"limit: {limit!r}; use touch or through")

        if intrabar not in ("pessimistic", "optimistic"):
            raise ValueError(f"intrabar: {intrabar!r}; use pessimistic or optimistic")

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


def periods_per_year(times):
    """Time, Annualisation: the clock's rows less one per 365 days of its span."""
    days = (times[-1] - times[0]) / 86_400e9
    return (len(times) - 1) * 365.0 / days


def mean(values):
    return sum(values) / len(values)


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
    if sum(tau) < 2 or po == 0.0 or pc == 0.0:
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


    def __init__(self, instrument, side, bar, time):
        self.instrument = instrument
        self.side       = side
        self.open_bar   = bar
        self.open_time  = time

        self.close_bar    = None
        self.close_time   = None
        self.size         = 0.0
        self.opened       = 0.0
        self.opened_value = 0.0
        self.closed       = 0.0
        self.closed_value = 0.0
        self.realized     = 0.0
        self.fees         = 0.0
        self.funding      = 0.0
        self.reason       = None


    def report(self, mark=None, entry=None, held=0.0):
        """The round trip's row. An open one is marked at the last close."""
        entry_price = self.opened_value / self.opened
        exit_price  = self.closed_value / self.closed if self.closed > 0.0 else math.nan
        unrealized  = 0.0
        if self.close_bar is None and mark is not None:
            unrealized = held * (mark - entry)

        pnl = self.realized + unrealized - self.fees - self.funding
        return dict(
            instrument=self.instrument, side=self.side, size=self.size,
            entry=entry_price, exit=exit_price,
            open_bar=self.open_bar, close_bar=self.close_bar,
            open_time=self.open_time, close_time=self.close_time,
            pnl=pnl, ret=pnl / (self.size * entry_price), fees=self.fees,
            funding=self.funding, reason=self.reason, open=self.close_bar is None,
        )


class Simulator:

    def __init__(self, times, bars, instruments, venue=None, interval=None,
                 funding=None):
        self.times       = [int(time) for time in times]
        self.bars        = bars                   # bars[t][i]: a Bar, or None
        self.instruments = list(instruments)
        self.venue       = venue or Venue()
        gaps             = [b - a for a, b in zip(self.times, self.times[1:])]
        self.interval    = int(interval or (statistics.median(gaps) if gaps else 0))
        n                = len(self.instruments)
        self.schedule_funding(funding)

        # Wallets: a number funds the one wallet a universe of one kind uses, and
        # a universe holding both kinds names each
        kinds   = {inst.kind for inst in self.instruments}
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
        self.budget_left = {}
        self.costs       = None

        self.see_row(0)
        self.ledger = [self.row(0)]


    def see_row(self, t):
        for i, bar in enumerate(self.bars[t]):
            if bar is not None:
                self.listed[i] = True
                self.mark[i]   = bar.close
                self.history[i].append(bar)


    def wallet_equity(self):
        spot   = self.spot_cash
        margin = self.margin_cash

        for i, inst in enumerate(self.instruments):
            if self.position[i] == 0.0:
                continue
            if inst.kind == "spot":
                spot += self.position[i] * self.mark[i]
            else:
                margin += self.position[i] * (self.mark[i] - self.entry[i])

        return spot, margin


    def equity(self):
        spot, margin = self.wallet_equity()
        return spot + margin


    def cap_requirement(self):
        """Margin, The cap: notional over leverage, summed over the perpetuals."""
        if self.venue.leverage <= 0.0:
            return 0.0
        return sum(abs(q) * self.mark[i] / self.venue.leverage
                   for i, q in enumerate(self.position)
                   if q != 0.0 and self.instruments[i].kind != "spot")


    def resting(self, instrument=None):
        return [o for o in self.slots if o is not None
                and (instrument is None or o["instrument"] == instrument)]


    def window(self, i):
        """Fills, The window: the instrument's last W bars before the fill's bar, their
        mean volume, and the sample deviation of their log returns close to close."""
        bars    = self.history[i][-self.venue.window:]
        closes  = [bar.close for bar in self.history[i][-(self.venue.window + 1):]]
        returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
        volume  = mean([bar.volume for bar in bars]) if bars else 0.0
        sigma   = statistics.stdev(returns) if len(returns) >= 2 else 0.0

        return bars, volume, sigma


    def row(self, t, costs=None):
        costs = costs or dict(fees=0.0, spread=0.0, slippage=0.0, impact=0.0,
                              funding=0.0, dust=0.0)

        spot, margin = self.wallet_equity()
        equity = spot + margin
        out    = dict(time=self.times[t], equity=equity, spot_cash=self.spot_cash,
                      margin_cash=self.margin_cash, terminal=equity <= 0.0,
                      instruments=[], **costs)

        for i in range(len(self.instruments)):
            q, mark = self.position[i], self.mark[i]
            out["instruments"].append(dict(
                position=q, entry=self.entry[i], mark=mark,
                notional=abs(q) * mark if q != 0.0 else 0.0,
                upnl=q * (mark - self.entry[i]) if q != 0.0 else 0.0,
                weight=q * mark / equity if q != 0.0 and equity > 0.0 else 0.0,
            ))

        return out


    def log(self, kind, order, reason, size=None, price=None):
        self.events.append(dict(
            time=self.times[self.t], bar=self.t, kind=kind,
            order=None if order is None else order["id"],
            instrument=None if order is None else order["instrument"],
            size=size, price=price, reason=reason,
        ))


    def lot_down(self, i, size):
        """Fills, Sizes and dust: with sizes on, down to a whole number of lots."""
        lot = self.instruments[i].lot
        if not self.venue.sizes or lot <= 0.0:
            return size
        return math.floor(round(size / lot, 9)) * lot


    def round_price(self, i, price):
        """Fills, Sizes and dust: to the nearest tick, or to Hyperliquid's rule."""
        inst = self.instruments[i]
        if not self.venue.sizes or price is None or not math.isfinite(price):
            return price

        if inst.sigfigs is not None:
            digits = inst.sigfigs - 1 - int(math.floor(math.log10(abs(price))))
            if inst.decimals is not None:
                digits = min(digits, inst.decimals)
            return round(price, digits)

        if inst.tick > 0.0:
            return round(price / inst.tick) * inst.tick

        return price


    def remainder_done(self, i, remaining):
        """An order retires when what remains of it can no longer fill."""
        lot = self.instruments[i].lot
        if self.venue.sizes and lot > 0.0:
            return remaining < lot - DUST
        return remaining <= DUST


    def new_order(self, instrument, side, kind, size=None, value=None, price=None,
                  trigger=None, tif="gtc", post_only=False, reduce_only=False,
                  reason="market"):
        order = dict(id=self.next_id, instrument=instrument, side=side, kind=kind,
                     size=None if size is None else float(size),
                     value=None if value is None else float(value),
                     price=None if price is None else float(price),
                     trigger=None if trigger is None else float(trigger),
                     filled=0.0, placed=self.t, seen=0, status="pending", tif=tif,
                     post_only=post_only, reduce_only=reduce_only, reason=reason,
                     pair=None, leg=None, armed=True)

        self.next_id += 1
        self.orders[order["id"]] = order

        return order


    def refuse(self, order, why):
        order["status"] = "refused"
        self.log("refused", order, why)
        return order["id"]


    def size_refusal(self, order):
        """Orders and Sizes: why an order's size cannot be placed, or None. With sizes
        on it is rounded down to the lot, held to the maximum order quantity and
        checked against the minimum notional at the last close."""
        i    = order["instrument"]
        inst = self.instruments[i]
        if not self.listed[i]:
            return "the instrument has not printed its first bar"

        if self.delisted[i]:
            return "the instrument is delisted"

        if order["size"] is None:
            return None

        if not math.isfinite(order["size"]) or order["size"] <= 0.0:
            return f"size: {order['size']} cannot fill"

        if not self.venue.sizes:
            return None

        rounded = self.lot_down(i, order["size"])
        if rounded <= 0.0:
            return f"size: {order['size']} rounds to nothing at a lot of {inst.lot}"

        if rounded > inst.max_qty:
            self.log("clamped", order, f"size: {rounded} held to {inst.max_qty}",
                     size=inst.max_qty)
            rounded = inst.max_qty

        order["size"] = rounded
        exempt = order["reduce_only"] and inst.reduce_only_exempt
        if rounded * self.mark[i] < inst.min_notional and not exempt:
            return (f"size: {rounded} is below the minimum notional of "
                    f"{inst.min_notional}")

        return None


    def place_market(self, order):
        """Orders, Order types: a market order fills at its instrument's next open,
        and a value order's size is its amount over its fill price (Units)."""
        why = self.size_refusal(order)

        if why is None and order["value"] is not None and (
                not math.isfinite(order["value"]) or order["value"] <= 0.0):
            why = f"amount: {order['value']} cannot fill"

        if why is None and len([o for o in self.pending if o["instrument"]
                                == order["instrument"]]) >= self.venue.max_open_orders:
            why = f"max_open_orders: {self.venue.max_open_orders} already waiting"

        if why is not None:
            return self.refuse(order, why)

        self.log("placed", order, order["reason"])
        self.pending.append(order)

        return order["id"]


    def resting_refusal(self, order, replacing=None):
        """Why a limit or a stop cannot rest, or None."""
        i, side = order["instrument"], order["side"]
        why = self.size_refusal(order)
        if why is not None:
            return why

        if order["kind"] == "limit" and not math.isfinite(order["price"]):
            return f"limit: {order['price']} is not a price"

        if order["kind"] == "stop" and not math.isfinite(order["trigger"]):
            return f"stop: {order['trigger']} is not a price"

        others = [o for o in self.resting(i) if o is not replacing]
        if len(others) >= self.venue.max_open_orders:
            return f"max_open_orders: {self.venue.max_open_orders} already resting"

        if order["kind"] == "limit" and side is not None:
            # Maker, taker and the book: an order that reaches the account's own
            # resting limit of the other side would trade against the same account
            for own in others:
                if own["kind"] != "limit" or own["side"] in (None, side):
                    continue
                if (side == "buy" and order["price"] >= own["price"]) or \
                        (side == "sell" and order["price"] <= own["price"]):
                    return f"it would trade against the account's own order {own['id']}"

            # Order types: post_only is checked against the last close
            if order["post_only"]:
                close = self.mark[i]
                if (side == "buy" and order["price"] >= close) or \
                        (side == "sell" and order["price"] <= close):
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
        self.log("placed", order, order["reason"])
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


    def buy_value(self, instrument, amount):
        return self.order(instrument, "buy", value=amount)


    def sell_value(self, instrument, amount):
        return self.order(instrument, "sell", value=amount)


    def close(self, instrument):
        """Orders: a market order for the whole position, sized when it is placed."""
        q = self.position[instrument]
        if q == 0.0:
            return None
        return self.order(instrument, "sell" if q > 0.0 else "buy", abs(q),
                          reduce_only=True)


    def bracket(self, instrument, stop=None, take_profit=None):
        """Order types: two sizeless reduce-only legs, one OCO pair, armed while the
        account holds the position they close and taking their side from it."""
        legs = []

        for leg, kind, price in (("stop", "stop", stop),
                                 ("take_profit", "limit", take_profit)):
            if price is None:
                continue

            order = self.new_order(instrument, None, kind,
                                   price=price if kind == "limit" else None,
                                   trigger=price if kind == "stop" else None,
                                   reduce_only=True, reason=leg)
            order["leg"]   = leg
            order["armed"] = False
            self.place_resting(order)
            if order["status"] == "resting":
                legs.append(order)

        if len(legs) == 2:
            legs[0]["pair"], legs[1]["pair"] = legs[1]["id"], legs[0]["id"]

        self.arm()
        return [order["id"] for order in legs]


    def arm(self):
        for order in self.resting():
            if not order["armed"] and self.position[order["instrument"]] != 0.0:
                order["armed"] = True
                self.log("armed", order, order["leg"])


    def cancel(self, order_id, reason="cancelled by the account"):
        order = self.orders.get(order_id)
        if order is None or order["status"] not in ("pending", "resting"):
            return False

        if order["status"] == "pending":
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
        if price is not None:
            moved["price"] = self.round_price(old["instrument"], float(price))
        if trigger is not None:
            moved["trigger"] = self.round_price(old["instrument"], float(trigger))

        if size is not None:
            moved["size"] = float(size) + old["filled"]
        if moved["kind"] == "limit" and price is not None:
            # a moved limit is a new placement for maker or taker and for post_only
            moved["placed"], moved["seen"] = self.t, 0

        why = self.resting_refusal(moved, replacing=old)
        if why is not None:
            self.log("refused", old, f"replace: {why}")
            return None

        self.slots[self.slots.index(old)] = None
        old.update(moved)
        self.rest(old)
        self.log("placed", old, "replace")

        return old["id"]


    def target(self, weights, tolerance=0.0):
        """Orders, Target: the legs that bring each named instrument to its weight."""
        if not isinstance(weights, dict):
            if isinstance(weights, (int, float)):
                weights = [weights]
            weights = dict(enumerate(weights))

        # a second target in one decision replaces the first's legs
        if self.decided == self.t:
            for order in self.target_legs:
                if order["status"] == "pending":
                    self.cancel(order["id"], "replaced by a later target")

        self.decided, self.target_legs = self.t, []
        equity = self.equity()
        reducing, growing = [], []

        for i, weight in sorted(weights.items()):
            inst = self.instruments[i]
            if inst.kind == "spot" and weight < 0.0:
                raise ValueError(f"weights: {weight} for {inst.name}, which cannot be "
                                 f"short; trade a perpetual instead")

            close = self.mark[i]
            leg   = weight * equity / close - self.position[i]
            if inst.lot > 0.0:
                leg = math.trunc(round(leg / inst.lot, 9)) * inst.lot
            if abs(leg) * close < tolerance * equity or abs(leg) <= DUST:
                continue

            q = self.position[i]
            if q != 0.0 and (q + leg) * q < 0.0:
                reducing.append((i, -q))
                grow = q + leg
            elif q != 0.0 and abs(q + leg) < abs(q):
                reducing.append((i, leg))
                grow = 0.0
            else:
                grow = leg

            # a growing leg below the minimum notional is skipped
            small = self.venue.sizes and abs(grow) * close < inst.min_notional
            if grow != 0.0 and not small:
                growing.append((i, grow))

        for legs, reduce_only in ((reducing, True), (growing, False)):
            for i, leg in legs:
                order_id = self.order(i, "buy" if leg > 0.0 else "sell", abs(leg),
                                      reduce_only=reduce_only, reason="rebalance")
                self.target_legs.append(self.orders[order_id])

        return [order["id"] for order in self.target_legs]


    def transfer(self, amount, to="margin"):
        """Wallets: applied at the start of the next bar, before any fill."""
        if to not in ("spot", "margin"):
            raise ValueError(f"to: {to!r} is not a wallet; use spot or margin")

        self.transfers.append(dict(amount=float(amount), to=to))


    def step(self):
        """The step, The order of events: from this row to the next."""
        if self.t + 1 >= len(self.times):
            return False

        self.t += 1
        t = self.t
        self.costs = dict(fees=0.0, spread=0.0, slippage=0.0, impact=0.0,
                          funding=0.0, dust=0.0)

        # Fills, The liquidity budget: what each instrument's fills may take this bar
        self.budget_left = {}
        for i, bar in enumerate(self.bars[t]):
            if bar is not None:
                _, volume, _ = self.window(i)
                self.budget_left[i] = self.venue.budget * volume

        self.apply_transfers()                   # 1
        self.charge_funding(self.between[t], t, at_close=False)     # 2
        # 3 the liquidation path is not in this scope
        self.delist(t)
        self.fill_market_orders(t)               # 4

        self.arm()
        self.fill_resting_orders(t)              # 5
        # 6 and 7, exits on a liquidating bar and the check, are not in this scope
        self.charge_funding(self.owned[t], t, at_close=True)        # 8
        self.see_row(t)                          # 9

        self.sweep_dust()
        self.ledger.append(self.row(t, self.costs))
        return True


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


    def charge_funding(self, events, t, at_close):
        """Funding, The charge: q * p * r for each perpetual held, at the close of the
        row that owns the event, or at the last mark where there is none."""
        for _, instrument, rate, _ in events:
            if instrument is None:
                held = [i for i, inst in enumerate(self.instruments)
                        if inst.kind != "spot"]
            else:
                held = [instrument]

            for i in held:
                q = self.position[i]
                if q == 0.0 or self.instruments[i].kind == "spot":
                    continue

                bar    = self.bars[t][i]
                price  = bar.close if at_close and bar is not None else self.mark[i]
                charge = q * price * rate
                self.margin_cash -= charge
                self.costs["funding"] += charge
                if self.trip[i] is not None:
                    self.trip[i].funding += charge


    def apply_transfers(self):
        for transfer in self.transfers:
            amount = transfer["amount"]
            if transfer["to"] == "margin":
                allowed = self.spot_cash
            else:
                spot, margin = self.wallet_equity()
                allowed = min(self.margin_cash, margin - self.cap_requirement())

            moved = max(0.0, min(amount, allowed))
            if moved < amount:
                self.log("clamped", None, f"transfer: {moved} of the {amount} asked",
                         size=moved)

            if transfer["to"] == "margin":
                self.spot_cash -= moved
                self.margin_cash += moved
            else:
                self.margin_cash -= moved
                self.spot_cash += moved

        self.transfers = []


    def delist(self, t):
        """Time, The clock: at the delisting time the waiting orders are cancelled and
        the position closes at its last close with no fee."""
        for i, inst in enumerate(self.instruments):
            if self.delisted[i] or inst.delist is None or self.times[t] < inst.delist:
                continue

            for order in [o for o in self.pending + self.resting()
                          if o["instrument"] == i]:
                self.cancel(order["id"], "delist")

            self.delisted[i] = True

            q = self.position[i]
            if q != 0.0:
                self.apply(i, "sell" if q > 0.0 else "buy", abs(q), self.mark[i],
                           rate=0.0, taker=False, order=None, reason="delist",
                           reference=self.mark[i])


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
        raw   = reference * (1.0 + sign * total)
        price = min(raw, bar.high) if side == "buy" else max(raw, bar.low)

        costs = dict(spread=reference * s / 2.0 * scale,
                     slippage=reference * slip * scale,
                     impact=reference * imp * scale, adjustment=price - raw)
        return price, costs


    def book(self, size, costs):
        """What a run records: a fill's costs are what it paid beyond its reference,
        so a price the bar's range held back pays each of them in proportion."""
        computed = costs["spread"] + costs["slippage"] + costs["impact"]
        if computed <= 0.0:
            return

        paid = (computed - abs(costs["adjustment"])) / computed
        self.costs["spread"] += size * costs["spread"] * paid
        self.costs["slippage"] += size * costs["slippage"] * paid
        self.costs["impact"] += size * costs["impact"] * paid


    def fill_market_orders(self, t):
        waiting = []
        for order in self.pending:
            i   = order["instrument"]
            bar = self.bars[t][i]
            if bar is None:
                waiting.append(order)       # Time, The clock: it waits for its bar
                continue

            side   = order["side"]
            budget = self.budget_left.get(i, 0.0)
            if order["value"] is not None:
                wanted = order["value"] / bar.open
            else:
                wanted = order["size"] - order["filled"]

            allowed = min(wanted, budget)
            price, costs = self.taker_price(i, side, bar.open, allowed, bar)
            if order["value"] is not None:
                wanted  = order["value"] / price
                allowed = min(wanted, budget)

            size = self.clamp(i, side, allowed, price, order["reduce_only"],
                              self.venue.taker)
            size          = self.lot_down(i, size)
            whole         = size >= self.lot_down(i, wanted) - DUST
            refused_whole = order["tif"] == "fok" and not whole
            if bar.volume <= 0.0 or size <= DUST or refused_whole:
                order["status"] = "expired"
                self.log("not_filled", order, "nothing could fill on its bar"
                         if order["tif"] != "fok" else "fok: the whole size could not")
                continue

            if not whole:
                self.log("clamped", order, f"{size} of the {wanted} asked fits",
                         size=size)

            self.budget_left[i] = budget - size
            self.book(size, costs)
            self.apply(i, side, size, price, self.venue.taker, True, order,
                       order["reason"], reference=bar.open, costs=costs)
            order["filled"] += size

            # a market order lives one bar: what it did not fill is gone
            if not whole:
                order["status"] = "expired"
                self.log("expired", order, "a market order takes one bar")
            else:
                order["status"] = "filled"

        self.pending = waiting


    def side_of(self, order):
        """A bracket leg's side is taken from the position it finds (Order types)."""
        if order["side"] is not None:
            return order["side"]

        q = self.position[order["instrument"]]
        if q == 0.0:
            return None
        return "sell" if q > 0.0 else "buy"


    def offer(self, order, bar):
        """Whether the bar reaches a resting order, at what reference, and whether it
        fills as a taker (Order types, Maker, taker and the book, Fills, Limits)."""
        side = self.side_of(order)
        if side is None or not order["armed"]:
            return None

        inst = self.instruments[order["instrument"]]
        if order["kind"] == "limit":
            price = order["price"]
            # through needs the market a tick past the limit; prices sit on the tick,
            # so half a tick past is a whole one without a float landing short of it
            past = (price - bar.low) if side == "buy" else (bar.high - price)
            if self.venue.limit == "through":
                reached = past >= inst.tick / 2.0
            else:
                reached = past >= 0.0
            opened_through = bar.open <= price if side == "buy" else bar.open >= price
            if not reached:
                return None
            return price, order["seen"] == 0 and opened_through

        trigger = order["trigger"]
        if side == "buy":
            if bar.high < trigger:
                return None
            return max(trigger, bar.open), True

        if bar.low > trigger:
            return None
        return min(trigger, bar.open), True


    def fill_resting_orders(self, t):
        """The step, 5: resting orders in slot order, a position's exits reordered by
        the PnL per unit each would realize, worst first unless optimistic."""
        offers = []
        for slot, order in enumerate(self.slots):
            if order is None:
                continue
            bar = self.bars[t][order["instrument"]]
            if bar is None:
                continue                     # it waits for its instrument's next bar

            if order["kind"] == "limit" and order["post_only"] and order["seen"] == 0:
                side = order["side"]
                if (side == "buy" and bar.open <= order["price"]) or \
                        (side == "sell" and bar.open >= order["price"]):
                    self.cancel(order["id"], "post_only: the first bar opened through")
                    continue
            if bar.volume <= 0.0:
                continue                     # a bar with no volume fills nothing

            found = self.offer(order, bar)
            if found is not None:
                offers.append((slot, order, found[0], found[1]))

        offers = self.worst_first(offers)
        for slot, order, reference, taker in offers:
            if order["status"] != "resting":
                continue                     # its OCO partner filled first

            i    = order["instrument"]
            bar  = self.bars[t][i]
            side = self.side_of(order)
            if side is None:
                continue

            remaining = (abs(self.position[i]) if order["size"] is None
                         else order["size"] - order["filled"])
            budget  = self.budget_left.get(i, 0.0)
            allowed = min(remaining, budget)
            if order["kind"] == "stop":
                price, costs = self.taker_price(i, side, reference, allowed, bar)
            else:
                price, costs = reference, dict(spread=0.0, slippage=0.0, impact=0.0,
                                               adjustment=0.0)

            rate = self.venue.taker if taker else self.venue.maker
            size = self.lot_down(i, self.clamp(i, side, allowed, price,
                                               order["reduce_only"], rate))
            whole = size >= self.lot_down(i, remaining) - DUST
            if size <= DUST or (order["tif"] == "fok" and not whole):
                continue

            if not whole:
                self.log("clamped", order, f"{size} of the {remaining} asked fits",
                         size=size)

            self.budget_left[i] = budget - size
            self.book(size, costs)
            reason = order["leg"] or order["reason"]
            self.apply(i, side, size, price, rate, taker, order, reason,
                       reference=reference, costs=costs)
            order["filled"] += size
            if order["pair"] is not None:
                self.cancel(order["pair"], "its OCO partner filled")

            if order["size"] is None:
                done = self.position[i] == 0.0
            else:
                done = self.remainder_done(i, order["size"] - order["filled"])
            if done:
                order["status"] = "filled"
                self.slots[self.slots.index(order)] = None

        for order in self.resting():
            if self.bars[t][order["instrument"]] is None:
                continue
            order["seen"] += 1
            # ioc and fok take one bar, filled or not, where gtc waits for the next
            if order["tif"] in ("ioc", "fok"):
                order["status"] = "expired"
                self.slots[self.slots.index(order)] = None
                self.log("expired", order, f"{order['tif']}: one bar")


    def worst_first(self, offers):
        """The step, 5: only a position's exits reorder, among the slots they hold."""
        def realize(item):
            _, order, price, _ = item
            i       = order["instrument"]
            q       = self.position[i]
            side    = self.side_of(order)
            reduces = (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")
            if not reduces:
                return None
            return (price - self.entry[i]) * (1.0 if q > 0.0 else -1.0)

        reverse = self.venue.intrabar == "optimistic"
        out     = list(offers)
        for i in range(len(self.instruments)):
            places = [k for k, item in enumerate(offers)
                      if item[1]["instrument"] == i and realize(item) is not None]
            ordered = sorted(places, key=lambda k: realize(offers[k]), reverse=reverse)
            for place, take in zip(places, ordered):
                out[place] = offers[take]
        return out


    def clamp(self, i, side, size, price, reduce_only, rate):
        """Wallets and Orders: every clamp a fill passes, in the page's order."""
        q = self.position[i]

        if reduce_only:
            reduces = (q > 0.0 and side == "sell") or (q < 0.0 and side == "buy")
            if not reduces:
                return 0.0
            size = min(size, abs(q))

        if self.instruments[i].kind == "spot":
            if side == "sell":
                size = min(size, max(q, 0.0))
            else:
                affordable = self.spot_cash / (price * (1.0 + rate))
                size       = min(size, max(affordable, 0.0))

        return max(size, 0.0)


    def sweep_dust(self):
        """Fills, Sizes and dust: a position floating point leaves below half a lot is
        zeroed at the close, and its value there is the dust adjustment."""
        if not self.venue.sizes:
            return

        for i, inst in enumerate(self.instruments):
            q = self.position[i]
            if q == 0.0 or inst.lot <= 0.0 or abs(q) >= inst.lot / 2.0:
                continue

            if inst.kind == "spot":
                value = q * self.mark[i]
            else:
                value = q * (self.mark[i] - self.entry[i])

            self.costs["dust"] += value
            self.position[i], self.entry[i] = 0.0, 0.0

            if self.trip[i] is not None:
                trip = self.trip[i]
                trip.close_bar, trip.close_time = self.t, self.times[self.t]
                trip.reason = "dust"
                self.trips.append(trip)
                self.trip[i] = None


    def apply(self, i, side, size, price, rate, taker, order, reason, reference,
              costs=None):
        """Accounting: a fill's fee, its cash, its position and its round trip."""
        t     = self.t
        costs = costs or dict(spread=0.0, slippage=0.0, impact=0.0, adjustment=0.0)
        fee   = size * price * rate
        self.costs["fees"] += fee

        signed    = size if side == "buy" else -size
        before    = self.position[i]
        closed    = min(size, abs(before)) if before * signed < 0.0 else 0.0
        opened    = size - closed
        direction = 1.0 if before > 0.0 else -1.0
        realized  = closed * (price - self.entry[i]) * direction

        # Accounting, Positions: on spot the notional itself changes hands, on a
        # perpetual only the realized PnL and the fee move the margin wallet
        if self.instruments[i].kind == "spot":
            self.spot_cash -= signed * price + fee
        else:
            self.margin_cash += realized - fee

        after = before + signed
        if abs(after) <= DUST:
            after = 0.0

        if closed > 0.0:
            trip = self.trip[i]
            trip.closed += closed
            trip.closed_value += closed * price
            trip.realized += realized
            trip.fees += fee * closed / size
            if after == 0.0 or after * before < 0.0:
                trip.close_bar, trip.close_time, trip.reason = t, self.times[t], reason
                self.trips.append(trip)
                self.trip[i] = None

        if after == 0.0:
            self.entry[i] = 0.0
        elif before == 0.0 or after * before < 0.0:
            self.entry[i] = price
        elif opened > 0.0:
            self.entry[i] = (self.entry[i] * abs(before) + price * opened) / abs(after)

        if opened > DUST and after != 0.0:
            if self.trip[i] is None:
                self.trip[i] = Trip(i, "long" if after > 0.0 else "short", t,
                                    self.times[t])
            trip = self.trip[i]
            trip.opened += opened
            trip.opened_value += opened * price
            trip.fees += fee * opened / size

        if self.trip[i] is not None:
            self.trip[i].size = max(self.trip[i].size, abs(after))

        self.position[i] = after
        self.fills.append(dict(
            time=self.times[t], bar=t, instrument=i,
            order=None if order is None else order["id"], side=side, size=size,
            price=price, reference=reference, spread=costs["spread"],
            slippage=costs["slippage"], impact=costs["impact"], fee=fee,
            adjustment=costs["adjustment"], taker=taker, reason=reason,
        ))
        if order is not None:
            self.log("filled", order, reason, size=size, price=price)

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
