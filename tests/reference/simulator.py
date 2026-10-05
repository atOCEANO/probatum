"""probatum's reference simulator, written from the Specification page.

It exists to disagree with the engine, so it is written from the rules as the
Specification states them and never from the Rust. Each step names the section it
implements, so a disagreement can be argued from the page rather than guessed at,
and every price, cash movement and log row is computed plainly, in the order the
page lists them. It is slow on purpose.

It takes what the low-level binding takes, the bars on a clock, the instruments,
the venue and the decisions made at each close, and it returns what the binding
returns: the ledger, the fill log, the order log and the round trips.

Scope: the clock, the step's order of events, market and value orders and
`target`, fees, the spot and margin wallets with transfers, listing and delisting,
round trips and the ledger. Orders that rest, the fill model's costs, funding, the
margin cap and liquidation, inverse contracts and several instruments on one path
are not here.
"""

import collections
import math
import statistics

DUST = 1e-9          # Fills, Sizes and dust: a fill this small moves nothing

Bar = collections.namedtuple("Bar", "open high low close volume")


class Instrument:

    def __init__(self, name, kind="linear", lot=0.0, delist=None):
        if kind not in ("spot", "linear"):
            raise ValueError(f"kind: {kind!r} is not modelled; use spot or linear")

        self.name   = name
        self.kind   = kind
        self.lot    = float(lot)
        self.delist = delist          # nanoseconds, or None


class Venue:

    def __init__(self, balance=10_000.0, maker=0.0002, taker=0.0006, leverage=10.0,
                 max_open_orders=8):
        self.balance         = balance
        self.maker           = float(maker)
        self.taker           = float(taker)
        self.leverage        = float(leverage)
        self.max_open_orders = int(max_open_orders)


def periods_per_year(times):
    """Time, Annualisation: the clock's rows less one per 365 days of its span."""
    days = (times[-1] - times[0]) / 86_400e9
    return (len(times) - 1) * 365.0 / days


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

    def __init__(self, times, bars, instruments, venue=None, interval=None):
        self.times       = [int(time) for time in times]
        self.bars        = bars                   # bars[t][i]: a Bar, or None
        self.instruments = list(instruments)
        self.venue       = venue or Venue()
        gaps             = [b - a for a, b in zip(self.times, self.times[1:])]
        self.interval    = interval or (statistics.median(gaps) if gaps else 0)
        n                = len(self.instruments)

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
        self.listed      = [False] * n
        self.delisted    = [False] * n
        self.trip        = [None] * n
        self.trips       = []
        self.pending     = []          # market orders, filling at each one's next bar
        self.transfers   = []
        self.orders      = {}
        self.next_id     = 1
        self.fills       = []
        self.events      = []
        self.decided     = -1          # the row whose close the last `target` came from
        self.target_legs = []

        self.see_row(0)
        self.ledger = [self.row(0, fees=0.0)]


    def see_row(self, t):
        for i, bar in enumerate(self.bars[t]):
            if bar is not None:
                self.listed[i] = True
                self.mark[i]   = bar.close


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


    def row(self, t, fees):
        spot, margin = self.wallet_equity()
        equity = spot + margin
        out    = dict(time=self.times[t], equity=equity, spot_cash=self.spot_cash,
                      margin_cash=self.margin_cash, fees=fees, terminal=equity <= 0.0,
                      instruments=[])

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


    def place(self, instrument, side, size=None, value=None, reduce_only=False,
              reason="market"):
        """Orders, Order types: a market order fills at its instrument's next open,
        and a value order's size is its amount over that fill price (Units)."""
        order = dict(id=self.next_id, instrument=instrument, side=side, kind="market",
                     size=None if size is None else float(size),
                     value=None if value is None else float(value), filled=0.0,
                     placed=self.t, status="pending", reduce_only=reduce_only,
                     reason=reason)
        self.next_id += 1
        self.orders[order["id"]] = order

        asked   = order["size"] if value is None else order["value"]
        refusal = None
        if asked is None or not math.isfinite(asked) or asked <= 0.0:
            refusal = f"size: {asked} cannot fill"
        elif not self.listed[instrument]:
            refusal = "the instrument has not printed its first bar"
        elif self.delisted[instrument]:
            refusal = "the instrument is delisted"
        elif sum(1 for o in self.pending
                 if o["instrument"] == instrument) >= self.venue.max_open_orders:
            refusal = f"max_open_orders: {self.venue.max_open_orders} already waiting"

        if refusal is not None:
            order["status"] = "refused"
            self.log("refused", order, refusal)
            return order["id"]

        self.log("placed", order, reason)
        self.pending.append(order)
        return order["id"]


    def buy(self, instrument, size, reduce_only=False):
        return self.place(instrument, "buy", size, reduce_only=reduce_only)


    def sell(self, instrument, size, reduce_only=False):
        return self.place(instrument, "sell", size, reduce_only=reduce_only)


    def buy_value(self, instrument, amount):
        return self.place(instrument, "buy", value=amount)


    def sell_value(self, instrument, amount):
        return self.place(instrument, "sell", value=amount)


    def close(self, instrument):
        """Orders: a market order for the whole position, sized when it is placed."""
        q = self.position[instrument]
        if q == 0.0:
            return None
        return self.place(instrument, "sell" if q > 0.0 else "buy", abs(q),
                          reduce_only=True)


    def cancel(self, order_id):
        order = self.orders.get(order_id)
        if order is None or order["status"] != "pending":
            return False

        order["status"] = "cancelled"
        self.pending.remove(order)
        self.log("cancelled", order, "cancelled by the account")
        return True


    def cancel_all(self, instrument=None):
        waiting = [o for o in self.pending
                   if instrument is None or o["instrument"] == instrument]
        for order in waiting:
            self.cancel(order["id"])

        return len(waiting)


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
                    order["status"] = "cancelled"
                    self.pending.remove(order)
                    self.log("cancelled", order, "replaced by a later target")

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
                growing.append((i, q + leg))
            elif q != 0.0 and abs(q + leg) < abs(q):
                reducing.append((i, leg))
            else:
                growing.append((i, leg))

        for legs, reduce_only in ((reducing, True), (growing, False)):
            for i, leg in legs:
                order_id = self.place(i, "buy" if leg > 0.0 else "sell", abs(leg),
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
        t    = self.t
        fees = 0.0

        self.apply_transfers()                   # 1
        # 2 funding between rows and 3 the liquidation path are not in this scope
        self.delist(t)
        fees += self.fill_market_orders(t)       # 4
        # 5 to 8, resting orders to funding at the closes, are not in this scope
        self.see_row(t)                          # 9
        self.ledger.append(self.row(t, fees))

        return True


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

            self.delisted[i] = True
            for order in [o for o in self.pending if o["instrument"] == i]:
                order["status"] = "cancelled"
                self.pending.remove(order)
                self.log("cancelled", order, "delist")

            q = self.position[i]
            if q != 0.0:
                self.apply(i, "sell" if q > 0.0 else "buy", abs(q), self.mark[i],
                           rate=0.0, taker=False, order=None, reason="delist",
                           reference=None)


    def fill_market_orders(self, t):
        fees    = 0.0
        waiting = []
        for order in self.pending:
            i   = order["instrument"]
            bar = self.bars[t][i]
            if bar is None:
                waiting.append(order)       # Time, The clock: it waits for its bar
                continue

            price = bar.open                # Fills, A taker's price, with no costs here
            if order["value"] is not None:
                asked = order["value"] / price
            else:
                asked = order["size"] - order["filled"]
            size = self.clamp(i, order["side"], asked, price, order["reduce_only"])
            if bar.volume <= 0.0 or size <= DUST:
                order["status"] = "expired"
                self.log("not_filled", order, "nothing could fill on its bar")
                continue

            if size < asked - DUST:
                self.log("clamped", order, f"{size} of the {asked} asked fits",
                         size=size)
            fees += self.apply(i, order["side"], size, price, self.venue.taker, True,
                               order, order["reason"], reference=bar.open)
            order["filled"] += size

            # a market order lives one bar: what it did not fill is gone
            if size < asked - DUST:
                order["status"] = "expired"
                self.log("expired", order, "a market order takes one bar")
            else:
                order["status"] = "filled"

        self.pending = waiting
        return fees


    def clamp(self, i, side, size, price, reduce_only):
        """Wallets and Orders: every clamp a market order passes, in the page's order."""
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
                affordable = self.spot_cash / (price * (1.0 + self.venue.taker))
                size       = min(size, max(affordable, 0.0))

        return size


    def apply(self, i, side, size, price, rate, taker, order, reason, reference):
        """Accounting: a fill's fee, its cash, its position and its round trip."""
        t         = self.t
        fee       = size * price * rate
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
            price=price, reference=reference, spread=0.0, slippage=0.0, impact=0.0,
            fee=fee, adjustment=0.0, taker=taker, reason=reason,
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


def run(times, bars, instruments, venue=None, script=None, interval=None):
    """Steps a simulator through every row, making the decisions `script` holds.

    `script` maps a row to the calls made at its close, each a verb and its keyword
    arguments, so a decision made at row t is filled on a later row, never on t.
    """
    sim    = Simulator(times, bars, instruments, venue, interval)
    script = script or {}

    while True:
        for verb, kwargs in script.get(sim.t, []):
            getattr(sim, verb)(**kwargs)
        if not sim.step():
            return sim
