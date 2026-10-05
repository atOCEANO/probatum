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
refusal, the worse exit first, fees, the spot and margin wallets with transfers,
listing and delisting, round trips and the ledger. The fill model's costs,
funding, the margin cap and liquidation, inverse contracts and several instruments
on one path are not here.
"""

import collections
import math
import statistics

DUST = 1e-9          # Fills, Sizes and dust: a fill this small moves nothing

Bar = collections.namedtuple("Bar", "open high low close volume")


class Instrument:

    def __init__(self, name, kind="linear", lot=0.0, tick=0.0, delist=None):
        if kind not in ("spot", "linear"):
            raise ValueError(f"kind: {kind!r} is not modelled; use spot or linear")

        self.name   = name
        self.kind   = kind
        self.lot    = float(lot)
        self.tick   = float(tick)
        self.delist = delist          # nanoseconds, or None


class Venue:

    def __init__(self, balance=10_000.0, maker=0.0002, taker=0.0006, leverage=10.0,
                 max_open_orders=8, limit="touch", intrabar="pessimistic"):
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
        self.intrabar        = intrabar


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
        self.slots       = []          # Orders, Maker, taker and the book: one table
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


    def resting(self, instrument=None):
        return [o for o in self.slots if o is not None
                and (instrument is None or o["instrument"] == instrument)]


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


    def common_refusal(self, order):
        i = order["instrument"]
        if not self.listed[i]:
            return "the instrument has not printed its first bar"

        if self.delisted[i]:
            return "the instrument is delisted"

        return None


    def place_market(self, order):
        """Orders, Order types: a market order fills at its instrument's next open,
        and a value order's size is its amount over that fill price (Units)."""
        asked = order["size"] if order["value"] is None else order["value"]
        why   = self.common_refusal(order)

        if asked is None or not math.isfinite(asked) or asked <= 0.0:
            why = f"size: {asked} cannot fill"
        elif why is None and len([o for o in self.pending
                                  if o["instrument"] == order["instrument"]]) \
                >= self.venue.max_open_orders:
            why = f"max_open_orders: {self.venue.max_open_orders} already waiting"

        if why is not None:
            return self.refuse(order, why)

        self.log("placed", order, order["reason"])
        self.pending.append(order)

        return order["id"]


    def resting_refusal(self, order, replacing=None):
        """Why a limit or a stop cannot rest, or None."""
        i, side = order["instrument"], order["side"]
        why = self.common_refusal(order)
        if why is not None:
            return why

        if order["size"] is not None and (not math.isfinite(order["size"])
                                          or order["size"] <= 0.0):
            return f"size: {order['size']} cannot fill"

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

        for leg, kind, price in (("stop", "stop", stop), ("take_profit", "limit",
                                                          take_profit)):
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
            moved["price"] = float(price)
        if trigger is not None:
            moved["trigger"] = float(trigger)

        if size is not None:
            moved["size"] = float(size) + old["filled"]
        if moved["kind"] == "limit" and price is not None:
            # a moved limit is a new placement for maker or taker and for post_only
            moved["placed"], moved["seen"] = self.t, 0

        why = self.resting_refusal(moved, replacing=old)
        if why is not None:
            self.log("refused", old, f"replace: {why}")
            return None

        slot = self.slots.index(old)
        self.slots[slot] = None
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
                growing.append((i, q + leg))
            elif q != 0.0 and abs(q + leg) < abs(q):
                reducing.append((i, leg))
            else:
                growing.append((i, leg))

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
        t    = self.t
        fees = 0.0

        self.apply_transfers()                   # 1
        # 2 funding between rows and 3 the liquidation path are not in this scope
        self.delist(t)
        fees += self.fill_market_orders(t)       # 4
        self.arm()
        fees += self.fill_resting_orders(t)      # 5
        # 6 to 8, exits on a liquidating bar, the check and funding, are not here
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

            for order in [o for o in self.pending + self.resting()
                          if o["instrument"] == i]:
                self.cancel(order["id"], "delist")

            self.delisted[i] = True

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
            size = self.clamp(i, order["side"], asked, price, order["reduce_only"],
                              self.venue.taker)
            if bar.volume <= 0.0 or size <= DUST or (order["tif"] == "fok"
                                                     and size < asked - DUST):
                order["status"] = "expired"
                self.log("not_filled", order, "nothing could fill on its bar"
                         if order["tif"] != "fok" else "fok: the whole size could not")
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


    def side_of(self, order):
        """A bracket leg's side is taken from the position it finds (Order types)."""
        if order["side"] is not None:
            return order["side"]

        q = self.position[order["instrument"]]
        if q == 0.0:
            return None
        return "sell" if q > 0.0 else "buy"


    def offer(self, order, bar):
        """Whether the bar reaches a resting order, and if so at what price, maker or
        taker (Order types, Maker, taker and the book, Fills, Limits)."""
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
            first = order["seen"] == 0
            return price, first and opened_through

        trigger = order["trigger"]
        if side == "buy":
            if bar.high < trigger:
                return None
            price = min(max(trigger, bar.open), bar.high)
        else:
            if bar.low > trigger:
                return None
            price = max(min(trigger, bar.open), bar.low)

        return price, True


    def fill_resting_orders(self, t):
        """The step, 5: resting orders in slot order, a position's exits reordered by
        the PnL per unit each would realize, worst first unless optimistic."""
        fees   = 0.0
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
        for slot, order, price, taker in offers:
            if order["status"] != "resting":
                continue                     # its OCO partner filled first

            i    = order["instrument"]
            side = self.side_of(order)
            if side is None:
                continue

            remaining = (abs(self.position[i]) if order["size"] is None
                         else order["size"] - order["filled"])
            rate = self.venue.taker if taker else self.venue.maker
            size = self.clamp(i, side, remaining, price, order["reduce_only"], rate)
            if order["tif"] == "fok" and size < remaining - DUST:
                continue
            if size <= DUST:
                continue
            if size < remaining - DUST:
                self.log("clamped", order, f"{size} of the {remaining} asked fits",
                         size=size)

            reason = order["leg"] or order["reason"]
            fees += self.apply(i, side, size, price, rate, taker, order, reason,
                               reference=price)
            order["filled"] += size
            if order["pair"] is not None:
                self.cancel(order["pair"], "its OCO partner filled")
            done = (self.position[i] == 0.0 if order["size"] is None
                    else order["size"] - order["filled"] <= DUST)
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

        return fees


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
