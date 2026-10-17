"""``Strategy``, a policy written as what it decides at each close, and the views its
``next`` reads: the bar just closed, and the account that close left, where every
order is placed (Specification, A strategy; ADR 0025)."""

import datetime
import inspect
import math
import numbers

import numpy as np

from ._data import Data

EPOCH  = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
FIELDS = ("open", "high", "low", "close", "volume")
TIFS   = ("gtc", "ioc", "fok")
# the status the order log's last word on an order leaves it with once it is gone
ENDS = {
    "filled":     "filled",
    "cancelled":  "cancelled",
    "expired":    "expired",
    "not_filled": "expired",
    "refused":    "refused",
}
# the round trips' columns an account hands out, a float's where none is named
TRIPS = ("instrument", "side", "size", "entry", "exit", "open_bar", "close_bar",
         "open_time", "close_time", "realized", "fees", "funding", "pnl", "ret", "best",
         "worst", "reason")
TYPES = {
    "instrument": object,
    "side":       object,
    "reason":     object,
    "open_bar":   np.int64,
    "close_bar":  np.int64,
    "open_time":  np.int64,
    "close_time": np.int64,
}


class Strategy:
    """A policy written as what it decides at each close.

    Run by `probatum.backtest`, ``init(data)`` runs once over the whole Data before
    the first bar, where indicators are computed over every bar at once, and
    ``next(bar, account)`` runs at the close of every bar from ``warmup`` on, counted
    in the Data's rows, except the last, which no bar follows to fill on, and a row
    none of the instruments printed. ``next`` reads the bar just closed and the
    account that close left, and places its orders through the account; they fill on
    a later bar, never on that one. ``warmup`` can be a class attribute or be set in
    ``init``, and should cover the longest history a decision reads, since an index
    below zero reads numpy's array from its end. ``on_inherit(account)`` is called
    before a validation window's first bar, so a policy can protect a position it did
    not open. Its parameters are its constructor's arguments, kept as attributes of
    the same names.

    `probatum.Backtester`, `tune` and `walk_forward` drive the class as emsl did:
    ``init(engine)`` once after the reset and ``next(state, engine)`` on every bar but
    the last, with ``warmup`` read the same way and the orders placed through the
    engine's own methods (ADR 0025).
    """

    warmup = 0


    def init(self, data):
        pass


    def next(self, bar, account):
        raise NotImplementedError(f"next: {type(self).__name__} does not write it; give "
                                  f"the strategy a next(bar, account)")


    def on_inherit(self, account):
        pass


def _warmup(strategy):
    """``strategy``'s warm-up, read after its ``init``."""
    warmup = getattr(strategy, "warmup", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, numbers.Integral) or warmup < 0:
        raise ValueError(f"warmup: {warmup!r}; give a whole number of bars from 0")
    return int(warmup)


def _called(strategy):
    """What a provenance calls ``strategy``: its own repr where its class writes one,
    and otherwise its class with the constructor's arguments it keeps."""
    kind = type(strategy)
    if kind.__repr__ is not object.__repr__:
        return repr(strategy)

    try:
        names = [name for name, parameter in inspect.signature(kind).parameters.items()
                 if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD,
                                       parameter.KEYWORD_ONLY)]
    except (TypeError, ValueError):
        names = []

    given = ", ".join(f"{name}={getattr(strategy, name)!r}" for name in names
                      if hasattr(strategy, name))
    return f"{kind.__name__}({given})"


def _play(strategy, run):
    """Drive ``strategy`` over ``run`` from its first bar: ``init`` on the Data, then
    ``next`` at each close from the warm-up on but the last."""
    strategy.init(run.data)
    warmup  = _warmup(strategy)
    account = Account(run)
    decide, step = strategy.next, run.engine.step
    rows = run.rows.tolist()

    for k in range(len(rows) - 1):
        if rows[k] >= warmup:
            run.begin(k)
            decide(Bar(run, rows[k]), account)
        step()

    # the account at its last bar, where a handle the copy keeps reads its order
    run.begin(len(rows) - 1)


def _number(value):
    """Whether ``value`` is a real number and no bool."""
    # a float or an int answers without the abstract class's check, which is slow
    if isinstance(value, (float, int)):
        return not isinstance(value, bool)
    return isinstance(value, numbers.Real)


class _Run:
    """What the views of one account share: the binding's engine, opened at the Data's
    first row its instruments printed and holding a bar for every such row; the
    Data's row of each of those bars; which instruments are spot; the handles of the
    orders placed; and what was read of the engine at the current close."""


    def __init__(self, engine, data, start, markets):
        self.engine = engine
        self.data   = data
        self.start  = start

        # a selection's fields are copies, so each is taken once
        self.available = data.available
        self.rows      = np.flatnonzero(self.available.any(axis=1))
        self.names     = data.names
        self.columns   = {name: k for k, name in enumerate(data.names)}
        self.width     = len(data.names)
        self.spot      = [market == "spot" for market in markets]
        self.inverse   = list(markets) == ["inverse"]
        self.bars      = data.bars
        self.time      = data.time
        self.rates     = None

        # what an error calls the view of the whole account
        self.named = "account"
        self.reset()


    def __copy__(self):
        return self


    def __deepcopy__(self, memo):
        # the engine a run holds cannot be copied, and a policy copied with a view of it
        # keeps the same run, whose handles the next run refuses
        return self


    def reset(self):
        """Open the account again at its first bar."""
        self.engine.reset_at(self.start)

        self.k       = 0
        self.handles = {}
        self.live    = set()
        self.seen    = 0
        self.last    = 0
        self.opening = self.engine.account()["equity"]
        self.curve   = np.empty(len(self.rows))
        self.curve[0] = self.opening
        self.points = 1
        self.store  = {name: np.empty(16, dtype=TYPES.get(name, np.float64))
                       for name in TRIPS}
        self.closed = 0
        self.trips  = None

        self.begin()


    def step(self):
        """Move the account to its next bar."""
        self.engine.step()
        self.k += 1


    def begin(self, k=None):
        """Open the decision at the close of bar ``k``, the current one where it is left
        out, with the events since the last decision and nothing yet read of the
        account."""
        if k is not None:
            self.k = k

        now         = self.engine.num_events()
        self.window = (self.last, now)
        self.last   = now
        self.state  = None
        self.events = None
        self.synced = False


    @property
    def row(self):
        """The Data's row the account is at."""
        return int(self.rows[self.k])


    def funding(self):
        """The last funding rate known at each bar's end, `(T, N)`."""
        if self.rates is None:
            self.rates = self.data.funding_rate

        return self.rates


    def bar(self, tick):
        """The Data's row of the engine's bar ``tick``."""
        return int(self.rows[tick - self.start])


    def account(self):
        """The account as the last close left it."""
        if self.state is None:
            self.state = self.engine.account()

        return self.state


    def column(self, name):
        """The column of the instrument ``name``."""
        if name not in self.columns:
            raise KeyError(f"{self.named}: no instrument {name!r}; use one of "
                           f"{', '.join(self.names)}")
        return self.columns[name]


    def one(self, field, plural=None):
        """The column of the one instrument, for a view of the whole account reading
        ``field``, refused with ``plural`` and an instrument's view as the fix where
        the account holds several."""
        if self.width != 1:
            fix = f"{self.named}.{plural} or " if plural else ""
            raise AttributeError(f"{self.named}.{field}: the account holds {self.width} "
                                 f"instruments; use {fix}{self.named}"
                                 f"[\"{self.names[0]}\"].{field}")

        return 0


    def sync(self):
        """Bring every handle to its order as it stands: the events since the last sync,
        a bracket's leg that armed on its entry's fill taking the side that closes what
        the fill opened, then the orders still open, an order no longer open taking the
        status the last of its events leaves it with."""
        if self.synced:
            return

        self.synced = True
        log         = self.engine.order_log_since(self.seen)
        self.seen += len(log)

        entered = {}
        for event in log:
            handle = self.adopt(event)
            if handle is None:
                continue

            kind, k = event["kind"], event["instrument"]
            if kind == "filled":
                handle._filled += event["size"]
                entered[k] = handle._side
            elif kind == "armed" and handle._side is None and k in entered:
                handle._side = "sell" if entered[k] == "buy" else "buy"

            handle._end    = ENDS.get(kind, handle._end)
            handle._reason = event["reason"]

        live = set()
        for given in self.engine.orders():
            handle          = self.handles[given["id"]]
            handle._side    = given["side"]
            handle._kind    = given["kind"]
            handle._size    = given["size"]
            handle._price   = given["price"]
            handle._trigger = given["trigger"]
            handle._status  = given["status"]
            live.add(given["id"])

        for order in self.live - live:
            handle         = self.handles[order]
            handle._status = handle._end

        self.live = live


    def adopt(self, event):
        """The handle of the order ``event`` names, or None for an event of no order: made
        where no verb handed one out, as for a target's leg refused as it was placed, a
        market order placed at the event's bar."""
        order = event["order"]
        if order is None:
            return None

        handle = self.handles.get(order)
        if handle is None:
            handle = self.handles[order] = Order(
                self, order, event["instrument"], self.bar(event["bar"]), "market", None,
                None, None, None, ENDS.get(event["kind"]))

        return handle


    def placed(self, before, order, k, kind, side, size=None, price=None, trigger=None,
               skip=0):
        """The handle of a placement on column ``k``, ``order`` its id, or, where the
        engine refused it, of the refusal it logged since event ``before``, past
        ``skip`` earlier ones, holding what the placement was given."""
        self.synced = False

        if order is not None:
            return self.made(order)

        refusals = [event for event in self.engine.order_log_since(before)
                    if event["kind"] == "refused"]
        if len(refusals) <= skip:
            return None

        event = refusals[skip]
        handle = self.handles[event["order"]] = Order(self, event["order"], k, self.row,
                                                      kind, side, size, price, trigger,
                                                      "refused")
        handle._end    = "refused"
        handle._reason = event["reason"]
        return handle


    def made(self, order):
        """The handle of ``order``, just placed, read as the engine holds it, its size
        and prices where the venue's sizes and price rule left them."""
        given = self.engine.open_order(order)
        handle = self.handles[order] = Order(
            self, order, given["instrument"], self.row, given["kind"], given["side"],
            given["size"], given["price"], given["trigger"], given["status"])
        self.live.add(order)

        return handle


    def price(self, name, value):
        """``value``, a price an order was given as ``name``, or None."""
        if value is None:
            return None

        if not _number(value):
            raise TypeError(f"{name}: {value!r}; give a price")

        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name}: {value}; give a finite price")

        if self.inverse and value <= 0.0:
            raise ValueError(f"{name}: {value}, which an inverse contract cannot trade "
                             f"at; give a price above 0")

        return value


    def size(self, size):
        """``size``, an order's size in base units or contracts."""
        if not _number(size):
            raise TypeError(f"size: {size!r}; give a number of base units, or of "
                            f"contracts on an inverse contract")

        return float(size)


    def order(self, side, size, limit, stop, tif, post_only, reduce_only, k):
        """A market, limit or stop order on column ``k``."""
        size = self.size(size)
        price, trigger = self.price("limit", limit), self.price("stop", stop)
        if price is not None and trigger is not None:
            raise ValueError(f"{side}: a limit of {price} and a stop of {trigger}; give "
                             f"one or the other")

        kind = ("limit" if price is not None else "stop" if trigger is not None
                else "market")

        if not isinstance(tif, str) or tif.lower() not in TIFS:
            raise ValueError(f"tif: {tif!r}; use \"gtc\", \"ioc\" or \"fok\"")

        if post_only and kind != "limit":
            raise ValueError(f"post_only: a {kind} order takes liquidity; give limit= "
                             f"to rest one")

        if kind == "stop" and tif.lower() != "gtc":
            raise ValueError(f"tif: {tif!r} for a stop, which rests until it triggers; "
                             f"leave tif out")

        before = self.engine.num_events()
        order  = self.engine.order(side, size, kind, price, trigger, reduce_only,
                                   post_only, tif, k)
        return self.placed(before, order, k, kind, side, size, price, trigger)


    def value(self, side, amount, tif, reduce_only, k):
        """A market order on column ``k`` for ``amount`` of the quote currency."""
        if not _number(amount):
            raise TypeError(f"amount: {amount!r}; give an amount of the quote currency")

        before = self.engine.num_events()
        order  = self.engine.order_value(side, float(amount), reduce_only, tif, k)
        return self.placed(before, order, k, "market", side)


    def bracket(self, stop, take_profit, k):
        """A bracket's two legs on column ``k``."""
        if stop is None and take_profit is None:
            raise ValueError("bracket: no stop and no take_profit; give either or both")

        stop        = self.price("stop", stop)
        take_profit = self.price("take_profit", take_profit)
        before      = self.engine.num_events()
        ids         = self.engine.bracket(stop, take_profit, k)

        legs, skip = [], 0
        for asked, order, kind, price, trigger in zip(
                (stop, take_profit), ids, ("stop", "limit"), (None, take_profit),
                (stop, None)):
            if asked is None:
                legs.append(None)
                continue

            legs.append(self.placed(before, order, k, kind, None, None, price, trigger,
                                    skip))
            skip += order is None

        return Exits(*legs)


    def close(self, k):
        """The order that flattens column ``k``, or None where it is flat."""
        held = self.account()["position"][k]
        if held == 0.0:
            return None

        before = self.engine.num_events()
        order  = self.engine.close(k)
        return self.placed(before, order, k, "market", "sell" if held > 0.0 else "buy",
                           abs(held))


    def own(self, order):
        """``order``, refused where it is not a handle this account handed out."""
        if not isinstance(order, Order) or order._run is not self:
            raise TypeError(f"order: {order!r}; pass an Order this {self.named}'s verbs "
                            f"returned")
        return order


    def cancel(self, order):
        """Whether ``order`` was resting and is cancelled."""
        order       = self.own(order)
        self.synced = False
        return self.engine.cancel(order.id)


    def cancel_all(self, k):
        """How many resting orders were cancelled, on column ``k`` or on every one."""
        self.synced = False
        return self.engine.cancel_all(k)


    def replace(self, order, price, trigger, size):
        """``order`` moved, or None where it was not resting or could not move."""
        order = self.own(order)
        price, trigger = self.price("price", price), self.price("trigger", trigger)
        kind = order.kind
        for name, given, takes in (("price", price, "limit"),
                                   ("trigger", trigger, "stop")):
            if given is not None and kind != takes:
                raise ValueError(f"{name}: {given} for a {kind} order, which has none; "
                                 f"move a limit by its price and a stop by its trigger")
        size        = None if size is None else self.size(size)
        self.synced = False
        moved       = self.engine.replace(order.id, size, price, trigger)
        return None if moved is None else order


    def target(self, weights, tolerance, transfer):
        """The legs that bring each instrument to its weight of equity."""
        if self.width == 1 and _number(weights):
            weight = self.weight(0, weights)
            before = self.engine.num_events()
            return self.legs(self.engine.target(weight, tolerance, transfer), before)

        if isinstance(weights, dict):
            given = {}
            for name, weight in weights.items():
                if not isinstance(name, str) or name not in self.columns:
                    raise KeyError(f"weights: no instrument {name!r}; use one of "
                                   f"{', '.join(self.names)}")
                given[self.columns[name]] = weight
        elif _number(weights):
            raise ValueError(f"weights: {weights!r}, one number, for {self.width} "
                             f"instruments; give one for each, or a dict from name to "
                             f"weight")
        else:
            values = np.asarray(weights, dtype=np.float64)
            if values.shape != (self.width,):
                raise ValueError(f"weights: an array of shape {values.shape} for "
                                 f"{self.width} instrument{'s' * (self.width > 1)}; "
                                 f"give one of ({self.width},)")

            given = dict(enumerate(values.tolist()))

        given  = {k: self.weight(k, weight) for k, weight in given.items()}
        before = self.engine.num_events()
        return self.legs(self.engine.target(given, tolerance, transfer), before)


    def weight(self, k, weight):
        """``weight``, a target's on column ``k``, as a float."""
        if not _number(weight):
            raise TypeError(f"weights: {weight!r} for {self.names[k]}; give a number")

        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError(f"weights: {weight} for {self.names[k]}; give a finite "
                             f"number")

        if weight < 0.0 and self.spot[k]:
            raise ValueError(f"weights: {weight} for {self.names[k]}, which is spot and "
                             f"cannot be short; trade a perpetual instead")

        return weight


    def legs(self, orders, before):
        """The handles of a target's legs, those it queued, ``orders`` their ids, and
        those refused as they were placed, logged since event ``before``, in the order
        they were placed."""
        self.synced = False
        out         = [self.made(order) for order in orders]

        if self.engine.num_events() - before > len(out):
            for event in self.engine.order_log_since(before):
                if event["kind"] == "refused":
                    handle = self.adopt(event)
                    handle._end, handle._reason = "refused", event["reason"]
                    out.append(handle)

        return sorted(out, key=lambda handle: handle.id)


    def open(self, k=None):
        """The handles of the orders still open, on column ``k`` or on every one."""
        self.sync()
        return [self.handles[order] for order in sorted(self.live)
                if k is None or self.handles[order]._k == k]


    def trades(self):
        """The round trips closed so far, each read once, as it closes, into columns that
        grow."""
        new = self.engine.trades_since(self.closed)

        if new or self.trips is None:
            count, columns = self.closed + len(new), {}
            for name in TRIPS:
                column = self.store[name]
                if count > len(column):
                    column = self.store[name] = np.resize(column, 2 * count)
                values = [trip[name] for trip in new]
                if name == "instrument":
                    values = [self.names[k] for k in values]
                elif name in ("open_bar", "close_bar"):
                    values = [self.bar(bar) for bar in values]
                column[self.closed:count] = values
                columns[name] = column[:count]
                columns[name].flags.writeable = False
            self.closed, self.trips = count, Trades(columns, count)

        return self.trips


    def equity_curve(self):
        """The equity at the opening and at each bar the account has seen since, read
        once each."""
        tail = self.engine.equity_since(self.points - 1)
        self.curve[self.points:self.points + len(tail)] = tail
        self.points += len(tail)

        view = self.curve[:self.points]
        view.flags.writeable = False
        return view


class Bar:
    """The bar just closed, across the Data's instruments.

    ``i`` is its row of the Data, by which arrays computed in ``init`` are read;
    ``time`` its open, a datetime in UTC, and ``ns`` the same in int64 nanoseconds
    since 1970; ``n`` the number of instruments. ``open``, ``high``, ``low``,
    ``close``, ``volume`` and ``funding_rate`` read a Data of one instrument, and
    their plural forms ``(N,)`` arrays over several, NaN where an instrument printed
    no bar; ``funding_rate`` is the last rate known at the bar's end, NaN before the
    first. ``available`` says which instruments printed it, ``bar["ETHUSDT"]`` is one
    instrument's bar, and ``history(n)`` the last ``n`` bars up to and including this
    one, or every bar so far, as a read-only Data.
    """

    __slots__ = ("_run", "i")


    def __init__(self, run, i):
        self._run = run
        self.i    = i


    def __getitem__(self, name):
        run = self._run
        if name not in run.columns:
            raise KeyError(f"bar: no instrument {name!r}; use one of "
                           f"{', '.join(run.names)}")
        return InstrumentBar(run, self.i, run.columns[name])


    def __repr__(self):
        return f"Bar({self.i:,} at {self.time:%Y-%m-%d %H:%M})"


    @property
    def time(self):
        return EPOCH + datetime.timedelta(microseconds=self.ns // 1_000)


    @property
    def ns(self):
        return int(self._run.time[self.i])


    @property
    def n(self):
        return self._run.width


    @property
    def available(self):
        return self._run.available[self.i]


    def _one(self, field):
        run = self._run
        if run.width != 1:
            raise AttributeError(f"bar.{field}: the universe has {run.width} "
                                 f"instruments; use bar.{field}s or "
                                 f"bar[\"{run.names[0]}\"].{field}")

        return getattr(InstrumentBar(run, self.i, 0), field)


    def _all(self, field):
        return self._run.bars[self.i, :, FIELDS.index(field)]


    @property
    def open(self):
        return self._one("open")


    @property
    def high(self):
        return self._one("high")


    @property
    def low(self):
        return self._one("low")


    @property
    def close(self):
        return self._one("close")


    @property
    def volume(self):
        return self._one("volume")


    @property
    def funding_rate(self):
        return self._one("funding_rate")


    @property
    def opens(self):
        return self._all("open")


    @property
    def highs(self):
        return self._all("high")


    @property
    def lows(self):
        return self._all("low")


    @property
    def closes(self):
        return self._all("close")


    @property
    def volumes(self):
        return self._all("volume")


    @property
    def funding_rates(self):
        return self._run.funding()[self.i]


    def history(self, n=None):
        data = self._run.data
        end  = data._rows[0] + self.i + 1
        if n is None:
            return Data._of(data._root, (0, end), data._cols)

        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError(f"n: {n!r}; give history a whole number of bars from 1, or "
                             f"leave n out for every bar so far")

        return Data._of(data._root, (max(0, end - int(n)), end), data._cols)


class InstrumentBar:
    """One instrument's bar just closed: its ``name``, ``open``, ``high``, ``low``,
    ``close`` and ``volume``, NaN where it printed none, ``available``, and
    ``funding_rate``, the last rate known at the bar's end, NaN before the first."""

    __slots__ = ("_run", "i", "_k")


    def __init__(self, run, i, k):
        self._run = run
        self.i    = i
        self._k   = k


    def __repr__(self):
        return f"InstrumentBar({self.name} at row {self.i:,})"


    @property
    def name(self):
        return self._run.names[self._k]


    @property
    def open(self):
        return float(self._run.bars[self.i, self._k, 0])


    @property
    def high(self):
        return float(self._run.bars[self.i, self._k, 1])


    @property
    def low(self):
        return float(self._run.bars[self.i, self._k, 2])


    @property
    def close(self):
        return float(self._run.bars[self.i, self._k, 3])


    @property
    def volume(self):
        return float(self._run.bars[self.i, self._k, 4])


    @property
    def available(self):
        return bool(self._run.available[self.i, self._k])


    @property
    def funding_rate(self):
        return float(self._run.funding()[self.i, self._k])


class _Verbs:
    """The verbs that place an order on one instrument, for a view whose ``_column``
    says which."""

    __slots__ = ()


    def buy(self, size, limit=None, stop=None, tif="gtc", post_only=False,
            reduce_only=False):
        """A buy of ``size`` base units, or contracts on an inverse contract: at market
        on the next open, at ``limit`` or better, or once a price reaches ``stop``,
        filled as a taker. ``tif`` is "gtc", "ioc" or "fok" on a market or a limit
        order; ``post_only`` keeps a limit from ever taking liquidity and
        ``reduce_only`` any order from growing the position. Returns its `Order`,
        refused ones included."""
        return self._run.order("buy", size, limit, stop, tif, post_only, reduce_only,
                               self._column("buy"))


    def sell(self, size, limit=None, stop=None, tif="gtc", post_only=False,
             reduce_only=False):
        """A sell, as `buy` places a buy."""
        return self._run.order("sell", size, limit, stop, tif, post_only, reduce_only,
                               self._column("sell"))


    def buy_value(self, amount, tif="ioc", reduce_only=False):
        """A market buy of ``amount`` of the quote currency, its size the amount over
        its fill price at the next open; ``tif`` is "ioc" or "fok". Returns its
        `Order`, whose size is None and whose ``filled`` is what the amount bought
        (Specification, Units)."""
        return self._run.value("buy", amount, tif, reduce_only,
                               self._column("buy_value"))


    def sell_value(self, amount, tif="ioc", reduce_only=False):
        """A market sell of ``amount`` of the quote currency, as `buy_value` buys."""
        return self._run.value("sell", amount, tif, reduce_only,
                               self._column("sell_value"))


    def bracket(self, stop=None, take_profit=None):
        """An OCO pair of reduce-only exits with no size of their own, a stop at
        ``stop`` and a take-profit at ``take_profit``, each closing the position it
        finds: armed at once on a position held, and otherwise once the entry fills.
        Returns the `Exits`, ``.stop`` and ``.take_profit``, None for one not asked
        for."""
        return self._run.bracket(stop, take_profit, self._column("bracket"))


    def close(self):
        """A reduce-only market order that flattens the position on the next open, or
        None where it is flat."""
        return self._run.close(self._column("close"))


    def cancel(self, order):
        """Cancel ``order``, an `Order` this account placed, returning whether it was
        resting."""
        return self._run.cancel(order)


    def replace(self, order, price=None, trigger=None, size=None):
        """Move ``order``, a resting `Order`, to a new ``size``, a limit to a new
        ``price`` or a stop to a new ``trigger``, under the same id, returning it, or None
        where it was no longer resting or the moved order could not rest, which leaves
        it as it was."""
        return self._run.replace(order, price, trigger, size)


class Account(_Verbs):
    """The account the last close left, and the place every order is made.

    ``equity`` and ``leverage``; ``position`` and ``entry`` with one instrument, and
    ``positions``, ``(N,)``, with any number; ``orders``, the orders still open, as
    live `Order` handles; ``events``, every `Event` of its orders since the last close,
    oldest first; ``trades``, the round trips closed so far, the columns of
    ``result.trades`` as read-only arrays; ``equity_curve``, the equity at the opening
    and at every bar since, one a row its instruments printed, as ``result.equity``;
    and the two wallets, ``spot`` and ``margin``, each with its ``cash`` and
    ``equity``, the margin wallet also with its maintenance ``requirement`` and
    ``available``, the room the cap still leaves. ``account["ETHUSDT"]`` is one
    instrument's view, with its own verbs. With several instruments, the verbs that
    place an order on one are taken through its view; ``target``, ``transfer`` and
    ``cancel_all`` act on the whole account. Every size and price is in the venue's
    terms, and every bar is the Data's row, as ``bar.i`` is.
    """

    __slots__ = ("_run",)


    def __init__(self, run):
        self._run = run


    def __getitem__(self, name):
        return InstrumentAccount(self._run, self._run.column(name))


    def _column(self, verb):
        run = self._run
        if run.width != 1:
            raise AttributeError(f"{run.named}.{verb}: the account holds {run.width} "
                                 f"instruments; use {run.named}[\"{run.names[0]}\"]."
                                 f"{verb}")
        return 0


    @property
    def equity(self):
        return self._run.account()["equity"]


    @property
    def leverage(self):
        return self._run.account()["leverage"]


    @property
    def position(self):
        run = self._run
        return run.account()["position"][run.one("position", "positions")]


    @property
    def entry(self):
        run = self._run
        return run.account()["entry"][run.one("entry")]


    @property
    def positions(self):
        return np.array(self._run.account()["position"])


    @property
    def orders(self):
        return self._run.open()


    @property
    def events(self):
        run = self._run
        if run.events is None:
            first, last = run.window
            log        = run.engine.order_log_since(first)[:last - first] if last > first else ()
            run.events = [Event(run, event) for event in log]

        return run.events


    @property
    def trades(self):
        return self._run.trades()


    @property
    def equity_curve(self):
        return self._run.equity_curve()


    @property
    def spot(self):
        state = self._run.account()
        return Wallet(state["spot_cash"], state["spot_equity"])


    @property
    def margin(self):
        state = self._run.account()
        return Margin(state["margin_cash"], state["margin_equity"],
                      state["maintenance"], state["available"])


    def target(self, weights, tolerance=0.0, transfer=False):
        """Bring each instrument to its weight of equity at the last close: one number
        with one instrument, one for each in order, or a dict from name to weight,
        which leaves the instruments it does not name as they are. Each leg is a
        market order for the next open, every reducing one before every growing one;
        a leg below ``tolerance`` of equity is skipped, and ``transfer`` moves what the
        spot legs need beyond the spot wallet's cash across from the margin wallet.
        Returns its legs' `Order` handles, refused ones included (ADR 0009)."""
        return self._run.target(weights, tolerance, transfer)


    def transfer(self, amount, to="margin"):
        """Move ``amount`` of cash into the ``to`` wallet, "spot" or "margin", at the
        start of the next bar, held to what the wallet it leaves may give (ADR 0008)."""
        self._run.engine.transfer(amount, to)


    def cancel_all(self):
        """Cancel every resting order, returning how many were."""
        return self._run.cancel_all(None)


class InstrumentAccount(_Verbs):
    """One instrument of the account: its ``name``, ``position``, ``entry``, ``upnl``,
    the unrealized PnL at the last close, ``weight``, its signed notional over
    equity, and ``orders``, its orders still open, with the verbs that place an
    order on it."""

    __slots__ = ("_run", "_k")


    def __init__(self, run, k):
        self._run = run
        self._k   = k


    def __repr__(self):
        return f"InstrumentAccount({self.name}, position {self.position})"


    def _column(self, verb):
        return self._k


    @property
    def name(self):
        return self._run.names[self._k]


    @property
    def position(self):
        return self._run.account()["position"][self._k]


    @property
    def entry(self):
        return self._run.account()["entry"][self._k]


    @property
    def upnl(self):
        return self._run.account()["unrealized"][self._k]


    @property
    def weight(self):
        return self._run.account()["weight"][self._k]


    @property
    def orders(self):
        return self._run.open(self._k)


    def cancel_all(self):
        """Cancel every order resting on the instrument, returning how many were."""
        return self._run.cancel_all(self._k)


class Order:
    """A live handle on one order, showing it as it stands: ``id``; ``instrument``, its
    name; ``side``, "buy" or "sell", None for a bracket's leg until it arms and for a
    target's leg refused before it took one; ``kind``, "market", "limit" or "stop";
    ``size``, None for a bracket's leg that closes the position it finds and for a
    value order, whose ``filled`` is what its amount bought;
    ``filled``, what has filled of it; ``price`` and ``trigger``, a limit's price and
    a stop's trigger; ``placed``, the Data's row it was placed at; ``status``,
    "pending" while a market order waits for the next open and "resting" while a
    limit or a stop is on the book, then "filled", "cancelled", "expired" or
    "refused"; and ``reason``, what the order log said of the last thing that
    happened to it."""

    __slots__ = ("_run", "id", "_k", "placed", "_kind", "_side", "_size", "_filled",
                 "_price", "_trigger", "_status", "_end", "_reason")


    def __init__(self, run, order, k, placed, kind, side, size, price, trigger, status):
        self._run     = run
        self.id       = order
        self._k       = k
        self.placed   = placed
        self._kind    = kind
        self._side    = side
        self._size    = size
        self._filled  = 0.0
        self._price   = price
        self._trigger = trigger
        self._status  = status
        self._end     = None
        self._reason  = None


    def __repr__(self):
        self._run.sync()
        size = "" if self._size is None else f"{self._size:g} "
        return (f"Order({self.id}, {self._side or 'unarmed'} {size}{self._kind} on "
                f"{self.instrument}, {self._status})")


    @property
    def instrument(self):
        return self._run.names[self._k]


    @property
    def side(self):
        self._run.sync()
        return self._side


    @property
    def kind(self):
        self._run.sync()
        return self._kind


    @property
    def size(self):
        self._run.sync()
        return self._size


    @property
    def filled(self):
        self._run.sync()
        return self._filled


    @property
    def price(self):
        self._run.sync()
        return self._price


    @property
    def trigger(self):
        self._run.sync()
        return self._trigger


    @property
    def status(self):
        self._run.sync()
        return self._status


    @property
    def reason(self):
        self._run.sync()
        return self._reason


class Exits:
    """A bracket's two legs, ``stop`` and ``take_profit``, each an `Order`, or None for
    one not asked for."""

    __slots__ = ("stop", "take_profit")


    def __init__(self, stop, take_profit):
        self.stop        = stop
        self.take_profit = take_profit


    def __iter__(self):
        return iter((self.stop, self.take_profit))


    def __repr__(self):
        return f"Exits(stop={self.stop!r}, take_profit={self.take_profit!r})"


class Event:
    """One thing that happened to the account's orders: ``kind``, "placed", "armed",
    "filled", "clamped", "refused", "not_filled", "expired" or "cancelled";
    ``order``, its `Order`, None for a transfer's clamp; ``instrument``, its name;
    ``bar``, the Data's row it happened at; ``size`` and ``price``, a fill's or a
    clamp's; and ``reason``."""

    __slots__ = ("kind", "order", "instrument", "bar", "size", "price", "reason")


    def __init__(self, run, event):
        self.kind       = event["kind"]
        self.order      = run.adopt(event)
        k               = event["instrument"]
        self.instrument = None if k is None else run.names[k]
        self.bar        = run.bar(event["bar"])
        self.size       = event["size"]
        self.price      = event["price"]
        self.reason     = event["reason"]


    def __repr__(self):
        order = "" if self.order is None else f" order {self.order.id}"
        return f"Event({self.kind}{order} at row {self.bar:,}, {self.reason})"


class Wallet:
    """The spot wallet at the last close: its ``cash`` and its ``equity``, the cash and
    what its holdings are worth at the closes."""

    __slots__ = ("cash", "equity")


    def __init__(self, cash, equity):
        self.cash   = cash
        self.equity = equity


    def __repr__(self):
        return (f"{type(self).__name__}(cash {self.cash:,.2f}, equity "
                f"{self.equity:,.2f})")


class Margin(Wallet):
    """The margin wallet at the last close: its ``cash``, its ``equity``, the cash with
    the perpetuals' unrealized PnL, its maintenance ``requirement``, and
    ``available``, its equity less the margin the cap asks, the room the cap still
    leaves, floored at zero (ADR 0015)."""

    __slots__ = ("requirement", "available")


    def __init__(self, cash, equity, requirement, available):
        super().__init__(cash, equity)
        self.requirement = requirement
        self.available   = available


class Trades:
    """The round trips closed so far, oldest first, read as ``result.trades`` is: a
    column by name, ``trades["pnl"]`` or ``trades.pnl``, as a read-only array, and a
    row by position, ``trades[-1]``, as a dict, iterating giving the rows. Its
    ``open_bar`` and ``close_bar`` are the Data's rows, as ``bar.i`` is, where the
    Result's count from its own first row, and ``instrument`` the names."""


    def __init__(self, columns, length):
        self._columns = columns
        self._length  = length


    def __len__(self):
        return self._length


    @property
    def columns(self):
        """The columns' names, in order."""
        return tuple(self._columns)


    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._columns[key]
            except KeyError:
                raise KeyError(f"trades: no column {key!r}; use one of "
                               f"{', '.join(self._columns)}") from None

        if isinstance(key, numbers.Integral) and not isinstance(key, bool):
            at = int(key) + self._length if key < 0 else int(key)
            if not self._length:
                raise IndexError(f"row: {int(key)} of no round trips; check "
                                 f"len(account.trades) first")
            if not 0 <= at < self._length:
                raise IndexError(f"row: {int(key)} of {self._length:,} round trips; give "
                                 f"one from {-self._length:,} to {self._length - 1:,}")
            return {name: column[at].item() if isinstance(column[at], np.generic)
                    else column[at] for name, column in self._columns.items()}

        raise TypeError(f"key: {key!r}; give a column's name or a row's position")


    def __iter__(self):
        return (self[at] for at in range(self._length))


    def __contains__(self, item):
        if isinstance(item, str):
            raise TypeError(f"{item!r} in trades: the round trips are rows, as dicts; ask "
                            f"{item!r} in trades.columns")

        return any(row == item for row in self)


    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        columns = self.__dict__["_columns"]
        if name not in columns:
            raise AttributeError(f"trades: no column {name!r}; use one of "
                                 f"{', '.join(columns)}")
        return columns[name]


    def __repr__(self):
        return f"Trades({self._length} closed)"
